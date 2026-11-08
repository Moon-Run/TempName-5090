// Copyright 2026. MIT licensed repository adapter.
// Flux's Tensor Core mainloop and BF16 compute visitor are used unchanged.
// Vendored Flux: Apache-2.0; CUTLASS: BSD-3-Clause (third_party/*/LICENSE*).
#include "comm_none/gemm_v2_comm_none.hpp"
#include <cuda/atomic>
#include <cuda.h>
#include <dlfcn.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace flux5090_native {
using namespace cute;
using namespace bytedance::flux;
constexpr int kMaxBlocks = 128;
thread_local std::string last_error;

void require(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}
void check_cuda(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
void check_cutlass(cutlass::Status status) {
  if (status != cutlass::Status::kSuccess)
    throw std::runtime_error(cutlassGetStatusString(status));
}

struct Schedule {
  int m = 0, n = 0, world = 0, chunk_rows = 0, blocks = 0, interleaved = 0;
  int order[kMaxBlocks] = {};

  CUTLASS_HOST_DEVICE
  cutlass::gemm::GemmCoord tile(int index) const {
    const int tiles_n = n / 128;
    const int tiles_per_macro = (chunk_rows / 128) * tiles_n;
    int slot, within;
    if (interleaved) {
      const int destination = index % world;
      const int within_destination = index / world;
      slot = (within_destination / tiles_per_macro) * world + destination;
      within = within_destination % tiles_per_macro;
    } else {
      slot = index / tiles_per_macro;
      within = index % tiles_per_macro;
    }
    return {order[slot] * (chunk_rows / 128) + within / tiles_n,
            within % tiles_n, 0};
  }
};

Schedule make_schedule(int m, int n, int world, int chunk_rows,
                       const int* order, int interleaved) {
  require(m > 0 && n > 0 && world > 0 && world <= 16,
          "Positive M/N and world in [1,16] required");
  require(chunk_rows >= 128 && chunk_rows % 128 == 0 && n % 128 == 0,
          "chunk_rows and N must be multiples of the 128x128 Tensor Core tile");
  require(m % world == 0 && (m / world) % chunk_rows == 0,
          "Each destination's M slice must contain complete macroblocks");
  const int blocks = m / chunk_rows;
  require(blocks <= kMaxBlocks && order, "Invalid or excessively large macroblock schedule");
  require(interleaved == 0 || interleaved == 1, "interleaved must be boolean");
  Schedule result;
  result.m=m; result.n=n; result.world=world; result.chunk_rows=chunk_rows;
  result.blocks=blocks; result.interleaved=interleaved;
  std::array<bool, kMaxBlocks> used{};
  const int per_destination = blocks / world;
  for (int slot = 0; slot < blocks; ++slot) {
    const int physical = order[slot];
    require(physical >= 0 && physical < blocks && !used[physical],
            "order must be a bijection over physical macroblocks");
    const int destination = interleaved ? slot % world : slot / per_destination;
    require(physical / per_destination == destination,
            "order changes the destination assigned to a schedule slot");
    used[physical]=true; result.order[slot]=physical;
  }
  return result;
}

// Host-only launch state is copied into each immutable CUTLASS Params object.
// No mutable device-global mapping and no per-forward plan upload are used.
thread_local const Schedule* launch_schedule = nullptr;

struct OrderedDPSwizzle : cutlass::gemm::threadblock::ThreadblockSwizzleStreamK {
  using Base = cutlass::gemm::threadblock::ThreadblockSwizzleStreamK;
  Schedule schedule;
  CUTLASS_HOST_DEVICE OrderedDPSwizzle() {}
  OrderedDPSwizzle(cutlass::gemm::GemmUniversalMode mode,
                  cutlass::gemm::GemmCoord problem, cutlass::gemm::GemmCoord tile,
                  int split, int occupancy, int device_sms, int available_sms,
                  size_t a_bytes, size_t b_bytes, size_t c_bytes, int fragments)
      : Base(mode, problem, tile, split, occupancy, device_sms, 1,
             a_bytes, b_bytes, c_bytes, fragments) {
    require(launch_schedule != nullptr, "Missing prepared launch schedule");
    require(split == 1 && mode == cutlass::gemm::GemmUniversalMode::kGemm,
            "Only complete-K data-parallel GEMM is supported");
    schedule = *launch_schedule;
    // Disable CUTLASS's optional cohort padding/remapping as well as split-K.
    // avail_sms=1 selects DP heuristics; it does NOT restrict execution to one SM.
    cohort_raster = false;
    remap_block_indices = false;
    dp_blocks = (problem.m() / 128) * (problem.n() / 128);
    dp_first_wave_tiles = 1;
    require(sk_waves == 0 && sk_tiles == 0 && reduction_blocks == 0,
            "Unexpected split-K work in the deterministic DP kernel");
  }
  CUTLASS_DEVICE cutlass::gemm::GemmCoord get_tile_offset(int index) const {
    return schedule.tile(index);
  }
};

// Only the store visitor is wrapped; Flux's accumulation, alpha conversion,
// BF16 rounding and vector stores remain the upstream implementation.
template<class ThreadMap, class Element, class Stride>
struct PublishStore : cutlass::epilogue::threadblock::VisitorAuxStore<
    ThreadMap, Element, cutlass::FloatRoundStyle::round_to_nearest, Stride> {
  using Base = cutlass::epilogue::threadblock::VisitorAuxStore<
      ThreadMap, Element, cutlass::FloatRoundStyle::round_to_nearest, Stride>;
  struct Arguments {
    typename Base::Arguments store;
    uint32_t* counters;
    uint32_t* ready;
    uint32_t epoch;
    int chunk_rows;
    int tiles_per_macro;
  };
  using Params = Arguments;
  using SharedStorage = typename Base::SharedStorage;
  const Params* publish;
  CUTLASS_HOST_DEVICE PublishStore() {}
  CUTLASS_HOST_DEVICE PublishStore(const Params& params, const SharedStorage& shared)
      : Base(params.store, shared), publish(&params) {}
  template<class Problem>
  static Params to_underlying_arguments(const Problem&, const Arguments& args, void*) {
    return args;
  }
  template<class Problem>
  static size_t get_workspace_size(const Problem&, const Arguments&) { return 0; }

  template<class ParentCallbacks>
  struct Callbacks : ParentCallbacks {
    const Params* publish;
    int macro, thread;
    CUTLASS_DEVICE Callbacks(ParentCallbacks&& base, const Params* p, int macro_, int thread_)
        : ParentCallbacks(cute::move(base)), publish(p), macro(macro_), thread(thread_) {}
    CUTLASS_DEVICE void end_epilogue() {
      ParentCallbacks::end_epilogue();
      // Every store-writing thread releases its writes, before its CTA adds
      // one completion. Acq/rel RMWs combine all contributing CTA releases.
      __threadfence();
      __syncthreads();
      if (thread == 0) {
        cuda::atomic_ref<uint32_t, cuda::thread_scope_device> count(publish->counters[macro]);
        const uint32_t previous = count.fetch_add(1, cuda::memory_order_acq_rel);
        if (previous + 1 == uint32_t(publish->tiles_per_macro)) {
          cuda::atomic_ref<uint32_t, cuda::thread_scope_system> flag(publish->ready[macro]);
          flag.store(publish->epoch, cuda::memory_order_release);
        }
      }
    }
  };
  template<class Problem>
  CUTLASS_DEVICE auto get_callbacks(cutlass::gemm::GemmCoord tile, int thread,
                                   Problem problem) {
    auto callbacks = Base::get_callbacks(tile, thread, problem);
    return Callbacks<decltype(callbacks)>(cute::move(callbacks), publish,
                                         tile.m() * 128 / publish->chunk_rows, thread);
  }
};

using Meta = decltype(make_gemm_meta(
    make_gemm_dtype_config(_BF16{}, _BF16{}, _Void{}, _BF16{}),
    _Sm80{}, _A100{}, _CommNone{}, _RCR{}, _GemmV2{}, make_gemm_v2_meta(_False{})));
using HParams = decltype(make_gemm_hparams(
    make_gemm_v2_hparams(Shape<_64,_64,_32>{}, Shape<_16,_8,_16>{}, _StreamkDP{}),
    None{}, Shape<_128,_128,_32>{}, _GemmStreamK{}, _3{}, _RasterHeuristic{}));
using FluxBuilder = GemmV2CommNone_Kernel<Meta,HParams>;
using DefaultParams = decltype(FluxBuilder{}.default_kernel_params());
using ThreadMap = decltype(FluxBuilder{}.output_tile_thread_map(DefaultParams{}));
using Compute = decltype(FluxBuilder{}.default_evt_d(DefaultParams{}));
using Store = PublishStore<ThreadMap,cutlass::bfloat16_t,Stride<int64_t,_1,int64_t>>;
using EVT = cutlass::epilogue::threadblock::Sm80EVT<Store,Compute>;
using Params = gemm_v2_impl::KernelParams<OrderedDPSwizzle,Int<8>,EVT>;
using Kernel = decltype(FluxBuilder{}.default_gemm_kernel(Params{}));
using Gemm = cutlass::gemm::device::GemmUniversalBase<Kernel>;

struct Plan {
  Schedule schedule;
  int k, device;
  Gemm op;
  void* workspace = nullptr;
  size_t workspace_bytes = 0;
  ~Plan() { if (workspace) cudaFree(workspace); }
  Gemm::Arguments arguments(const void* a, const void* b, void* d,
                            uint32_t* counters, uint32_t* ready, uint32_t epoch) const {
    const int m=schedule.m, n=schedule.n;
    typename EVT::Arguments callbacks{
      {{{1.0f}}, {}, {}},
      {{static_cast<cutlass::bfloat16_t*>(d), {int64_t(n), _1{}, int64_t(m)*n}},
       counters, ready, epoch, schedule.chunk_rows, (schedule.chunk_rows/128)*(n/128)}
    };
    return Gemm::Arguments(cutlass::gemm::GemmUniversalMode::kGemm,
        {m,n,k}, 1, callbacks, a,b,nullptr,nullptr,
        int64_t(m)*k,int64_t(n)*k,0,0,k,k,n,n,1);
  }
};

using WaitFn = CUresult (CUDAAPI*)(CUstream,CUdeviceptr,cuuint32_t,unsigned int);
WaitFn wait_function() {
  static WaitFn function = []() {
    void* library = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
    require(library != nullptr, "Cannot load NVIDIA CUDA driver libcuda.so.1");
    auto f = reinterpret_cast<WaitFn>(dlsym(library,"cuStreamWaitValue32_v2"));
    require(f != nullptr, "CUDA driver lacks cuStreamWaitValue32_v2");
    return f;
  }();
  return function;
}
void wait_value(void* stream, uint32_t* address, uint32_t epoch) {
  const auto status = wait_function()(reinterpret_cast<CUstream>(stream),
      reinterpret_cast<CUdeviceptr>(address), epoch, CU_STREAM_WAIT_VALUE_EQ);
  if (status != CUDA_SUCCESS)
    throw std::runtime_error("cuStreamWaitValue32_v2 failed, CUresult="+std::to_string(int(status)));
}

__global__ void mapping_probe(Schedule schedule, int* output, int count) {
  const int i=blockIdx.x*blockDim.x+threadIdx.x;
  if (i<count) {
    const auto coordinate=schedule.tile(i);
    output[2*i]=coordinate.m(); output[2*i+1]=coordinate.n();
  }
}
} // namespace

using namespace flux5090_native;

#define GUARDED_BEGIN try { last_error.clear();
#define GUARDED_END return 0; } catch(const std::exception& e) { last_error=e.what(); return -1; } catch(...) { last_error="Unknown native error"; return -1; }

extern "C" const char* fx_last_error() { return last_error.c_str(); }
extern "C" int fx_abi_version() { return 1; }
extern "C" int fx_map_host(int m,int n,int world,int chunk_rows,const int* order,
                            int interleaved,int* output) {
  GUARDED_BEGIN
  auto schedule=make_schedule(m,n,world,chunk_rows,order,interleaved);
  require(output != nullptr,"Null mapping output");
  for (int i=0;i<(m/128)*(n/128);++i) {
    const auto coordinate=schedule.tile(i);
    output[2*i]=coordinate.m(); output[2*i+1]=coordinate.n();
  }
  GUARDED_END
}
extern "C" int fx_prepare(int m,int n,int k,int world,int chunk_rows,
                           const int* order,int interleaved,void** output) {
  GUARDED_BEGIN
  require(output != nullptr,"Null plan output");
  require(k>0 && k%8==0,"K must be a positive multiple of 8");
  auto plan=std::make_unique<Plan>();
  plan->schedule=make_schedule(m,n,world,chunk_rows,order,interleaved); plan->k=k;
  check_cuda(cudaGetDevice(&plan->device));
  cudaDeviceProp properties{}; check_cuda(cudaGetDeviceProperties(&properties,plan->device));
  require(properties.major==12 && properties.minor==0,"Native GEMM requires SM120 (RTX5090)");
  launch_schedule=&plan->schedule;
  const auto args=plan->arguments(nullptr,nullptr,nullptr,nullptr,nullptr,1);
  plan->workspace_bytes=Gemm::get_workspace_size(args);
  if (plan->workspace_bytes) check_cuda(cudaMalloc(&plan->workspace,plan->workspace_bytes));
  *output=plan.release();
  GUARDED_END
}
extern "C" int fx_workspace_bytes(void* pointer,uint64_t* bytes) {
  GUARDED_BEGIN
  require(pointer && bytes,"Null workspace query");
  *bytes=static_cast<Plan*>(pointer)->workspace_bytes;
  GUARDED_END
}
extern "C" int fx_destroy(void* pointer) {
  GUARDED_BEGIN
  delete static_cast<Plan*>(pointer);
  GUARDED_END
}
extern "C" int fx_forward(void* pointer,const void* a,const void* b,void* d,
                           uint32_t* counters,uint32_t* ready,uint32_t epoch,void* stream) {
  GUARDED_BEGIN
  require(pointer && a && b && d && counters && ready,"Null GEMM buffer/plan");
  require(epoch>0 && epoch<0x80000000u,"epoch must be in [1, 2^31)");
  auto& plan=*static_cast<Plan*>(pointer);
  int device; check_cuda(cudaGetDevice(&device));
  require(device==plan.device,"Plan used on a different CUDA device");
  launch_schedule=&plan.schedule;
  const auto args=plan.arguments(a,b,d,counters,ready,epoch);
  check_cutlass(Gemm::can_implement(args));
  auto s=static_cast<cudaStream_t>(stream);
  check_cuda(cudaMemsetAsync(counters,0,plan.schedule.blocks*sizeof(uint32_t),s));
  check_cutlass(plan.op.initialize(args,plan.workspace,s));
  check_cutlass(plan.op.run(s));
  GUARDED_END
}
extern "C" int fx_wait(uint32_t* ready,int index,uint32_t epoch,void* stream) {
  GUARDED_BEGIN
  require(ready && index>=0 && epoch>0,"Invalid readiness wait");
  wait_value(stream,ready+index,epoch);
  GUARDED_END
}
extern "C" int fx_map_device(void* pointer,int* output,void* stream) {
  GUARDED_BEGIN
  require(pointer && output,"Null mapping probe buffer");
  const auto& schedule=static_cast<Plan*>(pointer)->schedule;
  const int count=(schedule.m/128)*(schedule.n/128);
  mapping_probe<<<(count+127)/128,128,0,static_cast<cudaStream_t>(stream)>>>(schedule,output,count);
  check_cuda(cudaGetLastError());
  GUARDED_END
}
extern "C" int fx_probe_wait() {
  GUARDED_BEGIN
  // Bounded functional support probe: the value is already written before
  // the wait is submitted. This tests availability, not GEMM/comm overlap.
  uint32_t* value=nullptr; cudaStream_t stream=nullptr;
  check_cuda(cudaMalloc(&value,sizeof(uint32_t)));
  try {
    check_cuda(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    check_cuda(cudaMemsetAsync(value,0,sizeof(uint32_t),stream));
    wait_value(stream,value,0);
    check_cuda(cudaStreamSynchronize(stream));
  } catch (...) {
    if(stream) cudaStreamDestroy(stream);
    cudaFree(value); throw;
  }
  check_cuda(cudaStreamDestroy(stream)); check_cuda(cudaFree(value));
  GUARDED_END
}
extern "C" int fx_occupancy(int* blocks) {
  GUARDED_BEGIN
  require(blocks != nullptr,"Null occupancy output");
  *blocks=Gemm::maximum_active_blocks();
  require(*blocks>0,"Native kernel cannot reside on this device");
  GUARDED_END
}
