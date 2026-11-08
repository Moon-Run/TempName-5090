// TACO H128/E4M3FN CUDA arithmetic; standalone SoA/ordered-reduction adapter.
// Copyright (c) 2025, Institute of Computing Technology, Chinese Academy of Sciences.
// SPDX-License-Identifier: BSD-3-Clause
// Algorithm: COCCL_TACO ca717e05b6dd6d21fff4d2617d209c95be63f3ef.
// Warp/register layout follows TempName-New/tools/5090/phase6/op3_codec.cu.
// See third_party/taco/LICENSE.txt and THIRD_PARTY_TACO_NOTICE.txt. The packet
// layout and reduction order differ from that legacy adapter; bitwise packet
// equivalence to its AoS kernels is NOT the numerical validation contract.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>
#include <limits>

namespace {
constexpr int kAbi = 1;
constexpr int kGroup = 128;
constexpr int kWarps = 4;
constexpr unsigned kMask = 0xffffffffU;
constexpr float kNorm = 0.08838834764831844055f;

__device__ __forceinline__ float clamp(float x, float lo, float hi) {
  return fminf(fmaxf(x, lo), hi);
}
__device__ __forceinline__ float warp_sum(float x) {
  #pragma unroll
  for (int d = 16; d; d >>= 1) x += __shfl_down_sync(kMask, x, d);
  return x;
}
__device__ __forceinline__ float quarter_sum(float (&x)[4]) {
  float part[4];
  #pragma unroll
  for (int j = 0; j < 4; ++j) {
    float sum = warp_sum(__fmul_rn(x[j], x[j]));
    part[j] = __shfl_sync(kMask, sum, 0);
  }
  const int lane = threadIdx.x & 31;
  float sum = lane < 4 ? part[lane] : 0.f;
  sum = warp_sum(sum);
  return __shfl_sync(kMask, sum, 0);
}
__device__ __forceinline__ float group_max(float (&x)[4]) {
  float value = fmaxf(fmaxf(fabsf(x[0]), fabsf(x[1])),
                      fmaxf(fabsf(x[2]), fabsf(x[3])));
  #pragma unroll
  for (int d = 16; d; d >>= 1)
    value = fmaxf(value, __shfl_down_sync(kMask, value, d));
  return __shfl_sync(kMask, value, 0);
}

// lane l owns coefficients l+32*j: transform distances 1..16 across lanes,
// then 32 and 64 within registers. All 32 lanes in each active warp participate.
__device__ __forceinline__ void hadamard_strided(float (&x)[4]) {
  const int lane = threadIdx.x & 31;
  #pragma unroll
  for (int d = 1; d < 32; d *= 2) {
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float other = __shfl_xor_sync(kMask, x[j], d);
      x[j] = lane & d ? other - x[j] : x[j] + other;
    }
  }
  float a = x[0], b = x[1], c = x[2], d = x[3];
  x[0] = a + b; x[1] = a - b; x[2] = c + d; x[3] = c - d;
  a = x[0]; b = x[1]; c = x[2]; d = x[3];
  x[0] = a + c; x[1] = b + d; x[2] = a - c; x[3] = b - d;
}

__global__ void encode_kernel(const __nv_bfloat16* input, uint8_t* packet,
                              int64_t elements) {
  const int64_t group = int64_t(blockIdx.x) * kWarps + threadIdx.x / 32;
  const int64_t groups = elements / kGroup;
  if (group >= groups) return; // Whole warp exits together; no CTA barrier.
  const int lane = threadIdx.x & 31;
  float x[4];
  #pragma unroll
  for (int j = 0; j < 4; ++j)
    x[j] = __bfloat162float(input[group * kGroup + lane + j * 32]);
  const float variance = __fadd_rn(quarter_sum(x) / 128.f, 1e-6f);
  const float adaptive = clamp(448.f * rsqrtf(fmaxf(variance, 1e-12f)), 1e-3f, 1e3f);
  #pragma unroll
  for (int j = 0; j < 4; ++j) x[j] = __fmul_rn(x[j], adaptive);
  hadamard_strided(x);
  #pragma unroll
  for (int j = 0; j < 4; ++j) x[j] = __fmul_rn(x[j], kNorm);
  const float quant = clamp(group_max(x) / 448.f, 1e-12f, 1e6f);
  if (!lane) {
    reinterpret_cast<float*>(packet + elements)[group] = quant;
    reinterpret_cast<float*>(packet + elements + 4 * groups)[group] = adaptive;
  }
  #pragma unroll
  for (int j = 0; j < 4; ++j) {
    // CUDA's native conversion is finite saturation, round-to-nearest-even.
    // NaN input remains NaN: the untimed finite-input/wire gate rejects it.
    const unsigned q = __nv_cvt_float_to_fp8(x[j] / quant, __NV_SATFINITE, __NV_E4M3);
    const unsigned b1 = __shfl_down_sync(kMask, q, 1);
    const unsigned b2 = __shfl_down_sync(kMask, q, 2);
    const unsigned b3 = __shfl_down_sync(kMask, q, 3);
    if (!(lane & 3))
      *reinterpret_cast<unsigned*>(packet + group * kGroup + lane + 32 * j) =
          q | b1 << 8 | b2 << 16 | b3 << 24;
  }
}

// Packed payload loads give each lane four consecutive coefficients. Distances
// 1/2 operate within registers; distances 4..64 operate across lanes. No shared
// memory, per-coefficient scale loads, or decoded global-memory scratch buffer.
__device__ __forceinline__ void decode_group(const uint8_t* packet, int64_t group,
                                            int64_t elements, __nv_bfloat16 (&out)[4]) {
  const int lane = threadIdx.x & 31;
  const unsigned raw = *reinterpret_cast<const unsigned*>(packet + group * kGroup + lane * 4);
  float quant = 0.f, adaptive = 0.f;
  if (!lane) {
    quant = reinterpret_cast<const float*>(packet + elements)[group];
    adaptive = reinterpret_cast<const float*>(packet + elements + elements / 32)[group];
  }
  quant = __shfl_sync(kMask, quant, 0);
  adaptive = __shfl_sync(kMask, adaptive, 0);
  const __half2 lo(__nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(raw), __NV_E4M3));
  const __half2 hi(__nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(raw >> 16), __NV_E4M3));
  float x[4] = {__fmul_rn(__low2float(lo), quant), __fmul_rn(__high2float(lo), quant),
                __fmul_rn(__low2float(hi), quant), __fmul_rn(__high2float(hi), quant)};
  float a = x[0], b = x[1], c = x[2], d = x[3];
  x[0] = a + b; x[1] = a - b; x[2] = c + d; x[3] = c - d;
  a = x[0]; b = x[1]; c = x[2]; d = x[3];
  x[0] = a + c; x[1] = b + d; x[2] = a - c; x[3] = b - d;
  #pragma unroll
  for (int stride = 1; stride < 32; stride *= 2) {
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float other = __shfl_xor_sync(kMask, x[j], stride);
      x[j] = lane & stride ? other - x[j] : x[j] + other;
    }
  }
  #pragma unroll
  for (int j = 0; j < 4; ++j)
    out[j] = __float2bfloat16_rn(__fmul_rn(x[j], kNorm) / adaptive);
}
__device__ __forceinline__ void store_group(__nv_bfloat16* out, int64_t index,
                                           const __nv_bfloat16 (&value)[4]) {
  unsigned long long raw = 0;
  #pragma unroll
  for (int j = 0; j < 4; ++j)
    raw |= static_cast<unsigned long long>(__bfloat16_as_ushort(value[j])) << (16 * j);
  *reinterpret_cast<unsigned long long*>(out + index) = raw;
}
__global__ void decode_kernel(const uint8_t* packet, __nv_bfloat16* out, int64_t elements) {
  const int64_t group = int64_t(blockIdx.x) * kWarps + threadIdx.x / 32;
  if (group >= elements / kGroup) return;
  __nv_bfloat16 value[4];
  decode_group(packet, group, elements, value);
  store_group(out, group * kGroup + (threadIdx.x & 31) * 4, value);
}

__global__ void decode_reduce_kernel(const __nv_bfloat16* local, const uint8_t* packets,
                                     __nv_bfloat16* out, int64_t elements, int rank, int world) {
  const int64_t group = int64_t(blockIdx.x) * kWarps + threadIdx.x / 32;
  if (group >= elements / kGroup) return;
  const int64_t index = group * kGroup + (threadIdx.x & 31) * 4;
  const int64_t packet_bytes = elements / kGroup * 136;
  __nv_bfloat16 accum[4];
  #pragma unroll
  for (int j = 0; j < 4; ++j) accum[j] = __float2bfloat16_rn(0.f);
  // Wire readiness may differ, but arithmetic source order NEVER does.
  #pragma unroll 1
  for (int source = 0; source < world; ++source) {
    __nv_bfloat16 value[4];
    if (source == rank) {
      const unsigned long long raw = *reinterpret_cast<const unsigned long long*>(local + index);
      #pragma unroll
      for (int j = 0; j < 4; ++j)
        value[j] = __ushort_as_bfloat16(static_cast<unsigned short>(raw >> (16 * j)));
    } else {
      const int slot = source - int(source > rank);
      decode_group(packets + int64_t(slot) * packet_bytes, group, elements, value);
    }
    #pragma unroll
    for (int j = 0; j < 4; ++j)
      accum[j] = __float2bfloat16_rn(__fadd_rn(__bfloat162float(accum[j]), __bfloat162float(value[j])));
  }
  store_group(out, index, accum);
}

__global__ void reduce_kernel(const __nv_bfloat16* local, const __nv_bfloat16* remote,
                              __nv_bfloat16* out, int64_t elements, int rank, int world) {
  const int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= elements) return;
  __nv_bfloat16 accum = __float2bfloat16_rn(0.f);
  #pragma unroll 1
  for (int source = 0; source < world; ++source) {
    const int slot = source - int(source > rank);
    const __nv_bfloat16 value = source == rank ? local[index] : remote[int64_t(slot) * elements + index];
    accum = __float2bfloat16_rn(__fadd_rn(__bfloat162float(accum), __bfloat162float(value)));
  }
  out[index] = accum;
}
__global__ void delay_kernel(unsigned long long cycles) {
  const unsigned long long start = clock64();
  while (clock64() - start < cycles) asm volatile("" : : : "memory");
}

bool aligned(const void* ptr, size_t alignment) {
  return ptr && reinterpret_cast<uintptr_t>(ptr) % alignment == 0;
}
bool shape(int64_t elements) {
  return elements > 0 && elements % kGroup == 0 && elements / 256 < std::numeric_limits<int>::max();
}
bool peers(int rank, int world) { return world >= 2 && world <= 128 && rank >= 0 && rank < world; }
bool separate(const void* a, size_t abytes, const void* b, size_t bbytes) {
  const uintptr_t ap = reinterpret_cast<uintptr_t>(a), bp = reinterpret_cast<uintptr_t>(b);
  if (ap > std::numeric_limits<uintptr_t>::max() - abytes || bp > std::numeric_limits<uintptr_t>::max() - bbytes)
    return false;
  return ap + abytes <= bp || bp + bbytes <= ap;
}
unsigned grid(int64_t elements) { return unsigned((elements / kGroup + kWarps - 1) / kWarps); }
bool reduction_args(const void* local, const void* remote, void* out, int64_t elements,
                    int rank, int world, bool compressed) {
  if (!shape(elements) || !peers(rank, world) || !aligned(local, 8) ||
      !aligned(remote, compressed ? 4 : 2) || !aligned(out, 8)) return false;
  const size_t local_bytes = size_t(elements) * 2;
  const size_t remote_bytes = size_t(world - 1) * (compressed ? size_t(elements / kGroup) * 136 : local_bytes);
  return separate(local, local_bytes, remote, remote_bytes) &&
         separate(local, local_bytes, out, local_bytes) && separate(remote, remote_bytes, out, local_bytes);
}
} // namespace

extern "C" int flux_5090_codec_abi() { return kAbi; }
extern "C" const char* flux_5090_codec_error(int status) { return cudaGetErrorString(cudaError_t(status)); }
extern "C" int flux_5090_encode(const void* input, void* packet, int64_t elements, void* stream) {
  if (!shape(elements) || !aligned(input, 2) || !aligned(packet, 4) ||
      !separate(input, size_t(elements) * 2, packet, size_t(elements / kGroup) * 136))
    return int(cudaErrorInvalidValue);
  encode_kernel<<<grid(elements), 128, 0, cudaStream_t(stream)>>>(
      static_cast<const __nv_bfloat16*>(input), static_cast<uint8_t*>(packet), elements);
  return int(cudaGetLastError());
}
extern "C" int flux_5090_decode(const void* packet, void* out, int64_t elements, void* stream) {
  if (!shape(elements) || !aligned(packet, 4) || !aligned(out, 8) ||
      !separate(packet, size_t(elements / kGroup) * 136, out, size_t(elements) * 2))
    return int(cudaErrorInvalidValue);
  decode_kernel<<<grid(elements), 128, 0, cudaStream_t(stream)>>>(
      static_cast<const uint8_t*>(packet), static_cast<__nv_bfloat16*>(out), elements);
  return int(cudaGetLastError());
}
extern "C" int flux_5090_reduce(const void* local, const void* remote, void* out,
                                 int64_t elements, int rank, int world, void* stream) {
  if (!reduction_args(local, remote, out, elements, rank, world, false)) return int(cudaErrorInvalidValue);
  reduce_kernel<<<unsigned((elements + 255) / 256), 256, 0, cudaStream_t(stream)>>>(
      static_cast<const __nv_bfloat16*>(local), static_cast<const __nv_bfloat16*>(remote),
      static_cast<__nv_bfloat16*>(out), elements, rank, world);
  return int(cudaGetLastError());
}
extern "C" int flux_5090_decode_reduce(const void* local, const void* packets, void* out,
                                        int64_t elements, int rank, int world, void* stream) {
  if (!reduction_args(local, packets, out, elements, rank, world, true)) return int(cudaErrorInvalidValue);
  decode_reduce_kernel<<<grid(elements), 128, 0, cudaStream_t(stream)>>>(
      static_cast<const __nv_bfloat16*>(local), static_cast<const uint8_t*>(packets),
      static_cast<__nv_bfloat16*>(out), elements, rank, world);
  return int(cudaGetLastError());
}
extern "C" int flux_5090_diagnostic_delay(unsigned long long cycles, void* stream) {
  if (!cycles || cycles > 100000000ULL) return int(cudaErrorInvalidValue);
  delay_kernel<<<1, 1, 0, cudaStream_t(stream)>>>(cycles);
  return int(cudaGetLastError());
}
