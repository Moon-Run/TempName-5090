// Include after op1_abc.cu. TACO arithmetic attribution and BSD-3-Clause license:
// tools/a800/phase4/TACO_LICENSE.txt. R follows tools/h100/phase6/ma2_decode.cuh;
// only addressing changes from peer SoA to completed local NCCL AoS packets.
namespace op3_r {
struct Packet { unsigned raw; float quant, adaptive; };

__device__ __forceinline__ Packet load_remote(
    const unsigned char* packets, int rank, int total, int group, int source) {
  const int lane = threadIdx.x & 31;
  const int slot = source - int(source > rank);
  const unsigned char* packet = packets + (size_t(slot) * total + group) * 136;
  Packet p;
  asm volatile(
      "ld.global.u32 %0, [%3];\n\t"
      "ld.global.f32 %1, [%4];\n\t"
      "ld.global.f32 %2, [%5];\n\t"
      : "=r"(p.raw), "=f"(p.quant), "=f"(p.adaptive)
      : "l"(packet + lane * 4), "l"(packet + 128), "l"(packet + 132)
      : "memory");
  return p;
}

__device__ __forceinline__ unsigned long long load_local(
    const __nv_bfloat16* local, int group) {
  const __nv_bfloat16* address = local + group * 128 + (threadIdx.x & 31) * 4;
  unsigned long long raw;
  asm volatile("ld.global.u64 %0, [%1];" : "=l"(raw) : "l"(address) : "memory");
  return raw;
}

__device__ __forceinline__ void fold(const Packet& p, __nv_bfloat16 (&accum)[4]) {
  const int lane = threadIdx.x & 31;
  const __half2 h01(__nv_cvt_fp8x2_to_halfraw2(
      static_cast<__nv_fp8x2_storage_t>(p.raw), __NV_E4M3));
  const __half2 h23(__nv_cvt_fp8x2_to_halfraw2(
      static_cast<__nv_fp8x2_storage_t>(p.raw >> 16), __NV_E4M3));
  // Explicit rounded products prevent a register butterfly from becoming an
  // FFMA. Coefficients experience distances 1,2,4,8,16,32,64 in original order.
  float x[4] = {__fmul_rn(__low2float(h01), p.quant),
                __fmul_rn(__high2float(h01), p.quant),
                __fmul_rn(__low2float(h23), p.quant),
                __fmul_rn(__high2float(h23), p.quant)};
  float a = x[0], b = x[1], c = x[2], d = x[3];
  x[0] = a + b; x[1] = a - b; x[2] = c + d; x[3] = c - d;
  a = x[0]; b = x[1]; c = x[2]; d = x[3];
  x[0] = a + c; x[1] = b + d; x[2] = a - c; x[3] = b - d;
  #pragma unroll
  for (int distance = 1; distance < 32; distance *= 2) {
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float other = __shfl_xor_sync(0xffffffff, x[j], distance);
      x[j] = (lane & distance) ? other - x[j] : x[j] + other;
    }
  }
  float adaptive = p.adaptive;
  if (adaptive < 1e-12f) adaptive = 1e-12f;
  #pragma unroll
  for (int j = 0; j < 4; ++j) {
    x[j] *= rsqrtf(128.f);
    const __nv_bfloat16 decoded = __float2bfloat16(check_nan_inf(x[j]) / adaptive);
    accum[j] = __float2bfloat16(__bfloat162float(accum[j]) + __bfloat162float(decoded));
  }
}

template<bool Consumer>
__device__ __forceinline__ void store(__nv_bfloat16* output,
    const __nv_bfloat16* bias, const __nv_bfloat16* residual,
    int group, const __nv_bfloat16 (&accum)[4]) {
  const int index = group * 128 + (threadIdx.x & 31) * 4;
  unsigned long long raw = 0;
  #pragma unroll
  for (int j = 0; j < 4; ++j) {
    __nv_bfloat16 value = accum[j];
    if constexpr (Consumer) {
      // CR3 consumer semantics: projection+bias rounds BF16, then residual+
      // that result rounds BF16 again. Never merge the three FP32 values.
      value = __float2bfloat16(__bfloat162float(value) +
                               __bfloat162float(bias[(index + j) % 4096]));
      value = __float2bfloat16(__bfloat162float(residual[index + j]) +
                               __bfloat162float(value));
    }
    raw |= static_cast<unsigned long long>(__bfloat16_as_ushort(value)) << (16 * j);
  }
  asm volatile("st.global.u64 [%0], %1;" : : "l"(output + index), "l"(raw) : "memory");
}
}  // namespace op3_r

template<bool Consumer>
__global__ void op3_r_decode_kernel(const unsigned char* packets,
    const __nv_bfloat16* local, __nv_bfloat16* output,
    const __nv_bfloat16* bias, const __nv_bfloat16* residual, int rank, int total) {
  const int group = blockIdx.x * 4 + threadIdx.x / 32;
  __nv_bfloat16 accum[4];
  #pragma unroll
  for (int j = 0; j < 4; ++j) accum[j] = __float2bfloat16(0.f);
  op3_r::Packet current = op3_r::load_remote(packets, rank, total, group, (rank + 1) & 7);
  #pragma unroll 1
  for (int offset = 1; offset < 7; ++offset) {
    const op3_r::Packet next = op3_r::load_remote(
        packets, rank, total, group, (rank + offset + 1) & 7);
    op3_r::fold(current, accum);
    current = next;
  }
  const unsigned long long last = op3_r::load_local(local, group);
  op3_r::fold(current, accum);
  #pragma unroll
  for (int j = 0; j < 4; ++j) {
    const __nv_bfloat16 value = __ushort_as_bfloat16(static_cast<unsigned short>(last >> (16 * j)));
    accum[j] = __float2bfloat16(__bfloat162float(accum[j]) + __bfloat162float(value));
  }
  op3_r::store<Consumer>(output, bias, residual, group, accum);
}

// One warp owns one H128 group. Lane l keeps original coefficients l+32*j.
// All original warp reductions retain their lane order. The final four-partial
// reduction also uses the upstream 32-lane helper (including its zero lanes).
__device__ __forceinline__ float op3_quarter_sum(float (&values)[4]) {
  float sums[4];
  #pragma unroll
  for (int j = 0; j < 4; ++j) {
    sums[j] = warpReduceSum(__fmul_rn(values[j], values[j]));
    sums[j] = __shfl_sync(0xffffffff, sums[j], 0);
  }
  const int lane = threadIdx.x & 31;
  float sum = lane == 0 ? sums[0] : lane == 1 ? sums[1] :
              lane == 2 ? sums[2] : lane == 3 ? sums[3] : 0.f;
  sum = warpReduceSum(sum);
  return __shfl_sync(0xffffffff, sum, 0);
}

__device__ __forceinline__ float op3_quarter_max(float (&values)[4]) {
  float maxima[4];
  #pragma unroll
  for (int j = 0; j < 4; ++j) {
    maxima[j] = warpReduceMax(fabsf(values[j]));
    maxima[j] = __shfl_sync(0xffffffff, maxima[j], 0);
  }
  const int lane = threadIdx.x & 31;
  float value = lane == 0 ? maxima[0] : lane == 1 ? maxima[1] :
                lane == 2 ? maxima[2] : lane == 3 ? maxima[3] : 0.f;
  value = warpReduceMax(value);
  return __shfl_sync(0xffffffff, value, 0);
}

__global__ void op3_warp_encode_kernel(const __nv_bfloat16* input,
    unsigned char* packets, int rank, int total) {
  const int group = blockIdx.x * 4 + threadIdx.x / 32;
  const int lane = threadIdx.x & 31;
  const int slot = blockIdx.y, destination = slot + int(slot >= rank);
  const __nv_bfloat16* source = input + (size_t(destination) * total + group) * 128;
  unsigned char* packet = packets + (size_t(slot) * total + group) * 136;
  float x[4];
  #pragma unroll
  for (int j = 0; j < 4; ++j) x[j] = __bfloat162float(source[lane + 32*j]);
  const float sum = op3_quarter_sum(x);
  // The upstream reduction separates squaring and addition; explicit RN
  // operations keep this kernel from contracting across the new register map.
  float variance = __fadd_rn(sum / 128.f, 1e-6f);
  if (variance < 1e-12f) variance = 1e-12f;
  const float adaptive = clampf(448.f * rsqrtf(variance), 1e-3f, 1e3f);
  #pragma unroll
  for (int j = 0; j < 4; ++j) x[j] = __fmul_rn(x[j], adaptive);
  #pragma unroll
  for (int distance = 1; distance < 32; distance *= 2) {
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float neighbor = __shfl_xor_sync(0xffffffff, x[j], distance);
      x[j] = (lane & distance) ? neighbor - x[j] : x[j] + neighbor;
    }
  }
  float a = x[0], b = x[1], c = x[2], d = x[3];
  x[0] = a + b; x[1] = a - b; x[2] = c + d; x[3] = c - d;  // d=32
  a = x[0]; b = x[1]; c = x[2]; d = x[3];
  x[0] = a + c; x[1] = b + d; x[2] = a - c; x[3] = b - d;  // d=64
  #pragma unroll
  for (int j = 0; j < 4; ++j) x[j] *= rsqrtf(128.f);
  const float maximum = op3_quarter_max(x);
  const float mv = maximum < 1e-12f ? 1e-12f : maximum;
  const float quant = clampf(safe_divide(mv, 448.f), 1e-12f, 1e6f);
  if (lane == 0) {
    *reinterpret_cast<float*>(packet + 128) = quant;
    *reinterpret_cast<float*>(packet + 132) = adaptive;
  }
  #pragma unroll
  for (int j = 0; j < 4; ++j) {
    const float scaled = clamp_fp8(safe_divide(check_nan_inf(x[j]), quant));
    const unsigned q8 = __nv_cvt_float_to_fp8(scaled, __NV_SATFINITE, __NV_E4M3);
    const unsigned b1 = __shfl_down_sync(0xffffffff, q8, 1);
    const unsigned b2 = __shfl_down_sync(0xffffffff, q8, 2);
    const unsigned b3 = __shfl_down_sync(0xffffffff, q8, 3);
    if ((lane & 3) == 0)
      *reinterpret_cast<unsigned*>(packet + lane + 32*j) = q8 | b1 << 8 | b2 << 16 | b3 << 24;
  }
}

static bool op3_shape(int rank, int total) { return rank >= 0 && rank < 8 && total == 8192; }
static bool op3_aligned(const void* address, size_t alignment) {
  return address && reinterpret_cast<uintptr_t>(address) % alignment == 0;
}

extern "C" int op3_r_decode(const void* wire, const void* local, void* output,
    const void* bias, const void* residual, int rank, int total, void* stream) {
  if (!op3_shape(rank, total) || !op3_aligned(wire, 4) || !op3_aligned(local, 8) ||
      !op3_aligned(output, 8) || output == local || output == residual ||
      ((bias == nullptr) != (residual == nullptr)) ||
      (bias && (!op3_aligned(bias, 8) || !op3_aligned(residual, 8))))
    return int(cudaErrorInvalidValue);
  if (bias)
    op3_r_decode_kernel<true><<<total/4, 128, 0, (cudaStream_t)stream>>>(
        static_cast<const unsigned char*>(wire), static_cast<const __nv_bfloat16*>(local),
        static_cast<__nv_bfloat16*>(output), static_cast<const __nv_bfloat16*>(bias),
        static_cast<const __nv_bfloat16*>(residual), rank, total);
  else
    op3_r_decode_kernel<false><<<total/4, 128, 0, (cudaStream_t)stream>>>(
        static_cast<const unsigned char*>(wire), static_cast<const __nv_bfloat16*>(local),
        static_cast<__nv_bfloat16*>(output), nullptr, nullptr, rank, total);
  return int(cudaGetLastError());
}

extern "C" int op3_encode_warp(const void* input, void* wire, int rank, int total, void* stream) {
  if (!op3_shape(rank, total) || !op3_aligned(input, 2) || !op3_aligned(wire, 4))
    return int(cudaErrorInvalidValue);
  op3_warp_encode_kernel<<<dim3(total/4, 7), 128, 0, (cudaStream_t)stream>>>(
      static_cast<const __nv_bfloat16*>(input), static_cast<unsigned char*>(wire), rank, total);
  return int(cudaGetLastError());
}
