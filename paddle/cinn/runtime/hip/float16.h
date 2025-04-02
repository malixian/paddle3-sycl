// Copyright (c) 2022 CINN Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef CINN_COMMON_FLOAT16_H
#define CINN_COMMON_FLOAT16_H

#ifdef __cplusplus
#pragma once
#endif  // __cplusplus

#if defined(_M_X64) || defined(__x86_64__) || defined(_M_IX86) || \
    defined(__i386__)
#define __CINN_x86__
#include <immintrin.h>
#endif

#include <stdint.h>

#include <cmath>

#ifdef CINN_WITH_HIP
#include <cuda.h>
#define CINN_HIP_FP16
#include <hip_fp16.h>
#endif  // CINN_WITH_HIP

#ifdef __cplusplus
#ifndef _WIN32
#define CINN_ALIGN(x) __attribute__((aligned(x)))
#else  // _WIN32
#define CINN_ALIGN(x) __declspec(align(x))
#endif  // _WIN32

#else  // __cplusplus
#define CINN_ALIGN(x)
#endif  // __cplusplus

// The `HOST` macro definition is not used here, it has a potential
// conflict with the enumeration `kHOST` representing the backend.
#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif

#ifdef __cplusplus
namespace cinn {
namespace common {
#endif  // __cplusplus

// Use CINN_ALIGNED(2) to ensure that each float16 will be allocated
// and aligned at least on a 2-byte boundary, which leads to efficient
// memory access of float16 struct and also makes float16 compatible
// with HIP half
struct CINN_ALIGN(2) float16 {
  uint16_t x;

#ifdef __cplusplus
  // The following defaulted special class member functions
  // are added to make float16 pass the std::is_trivial test
  float16() = default;
  float16(const float16& o) = default;
  float16& operator=(const float16& o) = default;
  float16(float16&& o) = default;
  float16& operator=(float16&& o) = default;
  ~float16() = default;

// Constructors
#ifdef CINN_HIP_FP16
  __host__ __device__ inline explicit float16(const half& h) {
    x = reinterpret_cast<__half_raw*>(const_cast<half*>(&h))->x;
  }
#endif  // CINN_HIP_FP16

  __host__ __device__ inline explicit float16(float val) {
    half tmp = __float2half(val);
    x = *reinterpret_cast<uint16_t*>(&tmp);
  }

  __host__ __device__ inline explicit float16(bool b) : x(b ? 0x3c00 : 0) {}

  template <class T>
  __host__ __device__ inline explicit float16(const T& val)
      : x(float16(static_cast<float>(val)).x) {}

// Assignment operators
#ifdef CINN_HIP_FP16
  __host__ __device__ inline float16& operator=(const half& rhs) {
    x = reinterpret_cast<__half_raw*>(const_cast<half*>(&rhs))->x;
    return *this;
  }
#endif

  __host__ __device__ inline float16& operator=(bool b) {
    x = b ? 0x3c00 : 0;
    return *this;
  }

  __host__ __device__ inline float16& operator=(int8_t val) {
    x = float16(val).x;
    return *this;
  }

  __host__ __device__ inline float16& operator=(uint8_t val) {
    x = float16(val).x;
    return *this;
  }

  __host__ __device__ inline float16& operator=(int16_t val) {
    x = float16(val).x;
    return *this;
  }

  __host__ __device__ inline float16& operator=(uint16_t val) {
    x = float16(val).x;
    return *this;
  }

  __host__ __device__ inline float16& operator=(int32_t val) {
    x = float16(val).x;
    return *this;
  }

  __host__ __device__ inline float16& operator=(uint32_t val) {
    x = float16(val).x;
    return *this;
  }

  __host__ __device__ inline float16& operator=(int64_t val) {
    x = float16(val).x;
    return *this;
  }

  __host__ __device__ inline float16& operator=(uint64_t val) {
    x = float16(val).x;
    return *this;
  }

  __host__ __device__ inline float16& operator=(float val) {
    x = float16(val).x;
    return *this;
  }

  __host__ __device__ inline float16& operator=(double val) {
    x = float16(val).x;
    return *this;
  }

// Conversion operators
#ifdef CINN_HIP_FP16
  __host__ __device__ inline half to_half() const {
    __half_raw h;
    h.x = x;
    return half(h);
  }
#endif  // CINN_HIP_FP16

  __host__ __device__ inline operator float() const {
    half tmp = *reinterpret_cast<const half*>(this);
    return __half2float(tmp);
  }

  __host__ __device__ inline explicit operator bool() const {
    return (x & 0x7fff) != 0;
  }

  __host__ __device__ inline explicit operator int8_t() const {
    return static_cast<int8_t>(static_cast<float>(*this));
  }

  __host__ __device__ inline explicit operator uint8_t() const {
    return static_cast<uint8_t>(static_cast<float>(*this));
  }

  __host__ __device__ inline explicit operator int16_t() const {
    return static_cast<int16_t>(static_cast<float>(*this));
  }

  __host__ __device__ inline explicit operator uint16_t() const {
    return static_cast<uint16_t>(static_cast<float>(*this));
  }

  __host__ __device__ inline explicit operator int32_t() const {
    return static_cast<int32_t>(static_cast<float>(*this));
  }

  __host__ __device__ inline explicit operator uint32_t() const {
    return static_cast<uint32_t>(static_cast<float>(*this));
  }

  __host__ __device__ inline explicit operator int64_t() const {
    return static_cast<int64_t>(static_cast<float>(*this));
  }

  __host__ __device__ inline explicit operator uint64_t() const {
    return static_cast<uint64_t>(static_cast<float>(*this));
  }

  __host__ __device__ inline operator double() const {
    return static_cast<double>(static_cast<float>(*this));
  }

 private:
  union Bits {
    float f;
    int32_t si;
    uint32_t ui;
  };

  static const int shift = 13;
  static const int shiftSign = 16;

  static const int32_t infN = 0x7F800000;
  static const int32_t maxN = 0x477FE000;  // max flt16 as flt32
  static const int32_t minN = 0x38800000;  // min flt16 normal as flt32
  static const int32_t sigN = 0x80000000;  // sign bit

  static constexpr int32_t infC = infN >> shift;
  static constexpr int32_t nanN = (infC + 1)
                                  << shift;  // minimum flt16 nan as float32
  static constexpr int32_t maxC = maxN >> shift;
  static constexpr int32_t minC = minN >> shift;
  static constexpr int32_t sigC = sigN >> shiftSign;

  static const int32_t mulN = 0x52000000;  // (1 << 23) / minN
  static const int32_t mulC = 0x33800000;  // minN / (1 << (23 - shift))
  static const int32_t subC = 0x003FF;     // max flt32 subnormal downshifted
  static const int32_t norC = 0x00400;     // min flt32 normal downshifted

  static constexpr int32_t maxD = infC - maxC - 1;
  static constexpr int32_t minD = minC - subC - 1;
#endif  // __cplusplus
};

struct CINN_ALIGN(32) float8 {
  float x, y, z, w, v, u, t, s;
};

struct CINN_ALIGN(16) half8 {
  float16 x, y, z, w, v, u, t, s;
};

struct CINN_ALIGN(8) half4 {
  float16 x, y, z, w;
};

#ifdef __cplusplus
// Arithmetic operators on GPU
// HIP 9.0 provides built-in arithmetic operators for half while
// HIP 7.5 and 8.0 do not. The arithmetic operators defined here are
// for users to write similar HIP code in HIP 7.5 and 8.0 as in
// HIP 9.0 regarding the half data type.
// ROCM has built-in arithmetic operators as not defined
// __HIP_NO_HALF_OPERATORS__
#if defined(CINN_HIP_FP16)
__device__ inline half operator+(const half& a, const half& b) {
  return __hadd(a, b);
}

__device__ inline half operator-(const half& a, const half& b) {
  return __hsub(a, b);
}

__device__ inline half operator*(const half& a, const half& b) {
  return __hmul(a, b);
}

__device__ inline half operator/(const half& a, const half& b) {
  float num = __half2float(a);
  float denom = __half2float(b);
  return __float2half(num / denom);
}

__device__ inline half operator-(const half& a) {
  return __hneg(a);
}

__device__ inline half& operator+=(half& a, const half& b) {  // NOLINT
  a = a + b;
  return a;
}

__device__ inline half& operator-=(half& a, const half& b) {  // NOLINT
  a = a - b;
  return a;
}

__device__ inline half& operator*=(half& a, const half& b) {  // NOLINT
  a = a * b;
  return a;
}

__device__ inline half& operator/=(half& a, const half& b) {  // NOLINT
  a = a / b;
  return a;
}

__device__ inline bool operator==(const half& a, const half& b) {
  return __heq(a, b);
}

__device__ inline bool operator!=(const half& a, const half& b) {
  return __hne(a, b);
}

__device__ inline bool operator<(const half& a, const half& b) {
  return __hlt(a, b);
}

__device__ inline bool operator<=(const half& a, const half& b) {
  return __hle(a, b);
}

__device__ inline bool operator>(const half& a, const half& b) {
  return __hgt(a, b);
}

__device__ inline bool operator>=(const half& a, const half& b) {
  return __hge(a, b);
}

#endif  // CINN_HIP_FP16

// Arithmetic operators for float16 on GPU
__host__ __device__ inline float16 operator+(const float16& a,
                                             const float16& b) {
  return float16(__hadd(a.to_half(), b.to_half()));
}

__host__ __device__ inline float16 operator-(const float16& a,
                                             const float16& b) {
  return float16(__hsub(a.to_half(), b.to_half()));
}

__host__ __device__ inline float16 operator*(const float16& a,
                                             const float16& b) {
  return float16(__hmul(a.to_half(), b.to_half()));
}

__host__ __device__ inline float16 operator/(const float16& a,
                                             const float16& b) {
  // TODO(kexinzhao): check which cuda version starts to support __hdiv
  float num = __half2float(a.to_half());
  float denom = __half2float(b.to_half());
  return float16(num / denom);
}

__host__ __device__ inline float16 operator-(const float16& a) {
  return float16(__hneg(a.to_half()));
}

__host__ __device__ inline float16& operator+=(float16& a,          // NOLINT
                                               const float16& b) {  // NOLINT
  a = a + b;
  return a;
}

__host__ __device__ inline float16& operator-=(float16& a,          // NOLINT
                                               const float16& b) {  // NOLINT
  a = a - b;
  return a;
}

__host__ __device__ inline float16& operator*=(float16& a,          // NOLINT
                                               const float16& b) {  // NOLINT
  a = a * b;
  return a;
}

__host__ __device__ inline float16& operator/=(float16& a,          // NOLINT
                                               const float16& b) {  // NOLINT
  a = a / b;
  return a;
}

__host__ __device__ inline bool operator==(const float16& a, const float16& b) {
  return __heq(a.to_half(), b.to_half());
}

__host__ __device__ inline bool operator!=(const float16& a, const float16& b) {
  return __hne(a.to_half(), b.to_half());
}

__host__ __device__ inline bool operator<(const float16& a, const float16& b) {
  return __hlt(a.to_half(), b.to_half());
}

__host__ __device__ inline bool operator<=(const float16& a, const float16& b) {
  return __hle(a.to_half(), b.to_half());
}

__host__ __device__ inline bool operator>(const float16& a, const float16& b) {
  return __hgt(a.to_half(), b.to_half());
}

__host__ __device__ inline bool operator>=(const float16& a, const float16& b) {
  return __hge(a.to_half(), b.to_half());
}
#endif  // __cplusplus

__host__ __device__ inline float16 raw_uint16_to_float16(uint16_t a) {
  float16 res;
  res.x = a;
  return res;
}

__host__ __device__ inline bool(isnan)(const float16& a) {
  return __hisnan(a.to_half());
}

__host__ __device__ inline bool(isinf)(const float16& a) {
  return (a.x & 0x7fff) == 0x7c00;
}

__host__ __device__ inline bool(isfinite)(const float16& a) {
  return !((isnan)(a)) && !((isinf)(a));
}

__host__ __device__ inline float16(abs)(const float16& a) {
  return static_cast<float16>(__habs(a.to_half()));
}

__host__ __device__ inline float16(log)(const float16& a) {
  return float16(std::log(static_cast<float>(a)));
}

#ifdef __cplusplus
}  // namespace common
}  // namespace cinn
#endif  // __cplusplus

#if defined(__cplusplus) && defined(CINN_HIP_FP16)
__device__ inline cinn::common::float16 __shfl_sync(unsigned mask,
                                                    cinn::common::float16 var,
                                                    int srcLane,
                                                    int width = warpSize) {
  return cinn::common::float16(
      __shfl_sync(mask, var.to_half(), srcLane, width));
}

__device__ inline cinn::common::float16 __shfl_up_sync(
    unsigned mask,
    cinn::common::float16 var,
    unsigned int delta,
    int width = warpSize) {
  return cinn::common::float16(
      __shfl_up_sync(mask, var.to_half(), delta, width));
}

__device__ inline cinn::common::float16 __shfl_down_sync(
    unsigned mask,
    cinn::common::float16 var,
    unsigned int delta,
    int width = warpSize) {
  return cinn::common::float16(
      __shfl_down_sync(mask, var.to_half(), delta, width));
}

__device__ inline cinn::common::float16 __shfl_xor_sync(
    unsigned mask,
    cinn::common::float16 var,
    int laneMask,
    int width = warpSize) {
  return cinn::common::float16(
      __shfl_xor_sync(mask, var.to_half(), laneMask, width));
}

__host__ __device__ inline cinn::common::float16 max(
    const cinn::common::float16& a, const cinn::common::float16& b) {
  return a > b ? a : b;
}
__host__ __device__ inline cinn::common::float16 min(
    const cinn::common::float16& a, const cinn::common::float16& b) {
  return a < b ? a : b;
}
#endif  // __cplusplus && CINN_HIP_FP16

#endif  // CINN_COMMON_FLOAT16_H
