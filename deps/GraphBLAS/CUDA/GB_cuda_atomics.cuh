//------------------------------------------------------------------------------
// GraphBLAS/CUDA/GB_cuda_atomics: CUDA atomics for GraphBLAS
//------------------------------------------------------------------------------

/*
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * * Redistributions of source code must retain the above copyright
 *   notice, this list of conditions and the following disclaimer.
 * * Redistributions in binary form must reproduce the above copyright
 *   notice, this list of conditions and the following disclaimer in the
 *   documentation and/or other materials provided with the distribution.
 * * Neither the name of NVIDIA CORPORATION nor the names of its
 *   contributors may be used to endorse or promote products derived
 *   from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

//------------------------------------------------------------------------------
// Specializations for different atomic operations on different types
//------------------------------------------------------------------------------

// No 1-byte methods are available (bool, uint8_t, int8_t), because CUDA does
// not support atomicCAS for a single byte.  Instead, to compute a single byte
// atomically, GraphBLAS must operate on a larger temporary type (typically
// uint32_t, but it could also use a 16-bit type), and when all results are
// computed and the kernel launch is done, the final value is copied to the
// single byte result on the host.
//
// The GxB_FC64_t type is supported only by GB_cuda_atomic_add.
//
// GB_cuda_atomic_write, GB_cuda_atomic_times:
//
//      int16_t, uint16_t, int32_t, uint32_t, int64_t, uint64_t,
//      float, double, and GxB_FC32_t (not GxB_FC64_t).
//
// GB_cuda_atomic_min, GB_cuda_atomic_max:
//
//      int16_t, uint16_t, int32_t, uint32_t, int64_t, uint64_t,
//      float, and double (not GxB_FC32_t or GxB_FC64_t).
//
// GB_cuda_atomic_add:
//
//      int16_t, uint16_t, int32_t, uint32_t, int64_t, uint64_t,
//      float, double, GxB_FC32_t, and GxB_FC64_t.
// 
// GB_cuda_atomic_bor, GB_cuda_atomic_band,
// GB_cuda_atomic_bxor, GB_cuda_atomic_bxnor :
//
//      uint16_t, uint32_t, uint64_t
//
// GB_cuda_atomic_lock, GB_cuda_atomic_unlock:
//
//      uint32_t only
//
// GB_PUN is #defined in GB_pun.h as:
// #define GB_PUN(type,value) (*((type *) (&(value))))
// which allows a value to be interpretted as another type but with no
// typecasting.  The value parameter must be an lvalue, not an expression.

#pragma once

template <typename T> __device__ void GB_cuda_atomic_write (T* ptr, T val) ;
template <typename T> __device__ void GB_cuda_atomic_add (T* ptr, T val) ;
template <typename T> __device__ void GB_cuda_atomic_times (T* ptr, T val) ;
template <typename T> __device__ void GB_cuda_atomic_min (T* ptr, T val) ;
template <typename T> __device__ void GB_cuda_atomic_max (T* ptr, T val) ;
template <typename T> __device__ void GB_cuda_atomic_bor (T* ptr, T val) ;
template <typename T> __device__ void GB_cuda_atomic_band (T* ptr, T val) ;
template <typename T> __device__ void GB_cuda_atomic_bxor (T* ptr, T val) ;
template <typename T> __device__ void GB_cuda_atomic_bxnor (T* ptr, T val) ;

__device__ __inline__ void GB_cuda_lock   (uint32_t *mutex) ;
__device__ __inline__ void GB_cuda_unlock (uint32_t *mutex) ;

//------------------------------------------------------------------------------
// GB_cuda_atomic_write
//------------------------------------------------------------------------------

// atomic write (16, 32, and 64 bits)
// no atomic write for GxB_FC64_t

template<> __device__ __inline__ void GB_cuda_atomic_write<int16_t>
(
    int16_t *ptr,   // target to modify
    int16_t val     // value to modify the target with
)
{
    unsigned short int *p = (unsigned short int *) ptr ;
    unsigned short int v = GB_PUN (unsigned short int, val) ;
    unsigned short int assumed ;
    unsigned short int old = *p ;
    do
    {
        // assume the old value
        assumed = old ;
        // modify it atomically:
        old = atomicCAS (p, assumed, v) ;
    }
    while (assumed != old) ;
}

template<> __device__ __inline__ void GB_cuda_atomic_write<uint16_t>
(
    uint16_t *ptr,  // target to modify
    uint16_t val    // value to modify the target with
)
{
    unsigned short int *p = (unsigned short int *) ptr ;
    unsigned short int v = (unsigned short int) val ;
    unsigned short int assumed ;
    unsigned short int old = *p ;
    do
    {
        // assume the old value
        assumed = old ;
        // modify it atomically:
        old = atomicCAS (p, assumed, v) ;
    }
    while (assumed != old) ;
}

template<> __device__ __inline__ void GB_cuda_atomic_write<int32_t>
(
    int32_t *ptr,   // target to modify
    int32_t val     // value to modify the target with
)
{
    // native CUDA method
    atomicExch ((int *) ptr, (int) val) ;
}

template<> __device__ __inline__ void GB_cuda_atomic_write<uint32_t>
(
    uint32_t *ptr,   // target to modify
    uint32_t val     // value to modify the target with
)
{
    // native CUDA method
    atomicExch ((unsigned int *) ptr, (unsigned int) val) ;
}

template<> __device__ __inline__ void GB_cuda_atomic_write<int64_t>
(
    int64_t *ptr,   // target to modify
    int64_t val     // value to modify the target with
)
{
    unsigned long long int *p = (unsigned long long int *) ptr ;
    unsigned long long int v = GB_PUN (unsigned long long int, val) ;
    atomicExch (p, v) ;
}

template<> __device__ __inline__ void GB_cuda_atomic_write<uint64_t>
(
    uint64_t *ptr,  // target to modify
    uint64_t val    // value to modify the target with
)
{
    // native CUDA method
    atomicExch ((unsigned long long int *) ptr, (unsigned long long int) val) ;
}

template<> __device__ __inline__ void GB_cuda_atomic_write<float>
(
    float *ptr,     // target to modify
    float val       // value to modify the target with
)
{
    // native CUDA method
    atomicExch (ptr, val) ;
}

template<> __device__ __inline__ void GB_cuda_atomic_write<double>
(
    double *ptr,    // target to modify
    double val      // value to modify the target with
)
{
    unsigned long long int *p = (unsigned long long int *) ptr ;
    unsigned long long int v = GB_PUN (unsigned long long int, val) ;
    atomicExch (p, v) ;
}

#if 0
template<> __device__ __inline__ void GB_cuda_atomic_write<GxB_FC32_t>
(
    GxB_FC32_t *ptr,     // target to modify
    GxB_FC32_t val       // value to modify the target with
)
{
    unsigned long long int *p = (unsigned long long int *) ptr ;
    unsigned long long int v = GB_PUN (unsigned long long int, val) ;
    atomicExch (p, v) ;
}
#endif

//------------------------------------------------------------------------------
// GB_cuda_atomic_add for built-in types
//------------------------------------------------------------------------------

// types: int and uint [16,32,64], float, double, GxB_FC32_t, complex double

template<> __device__ __inline__ void GB_cuda_atomic_add<int16_t>
(
    int16_t *ptr,   // target to modify
    int16_t val     // value to modify the target with
)
{
    unsigned short int *p = (unsigned short int *) ptr ;
    unsigned short int assumed ;
    unsigned short int old = *p ;
    do
    {
        // assume the old value
        assumed = old ;
        // compute the new value:
        int16_t new_value = GB_PUN (int16_t, assumed) + val ;
        // modify it atomically:
        old = atomicCAS (p, assumed, GB_PUN (unsigned short int, new_value)) ;
    }
    while (assumed != old) ;
}

template<> __device__ __inline__ void GB_cuda_atomic_add<uint16_t>
(
    uint16_t *ptr,  // target to modify
    uint16_t val    // value to modify the target with
)
{
    unsigned short int *p = (unsigned short int *) ptr ;
    unsigned short int v = (unsigned short int) val ;
    unsigned short int assumed ;
    unsigned short int old = *p ;
    do
    {
        // assume the old value
        assumed = old ;
        // modify it atomically:
        old = atomicCAS (p, assumed, assumed + v) ;
    }
    while (assumed != old) ;
}

template<> __device__ __inline__ void GB_cuda_atomic_add<int32_t>
(
    int32_t *ptr,   // target to modify
    int32_t val     // value to modify the target with
)
{
    // native CUDA method
    atomicAdd ((int *) ptr, (int) val) ;
}

template<> __device__ __inline__ void GB_cuda_atomic_add<uint32_t>
(
    uint32_t *ptr,   // target to modify
    uint32_t val     // value to modify the target with
)
{
    // native CUDA method
    atomicAdd ((unsigned int *) ptr, (unsigned int) val) ;
}

template<> __device__ __inline__ void GB_cuda_atomic_add<int64_t>
(
    int64_t *ptr,   // target to modify
    int64_t val     // value to modify the target with
)
{
    unsigned long long int *p = (unsigned long long int *) ptr ;
    unsigned long long int v = GB_PUN (unsigned long long int, val) ;
    atomicAdd (p, v) ;
}

template<> __device__ __inline__ void GB_cuda_atomic_add<uint64_t>
(
    uint64_t *ptr,  // target to modify
    uint64_t val    // value to modify the target with
)
{
    // native CUDA method
    atomicAdd ((unsigned long long int *) ptr, (unsigned long long int) val) ;
}

template<> __device__ __inline__ void GB_cuda_atomic_add<float>
(
    float *ptr,     // target to modify
    float val       // value to modify the target with
)
{
    // native CUDA method
    atomicAdd (ptr, val) ;
}

template<> __device__ __inline__ void GB_cuda_atomic_add<double>
(
    double *ptr,    // target to modify
    double val      // value to modify the target with
)
{
    // native CUDA method
    atomicAdd (ptr, val) ;
}

#if 0
template<> __device__ __inline__ void GB_cuda_atomic_add<GxB_FC32_t>
(
    GxB_FC32_t *ptr,     // target to modify
    GxB_FC32_t val       // value to modify the target with
)
{
    // native CUDA method on each float, real and imaginary parts
    float *p = (float *) ptr ;
    atomicAdd (p  , GB_crealf (val)) ;
    atomicAdd (p+1, GB_cimagf (val)) ;
}

template<> __device__ __inline__ void GB_cuda_atomic_add<GxB_FC64_t>
(
    GxB_FC64_t *ptr,    // target to modify
    GxB_FC64_t val      // value to modify the target with
)
{
    // native CUDA method on each double, real and imaginary parts
    double *p = (double *) ptr ;
    atomicAdd (p  , GB_creal (val)) ;
    atomicAdd (p+1, GB_cimag (val)) ;
}
#endif

//------------------------------------------------------------------------------
// GB_cuda_atomic_times for built-in types
//------------------------------------------------------------------------------

// types: int and uint [16,32,64], float, double, GxB_FC32_t
// no GxB_FC64_t.

template<> __device__ __inline__ void GB_cuda_atomic_times<int16_t>
(
    int16_t *ptr,   // target to modify
    int16_t val     // value to modify the target with
)
{
    unsigned short int *p = (unsigned short int *) ptr ;
    unsigned short int assumed ;
    unsigned short int old = *p ;
    do
    {
        // assume the old value
        assumed = old ;
        // compute the new value:
        int16_t new_value = GB_PUN (int16_t, assumed) * val ;
        // modify it atomically:
        old = atomicCAS (p, assumed, GB_PUN (unsigned short int, new_value)) ;
    }
    while (assumed != old) ;
}

template<> __device__ __inline__ void GB_cuda_atomic_times<uint16_t>
(
    uint16_t *ptr,  // target to modify
    uint16_t val    // value to modify the target with
)
{
    unsigned short int *p = (unsigned short int *) ptr ;
    unsigned short int v = (unsigned short int) val ;
    unsigned short int assumed ;
    unsigned short int old = *p ;
    do
    {
        // assume the old value
        assumed = old ;
        // modify it atomically:
        old = atomicCAS (p, assumed, assumed * v) ;
    }
    while (assumed != old) ;
}

template<> __device__ __inline__ void GB_cuda_atomic_times<int32_t>
(
    int32_t *ptr,   // target to modify
    int32_t val     // value to modify the target with
)
{
    int *p = (int *) ptr ;
    int assumed ;
    int old = *p ;
    do
    {
        // assume the old value
        assumed = old ;
        // compute the new value
        int32_t new_value = GB_PUN (int32_t, assumed) * val ;
        // modify it atomically:
        old = atomicCAS (p, assumed, GB_PUN (int, new_value)) ;
    }
    while (assumed != old) ;
}

template<> __device__ __inline__ void GB_cuda_atomic_times<uint32_t>
(
    uint32_t *ptr,   // target to modify
    uint32_t val     // value to modify the target with
)
{
    unsigned int *p = (unsigned int *) ptr ;
    unsigned int v = (unsigned int) val ;
    unsigned int assumed ;
    unsigned int old = *p ;
    do
    {
        // assume the old value
        assumed = old ;
        // modify it atomically:
        old = atomicCAS (p, assumed, assumed * v) ;
    }
    while (assumed != old) ;
}

template<> __device__ __inline__ void GB_cuda_atomic_times<int64_t>
(
    int64_t *ptr,   // target to modify
    int64_t val     // value to modify the target with
)
{
    unsigned long long int *p = (unsigned long long int *) ptr ;
    unsigned long long int assumed ;
    unsigned long long int old = *p ;
    do
    {
        // assume the old value
        assumed = old ;
        // compute the new value
        int64_t new_value = GB_PUN (int64_t, assumed) * val ;
        // modify it atomically:
        old = atomicCAS (p, assumed, GB_PUN (unsigned long long int, new_value)) ;
    }
    while (assumed != old) ;
}

template<> __device__ __inline__ void GB_cuda_atomic_times<uint64_t>
(
    uint64_t *ptr,  // target to modify
    uint64_t val    // value to modify the target with
)
{
    unsigned long long int *p = (unsigned long long int *) ptr ;
    unsigned long long int v = (unsigned long long int) val ;
    unsigned long long int assumed ;
    unsigned long long int old = *p ;
    do
    {
        // assume the old value
        assumed = old ;
        // modify it atomically:
        old = atomicCAS (p, assumed, assumed * v) ;
    }
    while (assumed != old) ;
}

template<> __device__ __inline__ void GB_cuda_atomic_times<float>
(
    float *ptr,     // target to modify
    float val       // value to modify the target with
)
{
    unsigned int *p = (unsigned int *) ptr ;
    unsigned int assumed ;
    unsigned int old = *p ;
    do
    {
        // assume the old value
        assumed = old ;
        // compute the new value
        float new_value = GB_PUN (float, assumed) * val ;
        // modify it atomically:
        old = atomicCAS (p, assumed, GB_PUN (unsigned int, new_value)) ;
    }
    while (assumed != old) ;
}

template<> __device__ __inline__ void GB_cuda_atomic_times<double>
(
    double *ptr,    // target to modify
    double val      // value to modify the target with
)
{
    unsigned long long int *p = (unsigned long long int *) ptr ;
    unsigned long long int assumed ;
    unsigned long long int old = *p ;
    do
    {
        // assume the old value
        assumed = old ;
        // compute the new value
        double new_value = GB_PUN (double, assumed) * val ;
        // modify it atomically:
        old = atomicCAS (p, assumed, GB_PUN (unsigned long long int, new_value)) ;
    }
    while (assumed != old) ;
}

#if 0
template<> __device__ __inline__ void GB_cuda_atomic_times<GxB_FC32_t>
(
    GxB_FC32_t *ptr,     // target to modify
    GxB_FC32_t val       // value to modify the target with
)
{
    unsigned long long int *p = (unsigned long long int *) ptr ;
    unsigned long long int assumed ;
    unsigned long long int old = *p ;
    do
    {
        // assume the old value
        assumed = old ;
        // compute the new value
        GxB_FC32_t new_value = GB_PUN (GxB_FC32_t, assumed) * val ;
        // modify it atomically:
        old = atomicCAS (p, assumed, GB_PUN (unsigned long long int, new_value)) ;
    }
    while (assumed != old) ;
}
#endif

//------------------------------------------------------------------------------
// GB_cuda_atomic_min
//------------------------------------------------------------------------------

// types: int and uint [16,32,64], float, and double
// no complex types

template<> __device__ __inline__ void GB_cuda_atomic_min<int16_t>
(
    int16_t *ptr,   // target to modify
    int16_t val     // value to modify the target with
)
{
    unsigned short int *p = (unsigned short int *) ptr ;
    unsigned short int assumed ;
    unsigned short int old = *p ;
    do
    {
        // assume the old value
        assumed = old ;
        // compute the new value
        int16_t assumed_int16 = GB_PUN (int16_t, assumed) ;
        int16_t new_value = GB_IMIN (assumed_int16, val) ;
        // modify it atomically:
        old = atomicCAS (p, assumed, GB_PUN (unsigned short int, new_value)) ;
    }
    while (assumed != old) ;
}

template<> __device__ __inline__ void GB_cuda_atomic_min<uint16_t>
(
    uint16_t *ptr,  // target to modify
    uint16_t val    // value to modify the target with
)
{
    unsigned short int *p = (unsigned short int *) ptr ;
    unsigned short int v = (unsigned short int) val ;
    unsigned short int assumed ;
    unsigned short int old = *p ;
    do
    {
        // assume the old value
        assumed = old ;
        // compute the new value
        unsigned short int new_value = GB_IMIN (assumed, v) ;
        // modify it atomically:
        old = atomicCAS (p, assumed, new_value) ;
    }
    while (assumed != old) ;
}

template<> __device__ __inline__ void GB_cuda_atomic_min<int32_t>
(
    int32_t *ptr,   // target to modify
    int32_t val     // value to modify the target with
)
{
    // native CUDA method
    atomicMin ((int *) ptr, (int) val) ;
}

template<> __device__ __inline__ void GB_cuda_atomic_min<uint32_t>
(
    uint32_t *ptr,   // target to modify
    uint32_t val     // value to modify the target with
)
{
    // native CUDA method
    atomicMin ((unsigned int *) ptr, (unsigned int) val) ;
}

template<> __device__ __inline__ void GB_cuda_atomic_min<int64_t>
(
    int64_t *ptr,   // target to modify
    int64_t val     // value to modify the target with
)
{
    // native CUDA method
    atomicMin ((long long int *) ptr, (long long int) val) ;
}

template<> __device__ __inline__ void GB_cuda_atomic_min<uint64_t>
(
    uint64_t *ptr,  // target to modify
    uint64_t val    // value to modify the target with
)
{
    // native CUDA method
    atomicMin ((unsigned long long int *)ptr, (unsigned long long int) val) ;
}

template<> __device__ __inline__ void GB_cuda_atomic_min<float>
(
    float *ptr,     // target to modify
    float val       // value to modify the target with
)
{
    unsigned int *p = (unsigned int *) ptr ;
    unsigned int assumed ;
    unsigned int old = *p ;
    do
    {
        // assume the old value
        assumed = old ;
        // compute the new value
        float new_value = fminf (GB_PUN (float, assumed), val) ;
        // modify it atomically:
        old = atomicCAS (p, assumed, GB_PUN (unsigned int, new_value)) ;
    }
    while (assumed != old) ;
}

template<> __device__ __inline__ void GB_cuda_atomic_min<double>
(
    double *ptr,    // target to modify
    double val      // value to modify the target with
)
{
    unsigned long long int *p = (unsigned long long int *) ptr ;
    unsigned long long int assumed ;
    unsigned long long int old = *p ;
    do
    {
        // assume the old value
        assumed = old ;
        // compute the new value
        double new_value = fmin (GB_PUN (double, assumed), val) ;
        // modify it atomically:
        old = atomicCAS (p, assumed, GB_PUN (unsigned long long int, new_value)) ;
    }
    while (assumed != old) ;
}

//------------------------------------------------------------------------------
// GB_cuda_atomic_max
//------------------------------------------------------------------------------

// types: int and uint [16,32,64], float, and double
// no complex types

template<> __device__ __inline__ void GB_cuda_atomic_max<int16_t>
(
    int16_t *ptr,   // target to modify
    int16_t val     // value to modify the target with
)
{
    unsigned short int *p = (unsigned short int *) ptr ;
    unsigned short int assumed ;
    unsigned short int old = *p ;
    do
    {
        // assume the old value
        assumed = old ;
        // compute the new value
        int16_t assumed_int16 = GB_PUN (int16_t, assumed) ;
        int16_t new_value = GB_IMIN (assumed_int16, val) ;
        // modify it atomically:
        old = atomicCAS (p, assumed, GB_PUN (unsigned short int, new_value)) ;
    }
    while (assumed != old) ;
}

template<> __device__ __inline__ void GB_cuda_atomic_max<uint16_t>
(
    uint16_t *ptr,  // target to modify
    uint16_t val    // value to modify the target with
)
{
    unsigned short int *p = (unsigned short int *) ptr ;
    unsigned short int v = (unsigned short int) val ;
    unsigned short int assumed ;
    unsigned short int old = *p ;
    do
    {
        // assume the old value
        assumed = old ;
        // compute the new value
        unsigned short int new_value = GB_IMIN (assumed, v) ;
        // modify it atomically:
        old = atomicCAS (p, assumed, new_value) ;
    }
    while (assumed != old) ;
}

template<> __device__ __inline__ void GB_cuda_atomic_max<int32_t>
(
    int32_t *ptr,   // target to modify
    int32_t val     // value to modify the target with
)
{
    // native CUDA method
    atomicMax ((int *) ptr, (int) val) ;
}

template<> __device__ __inline__ void GB_cuda_atomic_max<uint32_t>
(
    uint32_t *ptr,   // target to modify
    uint32_t val     // value to modify the target with
)
{
    // native CUDA method
    atomicMax ((unsigned int *) ptr, (unsigned int) val) ;
}

template<> __device__ __inline__ void GB_cuda_atomic_max<int64_t>
(
    int64_t *ptr,   // target to modify
    int64_t val     // value to modify the target with
)
{
    // native CUDA method
    atomicMax ((long long int *) ptr, (long long int) val) ;
}

template<> __device__ __inline__ void GB_cuda_atomic_max<uint64_t>
(
    uint64_t *ptr,  // target to modify
    uint64_t val    // value to modify the target with
)
{
    // native CUDA method
    atomicMax ((unsigned long long int *)ptr, (unsigned long long int) val) ;
}

template<> __device__ __inline__ void GB_cuda_atomic_max<float>
(
    float *ptr,     // target to modify
    float val       // value to modify the target with
)
{
    unsigned int *p = (unsigned int *) ptr ;
    unsigned int assumed ;
    unsigned int old = *p ;
    do
    {
        // assume the old value
        assumed = old ;
        // compute the new value
        float new_value = fmaxf (GB_PUN (float, assumed), val) ;
        // modify it atomically:
        old = atomicCAS (p, assumed, GB_PUN (unsigned int, new_value)) ;
    }
    while (assumed != old) ;
}

template<> __device__ __inline__ void GB_cuda_atomic_max<double>
(
    double *ptr,    // target to modify
    double val      // value to modify the target with
)
{
    unsigned long long int *p = (unsigned long long int *) ptr ;
    unsigned long long int assumed ;
    unsigned long long int old = *p ;
    do
    {
        // assume the old value
        assumed = old ;
        // compute the new value
        double new_value = fmax (GB_PUN (double, assumed), val) ;
        // modify it atomically:
        old = atomicCAS (p, assumed, GB_PUN (unsigned long long int, new_value)) ;
    }
    while (assumed != old) ;
}

//------------------------------------------------------------------------------
// GB_cuda_atomic_bor
//------------------------------------------------------------------------------

// bitwise: on uint [16,32,64]

template<> __device__ __inline__ void GB_cuda_atomic_bor<uint16_t>
(
    uint16_t *ptr,  // target to modify
    uint16_t val    // value to modify the target with
)
{
    unsigned short int *p = (unsigned short int *) ptr ;
    unsigned short int v = (unsigned short int) val ;
    unsigned short int assumed ;
    unsigned short int old = *p ;
    do
    {
        // assume the old value
        assumed = old ;
        // modify it atomically:
        old = atomicCAS (p, assumed, assumed | v) ;
    }
    while (assumed != old) ;
}

template<> __device__ __inline__ void GB_cuda_atomic_bor<uint32_t>
(
    uint32_t *ptr,   // target to modify
    uint32_t val     // value to modify the target with
)
{
    // native CUDA method
    atomicOr ((unsigned int *) ptr, (unsigned int) val) ;
}

template<> __device__ __inline__ void GB_cuda_atomic_bor<uint64_t>
(
    uint64_t *ptr,  // target to modify
    uint64_t val    // value to modify the target with
)
{
    // native CUDA method
    atomicOr ((unsigned long long int *) ptr, (unsigned long long int) val) ;
}

//------------------------------------------------------------------------------
// GB_cuda_atomic_band
//------------------------------------------------------------------------------

// bitwise: on uint [16,32,64]

template<> __device__ __inline__ void GB_cuda_atomic_band<uint16_t>
(
    uint16_t *ptr,  // target to modify
    uint16_t val    // value to modify the target with
)
{
    unsigned short int *p = (unsigned short int *) ptr ;
    unsigned short int v = (unsigned short int) val ;
    unsigned short int assumed ;
    unsigned short int old = *p ;
    do
    {
        // assume the old value
        assumed = old ;
        // modify it atomically:
        old = atomicCAS (p, assumed, assumed & v) ;
    }
    while (assumed != old) ;
}

template<> __device__ __inline__ void GB_cuda_atomic_band<uint32_t>
(
    uint32_t *ptr,   // target to modify
    uint32_t val     // value to modify the target with
)
{
    // native CUDA method
    atomicAnd ((unsigned int *) ptr, (unsigned int) val) ;
}

template<> __device__ __inline__ void GB_cuda_atomic_band<uint64_t>
(
    uint64_t *ptr,  // target to modify
    uint64_t val    // value to modify the target with
)
{
    // native CUDA method
    atomicAnd ((unsigned long long int *) ptr, (unsigned long long int) val) ;
}

//------------------------------------------------------------------------------
// GB_cuda_atomic_bxor
//------------------------------------------------------------------------------

// bitwise: on uint [16,32,64]

template<> __device__ __inline__ void GB_cuda_atomic_bxor<uint16_t>
(
    uint16_t *ptr,  // target to modify
    uint16_t val    // value to modify the target with
)
{
    unsigned short int *p = (unsigned short int *) ptr ;
    unsigned short int v = (unsigned short int) val ;
    unsigned short int assumed ;
    unsigned short int old = *p ;
    do
    {
        // assume the old value
        assumed = old ;
        // modify it atomically:
        old = atomicCAS (p, assumed, assumed ^ v) ;
    }
    while (assumed != old) ;
}

template<> __device__ __inline__ void GB_cuda_atomic_bxor<uint32_t>
(
    uint32_t *ptr,   // target to modify
    uint32_t val     // value to modify the target with
)
{
    // native CUDA method
    atomicXor ((unsigned int *) ptr, (unsigned int) val) ;
}

template<> __device__ __inline__ void GB_cuda_atomic_bxor<uint64_t>
(
    uint64_t *ptr,  // target to modify
    uint64_t val    // value to modify the target with
)
{
    // native CUDA method
    atomicXor ((unsigned long long int *) ptr, (unsigned long long int) val) ;
}

//------------------------------------------------------------------------------
// GB_cuda_atomic_bxnor
//------------------------------------------------------------------------------

// bitwise: on uint [16,32,64]

template<> __device__ __inline__ void GB_cuda_atomic_bxnor<uint16_t>
(
    uint16_t *ptr,  // target to modify
    uint16_t val    // value to modify the target with
)
{
    unsigned short int *p = (unsigned short int *) ptr ;
    unsigned short int v = (unsigned short int) val ;
    unsigned short int assumed ;
    unsigned short int old = *p ;
    do
    {
        // assume the old value
        assumed = old ;
        // modify it atomically:
        old = atomicCAS (p, assumed, ~(assumed ^ v)) ;
    }
    while (assumed != old) ;
}

template<> __device__ __inline__ void GB_cuda_atomic_bxnor<uint32_t>
(
    uint32_t *ptr,   // target to modify
    uint32_t val     // value to modify the target with
)
{
    unsigned int *p = (unsigned int *) ptr ;
    unsigned int v = (unsigned int) val ;
    unsigned int assumed ;
    unsigned int old = *p ;
    do
    {
        // assume the old value
        assumed = old ;
        // modify it atomically:
        old = atomicCAS (p, assumed, ~(assumed ^ v)) ;
    }
    while (assumed != old) ;
}

template<> __device__ __inline__ void GB_cuda_atomic_bxnor<uint64_t>
(
    uint64_t *ptr,  // target to modify
    uint64_t val    // value to modify the target with
)
{
    unsigned long long int *p = (unsigned long long int *) ptr ;
    unsigned long long int v = (unsigned long long int) val ;
    unsigned long long int assumed ;
    unsigned long long int old = *p ;
    do
    {
        // assume the old value
        assumed = old ;
        // modify it atomically:
        old = atomicCAS (p, assumed, ~(assumed ^ v)) ;
    }
    while (assumed != old) ;
}

//------------------------------------------------------------------------------
// GB_cuda_lock/unlock: set/clear a mutex for a critical section
//------------------------------------------------------------------------------

__device__ __inline__ void GB_cuda_lock (uint32_t *mutex)
{
    int old ;
    do
    {
        old = atomicCAS (mutex, 0, 1) ;
    }
    while (old == 1) ;
}

__device__ __inline__ void GB_cuda_unlock (uint32_t *mutex)
{
    int old ;
    do
    {
        old = atomicCAS (mutex, 1, 0) ;
    }
    while (old == 0) ;
}

