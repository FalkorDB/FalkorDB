//------------------------------------------------------------------------------
// GB_jit_kernel_AxB_saxpy5: C+=A*B, C is full, A bitmap/full, B sparse/hyper
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "include/GB_AxB_saxpy3_template.h"

GB_JIT_GLOBAL GB_JIT_KERNEL_AXB_SAXPY5_PROTO (GB_jit_kernel) ;

#if !GB_A_IS_PATTERN && !GB_A_ISO && !GB_A_IS_BITMAP

    #if GB_SEMIRING_HAS_AVX_IMPLEMENTATION

        //----------------------------------------------------------------------
        // saxpy5 method with vectors of length 8 for double, 16 for single
        //----------------------------------------------------------------------

        // AVX512F: vector registers are 512 bits, or 64 bytes, which can hold
        // 16 floats or 8 doubles.

        #define GB_V16_512 (16 * GB_Z_NBITS <= 512)
        #define GB_V8_512  ( 8 * GB_Z_NBITS <= 512)
        #define GB_V4_512  ( 4 * GB_Z_NBITS <= 512)

        #define GB_V16 GB_V16_512
        #define GB_V8  GB_V8_512
        #define GB_V4  GB_V4_512

        #if GB_COMPILER_SUPPORTS_AVX512F && GB_V4_512

            GB_TARGET_AVX512F static inline void GB_AxB_saxpy5_unrolled_avx512f
            (
                GrB_Matrix C,
                const GrB_Matrix A,
                const GrB_Matrix B,
                const int ntasks,
                const int nthreads,
                const int64_t *B_slice
            )
            {
                #include "template/GB_AxB_saxpy5_unrolled.c"
            }

        #endif

        //----------------------------------------------------------------------
        // saxpy5 method with vectors of length 4 for double, 8 for single
        //----------------------------------------------------------------------

        // AVX2: vector registers are 256 bits, or 32 bytes, which can hold
        // 8 floats or 4 doubles.

        #define GB_V16_256 (16 * GB_Z_NBITS <= 256)
        #define GB_V8_256  ( 8 * GB_Z_NBITS <= 256)
        #define GB_V4_256  ( 4 * GB_Z_NBITS <= 256)

        #undef  GB_V16
        #undef  GB_V8
        #undef  GB_V4

        #define GB_V16 GB_V16_256
        #define GB_V8  GB_V8_256
        #define GB_V4  GB_V4_256

        #if GB_COMPILER_SUPPORTS_AVX2 && GB_V4_256

            GB_TARGET_AVX2 static inline void GB_AxB_saxpy5_unrolled_avx2
            (
                GrB_Matrix C,
                const GrB_Matrix A,
                const GrB_Matrix B,
                const int ntasks,
                const int nthreads,
                const int64_t *B_slice
            )
            {
                #include "template/GB_AxB_saxpy5_unrolled.c"
            }

        #endif

        //----------------------------------------------------------------------
        // saxpy5 method with RISC-V vectors
        //----------------------------------------------------------------------

        #if GB_COMPILER_SUPPORTS_RVV1

            #include <riscv_vector.h>

            GB_TARGET_RVV1 static inline void GB_AxB_saxpy5_unrolled_rvv
            (
                GrB_Matrix C,
                const GrB_Matrix A,
                const GrB_Matrix B,
                const int ntasks,
                const int nthreads,
                const int64_t *B_slice
            )
            {
                #include "template/GB_AxB_saxpy5_lv.c"
            }

        #endif
    
    #endif

    //--------------------------------------------------------------------------
    // saxpy5 method unrolled, with no vectors
    //--------------------------------------------------------------------------

    #undef  GB_V16
    #undef  GB_V8
    #undef  GB_V4

    #define GB_V16 0
    #define GB_V8  0
    #define GB_V4  0

    static inline void GB_AxB_saxpy5_unrolled_vanilla
    (
        GrB_Matrix C,
        const GrB_Matrix A,
        const GrB_Matrix B,
        const int ntasks,
        const int nthreads,
        const int64_t *B_slice
    )
    {
        #include "template/GB_AxB_saxpy5_unrolled.c"
    }

#endif

//------------------------------------------------------------------------------
// GB_jit_kernel: for saxpy5 method
//------------------------------------------------------------------------------

GB_JIT_GLOBAL GB_JIT_KERNEL_AXB_SAXPY5_PROTO (GB_jit_kernel)
{
    GB_GET_CALLBACKS ;

    #if GB_A_IS_PATTERN || GB_A_ISO
    {

        //----------------------------------------------------------------------
        // saxpy5: C+=A*B where A is bitmap/full and iso or pattern
        //----------------------------------------------------------------------

        #include "template/GB_AxB_saxpy5_A_iso_or_pattern.c"

    }
    #elif GB_A_IS_BITMAP
    {

        //----------------------------------------------------------------------
        // saxpy5: C+=A*B where A is bitmap (but not iso or pattern)
        //----------------------------------------------------------------------

        #include "template/GB_AxB_saxpy5_A_bitmap.c"

    }
    #else
    {

        //----------------------------------------------------------------------
        // saxpy5: C+=A*B where A is sparse/hypersparse
        //----------------------------------------------------------------------

        #if GB_SEMIRING_HAS_AVX_IMPLEMENTATION
        {

            #if GB_COMPILER_SUPPORTS_AVX512F && GB_V4_512
            if (cpu_has_avx512f)
            {
                // x86_64 with AVX512f
                GB_AxB_saxpy5_unrolled_avx512f (C, A, B, ntasks, nthreads,
                    B_slice) ;
                return (GrB_SUCCESS) ;
            }
            #endif

            #if GB_COMPILER_SUPPORTS_AVX2 && GB_V4_256
            if (cpu_has_avx2)
            {
                // x86_64 with AVX2
                GB_AxB_saxpy5_unrolled_avx2 (C, A, B, ntasks, nthreads,
                    B_slice) ;
                return (GrB_SUCCESS) ;
            }
            #endif

            #if GB_COMPILER_SUPPORTS_RVV1
            if (cpu_has_rvv1)
            {
                // RISC-V64 with RVV1.0
                GB_AxB_saxpy5_unrolled_rvv (C, A, B, ntasks, nthreads,
                    B_slice) ;
                return (GrB_SUCCESS) ;
            }
            #endif
        }
        #endif

        // any architecture and any semiring
        GB_AxB_saxpy5_unrolled_vanilla (C, A, B, ntasks, nthreads, B_slice) ;

    }
    #endif
    return (GrB_SUCCESS) ;
}

