//------------------------------------------------------------------------------
// GB_AxB_saxpy5_meta.c: C+=A*B when C is full
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// C is full.
// A is bitmap or full.
// B is sparse or hypersparse.

// The monoid is identical to the accum op, and is not the ANY operator.
// The type of A must match the multiply operator input.
// The type of C must match the monoid/accum op.  B can be typecasted in
// general (in the JIT version), but not here for the FactoryKernel.

// This method is only used for built-in semirings with no typecasting, in
// the FactoryKernels.  It is not used for JIT kernels, but the JIT kernel
// (Source/JitKernels/GB_jit_kernel_AxB_saxpy5.c) has nearly identical logic.

#ifdef GB_GENERIC
#error "saxpy5 generic kernel undefined"
#endif

#if GB_IS_ANY_MONOID
#error "saxpy5 not defined for the ANY monoid"
#endif

#ifdef GB_JIT_KERNEL
#error "saxpy5 JIT kernel uses the lower-level saxpy5 templates, not this one"
#endif

{

    //--------------------------------------------------------------------------
    // get C, A, and B
    //--------------------------------------------------------------------------

    ASSERT (GB_IS_FULL (C)) ;
    ASSERT (C->vlen == A->vlen) ;
    ASSERT (C->vdim == B->vdim) ;
    ASSERT (A->vdim == B->vlen) ;
    ASSERT (GB_IS_BITMAP (A) || GB_IS_FULL (A)) ;
    ASSERT (GB_IS_SPARSE (B) || GB_IS_HYPERSPARSE (B)) ;

    const bool A_is_bitmap = GB_IS_BITMAP (A) ;
    const bool A_iso = A->iso ;

    //--------------------------------------------------------------------------
    // C += A*B, no mask, A bitmap/full, B sparse/hyper
    //--------------------------------------------------------------------------

    #if GB_A_IS_PATTERN
    {

        //----------------------------------------------------------------------
        // A is pattern-only
        //----------------------------------------------------------------------

        if (A_is_bitmap)
        { 
            // A is bitmap and pattern-only
            #undef  GB_A_IS_BITMAP
            #define GB_A_IS_BITMAP 1
            #include "GB_AxB_saxpy5_A_iso_or_pattern.c"
        }
        else
        { 
            // A is full and pattern-only
            #undef  GB_A_IS_BITMAP
            #define GB_A_IS_BITMAP 0
            #include "GB_AxB_saxpy5_A_iso_or_pattern.c"
        }

    }
    #else
    {

        //----------------------------------------------------------------------
        // A is valued
        //----------------------------------------------------------------------

        if (A_iso)
        {

            //------------------------------------------------------------------
            // A is iso-valued
            //------------------------------------------------------------------

            if (A_is_bitmap)
            { 
                // A is bitmap, iso-valued, B is sparse/hyper
                #undef  GB_A_IS_BITMAP
                #define GB_A_IS_BITMAP 1
                #include "GB_AxB_saxpy5_A_iso_or_pattern.c"
            }
            else
            { 
                // A is full, iso-valued, B is sparse/hyper
                #undef  GB_A_IS_BITMAP
                #define GB_A_IS_BITMAP 0
                #include "GB_AxB_saxpy5_A_iso_or_pattern.c"
            }

        }
        else
        {

            //------------------------------------------------------------------
            // general case: A is non-iso and valued
            //------------------------------------------------------------------

            if (A_is_bitmap)
            { 
                // A is bitmap, non-iso-valued, B is sparse/hyper
                #undef  GB_A_IS_BITMAP
                #define GB_A_IS_BITMAP 1
                #include "GB_AxB_saxpy5_A_bitmap.c"
                #undef  GB_A_IS_BITMAP
            }
            else
            { 
                // A is full, non-iso-valued, B is sparse/hyper
                #if GB_SEMIRING_HAS_AVX_IMPLEMENTATION
                    #if GB_COMPILER_SUPPORTS_AVX512F && GB_V4_512
                    if (GB_Global_cpu_features_avx512f ( ))
                    {
                        // x86_64 with AVX512f
                        GB_AxB_saxpy5_unrolled_avx512f (C, A, B,
                            ntasks, nthreads, B_slice) ;
                        return (GrB_SUCCESS) ;
                    }
                    #endif
                    #if GB_COMPILER_SUPPORTS_AVX2 && GB_V4_256
                    if (GB_Global_cpu_features_avx2 ( ))
                    {
                        // x86_64 with AVX2
                        GB_AxB_saxpy5_unrolled_avx2 (C, A, B,
                            ntasks, nthreads, B_slice) ;
                        return (GrB_SUCCESS) ;
                    }
                    #endif
                #endif
                // any architecture and any built-in semiring
                GB_AxB_saxpy5_unrolled_vanilla (C, A, B,
                    ntasks, nthreads, B_slice) ;
            }
        }
    }
    #endif
}

#undef GB_A_IS_BITMAP
#undef GB_B_IS_HYPER

