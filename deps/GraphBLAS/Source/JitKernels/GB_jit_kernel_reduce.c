//------------------------------------------------------------------------------
// GB_jit_kernel_reduce.c: JIT kernel for reduction to scalar
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// The GB_jitifyer constructs a *.c file with macro definitions specific to the
// problem instance, such as the excerpts for the GB_jit_reduce_2c1fbb2 kernel,
// below, which a kernel that computes the scalar reduce of a double matrix in
// bitmap form, using the GrB_PLUS_FP64_MONOID.  The code 2c1fbb2 is computed
// by GB_enumify_reduce.  The macros are followed by an #include with this
// file, to define the kernel routine itself.  The kernel is always called
// GB_jit_kernel, regardless of what it computes.

#ifdef for_comments_only    // only so vim will add color to the code below:

    // example file: GB_jit_reduce_2c1fbb2.c

    #include "GB_jit_kernel_reduce.h"

    // reduce: (plus, double)

    // monoid:
    #define GB_Z_TYPE double
    #define GB_ADD(z,x,y) z = (x) + (y)
    #define GB_UPDATE(z,y) z += y
    #define GB_DECLARE_IDENTITY(z) double z = 0
    #define GB_DECLARE_IDENTITY_CONST(z) const double z = 0
    #define GB_HAS_IDENTITY_BYTE 1
    #define GB_IDENTITY_BYTE 0x00
    #define GB_PRAGMA_SIMD_REDUCTION_MONOID(z) GB_PRAGMA_SIMD_REDUCTION (+,z)
    #define GB_Z_IGNORE_OVERFLOW 1
    #define GB_Z_NBITS 64
    #define GB_Z_ATOMIC_BITS 64
    #define GB_Z_HAS_ATOMIC_UPDATE 1
    #define GB_Z_HAS_OMP_ATOMIC_UPDATE 1
    #define GB_Z_HAS_CUDA_ATOMIC_BUILTIN 1
    #define GB_Z_CUDA_ATOMIC GB_cuda_atomic_add
    #define GB_Z_CUDA_ATOMIC_TYPE double
    #define GB_GETA_AND_UPDATE(z,Ax,p) GB_UPDATE (z, Ax [p])

    // A matrix: bitmap
    #define GB_A_IS_HYPER  0
    #define GB_A_IS_SPARSE 0
    #define GB_A_IS_BITMAP 1
    #define GB_A_IS_FULL   0
    #define GBP_A(Ap,k,vlen) ((k) * (vlen))
    #define GBH_A(Ah,k)      (k)
    #define GBI_A(Ai,p,vlen) ((p) % (vlen))
    #define GBB_A(Ab,p)      Ab [p]
    #define GB_A_ISO 0
    #define GB_A_HAS_ZOMBIES 0
    #define GB_A_TYPE double
    #define GB_A2TYPE double
    #define GB_DECLAREA(a) double a
    #define GB_GETA(a,Ax,p,iso) a = Ax [p]

    // panel size for reduction:
    #define GB_PANEL 32

    #include "GB_monoid_shared_definitions.h"

    #include "GB_jit_kernel_reduce.c"

#endif

//------------------------------------------------------------------------------
// reduce to a non-iso matrix to scalar, for monoids only
//------------------------------------------------------------------------------

// The two template files GB_reduce_to_scalar_template.c and GB_reduce_panel.c
// appear in GraphBLAS/Source/Template.  They are used by both the pre-compiled
// kernels in GraphBLAS/Source/FactoryKernels, and by the JIT kernel here.

GB_JIT_GLOBAL GB_JIT_KERNEL_REDUCE_PROTO (GB_jit_kernel) ;
GB_JIT_GLOBAL GB_JIT_KERNEL_REDUCE_PROTO (GB_jit_kernel)
{
    GB_Z_TYPE z = (* ((GB_Z_TYPE *) result)) ;
    GB_Z_TYPE *W = (GB_Z_TYPE *) Workspace ;
    #if GB_A_HAS_ZOMBIES || GB_A_IS_BITMAP || (GB_PANEL == 1)
    {
        #include "GB_reduce_to_scalar_template.c"
    }
    #else
    {
        #include "GB_reduce_panel.c"
    }
    #endif
    // (*result) = z ;
    memcpy (result, &z, sizeof (GB_Z_TYPE)) ;
    return (GrB_SUCCESS) ;
}

