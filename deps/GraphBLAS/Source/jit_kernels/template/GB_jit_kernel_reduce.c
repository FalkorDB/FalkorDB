//------------------------------------------------------------------------------
// GB_jit_kernel_reduce.c: JIT kernel for reduction to scalar
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// The GB_jitifyer constructs a *.c file with macro definitions specific to the
// problem instance, such as the GB_jit__reduce__14bb2 kernel, below, which a
// kernel that computes the scalar reduce of a double matrix in bitmap form,
// using the GrB_PLUS_FP64_MONOID.  The hex code 14bb2 is computed by
// GB_enumify_reduce.  The macros are followed by an #include with this file,
// to define the kernel routine itself.  The kernel is always called
// GB_jit_kernel, regardless of what it computes.  However, if this kernel is
// copied into GraphBLAS/PreJit, the name GB_jit_kernel is replaced with its
// full name, GB_jit__reduce__14bb2, which then appears as a compiled function
// in libgraphblas.so when the GraphBLAS library itself is recompiled.

// The GB_jit_query function provides a mechanism for GraphBLAS to query the
// kernels it has already compiled.  When a compiled kernel is loaded, its
// definitions are checked to make sure they haven't changed.  If the user
// application has changed the definition of a user-defined data type, for
// example, the string defn [0] would differ.  See the gauss_demo program for
// an example.  When GraphBLAS detects any such change, the old compiled kernel
// is discarded and a new one is compiled to match the expected definition.

#ifdef for_comments_only    // only so vim will add color to the code below:

    //--------------------------------------------------------------------------
    // GB_jit__reduce__14bb2.c
    //--------------------------------------------------------------------------
    // SuiteSparse:GraphBLAS v10.0.0, Timothy A. Davis, (c) 2017-2025,
    // All Rights Reserved.
    // SPDX-License-Identifier: Apache-2.0
    // The above copyright and license do not apply to any
    // user-defined types and operators defined below.
    //--------------------------------------------------------------------------

    #include "include/GB_jit_kernel.h"

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
    #define GB_Z_SIZE  8
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
    #define GBp_A(Ap,k,vlen) ((k) * (vlen))
    #define GBh_A(Ah,k)      (k)
    #define GBi_A(Ai,p,vlen) ((p) % (vlen))
    #define GBb_A(Ab,p)      Ab [p]
    #define GB_A_NVALS(e) int64_t e = A->nvals
    #define GB_A_NHELD(e) int64_t e = (A->vlen * A->vdim)
    #define GB_A_HAS_ZOMBIES 0
    #define GB_A_ISO 0
    #define GB_A_TYPE double
    #define GB_A2TYPE double
    #define GB_DECLAREA(a) double a
    #define GB_GETA(a,Ax,p,iso) a = Ax [p]
    #define GB_Ap_BITS 64
    #define GB_Ai_BITS 64

    // panel size for reduction:
    #define GB_PANEL 32

    #include "include/GB_monoid_shared_definitions.h"
    #ifndef GB_JIT_RUNTIME
    #define GB_jit_kernel GB_jit__reduce__14bb2
    #define GB_jit_query  GB_jit__reduce__14bb2_query
    #endif
    #include "template/GB_jit_kernel_reduce.c"
    GB_JIT_GLOBAL GB_JIT_QUERY_PROTO (GB_jit_query) ;
    GB_JIT_GLOBAL GB_JIT_QUERY_PROTO (GB_jit_query)
    {
        (*hash) = 0x753d2d93e48ef09e ;
        v [0] = 10 ; v [1] = 0 ; v [2] = 0 ;
        defn [0] = NULL ;
        defn [1] = NULL ;
        defn [2] = NULL ;
        defn [3] = NULL ;
        defn [4] = NULL ;
        return (true) ;
    }

#endif

//------------------------------------------------------------------------------
// reduce to a non-iso matrix to scalar, for monoids only
//------------------------------------------------------------------------------

// The two template files GB_reduce_to_scalar_template.c and GB_reduce_panel.c
// appear in GraphBLAS/Source/reduce/template.  They are used by both the
// pre-compiled kernels in GraphBLAS/FactoryKernels, and by the JIT kernel
// here.

// The prototype of this GB_jit_kernel is defined by a macro,
// GB_JIT_KERNEL_REDUCE_PROTO, defined in
// Source/jit_kernels/include/GB_jit_kernel_proto.h:

#if 0
#define GB_JIT_KERNEL_REDUCE_PROTO(GB_jit_kernel_reduce)                \
GrB_Info GB_jit_kernel_reduce                                           \
(                                                                       \
    GB_void *result,                                                    \
    const GrB_Matrix A,                                                 \
    GB_void *restrict Workspace,                                        \
    bool *restrict F,                                                   \
    const int ntasks,                                                   \
    const int nthreads,                                                 \
    const GB_callback_struct *restrict my_callback                      \
)
#endif

// This macro is used because the identical prototype must appear in many
// places, but with different function names.  For example, if this kernel is
// copied into GraphBLAS/PreJIT, then this macro is used to define the
// GB_jit__reduce__14bb2 function, with the same set of parameters as given by
// the GB_JIT_KERNEL_REDUCE_PROTO macro above.

GB_JIT_GLOBAL GB_JIT_KERNEL_REDUCE_PROTO (GB_jit_kernel) ;
GB_JIT_GLOBAL GB_JIT_KERNEL_REDUCE_PROTO (GB_jit_kernel)
{
    GB_GET_CALLBACKS ;

    GB_Z_TYPE z = (* ((GB_Z_TYPE *) result)) ;
    GB_Z_TYPE *W = (GB_Z_TYPE *) Workspace ;
    // The two templates below use the F and Workspace arrays to reduce A to
    // the scalar z.  For OpenMP parallelism, ntasks tasks are created, and
    // executed with nthreads OpenMP threads.
    #if GB_A_HAS_ZOMBIES || GB_A_IS_BITMAP || (GB_PANEL == 1)
    {
        // The matrix A either has zombies, is in bitmap format, or the
        // panel size is one.  In this case, use a simpler method
        // that does not use a panel workspace array.
        #include "template/GB_reduce_to_scalar_template.c"
    }
    #else
    {
        // This algorithm relies on a panel array for each thread, of size
        // GB_PANEL (defined above).  Each task grabs a set of entries from A,
        // of size GB_PANEL, and accumulates each of them in the panel.  The
        // task iterates over its part of the A matrix.  When the task is done,
        // it 'sums' its panel into a single scalar.  This method is faster for
        // some monoids such as (plus,double).  Some monoids or data types do
        // not benefit from a panel-based reduction.  In this case the panel
        // size is set (via GB_PANEL) to one.
        #include "template/GB_reduce_panel.c"
    }
    #endif
    // (*result) = z ;
    memcpy (result, &z, sizeof (GB_Z_TYPE)) ;
    return (GrB_SUCCESS) ;
}

