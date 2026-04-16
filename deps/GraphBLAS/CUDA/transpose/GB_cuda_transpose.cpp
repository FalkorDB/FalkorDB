//------------------------------------------------------------------------------
// GB_cuda_transpose: C=A' or C=op(A'), with typecasting, via the CUDA builder
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// CALLS:     GB_cuda_builder

// Transpose a matrix, C=A', using the GB_cuda_builder method, and optionally
// apply a unary operator and/or typecast the values.  The result is returned
// in the T = (*Thandle) output, which later transplanted into the C matrix by
// the caller, GB_transpose.

#define GB_FREE_WORKSPACE                           \
{                                                   \
    GB_FREE_MEMORY (&Key_input, Key_input_mem) ;    \
    GB_FREE_MEMORY (&Swork, Swork_mem) ;            \
    GB_cuda_stream_pool_release (&stream) ;         \
}

#define GB_FREE_ALL                             \
{                                               \
    GB_FREE_WORKSPACE ;                         \
    GB_Matrix_free (Thandle) ;                  \
}

#include "transpose/GB_cuda_transpose.hpp"
#include "builder/GB_cuda_builder.hpp"
extern "C"
{
    #include "apply/GB_apply.h"
}

//------------------------------------------------------------------------------
// GB_cuda_transpose
//------------------------------------------------------------------------------

GrB_Info GB_cuda_transpose      // T=A', T=(ctype)A' or T=op(A')
(
    GrB_Matrix *Thandle,        // output matrix T, header allocated on input
    GrB_Type ctype,             // desired type of T
    const bool C_is_csc,        // desired CSR/CSC format of C and T
    const bool C_iso,           // true if C (and T) is iso
    const GB_iso_code C_code_iso,   // iso code for C and T
    const GrB_Matrix A,         // input matrix; C == A if done in place
    const bool in_place,        // true if C and A are the same matrix
        // no operator is applied if op is NULL
        const GB_Operator op,       // unary/idxunop/binop to apply
        const GrB_Scalar scalar,    // scalar to bind to binary operator
        bool binop_bind1st,         // if true, binop(x,A) else binop(A,y)
        bool flipij,                // if true, flip i,j for user idxunop
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;
    ASSERT (Thandle != NULL) ;
    GrB_Matrix T = (*Thandle) ;     // just the header of T is given on input
    ASSERT (T != NULL) ;
    int memlane = GB_memlane (T->header_mem) ;
    uint64_t mem = GB_mem (memlane, 0) ;

    cudaStream_t stream = nullptr ;
    GB_void *Key_input = NULL ;
    uint64_t Key_input_mem = mem ;
    GB_void *Swork = NULL  ;
    uint64_t Swork_mem = mem ;

    GrB_Type atype = A->type ;
    int64_t anz = GB_nnz (A) ;
    bool Ap_is_32 = A->p_is_32 ;
    bool Aj_is_32 = A->j_is_32 ;
    bool Ai_is_32 = A->i_is_32 ;
    int64_t avlen = A->vlen ;
    int64_t avdim = A->vdim ;
    float A_hyper_switch = A->hyper_switch ;

    bool Cp_is_32, Cj_is_32, Ci_is_32 ;

//  size_t apsize = (Ap_is_32) ? sizeof (uint32_t) : sizeof (uint64_t) ;
    size_t ajsize = (Aj_is_32) ? sizeof (uint32_t) : sizeof (uint64_t) ;
    size_t aisize = (Ai_is_32) ? sizeof (uint32_t) : sizeof (uint64_t) ;

    size_t csize = ctype->size ;

    //--------------------------------------------------------------------------
    // construct Key_input
    //--------------------------------------------------------------------------

    bool Key_is_32 = GB_cuda_builder_key_is_32 (avlen, avdim) ;
    size_t key_size = 2 * ((Key_is_32) ? sizeof (uint32_t) : sizeof (uint64_t));

    // allocate iwork of size anz
    Key_input = (GB_void *) GB_MALLOC_MEMORY (anz+1, key_size,
        &Key_input_mem) ;
    if (Key_input == NULL)
    { 
        // out of memory
        GB_FREE_ALL ;
        return (GrB_OUT_OF_MEMORY) ;
    }

    GB_OK (GB_cuda_stream_pool_acquire (&stream)) ;

    // determine the geometry of the CUDA kernel launches
    int32_t number_of_sms = GB_Global_gpu_sm_get (0) ;
    int64_t raw_gridsz = GB_ICEIL (anz, GB_CUDA_TRANSPOSE_PREP_CHUNKSIZE) ;
    int32_t gridsz = std::min (raw_gridsz, (int64_t) (number_of_sms * 256)) ;
    gridsz = std::max (gridsz, 1) ;

    double t = GB_OPENMP_GET_WTIME ;        // FIXME

    GB_OK (GB_cuda_transpose_prep_jit (
        /* output: */ Key_input,
        /* input: */ Key_is_32, A, stream, gridsz)) ;

    GB_OK (GB_cuda_stream_pool_release (&stream)) ;

    t = GB_OPENMP_GET_WTIME - t ;
    printf ("CUDA transpose prep time: %g\n", t) ;  // FIXME

    //--------------------------------------------------------------------------
    // allocate the output matrix
    //--------------------------------------------------------------------------

    // T is created using the requested integers of C.
    GB_determine_pji_is_32 (&Cp_is_32, &Cj_is_32, &Ci_is_32,
        GxB_HYPERSPARSE, anz, avdim, avlen, Werk) ;

    // initialize the header of T, with no content,
    // and initialize the type and dimension of T.

    info = GB_new (Thandle, // hyper, existing header
        ctype, avdim, avlen, GB_ph_null, C_is_csc,
        GxB_HYPERSPARSE, A_hyper_switch, 0,
        Cp_is_32, Cj_is_32, Ci_is_32, memlane) ;
    ASSERT (info == GrB_SUCCESS) ;

    GB_void *X = NULL ;

    //------------------------------------------------------------------
    // construct Swork if necessary
    //------------------------------------------------------------------

    if (op != NULL && !C_iso)
    { 
        Swork = (GB_void *) GB_XALLOC_MEMORY (false, C_iso, anz, csize,
            &Swork_mem) ;
        if (Swork == NULL)
        {
            // out of memory
            GB_FREE_ALL ;
            return (GrB_OUT_OF_MEMORY) ;
        }
    }

    // numerical values: apply the op, typecast, or make shallow copy
    GrB_Type stype ;
    GB_void sscalar [GB_VLA(csize)] ;
    if (C_iso)
    { 
        // apply the op to the iso scalar
        GB_unop_iso (sscalar, ctype, C_code_iso, op, A, scalar) ;
        X = sscalar ;
        stype = ctype ;
    }
    else if (op != NULL)
    { 
        // Swork = op (A)
        // FIXME: tell GB_apply_op it "must" use the GPU
        info = GB_apply_op (Swork, ctype, C_code_iso, op, scalar,
            binop_bind1st, flipij, A, Werk) ;
        ASSERT (info == GrB_SUCCESS) ;
        // GB_cuda_builder will not need to typecast Swork to T->x, and it may
        // choose to transplant it into T->x
        X = Swork ;
        stype = ctype ;
    }
    else
    { 
        // GB_cuda_builder will typecast S_input from atype to ctype if needed.
        // S_input is a shallow copy of Ax, and must not be modified.
        ASSERT (!C_iso) ;
        ASSERT (!A->iso) ;
        X = (GB_void *) A->x ;
        stype = atype ;
    }

    //------------------------------------------------------------------
    // build the matrix: T = (ctype) A' or op ((xtype) A')
    //------------------------------------------------------------------

    GB_OK (GB_cuda_builder (
        Thandle,    // create T using an existing header
        ctype,      // T is of type ctype
        avdim,      // T->vlen = A->vdim, always > 1
        avlen,      // T->vdim = A->vlen, always > 1
        C_is_csc,   // T has the same CSR/CSC format as C
        true,       // is_matrix, since T->vdim is always > 1
        Key_input,  // (i,j) indices pre-loaded into Key_input workspace
        NULL,       // I indices: not used
        NULL,       // J indices: not used
        X,          // X values
        C_iso,      // iso property of T is the same as C->iso
        anz,        // number of tuples
        NULL,       // no dup operator needed (input has no duplicates)
        stype,      // type of X
        false,      // no burble (already burbled above)
        true,       // I_is_32: not used
        true,       // J_is_32: not used
        Cp_is_32, Cj_is_32, Ci_is_32, // integer sizes for T 
        true,       // tuples known to have no duplicates
        false       // tuples are not sorted on input
    )) ;

    //------------------------------------------------------------------
    // return result
    //------------------------------------------------------------------

    GB_FREE_WORKSPACE ;
    ASSERT (!GB_JUMBLED (*Thandle)) ;
    return (GrB_SUCCESS) ;
}

