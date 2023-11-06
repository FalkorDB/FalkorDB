//------------------------------------------------------------------------------
// GB_AxB_dot2: compute C<#M>=A'*B or C<#M>=A*B, where C is bitmap/full
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// This method always constructs C as bitmap or full; it then converts C to
// sparse or hyper if A or B are hypersparse.  The C<M>=A'*B dot product when C
// is sparse is computed by GB_AxB_dot3.  This method handles the case when C
// is bitmap or full.

// If A_not_transposed is true, then C<#>=A*B is computed for GB_AxB_saxpy.  A
// is bitmap or full, and the dot product method accesses A with a different
// stride than when computing C<#M>=A'*B.

// TODO:  this is slower than it could be if A and B are both bitmap/full, when
// A->vlen is large.  This is because the inner loop is a simple full/bitmap
// dot product, across the entire input vectors.  No tiling is used, so cache
// performance is not as good as it could be.  For large problems, C=(A')*B is
// faster with the saxpy3 method, as compared to this method with C=A'*B.

// JIT: done.

#define GB_FREE_WORKSPACE                   \
{                                           \
    GB_Matrix_free (&Mwork) ;               \
    GB_Matrix_free (&Awork) ;               \
    GB_Matrix_free (&Bwork) ;               \
    GB_WERK_POP (M_ek_slicing, int64_t) ;   \
    GB_WERK_POP (B_slice, int64_t) ;        \
    GB_WERK_POP (A_slice, int64_t) ;        \
}

#define GB_FREE_ALL                         \
{                                           \
    GB_FREE_WORKSPACE ;                     \
    GB_phybix_free (C) ;                    \
}

#include "GB_mxm.h"
#include "GB_subref.h"
#include "GB_ek_slice.h"
#include "GB_bitmap_assign_methods.h"
#include "GB_stringify.h"
#include "GB_AxB__include1.h"
#ifndef GBCOMPACT
#include "GB_AxB__include2.h"
#endif

GrB_Info GB_AxB_dot2                // C=A'*B or C<#M>=A'*B, dot product method
(
    GrB_Matrix C,                   // output matrix, static header
    const bool C_iso,               // true if C is iso
    const GB_void *cscalar,         // iso value of C
    const GrB_Matrix M_in,          // mask matrix for C<#M>=A'*B, may be NULL
    const bool Mask_comp,           // if true, use !M
    const bool Mask_struct,         // if true, use the only structure of M
    const bool A_not_transposed,    // if true, C=A*B, else C=A'*B
    const GrB_Matrix A_in,          // input matrix
    const GrB_Matrix B_in,          // input matrix
    const GrB_Semiring semiring,    // semiring that defines C=A*B
    const bool flipxy,              // if true, do z=fmult(b,a) vs fmult(a,b)
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;

    ASSERT (C != NULL && (C->static_header || GBNSTATIC)) ;
    ASSERT_MATRIX_OK_OR_NULL (M_in, "M for dot A'*B", GB0) ;
    ASSERT_MATRIX_OK (A_in, "A for dot A'*B", GB0) ;
    ASSERT_MATRIX_OK (B_in, "B for dot A'*B", GB0) ;

    ASSERT (!GB_ZOMBIES (M_in)) ;
    ASSERT (GB_JUMBLED_OK (M_in)) ;
    ASSERT (!GB_PENDING (M_in)) ;
    ASSERT (!GB_ZOMBIES (A_in)) ;
    ASSERT (!GB_JUMBLED (A_in)) ;
    ASSERT (!GB_PENDING (A_in)) ;
    ASSERT (!GB_ZOMBIES (B_in)) ;
    ASSERT (!GB_JUMBLED (B_in)) ;
    ASSERT (!GB_PENDING (B_in)) ;

    ASSERT_SEMIRING_OK (semiring, "semiring for numeric A'*B", GB0) ;

    struct GB_Matrix_opaque Awork_header, Bwork_header, Mwork_header ;
    GrB_Matrix M = NULL, Mwork = NULL ;
    GrB_Matrix A = NULL, Awork = NULL ;
    GrB_Matrix B = NULL, Bwork = NULL ;
    GB_WERK_DECLARE (A_slice, int64_t) ;
    GB_WERK_DECLARE (B_slice, int64_t) ;
    GB_WERK_DECLARE (M_ek_slicing, int64_t) ;

    // GB_AxB_saxpy punts to this dot2 method for for C=A*B, and in this case,
    // A is bitmap or full, and B is hypersparse or sparse
    bool A_is_full = GB_IS_FULL (A_in) ;
    bool B_is_full = GB_IS_FULL (B_in) ;
    bool A_bitmap_or_full = (GB_IS_BITMAP (A_in) || A_is_full) ;
    bool B_bitmap_or_full = (GB_IS_BITMAP (B_in) || B_is_full) ;
    ASSERT (GB_IMPLIES (A_not_transposed,
        (GB_IS_BITMAP (A_in) || GB_IS_FULL (A_in)) &&
        (GB_IS_SPARSE (B_in) || GB_IS_HYPERSPARSE (B_in)))) ;

    //--------------------------------------------------------------------------
    // construct hyper_shallow versions of A and B, if hypersparse
    //--------------------------------------------------------------------------

    // If A_in is hypersparse, a new sparse matrix A is constructed with
    // A->vdim = A_in->nvec and the same vlen as A_in, and then the
    // hyper_shallow C->vlen will equal A->vdim < cvlen_final.

    // If B_in is hypersparse, a new sparse matrix B is constructed with
    // B->vdim = B_in->nvec and the same vlen as B_in, and then the
    // hyper_shallow C->vdim will equal B->vdim < cvdim_final.

    int64_t cvlen_final = (A_not_transposed) ? A_in->vlen : A_in->vdim ;
    int64_t cvdim_final = B_in->vdim ;
    ASSERT (A_in->vlen > 0) ;
    bool A_is_hyper = GB_IS_HYPERSPARSE (A_in) ;
    bool B_is_hyper = GB_IS_HYPERSPARSE (B_in) ;
    bool A_or_B_hyper = A_is_hyper || B_is_hyper ;
    GrB_Index *restrict Ah = (GrB_Index *) A_in->h ;
    GrB_Index *restrict Bh = (GrB_Index *) B_in->h ;

    if (A_is_hyper)
    { 
        // A = hypershallow version of A_in
        GB_CLEAR_STATIC_HEADER (Awork, &Awork_header) ;
        A = GB_hyper_shallow (Awork, A_in) ;
    }
    else
    { 
        // use A_in as-is
        A = A_in ;
    }

    if (B_is_hyper)
    { 
        // B = hypershallow version of B_in
        GB_CLEAR_STATIC_HEADER (Bwork, &Bwork_header) ;
        B = GB_hyper_shallow (Bwork, B_in) ;
    }
    else
    { 
        // use B_in as-is
        B = B_in ;
    }

    ASSERT (!GB_IS_HYPERSPARSE (A)) ;
    ASSERT (!GB_IS_HYPERSPARSE (B)) ;
    ASSERT (GB_IMPLIES (A_not_transposed, !A_is_hyper && (A == A_in))) ;
    bool A_is_sparse = GB_IS_SPARSE (A) ;
    bool B_is_sparse = GB_IS_SPARSE (B) ;

    //--------------------------------------------------------------------------
    // determine the size of C
    //--------------------------------------------------------------------------

    int64_t cnvec = B->nvec ;
    int64_t cvlen = (A_not_transposed) ? A->vlen : A->vdim ;
    int64_t cvdim = B->vdim ;

    int64_t cnz ;
    bool ok = GB_int64_multiply ((GrB_Index *) (&cnz), cvlen, cvdim) ;

    //--------------------------------------------------------------------------
    // extract the submask if A or B are hypersparse 
    //--------------------------------------------------------------------------

    if (A_or_B_hyper && M_in != NULL)
    { 
        // Mwork = M_in (Ah, Bh), where Mwork has a static header
        // if Mask_struct then Mwork is extracted as iso
        GB_CLEAR_STATIC_HEADER (Mwork, &Mwork_header) ;
        GB_OK (GB_subref (Mwork, Mask_struct, M_in->is_csc, M_in,
            (A_is_hyper) ? Ah : GrB_ALL, cvlen,
            (B_is_hyper) ? Bh : GrB_ALL, cvdim, false, Werk)) ;
        M = Mwork ;
        ASSERT_MATRIX_OK_OR_NULL (M, "M submask dot A'*B", GB0) ;
    }
    else
    { 
        // use the mask as-is
        M = M_in ;
    }

    //--------------------------------------------------------------------------
    // determine the number of threads to use
    //--------------------------------------------------------------------------

    int64_t naslice = 0 ;
    int64_t nbslice = 0 ;

    int64_t anvec = (A_not_transposed) ? A->vlen : A->nvec ;
    double anz = (double) GB_nnz_held (A) ;

    int64_t bnvec = B->nvec ;
    double bnz = (double) GB_nnz_held (B) ;

    double work ;
    if (A_bitmap_or_full && !B_bitmap_or_full)
    { 
        // A is bitmap/full, B is sparse/hyper; only B is scanned
        work = bnz ;
    }
    else if (!A_bitmap_or_full && B_bitmap_or_full)
    {
        // A is sparse/hyper, B is bitmap/full; only A is scanned
        work = anz ;
    }
    else if (A_bitmap_or_full && B_bitmap_or_full)
    { 
        // All of A and B are scanned (unless the mask is present)
        work = ((double) cnz) * ((double) B->vlen) ;
    }
    else
    { 
        // In this case, most of both A and B are scanned.  This is a very
        // rough estimate of the work required.
        work = 10 * (anz + bnz) ;
    }

    int nthreads_max = GB_Context_nthreads_max ( ) ;
    double chunk = GB_Context_chunk ( ) ;
    int nthreads = GB_nthreads (work, chunk, nthreads_max) ;

    #define GB_NTASKS_PER_THREAD 32

    if (nthreads == 1)
    { 
        // do the entire computation with a single thread
        naslice = 1 ;
        nbslice = 1 ;
    }
    else
    {
        // determine number of slices for A' and B
        if (bnvec == 1)
        { 
            // C and B are single vectors
            naslice = GB_NTASKS_PER_THREAD * nthreads ;
            nbslice = 1 ;
        }
        else if (anvec == 1 || bnvec == 0
            || bnvec > GB_NTASKS_PER_THREAD * nthreads)
        { 
            // A is a single vector, or B is empty, or B is large: just slice B
            naslice = 1 ;
            nbslice = GB_NTASKS_PER_THREAD * nthreads ;
        }
        else
        { 
            // slice B into individual vectors
            nbslice = bnvec ;

            // slice A' to get a total of about 16*nthreads tasks
            naslice = (GB_NTASKS_PER_THREAD * nthreads) / nbslice ;

            // but do not slice A too finely
            naslice = GB_IMIN (naslice, anvec/4) ;
            naslice = GB_IMAX (naslice, nthreads) ;
        }
    }

    GBURBLE ("(nthreads: %d naslice %g nbslice %g) ", nthreads,
        (double) naslice, (double) nbslice) ;

    //--------------------------------------------------------------------------
    // get the semiring operators
    //--------------------------------------------------------------------------

    GrB_BinaryOp mult = semiring->multiply ;
    GrB_Monoid add = semiring->add ;
    ASSERT (mult->ztype == add->op->ztype) ;
    bool A_is_pattern, B_is_pattern ;
    GB_binop_pattern (&A_is_pattern, &B_is_pattern, flipxy, mult->opcode) ;

    //--------------------------------------------------------------------------
    // allocate workspace and slice A and B
    //--------------------------------------------------------------------------

    // A and B can have any sparsity: full, bitmap, sparse, or hypersparse.
    // C is always created as bitmap

    GB_WERK_PUSH (A_slice, naslice + 1, int64_t) ;
    GB_WERK_PUSH (B_slice, nbslice + 1, int64_t) ;
    if (A_slice == NULL || B_slice == NULL || !ok)
    { 
        // out of memory
        GB_FREE_ALL ;
        return (GrB_OUT_OF_MEMORY) ;
    }
    GB_pslice (A_slice, A->p, anvec, naslice, false) ;
    GB_pslice (B_slice, B->p, bnvec, nbslice, false) ;

    //--------------------------------------------------------------------------
    // allocate C
    //--------------------------------------------------------------------------

    // if M is sparse/hyper, then calloc C->b; otherwise use malloc
    bool M_is_sparse_or_hyper = (M != NULL) &&
        (GB_IS_SPARSE (M) || GB_IS_HYPERSPARSE (M)) ;
    GrB_Type ctype = add->op->ztype ;

    // determine the sparsity of C.  If M is present, C is always bitmap.
    // otherwise, C can be bitmap or full
    int C_sparsity = GxB_BITMAP ;
    if (M == NULL)
    {
        // no mask is present so C can be bitmap or full
        if (A_is_full && B_is_full)
        { 
            // C = A*B or A'*B, both A and B full: C is full
            C_sparsity = GxB_FULL ;
        }
        else if (A_is_full && B_is_sparse)
        {
            // C = A*B or A'*B, where A is full and B sparse
            if (B->nvec_nonempty < 0)
            { 
                B->nvec_nonempty = GB_nvec_nonempty (B) ;
            }
            // C is full if all vectors of B are present
            C_sparsity = (B->nvec_nonempty == B->vdim) ?
                GxB_FULL : GxB_BITMAP ;
        }
        else if (A_is_sparse && B_is_full)
        {
            // C = A'*B, where A is sparse and B is full
            if (A->nvec_nonempty < 0)
            { 
                // A->nvec_nonempty is used to select the method 
                A->nvec_nonempty = GB_nvec_nonempty (A) ;
            }
            // C is full if all vectors of A are present
            C_sparsity = (A->nvec_nonempty == A->vdim) ?
                GxB_FULL : GxB_BITMAP ;
        }
    }

    if (M_in == NULL)
    { 
        GBURBLE ("(dot %s = %s%s*%s) ",
            GB_sparsity_char (C_sparsity),
            GB_sparsity_char_matrix (A_in),
            A_not_transposed ? "" : "'",
            GB_sparsity_char_matrix (B_in)) ;
    }
    else
    { 
        GBURBLE ("(dot %s%s%s%s%s = %s%s*%s) ",
            GB_sparsity_char (C_sparsity),
            Mask_struct ? "{" : "<",
            Mask_comp ? "!" : "",
            GB_sparsity_char_matrix (M_in),
            Mask_struct ? "}" : ">",
            GB_sparsity_char_matrix (A_in),
            A_not_transposed ? "" : "'",
            GB_sparsity_char_matrix (B_in)) ;
    }

    // set C->iso = C_iso
    GB_OK (GB_new_bix (&C, // bitmap/full, existing header
        ctype, cvlen, cvdim, GB_Ap_malloc, true, C_sparsity,
        M_is_sparse_or_hyper, B->hyper_switch, cnvec, cnz, true, C_iso)) ;

    //--------------------------------------------------------------------------
    // if M is sparse/hyper, scatter it into the C bitmap
    //--------------------------------------------------------------------------

    if (M_is_sparse_or_hyper)
    { 
        // FUTURE:: could just set Cb [pC] = 2 since Cb has just been calloc'd.
        // However, in the future, this method might be able to modify C on
        // input, in which case C->b will not be all zero.

        ASSERT (C_sparsity == GxB_BITMAP) ;
        int M_ntasks, M_nthreads ;
        GB_SLICE_MATRIX (M, 8) ;

        // Cb [pC] += 2 for each entry M(i,j) in the mask
        GB_bitmap_M_scatter (C,
            NULL, 0, GB_ALL, NULL, NULL, 0, GB_ALL, NULL,
            M, Mask_struct, GB_ASSIGN, GB_BITMAP_M_SCATTER_PLUS_2,
            M_ek_slicing, M_ntasks, M_nthreads) ;
        // the bitmap of C now contains:
        //  Cb (i,j) = 0:   cij not present, mij zero
        //  Cb (i,j) = 1:   cij present, mij zero           (not used yet)
        //  Cb (i,j) = 2:   cij not present, mij 1
        //  Cb (i,j) = 3:   cij present, mij 1              (not used yet)
        GB_WERK_POP (M_ek_slicing, int64_t) ;
    }

    //--------------------------------------------------------------------------
    // C<#>=A'*B, computing each entry with a dot product, via builtin semiring
    //--------------------------------------------------------------------------

    if (C_iso)
    { 

        //----------------------------------------------------------------------
        // via the iso kernel
        //----------------------------------------------------------------------

        GBURBLE ("(iso dot2) ") ;
        memcpy (C->x, cscalar, ctype->size) ;
        info = GB (_Adot2B__any_pair_iso) (C, M, Mask_comp, Mask_struct,
            A_not_transposed, A, A_slice, B, B_slice,
            nthreads, naslice, nbslice) ;
        ASSERT (info != GrB_NO_VALUE) ;

    }
    else
    {

        //----------------------------------------------------------------------
        // via the factory kernel
        //----------------------------------------------------------------------

        info = GrB_NO_VALUE ;
        #ifndef GBCOMPACT
        GB_IF_FACTORY_KERNELS_ENABLED
        { 

            //------------------------------------------------------------------
            // define the worker for the switch factory
            //------------------------------------------------------------------

            #define GB_Adot2B(add,mult,xname) \
                GB (_Adot2B_ ## add ## mult ## xname)

            #define GB_AxB_WORKER(add,mult,xname)                           \
            {                                                               \
                info = GB_Adot2B (add,mult,xname) (C, M, Mask_comp,         \
                    Mask_struct, A_not_transposed, A, A_slice,              \
                    B, B_slice, nthreads, naslice, nbslice) ;               \
            }                                                               \
            break ;

            //------------------------------------------------------------------
            // launch the switch factory
            //------------------------------------------------------------------

            GB_Opcode mult_binop_code, add_binop_code ;
            GB_Type_code xcode, ycode, zcode ;

            if (GB_AxB_semiring_builtin (A, A_is_pattern, B, B_is_pattern,
                semiring, flipxy, &mult_binop_code, &add_binop_code, &xcode,
                &ycode, &zcode))
            { 
                #include "GB_AxB_factory.c"
            }
            ASSERT (info == GrB_SUCCESS || info == GrB_NO_VALUE) ;
        }
        #endif

        //----------------------------------------------------------------------
        // via the JIT or PreJIT kernel
        //----------------------------------------------------------------------

        if (info == GrB_NO_VALUE)
        {
            if (A_not_transposed)
            { 
                // C<#M> = A*B, C is bitmap/full
                info = GB_AxB_dot2n_jit (C, M, Mask_comp,
                    Mask_struct, A, A_slice, B, B_slice, semiring, flipxy,
                    nthreads, naslice, nbslice) ;
            }
            else
            { 
                // C<#M> = A'*B, C is bitmap
                info = GB_AxB_dot2_jit (C, M, Mask_comp,
                    Mask_struct, A, A_slice, B, B_slice, semiring, flipxy,
                    nthreads, naslice, nbslice) ;
            }
        }

        //----------------------------------------------------------------------
        // via the generic kernel
        //----------------------------------------------------------------------

        if (info == GrB_NO_VALUE)
        { 
            #define GB_DOT2_GENERIC
            GB_BURBLE_MATRIX (C, "(generic C%s=A%s*B, C %s) ",
                (M == NULL) ? "" : (Mask_comp ? "<!M>" : "<M>"),
                A_not_transposed ? "" : "'",
                (C_sparsity == GxB_BITMAP) ? "bitmap" : "full") ;
            #include "GB_AxB_dot_generic.c"
            info = GrB_SUCCESS ;
        }
    }

    if (info != GrB_SUCCESS)
    { 
        // out of memory, or other error
        GB_FREE_ALL ;
        return (info) ;
    }

    //--------------------------------------------------------------------------
    // free workspace
    //--------------------------------------------------------------------------

    GB_FREE_WORKSPACE ;
    C->magic = GB_MAGIC ;
    ASSERT_MATRIX_OK (C, "dot2: result C, before expand", GB0) ;
    ASSERT (!GB_ZOMBIES (C)) ;

    //--------------------------------------------------------------------------
    // convert C to sparse/hyper if A or B are hypersparse on input
    //--------------------------------------------------------------------------

    if (A_or_B_hyper)
    { 
        GB_OK (GB_bitmap_expand_to_hyper (C, cvlen_final, cvdim_final,
            A_in, B_in, Werk)) ;
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    ASSERT_MATRIX_OK (C, "dot2: result C, after expand", GB0) ;
    ASSERT (GB_ZOMBIES_OK (C)) ;
    ASSERT (!GB_JUMBLED (C)) ;
    ASSERT (!GB_PENDING (C)) ;
    ASSERT (C->nvec_nonempty >= 0) ;
    return (GrB_SUCCESS) ;
}

