//------------------------------------------------------------------------------
// GB_AxB_saxpy_sparsity: determine the sparsity structure for C<M or !M>=A*B
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Determines the sparsity structure for C, for computing C=A*B, C<M>=A*B, or
// C<!M>=A*B, based on the sparsity structures of C (on input), M, A, and B,
// and whether or not M is complemented.

// TODO: When A or B are bitmapped or full, they can be transposed in-place.
// TODO: give the user control over this decision

//------------------------------------------------------------------------------

#include "GB_AxB_saxpy.h"

void GB_AxB_saxpy_sparsity          // determine C_sparsity and method to use
(
    // output:
    int *C_sparsity,                // sparsity structure of C
    int *saxpy_method,              // saxpy method to use
    // input:
    const GrB_Matrix M,             // optional mask for C, unused if NULL
    const bool Mask_comp,           // if true, use !M
    const GrB_Matrix A,             // input A matrix
    const GrB_Matrix B              // input B matrix
)
{

    //--------------------------------------------------------------------------
    // determine the sparsity of C
    //--------------------------------------------------------------------------

    if (B->nvec_nonempty < 0)
    { 
        // B->nvec_nonempty is used to select the method
        B->nvec_nonempty = GB_nvec_nonempty (B) ;
    }
    double bnvec = B->nvec_nonempty ;

    double m = (double) A->vlen ;
    double n = (double) B->vdim ;
    double anz = (double) GB_nnz_held (A) ;
//  double bnz = (double) GB_nnz_held (B) ;

    int M_sparsity = (M == NULL) ? 0 : GB_sparsity (M) ;
    int B_sparsity = GB_sparsity (B) ;
    int A_sparsity = GB_sparsity (A) ;
    bool M_is_hyper  = (M_sparsity == GxB_HYPERSPARSE) ;
    bool M_is_sparse = (M_sparsity == GxB_SPARSE) ;

    if (M != NULL && !Mask_comp && (M_is_hyper || M_is_sparse))
    {

        //-----------------------------------------------------
        // C               <M>=             A     *     B
        //-----------------------------------------------------

        // hyper            sparse          any         hyper
        // hyper            hyper           any         hyper
        // sparse           hyper           any         sparse/bitmap/full
        // sparse           sparse          any         sparse/bitmap/full

        // The non-empty columns of C are a subset of the non-empty columns of
        // B, so in general, if B is hypersparse, so is C.  If B is sparse,
        // bitmap, or full, then C must be sparse, regardless of the sparsity
        // of A and B.  This is a restriction of GB_AxB_saxpy3.c.

        if (B_sparsity == GxB_HYPERSPARSE)
        { 
            (*C_sparsity) = GxB_HYPERSPARSE ;
        }
        else
        { 
            (*C_sparsity) = GxB_SPARSE ;
        }

        (*saxpy_method) = GB_SAXPY_METHOD_3 ;

    }
    else
    {

        //-----------------------------------------------------
        // C                =               A     *     B
        //-----------------------------------------------------

        // hyper            .               hyper       hyper
        // hyper            .               sparse      hyper
        // hyper            .               bitmap      hyper
        // hyper            .               full        hyper

        // sparse           .               hyper       sparse
        // sparse           .               sparse      sparse
        // sparse/bitmap    .               bitmap      sparse
        // sparse/bitmap    .               full        sparse

        // sparse/bitmap    .               hyper       bitmap
        // sparse/bitmap    .               sparse      bitmap
        // bitmap           .               bitmap      bitmap
        // bitmap           .               full        bitmap

        // sparse/bitmap    .               hyper       full 
        // sparse/bitmap    .               sparse      full
        // bitmap           .               bitmap      full
        // bitmap (***)     .               full        full

        //    (***): future, compute C as full

        //-----------------------------------------------------
        // C               <M>=             A     *     B
        //-----------------------------------------------------

        // hyper            any             hyper       hyper
        // hyper            any             sparse      hyper
        // hyper            any             bitmap      hyper
        // hyper            any             full        hyper

        // sparse           any             hyper       sparse
        // sparse           any             sparse      sparse
        // sparse/bitmap    any             bitmap      sparse
        // sparse/bitmap    any             full        sparse

        // sparse/bitmap    any             hyper       bitmap
        // sparse/bitmap    any             sparse      bitmap
        // bitmap           any             bitmap      bitmap
        // bitmap           any             full        bitmap

        // sparse/bitmap    bitmap/full     hyper       full    (*)
        // sparse/bitmap    bitmap/full     sparse      full    (*)
        // bitmap           bitmap/full     bitmap      full    (*)
        // bitmap           bitmap/full     full        full    (*)

        // (*): if M hyper/sparse, then C is hyper/sparse; see above

        //-----------------------------------------------------
        // C               <!M>=            A     *     B
        //-----------------------------------------------------

        // hyper            any             hyper       hyper
        // hyper            any             sparse      hyper
        // hyper            any             bitmap      hyper
        // hyper            any             full        hyper

        // sparse           any             hyper       sparse
        // sparse           any             sparse      sparse
        // sparse/bitmap    any             bitmap      sparse
        // sparse/bitmap    any             full        sparse

        // sparse/bitmap    any             hyper       bitmap
        // sparse/bitmap    any             sparse      bitmap
        // bitmap           any             bitmap      bitmap
        // bitmap           any             full        bitmap

        // sparse/bitmap    any             hyper       full 
        // sparse/bitmap    any             sparse      full
        // bitmap           any             bitmap      full
        // bitmap           any             full        full

        // If M is complemented, or not complemented and bitmap/full, then C
        // has the same sparsity as listed above, except when A and B are both
        // full.

        // For the cases where C is labelled as hyper/bitmap or sparse/bitmap:
        // If m*n is much larger than nnz(A)+nnz(B), then always construct C as
        // sparse/hyper, not bitmap.   TODO: give the user control over this
        // decision.

        // TODO:  for bitmap*hyper and hyper*bitmap, create a hyper_shallow
        // version of the hyper matrix (like dot does), and construct C as
        // bitmap.  Then expand into C into hyper.

        switch (B_sparsity)
        {
            case GxB_HYPERSPARSE : 

                // H = any * H
                (*C_sparsity) = GxB_HYPERSPARSE ;
                break ;

            case GxB_SPARSE : 

                switch (A_sparsity)
                {
                    case GxB_HYPERSPARSE : 
                    case GxB_SPARSE : 
                        // S = {S,H} * S : C has the same sparsity as B
                        (*C_sparsity) = GxB_SPARSE ;
                        break ;
                    case GxB_BITMAP : 
                    case GxB_FULL : 
                        // S = {B,F} * S : if B has many empty columns
                        // B = {B,F} * S : otherwise C is bitmap
                        (*C_sparsity) = (bnvec < n/4) ? GxB_SPARSE : GxB_BITMAP;
                        break ;
                    default: ;
                }
                break ;

            case GxB_BITMAP : 
            case GxB_FULL : 

                switch (A_sparsity)
                {
                    case GxB_HYPERSPARSE : 
                    case GxB_SPARSE : 
                        // S = {S,H} * {B,F} : if A is very sparse
                        // B = {S,H} * {B,F} : otherwise C is bitmap
                        (*C_sparsity) = (anz < m/20) ? GxB_SPARSE : GxB_BITMAP ;
                        break ;
                    case GxB_BITMAP : 
                    case GxB_FULL : 
                        // B = {B,F} * {B,F} : C is bitmap
                        (*C_sparsity) = GxB_BITMAP ;
                        break ;
                    default: ;
                }
                break ;

            default: ;
        }

        if ((*C_sparsity) == GxB_HYPERSPARSE || (*C_sparsity) == GxB_SPARSE)
        { 
            (*saxpy_method) = GB_SAXPY_METHOD_3 ;
        }
        else
        { 
            (*saxpy_method) = GB_SAXPY_METHOD_BITMAP ;
        }
    }

    if ((*C_sparsity) == GxB_HYPERSPARSE || (*C_sparsity) == GxB_SPARSE)
    {
        // If C is sparse or hypersparse, then it will be computed by
        // GB_AxB_saxpy3.  For this method, if B is hypersparse, C must also be
        // hypersparse.  Otherwise C must be sparse.  This is a requirement of
        // GB_AxB_saxpy3, and is also asserted there.
        ASSERT ((*C_sparsity) ==
            ((B_sparsity == GxB_HYPERSPARSE) ? GxB_HYPERSPARSE : GxB_SPARSE)) ;
    }
}

