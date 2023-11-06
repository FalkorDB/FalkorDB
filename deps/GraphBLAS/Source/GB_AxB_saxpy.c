//------------------------------------------------------------------------------
// GB_AxB_saxpy: compute C=A*B, C<M>=A*B, or C<!M>=A*B
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_mxm.h"
#include "GB_AxB_saxpy.h"
#include "GB_stringify.h"

// TODO: allow bitmap multiply to work in-place as well

GrB_Info GB_AxB_saxpy               // C = A*B using Gustavson/Hash/Bitmap
(
    GrB_Matrix C,                   // output, static header
    GrB_Matrix C_in,                // original input matrix
    const GrB_Matrix M,             // optional mask matrix
    const bool Mask_comp,           // if true, use !M
    const bool Mask_struct,         // if true, use the only structure of M
    const GrB_BinaryOp accum,
    const GrB_Matrix A,             // input matrix A
    const GrB_Matrix B,             // input matrix B
    const GrB_Semiring semiring,    // semiring that defines C=A*B
    const bool flipxy,              // if true, do z=fmult(b,a) vs fmult(a,b)
    bool *mask_applied,             // if true, then mask was applied
    bool *done_in_place,            // if true, C was computed in-place 
    const GrB_Desc_Value AxB_method,
    const int do_sort,              // if nonzero, try to sort in saxpy3
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;
    (*mask_applied) = false ;
    ASSERT (C != NULL && (C->static_header || GBNSTATIC)) ;

    ASSERT_MATRIX_OK_OR_NULL (M, "M for saxpy A*B", GB0) ;
    ASSERT (!GB_PENDING (M)) ;
    ASSERT (GB_JUMBLED_OK (M)) ;
    ASSERT (!GB_ZOMBIES (M)) ;

    ASSERT_MATRIX_OK (A, "A for saxpy A*B", GB0) ;
    ASSERT (!GB_PENDING (A)) ;
    ASSERT (GB_JUMBLED_OK (A)) ;
    ASSERT (!GB_ZOMBIES (A)) ;

    ASSERT_MATRIX_OK (B, "B for saxpy A*B", GB0) ;
    ASSERT (!GB_PENDING (B)) ;
    ASSERT (GB_JUMBLED_OK (B)) ;
    ASSERT (!GB_ZOMBIES (B)) ;

    ASSERT_SEMIRING_OK (semiring, "semiring for saxpy A*B", GB0) ;
    ASSERT (A->vdim == B->vlen) ;

    //--------------------------------------------------------------------------
    // determine the sparsity of C
    //--------------------------------------------------------------------------

    int C_sparsity, saxpy_method ;
    GB_AxB_saxpy_sparsity (&C_sparsity, &saxpy_method,
        M, Mask_comp, A, B) ;

    //--------------------------------------------------------------------------
    // determine if C is iso
    //--------------------------------------------------------------------------

    GrB_Type ztype = semiring->add->op->ztype ;
    size_t zsize = ztype->size ;
    GB_void cscalar [GB_VLA(zsize)] ;
    bool C_iso = GB_AxB_iso (cscalar, A, B, A->vdim, semiring, flipxy, false) ;
    if (C_iso)
    {
        // revise the method if A and B are both iso and as-if-full
        if (A->iso && GB_as_if_full (A) && B->iso && GB_as_if_full (B))
        { 
            saxpy_method = GB_SAXPY_METHOD_ISO_FULL ;
            C_sparsity = GxB_FULL ;
        }
    }

    //--------------------------------------------------------------------------
    // determine if saxpy4 or saxpy5 can be used: C += A*B where C is full
    //--------------------------------------------------------------------------

    if (!C_iso                              // C must be non-iso on output
        && C_in != NULL                     // GB_AxB_meta says it is OK
        && GB_IS_FULL (C_in)                // C must be full
        && M == NULL                        // no mask present
        && (accum != NULL)                  // accum is present
        && (accum == semiring->add->op)     // accum is same as monoid
        && (C_in->type == accum->ztype))    // no typecast from accum output
    {
        if ((GB_IS_SPARSE (A) || GB_IS_HYPERSPARSE (A))
        &&  (GB_IS_BITMAP (B) || GB_IS_FULL (B)))
        { 
            // GB_AxB_saxpy4 computes C += A*B where C is full, no mask
            // is present, accum is present and matches the monoid, no
            // typecasting, A is sparse or hypersparse, and B is bitmap or
            // full.  The ANY monoid is not supported since it would be
            // unusual to use ANY as the accum.  C may be iso on input but the
            // method is not used if C is iso on output.  The type of C must
            // match the ztype of the monoid, but the JIT may do any
            // typecasting with A and B.

            info = GB_AxB_saxpy4 (C_in, A, B, semiring, flipxy, done_in_place,
                Werk) ;
            if (info != GrB_NO_VALUE)
            { 
                // return if saxpy4 has handled this case, otherwise fall
                // through to saxpy3, dot2, or bitmap_saxpy below.
                return (info) ;
            }
        }
        else if ((GB_IS_BITMAP (A) || GB_IS_FULL (A))
             &&  (GB_IS_SPARSE (B) || GB_IS_HYPERSPARSE (B))
             && A->type == (flipxy ? semiring->multiply->ytype :
                                     semiring->multiply->xtype))
        {
            // GB_AxB_saxpy5 computes C+=A*B where C is full, just like
            // GB_AxB_saxpy4, except that the sparsity format of A and B are
            // reversed.  A is bitmap or full, and B is sparse or hypersparse.
            // A->type must match the multiply input type (xtype if flipxy
            // false, ytype if true).  The type of C must match the ztype of
            // the monoid.  The JIT may do any typecasting with B.

            info = GB_AxB_saxpy5 (C_in, A, B, semiring, flipxy, done_in_place,
                Werk) ;
            if (info != GrB_NO_VALUE)
            { 
                // return if saxpy5 has handled this case, otherwise fall
                // through to saxpy3, dot2, or bitmap_saxpy below.
                return (info) ;
            }
        }
    }

    //--------------------------------------------------------------------------
    // burble
    //--------------------------------------------------------------------------

    if (M == NULL)
    { 
        GBURBLE ("(%s = %s*%s, anz: %g bnz: %g) ",
            GB_sparsity_char (C_sparsity),
            GB_sparsity_char_matrix (A),
            GB_sparsity_char_matrix (B),
            (double) GB_nnz (A), (double) GB_nnz (B)) ;
    }
    else
    { 
        GBURBLE ("(%s%s%s%s%s = %s*%s) ",
            GB_sparsity_char (C_sparsity),
            Mask_struct ? "{" : "<",
            Mask_comp ? "!" : "",
            GB_sparsity_char_matrix (M),
            Mask_struct ? "}" : ">",
            GB_sparsity_char_matrix (A),
            GB_sparsity_char_matrix (B)) ;
    }

    //--------------------------------------------------------------------------
    // select the method to use
    //--------------------------------------------------------------------------

    if (saxpy_method == GB_SAXPY_METHOD_ISO_FULL)
    {

        //----------------------------------------------------------------------
        // C is iso and full; do not apply the mask
        //----------------------------------------------------------------------

        GBURBLE ("(iso full saxpy) ") ;
        ASSERT (C_sparsity == GxB_FULL) ;
        // set C->iso = true    OK
        info = GB_new_bix (&C, // existing header
            ztype, A->vlen, B->vdim, GB_Ap_null, true, GxB_FULL, false,
            GB_HYPER_SWITCH_DEFAULT, -1, 1, true, true) ;
        if (info == GrB_SUCCESS)
        { 
            C->magic = GB_MAGIC ;
            memcpy (C->x, cscalar, zsize) ;
        }

    }
    else if (saxpy_method == GB_SAXPY_METHOD_3)
    {

        //----------------------------------------------------------------------
        // saxpy3: general-purpose Gustavson/Hash method, C is sparse/hyper
        //----------------------------------------------------------------------

        // C is sparse or hypersparse

        // This method allocates its own workspace, which is very small if the
        // Hash method is used.  The workspace for Gustavson's method is
        // larger, but saxpy3 selects that method only if the total work is
        // high enough so that the time to initialize the space.  C is sparse
        // or hypersparse.

        ASSERT (C_sparsity == GxB_HYPERSPARSE || C_sparsity == GxB_SPARSE) ;
        info = GB_AxB_saxpy3 (C, C_iso, cscalar, C_sparsity, M, Mask_comp,
            Mask_struct, A, B, semiring, flipxy, mask_applied, AxB_method,
            do_sort, Werk) ;

        if (info == GrB_NO_VALUE)
        { 
            // The mask is present but has been discarded since it results in
            // too much work.  The analysis must be redone, which is done by
            // calling this function once again, recursively, without the mask.
            // GB_AxB_saxpy_sparsity will be called again, and it might choose
            // the bitmap method instead.  If saxpy3 is still chosen, this
            // results in a different analysis in GB_AxB_saxpy3, with no mask
            // present.  Otherwise, GB_AxB_saxbit, below, is called.
            ASSERT (M != NULL) ;
            info = GB_AxB_saxpy (C, NULL, NULL, false, false, NULL, A, B,
                semiring, flipxy, mask_applied, done_in_place, AxB_method,
                do_sort, Werk) ;
        }

    }
    else
    {

        //----------------------------------------------------------------------
        // bitmap method: C is bitmap
        //----------------------------------------------------------------------

        ASSERT (C_sparsity == GxB_BITMAP) ;

        if ((GB_IS_BITMAP (A) || GB_IS_FULL (A)) &&
            (GB_IS_SPARSE (B) || GB_IS_HYPERSPARSE (B)))
        { 
            // C<#M> = A*B via dot products, where A is bitmap or full and B is
            // sparse or hypersparse, using the dot2 method with A not
            // explicitly transposed.
            info = GB_AxB_dot2 (C, C_iso, cscalar, M, Mask_comp, Mask_struct,
                true, A, B, semiring, flipxy, Werk) ;
        }
        else
        { 

            // C<#M> = A*B via bitmap saxpy method
            info = GB_AxB_saxbit (C, C_iso, cscalar, M,
                Mask_comp, Mask_struct, A, B, semiring, flipxy, Werk) ;
        }

        // the mask is always applied if present
        (*mask_applied) = (M != NULL && info == GrB_SUCCESS) ;
    }

    return (info) ;
}

