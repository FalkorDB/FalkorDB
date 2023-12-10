//------------------------------------------------------------------------------
// GB_AxB_meta_adotb_control: determine method for computing C=A'*B
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_mxm.h"

void GB_AxB_meta_adotb_control
(
    // output:
    int *axb_method,
    // input:
    const GrB_Matrix C_in,
    const GrB_Matrix M,
    bool Mask_comp,
    const GrB_Matrix A,
    const GrB_Matrix B,
    const GrB_BinaryOp accum,
    const GrB_Semiring semiring,    // semiring that defines C=A*B
    bool flipxy,
    bool can_do_in_place,
    bool allow_scale,
    bool B_is_diagonal,
    GrB_Desc_Value AxB_method
)
{

    // C=A'*B is being computed: use the dot product without computing A'
    // or use the saxpy (Gustavson) method

    // use saxpy by default, unless selecting other methods below
    (*axb_method) = GB_USE_SAXPY ;

    // If the mask is present, only entries for which M(i,j)=1 are
    // computed, which makes this method very efficient when the mask is
    // very sparse (triangle counting, for example).  Each entry C(i,j) for
    // which M(i,j)=1 is computed via a dot product, C(i,j) =
    // A(:,i)'*B(:,j).  If the mask is not present, the dot-product method
    // is very slow in general, and thus the saxpy method is usually used
    // instead.

    if (allow_scale && M == NULL
        && !GB_IS_BITMAP (A)     // TODO: A'*D colscale with A bitmap
        && B_is_diagonal)
    { 
        // C = A'*D, col scale
        (*axb_method) = GB_USE_COLSCALE ;
    }
    else if (allow_scale && M == NULL
        && !GB_IS_BITMAP (B)     // TODO: D*B rowscale with B bitmap
        && GB_is_diagonal (A))
    { 
        // C = D*B, row scale
        (*axb_method) = GB_USE_ROWSCALE ;
    }
    else if (AxB_method == GxB_DEFAULT)
    {
        // auto selection for A'*B
        bool C_out_iso = false ;    // ignored unless C can be done in-place
        if (can_do_in_place && C_in != NULL)
        { 
            // check if C will be iso on output (for dot4 control only).
            // Ignored if dot4 C_in is not present or C cannot be
            // computed in-place.
            C_out_iso = GB_AxB_iso (NULL, A, B, A->vlen, semiring, flipxy,
                false) ;
        }
        if (GB_AxB_dot4_control (C_out_iso, can_do_in_place ? C_in : NULL,
            M, Mask_comp, accum, semiring))
        { 
            // C+=A'*B can be done with dot4
            (*axb_method) = GB_USE_DOT ;
        }
        else if (GB_AxB_dot3_control (M, Mask_comp))
        { 
            // C<M>=A'*B uses the masked dot product method (dot3)
            (*axb_method) = GB_USE_DOT ;
        }
        else if (GB_AxB_dot2_control (A, B))
        { 
            // C=A'*B or C<!M>=A'B* can efficiently use the dot2 method
            (*axb_method) = GB_USE_DOT ;
        }
    }
    else if (AxB_method == GxB_AxB_DOT)
    { 
        // user selection for A'*B
        (*axb_method) = GB_USE_DOT ;
    }
}

