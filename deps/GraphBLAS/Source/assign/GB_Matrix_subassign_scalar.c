//------------------------------------------------------------------------------
// GB_Matrix_subassign_scalar: assign to submatrix, via scalar expansion
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Assigns a single scalar to a submatrix:

// C(I,J)<M> = accum(C(I,J),x)

// The scalar x is implicitly expanded into a matrix A of size nI-by-nJ,
// with each entry in A equal to x.

// Compare with GrB_Matrix_assign_scalar,
// which uses M and C_Replace differently.

#include "assign/GB_subassign.h"
#include "ij/GB_ij.h"
#include "mask/GB_get_mask.h"
#define GB_FREE_ALL GB_Matrix_free (&A) ;

GrB_Info GB_Matrix_subassign_scalar   // C(I,J)<M> = accum (C(I,J),s)
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix Mask,          // optional mask for C, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C(I,J),x)
    const GrB_Scalar scalar,        // scalar to assign to C(I,J)
    const void *I,                  // row indices
    const bool I_is_32,
    uint64_t ni,                    // number of row indices
    const void *J,                  // column indices
    const bool J_is_32,
    uint64_t nj,                    // number of column indices
    const GrB_Descriptor desc,      // descriptor for C and Mask
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;
    GrB_Matrix A = NULL ;
    GB_RETURN_IF_NULL (C) ;
    GB_RETURN_IF_NULL (scalar) ;
    GB_RETURN_IF_NULL (I) ;
    GB_RETURN_IF_NULL (J) ;
    GB_RETURN_IF_OUTPUT_IS_READONLY (C) ;

    // if C has a user-defined type, its type must match the scalar type
    if (C->type->code == GB_UDT_code && C->type != scalar->type)
    { 
        GB_ERROR (GrB_DOMAIN_MISMATCH, "Input of type [%s]\n"
            "cannot be typecast to output of type [%s]",
            scalar->type->name, C->type->name) ;
    }

    // get the descriptor
    GB_GET_DESCRIPTOR (info, desc, C_replace, Mask_comp, Mask_struct,
        xx1, xx2, xx3, xx7) ;

    // get the mask
    GrB_Matrix M = GB_get_mask (Mask, &Mask_comp, &Mask_struct) ;

    //--------------------------------------------------------------------------
    // C(I,J)<M> = accum (C(I,J), scalar)
    //--------------------------------------------------------------------------

    uint64_t nvals ;
    GB_OK (GB_nvals (&nvals, (GrB_Matrix) scalar, Werk)) ;

    if (M == NULL && !Mask_comp && ni == 1 && nj == 1 && !C_replace)
    {

        //----------------------------------------------------------------------
        // scalar assignment
        //----------------------------------------------------------------------

        uint64_t row, col ;
        if (I_is_32)
        { 
            const uint32_t *I32 = (uint32_t *) I ;
            row = I32 [0] ;
        }
        else
        { 
            const uint64_t *I64 = (uint64_t *) I ;
            row = I64 [0] ;
        }
        if (J_is_32)
        { 
            const uint32_t *J32 = (uint32_t *) J ;
            col = J32 [0] ;
        }
        else
        { 
            const uint64_t *J64 = (uint64_t *) J ;
            col = J64 [0] ;
        }

        if (nvals == 1)
        { 
            // set the element: C(row,col) += scalar or C(row,col) = scalar
            info = GB_setElement (C, accum, scalar->x, row, col,
                scalar->type->code, Werk) ;
        }
        else if (accum == NULL)
        { 
            // delete the C(row,col) element
            info = GB_Matrix_removeElement (C, row, col, Werk) ;
        }

    }
    else if (nvals == 1)
    { 

        //----------------------------------------------------------------------
        // the opaque GrB_Scalar has a single entry
        //----------------------------------------------------------------------

        // This is identical to non-opaque scalar subassignment

        info = GB_subassign (
            C, C_replace,               // C matrix and its descriptor
            M, Mask_comp, Mask_struct,  // mask matrix and its descriptor
            false,                      // do not transpose the mask
            accum,                      // for accum (C(I,J),scalar)
            NULL, false,                // no explicit matrix A
            I, I_is_32, ni,             // row indices
            J, J_is_32, nj,             // column indices
            true,                       // do scalar expansion
            scalar->x,                  // scalar to assign, expands to become A
            scalar->type->code,         // type code of scalar to expand
            Werk) ;

    }
    else
    { 

        //----------------------------------------------------------------------
        // the opaque GrB_Scalar has no entry
        //----------------------------------------------------------------------

        // determine the properites of the I and J index lists
        int64_t nI, nJ, Icolon [3], Jcolon [3] ;
        int I_Kind, J_Kind ;
        GB_ijlength (I, I_is_32, ni, GB_NROWS (C), &nI, &I_Kind, Icolon) ;
        GB_ijlength (J, J_is_32, nj, GB_NCOLS (C), &nJ, &J_Kind, Jcolon) ;

        // create an empty matrix A of the right size, and use matrix assign
        struct GB_Matrix_opaque A_header ;
        GB_CLEAR_MATRIX_HEADER (A, &A_header) ;
        bool is_csc = C->is_csc ;
        int64_t vlen = is_csc ? nI : nJ ;
        int64_t vdim = is_csc ? nJ : nI ;
        GB_OK (GB_new (&A, // existing header
            scalar->type, vlen, vdim, GB_ph_calloc, is_csc, GxB_AUTO_SPARSITY,
            GB_HYPER_SWITCH_DEFAULT, 1, /* OK: */ false, false, false)) ;
        info = GB_subassign (
            C, C_replace,                   // C matrix and its descriptor
            M, Mask_comp, Mask_struct,      // mask matrix and its descriptor
            false,                          // do not transpose the mask
            accum,                          // for accum (C(I,J),A)
            A, false,                       // A matrix and its descriptor
            I, I_is_32, ni,                 // row indices
            J, J_is_32, nj,                 // column indices
            false, NULL, GB_ignore_code,    // no scalar expansion
            Werk) ;
        GB_FREE_ALL ;
    }

    return (info) ;
}

