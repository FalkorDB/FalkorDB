//------------------------------------------------------------------------------
// GB_Vector_subassign_scalar: assign scalar to vector, via scalar expansion
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Assigns a single scalar to a subvector, w(Rows)<M> = accum(w(Rows),x)
// The scalar x is implicitly expanded into a vector u of size nRows-by-1,
// with each entry in u equal to x.

// The actual work is done in GB_subassign_scalar.c.

#include "assign/GB_subassign.h"
#include "ij/GB_ij.h"
#include "mask/GB_get_mask.h"
#define GB_FREE_ALL GB_Matrix_free (&A) ;

// If the GrB_Scalar s is non-empty, then this is the same as the non-opapue
// scalar assignment above.

// If the GrB_Scalar s is empty of type stype, then this is identical to:
//  GrB_Vector_new (&A, stype, nRows) ;
//  GxB_Vector_subassign (w, M, accum, A, Rows, nRows, desc) ;
//  GrB_Vector_free (&A) ;

GrB_Info GB_Vector_subassign_scalar // w(I)><Mask> = accum (w(I),s)
(
    GrB_Vector w,                   // input/output matrix for results
    const GrB_Vector mask,          // optional mask for w, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(w(I),x)
    const GrB_Scalar scalar,        // scalar to assign to w(I)
    const uint64_t *I,              // row indices
    const bool I_is_32,
    uint64_t ni,                    // number of row indices
    const GrB_Descriptor desc,      // descriptor for w and Mask
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;
    GrB_Matrix A = NULL ;
    GB_RETURN_IF_NULL (w) ;
    GB_RETURN_IF_NULL (scalar) ;
    GB_RETURN_IF_NULL (I) ;
    GB_RETURN_IF_OUTPUT_IS_READONLY (w) ;

    ASSERT (GB_VECTOR_OK (w)) ;
    ASSERT (mask == NULL || GB_VECTOR_OK (mask)) ;

    // if w has a user-defined type, its type must match the scalar type
    if (w->type->code == GB_UDT_code && w->type != scalar->type)
    { 
        GB_ERROR (GrB_DOMAIN_MISMATCH, "Input of type [%s]\n"
            "cannot be typecast to output of type [%s]",
            scalar->type->name, w->type->name) ;
    }

    // get the descriptor
    GB_GET_DESCRIPTOR (info, desc, C_replace, Mask_comp, Mask_struct,
        xx1, xx2, xx3, xx7) ;

    // get the mask
    GrB_Matrix M = GB_get_mask ((GrB_Matrix) mask, &Mask_comp, &Mask_struct) ;

    //--------------------------------------------------------------------------
    // w(Rows)<M> = accum (w(Rows), scalar)
    //--------------------------------------------------------------------------

    uint64_t nvals ;
    GB_OK (GB_nvals (&nvals, (GrB_Matrix) scalar, Werk)) ;

    if (M == NULL && !Mask_comp && ni == 1 && !C_replace)
    {

        //----------------------------------------------------------------------
        // scalar assignment
        //----------------------------------------------------------------------

        const uint64_t row = I [0] ;
        if (nvals == 1)
        { 
            // set the element: w(row) += scalar or w(wrow) = scalar
            info = GB_setElement ((GrB_Matrix) w, accum, scalar->x, row, 0,
                scalar->type->code, Werk) ;
        }
        else if (accum == NULL)
        { 
            // delete the w(row) element
            info = GB_Vector_removeElement (w, row, Werk) ;
        }

    }
    else if (nvals == 1)
    { 

        //----------------------------------------------------------------------
        // the opaque GrB_Scalar has a single entry
        //----------------------------------------------------------------------

        // This is identical to non-opaque scalar assignment

        info = GB_subassign (
            (GrB_Matrix) w, C_replace,  // w vector and its descriptor
            M, Mask_comp, Mask_struct,  // mask vector and its descriptor
            false,                      // do not transpose the mask
            accum,                      // for accum (w(Rows),scalar)
            NULL, false,                // no explicit vector u
            I, false, ni,               // row indices
            GrB_ALL, false, 1,          // column indices
            true,                       // do scalar expansion
            scalar->x,                  // scalar to assign, expands to become u
            scalar->type->code,         // type code of scalar to expand
            Werk) ;

    }
    else
    { 

        //----------------------------------------------------------------------
        // the opaque GrB_Scalar has no entry
        //----------------------------------------------------------------------

        // determine the properites of the I index list
        int64_t nRows, RowColon [3] ;
        int RowsKind ;
        GB_ijlength (I, false, ni, GB_NROWS (w), &nRows, &RowsKind, RowColon) ;

        // create an empty matrix A of the right size, and use matrix assign
        struct GB_Matrix_opaque A_header ;
        GB_CLEAR_MATRIX_HEADER (A, &A_header) ;
        GB_OK (GB_new (&A,  // existing header
            scalar->type, nRows, 1, GB_ph_calloc, true, GxB_AUTO_SPARSITY,
            GB_HYPER_SWITCH_DEFAULT, 1, /* OK: */ false, false, false)) ;
        info = GB_subassign (
            (GrB_Matrix) w, C_replace,      // w vector and its descriptor
            M, Mask_comp, Mask_struct,      // mask matrix and its descriptor
            false,                          // do not transpose the mask
            accum,                          // for accum (w(Rows),scalar)
            A, false,                       // A matrix and its descriptor
            I, false, ni,                   // row indices
            GrB_ALL, false, 1,              // column indices
            false, NULL, GB_ignore_code,    // no scalar expansion
            Werk) ;
        GB_FREE_ALL ;
    }

    return (info) ;
}

