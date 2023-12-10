//------------------------------------------------------------------------------
// GB_unop_iso: apply a unary or binary op (with scalar) with an iso result
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// The input matrix A need not be entirely valid.  GB_transpose can be
// transposing the matrix in place, which case the contents of A have already
// been transplanted into T, and only A->x remains.

#include "GB.h"

void GB_unop_iso            // Cx [0] = unop (A), binop (s,A) or binop (A,s)
(
    // output
    GB_void *restrict Cx,   // output scalar of iso array
    // input
    GrB_Type ctype,         // type of Cx
    GB_iso_code C_code_iso, // defines how C iso value is to be computed
    GB_Operator op,         // unary/binop operator, if present
    GrB_Matrix A,           // input matrix
    GrB_Scalar scalar       // input scalar (may be NULL)
)
{

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    ASSERT (A != NULL && A->type != NULL) ;
    ASSERT_TYPE_OK (ctype, "ctype for GB_unop_iso", GB0) ;
    ASSERT (Cx != NULL) ;

    GrB_Type stype = (scalar != NULL) ? scalar->type : GrB_BOOL ;
//  const size_t csize = ctype->size ;
    const size_t asize = A->type->size ;
    const size_t ssize = stype->size ;
    const GB_Type_code ccode = ctype->code ;
    const GB_Type_code acode = A->type->code ;
    const GB_Type_code scode = stype->code ;

    //--------------------------------------------------------------------------
    // compute the C iso value
    //--------------------------------------------------------------------------

    if (C_code_iso == GB_ISO_1)
    { 

        //----------------------------------------------------------------------
        // Cx [0] = (ctype) 1, via the PAIR binary op or ONE unary op
        //----------------------------------------------------------------------

        GB_cast_one (Cx, ccode) ;

    }
    else if (C_code_iso == GB_ISO_S)
    { 

        //----------------------------------------------------------------------
        // Cx [0] = (ctype) scalar via FIRST(s,A), SECOND(A,s), ANY(..), ...
        //----------------------------------------------------------------------

        ASSERT_SCALAR_OK (scalar, "scalar for GB_unop_iso", GB0) ;
        GB_cast_scalar (Cx, ccode, scalar->x, scode, ssize) ;

    }
    else
    {

        //----------------------------------------------------------------------
        // Cx [0] depends on the iso value of A
        //----------------------------------------------------------------------

        ASSERT (A->x != NULL && A->x_size >= asize) ;
        ASSERT (A->iso) ;

        if (C_code_iso == GB_ISO_A)
        { 
            //------------------------------------------------------------------
            // Cx [0] = (ctype) A
            //------------------------------------------------------------------

            GB_cast_scalar (Cx, ccode, A->x, acode, asize) ;

        }
        else if (C_code_iso == GB_ISO_OP1_A)
        { 

            //------------------------------------------------------------------
            // Cx [0] = unop (A)
            //------------------------------------------------------------------

            ASSERT_OP_OK (op, "op for GB_unop_iso", GB0) ;

            // x = (xtype) Ax [0]
            GB_Type_code xcode = op->xtype->code ;
            size_t xsize = op->xtype->size ;
            GB_void x [GB_VLA(xsize)] ;
            GB_cast_scalar (x, xcode, A->x, acode, asize) ;

            // Cx [0] = op (x)
            GxB_unary_function fop = op->unop_function ;
            fop (Cx, x) ;

        }
        else
        { 

            //------------------------------------------------------------------
            // Cx [0] = binop (scalar,A) or binop (A,scalar)
            //------------------------------------------------------------------

            ASSERT_OP_OK (op, "op for GB_unop_iso", GB0) ;
            ASSERT_SCALAR_OK (scalar, "scalar for GB_unop_iso binop", GB0) ;
            GB_Type_code xcode = op->xtype->code ;
            GB_Type_code ycode = op->ytype->code ;
            size_t xsize = op->xtype->size ;
            size_t ysize = op->ytype->size ;
            GxB_binary_function fop = op->binop_function ;
            GB_void x [GB_VLA(xsize)] ;
            GB_void y [GB_VLA(ysize)] ;

            if (C_code_iso == GB_ISO_OP2_SA)
            { 

                //--------------------------------------------------------------
                // Cx [0] = binop (scalar, A)
                //--------------------------------------------------------------

                GB_cast_scalar (x, xcode, scalar->x, scode, ssize) ;
                GB_cast_scalar (y, ycode, A->x, acode, asize) ;

            }
            else // (C_code_iso == GB_ISO_OP2_AS)
            { 

                //--------------------------------------------------------------
                // Cx [0] = binop (A, scalar)
                //--------------------------------------------------------------

                GB_cast_scalar (x, xcode, A->x, acode, asize) ;
                GB_cast_scalar (y, ycode, scalar->x, scode, ssize) ;

            }

            // Cx [0] = binop (x, y)
            fop (Cx, x, y) ;
        }
    }
}

