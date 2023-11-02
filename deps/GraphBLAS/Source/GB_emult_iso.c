//------------------------------------------------------------------------------
// GB_emult_iso: apply a binary op and check for iso result for C=A.*B
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Compute c = op(a,b) for two matrices A and B, and return true if C=A.*B or
// C=kron(A,B) results in an iso matrix C.  If true, the output scalar c is the
// iso value for the matrix C.

#include "GB_emult.h"

bool GB_emult_iso           // c = op(a,b), return true if C is iso
(
    // output
    GB_void *restrict c,    // output scalar of iso array
    // input
    GrB_Type ctype,         // type of c
    GrB_Matrix A,           // input matrix
    GrB_Matrix B,           // input matrix
    GrB_BinaryOp op         // binary operator
)
{

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    ASSERT_MATRIX_OK (A, "A for GB_emult_iso", GB0) ;
    ASSERT_MATRIX_OK (B, "B for GB_emult_iso", GB0) ;
    ASSERT_TYPE_OK (ctype, "ctype for GB_emult_iso", GB0) ;
    ASSERT_BINARYOP_OK (op, "op for GB_emult_iso", GB0) ;
    ASSERT (c != NULL) ;

    //--------------------------------------------------------------------------
    // quick return if op is positional
    //--------------------------------------------------------------------------

    if (GB_OP_IS_POSITIONAL (op))
    { 
        // C is not iso if the op is positional
        return (false) ;
    }

    //--------------------------------------------------------------------------
    // get the binary operator and the types of C, A, and B
    //--------------------------------------------------------------------------

    const GxB_binary_function femult = op->binop_function ;
    GB_Opcode opcode = op->opcode ;

    const GrB_Type xtype = op->xtype ;
    const GrB_Type ytype = op->ytype ;
    const GrB_Type ztype = op->ztype ;

    const GB_Type_code xcode = xtype->code ;
    const GB_Type_code ycode = ytype->code ;
    const GB_Type_code zcode = ztype->code ;
    const GB_Type_code ccode = ctype->code ;
    const GB_Type_code acode = A->type->code ;
    const GB_Type_code bcode = B->type->code ;

    const size_t xsize = xtype->size ;
    const size_t ysize = ytype->size ;
    const size_t zsize = ztype->size ;
    const size_t csize = ctype->size ;
    const size_t asize = A->type->size ;
    const size_t bsize = B->type->size ;

    //--------------------------------------------------------------------------
    // determine if C is iso
    //--------------------------------------------------------------------------

    if (opcode == GB_PAIR_binop_code)
    { 

        //----------------------------------------------------------------------
        // C is iso, with c = 1
        //----------------------------------------------------------------------

        GB_cast_one (c, ccode) ;
        return (true) ;

    }
    else if (B->iso &&
            (opcode == GB_SECOND_binop_code || opcode == GB_ANY_binop_code))
    { 

        //----------------------------------------------------------------------
        // C is iso, with c = b
        //----------------------------------------------------------------------

        if (ccode == ycode && bcode == ycode)
        { 
            // c = Bx [0]
            memcpy (c, B->x, csize) ;
        }
        else
        { 
            // c = (ctype) ((ytype) Bx [0])
            GB_void y [GB_VLA(ysize)] ;
            GB_cast_scalar (y, ycode, B->x, bcode, bsize) ;
            GB_cast_scalar (c, ccode, y, ycode, ysize) ;
        }
        return (true) ;

    }
    else if (A->iso &&
            (opcode == GB_FIRST_binop_code || opcode == GB_ANY_binop_code))
    { 

        //----------------------------------------------------------------------
        // C is iso, with c = a
        //----------------------------------------------------------------------

        if (ccode == xcode && acode == xcode)
        { 
            // c = Ax [0]
            memcpy (c, A->x, csize) ;
        }
        else
        { 
            // c = (ctype) ((xtype) Ax [0])
            GB_void x [GB_VLA(xsize)] ;
            GB_cast_scalar (x, xcode, A->x, acode, asize) ;
            GB_cast_scalar (c, ccode, x, xcode, xsize) ;
        }
        return (true) ;

    }
    else if (A->iso && B->iso)
    { 

        //----------------------------------------------------------------------
        // C is iso, with c = op(a,b), for any op, including user-defined
        //----------------------------------------------------------------------

        if (acode == xcode && bcode == ycode && ccode == zcode)
        { 
            // c = op (Ax [0], Bx [0])
            femult (c, A->x, B->x) ;
        }
        else
        { 
            // x = (xtype) Ax [0]
            GB_void x [GB_VLA(xsize)] ;
            GB_cast_scalar (x, xcode, A->x, acode, asize) ;
            // y = (ytype) Bx [0]
            GB_void y [GB_VLA(ysize)] ;
            GB_cast_scalar (y, ycode, B->x, bcode, bsize) ;
            // z = op (x,y)
            GB_void z [GB_VLA(zsize)] ;
            femult (z, x, y) ;
            // c = (ctype) z
            GB_cast_scalar (c, ccode, z, zcode, zsize) ;
        }
        return (true) ;
    }

    //--------------------------------------------------------------------------
    // otherwise, C is not iso
    //--------------------------------------------------------------------------

    return (false) ;
}

