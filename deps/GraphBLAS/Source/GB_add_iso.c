//------------------------------------------------------------------------------
// GB_add_iso: apply a binary op and check for iso result for C=A+B
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Compute c = op(a,b) for two matrices A and B, and return true if C=A+B
// results in an iso matrix C.  If true, the output scalar c is the iso value
// for the matrix C.

#include "GB_add.h"
#include "GB_emult.h"

bool GB_add_iso             // c = op(a,b), return true if C is iso
(
    // output
    GB_void *restrict c,    // output scalar of iso array
    // input
    GrB_Type ctype,         // type of c
    GrB_Matrix A,           // input matrix
    const GB_void *restrict alpha_scalar,   // of type op->xtype
    GrB_Matrix B,           // input matrix
    const GB_void *restrict beta_scalar,    // of type op->ytype
    GrB_BinaryOp op,        // binary operator
    const bool A_and_B_are_disjoint,
    const bool is_eWiseUnion
)
{

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    ASSERT_MATRIX_OK (A, "A for GB_add_iso", GB0) ;
    ASSERT_MATRIX_OK (B, "B for GB_add_iso", GB0) ;
    ASSERT_TYPE_OK (ctype, "ctype for GB_add_iso", GB0) ;
    ASSERT_BINARYOP_OK (op, "op for GB_add_iso", GB0) ;
    ASSERT (c != NULL) ;

    //--------------------------------------------------------------------------
    // special case if both A and B are as-if-full
    //--------------------------------------------------------------------------

    if (GB_as_if_full (A) && GB_as_if_full (B))
    { 
        // A and B are both full (or as-if-full), and eWiseMult, eWiseAdd, and
        // eWiseUnion all compute the same thing.  GB_emult detects this
        // condition and calls GB_add, so that GB_emult doesn't have to handle
        // this case.  GB_add tests the iso condition as if it were computing
        // C=A.*B.  The alpha_scalar and beta_scalar values are not used.
        return (GB_emult_iso (c, ctype, A, B, op)) ;
    }

    //--------------------------------------------------------------------------
    // quick return if the op is positional
    //--------------------------------------------------------------------------

    if (GB_OP_IS_POSITIONAL (op))
    { 
        // C is not iso
        return (false) ;
    }

    if (!is_eWiseUnion && (!A->iso || !B->iso))
    { 
        // for eWiseAdd: A and B must be iso for C to be iso
        return (false) ;
    }

    //--------------------------------------------------------------------------
    // get the scalar codes and sizes for C, A, and B
    //--------------------------------------------------------------------------

    const size_t csize = ctype->size ;
    const size_t asize = A->type->size ;
    const size_t bsize = B->type->size ;

    const GB_Type_code ccode = ctype->code ;
    const GB_Type_code acode = A->type->code ;
    const GB_Type_code bcode = B->type->code ;

    // c = zero
    memset (c, 0, csize) ;

    //--------------------------------------------------------------------------
    // handle the iso case for eWiseUnion and eWiseAdd
    //--------------------------------------------------------------------------

    if (is_eWiseUnion)
    {

        //======================================================================
        // eWiseUnion:
        //======================================================================

        // for eWiseUnion: the type of C is always identical to op->ztype
        ASSERT (op != NULL) ;
        ASSERT (ctype == op->ztype) ;

        // get the binary operator
        const GxB_binary_function fadd = op->binop_function ;
        const GB_Opcode opcode = op->opcode ;

        const GrB_Type xtype = op->xtype ;
        const GrB_Type ytype = op->ytype ;

        const GB_Type_code xcode = xtype->code ;
        const GB_Type_code ycode = ytype->code ;

        const size_t xsize = xtype->size ;
        const size_t ysize = ytype->size ;

        if (opcode == GB_PAIR_binop_code)
        { 

            //------------------------------------------------------------------
            // C is iso, with c = 1
            //------------------------------------------------------------------

            GB_cast_one (c, ccode) ;
            return (true) ;

        }
        else if (B->iso &&
            (opcode == GB_SECOND_binop_code || opcode == GB_ANY_binop_code))
        {

            //------------------------------------------------------------------
            // if b == beta: C is iso, with c = b
            //------------------------------------------------------------------

            // b = (ytype) Bx [0]
            GB_void b [GB_VLA(ysize)] ;
            GB_cast_scalar (b, ycode, B->x, bcode, bsize) ;

            if (memcmp (b, beta_scalar, ysize) == 0)
            { 
                // c = b
                memcpy (c, b, ysize) ;
                return (true) ;
            }

        }
        else if (A->iso &&
            (opcode == GB_FIRST_binop_code || opcode == GB_ANY_binop_code))
        {

            //------------------------------------------------------------------
            // if a == alpha: C is iso, with c = a
            //------------------------------------------------------------------

            // a = (xtype) Ax [0]
            GB_void a [GB_VLA(xsize)] ;
            GB_cast_scalar (a, xcode, A->x, acode, asize) ;

            if (memcmp (a, alpha_scalar, xsize) == 0)
            { 
                // c = a
                memcpy (c, a, xsize) ;
                return (true) ;
            }

        }
        else if (A->iso && B->iso)
        {

            //------------------------------------------------------------------
            // C is iso if op(a,beta) == op(a,b) == op(alpha,b)
            //------------------------------------------------------------------

            // a = (xtype) Ax [0]
            GB_void a [GB_VLA(xsize)] ;
            GB_cast_scalar (a, xcode, A->x, acode, asize) ;

            // b = (ytype) Bx [0]
            GB_void b [GB_VLA(ysize)] ;
            GB_cast_scalar (b, ycode, B->x, bcode, bsize) ;

            // c = op(a,b)
            fadd (c, a, b) ;

            // t = op(a,beta)
            GB_void t [GB_VLA(csize)] ;
            fadd (t, a, beta_scalar) ;
            if (memcmp (c, t, csize) != 0)
            { 
                // op(a,b) != op(a,beta) so C is not iso
                return (false) ;
            }

            // t = op(alpha,b)
            fadd (t, alpha_scalar, b) ;
            if (memcmp (c, t, csize) != 0)
            { 
                // op(a,b) != op(alpha,b) so C is not iso
                return (false) ;
            }

            // C is iso with c = op(a,b)
            return (true) ;
        }

    }
    else
    {

        //======================================================================
        // eWiseAdd: both A and B must be iso
        //======================================================================

        //----------------------------------------------------------------------
        // compare the iso values of A and B, typecasted to C
        //----------------------------------------------------------------------

        // a = (ctype) Ax [0]
        GB_void a [GB_VLA(csize)] ;
        GB_cast_scalar (a, ccode, A->x, acode, asize) ;

        // b = (ctype) Bx [0]
        GB_void b [GB_VLA(csize)] ;
        GB_cast_scalar (b, ccode, B->x, bcode, bsize) ;

        if (memcmp (a, b, csize) != 0)
        {
            // the iso values of A and B differ, when typecasted to C
            return (false) ;
        }

        //----------------------------------------------------------------------
        // compute the C iso value and compare with A and B
        //----------------------------------------------------------------------

        if (A_and_B_are_disjoint)
        { 

            // For GB_wait, the pattern of A and B are known to be disjoint.
            // No typecasting is done in this case.
            ASSERT (ctype == A->type) ;
            memcpy (c, a, csize) ;
            return (true) ;

        }
        else
        {

            // get the binary operator
            const GxB_binary_function fadd = op->binop_function ;

            const GrB_Type xtype = op->xtype ;
            const GrB_Type ytype = op->ytype ;
            const GrB_Type ztype = op->ztype ;

            const GB_Type_code xcode = xtype->code ;
            const GB_Type_code ycode = ytype->code ;
            const GB_Type_code zcode = ztype->code ;

            const size_t xsize = xtype->size ;
            const size_t ysize = ytype->size ;
            const size_t zsize = ztype->size ;

            // x = (xtype) Ax [0]
            GB_void x [GB_VLA(xsize)] ;
            GB_cast_scalar (x, xcode, A->x, acode, asize) ;

            // y = (ytype) Bx [0]
            GB_void y [GB_VLA(ysize)] ;
            GB_cast_scalar (y, ycode, B->x, bcode, bsize) ;

            // z = op (x,y)
            GB_void z [GB_VLA(zsize)] ;
            fadd (z, x, y) ;

            // c = (ctype) z
            GB_cast_scalar (c, ccode, z, zcode, zsize) ;

            if (memcmp (c, a, csize) == 0)
            { 
                // the iso values of C and A match
                return (true) ;
            }
        }
    }

    // if the tests above fall through, C is not iso
    return (false) ;
}

