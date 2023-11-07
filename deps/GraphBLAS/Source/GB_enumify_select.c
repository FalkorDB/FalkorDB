//------------------------------------------------------------------------------
// GB_enumify_select: enumerate a GrB_select problem
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "GB_stringify.h"

// Currently, the mask M and the accum are not present, and C and A have the
// same type, but these conditions may change in the future.

void GB_enumify_select      // enumerate a GrB_selectproblem
(
    // output:
    uint64_t *scode,        // unique encoding of the entire operation
    // input:
    bool C_iso,
    bool in_place_A,
    // operator:
    GrB_IndexUnaryOp op,    // the index unary operator to enumify
    bool flipij,            // if true, flip i and j
    // A matrix:
    GrB_Matrix A
)
{

    //--------------------------------------------------------------------------
    // get the types of A, X, Y, and Z
    //--------------------------------------------------------------------------

    GrB_Type atype = A->type ;
    GB_Opcode opcode = op->opcode ;
    GB_Type_code zcode = op->ztype->code ;
    GB_Type_code xcode = (op->xtype == NULL) ? 0 : op->xtype->code ;
    GB_Type_code ycode = op->ytype->code ;

    //--------------------------------------------------------------------------
    // enumify the idxunop operator
    //--------------------------------------------------------------------------

    bool depends_on_x, depends_on_i, depends_on_j, depends_on_y ;
    int idxop_ecode ;
    GB_enumify_unop (&idxop_ecode, &depends_on_x, &depends_on_i,
        &depends_on_j, &depends_on_y, flipij, opcode, xcode) ;

    ASSERT (idxop_ecode >= 231 && idxop_ecode <= 254) ;

    if (!depends_on_x)
    { 
        // VALUE* ops and user-defined index unary ops depend on x.  The
        // positional ops (tril, triu, row*, col*, diag*) do not.
        xcode = 0 ;
    }

    if (!depends_on_y)
    { 
        // All index unary ops depend on y except for NONZOMBIE
        ycode = 0 ;
    }

    int i_dep = (depends_on_i) ? 1 : 0 ;
    int j_dep = (depends_on_j) ? 1 : 0 ;

    //--------------------------------------------------------------------------
    // enumify the types
    //--------------------------------------------------------------------------

    int acode = atype->code ;               // 1 to 14
    int ccode = acode ;                     // this may change in the future
    int A_iso_code = (A->iso) ? 1 : 0 ;
    int C_iso_code = (C_iso) ? 1 : 0 ;

    //--------------------------------------------------------------------------
    // enumify the sparsity structure of A
    //--------------------------------------------------------------------------

    int A_sparsity = GB_sparsity (A) ;
    int C_sparsity ;

    if (opcode == GB_DIAG_idxunop_code)
    { 
        C_sparsity = (A_sparsity == GxB_FULL) ? GxB_SPARSE : A_sparsity ;
    }
    else
    { 
        C_sparsity = (A_sparsity == GxB_FULL) ? GxB_BITMAP : A_sparsity ;
    }

    int asparsity, csparsity ;
    GB_enumify_sparsity (&csparsity, C_sparsity) ;
    GB_enumify_sparsity (&asparsity, A_sparsity) ;

    int inplace = (in_place_A) ? 1 : 0 ;

    //--------------------------------------------------------------------------
    // construct the select scode
    //--------------------------------------------------------------------------

    // total scode bits:  38 (10 hex digits)

    (*scode) =
                                               // range        bits
                // iso of A aand C (2 bits)
                GB_LSHIFT (C_iso_code , 37) |  // 0 or 1       1
                GB_LSHIFT (A_iso_code , 36) |  // 0 or 1       1

                // inplace, i/j dependency and flipij (1 hex digit)
                GB_LSHIFT (inplace    , 35) |  // 0 or 1       1
                GB_LSHIFT (i_dep      , 34) |  // 0 or 1       1
                GB_LSHIFT (j_dep      , 33) |  // 0 or 1       1
                GB_LSHIFT (flipij     , 32) |  // 0 or 1       1

                // op, z = f(x,i,j,y) (5 hex digits)
                GB_LSHIFT (idxop_ecode, 24) |  // 231 to 254   8
                GB_LSHIFT (zcode      , 20) |  // 0 to 14      4
                GB_LSHIFT (xcode      , 16) |  // 0 to 14      4
                GB_LSHIFT (ycode      , 12) |  // 0 to 14      4

                // types of C and A (2 hex digits)
                GB_LSHIFT (ccode      ,  8) |  // 0 to 15      4
                GB_LSHIFT (acode      ,  4) |  // 0 to 15      4

                // sparsity structures of C and A (1 hex digit)
                GB_LSHIFT (csparsity  ,  2) |  // 0 to 3       2
                GB_LSHIFT (asparsity  ,  0) ;  // 0 to 3       2
}

