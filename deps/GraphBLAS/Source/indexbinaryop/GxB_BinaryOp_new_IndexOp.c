//------------------------------------------------------------------------------
// GxB_BinaryOp_new_IndexOp: create a new user-defined binary op
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"

// GxB_BinaryOp_new_IndexOp: create a new binary op from an index binary op
GrB_Info GxB_BinaryOp_new_IndexOp
(
    GrB_BinaryOp *binop_handle,     // handle of binary op to create
    GxB_IndexBinaryOp idxbinop,     // based on this index binary op
    GrB_Scalar theta                // theta value to bind to the new binary op
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;
    GB_CHECK_INIT ;
    GB_RETURN_IF_NULL (binop_handle) ;
    (*binop_handle) = NULL ;
    GB_RETURN_IF_NULL_OR_FAULTY (idxbinop) ;
    GB_RETURN_IF_NULL_OR_INVALID (theta) ;

    if (!GB_Type_compatible (idxbinop->theta_type, theta->type))
    { 
        return (GrB_DOMAIN_MISMATCH) ;
    }

    //--------------------------------------------------------------------------
    // allocate the binary op
    //--------------------------------------------------------------------------

    size_t header_size ;
    GrB_BinaryOp
        binop = GB_CALLOC_MEMORY (1, sizeof (struct GB_BinaryOp_opaque),
            &header_size) ;
    if (binop == NULL)
    { 
        // out of memory
        return (GrB_OUT_OF_MEMORY) ;
    }
    binop->header_size = header_size ;

    //--------------------------------------------------------------------------
    // create the binary op
    //--------------------------------------------------------------------------

    // copy the index binary op contents into the binary op
    memcpy (binop, idxbinop, sizeof (struct GB_BinaryOp_opaque)) ;

    // remove the components owned by the index binary op
    binop->user_name = NULL ;
    binop->user_name_size = 0 ;
    binop->defn = NULL ;
    binop->defn_size = 0 ;

    bool jitable = (idxbinop->hash != UINT64_MAX) ;

    info = GB_op_name_and_defn (
        // output:
        binop->name, &(binop->name_len), &(binop->hash),
        &(binop->defn), &(binop->defn_size),
        // input:
        idxbinop->name, idxbinop->defn, true, jitable) ;
    if (info != GrB_SUCCESS)
    { 
        // out of memory
        GB_FREE_MEMORY (&binop, header_size) ;
        return (info) ;
    }

    //--------------------------------------------------------------------------
    // copy theta into the new binary op
    //--------------------------------------------------------------------------

    binop->theta = GB_MALLOC_MEMORY (1, binop->theta_type->size,
        &(binop->theta_size)) ;
    if (binop->theta == NULL)
    { 
        // out of memory
        GB_Op_free ((GB_Operator *) (&binop)) ;
        return (GrB_OUT_OF_MEMORY) ;
    }

    GB_cast_scalar (binop->theta, binop->theta_type->code,
        theta->x, theta->type->code, theta->type->size) ;

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    ASSERT_BINARYOP_OK (binop, "new user-defined binary op (based on idxbinop)",
        GB0) ;
    (*binop_handle) = binop ;
    return (GrB_SUCCESS) ;
}

