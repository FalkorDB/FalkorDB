//------------------------------------------------------------------------------
// GB_Monoid_new: create a GrB_Monoid
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Create a monoid with an operator, (optionally) an identity value, and
// (optionally) a terminal value.  If using a built-in operator, a duplicate
// boolean operator is first replaced with its unique equivalent.  If the
// operator is built-in and corresponds to a known monoid, then the identity
// value and terminal value provided on input are silently ignored, and the
// known values are used instead.  This is to allow the use of the hard-coded
// functions for built-in monoids.

// User-defined monoids may have a NULL terminal value, which denotes that the
// monoid does not have a terminal value.

#include "GB.h"
#include "binaryop/GB_binop.h"
#include "monoid/GB_Monoid_new.h"
#include "jitifyer/GB_jitifyer.h"

GrB_Info GB_Monoid_new          // create a monoid
(
    GrB_Monoid *monoid,         // handle of monoid to create
    GrB_BinaryOp op,            // binary operator of the monoid
    const void *identity,       // identity value, if any
    const void *terminal,       // terminal value, if any (may be NULL)
    GB_Type_code idcode,        // identity and terminal type code
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_RETURN_IF_NULL (monoid) ;
    (*monoid) = NULL ;
    GB_RETURN_IF_NULL (identity) ;
    GB_RETURN_IF_NULL_OR_FAULTY (op) ;

    ASSERT_BINARYOP_OK (op, "op for monoid", GB0) ;
    ASSERT (idcode <= GB_UDT_code) ;

    //--------------------------------------------------------------------------
    // rename built-in binary operators
    //--------------------------------------------------------------------------

    // If the user requests the creation of a monoid based on a duplicate
    // built-in binary operator, the unique boolean operator is used instead.
    // See also GB_boolean_rename, which does this for opcodes, not operators.

    op = GB_boolean_rename_op (op) ;
    ASSERT_BINARYOP_OK (op, "revised op", GB0) ;

    //--------------------------------------------------------------------------
    // continue checking inputs
    //--------------------------------------------------------------------------

    // check operator types; all must be identical, and the operator cannot
    // be positional
    if (op->xtype != op->ztype || op->ytype != op->ztype ||
        GB_OP_IS_POSITIONAL (op))
    { 
        return (GrB_DOMAIN_MISMATCH) ;
    }

    // The idcode must match the monoid->op->ztype->code for built-in types,
    // and this can be rigourously checked.  For all user-defined types,
    // identity is a mere void * pointer, and its actual type cannot be
    // compared with the input op->ztype parameter.  Only the type code,
    // GB_UDT_code, can be checked to see if it matches.  In
    // that case, all that is known is that identity is a void * pointer that
    // points to something; it must a scalar of the proper user-defined type.

    GB_Type_code zcode = op->ztype->code ;
    if (idcode != zcode)
    { 
        return (GrB_DOMAIN_MISMATCH) ;
    }

    //--------------------------------------------------------------------------
    // create the monoid
    //--------------------------------------------------------------------------

    // allocate the monoid
    size_t header_size ;
    (*monoid) = GB_MALLOC_MEMORY (1, sizeof (struct GB_Monoid_opaque),
        &header_size) ;
    if (*monoid == NULL)
    { 
        // out of memory
        return (GrB_OUT_OF_MEMORY) ;
    }

    // initialize the monoid
    GrB_Monoid mon = *monoid ;
    mon->magic = GB_MAGIC ;
    mon->header_size = header_size ;
    mon->user_name = NULL ;                 // user_name for GrB_get/GrB_set
    mon->user_name_size = 0 ;
    mon->op = op ;
    size_t zsize = op->ztype->size ;
    mon->identity = NULL ;                  // defined below (if present)
    mon->terminal = NULL ;                  // defined below (if present)
    mon->identity_size = 0 ;
    mon->terminal_size = 0 ;
    bool builtin = false ;  // set true below if using a builtin binary op
    mon->hash = 0 ;         // builtin monoids have a hash value of 0

    //--------------------------------------------------------------------------
    // allocation of identity and terminal values
    //--------------------------------------------------------------------------

    // allocate the identity value
    #define GB_ALLOC_IDENTITY                                               \
    {                                                                       \
        mon->identity = GB_MALLOC_MEMORY (1, zsize, &(mon->identity_size)) ; \
        if (mon->identity == NULL)                                          \
        {                                                                   \
            /* out of memory */                                             \
            GB_FREE_MEMORY (&(mon->terminal), mon->terminal_size) ;                \
            GB_FREE_MEMORY (monoid, header_size) ;                                 \
            return (GrB_OUT_OF_MEMORY) ;                                    \
        }                                                                   \
    }

    // allocate the terminal value
    #define GB_ALLOC_TERMINAL                                               \
    {                                                                       \
        mon->terminal = GB_MALLOC_MEMORY (1, zsize, &(mon->terminal_size)) ; \
        if (mon->terminal == NULL)                                          \
        {                                                                   \
            /* out of memory */                                             \
            GB_FREE_MEMORY (&(mon->identity), mon->identity_size) ;                \
            GB_FREE_MEMORY (monoid, header_size) ;                                 \
            return (GrB_OUT_OF_MEMORY) ;                                    \
        }                                                                   \
    }

    //--------------------------------------------------------------------------
    // create a user-defined monoid based on a known built-in monoid
    //--------------------------------------------------------------------------

    // If the user requests the creation of a monoid based on a built-in
    // operator that corresponds to known built-in monoid, the identity and
    // terminal values provided by the user to GrB_Monoid_new are ignored,
    // since these are all handled by the same hard-coded functions as the
    // built-in monoids based on the same operator.

    // All of the monoids created in the switch statement below have non-NULL
    // identity values; some have terminal values and some do not.

    // create a monoid with both an identity and a terminal value,
    // based on a built-in operator that is a known monoid
    #define GB_IT(ztype,identity_value,terminal_value)                      \
    {                                                                       \
        GB_ALLOC_TERMINAL ;                                                 \
        GB_ALLOC_IDENTITY ;                                                 \
        ztype *identity = (ztype *) mon->identity ;                         \
        ztype *terminal = (ztype *) mon->terminal ;                         \
        (*identity) = identity_value ;                                      \
        (*terminal) = terminal_value ;                                      \
        builtin = true ;                                                    \
    }                                                                       \
    break ;

    // create a monoid with just an identity but no terminal value,
    // based on a built-in operator that is a known monoid
    #define GB_IN(ztype,identity_value)                                     \
    {                                                                       \
        GB_ALLOC_IDENTITY ;                                                 \
        ztype *identity = (ztype *) mon->identity ;                         \
        (*identity) = identity_value ;                                      \
        builtin = true ;                                                    \
    }                                                                       \
    break ;

    switch (op->opcode)
    {
        case GB_MIN_binop_code : 

            // MIN monoid:  identity is +inf, terminal is -inf
            // note there is no MIN monoid for complex types
            switch (zcode)
            {
                case GB_INT8_code   : GB_IT (int8_t  , INT8_MAX  , INT8_MIN  )
                case GB_INT16_code  : GB_IT (int16_t , INT16_MAX , INT16_MIN )
                case GB_INT32_code  : GB_IT (int32_t , INT32_MAX , INT32_MIN )
                case GB_INT64_code  : GB_IT (int64_t , INT64_MAX , INT64_MIN )
                case GB_UINT8_code  : GB_IT (uint8_t , UINT8_MAX , 0         )
                case GB_UINT16_code : GB_IT (uint16_t, UINT16_MAX, 0         )
                case GB_UINT32_code : GB_IT (uint32_t, UINT32_MAX, 0         )
                case GB_UINT64_code : GB_IT (uint64_t, UINT64_MAX, 0         )
                // floating-point min is not terminal:
                case GB_FP32_code   : GB_IN (float   , INFINITY )
                case GB_FP64_code   : GB_IN (double  , ((double) INFINITY))
                default: ;
            }
            break ;

        case GB_MAX_binop_code : 

            // MAX monoid:  identity is -inf, terminal is +inf
            // note there is no MAX monoid for complex types
            switch (zcode)
            {
                case GB_INT8_code   : GB_IT (int8_t  , INT8_MIN  , INT8_MAX  )
                case GB_INT16_code  : GB_IT (int16_t , INT16_MIN , INT16_MAX )
                case GB_INT32_code  : GB_IT (int32_t , INT32_MIN , INT32_MAX )
                case GB_INT64_code  : GB_IT (int64_t , INT64_MIN , INT64_MAX )
                case GB_UINT8_code  : GB_IT (uint8_t , 0         , UINT8_MAX )
                case GB_UINT16_code : GB_IT (uint16_t, 0         , UINT16_MAX)
                case GB_UINT32_code : GB_IT (uint32_t, 0         , UINT32_MAX)
                case GB_UINT64_code : GB_IT (uint64_t, 0         , UINT64_MAX)
                // floating-point max is not terminal:
                case GB_FP32_code   : GB_IN (float   , -INFINITY)
                case GB_FP64_code   : GB_IN (double  , ((double) -INFINITY)) ;
                default: ;
            }
            break ;

        case GB_PLUS_binop_code : 

            // PLUS monoid:  identity is zero, no terminal value
            switch (zcode)
            {
                case GB_INT8_code   : GB_IN (int8_t  , 0 )
                case GB_INT16_code  : GB_IN (int16_t , 0 )
                case GB_INT32_code  : GB_IN (int32_t , 0 )
                case GB_INT64_code  : GB_IN (int64_t , 0 )
                case GB_UINT8_code  : GB_IN (uint8_t , 0 )
                case GB_UINT16_code : GB_IN (uint16_t, 0 )
                case GB_UINT32_code : GB_IN (uint32_t, 0 )
                case GB_UINT64_code : GB_IN (uint64_t, 0 )
                case GB_FP32_code   : GB_IN (float   , 0 )
                case GB_FP64_code   : GB_IN (double  , 0 )
                case GB_FC32_code   : GB_IN (GxB_FC32_t, GxB_CMPLXF(0,0) )
                case GB_FC64_code   : GB_IN (GxB_FC64_t, GxB_CMPLX(0,0) )
                default: ;
            }
            break ;

        case GB_TIMES_binop_code : 

            // TIMES monoid:  identity is 1, terminal is 0 for integers
            switch (zcode)
            {
                case GB_INT8_code   : GB_IT (int8_t  , 1, 0)
                case GB_INT16_code  : GB_IT (int16_t , 1, 0)
                case GB_INT32_code  : GB_IT (int32_t , 1, 0)
                case GB_INT64_code  : GB_IT (int64_t , 1, 0)
                case GB_UINT8_code  : GB_IT (uint8_t , 1, 0)
                case GB_UINT16_code : GB_IT (uint16_t, 1, 0)
                case GB_UINT32_code : GB_IT (uint32_t, 1, 0)
                case GB_UINT64_code : GB_IT (uint64_t, 1, 0)
                // floating-point times is not terminal:
                case GB_FP32_code   : GB_IN (float   , 1 )
                case GB_FP64_code   : GB_IN (double  , 1 )
                case GB_FC32_code   : GB_IN (GxB_FC32_t, GxB_CMPLXF(1,0) )
                case GB_FC64_code   : GB_IN (GxB_FC64_t, GxB_CMPLX(1,0) )
                default: ;
            }
            break ;

        case GB_ANY_binop_code : 

            // ANY monoid:  identity is anything, terminal value is anything
            switch (zcode)
            {
                case GB_BOOL_code   : GB_IT (bool    , 0, 0 )
                case GB_INT8_code   : GB_IT (int8_t  , 0, 0 )
                case GB_INT16_code  : GB_IT (int16_t , 0, 0 )
                case GB_INT32_code  : GB_IT (int32_t , 0, 0 )
                case GB_INT64_code  : GB_IT (int64_t , 0, 0 )
                case GB_UINT8_code  : GB_IT (uint8_t , 0, 0 )
                case GB_UINT16_code : GB_IT (uint16_t, 0, 0 )
                case GB_UINT32_code : GB_IT (uint32_t, 0, 0 )
                case GB_UINT64_code : GB_IT (uint64_t, 0, 0 )
                case GB_FP32_code   : GB_IT (float   , 0, 0 )
                case GB_FP64_code   : GB_IT (double  , 0, 0 )
                case GB_FC32_code   : 
                    GB_IT (GxB_FC32_t, GxB_CMPLXF(0,0), GxB_CMPLXF(0,0))
                case GB_FC64_code   : 
                    GB_IT (GxB_FC64_t, GxB_CMPLX(0,0), GxB_CMPLX(0,0))
                default: ;
            }
            break ;

        case GB_LOR_binop_code : 

            // boolean OR monoid:  identity is false, terminal is true
            if (zcode == GB_BOOL_code) GB_IT (bool, false, true)

        case GB_LAND_binop_code : 

            // boolean AND monoid:  identity is true, terminal is false
            if (zcode == GB_BOOL_code) GB_IT (bool, true, false)

        case GB_LXOR_binop_code : 

            // boolean XOR monoid:  identity is false, no terminal value
            if (zcode == GB_BOOL_code) GB_IN (bool, false)

        case GB_EQ_binop_code : 

            // boolean EQ monoid:  identity is true, no terminal value
            if (zcode == GB_BOOL_code) GB_IN (bool, true)

        case GB_BOR_binop_code      : 

            // BOR monoids (bitwise or):
            switch (zcode)
            {
                case GB_UINT8_code  : GB_IT (uint8_t , 0, 0xFF               )
                case GB_UINT16_code : GB_IT (uint16_t, 0, 0xFFFF             )
                case GB_UINT32_code : GB_IT (uint32_t, 0, 0xFFFFFFFF         ) 
                case GB_UINT64_code : GB_IT (uint64_t, 0, 0xFFFFFFFFFFFFFFFF ) 
                default: ;
            }
            break ;

        case GB_BAND_binop_code     : 

            // BAND monoids (bitwise and):
            switch (zcode)
            {
                case GB_UINT8_code  : GB_IT (uint8_t , 0xFF              , 0 )
                case GB_UINT16_code : GB_IT (uint16_t, 0xFFFF            , 0 )
                case GB_UINT32_code : GB_IT (uint32_t, 0xFFFFFFFF        , 0 ) 
                case GB_UINT64_code : GB_IT (uint64_t, 0xFFFFFFFFFFFFFFFF, 0 ) 
                default: ;
            }
            break ;

        case GB_BXOR_binop_code     : 

            // BXOR monoids (bitwise xor): not terminal
            switch (zcode)
            {
                case GB_UINT8_code  : GB_IN (uint8_t , 0 )
                case GB_UINT16_code : GB_IN (uint16_t, 0 )
                case GB_UINT32_code : GB_IN (uint32_t, 0 ) 
                case GB_UINT64_code : GB_IN (uint64_t, 0 ) 
                default: ;
            }
            break ;

        case GB_BXNOR_binop_code    : 

            // BXNOR monoids (bitwise xnor): not terminal
            switch (zcode)
            {
                case GB_UINT8_code  : GB_IN (uint8_t , 0xFF               )
                case GB_UINT16_code : GB_IN (uint16_t, 0xFFFF             )
                case GB_UINT32_code : GB_IN (uint32_t, 0xFFFFFFFF         )
                case GB_UINT64_code : GB_IN (uint64_t, 0xFFFFFFFFFFFFFFFF )
                default: ;
            }
            break ;

        default : ;

            // monoid identity and terminal value defined below
    }

    //--------------------------------------------------------------------------
    // user-defined operators or unknown monoids
    //--------------------------------------------------------------------------

    // The monoid is based on a user-defined operator, or on a built-in
    // operator that is not a known monoid.  Use the identity and terminal
    // values provided on input.

    if (!builtin)
    {
        // create the monoid identity value
        GB_ALLOC_IDENTITY ;
        memcpy (mon->identity, identity, zsize) ;
        uint64_t hashes [3] ;
        hashes [0] = mon->op->hash ;
        hashes [1] = GB_jitifyer_hash (identity, zsize, true) ;
        hashes [2] = 0 ;
        if (terminal != NULL)
        { 
            // create the monoid terminal value
            GB_ALLOC_TERMINAL ;
            memcpy (mon->terminal, terminal, zsize) ;
            hashes [2] = GB_jitifyer_hash (terminal, zsize, true) ;
        }
        // create the monoid hash: based on binary op hash, id, and terminal.
        // the monoid is JIT'able only if the op is JIT'able
        mon->hash = GB_jitifyer_hash (hashes, 3*sizeof (uint64_t),
            mon->op->hash != UINT64_MAX) ;
    }

    ASSERT_MONOID_OK (mon, "new monoid", GB0) ;
    return (GrB_SUCCESS) ;
}

