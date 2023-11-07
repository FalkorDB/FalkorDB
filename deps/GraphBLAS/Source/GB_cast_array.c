//------------------------------------------------------------------------------
// GB_cast_array: typecast an array
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// JIT: done.

// Casts an input array A->x to an output array Cx with a different type.  The
// two types are always different, so this does not need to handle user-defined
// types.  The iso case is not handled; A->x and Cx must be the same size and no
// iso expansion is done.

#include "GB.h"
#include "GB_apply.h"
#include "GB_stringify.h"
#ifndef GBCOMPACT
#include "GB_unop__include.h"
#endif

GrB_Info GB_cast_array              // typecast an array
(
    GB_void *Cx,                // output array
    const GB_Type_code code1,   // type code for Cx
    GrB_Matrix A,
    const int A_nthreads        // number of threads to use
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT_MATRIX_OK (A, "A for cast_array", GB0) ;
    const GB_void *restrict Ax = A->x ;
    const int8_t *restrict Ab = A->b ;
    const int64_t anz = GB_nnz_held (A) ;
    const GB_Type_code code2 = A->type->code ;

    if (anz == 0 || Cx == Ax)
    { 
        // no work to do
        return (GrB_SUCCESS) ;
    }

    ASSERT (Cx != NULL) ;
    ASSERT (Ax != NULL) ;
    ASSERT (anz > 0) ;
    ASSERT (GB_code_compatible (code1, code2)) ;
    ASSERT (code1 != code2) ;
    ASSERT (code1 != GB_UDT_code) ;

    //--------------------------------------------------------------------------
    // via the factory kernel
    //--------------------------------------------------------------------------

    GrB_Info info = GrB_NO_VALUE ;

    #ifndef GBCOMPACT
    GB_IF_FACTORY_KERNELS_ENABLED
    { 

        //----------------------------------------------------------------------
        // define the worker for the switch factory
        //----------------------------------------------------------------------

        #define GB_unop_apply(zname,xname)                                  \
            GB (_unop_apply__identity ## zname ## xname)

        #define GB_WORKER(ignore1,zname,ztype,xname,xtype)                  \
        {                                                                   \
            info = GB_unop_apply (zname,xname) (Cx, Ax, Ab, anz,            \
                A_nthreads) ;                                               \
        }                                                                   \
        break ;

        //----------------------------------------------------------------------
        // launch the switch factory
        //----------------------------------------------------------------------

        #define GB_EXCLUDE_SAME_TYPES
        #include "GB_twotype_factory.c"
    }
    #endif

    //--------------------------------------------------------------------------
    // via the JIT or PreJIT kernel
    //--------------------------------------------------------------------------

    if (info == GrB_NO_VALUE)
    { 
        GrB_Type ctype = GB_code_type (code1, NULL) ;
        GB_Operator op = GB_unop_identity (ctype, NULL) ;
        ASSERT_OP_OK (op, "id op for cast_array", GB0) ;
        info = GB_apply_unop_jit (Cx, ctype, op, false, A, NULL, NULL, 0,
            A_nthreads) ;
    }

    //--------------------------------------------------------------------------
    // via the generic kernel
    //--------------------------------------------------------------------------

    if (info == GrB_NO_VALUE)
    { 
        GB_BURBLE_N (anz, "(generic cast array) ") ;
        int64_t csize = GB_code_size (code1, 0) ;
        int64_t asize = GB_code_size (code2, 0) ;
        GB_cast_function cast_A_to_C = GB_cast_factory (code1, code2) ;
        #define GB_APPLY_OP(p) \
            cast_A_to_C (Cx +(p*csize), Ax +(p*asize), asize)
        #include "GB_apply_unop_ip.c"
        info = GrB_SUCCESS ;
    }

    return (info) ;
}

