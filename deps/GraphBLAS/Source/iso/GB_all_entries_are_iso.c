//------------------------------------------------------------------------------
// GB_all_entries_are_iso: check if all entries in a matrix are identical
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Returns true if all entries in A are the same, and A can then be converted
// to iso if currently non-iso.  Returns false if A is bitmap, has any zombies,
// or has or pending tuples, since these are more costly to check.

// User-defined types of sizes 1, 2, 4, 8, and 16 bytes can be tested by using
// the built-in uint* types of those sizes.  Different sizes cannot be checked
// with a JIT since "a == b" is not defined in C11 if a and b are structs.
// Instead, memcmp (a, b, sizeof (type)) is used instead.

#include "GB.h"

bool GB_all_entries_are_iso // return true if A is iso, false otherwise
(
    const GrB_Matrix A      // matrix to test if all entries are the same
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    if (A == NULL || GB_nnz (A) == 0 || GB_nnz_held (A) == 0)
    { 
        // empty matrices cannot be iso
        return (false) ;
    }
    else if (A->iso)
    { 
        // nothing to do; A is already iso
        return (true) ;
    }
    else if (GB_PENDING (A) || GB_ZOMBIES (A) || GB_IS_BITMAP (A))
    { 
        // Non-iso matrices with pending work are assumed to be non-iso.
        // Bitmap matrices and matrices with zombies could be checked, but
        // finding the first entry is tedious so this is skipped.  Matrices
        // with pending work could be finished first, but this is costly so it
        // is skipped.
        return (false) ;
    }

    ASSERT (!GB_ZOMBIES (A)) ;
    ASSERT (GB_JUMBLED_OK (A)) ;
    ASSERT (!GB_PENDING (A)) ;
    ASSERT (!GB_IS_BITMAP (A)) ;
    ASSERT_MATRIX_OK (A, "A input for GB_all_entries_are_iso", GB0) ;

    //--------------------------------------------------------------------------
    // get A
    //--------------------------------------------------------------------------

    int64_t asize = A->type->size ;
    int64_t anz = GB_nnz_held (A) ;

    //--------------------------------------------------------------------------
    // determine the number of threads to use
    //--------------------------------------------------------------------------

    int nthreads = 0, ntasks = 0 ;
    int nthreads_max = GB_Context_nthreads_max ( ) ;
    double chunk = GB_Context_chunk ( ) ;
    nthreads = GB_nthreads (anz, chunk, nthreads_max) ;
    ntasks = (nthreads == 1) ? 1 : (64 * nthreads) ;
    ntasks = GB_IMIN (ntasks, anz) ;
    ntasks = GB_IMAX (ntasks, 1) ;

    //--------------------------------------------------------------------------
    // check if A is iso: built-in types and user-defined with sizes 1,2,4,8,16
    //--------------------------------------------------------------------------

    #define GB_GET_FIRST_VALUE(atype_t, a, Ax)                      \
        const atype_t a = Ax [0]
    #define GB_COMPARE_WITH_FIRST_VALUE(my_iso, a, Ax, p)           \
        my_iso = my_iso & (a == Ax [p])

    bool iso = true ;       // A is iso until proven otherwise

    switch (asize)
    {
        case GB_1BYTE : // uint8, int8, bool, or 1-byte user
            #define GB_A_TYPE uint8_t
            #include "iso/factory/GB_all_entries_are_iso_template.c"
            break ;

        case GB_2BYTE : // uint16, int16, or 2-byte user
            #define GB_A_TYPE uint16_t
            #include "iso/factory/GB_all_entries_are_iso_template.c"
            break ;

        case GB_4BYTE : // uint32, int32, float, or 4-byte user
            #define GB_A_TYPE uint32_t
            #include "iso/factory/GB_all_entries_are_iso_template.c"
            break ;

        case GB_8BYTE : // uint64, int64, double, float complex,
                        // or 8-byte user defined
            #define GB_A_TYPE uint64_t
            #include "iso/factory/GB_all_entries_are_iso_template.c"
            break ;

        case GB_16BYTE : // double complex or 16-byte user
            #define GB_A_TYPE uint64_t
            #undef  GB_GET_FIRST_VALUE
            #define GB_GET_FIRST_VALUE(atype_t, a, Ax)              \
                const atype_t a ## 0 = Ax [0] ;                     \
                const atype_t a ## 1 = Ax [1] ;
            #undef  GB_COMPARE_WITH_FIRST_VALUE
            #define GB_COMPARE_WITH_FIRST_VALUE(my_iso, a, Ax, p)   \
                my_iso = my_iso & (a ## 0 == Ax [2*(p)  ])          \
                                & (a ## 1 == Ax [2*(p)+1])
            #include "iso/factory/GB_all_entries_are_iso_template.c"
            break ;

        default : // with user-defined types of any size

            #define GB_A_TYPE GB_void
            #undef  GB_GET_FIRST_VALUE
            #define GB_GET_FIRST_VALUE(atype_t, a, Ax)              \
                GB_void a [GB_VLA(asize)] ;                         \
                memcpy (a, Ax, asize) ;
            #undef  GB_COMPARE_WITH_FIRST_VALUE
            #define GB_COMPARE_WITH_FIRST_VALUE(my_iso, a, Ax, p)   \
                my_iso = my_iso & (memcmp (a, Ax + (p)*asize, asize) == 0)
            #include "iso/factory/GB_all_entries_are_iso_template.c"
            break ;
    }

    return (iso) ;
}

