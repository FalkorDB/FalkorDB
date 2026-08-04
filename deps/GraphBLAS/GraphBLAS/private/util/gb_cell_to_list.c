//------------------------------------------------------------------------------
// gb_cell_to_list: convert cell array to index list I or colon expression
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Get a list of indices from a built-in MATLAB/Octave cell array.

// I is a cell array.  I contains 0, 1, 2, or 3 items:
//
//      0:  { }     This is the built-in ':', like C(:,J).
//      1:  { list }  A 1D list of row indices, like C(I,J).
//      2:  { start,fini }  start and fini are scalars.
//                  This defines I = start:1:fini in colon notation.
//      3:  { start,inc,fini } start, inc, and fini are scalars.
//                  This defines I = start:inc:fini in colon notation.
//
// If the cell contains 2 or 3 items, I is returned as an int64_t GrB_Vector of
// length 3, and the descriptor must use GxB_IS_STRIDE for the call to
// GrB_assign, GxB_subassign, or GrB_extract.

#undef  FREE_WORK
#define FREE_WORK                       \
    GrB_Vector_free (&Start_to_free) ;  \
    GrB_Vector_free (&Fini_to_free) ;   \
    GrB_Vector_free (&Inc_to_free) ;

#undef  FREE_ALL
#define FREE_ALL                        \
    FREE_WORK ;                         \
    GrB_Vector_free (&I_to_free) ;

GrB_Info gb_cell_to_list
(
    // output
    GrB_Vector *I_handle,
    GrB_Vector *I_to_free_handle,
    uint64_t *nI,               // # of items in the list
    int64_t *I_max,             // largest item in the list (NULL if not needed)
    // input
    struct gb_matrix_struct Cell_Matrix [3],    // contents of the Cell
    const int len,              // # of items in Cell_Matrix
    const int base_offset,      // 1 or 0
    const uint64_t n,           // dimension of the matrix
    const int arena,
    char err [ERRLEN]
)
{

    //--------------------------------------------------------------------------
    // parse the lists in the cell array
    //--------------------------------------------------------------------------

    GrB_Vector I = NULL, I_to_free = NULL, Start = NULL, Inc = NULL,
        Fini = NULL, Start_to_free = NULL, Fini_to_free = NULL,
        Inc_to_free = NULL ;
    bool Start_is_scalar, Fini_is_scalar, Inc_is_scalar ;

    (*I_handle) = NULL ;
    (*I_to_free_handle) = NULL ;

    if (len == 0)
    { 

        //----------------------------------------------------------------------
        // I = { }, a NULL vector I, representing I = 0:n-1 = GrB_ALL
        //----------------------------------------------------------------------

        (*nI) = n ;
        if (I_max != NULL)
        { 
            (*I_max) = n-1 ;
        }
        return (GrB_SUCCESS) ;

    }
    else if (len == 1)
    { 

        //----------------------------------------------------------------------
        // I = { list }
        //----------------------------------------------------------------------

        OK (gb_matrix_to_list (&I, &I_to_free, &(Cell_Matrix [0]),
            base_offset, arena, err)) ;

        if (I_max != NULL)
        { 
            // I_max = max (list)
            OK (GrB_Vector_reduce_INT64 (I_max, NULL, GrB_MAX_MONOID_INT64, I,
                NULL)) ;
        }
        OK (GrB_Vector_size (nI, I)) ;

    }
    else // if (len == 2 || len == 3)
    { 

        //----------------------------------------------------------------------
        // I = { start, fini } or I = { start, inc, fini }
        //----------------------------------------------------------------------

        // Start = Cell {0}, Fini = Cell {1}, and Inc = Cell {2} if present
        int64_t ibegin = 0, iinc = 1, iend = 0 ;
        if (len == 2)
        { 
            OK (gb_matrix_to_list (&Start, &Start_to_free, &(Cell_Matrix [0]),
                0, arena, err)) ;
            OK (gb_matrix_to_list (&Fini , &Fini_to_free , &(Cell_Matrix [1]),
                0, arena, err)) ;
            OK (gb_is_scalar (&Start_is_scalar, (GrB_Matrix) Start, err)) ;
            OK (gb_is_scalar (&Fini_is_scalar, (GrB_Matrix) Fini, err)) ;
            CHECK_ERROR (!Start_is_scalar || !Fini_is_scalar,
                "cell entries must be scalars for start:fini") ;
        }
        else // if (len == 3)
        { 
            OK (gb_matrix_to_list (&Start, &Start_to_free, &(Cell_Matrix [0]),
                0, arena, err)) ;
            OK (gb_matrix_to_list (&Inc  , &Inc_to_free  , &(Cell_Matrix [1]),
                0, arena, err)) ;
            OK (gb_matrix_to_list (&Fini , &Fini_to_free , &(Cell_Matrix [2]),
                0, arena, err)) ;
            OK (gb_is_scalar (&Start_is_scalar, (GrB_Matrix) Start, err)) ;
            OK (gb_is_scalar (&Inc_is_scalar, (GrB_Matrix) Inc, err)) ;
            OK (gb_is_scalar (&Fini_is_scalar, (GrB_Matrix) Fini, err)) ;
            CHECK_ERROR (!Start_is_scalar || !Fini_is_scalar || !Inc_is_scalar,
                "cell entries must be scalars for start:inc:fini") ;
        }

        // get ibegin, iend, and iinc
        OK (GrB_Vector_extractElement_INT64 (&ibegin, Start, 0)) ;
        OK (GrB_Vector_extractElement_INT64 (&iend, Fini, 0)) ;
        if (len == 3)
        { 
            OK (GrB_Vector_extractElement_INT64 (&iinc, Inc, 0)) ;
        }

        // handle the base_offset
        if (base_offset == 1)
        { 
            ibegin-- ;
            iend-- ;
        }

        // I = [ibegin, iend, iinc], to be freed by the caller
        OK (GxB_Vector_new_arena (&I, GrB_INT64, 3, arena, arena)) ;
        OK (GrB_Vector_setElement_INT64 (I, ibegin, GxB_BEGIN)) ;
        OK (GrB_Vector_setElement_INT64 (I, iend  , GxB_END)) ;
        OK (GrB_Vector_setElement_INT64 (I, iinc  , GxB_INC)) ;
        I_to_free = I ;

        //----------------------------------------------------------------------
        // determine the properties of ibegin:iinc:iend
        //----------------------------------------------------------------------

        int64_t imax = -1 ;
        (*nI) = 0 ;
        if (iinc < 0)
        { 
            if (ibegin >= iend)
            { 
                // the list is non-empty, for example, 7:-2:4 = [7 5]
                (*nI) = ((ibegin - iend) / (-iinc)) + 1 ;
                imax = ibegin ;
            }
        }
        else if (iinc > 0)
        { 
            if (ibegin <= iend)
            { 
                // the list is non-empty, for example, 4:2:9 = [4 6 8]
                // nI = length of the expanded list,
                // which is 3 for the list 4:2:9.
                (*nI) = ((iend - ibegin) / iinc) + 1 ;
                // imax is 8 for the list 4:2:9
                imax = ibegin + ((*nI)-1) * iinc ;
            }
        }
        if (I_max != NULL)
        { 
            (*I_max) = imax ;
        }
    }

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    FREE_WORK ;
    (*I_handle) = I ;
    (*I_to_free_handle) = I_to_free ;
    return (GrB_SUCCESS) ;
}

#undef  FREE_WORK
#define FREE_WORK
#undef  FREE_ALL
#define FREE_ALL

