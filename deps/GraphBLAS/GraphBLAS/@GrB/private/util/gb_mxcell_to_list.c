//------------------------------------------------------------------------------
// gb_mxcell_to_list: convert cell array to index list I or colon expression
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
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

#include "gb_interface.h"

GrB_Vector gb_mxcell_to_list    // return index list I
(
    // input
    const mxArray *Cell,        // built-in MATLAB cell array
    const int base_offset,      // 1 or 0
    const uint64_t n,           // dimension of the matrix
    // output
    uint64_t *nI,               // # of items in the list
    int64_t *I_max              // largest item in the list (NULL if not needed)
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    CHECK_ERROR (Cell == NULL || !mxIsCell (Cell), "internal error 6") ;

    //--------------------------------------------------------------------------
    // get the contents of Cell
    //--------------------------------------------------------------------------

    int len = mxGetNumberOfElements (Cell) ;
    CHECK_ERROR (len > 3, "index must be a cell array of length 0 to 3") ;

    //--------------------------------------------------------------------------
    // parse the lists in the cell array
    //--------------------------------------------------------------------------

    GrB_Vector I = NULL ;

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
        return (NULL) ;

    }
    else if (len == 1)
    { 

        //----------------------------------------------------------------------
        // I = { list }
        //----------------------------------------------------------------------

        I = gb_mxarray_to_list (mxGetCell (Cell, 0), base_offset) ;
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
        GrB_Vector Start = NULL, Inc = NULL, Fini = NULL ;
        if (len == 2)
        { 
            mxArray *cell0 = mxGetCell (Cell, 0) ;
            mxArray *cell1 = mxGetCell (Cell, 1) ;
            CHECK_ERROR (!gb_mxarray_is_scalar (cell0)
                      || !gb_mxarray_is_scalar (cell1),
                "cell entries must be scalars for start:fini") ;
            Start = gb_mxarray_to_list (cell0, 0) ;
            Fini  = gb_mxarray_to_list (cell1, 0) ;
        }
        else // if (len == 3)
        { 
            mxArray *cell0 = mxGetCell (Cell, 0) ;
            mxArray *cell1 = mxGetCell (Cell, 1) ;
            mxArray *cell2 = mxGetCell (Cell, 2) ;
            CHECK_ERROR (!gb_mxarray_is_scalar (cell0)
                      || !gb_mxarray_is_scalar (cell1)
                      || !gb_mxarray_is_scalar (cell2),
                "cell entries must be scalars for start:inc:fini") ;
            Start = gb_mxarray_to_list (cell0, 0) ;
            Inc   = gb_mxarray_to_list (cell1, 0) ;
            Fini  = gb_mxarray_to_list (cell2, 0) ;
        }

        // get ibegin, iend, and iinc
        OK (GrB_Vector_extractElement_INT64 (&ibegin, Start, 0)) ;
        OK (GrB_Vector_extractElement_INT64 (&iend, Fini, 0)) ;
        if (len == 3)
        { 
            OK (GrB_Vector_extractElement_INT64 (&iinc, Inc, 0)) ;
        }

        GrB_Vector_free (&Start) ;
        GrB_Vector_free (&Fini) ;
        GrB_Vector_free (&Inc) ;

        // handle the base_offset
        if (base_offset == 1)
        {
            ibegin-- ;
            iend-- ;
        }

        // I = [ibegin, iend, iinc]
        OK (GrB_Vector_new (&I, GrB_INT64, 3)) ;
        OK (GrB_Vector_setElement_INT64 (I, ibegin, GxB_BEGIN)) ;
        OK (GrB_Vector_setElement_INT64 (I, iend  , GxB_END)) ;
        OK (GrB_Vector_setElement_INT64 (I, iinc  , GxB_INC)) ;

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
    // return result
    //--------------------------------------------------------------------------

    return (I) ;
}

