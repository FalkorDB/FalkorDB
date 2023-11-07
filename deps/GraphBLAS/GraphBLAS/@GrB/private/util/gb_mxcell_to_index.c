//------------------------------------------------------------------------------
// gb_mxcell_to_index: convert cell array to index list I or colon expression
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Get a list of indices from a built-in cell array.

// I is a cell array.  I contains 0, 1, 2, or 3 items:
//
//      0:  { }     This is the built-in ':', like C(:,J), refering to all m
//                  rows, if C is m-by-n.
//      1:  { list }  A 1D list of row indices, like C(I,J).
//      2:  { start,fini }  start and fini are scalars (either double, int64,
//                  or uint64).  This defines I = start:fini in colon
//                  notation.
//      3:  { start,inc,fini } start, inc, and fini are scalars (double, int64,
//                  or uint64).  This defines I = start:inc:fini in colon
//                  notation.

#include "gb_interface.h"

GrB_Index *gb_mxcell_to_index   // return index list I
(
    const mxArray *I_cell,      // built-in cell array
    base_enum_t base,           // I is one-based or zero-based
    const GrB_Index n,          // dimension of matrix being indexed
    bool *I_allocated,          // true if output array I is allocated
    GrB_Index *ni,              // length (I)
    int64_t *I_max              // max (I) is computed if I_max is not NULL.
                                // I_max is 0-based.
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    CHECK_ERROR (I_cell == NULL || !mxIsCell (I_cell), "internal error 6") ;

    //--------------------------------------------------------------------------
    // get the contents of I_cell
    //--------------------------------------------------------------------------

    int len = mxGetNumberOfElements (I_cell) ;
    CHECK_ERROR (len > 3, "index must be a cell array of length 0 to 3") ;

    bool Item_allocated [3] = { false, false, false } ;
    int64_t Item_len [3] = { 0, 0, 0 } ;
    int64_t Item_max [3] = { -1, -1, -1 } ;
    GrB_Index *Item [3] = { NULL, NULL, NULL } ;

    for (int k = 0 ; k < len ; k++)
    { 
        // convert I_cell {k} content to an integer list
        Item [k] = (GrB_Index *) gb_mxarray_to_list (mxGetCell (I_cell, k),
            base, &Item_allocated [k], &Item_len [k], &Item_max [k]) ;
    }

    //--------------------------------------------------------------------------
    // parse the lists in the cell array
    //--------------------------------------------------------------------------

    GrB_Index *I ;

    if (len == 0)
    { 

        //----------------------------------------------------------------------
        // I = { }
        //----------------------------------------------------------------------

        (*ni) = n ;
        (*I_allocated) = false ;
        I = (GrB_Index *) GrB_ALL ;
        if (I_max != NULL)
        { 
            // I_max is the last index in the matrix, based on its dimension.
            (*I_max) = ((int64_t) n) - 1 ;
        }

    }
    else if (len == 1)
    { 

        //----------------------------------------------------------------------
        // I = { list }
        //----------------------------------------------------------------------

        (*ni) = Item_len [0] ;
        (*I_allocated) = Item_allocated [0] ;
        I = (GrB_Index *) (Item [0]) ;
        if (I_max != NULL)
        {
            // find the max entry in the list
            if (Item_max [0] >= 0)
            { 
                // the max entry has already been computed (1-based)
                // convert from 1-based to 0-based
                (*I_max) = Item_max [0] - 1 ;
            }
            else
            { 
                // find the max entry (0-based)
                GrB_Index List_max = 0 ;
                GB_helper4 (I, (*ni), &List_max) ;
                (*I_max) = ((int64_t) List_max) - 1 ;
            }
        }

    }
    else if (len == 2)
    { 

        //----------------------------------------------------------------------
        // I = { start, fini }, defining start:fini
        //----------------------------------------------------------------------

        CHECK_ERROR (Item_len [0] != 1 || Item_len [1] != 1,
            "start and fini must be scalars for start:fini") ;

        I = mxMalloc (3 * sizeof (GrB_Index)) ;
        (*I_allocated) = true ;

        I [GxB_BEGIN] = Item [0][0] ;
        I [GxB_END  ] = Item [1][0] ;
        I [GxB_INC  ] = 0 ;             // unused

        if (Item_allocated [0]) gb_mxfree ((void **) (& (Item [0]))) ;
        if (Item_allocated [1]) gb_mxfree ((void **) (& (Item [1]))) ;

        (*ni) = GxB_RANGE ;
        if (I_max != NULL)
        { 
            // find the last index in the start:fini list
            (*I_max) = (int64_t) I [GxB_END] ;
        }

    }
    else // if (len == 3)
    {

        //----------------------------------------------------------------------
        // I = { start, inc, fini }, defining start:inc:fini
        //----------------------------------------------------------------------

        CHECK_ERROR (Item_len [0] != 1 || Item_len [1] != 1 ||
            Item_len [2] != 1,
            "start, inc, and fini must be scalars for start:inc:fini") ;

        I = mxMalloc (3 * sizeof (GrB_Index)) ;
        (*I_allocated) = true ;

        I [GxB_BEGIN] = Item [0][0] ;
        I [GxB_END  ] = Item [2][0] ;
        I [GxB_INC  ] = 0 ;
        int64_t iinc = Item [1][0] ;

        if (Item_allocated [1])
        { 
            // the 2nd item in the list is iinc, and if it was passed in as
            // 1-based, it has been decremented.  So increment it to get back
            // to the correct value.
            iinc++ ;
        }

        if (Item_allocated [0]) gb_mxfree ((void **) (& (Item [0]))) ;
        if (Item_allocated [1]) gb_mxfree ((void **) (& (Item [1]))) ;
        if (Item_allocated [2]) gb_mxfree ((void **) (& (Item [2]))) ;

        if (iinc < 0)
        { 
            I [GxB_INC] = (GrB_Index) (-iinc) ;
            (*ni) = GxB_BACKWARDS ;
            if (I_max != NULL)
            {
                // find the first entry in the list ibegin:iinc:iend.
                (*I_max) = -1 ;
                int64_t ibegin = (int64_t) I [GxB_BEGIN] ;
                int64_t iend   = (int64_t) I [GxB_END] ;
                if (iinc != 0 && ibegin >= iend)
                { 
                    // the list is non-empty, for example, 7:-2:4 = [7 5]
                    // I_max = GB_ijlist (NULL, 0, GB_STRIDE, I)
                    (*I_max) = ibegin ;
                }
            }
        }
        else
        { 
            I [GxB_INC] = (GrB_Index) (iinc) ;
            (*ni) = GxB_STRIDE ;
            if (I_max != NULL)
            {
                // find the last entry in the list ibegin:iinc:iend.
                (*I_max) = -1 ;
                int64_t ibegin = (int64_t) I [GxB_BEGIN] ;
                int64_t iend   = (int64_t) I [GxB_END] ;
                if (iinc != 0 && ibegin <= iend)
                { 
                    // the list is non-empty, for example, 4:2:9 = [4 6 8]
                    // nI = length of the expanded list (see GB_ijproperties),
                    // which is 3 for the list 4:2:9.
                    int64_t nI = ((iend - ibegin) / iinc) + 1 ;
                    // I_max = GB_ijlist (NULL, nI-1, GB_STRIDE, I),
                    // which is 8 for the list 4:2:9
                    (*I_max) = ibegin + (nI-1) * iinc ;
                }
            }
        }
    }

    return (I) ;
}

