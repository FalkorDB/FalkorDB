//------------------------------------------------------------------------------
// GB_Matrix_extractElement: x = A(row,col)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Extract the value of single scalar, x = A(row,col), typecasting from the
// type of A to the type of x, as needed.

// Returns GrB_SUCCESS if A(row,col) is present, and sets x to its value.
// Returns GrB_NO_VALUE if A(row,col) is not present, and x is unmodified.

// This template constructs GrB_Matrix_extractElement_[TYPE] for each of the
// 13 built-in types, and the _UDT method for all user-defined types.
// It also constructs GxB_Matrix_isStoredElement.

// The search strategy has been revised in GraphBLAS 10.4.0:  Do not wait
// unless jumbled.  First try to find the element.  If found (live or zombie),
// no need to wait.  If not found and pending tuples exist, wait and then
// search for it again.  Prior to this version, any call to this method would
// always finish any pending work.

GrB_Info GB_EXTRACT_ELEMENT     // extract a single entry, x = A(row,col)
(
    #ifdef GB_XTYPE
    GB_XTYPE *x,                // scalar to extract, not modified if not found
    #endif
    const GrB_Matrix A,         // matrix to extract a scalar from
    uint64_t row,               // row index
    uint64_t col                // column index
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;
    GB_RETURN_IF_NULL_OR_INVALID (A) ;
    #ifdef GB_XTYPE
    GB_RETURN_IF_NULL (x) ;
    #endif

    if (A->jumbled)
    { 
        GB_WHERE_1 (A, GB_WHERE_STRING) ;
        GB_BURBLE_START ("GrB_Matrix_extractElement") ;
        GB_OK (GB_wait (A, "A", Werk)) ;
        GB_BURBLE_END ;
    }

    // look for index i in vector j
    int64_t i, j ;
    const int64_t vlen = A->vlen ;
    if (A->is_csc)
    { 
        i = row ;
        j = col ;
        if (row >= vlen || col >= A->vdim)
        { 
            return (GrB_INVALID_INDEX) ;
        }
    }
    else
    { 
        i = col ;
        j = row ;
        if (col >= vlen || row >= A->vdim)
        { 
            return (GrB_INVALID_INDEX) ;
        }
    }

    //--------------------------------------------------------------------------
    // find the entry A(i,j)
    //--------------------------------------------------------------------------

    int64_t pleft ;
    bool found, is_zombie ;
    GB_Matrix_find_entry (&pleft, &found, &is_zombie, A, i, j) ;

    //--------------------------------------------------------------------------
    // check again if not found and the matrix has pending tuples
    //--------------------------------------------------------------------------

    if (!found && A->Pending != NULL)
    { 
        GB_WHERE_1 (A, GB_WHERE_STRING) ;
        GB_BURBLE_START ("GrB_Matrix_extractElement") ;
        GB_OK (GB_wait (A, "A", Werk)) ;
        GB_BURBLE_END ;
        GB_Matrix_find_entry (&pleft, &found, &is_zombie, A, i, j) ;
        ASSERT (!is_zombie) ;   // GB_wait has removed all zombies
    }

    //--------------------------------------------------------------------------
    // extract the element
    //--------------------------------------------------------------------------

    if (found && !is_zombie)
    {
        // entry found
        #ifdef GB_XTYPE
        GB_Type_code acode = A->type->code ;
        #if !defined ( GB_UDT_EXTRACT )
        if (GB_XCODE == acode)
        { 
            // copy Ax [pleft] into x, no typecasting, for built-in types only.
            GB_XTYPE *restrict Ax = ((GB_XTYPE *) (A->x)) ;
            (*x) = Ax [A->iso ? 0:pleft] ;
        }
        else
        #endif
        { 
            // typecast the value from Ax [pleft] into x
            if (!GB_code_compatible (GB_XCODE, acode))
            { 
                // x (GB_XCODE) and A (acode) must be compatible
                return (GrB_DOMAIN_MISMATCH) ;
            }
            size_t asize = A->type->size ;
            void *ax = ((GB_void *) A->x) + (A->iso ? 0 : (pleft*asize)) ;
            GB_cast_scalar (x, GB_XCODE, ax, acode, asize) ;
        }
        #pragma omp flush
        #endif
        return (GrB_SUCCESS) ;
    }
    else
    { 
        // entry not found
        return (GrB_NO_VALUE) ;
    }
}

#undef GB_UDT_EXTRACT
#undef GB_EXTRACT_ELEMENT
#undef GB_XTYPE
#undef GB_XCODE

