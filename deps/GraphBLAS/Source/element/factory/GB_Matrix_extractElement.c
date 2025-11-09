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

// FUTURE: tolerate zombies

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

    // TODO: do not wait unless jumbled.  First try to find the element.
    // If found (live or zombie), no need to wait.  If not found and pending
    // tuples exist, wait and then extractElement again.

    // delete any lingering zombies, assemble any pending tuples, and unjumble
    if (A->Pending != NULL || A->nzombies > 0 || A->jumbled)
    { 
        GB_WHERE_1 (A, GB_WHERE_STRING) ;
        GB_BURBLE_START ("GrB_Matrix_extractElement") ;
        GB_OK (GB_wait (A, "A", Werk)) ;
        GB_BURBLE_END ;
    }

    ASSERT (!GB_ANY_PENDING_WORK (A)) ;

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
    bool found ;
    GB_Ap_DECLARE (Ap, const) ; GB_Ap_PTR (Ap, A) ;

    if (Ap != NULL)
    {

        //----------------------------------------------------------------------
        // A is sparse or hypersparse
        //----------------------------------------------------------------------

        int64_t pA_start, pA_end ;
        if (A->h != NULL)
        {

            //------------------------------------------------------------------
            // A is hypersparse: look for j in hyperlist A->h [0 ... A->nvec-1]
            //------------------------------------------------------------------

            void *A_Yp = (A->Y == NULL) ? NULL : A->Y->p ;
            void *A_Yi = (A->Y == NULL) ? NULL : A->Y->i ;
            void *A_Yx = (A->Y == NULL) ? NULL : A->Y->x ;
            const int64_t A_hash_bits = (A->Y == NULL) ? 0 : (A->Y->vdim - 1) ;
            int64_t k = GB_hyper_hash_lookup (A->p_is_32, A->j_is_32,
                A->h, A->nvec, Ap, A_Yp, A_Yi, A_Yx, A_hash_bits,
                j, &pA_start, &pA_end) ;
            found = (k >= 0) ;
            if (!found)
            { 
                // vector j is empty
                return (GrB_NO_VALUE) ;
            }
            #ifdef GB_DEBUG
            GB_Ah_DECLARE (Ah, const) ; GB_Ah_PTR (Ah, A) ;
            ASSERT (j == GB_IGET (Ah, k)) ;
            #endif

        }
        else
        { 

            //------------------------------------------------------------------
            // A is sparse: look in the jth vector
            //------------------------------------------------------------------

            pA_start = GB_IGET (Ap, j);
            pA_end   = GB_IGET (Ap, j+1) ;
        }

        // vector j has been found, now look for index i
        pleft = pA_start ;
        int64_t pright = pA_end - 1 ;

        // Time taken for this step is at most O(log(nnz(A(:,j))).
        found = GB_binary_search (i, A->i, A->i_is_32, &pleft, &pright) ;

    }
    else
    {

        //----------------------------------------------------------------------
        // A is bitmap or full
        //----------------------------------------------------------------------

        pleft = i + j * vlen ;
        const int8_t *restrict Ab = A->b ;
        if (Ab != NULL)
        { 
            // A is bitmap
            found = (Ab [pleft] == 1) ;
        }
        else
        { 
            // A is full
            found = true ;
        }
    }

    //--------------------------------------------------------------------------
    // extract the element
    //--------------------------------------------------------------------------

    if (found)
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
        // TODO: do not flush if extracting to GrB_Scalar
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

