//------------------------------------------------------------------------------
// GB_extractTuples: extract all the tuples from a matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Extracts all tuples from a matrix, like [I,J,X] = find (A).  If any
// parameter I, J and/or X is NULL, then that component is not extracted.  The
// size of the I, J, and X arrays (those that are not NULL) is given by nvals,
// which must be at least as large as GrB_nvals (&nvals, A).  The values in the
// matrix are typecasted to the type of X, as needed.

// This function does the work for the user-callable GrB_*_extractTuples
// functions, and helps build the tuples for GB_concat_hyper.

// If A is iso and X is not NULL, the iso scalar Ax [0] is expanded into X.

// FUTURE: pass in parameters I_offset and J_offset to add to I and J

#include "GB.h"
#include "extractTuples/GB_extractTuples.h"

#define GB_FREE_ALL                             \
{                                               \
    GB_FREE_MEMORY (&Cp, Cp_size) ;               \
}

GrB_Info GB_extractTuples       // extract all tuples from a matrix
(
    void *I_out,                // array for returning row indices of tuples
    bool I_is_32_out,           // if true, I is 32-bit; else 64 bit
    void *J_out,                // array for returning col indices of tuples
    bool J_is_32_out,           // if true, J is 32-bit; else 64 bit
    void *X,                    // array for returning values of tuples
    uint64_t *p_nvals,          // I,J,X size on input; # tuples on output
    const GrB_Type xtype,       // type of array X
    const GrB_Matrix A,         // matrix to extract tuples from
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;
    void *Cp = NULL ; size_t Cp_size = 0 ;
    ASSERT_MATRIX_OK (A, "A to extract", GB0) ;
    ASSERT_TYPE_OK (xtype, "xtype to extract", GB0) ;
    ASSERT (p_nvals != NULL) ;

    // delete any lingering zombies and assemble any pending tuples;
    // allow A to remain jumbled
    GB_MATRIX_WAIT_IF_PENDING_OR_ZOMBIES (A) ;

    // get the types
    GB_BURBLE_DENSE (A, "(A %s) ") ;
    const GB_Type_code xcode = xtype->code ;
    const GB_Type_code acode = A->type->code ;
    const size_t asize = A->type->size ;

    const int64_t anz = GB_nnz (A) ;
    if (anz == 0)
    { 
        // no work to do
        (*p_nvals) = 0 ;
        return (GrB_SUCCESS) ;
    }

    int64_t nvals = *p_nvals ;          // size of I,J,X on input
    if (nvals < anz && (I_out != NULL || J_out != NULL || X != NULL))
    { 
        // output arrays are not big enough
        return (GrB_INSUFFICIENT_SPACE) ;
    }

    //-------------------------------------------------------------------------
    // determine the number of threads to use
    //-------------------------------------------------------------------------

    int nthreads_max = GB_Context_nthreads_max ( ) ;
    double chunk = GB_Context_chunk ( ) ;
    int nthreads = GB_nthreads (anz + A->nvec, chunk, nthreads_max) ;

    //-------------------------------------------------------------------------
    // handle the CSR/CSC format
    //--------------------------------------------------------------------------

    void *I, *J ;
    bool I_is_32, J_is_32 ;
    if (A->is_csc)
    { 
        I = I_out ;
        J = J_out ;
        I_is_32 = I_is_32_out ;
        J_is_32 = J_is_32_out ;
    }
    else
    { 
        I = J_out ;
        J = I_out ;
        I_is_32 = J_is_32_out ;
        J_is_32 = I_is_32_out ;
    }

    //--------------------------------------------------------------------------
    // bitmap case
    //--------------------------------------------------------------------------

    if (GB_IS_BITMAP (A))
    {

        //----------------------------------------------------------------------
        // allocate workspace
        //----------------------------------------------------------------------

        bool Cp_is_32 = GB_determine_p_is_32 (true, anz) ;   // OK
        size_t cpsize = (Cp_is_32) ? sizeof (uint32_t) : sizeof (uint64_t) ;
        Cp = GB_MALLOC_MEMORY (A->vdim+1, cpsize, &Cp_size) ;
        if (Cp == NULL)
        { 
            // out of memory
            GB_FREE_ALL ;
            return (GrB_OUT_OF_MEMORY) ;
        }

        //----------------------------------------------------------------------
        // extract the tuples
        //----------------------------------------------------------------------

        // Extract the pattern and the values, typecasting if needed.  If A is
        // iso or X is NULL, GB_convert_b2s only does the symbolic work.

        // FUTURE: either extract the tuples directly from the bitmap, without
        // the need for Cp, or revise GB_convert_b2s to take in offsets
        // to add to I and J.

        GB_OK (GB_convert_b2s (Cp, I, J, (GB_void *) X, NULL,
            Cp_is_32, I_is_32, J_is_32, xtype, A, Werk)) ;

        if (A->iso && X != NULL)
        { 
            // A is iso but a non-iso X has been requested;
            // typecast the iso scalar and expand it into X
            const size_t xsize = xtype->size ;
            GB_void scalar [GB_VLA(xsize)] ;
            GB_cast_scalar (scalar, xcode, A->x, acode, asize) ;
            GB_OK (GB_iso_expand (X, anz, scalar, xtype)) ;
        }

    }
    else
    {

        //----------------------------------------------------------------------
        // sparse, hypersparse, or full case
        //----------------------------------------------------------------------

        //----------------------------------------------------------------------
        // extract the row indices
        //----------------------------------------------------------------------

        if (I != NULL)
        {
            GB_IDECL (I, , u) ; GB_IPTR (I, I_is_32) ;
            if (A->i == NULL)
            {
                // A is full; construct the row indices
                int64_t avlen = A->vlen ;
                int64_t p ;
                #pragma omp parallel for num_threads(nthreads) schedule(static)
                for (p = 0 ; p < anz ; p++)
                { 
                    int64_t i = (p % avlen) ;
                    // I [p] = i ;
                    GB_ISET (I, p, i) ;
                }
            }
            else
            { 
                // A is sparse or hypersparse; copy/cast A->i into I
                GB_cast_int (
                    I,       I_is_32 ? GB_UINT32_code : GB_UINT64_code,
                    A->i, A->i_is_32 ? GB_UINT32_code : GB_UINT64_code,
                    anz, nthreads_max) ;
            }
        }

        //----------------------------------------------------------------------
        // extract the column indices
        //----------------------------------------------------------------------

        if (J != NULL)
        { 
            GB_OK (GB_extract_vector_list (J, J_is_32, A, Werk)) ;
        }

        //----------------------------------------------------------------------
        // extract the values
        //----------------------------------------------------------------------

        if (X != NULL)
        {
            if (A->iso)
            { 
                // A is iso but a non-iso X has been requested;
                // typecast the iso scalar and expand it into X
                const size_t xsize = xtype->size ;
                GB_void scalar [GB_VLA(xsize)] ;
                GB_cast_scalar (scalar, xcode, A->x, acode, asize) ;
                GB_OK (GB_iso_expand (X, anz, scalar, xtype)) ;
            }
            else if (xcode == acode)
            { 
                // copy the values from A into X, no typecast
                GB_memcpy (X, A->x, anz * asize, nthreads) ;
            }
            else
            { 
                // typecast the values from A into X
                ASSERT (X != NULL) ;
                ASSERT_MATRIX_OK (A, "A to cast_array", GB0) ;
                GB_OK (GB_cast_array ((GB_void *) X, xcode, A, nthreads)) ;
            }
        }
    }

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    *p_nvals = anz ;            // number of tuples extracted
    GB_FREE_ALL ;
    return (GrB_SUCCESS) ;
}

