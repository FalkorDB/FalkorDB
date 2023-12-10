//------------------------------------------------------------------------------
// GB_reshape:  reshape a matrix into another matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// JIT: not needed, but variants possible.

// If the input matrix is nrows-by-ncols, and the size of the newly-created
// matrix C is nrows_new-by-ncols_new, then nrows*ncols must equal
// nrows_new*ncols_new.

#include "GB.h"
#include "GB_reshape.h"
#include "GB_transpose.h"
#include "GB_ek_slice.h"
#include "GB_build.h"

#define GB_FREE_WORKSPACE                       \
{                                               \
    GB_WERK_POP (T_ek_slicing, int64_t) ;       \
    GB_FREE (&I_work, I_work_size) ;            \
    GB_FREE (&J_work, J_work_size) ;            \
    GB_FREE (&S_work, S_work_size) ;            \
    if (T != A && T != C)                       \
    {                                           \
        GB_Matrix_free (&T) ;                   \
    }                                           \
}

#define GB_FREE_ALL                             \
{                                               \
    GB_FREE_WORKSPACE ;                         \
    if (Chandle == NULL)                        \
    {                                           \
        GB_phybix_free (A) ;                    \
    }                                           \
    else                                        \
    {                                           \
        GB_Matrix_free (&C) ;                   \
    }                                           \
}

GrB_Info GB_reshape         // reshape a GrB_Matrix into another GrB_Matrix
(
    // output, if not in-place:
    GrB_Matrix *Chandle,    // output matrix, in place if Chandle == NULL
    // input, or input/output:
    GrB_Matrix A,           // input matrix, or input/output if in-place
    // input:
    bool by_col,            // true if reshape by column, false if by row
    int64_t nrows_new,      // number of rows of C
    int64_t ncols_new,      // number of columns of C
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;
    ASSERT_MATRIX_OK (A, "A for reshape", GB0) ;

    int64_t *I_work = NULL, *J_work = NULL ;
    GB_void *S_work = NULL, *S_input = NULL ;
    size_t I_work_size = 0, J_work_size = 0, S_work_size = 0 ;
    GB_WERK_DECLARE (T_ek_slicing, int64_t) ;
    GrB_Matrix C = NULL, T = NULL ;

    bool in_place = (Chandle == NULL) ;
    if (!in_place)
    { 
        (*Chandle) = NULL ;
    }

    GrB_Index matrix_size, s ;
    int64_t nrows_old = GB_NROWS (A) ;
    int64_t ncols_old = GB_NCOLS (A) ;
    bool ok = GB_int64_multiply (&matrix_size, nrows_old, ncols_old) ;
    if (!ok)
    { 
        // problem too large
        return (GrB_OUT_OF_MEMORY) ;
    }

    ok = GB_int64_multiply (&s, nrows_new, ncols_new) ;
    if (!ok || s != matrix_size)
    { 
        // dimensions are invalid
        return (GrB_DIMENSION_MISMATCH) ;
    }

    //--------------------------------------------------------------------------
    // finish any pending work, and transpose the input matrix if needed
    //--------------------------------------------------------------------------

    GB_MATRIX_WAIT (A) ;

    GrB_Type type = A->type ;
    bool A_is_csc = A->is_csc ;
    if (A_is_csc != by_col)
    {
        // transpose the input matrix
        if (in_place)
        { 
            // transpose A in-place
            GB_OK (GB_transpose_in_place (A, by_col, Werk)) ;
            T = A ;
        }
        else
        { 
            // T = A'
            GB_OK (GB_new (&T,  // new header
                type, A->vdim, A->vlen, GB_Ap_null, by_col, GxB_AUTO_SPARSITY,
                GB_Global_hyper_switch_get ( ), 0)) ;
            GB_OK (GB_transpose_cast (T, type, by_col, A, false, Werk)) ;
            // now T can be reshaped in-place to construct C
            in_place = true ;
        }
    }
    else
    { 
        // use T = A as-is, and reshape it either in-place or not in-place
        T = A ;
    }

    // T is now in the format required for the reshape
    ASSERT_MATRIX_OK (T, "T for reshape", GB0) ;
    ASSERT (T->is_csc == by_col) ;

    //--------------------------------------------------------------------------
    // determine the dimensions of C
    //--------------------------------------------------------------------------

    int64_t vlen_new, vdim_new ;
    bool T_is_csc = T->is_csc ;
    if (T_is_csc)
    { 
        vlen_new = nrows_new ;
        vdim_new = ncols_new ;
    }
    else
    { 
        vlen_new = ncols_new ;
        vdim_new = nrows_new ;
    }

    //--------------------------------------------------------------------------
    // C = reshape (T), keeping the same format (by_col)
    //--------------------------------------------------------------------------

    if (GB_IS_FULL (T) || GB_IS_BITMAP (T))
    {

        //----------------------------------------------------------------------
        // T and C are both full or both bitmap
        //----------------------------------------------------------------------

        if (in_place)
        { 
            // move T into C
            C = T ;
            T = NULL ;
        }
        else
        { 
            // copy T into C
            GB_OK (GB_dup (&C, T, Werk)) ;
        }
        // change the size of C
        C->vlen = vlen_new ;
        C->vdim = vdim_new ;
        C->nvec = vdim_new ;
        C->nvec_nonempty = (vlen_new == 0) ? 0 : vdim_new ;

    }
    else
    {

        //----------------------------------------------------------------------
        // sparse/hypersparse case
        //----------------------------------------------------------------------

        int64_t nvals = GB_nnz (T) ;
        int64_t *Tp = T->p ;
        int64_t *Th = T->h ;
        int64_t *Ti = T->i ;
        bool T_iso = T->iso ;
        int64_t tvlen = T->vlen ;
        bool T_jumbled = T->jumbled ;

        int nthreads_max = GB_Context_nthreads_max ( ) ;
        double chunk = GB_Context_chunk ( ) ;
        int T_nthreads, T_ntasks ;
        GB_SLICE_MATRIX (T, 1) ;

        //----------------------------------------------------------------------
        // allocate output and workspace
        //----------------------------------------------------------------------

        if (in_place)
        { 

            //------------------------------------------------------------------
            // Remove T->i and T->x from T; these become I_work and S_work
            //------------------------------------------------------------------

            // remove T->i from T; it becomes I_work
            I_work = T->i ; I_work_size = T->i_size ;
            T->i = NULL   ; T->i_size = 0 ;
            // remove T->x from T; it becomes S_work
            S_work = T->x ; S_work_size = T->x_size ;
            T->x = NULL   ; T->x_size = 0 ;
            S_input = NULL ;

            // move T into C
            C = T ;
            T = NULL ;

        }
        else
        {

            //------------------------------------------------------------------
            // create a new matrix C for GB_builder and allocate I_work
            //------------------------------------------------------------------

            // create the output matrix (just the header; no content)
            GB_OK (GB_new (&C, // new header
                type, vlen_new, vdim_new, GB_Ap_null, T_is_csc,
                GxB_AUTO_SPARSITY, GB_Global_hyper_switch_get ( ), 0)) ;
            // allocate new space for the future C->i
            I_work = GB_MALLOC (nvals, int64_t, &I_work_size) ;
            if (I_work == NULL)
            { 
                // out of memory
                GB_FREE_ALL ;
                return (GrB_OUT_OF_MEMORY) ;
            }
            // use T->x as S_input to GB_builder, which is not modified
            S_input = T->x ;
        }

        if (vdim_new > 1)
        {
            // J_work is not needed if vdim_new == 1
            J_work = GB_MALLOC (nvals, int64_t, &J_work_size) ;
            if (J_work == NULL)
            { 
                // out of memory
                GB_FREE_ALL ;
                return (GrB_OUT_OF_MEMORY) ;
            }
        }

        //----------------------------------------------------------------------
        // construct the new indices
        //----------------------------------------------------------------------

        int tid ;

        if (vdim_new == 1)
        { 

            //------------------------------------------------------------------
            // C is a single vector: no J_work is needed, and new index is 1D
            //------------------------------------------------------------------

            #pragma omp parallel for num_threads(T_nthreads) schedule(static)
            for (tid = 0 ; tid < T_ntasks ; tid++)
            {
                int64_t kfirst = kfirst_Tslice [tid] ;
                int64_t klast  = klast_Tslice  [tid] ;
                for (int64_t k = kfirst ; k <= klast ; k++)
                {
                    int64_t jold = GBH (Th, k) ;
                    int64_t pT_start, pT_end ;
                    GB_get_pA (&pT_start, &pT_end, tid, k,
                        kfirst, klast, pstart_Tslice, Tp, tvlen) ;
                    for (int64_t p = pT_start ; p < pT_end ; p++)
                    {
                        int64_t iold = Ti [p] ;
                        // convert (iold,jold) to a 1D index
                        int64_t index_1d = iold + jold * tvlen ;
                        // save the new 1D index
                        I_work [p] = index_1d ;
                    }
                }
            }

        }
        else
        { 

            //------------------------------------------------------------------
            // C is a matrix
            //------------------------------------------------------------------

            #pragma omp parallel for num_threads(T_nthreads) schedule(static)
            for (tid = 0 ; tid < T_ntasks ; tid++)
            {
                int64_t kfirst = kfirst_Tslice [tid] ;
                int64_t klast  = klast_Tslice  [tid] ;
                for (int64_t k = kfirst ; k <= klast ; k++)
                {
                    int64_t jold = GBH (Th, k) ;
                    int64_t pT_start, pT_end ;
                    GB_get_pA (&pT_start, &pT_end, tid, k,
                        kfirst, klast, pstart_Tslice, Tp, tvlen) ;
                    for (int64_t p = pT_start ; p < pT_end ; p++)
                    {
                        int64_t iold = Ti [p] ;
                        // convert (iold,jold) to a 1D index
                        int64_t index_1d = iold + jold * tvlen ;
                        // convert the 1D index to the 2d index: (inew,jnew)
                        int64_t inew = index_1d % vlen_new ;
                        int64_t jnew = (index_1d - inew) / vlen_new ;
                        // save the new indices
                        I_work [p] = inew ;
                        J_work [p] = jnew ;
                    }
                }
            }
        }

        //----------------------------------------------------------------------
        // free the old C->p and C->h, if constructing C in place
        //----------------------------------------------------------------------

        if (in_place)
        { 
            GB_phybix_free (C) ;
        }

        //----------------------------------------------------------------------
        // build the output matrix C
        //----------------------------------------------------------------------

        GB_OK (GB_builder (
            C,              // output matrix
            type,           // same type as T
            vlen_new,       // new vlen
            vdim_new,       // new vdim
            T_is_csc,       // same format as T
            &I_work,        // transplanted into C->i
            &I_work_size,
            &J_work,        // freed when done
            &J_work_size,
            &S_work,        // array of values; transplanted into C->x in-place
            &S_work_size,
            !T_jumbled,     // indices may be jumbled on input
            true,           // no duplicates exist
            nvals,          // number of entries in T and C 
            true,           // C is a matrix
            NULL,           // I_input is not used
            NULL,           // J_input is not used
            S_input,        // S_input is used if not in-place; NULL if in-place
            T_iso,          // true if T and C are iso-valued
            nvals,          // number of entries in T and C 
            NULL,           // no dup operator
            type,           // type of S_work and S_input
            true,           // burble is allowed
            Werk
        )) ;

        ASSERT (I_work == NULL) ;   // transplanted into C->i
        ASSERT (J_work == NULL) ;   // freed by GB_builder
        ASSERT (S_work == NULL) ;   // freed by GB_builder
    }

    //--------------------------------------------------------------------------
    // transpose C if needed, to change its format to match the format of A
    //--------------------------------------------------------------------------

    ASSERT_MATRIX_OK (C, "C for reshape before transpose", GB0) ;
    ASSERT (C->is_csc == T_is_csc) ;
    if (A_is_csc != T_is_csc)
    { 
        GB_OK (GB_transpose_in_place (C, A_is_csc, Werk)) ;
    }

    //--------------------------------------------------------------------------
    // free workspace, conform C, and return results
    //--------------------------------------------------------------------------

    GB_FREE_WORKSPACE ;
    GB_OK (GB_conform (C, Werk)) ;
    ASSERT_MATRIX_OK (C, "C result for reshape", GB0) ;
    if (Chandle != NULL)
    { 
        (*Chandle) = C ;
    }
    return (GrB_SUCCESS) ;
}

