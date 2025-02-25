//------------------------------------------------------------------------------
// GB_Matrix_diag: construct a diagonal matrix from a vector
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#define GB_FREE_WORKSPACE   \
{                           \
    GB_Matrix_free (&T) ;   \
}

#define GB_FREE_ALL         \
{                           \
    GB_FREE_WORKSPACE ;     \
    GB_phybix_free (C) ;    \
}

#include "diag/GB_diag.h"

GrB_Info GB_Matrix_diag     // build a diagonal matrix from a vector
(
    GrB_Matrix C,           // output matrix
    const GrB_Matrix V_in,  // input vector (as an n-by-1 matrix)
    int64_t k,
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;
    ASSERT_MATRIX_OK (C, "C input for GB_Matrix_diag", GB0) ;
    ASSERT_MATRIX_OK (V_in, "V input for GB_Matrix_diag", GB0) ;
    ASSERT (GB_VECTOR_OK (V_in)) ;       // V_in is a vector on input
    ASSERT (!GB_any_aliased (C, V_in)) ; // C and V_in cannot be aliased
    ASSERT (!GB_IS_HYPERSPARSE (V_in)) ; // vectors cannot be hypersparse

    struct GB_Matrix_opaque T_header ;
    GrB_Matrix T = NULL ;

    GrB_Type ctype = C->type ;
    int64_t n = V_in->vlen + GB_IABS (k) ;     // C must be n-by-n

    ASSERT (GB_NROWS (C) == GB_NCOLS (C))
    ASSERT (GB_NROWS (C) == n)
    ASSERT (GB_Type_compatible (ctype, V_in->type)) ;

    //--------------------------------------------------------------------------
    // finish any pending work in V_in and clear the output matrix C
    //--------------------------------------------------------------------------

    GB_MATRIX_WAIT (V_in) ;
    GB_phybix_free (C) ;

    //--------------------------------------------------------------------------
    // ensure V is not bitmap
    //--------------------------------------------------------------------------

    GrB_Matrix V ;
    if (GB_IS_BITMAP (V_in))
    { 
        // make a deep copy of V_in and convert to CSC
        GB_CLEAR_MATRIX_HEADER (T, &T_header) ;
        GB_OK (GB_dup_worker (&T, V_in->iso, V_in, true, NULL)) ;
        GB_OK (GB_convert_bitmap_to_sparse (T, Werk)) ;
        V = T ;
    }
    else
    { 
        // use V_in as-is
        V = V_in ;
    }

    //--------------------------------------------------------------------------
    // allocate C as sparse or hypersparse with vnz entries and vnz vectors
    //--------------------------------------------------------------------------

    // C is sparse if V is dense and k == 0, and hypersparse otherwise
    const int64_t vnz = GB_nnz (V) ;
    const bool V_is_full = GB_as_if_full (V) ;
    const int C_sparsity = (V_is_full && k == 0) ? GxB_SPARSE : GxB_HYPERSPARSE;
    const bool C_iso = V->iso ;
    if (C_iso)
    { 
        GBURBLE ("(iso diag) ") ;
    }
    const bool csc = C->is_csc ;
    const float bitmap_switch = C->bitmap_switch ;
    const int sparsity_control = C->sparsity_control ;

    // determine the p_is_32, j_is_32, and i_is_32 settings for the new matrix
    bool Cp_is_32, Cj_is_32, Ci_is_32 ;
    GB_determine_pji_is_32 (&Cp_is_32, &Cj_is_32, &Ci_is_32,
        C_sparsity, vnz, n, n, Werk) ;

    GB_OK (GB_new_bix (&C, // existing header
        ctype, n, n, GB_ph_malloc, csc, C_sparsity, false,
        C->hyper_switch, vnz, vnz, true, C_iso, Cp_is_32, Cj_is_32, Ci_is_32)) ;
    C->sparsity_control = sparsity_control ;
    C->bitmap_switch = bitmap_switch ;

    //--------------------------------------------------------------------------
    // handle the CSR/CSC format of C and determine position of diagonal
    //--------------------------------------------------------------------------

    if (!csc)
    { 
        // The kth diagonal of a CSC matrix is the same as the (-k)th diagonal
        // of the CSR format, so if C is CSR, negate the value of k.  Then
        // treat C as if it were CSC in the rest of this method.
        k = -k ;
    }

    int64_t kpositive, knegative ;
    if (k >= 0)
    { 
        kpositive = k ;
        knegative = 0 ;
    }
    else
    { 
        kpositive = 0 ;
        knegative = -k ;
    }

    //--------------------------------------------------------------------------
    // get the contents of C and determine # of threads to use
    //--------------------------------------------------------------------------

    int nthreads_max = GB_Context_nthreads_max ( ) ;
    double chunk = GB_Context_chunk ( ) ;
    int nthreads = GB_nthreads (vnz, chunk, nthreads_max) ;

    GB_Cp_DECLARE   (Cp, ) ; GB_Cp_PTR (Cp, C) ;
    GB_Ch_DECLARE   (Ch, ) ; GB_Ch_PTR (Ch, C) ;
    GB_Ci_DECLARE_U (Ci, ) ; GB_Ci_PTR (Ci, C) ;

    //--------------------------------------------------------------------------
    // copy the contents of V into the kth diagonal of C
    //--------------------------------------------------------------------------

    // C->x = (ctype) V->x
    GB_OK (GB_cast_matrix (C, V)) ;

    if (C_sparsity == GxB_SPARSE)
    {

        //----------------------------------------------------------------------
        // V is full, or can be treated as full, and k == 0
        //----------------------------------------------------------------------

        // construct Cp and Ci
        int64_t p ;
        #pragma omp parallel for num_threads(nthreads) schedule(static)
        for (p = 0 ; p < vnz ; p++)
        { 
            GB_ISET (Cp, p, p) ;    // Cp [p] = p ;
            GB_ISET (Ci, p, p) ;    // Ci [p] = p ;
        }

    }
    else if (V_is_full)
    {

        //----------------------------------------------------------------------
        // V is full, or can be treated as full, and k != 0
        //----------------------------------------------------------------------

        // construct Cp, Ch, and Ci
        int64_t p ;
        #pragma omp parallel for num_threads(nthreads) schedule(static)
        for (p = 0 ; p < vnz ; p++)
        { 
            int64_t j = p + kpositive ;
            int64_t i = p + knegative ;
            GB_ISET (Cp, p, p) ;        // Cp [p] = p ;
            GB_ISET (Ch, p, j) ;        // Ch [p] = j ;
            GB_ISET (Ci, p, i) ;        // Ci [p] = i ;
        }

    }
    else
    {

        //----------------------------------------------------------------------
        // V is sparse
        //----------------------------------------------------------------------

        GB_Ai_DECLARE (Vi, const) ; GB_Ai_PTR (Vi, V) ;

        // construct Cp, Ch, and Ci
        int64_t p ;
        #pragma omp parallel for num_threads(nthreads) schedule(static)
        for (p = 0 ; p < vnz ; p++)
        { 
            int64_t i = GB_IGET (Vi, p) ;   // i = Vi [p]
            int64_t j = i + kpositive ;
            i += knegative ;
            GB_ISET (Cp, p, p) ;            // Cp [p] = p ;
            GB_ISET (Ch, p, j) ;            // Ch [p] = j ;
            GB_ISET (Ci, p, i) ;            // Ci [p] = i ;
        }
    }

    //--------------------------------------------------------------------------
    // finalize the matrix C
    //--------------------------------------------------------------------------

    GB_ISET (Cp, vnz, vnz) ;    // Cp [vnz] = vnz ;
    C->nvals = vnz ;
    C->nvec = vnz ;
//  C->nvec_nonempty = vnz ;
    GB_nvec_nonempty_set (C, vnz) ;
    C->magic = GB_MAGIC ;

    //--------------------------------------------------------------------------
    // free workspace, conform C to its desired format, and return result
    //--------------------------------------------------------------------------

    GB_FREE_WORKSPACE ;
    GB_OK (GB_conform (C, Werk)) ;
    ASSERT_MATRIX_OK (C, "C output for GB_Matrix_diag", GB0) ;
    return (GrB_SUCCESS) ;
}

