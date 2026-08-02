//------------------------------------------------------------------------------
// gbmex_logextract: logical extraction: C = A(M)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// gbmex_logextract computes the built-in logical indexing expression C = A(M).  The
// matrices A and M must be the same size.  M is normally logical but it can be
// of any type in this mexFunction.  M should not have any explicit zeros.  C
// has the same type as A, and is a sparse vector of size nnz(M)-by-1.

// This function accesses opaque content and GB_methods inside GraphBLAS.

// Usage:

// C = gbmex_logextract (ghb, A, M)

//  This function is the C equivalent of the following m-function:

/*

    function C = gbmex_logextract (A, M_input)
    % Computing the built-in logical indexing expression C = A(M) in GraphBLAS.
    % C is a sparse vector of size nnz(M)-by-1.  M is normally a sparse logical
    % matrix, either GraphBLAS or built-in, but it can be of any type.
    % A and M have the same size.

    [m n] = size (A) ;

    % make sure all input, internal, and output matrices are all stored by
    % column
    save = GrB.format ;
    GrB.format ('by col') ;
    M = GrB (m, n, 'logical') ;
    M = GrB.select (M, '2nd', 'nonzero', M_input) ;
    if (isequal (GrB.format (A), 'by row'))
        A = GrB (A) ;
    end
    mnz = nnz (M) ;         % C will be mnz-by-1

    % G<M> = A
    % G has the same type and size as A, but G is always stored by column
    G = GrB (m, n, GrB.type (A)) ;
    G = GrB.subassign (G, M, A) ;

    % extract gx = the entries of G
    [~, ~, gx] = GrB.extracttuples (G) ;

    % convert G to logical
    G = spones (G, 'logical') ;

    % K = symbolic structure of M, where the kth entry in K(:) is equal to k.
    desc.base = 'zero-based' ;
    [mi, mj] = GrB.extracttuples (M, desc) ;
    K = GrB.build (mi, mj, int64 (0:mnz-1), m, n, desc) ;

    % T<G> = K
    T = GrB (m, n, 'uint64') ;
    T = GrB.subassign (T, G, K) ;

    % extract the values from T
    [~, ~, tx] = GrB.extracttuples (T) ;

    % construct the result C (always a column vector)
    C = GrB.build (tx, zeros(length(gx),1,'uint64'), gx, mnz, 1) ;

    % restore the format to its original state
    GrB.format (save) ;

*/

// This C mexFunction is faster than the above m-function, since it avoids the
// use of GrB.extracttuples and GrB.build.  Instead, it accesses the internal
// structure of the GrB_Matrix objects, and creates shallow copies.  The
// m-file above is useful for understanding that this C mexFunction does.

// C is always returned as a GrB matrix.

#include "gb_interface.h"
#include "gbmx_interface.h"
#include "GB_transpose.h"

#undef  FREE_WORK
#define FREE_WORK                       \
    gb_free ((void **) &Kx, arena) ;    \
    GrB_Matrix_free (&G) ;              \
    GrB_Matrix_free (&K) ;              \
    GrB_Matrix_free (&T) ;              \
    GrB_Matrix_free (&M) ;              \
    GrB_Matrix_free (&M_to_free) ;      \
    GrB_Matrix_free (&A_copy) ;         \
    GrB_Matrix_free (&A_to_free) ;

#undef  FREE_ALL
#define FREE_ALL                        \
    FREE_WORK ;                         \
    GrB_Vector_free (&V) ;              \
    GrB_Matrix_free (&C) ;

#define USAGE "usage: C = gbmex_logextract (ghb, A, M)"

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{

    //--------------------------------------------------------------------------
    // check inputs and construct outputs
    //--------------------------------------------------------------------------

    GrB_Matrix *C_opaque = NULL, K = NULL, M = NULL, A = NULL, A_copy = NULL,
        G = NULL, T = NULL, C = NULL, A_input = NULL, A_to_free = NULL,
        M_input = NULL, M_to_free = NULL ;
    GrB_Vector V = NULL ;
    uint64_t *Kx = NULL ;

    GBMX_USAGE (nargin == 3 && nargout <= 1, USAGE) ;
    bool ghb = (bool) mxGetScalar (pargin [0]) ;
    int arena = ghb ? GrB_DEFAULT : MXARENA ;

    if (ghb) pargout [0] = gbmx_export_ghb_mxstruct (&C_opaque) ;

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    struct gb_matrix_struct Matrix [2] ;
    gbmx_get_matrix (&(Matrix [0]), pargin [1]) ;
    gbmx_get_matrix (&(Matrix [1]), pargin [2]) ;

    ////////////////////////////////////////////////////////////////////////////

    GB_WERK ("gbmex_logextract") ;

    //--------------------------------------------------------------------------
    // get A
    //--------------------------------------------------------------------------

    // make sure A is stored by column
    OK (gb_get_matrix (&A_input, &A_to_free, &(Matrix [0]), arena, err)) ;

    OK (gb_by_col (&A, &A_copy, A_input, arena, err)) ;

    GrB_Index nrows, ncols ;
    OK (GrB_Matrix_nrows (&nrows, A)) ;
    OK (GrB_Matrix_ncols (&ncols, A)) ;

    //--------------------------------------------------------------------------
    // get M
    //--------------------------------------------------------------------------

    // M can be hypersparse, sparse, or full, but not bitmap
    int not_bitmap = GxB_HYPERSPARSE + GxB_SPARSE + GxB_FULL ;

    // make M boolean, stored by column, and drop explicit zeros
    OK (gb_get_matrix (&M_input, &M_to_free, &(Matrix [1]), arena, err)) ;
    OK (gb_new (&M, GrB_BOOL, nrows, ncols, GxB_BY_COL, not_bitmap, arena,
        err)) ;
    OK1 (M, GrB_Matrix_select_BOOL (M, NULL, NULL, GrB_VALUENE_BOOL, M_input,
        0, NULL)) ;
    GrB_Matrix_free (&M_to_free) ;

    GrB_Index mnz ;
    OK (GrB_Matrix_nvals (&mnz, M)) ;
    int sparsity ;
    OK (GrB_Matrix_get_INT32 (M, &sparsity, GxB_SPARSITY_STATUS)) ;
    CHECK_ERROR (sparsity == GxB_BITMAP, "internal error 8") ;
    CHECK_ERROR (!M->iso, "internal error 9")  ;            	

    //--------------------------------------------------------------------------
    // G<M> = A
    //--------------------------------------------------------------------------

    // G has the same type and size as A, but it is always stored by column.
    // Also ensure the G is not bitmap.
    GrB_Type type ;
    OK (GxB_Matrix_type (&type, A)) ;
    OK (gb_new (&G, type, nrows, ncols, GxB_BY_COL, not_bitmap, arena, err)) ;
    OK1 (G, GxB_Matrix_subassign (G, M, NULL,
        A, GrB_ALL, nrows, GrB_ALL, ncols, NULL)) ;
    GrB_Matrix_free (&A_copy) ;
    GrB_Matrix_free (&A_to_free) ;

    //--------------------------------------------------------------------------
    // extract Gx, the values of G
    //--------------------------------------------------------------------------

    GrB_Index gnvals ;
    OK1 (G, GrB_Matrix_wait (G, GrB_MATERIALIZE)) ;
    OK (GrB_Matrix_nvals (&gnvals, G)) ;
    OK (GrB_Matrix_get_INT32 (G, &sparsity, GxB_SPARSITY_STATUS)) ;
    CHECK_ERROR (sparsity == GxB_BITMAP, "internal error 10") ;

    // Remove G->x from G
    void *Gx = G->x ;
    uint64_t Gx_mem = G->x_mem ;
    GBMDUMP ("remove G->x from memtable: %p\n", G->x) ;
    GB_Global_memtable_remove (G->x) ;
    G->x = NULL ; G->x_mem = 0 ;
    bool G_iso = G->iso  ;            	

    //--------------------------------------------------------------------------
    // change G to boolean (all true and iso)
    //--------------------------------------------------------------------------

    bool Gbool = true ;        							
    G->type = GrB_BOOL ;       	             	 	                 	
    G->x = &Gbool ;            		 	 	 	 	 	
    G->iso = true ;            		 	 	 	 	 	
    G->x_shallow = true ;      		 	 	 	 	 	
    G->x_mem = GB_mem (0, sizeof (bool)) ;      // G->x is on the stack

    //--------------------------------------------------------------------------
    // K = structure of M, where the kth entry in K is equal to k
    //--------------------------------------------------------------------------

    // K is a shallow copy of M, except for its numerical values
    OK (GB_matrix_header_new (&K, GrB_DEFAULT, GrB_DEFAULT)) ;

    OK (GB_shallow_copy (K, GxB_BY_COL, M, NULL)) ;
    OK (GrB_Matrix_get_INT32 (K, &sparsity, GxB_SPARSITY_STATUS)) ;
    CHECK_ERROR (sparsity == GxB_BITMAP, "internal error 11") ;

    // Kx = uint64 (0:mnz-1)
    size_t Kx_memsize = (MAX (mnz, 1) * sizeof (uint64_t)) ;
    uint64_t Kx_mem = GB_mem (arena, Kx_memsize) ;
    Kx = gb_malloc (Kx_memsize, arena) ;
    if (Kx == NULL) ERROR ("out of memory", GrB_OUT_OF_MEMORY) ;
    OK (GB_helper7 (Kx, mnz)) ;

    // add a new K->x to K
    K->x = Kx ; Kx = NULL ;
    K->x_shallow = false ;
    K->type = GrB_UINT64 ;
    K->x_mem = Kx_mem ;
    GBMDUMP ("add K->x to memtable: %p\n", K->x) ;
    GB_Global_memtable_add (K->x, K->x_mem) ;
    K->iso = false  ;            	

    //--------------------------------------------------------------------------
    // T<G> = K
    //--------------------------------------------------------------------------

    OK (gb_new (&T, GrB_UINT64, nrows, ncols, GxB_BY_COL, not_bitmap, arena,
        err)) ;
    OK1 (T, GxB_Matrix_subassign (T, G, NULL,
        K, GrB_ALL, nrows, GrB_ALL, ncols, NULL)) ;

    //--------------------------------------------------------------------------
    // extract Tx, the values of T
    //--------------------------------------------------------------------------

    GrB_Index tnvals ;
    OK1 (T, GrB_Matrix_wait (T, GrB_MATERIALIZE)) ;
    OK (GrB_Matrix_nvals (&tnvals, T)) ;
    uint64_t *Tx = T->x ;
    size_t Tx_mem = T->x_mem ;
    GBMDUMP ("remove T->x from memtable: %p\n", T->x) ;
    GB_Global_memtable_remove (T->x) ;
    T->x = NULL ; T->x_mem = 0 ;

    // gnvals and tnvals are identical, by construction
    CHECK_ERROR (gnvals != tnvals, "internal error 12") ;

    //--------------------------------------------------------------------------
    // construct the result C
    //--------------------------------------------------------------------------

    // Vectors are always stored by column, and are never hypersparse.  This
    // step takes constant time, using a transplant of the row indices Tx from
    // T and the values Gx from G.  V is sparse (not full, not hypersparse).

    OK (GxB_Vector_new_arena (&V, type, mnz, arena, arena)) ;
    OK (GrB_Vector_set_INT32 (V, GxB_SPARSE, GxB_SPARSITY_CONTROL)) ;

    GBMDUMP ("remove V->i from memtable: %p\n", V->i) ;
    GBMDUMP ("remove V->x from memtable: %p\n", V->x) ;
    GB_Global_memtable_remove (V->i) ;
    gb_free ((void **) (&V->i), arena) ;
    GB_Global_memtable_remove (V->x) ;
    gb_free ((void **) (&V->x), arena) ;

    // transplant values of T as the row indices of V
    V->i = (void *) Tx ;
    V->i_mem = Tx_mem ;
    V->i_shallow = false ;
    V->i_is_32 = false ;
    GBMDUMP ("add V->i to memtable: %p\n", V->i) ;
    GB_Global_memtable_add (V->i, V->i_mem) ;

    // transplant the values of G as the values of V
    V->x = Gx ;
    V->x_mem = Gx_mem ;
    V->x_shallow = false ;
    V->iso = G_iso  ;            	
    GBMDUMP ("add V->x to memtable: %p\n", V->x) ;
    GB_Global_memtable_add (V->x, V->x_mem) ;

    GB_Ap_DECLARE (Vp, ) ; GB_Ap_PTR (Vp, V) ;
    GB_ISET (Vp, 0, 0) ;        // Vp [0] = 0 ;
    GB_ISET (Vp, 1, tnvals) ;   // Vp [1] = tnvals ;

    V->nvals = tnvals ;
    V->magic = GB_MAGIC ;
    GB_nvec_nonempty_set ((GrB_Matrix) V, (tnvals > 0) ? 1 : 0) ;

    // typecast V to a matrix C, for export
    C = (GrB_Matrix) V ;
    V = NULL ;

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    FREE_WORK ;

    OK (gb_export (C_opaque, &C, KIND_GRB, ghb, err)) ;
    ////////////////////////////////////////////////////////////////////////////
    if (!ghb)
    { 
        pargout [0] = gbmx_export_grb_mxstruct (&C) ;
    }

    gb_wrapup ( ) ;
}

