//------------------------------------------------------------------------------
// gblogextract: logical extraction: C = A(M)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// gblogextract computes the built-in logical indexing expression C = A(M).  The
// matrices A and M must be the same size.  M is normally logical but it can be
// of any type in this mexFunction.  M should not have any explicit zeros.  C
// has the same type as A, and is a sparse vector of size nnz(M)-by-1.

// Usage:

// C = gblogextract (A, M)

//  This function is the C equivalent of the following m-function:

/*

    function C = gblogextract (A, M_input)
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
#include "GB_transpose.h"

#define USAGE "usage: C = gblogextract (A, M)"

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    gb_usage (nargin == 2 && nargout <= 1, USAGE) ;
    GB_WERK ("gblogextract") ;

    //--------------------------------------------------------------------------
    // get A
    //--------------------------------------------------------------------------

    // make sure A is stored by column
    GrB_Matrix A_input = gb_get_shallow (pargin [0]) ;
    GrB_Matrix A, A_copy ;
    A = gb_by_col (&A_copy, A_input) ;

    GrB_Index nrows, ncols ;
    OK (GrB_Matrix_nrows (&nrows, A)) ;
    OK (GrB_Matrix_ncols (&ncols, A)) ;

    //--------------------------------------------------------------------------
    // get M
    //--------------------------------------------------------------------------

    // M can be hypersparse, sparse, or full, but not bitmap
    int not_bitmap = GxB_HYPERSPARSE + GxB_SPARSE + GxB_FULL ;

    // make M boolean, stored by column, and drop explicit zeros
    GrB_Matrix M_input = gb_get_shallow (pargin [1]) ;
    GrB_Matrix M = gb_new (GrB_BOOL, nrows, ncols, GxB_BY_COL, not_bitmap) ;
//  OK1 (M, GxB_Matrix_select (M, NULL, NULL, GxB_NONZERO, M_input,
//      NULL, NULL)) ;
    OK1 (M, GrB_Matrix_select_BOOL (M, NULL, NULL, GrB_VALUENE_BOOL, M_input,
        0, NULL)) ;
    OK (GrB_Matrix_free (&M_input)) ;

    GrB_Index mnz ;
    OK (GrB_Matrix_nvals (&mnz, M)) ;
    int sparsity ;
    OK (GxB_Matrix_Option_get (M, GxB_SPARSITY_STATUS, &sparsity)) ;
    CHECK_ERROR (sparsity == GxB_BITMAP, "internal error 5") ;
    CHECK_ERROR (!M->iso, "internal error 42")  ;            	

    //--------------------------------------------------------------------------
    // G<M> = A
    //--------------------------------------------------------------------------

    // G has the same type and size as A, but it is always stored by column.
    // Also ensure the G is not bitmap.
    GrB_Type type ;
    OK (GxB_Matrix_type (&type, A)) ;
    GrB_Matrix G = gb_new (type, nrows, ncols, GxB_BY_COL, not_bitmap) ;
    OK1 (G, GxB_Matrix_subassign (G, M, NULL,
        A, GrB_ALL, nrows, GrB_ALL, ncols, NULL)) ;
    OK (GrB_Matrix_free (&A_copy)) ;
    OK (GrB_Matrix_free (&A_input)) ;

    //--------------------------------------------------------------------------
    // extract Gx, the values of G
    //--------------------------------------------------------------------------

    GrB_Index gnvals ;
    OK1 (G, GrB_Matrix_wait (G, GrB_MATERIALIZE)) ;
    OK (GrB_Matrix_nvals (&gnvals, G)) ;
    OK (GxB_Matrix_Option_get (G, GxB_SPARSITY_STATUS, &sparsity)) ;
    CHECK_ERROR (sparsity == GxB_BITMAP, "internal error 0") ;

    // Remove G->x from G
    void *Gx = G->x ;
    size_t Gx_size = G->x_size ;
    #ifdef GB_MEMDUMP
    printf ("remove G->x from memtable: %p\n", G->x) ;
    #endif
    GB_Global_memtable_remove (G->x) ;
    G->x = NULL ; G->x_size = 0 ;
    bool G_iso = G->iso  ;            	

    //--------------------------------------------------------------------------
    // change G to boolean (all true and iso)
    //--------------------------------------------------------------------------

    // Tim: use G as structural instead
    bool Gbool = true ;        							
    G->type = GrB_BOOL ;       	             	 	                 	
    G->x = &Gbool ;            		 	 	 	 	 	
    G->iso = true ;            		 	 	 	 	 	
    G->x_shallow = true ;      		 	 	 	 	 	
    G->x_size = sizeof (bool) ; 						

    //--------------------------------------------------------------------------
    // K = structure of M, where the kth entry in K is equal to k
    //--------------------------------------------------------------------------

    // K is a shallow copy of M, except for its numerical values
    struct GB_Matrix_opaque K_header ;
    GrB_Matrix K = GB_clear_static_header (&K_header) ;

    OK (GB_shallow_copy (K, GxB_BY_COL, M, NULL)) ;
    OK (GxB_Matrix_Option_get (K, GxB_SPARSITY_STATUS, &sparsity)) ;
    CHECK_ERROR (sparsity == GxB_BITMAP, "internal error 10") ;

    // Kx = uint64 (0:mnz-1)
    size_t Kx_size = (MAX (mnz, 1) * sizeof (uint64_t)) ;
    uint64_t *Kx = mxMalloc (Kx_size) ;
    GB_helper7 (Kx, mnz) ;

    // add a new K->x to K
    K->x = Kx ;
    K->x_shallow = false ;
    K->type = GrB_UINT64 ;
    K->x_size = Kx_size ;
    #ifdef GB_MEMDUMP
    printf ("add K->x to memtable: %p\n", K->x) ;
    #endif
    GB_Global_memtable_add (K->x, K->x_size) ;
    K->iso = false  ;            	

    //--------------------------------------------------------------------------
    // T<G> = K
    //--------------------------------------------------------------------------

    GrB_Matrix T = gb_new (GrB_UINT64, nrows, ncols, GxB_BY_COL, not_bitmap) ;
    OK1 (T, GxB_Matrix_subassign (T, G, NULL,
        K, GrB_ALL, nrows, GrB_ALL, ncols, NULL)) ;

    //--------------------------------------------------------------------------
    // extract Tx, the values of T
    //--------------------------------------------------------------------------

    GrB_Index tnvals ;
    OK1 (T, GrB_Matrix_wait (T, GrB_MATERIALIZE)) ;
    OK (GrB_Matrix_nvals (&tnvals, T)) ;
    uint64_t *Tx = T->x ;
    size_t Tx_size = T->x_size ;
    #ifdef GB_MEMDUMP
    printf ("remove T->x from memtable: %p\n", T->x) ;
    #endif
    GB_Global_memtable_remove (T->x) ;
    T->x = NULL ; T->x_size = 0 ;

    // gnvals and tnvals are identical, by construction
    CHECK_ERROR (gnvals != tnvals, "internal error 1") ;

    //--------------------------------------------------------------------------
    // construct the result C
    //--------------------------------------------------------------------------

    // Vectors are always stored by column, and are never hypersparse.  This
    // step takes constant time, using a transplant of the row indices Tx from
    // T and the values Gx from G.  V is sparse (not full, not hypersparse).

    GrB_Vector V ;
    OK (GrB_Vector_new (&V, type, mnz)) ;
    OK (GxB_Vector_Option_set (V, GxB_SPARSITY_CONTROL, GxB_SPARSE)) ;

    #ifdef GB_MEMDUMP
    printf ("remove V->i from memtable: %p\n", V->i) ;
    printf ("remove V->x from memtable: %p\n", V->x) ;
    #endif
    GB_Global_memtable_remove (V->i) ;
    gb_mxfree ((void **) (&V->i)) ;
    GB_Global_memtable_remove (V->x) ;
    gb_mxfree ((void **) (&V->x)) ;

    // transplant values of T as the row indices of V
    V->i = (int64_t *) Tx ;
    V->i_size = Tx_size ;
    V->i_shallow = false ;
    #ifdef GB_MEMDUMP
    printf ("add V->i to memtable: %p\n", V->i) ;
    #endif
    GB_Global_memtable_add (V->i, V->i_size) ;  // this was the old T->x

    // transplant the values of G as the values of V
    V->x = Gx ;
    V->x_size = Gx_size ;
    V->x_shallow = false ;
    V->iso = G_iso  ;            	
    #ifdef GB_MEMDUMP
    printf ("add V->x to memtable: %p\n", V->x) ;
    #endif
    GB_Global_memtable_add (V->x, V->x_size) ;  // this was the old G->x

    int64_t *Vp = V->p ;
    Vp [0] = 0 ;
    Vp [1] = tnvals ;
    V->nvals = tnvals ;
    V->magic = GB_MAGIC ;
    V->nvec_nonempty = (tnvals > 0) ? 1 : 0 ;

    // typecast V to a matrix C, for export
    GrB_Matrix C = (GrB_Matrix) V ;
    V = NULL ;

    //--------------------------------------------------------------------------
    // free shallow copies and temporary matrices
    //--------------------------------------------------------------------------

    OK (GrB_Matrix_free (&G)) ;
    OK (GrB_Matrix_free (&K)) ;
    OK (GrB_Matrix_free (&T)) ;
    OK (GrB_Matrix_free (&M)) ;

    //--------------------------------------------------------------------------
    // export the output matrix C as a GraphBLAS matrix
    //--------------------------------------------------------------------------

    pargout [0] = gb_export (&C, KIND_GRB) ;
    GB_WRAPUP ;
}

