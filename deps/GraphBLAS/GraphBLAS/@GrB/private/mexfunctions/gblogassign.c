//------------------------------------------------------------------------------
// gblogassign: logical assignment: C(M) = A
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// gblogassign computes the built-in logical indexing expression C(M) = A.  The
// matrices C and M must be the same size.  M is normally logical but it can be
// of any type in this mexFunction.  M should not have any explicit zeros.  A
// is a sparse vector of size nnz(M)-by-1.  Scalar expansion is not handled.
// Use GrB.subassign (C, M, scalar) for that case.

// Usage:

//      C = gblogassign (C, M, A)

//  This function is the C equivalent of the following m-function:

/*

    function C = gblogassign (C, M_input, A)
    % Computing the built-in logical indexing expression C(M) = A in GraphBLAS.
    % A is a sparse vector of size nnz(M)-by-1 (scalar expansion is not
    % handled). M is normally a sparse logical matrix, either GraphBLAS or
    % built-in, but it can be of any type.  C and M have the same size.

    % make sure all matrices are stored by column
    save = GrB.format ;
    GrB.format ('by col') ;
    M = GrB (m, n, 'logical') ;
    M = GrB.select (M, '2nd', 'nonzero', M_input) ;
    if (isequal (GrB.format (A), 'by row'))
        A = GrB (A) ;
    end

    [m n] = size (C) ;
    mnz = nnz (M) ;         % A must be mnz-by-1
    if (~isequal (size (A), [mnz 1]))
        error ('GrB:error', 'A must be nnz(M)-by-1')
    end

    [ai,  ~, ax] = GrB.extracttuples (A) ;
    [mi, mj,  ~] = GrB.extracttuples (M) ;

    % construct a subset of the entries of the mask M corresponding to the
    % entries in A
    si = mi (ai) ;
    sj = mj (ai) ;
    S = GrB.build (si, sj, ax, m, n) ;

    GrB.format (save) ;

    % C<M> = S
    C = GrB.subassign (C, M, S) ;

*/

// This C mexFunction is faster than the above m-function, since it avoids the
// use of GrB.extracttuples.  Instead, it accesses the internal structure of the
// GrB_Matrix objects.  The m-file above is useful for understanding that this
// C mexFunction does.

// C is always returned as a GrB matrix.

#include "gb_interface.h"

#define USAGE "usage: C = gblogassign (C, M, A)"
#define ERR "A must be a vector of length nnz(M) for logical indexing, C(M)=A"

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

    gb_usage (nargin == 3 && nargout <= 1, USAGE) ;

    //--------------------------------------------------------------------------
    // get a deep copy of C, of any sparsity structure
    //--------------------------------------------------------------------------

    GrB_Matrix C = gb_get_deep (pargin [0]) ;
    GrB_Index nrows, ncols ;
    OK (GrB_Matrix_nrows (&nrows, C)) ;
    OK (GrB_Matrix_ncols (&ncols, C)) ;

    //--------------------------------------------------------------------------
    // get M
    //--------------------------------------------------------------------------

    // make M boolean, sparse/hyper, stored by column, and drop explicit zeros
    GrB_Matrix M_input = gb_get_shallow (pargin [1]) ;
    GrB_Matrix M = gb_new (GrB_BOOL, nrows, ncols, GxB_BY_COL,
        GxB_SPARSE + GxB_HYPERSPARSE) ;
//  OK1 (M, GxB_Matrix_select (M, NULL, NULL, GxB_NONZERO, M_input,
//      NULL, NULL)) ;
    OK1 (M, GrB_Matrix_select_BOOL (M, NULL, NULL, GrB_VALUENE_BOOL, M_input,
        0, NULL)) ;

    OK (GrB_Matrix_free (&M_input)) ;
    GrB_Index mnz ;
    OK (GrB_Matrix_nvals (&mnz, M)) ;

    //--------------------------------------------------------------------------
    // get A
    //--------------------------------------------------------------------------

    GrB_Matrix A_input = gb_get_shallow (pargin [2]) ;
    GrB_Matrix A = A_input ;
    GrB_Type atype ;
    GrB_Index anrows, ancols, anz ;
    GxB_Format_Value fmt ;
    int A_sparsity ;
    OK (GrB_Matrix_nrows (&anrows, A)) ;
    OK (GrB_Matrix_ncols (&ancols, A)) ;
    OK (GxB_Matrix_type (&atype, A)) ;
    OK (GrB_Matrix_nvals (&anz, A)) ;
    OK (GxB_Matrix_Option_get (A, GxB_FORMAT, &fmt)) ;
    OK (GxB_Matrix_Option_get (A, GxB_SPARSITY_STATUS, &A_sparsity)) ;

    GrB_Matrix A_copy = NULL ;
    GrB_Matrix A_copy2 = NULL ;

    // make sure A is not bitmap; it can be sparse, hypersparse, or full
    if (A_sparsity == GxB_BITMAP)
    {
        OK (GrB_Matrix_dup (&A_copy2, A)) ;
        OK1 (A_copy2, GxB_Matrix_Option_set (A_copy2, GxB_SPARSITY_CONTROL,
            GxB_SPARSE + GxB_HYPERSPARSE + GxB_FULL)) ;
        A = A_copy2 ;
    }

    // make sure A is a vector of the right size
    if (mnz == 0)
    { 
        // M is empty, so A must have no entries.  The dimensions and format of
        // A are not relevant, since the content of A will not be accessed.
        CHECK_ERROR (anz != 0, ERR) ;
    }
    else if (anrows == 1)
    {
        // A is 1-by-ancols; ensure it is has length nnz(M), and held by row,
        // or transpose to ancols-by-1 and held by column.
        CHECK_ERROR (ancols != mnz, ERR) ;
        if (fmt == GxB_BY_COL)
        { 
            // A is 1-by-ancols and held by column: transpose it
            A_copy = gb_new (atype, mnz, 1, GxB_BY_COL, 
                GxB_SPARSE + GxB_HYPERSPARSE + GxB_FULL) ;
            OK1 (A_copy, GrB_transpose (A_copy, NULL, NULL, A, NULL)) ;
            OK1 (A_copy, GrB_Matrix_wait (A_copy, GrB_MATERIALIZE)) ;
            A = A_copy ;
        }
    }
    else if (ancols == 1)
    {
        // A is anrows-by-1; ensure it is has length nnz(M), and held by col
        // or transpose to 1-by-anrows and held by row.
        CHECK_ERROR (anrows != mnz, ERR) ;
        if (fmt == GxB_BY_ROW)
        { 
            // A is anrows-by-1 and held by row: transpose it
            A_copy = gb_new (atype, 1, mnz, GxB_BY_ROW,
                GxB_SPARSE + GxB_HYPERSPARSE + GxB_FULL) ;
            OK1 (A_copy, GrB_transpose (A_copy, NULL, NULL, A, NULL)) ;
            OK1 (A_copy, GrB_Matrix_wait (A_copy, GrB_MATERIALIZE)) ;
            A = A_copy ;
        }
    }
    else
    {
        ERROR (ERR) ;
    }

    //--------------------------------------------------------------------------
    // extract the values and pattern of A; handle iso case
    //--------------------------------------------------------------------------

    // Tim: use a shallow variant of GxB*export to access content of M and A
    GrB_Index *Ai =            							
        (GrB_Index *) A->i ;   	             	 	                 	
    void *Ax = A->x ;          		 	 	 	 	 	
    char nil [16] =            		 	 	 	 	 	
     "iso logassign  " ;       		 	 	 	 	 	
    if (Ax == NULL) Ax = &nil ;							

    //--------------------------------------------------------------------------
    // extract the pattern of M
    //--------------------------------------------------------------------------

    GrB_Index *Mi = (GrB_Index *) (M->i) ;
    GrB_Index *Mj = mxMalloc (MAX (mnz, 1) * sizeof (GrB_Index)) ;
    OK (GrB_Matrix_extractTuples_BOOL (NULL, Mj, NULL, &mnz, M)) ;

    //--------------------------------------------------------------------------
    // construct a subset of the pattern of M corresponding to the entries of A
    //--------------------------------------------------------------------------

    GrB_Index *Si = mxMalloc (MAX (anz, 1) * sizeof (GrB_Index)) ;
    GrB_Index *Sj = mxMalloc (MAX (anz, 1) * sizeof (GrB_Index)) ;
    GB_helper5 (Si, Sj, Mi, Mj, M->vlen, Ai, A->vlen, anz) ;
    GrB_Matrix S = gb_new (atype, nrows, ncols, GxB_BY_COL, 0) ;

    if (A->iso)
    {
        // build S as an iso matrix
        GrB_Scalar s = NULL ;
        OK (GrB_Scalar_new (&s, atype)) ;
        if (atype == GrB_BOOL)
        { 
            OK (GrB_Scalar_setElement_BOOL (s, (* ((bool *) Ax)))) ;
        }
        else if (atype == GrB_INT8)
        { 
            OK (GrB_Scalar_setElement_INT8 (s, (* ((int8_t *) Ax)))) ;
        }
        else if (atype == GrB_INT16)
        { 
            OK (GrB_Scalar_setElement_INT16 (s, (* ((int16_t *) Ax)))) ;
        }
        else if (atype == GrB_INT32)
        { 
            OK (GrB_Scalar_setElement_INT32 (s, (* ((int32_t *) Ax)))) ;
        }
        else if (atype == GrB_INT64)
        { 
            OK (GrB_Scalar_setElement_INT64 (s, (* ((int64_t *) Ax)))) ;
        }
        else if (atype == GrB_UINT8)
        { 
            OK (GrB_Scalar_setElement_UINT8 (s, (* ((uint8_t *) Ax)))) ;
        }
        else if (atype == GrB_UINT16)
        { 
            OK (GrB_Scalar_setElement_UINT16 (s, (* ((uint16_t *) Ax)))) ;
        }
        else if (atype == GrB_UINT32)
        { 
            OK (GrB_Scalar_setElement_UINT32 (s, (* ((uint32_t *) Ax)))) ;
        }
        else if (atype == GrB_UINT64)
        { 
            OK (GrB_Scalar_setElement_UINT64 (s, (* ((uint64_t *) Ax)))) ;
        }
        else if (atype == GrB_FP32)
        { 
            OK (GrB_Scalar_setElement_FP32 (s, (* ((float *) Ax)))) ;
        }
        else if (atype == GrB_FP64)
        { 
            OK (GrB_Scalar_setElement_FP64 (s, (* ((double *) Ax)))) ;
        }
        else if (atype == GxB_FC32)
        { 
            OK (GxB_Scalar_setElement_FC32 (s, (* ((GxB_FC32_t *) Ax)))) ;
        }
        else if (atype == GxB_FC64)
        { 
            OK (GxB_Scalar_setElement_FC64 (s, (* ((GxB_FC64_t *) Ax)))) ;
        }
        else
        {
            ERROR ("unsupported type") ;
        }
        OK1 (S, GxB_Matrix_build_Scalar (S, Si, Sj, s, anz)) ;
        OK (GrB_Scalar_free (&s)) ;
    }
    else if (atype == GrB_BOOL)
    { 
        OK1 (S, GrB_Matrix_build_BOOL (S, Si, Sj, Ax, anz, GrB_LOR)) ;
    }
    else if (atype == GrB_INT8)
    { 
        OK1 (S, GrB_Matrix_build_INT8 (S, Si, Sj, Ax, anz, GrB_PLUS_INT8)) ;
    }
    else if (atype == GrB_INT16)
    { 
        OK1 (S, GrB_Matrix_build_INT16 (S, Si, Sj, Ax, anz, GrB_PLUS_INT16)) ;
    }
    else if (atype == GrB_INT32)
    { 
        OK1 (S, GrB_Matrix_build_INT32 (S, Si, Sj, Ax, anz, GrB_PLUS_INT32)) ;
    }
    else if (atype == GrB_INT64)
    { 
        OK1 (S, GrB_Matrix_build_INT64 (S, Si, Sj, Ax, anz, GrB_PLUS_INT64)) ;
    }
    else if (atype == GrB_UINT8)
    { 
        OK1 (S, GrB_Matrix_build_UINT8 (S, Si, Sj, Ax, anz, GrB_PLUS_UINT8)) ;
    }
    else if (atype == GrB_UINT16)
    { 
        OK1 (S, GrB_Matrix_build_UINT16 (S, Si, Sj, Ax, anz, GrB_PLUS_UINT16)) ;
    }
    else if (atype == GrB_UINT32)
    { 
        OK1 (S, GrB_Matrix_build_UINT32 (S, Si, Sj, Ax, anz, GrB_PLUS_UINT32)) ;
    }
    else if (atype == GrB_UINT64)
    { 
        OK1 (S, GrB_Matrix_build_UINT64 (S, Si, Sj, Ax, anz, GrB_PLUS_UINT64)) ;
    }
    else if (atype == GrB_FP32)
    { 
        OK1 (S, GrB_Matrix_build_FP32 (S, Si, Sj, Ax, anz, GrB_PLUS_FP32)) ;
    }
    else if (atype == GrB_FP64)
    { 
        OK1 (S, GrB_Matrix_build_FP64 (S, Si, Sj, Ax, anz, GrB_PLUS_FP64)) ;
    }
    else if (atype == GxB_FC32)
    { 
        OK1 (S, GxB_Matrix_build_FC32 (S, Si, Sj, Ax, anz, GxB_PLUS_FC32)) ;
    }
    else if (atype == GxB_FC64)
    { 
        OK1 (S, GxB_Matrix_build_FC64 (S, Si, Sj, Ax, anz, GxB_PLUS_FC64)) ;
    }
    else
    {
        ERROR ("unsupported type") ;
    }

    OK (GrB_Matrix_free (&A_copy)) ;
    OK (GrB_Matrix_free (&A_copy2)) ;

    //--------------------------------------------------------------------------
    // C<M> = S
    //--------------------------------------------------------------------------

    OK1 (C, GxB_Matrix_subassign (C, M, NULL,
        S, GrB_ALL, nrows, GrB_ALL, ncols, NULL)) ;

    //--------------------------------------------------------------------------
    // free shallow copies and temporary matrices
    //--------------------------------------------------------------------------

    // OK: Si, Sj, and Mj were allocated above from mxMalloc, never in a
    // GrB_Matrix
    gb_mxfree ((void **) (&Si)) ;
    gb_mxfree ((void **) (&Sj)) ;
    gb_mxfree ((void **) (&Mj)) ;
    OK (GrB_Matrix_free (&S)) ;
    OK (GrB_Matrix_free (&M)) ;
    OK (GrB_Matrix_free (&A_input)) ;

    //--------------------------------------------------------------------------
    // export the output matrix C as a GraphBLAS matrix
    //--------------------------------------------------------------------------

    pargout [0] = gb_export (&C, KIND_GRB) ;
    GB_WRAPUP ;
}

