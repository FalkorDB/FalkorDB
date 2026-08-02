//------------------------------------------------------------------------------
// gb_get_matlab_or_grb_matrix: get a MATLAB matrix or GrB value matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// The A->[phybix] content is tagged GxB_IS_READONLY, so the arena doesn't
// matter.  However, all of the content is mxMalloc'd for both the MATLAB
// matrix and GrB value matrix objects, so the data is tagged with the
// MXARENA. The header of A is placed in the arena determined by the input
// parameter.

#undef  FREE_WORK
#define FREE_WORK                       \
    GxB_Container_free (&Container) ;

#undef  FREE_ALL
#define FREE_ALL                        \
    FREE_WORK                           \
    GrB_Matrix_free (&Y) ;              \
    GrB_Matrix_free (&A) ;

GrB_Info gb_get_matlab_or_grb_matrix   // shallow copy of MATLAB or GrB matrix
(
    // output
    GrB_Matrix *A_handle,   // content of A is tagged GxB_IS_READONLY
    // input
    gb_matrix matrix,       // contents of a MATLAB or GrB matrix
    const int arena,
    char err [ERRLEN]
)
{ 

    //--------------------------------------------------------------------------
    // load the content of the matrix into the Container, as readonly
    //--------------------------------------------------------------------------

    GrB_Matrix A = NULL, Y = NULL ;
    GxB_Container Container = NULL ;

    OK (GxB_Container_new_arena (&Container, arena, arena)) ;

    GrB_Type ptype = (matrix->p_is_32) ? GrB_UINT32 : GrB_UINT64 ;
    GrB_Type jtype = (matrix->j_is_32) ? GrB_UINT32 : GrB_UINT64 ;
    GrB_Type itype = (matrix->i_is_32) ? GrB_UINT32 : GrB_UINT64 ;

    size_t psize = (matrix->p_is_32) ? sizeof (uint32_t) : sizeof (uint64_t) ;
    size_t jsize = (matrix->j_is_32) ? sizeof (uint32_t) : sizeof (uint64_t) ;
    size_t isize = (matrix->i_is_32) ? sizeof (uint32_t) : sizeof (uint64_t) ;

    if (matrix->Yp != NULL)
    { 
        // import the A->Y matrix using the Container
        OK (GrB_Matrix_new (&Y, GrB_UINT64, 0, 0)) ;
        Container->nrows = matrix->ynrows ;
        Container->ncols = matrix->yncols ;
        Container->nrows_nonempty = -1 ;
        Container->ncols_nonempty = -1 ;
        Container->nvals = matrix->nvec ;
        Container->format = GxB_SPARSE ;
        Container->orientation = GrB_COLMAJOR ;
        Container->iso = false ;
        Container->jumbled = false ;
        OK (GxB_Vector_load (Container->p, &(matrix->Yp), jtype,
            matrix->yncols + 1, (matrix->yncols + 1) * jsize,
            GxB_IS_READONLY + MXARENA, NULL)) ;
        OK (GxB_Vector_load (Container->i, &(matrix->Yi), jtype, matrix->nvec,
            (matrix->nvec * jsize), GxB_IS_READONLY + MXARENA, NULL)) ;
        OK (GxB_Vector_load (Container->x, &(matrix->Yx), jtype, matrix->nvec,
            (matrix->nvec * jsize), GxB_IS_READONLY + MXARENA, NULL)) ;
        OK (GxB_load_Matrix_from_Container (Y, Container, NULL)) ;
    }

    // import the A matrix using the Container
    Container->nrows = matrix->nrows ;
    Container->ncols = matrix->ncols ;
    Container->nvals = matrix->nvals ;
    Container->nrows_nonempty = (matrix->by_col) ? -1 : matrix->nvec_nonempty ;
    Container->ncols_nonempty = (matrix->by_col) ? matrix->nvec_nonempty : -1 ;
    Container->format = matrix->sparsity ;
    Container->orientation = (matrix->by_col) ? GrB_COLMAJOR : GrB_ROWMAJOR ;
    Container->iso = matrix->iso ;
    Container->jumbled = false ;

    int64_t x_len = 0 ;

    switch (matrix->sparsity)
    {
        case GxB_HYPERSPARSE : 

            Container->Y = Y ;
            Y = NULL ;
            OK (GxB_Vector_load (Container->h, &(matrix->h), jtype,
                matrix->plen, matrix->plen * jsize,
                GxB_IS_READONLY + MXARENA, NULL)) ;
            // fall through to sparse case

        case GxB_SPARSE : 

            OK (GxB_Vector_load (Container->p, &(matrix->p), ptype,
                matrix->plen + 1, (matrix->plen + 1) * psize,
                GxB_IS_READONLY + MXARENA, NULL)) ;
            OK (GxB_Vector_load (Container->i, &(matrix->i), itype,
                matrix->nvals, matrix->nvals * isize,
                GxB_IS_READONLY + MXARENA, NULL)) ;
            x_len = matrix->nvals ;
            break ;

        case GxB_BITMAP : 

            OK (GxB_Vector_load (Container->b, &(matrix->b), GrB_INT8,
                matrix->nrows * matrix->ncols,
                matrix->nrows * matrix->ncols * sizeof (uint8_t),
                GxB_IS_READONLY + MXARENA, NULL)) ;
            // fall through to full case

        case GxB_FULL : 
            x_len = matrix->nrows * matrix->ncols ;
            break ;

        default: ;
    }

    if (matrix->iso)
    { 
        x_len = 1 ;
    }

    OK (GxB_Vector_load (Container->x, &(matrix->x), matrix->type,
        x_len, x_len * matrix->typesize, GxB_IS_READONLY + MXARENA, NULL)) ;

    //--------------------------------------------------------------------------
    // unload the Container into A
    //--------------------------------------------------------------------------

    OK (GxB_Matrix_new_arena (&A, GrB_BOOL, 0, 0, arena, arena)) ;
    OK (GxB_load_Matrix_from_Container (A, Container, NULL)) ;

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    FREE_WORK ;
    (*A_handle) = A ;
    return (GrB_SUCCESS) ;
}

#undef  FREE_WORK
#define FREE_WORK
#undef  FREE_ALL
#define FREE_ALL

