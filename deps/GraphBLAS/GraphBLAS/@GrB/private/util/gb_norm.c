//------------------------------------------------------------------------------
// gb_norm: compute the norm of a GraphBLAS matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "gb_interface.h"

double gb_norm              // compute norm (A,kind)
(
    GrB_Matrix A,
    int64_t norm_kind       // 0, 1, 2, INT64_MAX, or INT64_MIN
)
{

    //--------------------------------------------------------------------------
    // get input matrix, select types and operators, and allocate X
    //--------------------------------------------------------------------------

    GrB_Index nrows, ncols, nvals ;
    OK (GrB_Matrix_nvals (&nvals, A)) ;
    if (nvals == 0) return ((double) 0) ;

    GrB_Type atype, xtype ;
    OK (GrB_Matrix_nrows (&nrows, A)) ;
    OK (GrB_Matrix_ncols (&ncols, A)) ;
    OK (GxB_Matrix_type (&atype, A)) ;

    GrB_UnaryOp absop ;
    GrB_Monoid sumop, maxop, minop ;
    GrB_Vector t = NULL ;
    GrB_Matrix X = NULL ;
    bool is_complex = false ;

    if (atype == GrB_FP32)
    { 
        // if A is FP32, use the FP32 type and operators
        xtype = GrB_FP32 ;
        absop = GrB_ABS_FP32 ;
        sumop = GrB_PLUS_MONOID_FP32 ;
        maxop = GrB_MAX_MONOID_FP32 ;
        minop = GrB_MIN_MONOID_FP32 ;
    }
    else if (atype == GxB_FC32)
    { 
        // if A is FC32, use the FP32/FC32 type and operators
        is_complex = true ;
        xtype = GrB_FP32 ;
        absop = GxB_ABS_FC32 ;
        sumop = GrB_PLUS_MONOID_FP32 ;
        maxop = GrB_MAX_MONOID_FP32 ;
        minop = GrB_MIN_MONOID_FP32 ;
    }
    else if (atype == GxB_FC64)
    { 
        // if A is FC64, use the FP64/FC64 type and operators
        is_complex = true ;
        xtype = GrB_FP64 ;
        absop = GxB_ABS_FC64 ;
        sumop = GrB_PLUS_MONOID_FP64 ;
        maxop = GrB_MAX_MONOID_FP64 ;
        minop = GrB_MIN_MONOID_FP64 ;
    }
    else
    { 
        // otherwise, use FP64 type and operators; this will typecast the 
        // input matrix to FP64 if A is not of that type.
        xtype = GrB_FP64 ;
        absop = GrB_ABS_FP64 ;
        sumop = GrB_PLUS_MONOID_FP64 ;
        maxop = GrB_MAX_MONOID_FP64 ;
        minop = GrB_MIN_MONOID_FP64 ;
    }

    OK (GrB_Matrix_new (&X, xtype, nrows, ncols)) ;

    //--------------------------------------------------------------------------
    // compute the norm
    //--------------------------------------------------------------------------

    double s = 0 ;

    if (nrows == 1 || ncols == 1 || norm_kind == 0)
    {

        //----------------------------------------------------------------------
        // vector or Frobenius norm
        //----------------------------------------------------------------------

        switch (norm_kind)
        {

            case 0 :    // Frobenius norm
            case 2 :    // 2-norm

                if (is_complex)
                { 
                    // X = abs (A)
                    OK1 (X, GrB_Matrix_apply (X, NULL, NULL, absop, A, NULL)) ;
                    // X = X.^2
                    if (atype == GxB_FC32)
                    {
                        OK1 (X, GrB_Matrix_apply_BinaryOp2nd_FP32 (X, NULL,
                            NULL, GxB_POW_FP32, X, (float) 2.0, NULL)) ;
                    }
                    else
                    {
                        OK1 (X, GrB_Matrix_apply_BinaryOp2nd_FP64 (X, NULL,
                            NULL, GxB_POW_FP64, X, (double) 2.0, NULL)) ;
                    }
                }
                else
                { 
                    // X = A.^2
                    OK1 (X, GrB_Matrix_apply_BinaryOp2nd_FP64 (X, NULL, NULL,
                        GxB_POW_FP64, A, (double) 2.0, NULL)) ;
                }
                // s = sum (X)
                OK (GrB_Matrix_reduce_FP64 (&s, NULL, sumop, X, NULL)) ;
                s = sqrt (s) ;
                break ;

            case 1 :    // 1-norm

                // X = abs (A)
                OK1 (X, GrB_Matrix_apply (X, NULL, NULL, absop, A, NULL)) ;
                // s = sum (X)
                OK (GrB_Matrix_reduce_FP64 (&s, NULL, sumop, X, NULL)) ;
                break ;

            case INT64_MAX :    // inf-norm

                // X = abs (A)
                OK1 (X, GrB_Matrix_apply (X, NULL, NULL, absop, A, NULL)) ;
                // s = max (X)
                OK (GrB_Matrix_reduce_FP64 (&s, NULL, maxop, X, NULL)) ;
                break ;

            case INT64_MIN :    // (-inf)-norm

                if (GB_is_dense (A))
                { 
                    // X = abs (A)
                    OK1 (X, GrB_Matrix_apply (X, NULL, NULL, absop, A, NULL)) ;
                    // s = min (X)
                    OK (GrB_Matrix_reduce_FP64 (&s, NULL, minop, X, NULL)) ;
                }
                break ;

            default:

                ERROR ("unknown norm") ;
                break ;
        }

    }
    else
    {

        //----------------------------------------------------------------------
        // matrix norm
        //----------------------------------------------------------------------

        switch (norm_kind)
        {

            case 2 :    // 2-norm

                ERROR ("2-norm not available for GrB matrices") ;
                break ;

            case 1 :    // 1-norm:  max sum of columns of abs (A)

                // X = abs (A)
                OK1 (X, GrB_Matrix_apply (X, NULL, NULL, absop, A, NULL)) ;
                // t = zeros (ncols,1)
                OK (GrB_Vector_new (&t, xtype, ncols)) ;
                // t(j) = sum of the ith column, X(:,j)
                OK (GrB_Matrix_reduce_Monoid (t, NULL, NULL, sumop, X,
                    GrB_DESC_T0)) ;
                // s = max (t)
                OK (GrB_Vector_reduce_FP64 (&s, NULL, maxop, t, NULL)) ;
                break ;

            case INT64_MAX :    // inf-norm:  max sum of rows of abs (A)

                // X = abs (A)
                OK1 (X, GrB_Matrix_apply (X, NULL, NULL, absop, A, NULL)) ;
                // t = zeros (nrows,1)
                OK (GrB_Vector_new (&t, xtype, nrows)) ;
                // t(i) = sum of the ith row, X(i,:)
                OK (GrB_Matrix_reduce_Monoid (t, NULL, NULL, sumop, X, NULL)) ;
                // s = max (t)
                OK (GrB_Vector_reduce_FP64 (&s, NULL, maxop, t, NULL)) ;
                break ;

            case INT64_MIN :

                ERROR ("(-inf)-norm not available for GrB matrices") ;
                break ;

            default :

                ERROR ("unknown norm") ;
                break ;
        }
    }

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    OK (GrB_Matrix_free (&X)) ;
    OK (GrB_Vector_free (&t)) ;
    return (s) ;
}

