//------------------------------------------------------------------------------
// GraphBLAS/Demo/Include/wathen.c: a finite-element matrix on a regular mesh
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Create a finite-element matrix on an nx-by-ny 2D mesh, as computed by
// wathen.m.

#include "GraphBLAS.h"
#include "graphblas_demos.h"

//------------------------------------------------------------------------------
// scale by rho
//------------------------------------------------------------------------------

double r = 0 ;
void rho_scale (double *f, const double *e)
{
    (*f) = r * (*e) ;
}

//------------------------------------------------------------------------------
// Wathen function
//------------------------------------------------------------------------------

GrB_Info wathen             // construct a random Wathen matrix
(
    GrB_Matrix *A_output,   // output matrix
    int64_t nx,             // grid dimension nx
    int64_t ny,             // grid dimension ny
    bool scale,             // if true, scale the rows
    int method,             // 0 to 3
    double *rho_given       // nx-by-ny dense matrix, if NULL use random rho
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    if (nx < 0 || ny < 0 || A_output == NULL || method < 0 || method > 3)
    {
        return (GrB_INVALID_VALUE) ;
    }

    // macro to free all workspace.  Not every method uses every object
    #undef  FREE_ALL
    #define FREE_ALL                            \
        GrB_Matrix_free (&A) ;                  \
        GrB_Matrix_free (&F) ;                  \
        GrB_Matrix_free (&D) ;                  \
        GrB_Matrix_free (&E) ;                  \
        GrB_UnaryOp_free (&rho_op) ;            \
        if (rho_rand != NULL) free (rho_rand) ; \
        if (I != NULL) free (I) ;               \
        if (J != NULL) free (J) ;               \
        if (X != NULL) free (X) ;

    GrB_Info info ;
    GrB_Matrix A = NULL, F = NULL, E = NULL, D = NULL ;
    GrB_UnaryOp rho_op = NULL ;
    double *rho_rand = NULL, *X = NULL, *rho ;
    GrB_Index *I = NULL, *J = NULL ;

    //--------------------------------------------------------------------------
    // construct the coefficients
    //--------------------------------------------------------------------------

    #define d ((double) 45),
    static const double e [8][8] = {
        {  6/d   -6/d    2/d   -8/d    3/d   -8/d    2/d   -6/d   },
        { -6/d   32/d   -6/d   20/d   -8/d   16/d   -8/d   20/d   },
        {  2/d   -6/d    6/d   -6/d    2/d   -8/d    3/d   -8/d   },
        { -8/d   20/d   -6/d   32/d   -6/d   20/d   -8/d   16/d   },
        {  3/d   -8/d    2/d   -6/d    6/d   -6/d    2/d   -8/d   },
        { -8/d   16/d   -8/d   20/d   -6/d   32/d   -6/d   20/d   },
        {  2/d   -8/d    3/d   -8/d    2/d   -6/d    6/d   -6/d   },
        { -6/d   20/d   -8/d   16/d   -8/d   20/d   -6/d   32/d   } } ;

    //--------------------------------------------------------------------------
    // A = sparse (n,n) ;
    //--------------------------------------------------------------------------

    int64_t n = 3*nx*ny + 2*nx + 2*ny +1 ;
    OK (GrB_Matrix_new (&A, GrB_FP64, n, n)) ;

    //--------------------------------------------------------------------------
    // RHO = 100 * rand (nx,ny) ;
    //--------------------------------------------------------------------------

    // i and j are 1-based, so the same index computations from wathen.m
    // can be used
    #define RHO(i,j) rho [(i-1)+((j-1)*nx)]

    if (rho_given == NULL)
    {
        // compute a random RHO matrix
        rho_rand = (double *) malloc (nx * ny * sizeof (double)) ;
        if (rho_rand == NULL)
        {   // out of memory
            FREE_ALL ;
            return (GrB_OUT_OF_MEMORY) ;
        }
        rho = rho_rand ;
        for (int j = 1 ; j <= ny ; j++)
        {
            for (int i = 1 ; i <= nx ; i++)
            {
                RHO (i,j) = 100 * simple_rand_x ( ) ;
            }
        }
    }
    else
    {
        // use rho_given on input
        rho = rho_given ;
    }

    #define em(krow,kcol) (e [krow][kcol] * RHO (i,j))

    //--------------------------------------------------------------------------
    // nn = zeros (8,1) ;
    //--------------------------------------------------------------------------

    GrB_Index nn [8] ;

    //--------------------------------------------------------------------------
    // construct the Wathen matrix, using one of four equivalent methods
    //--------------------------------------------------------------------------

    switch (method)
    {

        //----------------------------------------------------------------------
        // create tuples and use build, just like wathen.m
        //----------------------------------------------------------------------

        case 0:
        {
            // This method is fastest of the 4 methods here.  The code here is
            // nearly identical to the wathen.m M-file, except that here an
            // adjustment to the indices must be made since GraphBLAS matrices
            // are indexed starting at row and column 0, not 1.  It requires
            // more code on the part of the user application, however, as
            // compared to methods 1, 2, and 3.

            // allocate the tuples
            int64_t ntriplets = nx*ny*64 ;
            I = (GrB_Index *) malloc (ntriplets * sizeof (GrB_Index)) ;
            J = (GrB_Index *) malloc (ntriplets * sizeof (GrB_Index)) ;
            X = (double *) malloc (ntriplets * sizeof (double )) ;
            if (I == NULL || J == NULL || X == NULL)
            {   // out of memory
                FREE_ALL ;
                return (GrB_OUT_OF_MEMORY) ;
            }

            ntriplets = 0 ;

            for (int j = 1 ; j <= ny ; j++)
            {
                for (int i = 1 ; i <= nx ; i++)
                {
                    nn [0] = 3*j*nx + 2*i + 2*j + 1 ;
                    nn [1] = nn [0] - 1 ;
                    nn [2] = nn [1] - 1 ;
                    nn [3] = (3*j-1)*nx + 2*j + i - 1 ;
                    nn [4] = 3*(j-1)*nx + 2*i + 2*j - 3 ;
                    nn [5] = nn [4] + 1 ;
                    nn [6] = nn [5] + 1 ;
                    nn [7] = nn [3] + 1 ;
                    for (int krow = 0 ; krow < 8 ; krow++) nn [krow]-- ;

                    for (int krow = 0 ; krow < 8 ; krow++)
                    {
                        for (int kcol = 0 ; kcol < 8 ; kcol++)
                        {
                            I [ntriplets] = nn [krow] ;
                            J [ntriplets] = nn [kcol] ;
                            X [ntriplets] = em (krow,kcol) ;
                            ntriplets++ ;
                        }
                    }
                }
            }

            // A = sparse (I,J,X,n,n) ;
            OK (GrB_Matrix_build_FP64 (A, I, J, X, ntriplets, GrB_PLUS_FP64)) ;

        }
        break ;

        //----------------------------------------------------------------------
        // scalar assignment
        //----------------------------------------------------------------------

        case 1:
        {
            // This method is the simplest, and only takes about 2x the time as
            // method 0.  It would be impossibly slow in the equivalent MATLAB.

            for (int j = 1 ; j <= ny ; j++)
            {
                for (int i = 1 ; i <= nx ; i++)
                {
                    nn [0] = 3*j*nx + 2*i + 2*j + 1 ;
                    nn [1] = nn [0] - 1 ;
                    nn [2] = nn [1] - 1 ;
                    nn [3] = (3*j-1)*nx + 2*j + i - 1 ;
                    nn [4] = 3*(j-1)*nx + 2*i + 2*j - 3 ;
                    nn [5] = nn [4] + 1 ;
                    nn [6] = nn [5] + 1 ;
                    nn [7] = nn [3] + 1 ;
                    for (int krow = 0 ; krow < 8 ; krow++) nn [krow]-- ;

                    for (int krow = 0 ; krow < 8 ; krow++)
                    {
                        for (int kcol = 0 ; kcol < 8 ; kcol++)
                        {
                            // A (nn[krow],nn[kcol]) += em (krow,kcol)
                            OK (GrB_Matrix_assign_FP64 (A, NULL,
                                GrB_PLUS_FP64, em (krow,kcol),
                                (&nn [krow]), 1, (&nn [kcol]), 1, NULL)) ;
                        }
                    }
                }
            }
        }
        break ;

        //----------------------------------------------------------------------
        // matrix assignment, create F one entry at a time
        //----------------------------------------------------------------------

        case 2:
        {
            // This method constructs F and then assigns it all at once into A.
            // It is about 2x to 3x slower than method 1.

            // create a single 8-by-8 finite-element matrix F
            OK (GrB_Matrix_new (&F, GrB_FP64, 8, 8)) ;

            for (int j = 1 ; j <= ny ; j++)
            {
                for (int i = 1 ; i <= nx ; i++)
                {
                    nn [0] = 3*j*nx + 2*i + 2*j + 1 ;
                    nn [1] = nn [0] - 1 ;
                    nn [2] = nn [1] - 1 ;
                    nn [3] = (3*j-1)*nx + 2*j + i - 1 ;
                    nn [4] = 3*(j-1)*nx + 2*i + 2*j - 3 ;
                    nn [5] = nn [4] + 1 ;
                    nn [6] = nn [5] + 1 ;
                    nn [7] = nn [3] + 1 ;
                    for (int krow = 0 ; krow < 8 ; krow++) nn [krow]-- ;

                    for (int krow = 0 ; krow < 8 ; krow++)
                    {
                        for (int kcol = 0 ; kcol < 8 ; kcol++)
                        {
                            // F (krow,kcol) = em (krow, kcol)
                            OK (GrB_Matrix_setElement_FP64 (F,
                                em (krow,kcol), krow, kcol)) ;
                        }
                    }

                    // A (nn,nn) += F
                    OK (GrB_Matrix_assign (A, NULL, GrB_PLUS_FP64,
                        F, nn, 8, nn, 8, NULL)) ;
                }
            }
        }
        break ;

        //----------------------------------------------------------------------
        // matrix assignment, create F all at once
        //----------------------------------------------------------------------

        case 3:
        {
            // This method is as fast as method 2 (that is, 2x to 3x slower
            // than method 1).  It is very flexible since any method can be
            // used to construct the finite-element matrix.  Then A(nn,nn)+=F
            // is very efficient when F is a matrix.

            // create a single 8-by-8 finite-element matrix F
            OK (GrB_Matrix_new (&F, GrB_FP64, 8, 8)) ;

            // create a single 8-by-8 coefficient matrix E
            OK (GrB_Matrix_new (&E, GrB_FP64, 8, 8)) ;
            for (int krow = 0 ; krow < 8 ; krow++)
            {
                for (int kcol = 0 ; kcol < 8 ; kcol++)
                {
                    double ex = e [krow][kcol] ;
                    OK (GrB_Matrix_setElement_FP64 (E, ex, krow, kcol)) ;
                }
            }

            // create a unary operator to scale by RHO(i,j)
            OK (GrB_UnaryOp_new (&rho_op, 
                (GxB_unary_function) rho_scale, GrB_FP64, GrB_FP64)) ;

            for (int j = 1 ; j <= ny ; j++)
            {
                for (int i = 1 ; i <= nx ; i++)
                {
                    nn [0] = 3*j*nx + 2*i + 2*j + 1 ;
                    nn [1] = nn [0] - 1 ;
                    nn [2] = nn [1] - 1 ;
                    nn [3] = (3*j-1)*nx + 2*j + i - 1 ;
                    nn [4] = 3*(j-1)*nx + 2*i + 2*j - 3 ;
                    nn [5] = nn [4] + 1 ;
                    nn [6] = nn [5] + 1 ;
                    nn [7] = nn [3] + 1 ;
                    for (int krow = 0 ; krow < 8 ; krow++) nn [krow]-- ;

                    // F = E * RHO(i,j)
                    // note that this computation on F does not force
                    // A to be assembled.
                    r = RHO (i,j) ;
                    OK (GrB_Matrix_apply (F, NULL, NULL, rho_op, E, NULL)) ;

                    // A (nn,nn) += F
                    OK (GrB_Matrix_assign (A, NULL, GrB_PLUS_FP64,
                        F, nn, 8, nn, 8, NULL)) ;
                }
            }
        }
        break ;

        default:
            CHECK (false, GrB_INVALID_VALUE) ;
            break ;
    }

    //--------------------------------------------------------------------------
    // scale the matrix, if requested
    //--------------------------------------------------------------------------

    // Another way to multiply by the inverse of the diagonal would be to
    // compute A=A/D using the PLUS_DIV_FP64 semiring, which scales the columns
    // instead of the rows, and then transposing the result, since A is
    // symmetric but D\A and A/D are not.  Alternatively, a user-defined
    // operator z=f(x,y) that computes z=y/x could be used, along with a
    // user-defined semiring.

    if (scale)
    {
        // D = sparse (n,n)
        OK (GrB_Matrix_new (&D, GrB_FP64, n, n)) ;
        for (int64_t i = 0 ; i < n ; i++)
        {
            // D (i,i) = 1 / A (i,i) ;
            double di ;
            OK (GrB_Matrix_extractElement_FP64 (&di, A, i, i)) ;
            OK (GrB_Matrix_setElement_FP64 (D, 1/di, i, i)) ;
        }
        // A = D*A
        OK (GrB_mxm (A, NULL, NULL, GxB_PLUS_TIMES_FP64, D, A, NULL)) ;
    }

    // force completion
    OK (GrB_Matrix_wait (A, GrB_MATERIALIZE)) ;

    //--------------------------------------------------------------------------
    // free workspace and return the result
    //--------------------------------------------------------------------------

    *A_output = A ;
    A = NULL ;
    FREE_ALL ;
    return (GrB_SUCCESS) ;
}

