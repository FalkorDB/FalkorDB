//------------------------------------------------------------------------------
// LAGraph_MMWrite:  write a matrix to a Matrix Market file
//------------------------------------------------------------------------------

// LAGraph, (c) 2019-2022 by The LAGraph Contributors, All Rights Reserved.
// SPDX-License-Identifier: BSD-2-Clause
//
// For additional details (including references to third party source code and
// other files) see the LICENSE file or contact permission@sei.cmu.edu. See
// Contributors.txt for a full list of contributors. Created, in part, with
// funding and support from the U.S. Government (see Acknowledgments.txt file).
// DM22-0790

// Contributed by Timothy A. Davis, Texas A&M University

//------------------------------------------------------------------------------

// LAGraph_MMWrite:  write a matrix to a Matrix Market file.

// Writes a matrix to a file in the Matrix Market format.  See LAGraph_MMRead
// for a description of the format.

// The Matrix Market format is described at:
// https://math.nist.gov/MatrixMarket/formats.html

// Parts of this code are from SuiteSparse/CHOLMOD/Check/cholmod_write.c, and
// are used here by permission of the author of CHOLMOD/Check (T. A. Davis).

#include "LG_internal.h"

#undef  LG_FREE_WORK
#define LG_FREE_WORK                    \
{                                       \
    LAGraph_Free ((void **) &I, NULL) ; \
    LAGraph_Free ((void **) &J, NULL) ; \
    LAGraph_Free ((void **) &K, NULL) ; \
    LAGraph_Free ((void **) &X, NULL) ; \
    GrB_free (&AT) ;                    \
    GrB_free (&M) ;                     \
    GrB_free (&C) ;                     \
}

#undef  LG_FREE_ALL
#define LG_FREE_ALL LG_FREE_WORK

//------------------------------------------------------------------------------
// print_double
//------------------------------------------------------------------------------

// Print a double value to the file, using the shortest format that ensures the
// value is written precisely.  Returns true if successful, false if an I/O
// error occurred.

static bool print_double
(
    FILE *f,        // file to print to
    double x        // value to print
)
{

    char s [MAXLINE], *p ;
    int64_t i, dest = 0, src = 0 ;
    int width, ok ;

    //--------------------------------------------------------------------------
    // handle Inf and NaN
    //--------------------------------------------------------------------------

    if (isnan (x))
    {
        return (fprintf (f, "nan") > 0) ;
    }
    if (isinf (x))
    {
        return (fprintf (f, (x < 0) ? "-inf" : "inf") > 0) ;
    }

    //--------------------------------------------------------------------------
    // find the smallest acceptable precision
    //--------------------------------------------------------------------------

    for (width = 6 ; width < 20 ; width++)
    {
        double y ;
        sprintf (s, "%.*g", width, x) ;
        sscanf (s, "%lg", &y) ;
        if (x == y) break ;
    }

    //--------------------------------------------------------------------------
    // shorten the string
    //--------------------------------------------------------------------------

    // change "e+0" to "e", change "e+" to "e", and change "e-0" to "e-"
    for (i = 0 ; i < MAXLINE && s [i] != '\0' ; i++)
    {
        if (s [i] == 'e')
        {
            if (s [i+1] == '+')
            {
                dest = i+1 ;
                if (s [i+2] == '0')
                {
                    // delete characters s[i+1] and s[i+2]
                    src = i+3 ;
                }
                else
                {
                    // delete characters s[i+1]
                    src = i+2 ;
                }
            }
            else if (s [i+1] == '-')
            {
                dest = i+2 ;
                if (s [i+2] == '0')
                {
                    // delete character s[i+2]
                    src = i+3 ;
                }
                else
                {
                    // no change
                    break ;
                }
            }
            while (s [src] != '\0')
            {
                s [dest++] = s [src++] ;
            }
            s [dest] = '\0' ;
            break ;
        }
    }

    // delete the leading "0" if present and not necessary
    p = s ;
    s [MAXLINE-1] = '\0' ;
    i = strlen (s) ;
    if (i > 2 && s [0] == '0' && s [1] == '.')
    {
        // change "0.x" to ".x"
        p = s + 1 ;
    }
    else if (i > 3 && s [0] == '-' && s [1] == '0' && s [2] == '.')
    {
        // change "-0.x" to "-.x"
        s [1] = '-' ;
        p = s + 1 ;
    }

#if 0
    // double-check
    i = sscanf (p, "%lg", &z) ;
    if (i != 1 || y != z)
    {
        // oops! something went wrong in the "e+0" edit, above.
        // this "cannot" happen
        sprintf (s, "%.*g", width, x) ;
        p = s ;
    }
#endif

    //--------------------------------------------------------------------------
    // print the value to the file
    //--------------------------------------------------------------------------

    return (fprintf (f, "%s", p) > 0) ;
}

//------------------------------------------------------------------------------
// LAGraph_MMWrite: write a matrix to a MatrixMarket file
//------------------------------------------------------------------------------

int LAGraph_MMWrite
(
    // input:
    GrB_Matrix A,       // matrix to write to the file
    FILE *f,            // file to write it to, must be already open
    FILE *fcomments,    // optional file with extra comments, may be NULL
    char *msg
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    LG_CLEAR_MSG ;
    void *X = NULL ;
    GrB_Index *I = NULL, *J = NULL, *K = NULL ;
    GrB_Matrix M = NULL, AT = NULL, C = NULL ;
    LG_ASSERT (A != NULL, GrB_NULL_POINTER) ;
    LG_ASSERT (f != NULL, GrB_NULL_POINTER) ;

    //--------------------------------------------------------------------------
    // determine the basic matrix properties
    //--------------------------------------------------------------------------

    GrB_Index nrows, ncols, nvals ;
    GRB_TRY (GrB_Matrix_nrows (&nrows, A)) ;
    GRB_TRY (GrB_Matrix_ncols (&ncols, A)) ;
    GRB_TRY (GrB_Matrix_nvals (&nvals, A)) ;
    GrB_Index n = nrows ;

    //--------------------------------------------------------------------------
    // determine if the matrix is dense
    //--------------------------------------------------------------------------

    MM_fmt_enum MM_fmt = MM_coordinate ;

    // guard against integer overflow
    if (((double) nrows * (double) ncols < (double) INT64_MAX) &&
        (nvals == nrows * ncols))
    {
        MM_fmt = MM_array ;
    }

    //--------------------------------------------------------------------------
    // determine the entry type
    //--------------------------------------------------------------------------

    GrB_Type type ;
    char atype_name [LAGRAPH_MAX_NAME_LEN] ;
    LG_TRY (LAGraph_Matrix_TypeName (atype_name, A, msg)) ;
    LG_TRY (LAGraph_TypeFromName (&type, atype_name, msg)) ;

    MM_type_enum MM_type = MM_integer ;

    if (type == GrB_BOOL   || type == GrB_INT8   || type == GrB_INT16  ||
        type == GrB_INT32  || type == GrB_INT64  || type == GrB_UINT8  ||
        type == GrB_UINT16 || type == GrB_UINT32 || type == GrB_UINT64)
    {
        MM_type = MM_integer ;
    }
    else if (type == GrB_FP32 || type == GrB_FP64)
    {
        MM_type = MM_real ;
    }
    #if 0
    #if LAGRAPH_SUITESPARSE
    else if (type == GxB_FC32 || type == GxB_FC64)
    {
        MM_type = MM_complex ;
    }
    #endif
    #endif
    else
    {
        LG_ASSERT_MSG (false, GrB_NOT_IMPLEMENTED, "type not supported") ;
    }

    //--------------------------------------------------------------------------
    // determine symmetry
    //--------------------------------------------------------------------------

    MM_storage_enum MM_storage = MM_general ;

    if (nrows == ncols)
    {
        // AT = A'
        GRB_TRY (GrB_Matrix_new (&AT, type, n, n)) ;
        GRB_TRY (GrB_transpose (AT, NULL, NULL, A, NULL)) ;

        //----------------------------------------------------------------------
        // check for symmetry
        //----------------------------------------------------------------------

        bool isequal = false ;
        LG_TRY (LAGraph_Matrix_IsEqual (&isequal, A, AT, msg)) ;
        if (isequal)
        {
            MM_storage = MM_symmetric ;
        }

        //----------------------------------------------------------------------
        // check for skew-symmetry
        //----------------------------------------------------------------------

        // for signed types only
        if (MM_storage == MM_general)
        {
            // select the operator
            GrB_UnaryOp op = NULL ;
            if      (type == GrB_INT8 ) op = GrB_AINV_INT8  ;
            else if (type == GrB_INT16) op = GrB_AINV_INT16 ;
            else if (type == GrB_INT32) op = GrB_AINV_INT32 ;
            else if (type == GrB_INT64) op = GrB_AINV_INT64 ;
            else if (type == GrB_FP32 ) op = GrB_AINV_FP32  ;
            else if (type == GrB_FP64 ) op = GrB_AINV_FP64  ;
            #if 0
            else if (type == GxB_FC32 ) op = GxB_AINV_FC32 ;
            else if (type == GxB_FC64 ) op = GxB_AINV_FC64 ;
            #endif
            if (op != NULL)
            {
                GRB_TRY (GrB_apply (AT, NULL, NULL, op, AT, NULL)) ;
                LG_TRY (LAGraph_Matrix_IsEqual (&isequal, A, AT, msg)) ;
                if (isequal)
                {
                    MM_storage = MM_skew_symmetric ;
                }
            }
        }

        //----------------------------------------------------------------------
        // check for Hermitian (not yet supported)
        //----------------------------------------------------------------------

        #if 0
        if (MM_type == MM_complex && MM_storage == MM_general)
        {
            LG_TRY (LAGraph_Matrix_IsEqualOp (&isequal, A, AT,
                LAGraph_HERMITIAN_ComplexFP64, msg)) ;
            if (isequal)
            {
                MM_storage = MM_hermitian ;
            }
        }
        #endif

        GrB_free (&AT) ;
    }

    //--------------------------------------------------------------------------
    // determine if the matrix is structural-only
    //--------------------------------------------------------------------------

    bool is_structural = false ;
    if (! (MM_storage == MM_skew_symmetric || MM_storage == MM_hermitian))
    {
        if (type == GrB_BOOL)
        {
            GRB_TRY (GrB_reduce (&is_structural, NULL, GrB_LAND_MONOID_BOOL,
                A, NULL)) ;
        }
        else
        {
            GRB_TRY (GrB_Matrix_new (&C, GrB_BOOL, nrows, ncols)) ;
            GrB_BinaryOp op = NULL ;
            if      (type == GrB_INT8  ) op = GrB_EQ_INT8   ;
            else if (type == GrB_INT16 ) op = GrB_EQ_INT16  ;
            else if (type == GrB_INT32 ) op = GrB_EQ_INT32  ;
            else if (type == GrB_INT64 ) op = GrB_EQ_INT64  ;
            else if (type == GrB_UINT8 ) op = GrB_EQ_UINT8  ;
            else if (type == GrB_UINT16) op = GrB_EQ_UINT16 ;
            else if (type == GrB_UINT32) op = GrB_EQ_UINT32 ;
            else if (type == GrB_UINT64) op = GrB_EQ_UINT64 ;
            else if (type == GrB_FP32  ) op = GrB_EQ_FP32   ;
            else if (type == GrB_FP64  ) op = GrB_EQ_FP64   ;
            #if 0
            else if (type == GxB_FC32  ) op = GrB_EQ_FC32 ;
            else if (type == GxB_FC64  ) op = GrB_EQ_FC64 ;
            #endif
            GRB_TRY (GrB_apply (C, NULL, NULL, op, A, 1, NULL)) ;
            GRB_TRY (GrB_reduce (&is_structural, NULL, GrB_LAND_MONOID_BOOL,
                C, NULL)) ;
            GrB_free (&C) ;
        }
        if (is_structural)
        {
            MM_type = MM_pattern ;
            MM_fmt = MM_coordinate ;
        }
    }

    //--------------------------------------------------------------------------
    // write the Matrix Market header
    //--------------------------------------------------------------------------

    FPRINTF (f, "%%%%MatrixMarket matrix") ;

    switch (MM_fmt)
    {
        default :
        case MM_coordinate      : FPRINTF (f, " coordinate")        ; break ;
        case MM_array           : FPRINTF (f, " array")             ; break ;
    }

    switch (MM_type)
    {
        default :
        case MM_real            : FPRINTF (f, " real")              ; break ;
        case MM_integer         : FPRINTF (f, " integer")           ; break ;
//      case MM_complex         : FPRINTF (f, " complex")           ; break ;
        case MM_pattern         : FPRINTF (f, " pattern")           ; break ;
    }

    switch (MM_storage)
    {
        default :
        case MM_general         : FPRINTF (f, " general\n")         ; break ;
        case MM_symmetric       : FPRINTF (f, " symmetric\n")       ; break ;
        case MM_skew_symmetric  : FPRINTF (f, " skew-symmetric\n")  ; break ;
//      case MM_hermitian       : FPRINTF (f, " Hermitian\n")       ; break ;
    }

    FPRINTF (f, "%%%%GraphBLAS type ") ;
    if      (type == GrB_BOOL  ) { FPRINTF (f, "bool\n")   ; }
    else if (type == GrB_INT8  ) { FPRINTF (f, "int8_t\n")   ; }
    else if (type == GrB_INT16 ) { FPRINTF (f, "int16_t\n")  ; }
    else if (type == GrB_INT32 ) { FPRINTF (f, "int32_t\n")  ; }
    else if (type == GrB_INT64 ) { FPRINTF (f, "int64_t\n")  ; }
    else if (type == GrB_UINT8 ) { FPRINTF (f, "uint8_t\n")  ; }
    else if (type == GrB_UINT16) { FPRINTF (f, "uint16_t\n") ; }
    else if (type == GrB_UINT32) { FPRINTF (f, "uint32_t\n") ; }
    else if (type == GrB_UINT64) { FPRINTF (f, "uint64_t\n") ; }
    else if (type == GrB_FP32  ) { FPRINTF (f, "float\n")   ; }
    else if (type == GrB_FP64  ) { FPRINTF (f, "double\n")   ; }
    #if 0
    else if (type == GxB_FC32  ) { FPRINTF (f, "float complex\n")  ; }
    else if (type == GxB_FC64  ) { FPRINTF (f, "double complex\n") ; }
    #endif

#if 0
    if      (type == GrB_BOOL  ) { FPRINTF (f, "GrB_BOOL\n")   ; }
    else if (type == GrB_INT8  ) { FPRINTF (f, "GrB_INT8\n")   ; }
    else if (type == GrB_INT16 ) { FPRINTF (f, "GrB_INT16\n")  ; }
    else if (type == GrB_INT32 ) { FPRINTF (f, "GrB_INT32\n")  ; }
    else if (type == GrB_INT64 ) { FPRINTF (f, "GrB_INT64\n")  ; }
    else if (type == GrB_UINT8 ) { FPRINTF (f, "GrB_UINT8\n")  ; }
    else if (type == GrB_UINT16) { FPRINTF (f, "GrB_UINT16\n") ; }
    else if (type == GrB_UINT32) { FPRINTF (f, "GrB_UINT32\n") ; }
    else if (type == GrB_UINT64) { FPRINTF (f, "GrB_UINT64\n") ; }
    else if (type == GrB_FP32  ) { FPRINTF (f, "GrB_FP32\n")   ; }
    else if (type == GrB_FP64  ) { FPRINTF (f, "GrB_FP64\n")   ; }
    #if 0
    else if (type == GxB_FC32  ) { FPRINTF (f, "GxB_FC32\n")   ; }
    else if (type == GxB_FC64  ) { FPRINTF (f, "GxB_FC64\n")   ; }
    #endif
#endif

    //--------------------------------------------------------------------------
    // include any additional comments
    //--------------------------------------------------------------------------

    if (fcomments != NULL)
    {
        char buffer [MAXLINE] ;
        while (fgets (buffer, MAXLINE-1, fcomments) != NULL)
        {
            FPRINTF (f, "%%%s", buffer) ;
        }
    }

    //--------------------------------------------------------------------------
    // print the first line
    //--------------------------------------------------------------------------

    bool is_general = (MM_storage == MM_general) ;
    GrB_Index nvals_to_print = nvals ;

    if (!is_general)
    {
        // count the entries on the diagonal
        int64_t nself_edges = 0 ;
        LG_TRY (LG_nself_edges (&nself_edges, A, msg)) ;
        // nvals_to_print = # of entries in tril(A), including diagonal
        nvals_to_print = nself_edges + (nvals - nself_edges) / 2 ;
    }

    if (MM_fmt == MM_array)
    {
        // write `nrows ncols` if the array format is used
        FPRINTF (f, "%" PRIu64 " %" PRIu64 "\n",
            nrows, ncols) ;
    }
    else
    {
        // otherwise write `nrows ncols nvals` for the coordinate format
        FPRINTF (f, "%" PRIu64 " %" PRIu64 " %" PRIu64 "\n",
            nrows, ncols, nvals_to_print) ;
    }

    if (nvals_to_print == 0)
    {
        // quick return if nothing more to do
        LG_FREE_ALL ;
        return (GrB_SUCCESS) ;
    }

    //--------------------------------------------------------------------------
    // extract and print tuples
    //--------------------------------------------------------------------------

    LG_TRY (LAGraph_Malloc ((void **) &I, nvals, sizeof (GrB_Index), msg)) ;
    LG_TRY (LAGraph_Malloc ((void **) &J, nvals, sizeof (GrB_Index), msg)) ;
    LG_TRY (LAGraph_Malloc ((void **) &K, nvals, sizeof (GrB_Index), msg)) ;
    for (int64_t k = 0 ; k < nvals ; k++)
    {
        K [k] = k ;
    }

    GrB_Index nvals_printed = 0 ;
    bool coord = (MM_fmt == MM_coordinate) ;

    #define WRITE_TUPLES(ctype,is_unsigned,is_signed,is_real,is_complex)    \
    {                                                                       \
        ctype *X = NULL ;                                                   \
        LG_TRY (LAGraph_Malloc ((void **) &X, nvals, sizeof (ctype), msg)) ;\
        GRB_TRY (GrB_Matrix_extractTuples (I, J, X, &nvals, A)) ;           \
        LG_TRY (LG_msort3 ((int64_t *) J, (int64_t *) I,                    \
            (int64_t *) K, nvals, msg)) ;                                   \
        for (int64_t k = 0 ; k < nvals ; k++)                               \
        {                                                                   \
            /* convert the row and column index to 1-based */               \
            GrB_Index i = I [k] + 1 ;                                       \
            GrB_Index j = J [k] + 1 ;                                       \
            ctype     x = X [K [k]] ;                                       \
            if (is_general || i >= j)                                       \
            {                                                               \
                /* print the row and column index of the tuple */           \
                if (coord) FPRINTF (f, "%" PRIu64 " %" PRIu64 " ", i, j) ;  \
                /* print the value of the tuple */                          \
                if (is_structural)                                          \
                {                                                           \
                    /* print nothing */ ;                                   \
                }                                                           \
                else if (is_unsigned)                                       \
                {                                                           \
                    FPRINTF (f, "%" PRIu64, (uint64_t) x) ;                 \
                }                                                           \
                else if (is_signed)                                         \
                {                                                           \
                    FPRINTF (f, "%" PRId64, (int64_t) x) ;                  \
                }                                                           \
                else if (is_real)                                           \
                {                                                           \
                    LG_ASSERT_MSG (print_double (f, (double) x),            \
                        LAGRAPH_IO_ERROR, "Unable to write to file") ;      \
                }                                                           \
            /*  else if (is_complex)                                 */     \
            /*  {                                                    */     \
            /*      LG_ASSERT_MSG (print_double (f, creal (x)),      */     \
            /*          LAGRAPH_IO_ERROR,                            */     \
            /*          "Unable to write to file") ;                 */     \
            /*      FPRINTF (f, " ") ;                               */     \
            /*      LG_ASSERT_MSG (print_double (f, cimag (x)),      */     \
            /*          LAGRAPH_IO_ERROR,                            */     \
            /*          "Unable to write to file") ;                 */     \
            /*  }                                                    */     \
                FPRINTF (f, "\n") ;                                         \
            }                                                               \
            nvals_printed++ ;                                               \
        }                                                                   \
        LG_TRY (LAGraph_Free ((void **) &X, NULL)) ;                        \
    }

    if      (type == GrB_BOOL   ) WRITE_TUPLES (bool    , 1, 0, 0, 0)
    else if (type == GrB_INT8   ) WRITE_TUPLES (int8_t  , 0, 1, 0, 0)
    else if (type == GrB_INT16  ) WRITE_TUPLES (int16_t , 0, 1, 0, 0)
    else if (type == GrB_INT32  ) WRITE_TUPLES (int32_t , 0, 1, 0, 0)
    else if (type == GrB_INT64  ) WRITE_TUPLES (int64_t , 0, 1, 0, 0)
    else if (type == GrB_UINT8  ) WRITE_TUPLES (uint8_t , 1, 0, 0, 0)
    else if (type == GrB_UINT16 ) WRITE_TUPLES (uint16_t, 1, 0, 0, 0)
    else if (type == GrB_UINT32 ) WRITE_TUPLES (uint32_t, 1, 0, 0, 0)
    else if (type == GrB_UINT64 ) WRITE_TUPLES (uint64_t, 1, 0, 0, 0)
    else if (type == GrB_FP32   ) WRITE_TUPLES (float   , 0, 0, 1, 0)
    else if (type == GrB_FP64   ) WRITE_TUPLES (double  , 0, 0, 1, 0)
    #if 0
    else if (type == GxB_FC32   ) WRITE_TUPLES (GxB_FC32_t, 0, 0, 0, 1) ;
    else if (type == GxB_FC64   ) WRITE_TUPLES (GxB_FC64_t, 0, 0, 0, 1) ;
    #endif

    ASSERT (nvals_to_print == nvals_printed) ;

    //--------------------------------------------------------------------------
    // free workspace and return
    //--------------------------------------------------------------------------

    LG_FREE_ALL ;
    return (GrB_SUCCESS) ;
}
