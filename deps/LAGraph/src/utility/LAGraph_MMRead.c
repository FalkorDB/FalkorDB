//------------------------------------------------------------------------------
// LAGraph_MMRead: read a matrix from a Matrix Market file
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

// LAGraph_MMRead: read a matrix from a Matrix Market file

// Parts of this code are from SuiteSparse/CHOLMOD/Check/cholmod_read.c, and
// are used here by permission of the author of CHOLMOD/Check (T. A. Davis).

// The Matrix Market format is described at:
// https://math.nist.gov/MatrixMarket/formats.html

// Return values:
//  GrB_SUCCESS: input file and output matrix are valid
//  LAGRAPH_IO_ERROR: the input file cannot be read or has invalid content
//  GrB_NULL_POINTER:  A or f are NULL on input
//  GrB_NOT_IMPLEMENTED: complex types not yet supported
//  other: return values directly from GrB_* methods

#define LG_FREE_WORK                    \
{                                       \
    LAGraph_Free ((void **) &I, NULL) ; \
    LAGraph_Free ((void **) &J, NULL) ; \
    LAGraph_Free ((void **) &X, NULL) ; \
}

#define LG_FREE_ALL                     \
{                                       \
    LG_FREE_WORK ;                      \
    GrB_free (A) ;                      \
}

#include "LG_internal.h"

//------------------------------------------------------------------------------
// get_line
//------------------------------------------------------------------------------

// Read one line of the file, return true if successful, false if EOF.
// The string is returned in buf, converted to lower case.

static inline bool get_line
(
    FILE *f,        // file open for reading
    char *buf       // size MAXLINE+1
)
{

    // check inputs
    ASSERT (f != NULL) ;
    ASSERT (buf != NULL) ;

    // read the line from the file
    buf [0] = '\0' ;
    buf [1] = '\0' ;
    if (fgets (buf, MAXLINE, f) == NULL)
    {
        // EOF or other I/O error
        return (false) ;
    }
    buf [MAXLINE] = '\0' ;

    // convert the string to lower case
    for (int k = 0 ; k < MAXLINE && buf [k] != '\0' ; k++)
    {
        buf [k] = tolower (buf [k]) ;
    }
    return (true) ;
}

//------------------------------------------------------------------------------
// is_blank_line
//------------------------------------------------------------------------------

// returns true if buf is a blank line or comment, false otherwise.

static inline bool is_blank_line
(
    char *buf       // size MAXLINE+1, never NULL
)
{

    // check inputs
    ASSERT (buf != NULL) ;

    // check if comment line
    if (buf [0] == '%')
    {
        // line is a comment
        return (true) ;
    }

    // check if blank line
    for (int k = 0 ; k <= MAXLINE ; k++)
    {
        int c = buf [k] ;
        if (c == '\0')
        {
            // end of line
            break ;
        }
        if (!isspace (c))
        {
            // non-space character; this is not an error
            return (false) ;
        }
    }

    // line is blank
    return (true) ;
}

//------------------------------------------------------------------------------
// read_double
//------------------------------------------------------------------------------

// Read a single double value from a string.  The string may be any string
// recognized by sscanf, or inf, -inf, +inf, or nan.  The token infinity is
// also OK instead of inf (only the first 3 letters of inf* or nan* are
// significant, and the rest are ignored).

static inline bool read_double      // true if successful, false if failure
(
    char *p,        // string containing the value
    double *rval    // value to read in
)
{
    while (*p && isspace (*p)) p++ ;   // skip any spaces

    if (MATCH (p, "inf", 3) || MATCH (p, "+inf", 4))
    {
        (*rval) = INFINITY ;
    }
    else if (MATCH (p, "-inf", 4))
    {
        (*rval) = -INFINITY ;
    }
    else if (MATCH (p, "nan", 3))
    {
        (*rval) = NAN ;
    }
    else
    {
        if (sscanf (p, "%lg", rval) != 1)
        {
            // invalid file format, EOF, or other I/O error
            return (false) ;
        }
    }
    return (true) ;
}

//------------------------------------------------------------------------------
// read_entry: read a numerical value and typecast to the given type
//------------------------------------------------------------------------------

static inline bool read_entry   // returns true if successful, false if failure
(
    char *p,        // string containing the value
    GrB_Type type,  // type of value to read
    bool structural,   // if true, then the value is 1
    uint8_t *x      // value read in, a pointer to space of size of the type
)
{

    int64_t ival = 1 ;
    double rval = 1, zval = 0 ;

    while (*p && isspace (*p)) p++ ;   // skip any spaces

    if (type == GrB_BOOL)
    {
        if (!structural && sscanf (p, "%" SCNd64, &ival) != 1) return (false) ;
        if (ival < 0 || ival > 1)
        {
            // entry out of range
            return (false) ;
        }
        bool *result = (bool *) x ;
        result [0] = (bool) ival ;
    }
    else if (type == GrB_INT8)
    {
        if (!structural && sscanf (p, "%" SCNd64, &ival) != 1) return (false) ;
        if (ival < INT8_MIN || ival > INT8_MAX)
        {
            // entry out of range
            return (false) ;
        }
        int8_t *result = (int8_t *) x ;
        result [0] = (int8_t) ival ;
    }
    else if (type == GrB_INT16)
    {
        if (!structural && sscanf (p, "%" SCNd64, &ival) != 1) return (false) ;
        if (ival < INT16_MIN || ival > INT16_MAX)
        {
            // entry out of range
            return (false) ;
        }
        int16_t *result = (int16_t *) x ;
        result [0] = (int16_t) ival ;
    }
    else if (type == GrB_INT32)
    {
        if (!structural && sscanf (p, "%" SCNd64, &ival) != 1) return (false) ;
        if (ival < INT32_MIN || ival > INT32_MAX)
        {
            // entry out of range
            return (false) ;
        }
        int32_t *result = (int32_t *) x ;
        result [0] = (int32_t) ival ;
    }
    else if (type == GrB_INT64)
    {
        if (!structural && sscanf (p, "%" SCNd64, &ival) != 1) return (false) ;
        int64_t *result = (int64_t *) x ;
        result [0] = (int64_t) ival ;
    }
    else if (type == GrB_UINT8)
    {
        if (!structural && sscanf (p, "%" SCNd64, &ival) != 1) return (false) ;
        if (ival < 0 || ival > UINT8_MAX)
        {
            // entry out of range
            return (false) ;
        }
        uint8_t *result = (uint8_t *) x ;
        result [0] = (uint8_t) ival ;
    }
    else if (type == GrB_UINT16)
    {
        if (!structural && sscanf (p, "%" SCNd64, &ival) != 1) return (false) ;
        if (ival < 0 || ival > UINT16_MAX)
        {
            // entry out of range
            return (false) ;
        }
        uint16_t *result = (uint16_t *) x ;
        result [0] = (uint16_t) ival ;
    }
    else if (type == GrB_UINT32)
    {
        if (!structural && sscanf (p, "%" SCNd64, &ival) != 1) return (false) ;
        if (ival < 0 || ival > UINT32_MAX)
        {
            // entry out of range
            return (false) ;
        }
        uint32_t *result = (uint32_t *) x ;
        result [0] = (uint32_t) ival ;
    }
    else if (type == GrB_UINT64)
    {
        uint64_t uval = 1 ;
        if (!structural && sscanf (p, "%" SCNu64, &uval) != 1) return (false) ;
        uint64_t *result = (uint64_t *) x ;
        result [0] = (uint64_t) uval ;
    }
    else if (type == GrB_FP32)
    {
        if (!structural && !read_double (p, &rval)) return (false) ;
        float *result = (float *) x ;
        result [0] = (float) rval ;
    }
    else if (type == GrB_FP64)
    {
        if (!structural && !read_double (p, &rval)) return (false) ;
        double *result = (double *) x ;
        result [0] = rval ;
    }
#if 0
    else if (type == GxB_FC32)
    {
        if (!structural && !read_double (p, &rval)) return (false) ;
        while (*p && !isspace (*p)) p++ ;   // skip real part
        if (!structural && !read_double (p, &zval)) return (false) ;
        float *result = (float *) x ;
        result [0] = (float) rval ;     // real part
        result [1] = (float) zval ;     // imaginary part
    }
    else if (type == GxB_FC64)
    {
        if (!structural && !read_double (p, &rval)) return (false) ;
        while (*p && !isspace (*p)) p++ ;   // skip real part
        if (!structural && !read_double (p, &zval)) return (false) ;
        double *result = (double *) x ;
        result [0] = rval ;     // real part
        result [1] = zval ;     // imaginary part
    }
#endif

    return (true) ;
}

//------------------------------------------------------------------------------
// negate_scalar: negate a scalar value
//------------------------------------------------------------------------------

// negate the scalar x.  Do nothing for bool or uint*.

static inline void negate_scalar
(
    GrB_Type type,
    uint8_t *x
)
{

    if (type == GrB_INT8)
    {
        int8_t *value = (int8_t *) x ;
        (*value) = - (*value) ;
    }
    else if (type == GrB_INT16)
    {
        int16_t *value = (int16_t *) x ;
        (*value) = - (*value) ;
    }
    else if (type == GrB_INT32)
    {
        int32_t *value = (int32_t *) x ;
        (*value) = - (*value) ;
    }
    else if (type == GrB_INT64)
    {
        int64_t *value = (int64_t *) x ;
        (*value) = - (*value) ;
    }
    else if (type == GrB_FP32)
    {
        float *value = (float *) x ;
        (*value) = - (*value) ;
    }
    else if (type == GrB_FP64)
    {
        double *value = (double *) x ;
        (*value) = - (*value) ;
    }
#if 0
    else if (type == GxB_FC32)
    {
        float complex *value = (float complex *) x ;
        (*value) = - (*value) ;
    }
    else if (type == GxB_FC64)
    {
        double complex *value = (double complex *) x ;
        (*value) = - (*value) ;
    }
#endif
}

//------------------------------------------------------------------------------
// set_value
//------------------------------------------------------------------------------

// Add the (i,j,x) triplet to the I,J,X arrays as the kth triplet, and
// increment k.  No typecasting is done.

static inline void set_value
(
    size_t typesize,        // size of the numerical type, in bytes
    GrB_Index i,
    GrB_Index j,
    uint8_t *x,             // scalar, an array of size at least typesize
    GrB_Index *I,
    GrB_Index *J,
    uint8_t *X,
    GrB_Index *k            // # of triplets
)
{
    I [*k] = i ;
    J [*k] = j ;
    memcpy (X + ((*k) * typesize), x, typesize) ;
    (*k)++ ;
}

//------------------------------------------------------------------------------
// LAGraph_MMRead
//------------------------------------------------------------------------------

int LAGraph_MMRead
(
    // output:
    GrB_Matrix *A,  // handle of matrix to create
    // input:
    FILE *f,        // file to read from, already open
    char *msg
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Index *I = NULL, *J = NULL ;
    uint8_t *X = NULL ;
    LG_CLEAR_MSG ;
    LG_ASSERT (A != NULL, GrB_NULL_POINTER) ;
    LG_ASSERT (f != NULL, GrB_NULL_POINTER) ;
    (*A) = NULL ;

    //--------------------------------------------------------------------------
    // set the default properties
    //--------------------------------------------------------------------------

    MM_fmt_enum     MM_fmt     = MM_coordinate ;
    MM_type_enum    MM_type    = MM_real ;
    MM_storage_enum MM_storage = MM_general ;
    GrB_Type type = GrB_FP64 ;
    size_t typesize = sizeof (double) ;
    GrB_Index nrows = 0 ;
    GrB_Index ncols = 0 ;
    GrB_Index nvals = 0 ;

    //--------------------------------------------------------------------------
    // read the Matrix Market header
    //--------------------------------------------------------------------------

    // Read the header.  This consists of zero or more comment lines (blank, or
    // starting with a "%" in the first column), followed by a single data line
    // containing two or three numerical values.  The first line is normally:
    //
    //          %%MatrixMarket matrix <fmt> <type> <storage>
    //
    // but this is optional.  The 2nd line is also optional (the %%MatrixMarket
    // line is required for this 2nd line to be recognized):
    //
    //          %%GraphBLAS type <Ctype>
    //
    // where the Ctype is one of: bool, int8_t, int16_t, int32_t, int64_t,
    // uint8_t, uint16_t, uint32_t, uint64_t, float, or double.
    //
    // If the %%MatrixMarket line is not present, then the <fmt> <type> and
    // <storage> are implicit.  If the first data line contains 3 items,
    // then the implicit header is:
    //
    //          %%MatrixMarket matrix coordinate real general
    //          %%GraphBLAS type double
    //
    // If the first data line contains 2 items (nrows ncols), then the implicit
    // header is:
    //
    //          %%MatrixMarket matrix array real general
    //          %%GraphBLAS type double
    //
    // The implicit header is an extension of the Matrix Market format.

    char buf [MAXLINE+1] ;

    bool got_mm_header = false ;
    bool got_first_data_line = false ;
    int64_t line ;

    for (line = 1 ; get_line (f, buf) ; line++)
    {

        //----------------------------------------------------------------------
        // parse the line
        //----------------------------------------------------------------------

        if ((line == 1) && MATCH (buf, "%%matrixmarket", 14))
        {

            //------------------------------------------------------------------
            // read a Matrix Market header
            //------------------------------------------------------------------

            //  %%MatrixMarket matrix <fmt> <type> <storage>
            //  if present, it must be the first line in the file.

            got_mm_header = true ;
            char *p = buf + 14 ;

            //------------------------------------------------------------------
            // get "matrix" token and discard it
            //------------------------------------------------------------------

            while (*p && isspace (*p)) p++ ;        // skip any leading spaces

            if (!MATCH (p, "matrix", 6))
            {
                // invalid Matrix Market object
                LG_ASSERT_MSG (false,
                    LAGRAPH_IO_ERROR, "invalid MatrixMarket header"
                    " ('matrix' token missing)") ;
            }
            p += 6 ;                                // skip past token "matrix"

            //------------------------------------------------------------------
            // get the fmt token
            //------------------------------------------------------------------

            while (*p && isspace (*p)) p++ ;        // skip any leading spaces

            if (MATCH (p, "coordinate", 10))
            {
                MM_fmt = MM_coordinate ;
                p += 10 ;
            }
            else if (MATCH (p, "array", 5))
            {
                MM_fmt = MM_array ;
                p += 5 ;
            }
            else
            {
                // invalid Matrix Market format
                LG_ASSERT_MSG (false,
                    LAGRAPH_IO_ERROR, "invalid format in MatrixMarket header"
                    " (format must be 'coordinate' or 'array')") ;
            }

            //------------------------------------------------------------------
            // get the Matrix Market type token
            //------------------------------------------------------------------

            while (*p && isspace (*p)) p++ ;        // skip any leading spaces

            if (MATCH (p, "real", 4))
            {
                MM_type = MM_real ;
                type = GrB_FP64 ;
                typesize = sizeof (double) ;
                p += 4 ;
            }
            else if (MATCH (p, "integer", 7))
            {
                MM_type = MM_integer ;
                type = GrB_INT64 ;
                typesize = sizeof (int64_t) ;
                p += 7 ;
            }
            else if (MATCH (p, "complex", 7))
            {
                MM_type = MM_complex ;
#if 0
                type = GxB_FC64 ;
                typesize = sizeof (GxB_FC64_t) ;
                p += 7 ;
#endif
                LG_ASSERT_MSG (false,
                GrB_NOT_IMPLEMENTED, "complex types not supported") ;
            }
            else if (MATCH (p, "pattern", 7))
            {
                MM_type = MM_pattern ;
                type = GrB_BOOL ;
                typesize = sizeof (bool) ;
                p += 7 ;
            }
            else
            {
                // invalid Matrix Market type
                LG_ASSERT_MSG (false,
                    LAGRAPH_IO_ERROR, "invalid MatrixMarket type") ;
            }

            //------------------------------------------------------------------
            // get the storage token
            //------------------------------------------------------------------

            while (*p && isspace (*p)) p++ ;        // skip any leading spaces

            if (MATCH (p, "general", 7))
            {
                MM_storage = MM_general ;
            }
            else if (MATCH (p, "symmetric", 9))
            {
                MM_storage = MM_symmetric ;
            }
            else if (MATCH (p, "skew-symmetric", 14))
            {
                MM_storage = MM_skew_symmetric ;
            }
            else if (MATCH (p, "hermitian", 9))
            {
                MM_storage = MM_hermitian ;
            }
            else
            {
                // invalid Matrix Market storage
                LG_ASSERT_MSG (false,
                    LAGRAPH_IO_ERROR, "invalid MatrixMarket storage") ;
            }

            //------------------------------------------------------------------
            // ensure the combinations are valid
            //------------------------------------------------------------------

            if (MM_type == MM_pattern)
            {
                // (coodinate) x (pattern) x (general or symmetric)
                LG_ASSERT_MSG (
                    (MM_fmt == MM_coordinate &&
                    (MM_storage == MM_general || MM_storage == MM_symmetric)),
                    LAGRAPH_IO_ERROR,
                    "invalid MatrixMarket pattern combination") ;
            }

            if (MM_storage == MM_hermitian)
            {
                // (coordinate or array) x (complex) x (Hermitian)
                LG_ASSERT_MSG (MM_type == MM_complex,
                    LAGRAPH_IO_ERROR,
                    "invalid MatrixMarket complex combination") ;
            }

        }
        else if (got_mm_header && MATCH (buf, "%%graphblas", 11))
        {

            //------------------------------------------------------------------
            // %%GraphBLAS structured comment
            //------------------------------------------------------------------

            char *p = buf + 11 ;
            while (*p && isspace (*p)) p++ ;        // skip any leading spaces

            if (MATCH (p, "type", 4) && !got_first_data_line)
            {

                //--------------------------------------------------------------
                // %%GraphBLAS type <Ctype>
                //--------------------------------------------------------------

                // This must appear after the %%MatrixMarket header and before
                // the first data line.  Otherwise the %%GraphBLAS line is
                // treated as a pure comment.

                p += 4 ;
                while (*p && isspace (*p)) p++ ;    // skip any leading spaces

                // Ctype is one of: bool, int8_t, int16_t, int32_t, int64_t,
                // uint8_t, uint16_t, uint32_t, uint64_t, float, or double.
                // The complex types "float complex", or "double complex" are
                // not yet supported.

                if (MATCH (p, "bool", 4))
                {
                    type = GrB_BOOL ;
                    typesize = sizeof (bool) ;
                }
                else if (MATCH (p, "int8_t", 6))
                {
                    type = GrB_INT8 ;
                    typesize = sizeof (int8_t) ;
                }
                else if (MATCH (p, "int16_t", 7))
                {
                    type = GrB_INT16 ;
                    typesize = sizeof (int16_t) ;
                }
                else if (MATCH (p, "int32_t", 7))
                {
                    type = GrB_INT32 ;
                    typesize = sizeof (int32_t) ;
                }
                else if (MATCH (p, "int64_t", 7))
                {
                    type = GrB_INT64 ;
                    typesize = sizeof (int64_t) ;
                }
                else if (MATCH (p, "uint8_t", 7))
                {
                    type = GrB_UINT8 ;
                    typesize = sizeof (uint8_t) ;
                }
                else if (MATCH (p, "uint16_t", 8))
                {
                    type = GrB_UINT16 ;
                    typesize = sizeof (uint16_t) ;
                }
                else if (MATCH (p, "uint32_t", 8))
                {
                    type = GrB_UINT32 ;
                    typesize = sizeof (uint32_t) ;
                }
                else if (MATCH (p, "uint64_t", 8))
                {
                    type = GrB_UINT64 ;
                    typesize = sizeof (uint64_t) ;
                }
                else if (MATCH (p, "float complex", 13))
                {
#if 0
                    type = GxB_FC32 ;
                    typesize = sizeof (GxB_FC32_t) ;
#endif
                    LG_ASSERT_MSG (false,
                        GrB_NOT_IMPLEMENTED, "complex types not supported") ;
                }
                else if (MATCH (p, "double complex", 14))
                {
#if 0
                    type = GxB_FC64 ;
                    typesize = sizeof (GxB_FC64_t) ;
#endif
                    LG_ASSERT_MSG (false,
                        GrB_NOT_IMPLEMENTED, "complex types not supported") ;
                }
                else if (MATCH (p, "float", 5))
                {
                    type = GrB_FP32 ;
                    typesize = sizeof (float) ;
                }
                else if (MATCH (p, "double", 6))
                {
                    type = GrB_FP64 ;
                    typesize = sizeof (double) ;
                }
                else
                {
                    // unknown type
                    LG_ASSERT_MSG (false,
                        LAGRAPH_IO_ERROR, "unknown type") ;
                }

                if (MM_storage == MM_skew_symmetric && (type == GrB_BOOL ||
                    type == GrB_UINT8  || type == GrB_UINT16 ||
                    type == GrB_UINT32 || type == GrB_UINT64))
                {
                    // matrices with unsigned types cannot be skew-symmetric
                    LG_ASSERT_MSG (false, LAGRAPH_IO_ERROR,
                        "skew-symmetric matrices cannot have an unsigned type");
                }
            }
            else
            {
                // %%GraphBLAS line but no "type" as the 2nd token; ignore it
                continue ;
            }

        }
        else if (is_blank_line (buf))
        {

            // -----------------------------------------------------------------
            // blank line or comment line
            // -----------------------------------------------------------------

            continue ;

        }
        else
        {

            // -----------------------------------------------------------------
            // read the first data line
            // -----------------------------------------------------------------

            // format: [nrows ncols nvals] or just [nrows ncols]

            got_first_data_line = true ;
            int nitems = sscanf (buf, "%" SCNu64 " %" SCNu64 " %" SCNu64,
                &nrows, &ncols, &nvals) ;

            if (nitems == 2)
            {
                // a dense matrix
                if (!got_mm_header)
                {
                    // if no header, treat it as if it were
                    // %%MatrixMarket matrix array real general
                    MM_fmt = MM_array ;
                    MM_type = MM_real ;
                    MM_storage = MM_general ;
                    type = GrB_FP64 ;
                    typesize = sizeof (double) ;
                }
                if (MM_storage == MM_general)
                {
                    // dense general matrix
                    nvals = nrows * ncols ;
                }
                else
                {
                    // dense symmetric, skew-symmetric, or hermitian matrix
                    nvals = nrows + ((nrows * nrows - nrows) / 2) ;
                }
            }
            else if (nitems == 3)
            {
                // a sparse matrix
                if (!got_mm_header)
                {
                    // if no header, treat it as if it were
                    // %%MatrixMarket matrix coordinate real general
                    MM_fmt = MM_coordinate ;
                    MM_type = MM_real ;
                    MM_storage = MM_general ;
                    type = GrB_FP64 ;
                    typesize = sizeof (double) ;
                }
            }
            else
            {
                // wrong number of items in first data line
                LG_ASSERT_MSGF (false,
                    LAGRAPH_IO_ERROR, "invalid 1st data line"
                    " (line %" PRId64 " of input file)", line) ;
            }

            if (nrows != ncols)
            {
                // a rectangular matrix must be in the general storage
                LG_ASSERT_MSG (MM_storage == MM_general,
                    LAGRAPH_IO_ERROR, "invalid rectangular storage") ;
            }

            //------------------------------------------------------------------
            // header has been read in
            //------------------------------------------------------------------

            break ;
        }
    }

    //--------------------------------------------------------------------------
    // create the matrix
    //--------------------------------------------------------------------------

    GRB_TRY (GrB_Matrix_new (A, type, nrows, ncols)) ;

    //--------------------------------------------------------------------------
    // quick return for empty matrix
    //--------------------------------------------------------------------------

    if (nrows == 0 || ncols == 0 || nvals == 0)
    {
        // success: return an empty matrix.  This is not an error.
        return (GrB_SUCCESS) ;
    }

    //--------------------------------------------------------------------------
    // allocate space for the triplets
    //--------------------------------------------------------------------------

    GrB_Index nvals3 = ((MM_storage == MM_general) ? 1 : 2) * (nvals + 1) ;
    LG_TRY (LAGraph_Malloc ((void **) &I, nvals3, sizeof (GrB_Index), msg)) ;
    LG_TRY (LAGraph_Malloc ((void **) &J, nvals3, sizeof (GrB_Index), msg)) ;
    LG_TRY (LAGraph_Malloc ((void **) &X, nvals3, typesize, msg)) ;

    //--------------------------------------------------------------------------
    // read in the triplets
    //--------------------------------------------------------------------------

    GrB_Index i = -1, j = 0 ;
    GrB_Index nvals2 = 0 ;
    for (int64_t k = 0 ; k < nvals ; k++)
    {

        //----------------------------------------------------------------------
        // get the next triplet, skipping blank lines and comment lines
        //----------------------------------------------------------------------

        uint8_t x [MAXLINE] ;       // scalar value

        while (true)
        {

            //------------------------------------------------------------------
            // read the file until finding the next triplet
            //------------------------------------------------------------------

            bool ok = get_line (f, buf) ;
            line++ ;
            LG_ASSERT_MSG (ok, LAGRAPH_IO_ERROR, "premature EOF") ;
            if (is_blank_line (buf))
            {
                // blank line or comment
                continue ;
            }

            //------------------------------------------------------------------
            // get the row and column index
            //------------------------------------------------------------------

            char *p = buf ;
            if (MM_fmt == MM_array)
            {
                // array format, column major order
                i++ ;
                if (i == nrows)
                {
                    j++ ;
                    if (MM_storage == MM_general)
                    {
                        // dense matrix in column major order
                        i = 0 ;
                    }
                    else
                    {
                        // dense matrix in column major order, only the lower
                        // triangular form is present, including the diagonal
                        i = j ;
                    }
                }
            }
            else
            {
                // coordinate format; read the row index and column index
                int inputs = sscanf (p, "%" SCNu64 " %" SCNu64, &i, &j) ;
                LG_ASSERT_MSGF (inputs == 2, LAGRAPH_IO_ERROR,
                    "line %" PRId64 " of input file: indices invalid", line) ;
                // check the indices (they are 1-based in the MM file format)
                LG_ASSERT_MSGF (i >= 1 && i <= nrows, GrB_INDEX_OUT_OF_BOUNDS,
                    "line %" PRId64 " of input file: row index %" PRIu64
                    " out of range (must be in range 1 to %" PRIu64")",
                    line, i, nrows) ;
                LG_ASSERT_MSGF (j >= 1 && j <= ncols, GrB_INDEX_OUT_OF_BOUNDS,
                    "line %" PRId64 " of input file: column index %" PRIu64
                    " out of range (must be in range 1 to %" PRIu64")",
                    line, j, ncols) ;
                // convert from 1-based to 0-based.
                i-- ;
                j-- ;
                // advance p to the 3rd token to get the value of the entry
                while (*p &&  isspace (*p)) p++ ;   // skip any leading spaces
                while (*p && !isspace (*p)) p++ ;   // skip the row index
                while (*p &&  isspace (*p)) p++ ;   // skip any spaces
                while (*p && !isspace (*p)) p++ ;   // skip the column index
            }

            //------------------------------------------------------------------
            // read the value of the entry
            //------------------------------------------------------------------

            while (*p && isspace (*p)) p++ ;        // skip any spaces

            ok = read_entry (p, type, MM_type == MM_pattern, x) ;
            LG_ASSERT_MSGF (ok, LAGRAPH_IO_ERROR, "entry value invalid on line"
                " %" PRId64 " of input file", line) ;

            //------------------------------------------------------------------
            // set the value in the matrix
            //------------------------------------------------------------------

            set_value (typesize, i, j, x, I, J, X, &nvals2) ;

            //------------------------------------------------------------------
            // also set the A(j,i) entry, if symmetric
            //------------------------------------------------------------------

            if (i != j && MM_storage != MM_general)
            {
                if (MM_storage == MM_symmetric)
                {
                    set_value (typesize, j, i, x, I, J, X, &nvals2) ;
                }
                else if (MM_storage == MM_skew_symmetric)
                {
                    negate_scalar (type, x) ;
                    set_value (typesize, j, i, x, I, J, X, &nvals2) ;
                }
                #if 0
                else if (MM_storage == MM_hermitian)
                {
                    double complex *value = (double complex *) x ;
                    (*value) = conj (*value) ;
                    set_value (typesize, j, i, x, I, J, X, &nvals2) ;
                }
                #endif
            }

            // one more entry has been read in
            break ;
        }
    }

    //--------------------------------------------------------------------------
    // build the final matrix
    //--------------------------------------------------------------------------

    if (type == GrB_BOOL)
    {
        GRB_TRY (GrB_Matrix_build_BOOL (*A, I, J, (bool *) X, nvals2, NULL)) ;
    }
    else if (type == GrB_INT8)
    {
        GRB_TRY (GrB_Matrix_build_INT8 (*A, I, J, (int8_t *) X, nvals2, NULL)) ;
    }
    else if (type == GrB_INT16)
    {
        GRB_TRY (GrB_Matrix_build_INT16 (*A, I, J, (int16_t *) X, nvals2, NULL)) ;
    }
    else if (type == GrB_INT32)
    {
        GRB_TRY (GrB_Matrix_build_INT32 (*A, I, J, (int32_t *) X, nvals2, NULL)) ;
    }
    else if (type == GrB_INT64)
    {
        GRB_TRY (GrB_Matrix_build_INT64 (*A, I, J, (int64_t *) X, nvals2, NULL)) ;
    }
    else if (type == GrB_UINT8)
    {
        GRB_TRY (GrB_Matrix_build_UINT8 (*A, I, J, (uint8_t *) X, nvals2, NULL)) ;
    }
    else if (type == GrB_UINT16)
    {
        GRB_TRY (GrB_Matrix_build_UINT16 (*A, I, J, (uint16_t *) X, nvals2, NULL)) ;
    }
    else if (type == GrB_UINT32)
    {
        GRB_TRY (GrB_Matrix_build_UINT32 (*A, I, J, (uint32_t *) X, nvals2, NULL)) ;
    }
    else if (type == GrB_UINT64)
    {
        GRB_TRY (GrB_Matrix_build_UINT64 (*A, I, J, (uint64_t *) X, nvals2, NULL)) ;
    }
    else if (type == GrB_FP32)
    {
        GRB_TRY (GrB_Matrix_build_FP32 (*A, I, J, (float *) X, nvals2, NULL)) ;
    }
    else if (type == GrB_FP64)
    {
        GRB_TRY (GrB_Matrix_build_FP64 (*A, I, J, (double *) X, nvals2, NULL)) ;
    }
#if 0
    else if (type == GxB_FC32)
    {
        GRB_TRY (GxB_Matrix_build_FC32 (*A, I, J, (GxB_FC32_t *) X, nvals2, NULL)) ;
    }
    else if (type == GxB_FC64)
    {
        GRB_TRY (GxB_Matrix_build_FC64 (*A, I, J, (GxB_FC64_t *) X, nvals2, NULL)) ;
    }
#endif

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    LG_FREE_WORK ;
    return (GrB_SUCCESS) ;
}
