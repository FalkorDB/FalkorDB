//------------------------------------------------------------------------------
// gbmx_interface.h: the SuiteSparse:GraphBLAS MATLAB/Octave interface
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------
// error handling for mexFunctions and gbmx_* utilities
//------------------------------------------------------------------------------

#define GBMX_USAGE(ok,usage)                                \
    char err [ERRLEN] ;                                     \
    err [0] = '\0' ;                                        \
    gbmx_usage (ok, usage, err) ;

// error handling for mexFunctions
#undef  ERROR2
#define ERROR2(errmsg,arg,info)                             \
{                                                           \
    gbcov_put ( ) ;                                         \
    FREE_ALL ;                                              \
    mexErrMsgIdAndTxt ("GrB:error", errmsg, arg) ;          \
}

#undef  ERROR
#define ERROR(errmsg,info)                                  \
{                                                           \
    gbcov_put ( ) ;                                         \
    FREE_ALL ;                                              \
    mexErrMsgIdAndTxt ("GrB:error", errmsg) ;               \
}

//------------------------------------------------------------------------------
// mx-based utilties
//------------------------------------------------------------------------------

// These methods can use mxMalloc and mexErrMsgIdAndTxt.  They cannot use any
// GraphBLAS methods (or if they do, those methods should not allocate any
// memory in the default malloc/free arena).  These methods do not return an
// error if mxMalloc fails or if mexErrMsgIdAndTxt is called.  Instead, control
// is returned directly to MATLAB.

void gbmx_abort ( void ) ;  // terminate immediately (debug assertions only)

GrB_Info gbmx_defaults           // set global GraphBLAS defaults for MATLAB
(
    char err [ERRLEN]
) ;

mxArray *gbmx_export_ghb_mxstruct   // construct an mxArray struct for GhB
(
    GrB_Matrix **C_opaque_handle
) ;

mxArray *gbmx_export_grb_mxstruct   // construct an mxArray struct for GrB
(
    GrB_Matrix *C_handle            // matrix to export; freed on output
) ;

void gbmx_free                  // mxFree wrapper
(
    void **p_handle             // handle to pointer to be freed
) ;

int gbmx_flush ( void ) ;       // flush mexPrintf output to Command Window

GrB_Matrix gbmx_get_ghb_matrix  // the content of a MATLAB GhB handle object
(
    // input
    const mxArray *G            // must be a GhB object
) ;

mxArray *gbmx_get_ghb_handle    // the MATLAB GhB opaque handle
(
    // input
    const mxArray *G            // must be a GhB object
) ;

void gbmx_get_grb_matrix        // get content of a GrB matrix
(
    // output
    gb_matrix matrix,
    // input
    const mxArray *X
) ;

int64_t gbmx_get_int64_scalar   // return int64 value of a MATLAB scalar
(
    const mxArray *mxscalar,    // MATLAB scalar to extract
    char *name                  // name of the scalar
) ;

uint64_t *gbmx_get_integer_list
(
    const mxArray *mxList,
    uint64_t *len
) ;

kind_enum_t gbmx_get_kind
(
    const mxArray *mxdesc
) ;

void gbmx_get_matrix
(
    // output
    gb_matrix matrix,       // either a GraphBLAS or MATLAB matrix, statically
                            // allocated (but undefined) on input
    // input
    const mxArray *X        // GrB, GhB, or MATLAB matrix
) ;

void gbmx_get_mxargs
(
    // input:
    int nargin,                 // # inputs for mexFunction (must be > 0)
    const mxArray *pargin [ ],  // input arguments for mexFunction
    const char *usage,          // usage to print, if too many args appear
    // output:
    struct gb_matrix_struct Matrix [6], // matrix arguments
    int *nmatrices,             // # of matrix arguments
    char String [2][LEN+2],     // string arguments
    int *nstrings,              // # of string arguments
    mxArray *Cell [2],          // cell array arguments
    int *ncells,                // # of cell array arguments
    gb_descriptor gbdesc        // gb_descriptor struct
) ;

uint64_t gbmx_get_uint64_scalar // return uint64 value of a MATLAB scalar
(
    const mxArray *mxscalar,    // MATLAB scalar to extract
    char *name                  // name of the scalar
) ;

bool gbmx_mxarray_is_scalar     // true if built-in array is a scalar
(
    const mxArray *S
) ;

bool gbmx_mxarray_to_descriptor // true if descriptor present in pargin [...]
(
    // output:
    gb_descriptor gbdesc,   // statically allocated on input
    // input:
    const mxArray *mxdesc   // MATLAB struct with possible descriptor
) ;

GrB_Type gbmx_mxarray_type      // return the GrB_Type of a built-in matrix
(
    const mxArray *X
) ;

void gbmx_mxcell_to_matrices
(
    // output
    struct gb_matrix_struct Cell_Matrix [3], // matrix contents of the Cell
    int *len,                   // # of items in the Cell
    // input
    const mxArray *Cell         // built-in MATLAB cell array (at most 3 items)
) ;

void gbmx_mxstring_to_string  // copy a built-in string into a C string
(
    // output:
    char *string,           // size at least maxlen+1
    // input:
    const size_t maxlen,    // length of string
    const mxArray *S,       // built-in mxArray containing a string
    const char *name        // name of the mxArray
) ;

mxArray *gbmx_new_matlab_matrix // return new MATLAB full matrix
(
    const uint64_t nrows,       // dimensions
    const uint64_t ncols,
    GrB_Type type               // type of the array
) ;

int64_t gbmx_norm_kind      // determine the kind of norm to compute
(   
    const mxArray *arg
) ;

mxArray * gbmx_type_to_mxstring // return the built-in string from a GrB_Type
(
    const GrB_Type type
) ;

void gbmx_usage       // check usage and make sure GxB_init has been called
(
    bool ok,                // if false, then usage is not correct
    const char *message,    // error message if usage is not correct
    char err [ERRLEN]
) ;

//------------------------------------------------------------------------------
// mexFunctions in the util folder 
//------------------------------------------------------------------------------

void gbmx_assign_mexFunction    // gbmex_assign or gbmex_subassign mexFunctions
(
    int nargout,                // # output arguments for mexFunction
    mxArray *pargout [ ],       // output arguments for mexFunction
    int nargin,                 // # input arguments for mexFunction
    const mxArray *pargin [ ],  // input arguments for mexFunction
    bool do_subassign,          // true: do subassign, false: do assign
    const char *usage           // usage string to print if error
) ;

void gbmx_ewise_mexFunction
(
    int nargout,                // # output arguments for mexFunction
    mxArray *pargout [ ],       // output arguments for mexFunction
    int nargin,                 // # input arguments for mexFunction
    const mxArray *pargin [ ],  // input arguments for mexFunction
    const bool do_eadd,         // true: eadd, false: emult
    const char *usage           // usage string to print if error
) ;

//------------------------------------------------------------------------------
// gbmx_* source
//------------------------------------------------------------------------------

#include "gbmx_abort.c"
#include "gbmx_defaults.c"
#include "gbmx_export_ghb_struct.c"
#include "gbmx_export_grb_mxstruct.c"
#include "gbmx_flush.c"
#include "gbmx_free.c"
#include "gbmx_get_ghb_handle.c"
#include "gbmx_get_ghb_matrix.c"
#include "gbmx_get_grb_matrix.c"
#include "gbmx_get_int64_scalar.c"
#include "gbmx_get_integer_list.c"
#include "gbmx_get_kind.c"
#include "gbmx_get_matrix.c"
#include "gbmx_get_mxargs.c"
#include "gbmx_get_uint64_scalar.c"
#include "gbmx_mxarray_is_scalar.c"
#include "gbmx_mxarray_to_descriptor.c"
#include "gbmx_mxarray_type.c"
#include "gbmx_mxcell_to_matrices.c"
#include "gbmx_mxstring_to_string.c"
#include "gbmx_new_matlab_matrix.c"
#include "gbmx_norm_kind.c"
#include "gbmx_type_to_mxstring.c"
#include "gbmx_usage.c"

