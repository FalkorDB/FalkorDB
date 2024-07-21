// SuiteSparse:GraphBLAS 9.2.0
//------------------------------------------------------------------------------
// GraphBLAS.h: definitions for the GraphBLAS package
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2024, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS is a complete implementation of the GraphBLAS
// standard, which defines a set of sparse matrix operations on an extended
// algebra of semirings, using an almost unlimited variety of operators and
// types.  When applied to sparse adjacency matrices, these algebraic
// operations are equivalent to computations on graphs.  GraphBLAS provides a
// powerful and expressive framework creating graph algorithms based on the
// elegant mathematics of sparse matrix operations on a semiring.

// This GraphBLAS.h file contains GraphBLAS definitions for user applications
// to #include.  A few functions and variables with the prefix GB_ need to be
// defined in this file and are thus technically visible to the user, but they
// must not be accessed in user code.  They are here only so that the C11
// _Generic feature can be used in the user-accessible polymorphic functions,
// or to implement a fast GxB_Iterator using macros.

// This implementation conforms to the GraphBLAS API Specification and also
// includes functions and features that are extensions to the spec, which are
// given names of the form GxB_* for functions, built-in objects, and macros,
// so it is clear which are in the spec and which are extensions.  Extensions
// with the name GxB_* are user-accessible in SuiteSparse:GraphBLAS but cannot
// be guaranteed to appear in all GraphBLAS implementations.

// Regarding "historical" functions and symbols:  when a GxB_* function or
// symbol is added to the C API Specification, the new GrB_* name should be
// used instead.  The old GxB_* name will be kept in working order for
// historical reasons; it might no longer be mentioned in the user guide.
// Historical functions and symbols would only be removed in the rare case that
// they cause a serious conflict with future methods.

#ifndef GRAPHBLAS_H
#define GRAPHBLAS_H

//==============================================================================
//=== GraphBLAS macros, typedefs, enums, and global variables  =================
//==============================================================================

// This GraphBLAS.h header file is split into two parts.  The first part
// defines macros, typedefs, enums, global variables, and pulls in standard C
// #include files.  The second part defines all of the user-callable GrB_*,
// with access controlled by the internal definition, GB_CUDA_FOLDER.

//------------------------------------------------------------------------------
// include files required by GraphBLAS
//------------------------------------------------------------------------------

#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdint.h>
#include <inttypes.h>
#include <stddef.h>
#include <limits.h>
#include <math.h>
#include <stdarg.h>

//------------------------------------------------------------------------------
// helper macros
//------------------------------------------------------------------------------

// These are required for internal use in GraphBLAS.h, but should not be used
// by the end user application.

#define GB_CAT2(x,y) x ## y
#define GB_EVAL2(x,y) GB_CAT2 (x,y)

// GB_STR: convert the content of x into a string "x"
#define GB_XSTR(x) GB_STR(x)
#define GB_STR(x) #x

//------------------------------------------------------------------------------
// renaming for use in MATLAB R2021a or later
//------------------------------------------------------------------------------

#if defined ( GBMATLAB ) && !defined ( GB_JIT_RUNTIME )
    // All symbols must be renamed for the @GrB interface when using MATLAB
    // R2021a and following, since those versions include an earlier version of
    // SuiteSparse:GraphBLAS.  The renaming does not need to occur for the JIT
    // kernels, however.  Those can be shared between MATLAB and non-MATLAB
    // applications.
    #define GB(x)   GB_EVAL2 (GM_, x)
    #define GRB(x)  GB_EVAL2 (GrM_, x)
    #define GXB(x)  GB_EVAL2 (GxM_, x)
    #define GrB GrM
    #define GxB GxM
    #include "GB_rename.h"
#else
    // Use the standard GraphBLAS prefix.
    #define GB(x)   GB_EVAL2 (GB_, x)
    #define GRB(x)  GB_EVAL2 (GrB_, x)
    #define GXB(x)  GB_EVAL2 (GxB_, x)
#endif

//==============================================================================
// compiler variations
//==============================================================================

// GB_GLOBAL: for declaring global variables visible to the user application.
// These are not used for functions, just global variables like the predefined
// operators (GrB_PLUS_FP32), types, monoids, semirings, and descriptors.
#if defined (_MSC_VER) && !(defined (__INTEL_COMPILER) || defined(__INTEL_CLANG_COMPILER))
    #if defined ( GB_DLL_EXPORT )
        // Compiling SuiteSparse:GraphBLAS as a Windows DLL, exporting symbols
        // to user apps.
        #define GB_GLOBAL extern __declspec ( dllexport )
    #elif defined ( GB_STATIC )
        // Compiling the user application on Windows, importing symbols from
        // a static GraphBLAS library on Windows. The user application must
        // define GB_STATIC (e.g., with the pre-processor flag -DGB_STATIC) for
        // all compilation units that include "GraphBLAS.h".
        #define GB_GLOBAL extern
    #else
        // Compiling the user application on Windows, importing symbols from
        // the SuiteSparse:GraphBLAS DLL.  This is the default.
        #define GB_GLOBAL extern __declspec ( dllimport )
    #endif
#else
    // for other compilers
    #define GB_GLOBAL extern
#endif

// GraphBLAS requires an C11 compiler for its polymorphic functions (using
// the _Generic keyword), but it can be used in an C90 compiler if those
// functions are disabled.

// With C11 and later, _Generic keyword and polymorphic functions can be
// used.  Earlier versions of the language do not have this feature.

#ifdef __STDC_VERSION__
// C17: 201710L
// C11: 201112L
// C99: 199901L
// C95: 199409L
#define GxB_STDC_VERSION __STDC_VERSION__
#else
// assume C90 / C89
#define GxB_STDC_VERSION 199001L
#endif

//------------------------------------------------------------------------------
// CUDA (currently experimental, not for production use)
//------------------------------------------------------------------------------

#ifndef GRAPHBLAS_HAS_CUDA
/* #undef GRAPHBLAS_HAS_CUDA */
#endif

//------------------------------------------------------------------------------
// definitions for complex types
//------------------------------------------------------------------------------

#ifndef GXB_COMPLEX_H
#define GXB_COMPLEX_H

// Compiler has support for C99 floating point number arithmetic
#define GxB_HAVE_COMPLEX_C99

// Compiler has support for MSVC-style complex numbers
/* #undef GxB_HAVE_COMPLEX_MSVC */

    #if defined (GxB_HAVE_COMPLEX_MSVC)

        // Microsoft Windows complex types for C
        #include <complex.h>
        typedef _Fcomplex GxB_FC32_t ;
        typedef _Dcomplex GxB_FC64_t ;
        #define GxB_CMPLXF(r,i) (_FCbuild (r,i))
        #define GxB_CMPLX(r,i)  ( _Cbuild (r,i))
        #define GB_HAS_CMPLX_MACROS 1

    #elif defined (GxB_HAVE_COMPLEX_C99)

        // C11 complex types
        #include <complex.h>
        typedef float  _Complex GxB_FC32_t ;
        typedef double _Complex GxB_FC64_t ;
        #if (defined (CMPLX) && defined (CMPLXF))
            // use the C11 CMPLX and CMPLXF macros
            #define GxB_CMPLX(r,i) CMPLX (r,i)
            #define GxB_CMPLXF(r,i) CMPLXF (r,i)
            #define GB_HAS_CMPLX_MACROS 1
        #else
            // gcc 6.2 on the the Mac doesn't #define CMPLX
            #define GB_HAS_CMPLX_MACROS 0
            #define GxB_CMPLX(r,i) \
            ((GxB_FC64_t)((double)(r)) + (GxB_FC64_t)((double)(i) * _Complex_I))
            #define GxB_CMPLXF(r,i) \
            ((GxB_FC32_t)((float)(r)) + (GxB_FC32_t)((float)(i) * _Complex_I))
        #endif

    #else

        #error "Unknown or unsupported complex number arithmetic"

    #endif
#endif

//------------------------------------------------------------------------------
// restrict keyword
//------------------------------------------------------------------------------

#undef GB_restrict
#if defined ( __cplusplus )
    #define GB_restrict
#elif defined (_MSC_VER) && !(defined (__INTEL_COMPILER) || defined(__INTEL_CLANG_COMPILER))
    #define GB_restrict __restrict
#elif defined ( __NVCC__ )
    // NVIDIA nvcc
    #define GB_restrict __restrict__
#elif GxB_STDC_VERSION >= 199901L
    // C99 or later
    #define GB_restrict restrict
#else
    // C95 and earlier: no restrict keyword
    #define GB_restrict
#endif

//==============================================================================
// version control
//==============================================================================

// There are two version numbers that user codes can check against with
// compile-time #if tests:  the version of this GraphBLAS implementation,
// and the version of the GraphBLAS specification it conforms to.  User code
// can use tests like this:
//
//      #if GxB_SPEC_VERSION >= GxB_VERSION (2,0,3)
//      ... use features in GraphBLAS specification 2.0.3 ...
//      #else
//      ... only use features in early specifications
//      #endif
//
//      #if GxB_IMPLEMENTATION > GxB_VERSION (1,4,0)
//      ... use features from version 1.4.0 of a GraphBLAS package
//      #endif

// X_GRAPHBLAS: names this particular implementation:
#define GxB_SUITESPARSE_GRAPHBLAS

// GxB_VERSION: a single integer for comparing spec and version levels
#define GxB_VERSION(major,minor,sub) \
    (((major)*1000ULL + (minor))*1000ULL + (sub))

// The version of this implementation, and the GraphBLAS API version:
#define GxB_IMPLEMENTATION_NAME "SuiteSparse:GraphBLAS"
#define GxB_IMPLEMENTATION_DATE "May 22, 2024"
#define GxB_IMPLEMENTATION_MAJOR 9
#define GxB_IMPLEMENTATION_MINOR 2
#define GxB_IMPLEMENTATION_SUB   0
#define GxB_SPEC_DATE "Dec 22, 2023"
#define GxB_SPEC_MAJOR 2
#define GxB_SPEC_MINOR 1
#define GxB_SPEC_SUB   0

// compile-time access to the C API Version number of this library.
#define GRB_VERSION     GxB_SPEC_MAJOR
#define GRB_SUBVERSION  GxB_SPEC_MINOR

#define GxB_IMPLEMENTATION \
        GxB_VERSION (GxB_IMPLEMENTATION_MAJOR, \
                     GxB_IMPLEMENTATION_MINOR, \
                     GxB_IMPLEMENTATION_SUB)

// The 'about' string the describes this particular implementation of GraphBLAS:
#define GxB_IMPLEMENTATION_ABOUT \
"SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2024, All Rights Reserved." \
"\nhttp://suitesparse.com  Dept of Computer Sci. & Eng, Texas A&M University.\n"

// The GraphBLAS license for this particular implementation of GraphBLAS:
#define GxB_IMPLEMENTATION_LICENSE \
"SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2024, All Rights Reserved." \
"\nLicensed under the Apache License, Version 2.0 (the \"License\"); you may\n"\
"not use SuiteSparse:GraphBLAS except in compliance with the License.  You\n"  \
"may obtain a copy of the License at\n\n"                                      \
"    http://www.apache.org/licenses/LICENSE-2.0\n\n"                           \
"Unless required by applicable law or agreed to in writing, software\n"        \
"distributed under the License is distributed on an \"AS IS\" BASIS,\n"        \
"WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n"   \
"See the License for the specific language governing permissions and\n"        \
"limitations under the License.\n"

//------------------------------------------------------------------------------
// GraphBLAS C API version
//------------------------------------------------------------------------------

#define GxB_SPEC_VERSION GxB_VERSION(GxB_SPEC_MAJOR,GxB_SPEC_MINOR,GxB_SPEC_SUB)

// The 'spec' string describes the GraphBLAS spec:
#define GxB_SPEC_ABOUT \
"GraphBLAS C API, by Benjamin Brock, Aydin Buluc, Raye Kimmerer,\n" \
"Jim Kitchen, Manoj Kumar, Timothy Mattson, Scott McMillan, Jose' Moreira,\n" \
"Erik Welch, and Carl Yang.  Based on 'GraphBLAS Mathematics by Jeremy\n" \
"Kepner.  See also 'Graph Algorithms in the Language of Linear Algebra,'\n" \
"edited by J. Kepner and J. Gilbert, SIAM, 2011.\n"

//==============================================================================
// GrB_Index: the GraphBLAS integer
//==============================================================================

// GrB_Index: row or column index, or matrix dimension.  This typedef is used
// for row and column indices, or matrix and vector dimensions.

typedef uint64_t GrB_Index ;

// GrB_INDEX_MAX is the largest permissible index value.  The largest valid
// matrix or vector dimension is GrB_INDEX_MAX+1, or 2^60 in SuiteSparse:GrB.
#define GrB_INDEX_MAX ((GrB_Index) (1ULL << 60) - 1)

// GxB_INDEX_MAX is historical; use GrB_INDEX_MAX+1 instead.  It differs by one
// from GrB_INDEX_MAX, since it defined the largest valid matrix or vector
// dimension.
#define GxB_INDEX_MAX ((GrB_Index) (1ULL << 60))

//==============================================================================
// GraphBLAS error and informational codes
//==============================================================================

// All GraphBLAS functions return a code that indicates if it was successful
// or not.  If more information is required, the GrB_error function can be
// called, which returns a string that provides more information on the last
// return value from GraphBLAS.

typedef enum
{

    GrB_SUCCESS = 0,            // all is well

    //--------------------------------------------------------------------------
    // informational codes, not an error:
    //--------------------------------------------------------------------------

    GrB_NO_VALUE = 1,           // A(i,j) requested but not there
    GxB_EXHAUSTED = 7089,       // iterator is exhausted

    //--------------------------------------------------------------------------
    // errors:
    //--------------------------------------------------------------------------

    GrB_UNINITIALIZED_OBJECT = -1,  // object has not been initialized
    GrB_NULL_POINTER = -2,          // input pointer is NULL
    GrB_INVALID_VALUE = -3,         // generic error; some value is bad
    GrB_INVALID_INDEX = -4,         // row or column index is out of bounds
    GrB_DOMAIN_MISMATCH = -5,       // object domains are not compatible
    GrB_DIMENSION_MISMATCH = -6,    // matrix dimensions do not match
    GrB_OUTPUT_NOT_EMPTY = -7,      // output matrix already has values
    GrB_NOT_IMPLEMENTED = -8,       // method not implemented
    GrB_ALREADY_SET = -9,           // field already written to
    GrB_PANIC = -101,               // unknown error
    GrB_OUT_OF_MEMORY = -102,       // out of memory
    GrB_INSUFFICIENT_SPACE = -103,  // output array not large enough
    GrB_INVALID_OBJECT = -104,      // object is corrupted
    GrB_INDEX_OUT_OF_BOUNDS = -105, // row or col index out of bounds
    GrB_EMPTY_OBJECT = -106         // an object does not contain a value

}
GrB_Info ;

// for null and invalid objects
#define GrB_NULL NULL
#define GrB_INVALID_HANDLE NULL

//==============================================================================
// GrB_init / GrB_finalize
//==============================================================================

// GrB_init must called before any other GraphBLAS operation.  GrB_finalize
// must be called as the last GraphBLAS operation.

// GrB_init defines the mode that GraphBLAS will use:  blocking or
// non-blocking.  With blocking mode, all operations finish before returning to
// the user application.  With non-blocking mode, operations can be left
// pending, and are computed only when needed.

// The extension GxB_init does the work of GrB_init, but it also defines the
// memory management functions that SuiteSparse:GraphBLAS will use internally.

typedef enum
{
    GrB_NONBLOCKING = 0,        // methods may return with pending computations
    GrB_BLOCKING = 1,           // no computations are ever left pending
//  DRAFT: in progress, do not use:
    GxB_NONBLOCKING_GPU = 7099, // non-blocking mode, allow use of GPU(s)
    GxB_BLOCKING_GPU = 7098,    // blocking mode, allow use of GPU(s)
}
GrB_Mode ;

//==============================================================================
// GrB_Descriptor: the GraphBLAS descriptor
//==============================================================================

// The GrB_Descriptor is used to modify the behavior of GraphBLAS operations.
//
// GrB_OUTP: can be GrB_DEFAULT or GrB_REPLACE.  If GrB_REPLACE, then C is
//       cleared after taking part in the accum operation but before the mask.
//       In other words, C<Mask> = accum (C,T) is split into Z = accum(C,T) ;
//       C=0 ; C<Mask> = Z.
//
// GrB_MASK: can be GrB_DEFAULT, GrB_COMP, GrB_STRUCTURE, or set to both
//      GrB_COMP and GrB_STRUCTURE.  If GrB_DEFAULT, the mask is used
//      normally, where Mask(i,j)=1 means C(i,j) can be modified by C<Mask>=Z,
//      and Mask(i,j)=0 means it cannot be modified even if Z(i,j) is has been
//      computed and differs from C(i,j).  If GrB_COMP, this is the same as
//      taking the logical complement of the Mask.  If GrB_STRUCTURE is set,
//      the value of the mask is not considered, just its pattern.  The
//      GrB_COMP and GrB_STRUCTURE settings can be combined.
//
// GrB_INP0: can be GrB_DEFAULT or GrB_TRAN.  If GrB_DEFAULT, the first input
//      is used as-is.  If GrB_TRAN, it is transposed.  Only matrices are
//      transposed this way.  Vectors are never transposed via the
//      GrB_Descriptor.
//
// GrB_INP1: the same as GrB_INP0 but for the second input
//
// GxB_AxB_METHOD: this is a hint to SuiteSparse:GraphBLAS on which algorithm
//      it should use to compute C=A*B, in GrB_mxm, GrB_mxv, and GrB_vxm.
//      SuiteSparse:GraphBLAS has four different heuristics, and the default
//      method (GrB_DEFAULT) selects between them automatically.  The complete
//      rule is in the User Guide.  The brief discussion here assumes all
//      matrices are stored by column.  All methods compute the same result,
//      except that floating-point roundoff may differ when working on
//      floating-point data types.
//
//      GxB_AxB_SAXPY:  C(:,j)=A*B(:,j) is computed using a mix of Gustavson
//          and Hash methods.  Each task in the parallel computation makes its
//          own decision between these two methods, via a heuristic.
//
//      GxB_AxB_GUSTAVSON:  This is the same as GxB_AxB_SAXPY, except that
//          every task uses Gustavon's method, computing C(:,j)=A*B(:,j) via a
//          gather/scatter workspace of size equal to the number of rows of A.
//          Very good general-purpose method, but sometimes the workspace can
//          be too large when many threads are used.
//
//      GxB_AxB_HASH: This is the same as GxB_AxB_SAXPY, except that every
//          task uses the Hash method.  It is very good for hypersparse
//          matrices and uses very little workspace, and so it scales well to
//          many threads.
//
//      GxB_AxB_DOT: computes C(i,j) = A(:,i)'*B(:,j), for each entry C(i,j).
//          A very specialized method that works well only if the mask is
//          present, very sparse, and not complemented, or when C is a dense
//          vector or matrix, or when C is small.
//
// GxB_SORT: GrB_mxm and other methods may return a matrix in a 'jumbled'
//      state, with indices out of order.  The sort is left pending.  Some
//      methods can tolerate jumbled matrices on input, so this can be faster.
//      However, in some cases, it can be faster for GrB_mxm to sort its output
//      as it is computed.  With GxB_SORT set to GrB_DEFAULT, the sort is left
//      pending.  With GxB_SORT set to a nonzero value, GrB_mxm typically sorts
//      the resulting matrix C (but not always; this is just a hint).  If
//      GrB_init is called with GrB_BLOCKING mode, the sort will always be
//      done, and this setting has no effect.
//
// GxB_COMPRESSION: compression method for GxB_Matrix_serialize and
//      GxB_Vector_serialize.  The default is ZSTD (level 1).
//
// GxB_IMPORT:  GxB_FAST_IMPORT (faster, for trusted input data) or
//      GxB_SECURE_IMPORT (slower, for untrusted input data), for the
//      GxB*_pack* methods.

// The following are enumerated values in both the GrB_Desc_Field and the
// GxB_Option_Field for global options.  They are defined with the same integer
// value for both enums, so the user can use them for both.
#define GxB_NTHREADS 7086
#define GxB_CHUNK 7087

// GPU control (DRAFT: in progress, do not use)
#define GxB_GPU_ID 7088

typedef enum
{
    GrB_OUTP = 0,   // descriptor for output of a method
    GrB_MASK = 1,   // descriptor for the mask input of a method
    GrB_INP0 = 2,   // descriptor for the first input of a method
    GrB_INP1 = 3,   // descriptor for the second input of a method
    GxB_AxB_METHOD = 7090,  // descriptor for selecting C=A*B algorithm
    GxB_SORT = 7091,          // control sort in GrB_mxm
    GxB_COMPRESSION = 7092,   // select compression for serialize
    GxB_IMPORT = 7093,        // secure vs fast import
}
GrB_Desc_Field ;

typedef enum
{
    // for all GrB_Descriptor fields:
    GrB_DEFAULT = 0,    // default behavior of the method
    GxB_DEFAULT = 0,    // historical; use GrB_DEFAULT instead

    // for GrB_OUTP only:
    GrB_REPLACE = 1,    // clear the output before assigning new values to it

    // for GrB_MASK only:
    GrB_COMP = 2,       // use the structural complement of the input
    GrB_STRUCTURE = 4,  // use the only pattern of the mask, not its values
    GrB_COMP_STRUCTURE = 6, // shorthand for GrB_COMP + GrB_STRUCTURE

    // for GrB_INP0 and GrB_INP1 only:
    GrB_TRAN = 3,       // use the transpose of the input

    // for GxB_AxB_METHOD only:
    GxB_AxB_GUSTAVSON = 7081,   // gather-scatter saxpy method
    GxB_AxB_DOT       = 7083,   // dot product
    GxB_AxB_HASH      = 7084,   // hash-based saxpy method
    GxB_AxB_SAXPY     = 7085,   // saxpy method (any kind)

    // for GxB_IMPORT only:
    GxB_SECURE_IMPORT = 7080    // GxB*_pack* methods trust their input data
}
GrB_Desc_Value ;

// default for GxB pack is to trust the input data
#define GxB_FAST_IMPORT ((int) GrB_DEFAULT)

typedef struct GB_Descriptor_opaque *GrB_Descriptor ;

// Predefined descriptors and their values:

GB_GLOBAL GrB_Descriptor
                   // OUTP         MASK           MASK       INP0      INP1
                   //              structural     complement
                   // ===========  ============== ========== ========  ========

// GrB_NULL        // -            -              -          -         -
GrB_DESC_T1      , // -            -              -          -         GrB_TRAN
GrB_DESC_T0      , // -            -              -          GrB_TRAN  -
GrB_DESC_T0T1    , // -            -              -          GrB_TRAN  GrB_TRAN

GrB_DESC_C       , // -            -              GrB_COMP   -         -
GrB_DESC_CT1     , // -            -              GrB_COMP   -         GrB_TRAN
GrB_DESC_CT0     , // -            -              GrB_COMP   GrB_TRAN  -
GrB_DESC_CT0T1   , // -            -              GrB_COMP   GrB_TRAN  GrB_TRAN

GrB_DESC_S       , // -            GrB_STRUCTURE  -          -         -
GrB_DESC_ST1     , // -            GrB_STRUCTURE  -          -         GrB_TRAN
GrB_DESC_ST0     , // -            GrB_STRUCTURE  -          GrB_TRAN  -
GrB_DESC_ST0T1   , // -            GrB_STRUCTURE  -          GrB_TRAN  GrB_TRAN

GrB_DESC_SC      , // -            GrB_STRUCTURE  GrB_COMP   -         -
GrB_DESC_SCT1    , // -            GrB_STRUCTURE  GrB_COMP   -         GrB_TRAN
GrB_DESC_SCT0    , // -            GrB_STRUCTURE  GrB_COMP   GrB_TRAN  -
GrB_DESC_SCT0T1  , // -            GrB_STRUCTURE  GrB_COMP   GrB_TRAN  GrB_TRAN

GrB_DESC_R       , // GrB_REPLACE  -              -          -         -
GrB_DESC_RT1     , // GrB_REPLACE  -              -          -         GrB_TRAN
GrB_DESC_RT0     , // GrB_REPLACE  -              -          GrB_TRAN  -
GrB_DESC_RT0T1   , // GrB_REPLACE  -              -          GrB_TRAN  GrB_TRAN

GrB_DESC_RC      , // GrB_REPLACE  -              GrB_COMP   -         -
GrB_DESC_RCT1    , // GrB_REPLACE  -              GrB_COMP   -         GrB_TRAN
GrB_DESC_RCT0    , // GrB_REPLACE  -              GrB_COMP   GrB_TRAN  -
GrB_DESC_RCT0T1  , // GrB_REPLACE  -              GrB_COMP   GrB_TRAN  GrB_TRAN

GrB_DESC_RS      , // GrB_REPLACE  GrB_STRUCTURE  -          -         -
GrB_DESC_RST1    , // GrB_REPLACE  GrB_STRUCTURE  -          -         GrB_TRAN
GrB_DESC_RST0    , // GrB_REPLACE  GrB_STRUCTURE  -          GrB_TRAN  -
GrB_DESC_RST0T1  , // GrB_REPLACE  GrB_STRUCTURE  -          GrB_TRAN  GrB_TRAN

GrB_DESC_RSC     , // GrB_REPLACE  GrB_STRUCTURE  GrB_COMP   -         -
GrB_DESC_RSCT1   , // GrB_REPLACE  GrB_STRUCTURE  GrB_COMP   -         GrB_TRAN
GrB_DESC_RSCT0   , // GrB_REPLACE  GrB_STRUCTURE  GrB_COMP   GrB_TRAN  -
GrB_DESC_RSCT0T1 ; // GrB_REPLACE  GrB_STRUCTURE  GrB_COMP   GrB_TRAN  GrB_TRAN

// GrB_NULL is the default descriptor, with all settings at their defaults:
//
//      OUTP: do not replace the output
//      MASK: mask is valued and not complemented
//      INP0: first input not transposed
//      INP1: second input not transposed

// Predefined descriptors may not be modified or freed.  Attempting to modify
// them results in an error (GrB_INVALID_VALUE).  Attempts to free them are
// silently ignored.

//==============================================================================
// GrB_Type: data types
//==============================================================================

typedef struct GB_Type_opaque *GrB_Type ;

// GraphBLAS predefined types and their counterparts in pure C:
GB_GLOBAL GrB_Type
    GrB_BOOL   ,        // in C: bool
    GrB_INT8   ,        // in C: int8_t
    GrB_INT16  ,        // in C: int16_t
    GrB_INT32  ,        // in C: int32_t
    GrB_INT64  ,        // in C: int64_t
    GrB_UINT8  ,        // in C: uint8_t
    GrB_UINT16 ,        // in C: uint16_t
    GrB_UINT32 ,        // in C: uint32_t
    GrB_UINT64 ,        // in C: uint64_t
    GrB_FP32   ,        // in C: float
    GrB_FP64   ,        // in C: double
    GxB_FC32   ,        // in C: float complex
    GxB_FC64   ;        // in C: double complex

#define GxB_MAX_NAME_LEN 128

//==============================================================================
// GrB_UnaryOp: unary operators
//==============================================================================

// GrB_UnaryOp: a function z=f(x).  The function f must have the signature:

//      void f (void *z, const void *x) ;

// The pointers are void * but they are always of pointers to objects of type
// ztype and xtype, respectively.  The function must typecast its arguments as
// needed from void* to ztype* and xtype*.

typedef struct GB_UnaryOp_opaque *GrB_UnaryOp ;

//------------------------------------------------------------------------------
// built-in unary operators, z = f(x)
//------------------------------------------------------------------------------

GB_GLOBAL GrB_UnaryOp
    // For these functions z=f(x), z and x have the same type.
    // The suffix in the name is the type of x and z.
    // z = x             z = -x             z = 1/x             z = ! (x != 0)
    // identity          additive           multiplicative      logical
    //                   inverse            inverse             negation
    GrB_IDENTITY_BOOL,   GrB_AINV_BOOL,     GrB_MINV_BOOL,      GxB_LNOT_BOOL,
    GrB_IDENTITY_INT8,   GrB_AINV_INT8,     GrB_MINV_INT8,      GxB_LNOT_INT8,
    GrB_IDENTITY_INT16,  GrB_AINV_INT16,    GrB_MINV_INT16,     GxB_LNOT_INT16,
    GrB_IDENTITY_INT32,  GrB_AINV_INT32,    GrB_MINV_INT32,     GxB_LNOT_INT32,
    GrB_IDENTITY_INT64,  GrB_AINV_INT64,    GrB_MINV_INT64,     GxB_LNOT_INT64,
    GrB_IDENTITY_UINT8,  GrB_AINV_UINT8,    GrB_MINV_UINT8,     GxB_LNOT_UINT8,
    GrB_IDENTITY_UINT16, GrB_AINV_UINT16,   GrB_MINV_UINT16,    GxB_LNOT_UINT16,
    GrB_IDENTITY_UINT32, GrB_AINV_UINT32,   GrB_MINV_UINT32,    GxB_LNOT_UINT32,
    GrB_IDENTITY_UINT64, GrB_AINV_UINT64,   GrB_MINV_UINT64,    GxB_LNOT_UINT64,
    GrB_IDENTITY_FP32,   GrB_AINV_FP32,     GrB_MINV_FP32,      GxB_LNOT_FP32,
    GrB_IDENTITY_FP64,   GrB_AINV_FP64,     GrB_MINV_FP64,      GxB_LNOT_FP64,
    // complex unary operators:
    GxB_IDENTITY_FC32,   GxB_AINV_FC32,     GxB_MINV_FC32,      // no LNOT
    GxB_IDENTITY_FC64,   GxB_AINV_FC64,     GxB_MINV_FC64,      // for complex

    // z = 1             z = abs(x)         z = bnot(x)         z = signum
    // one               absolute value     bitwise negation
    GxB_ONE_BOOL,        GrB_ABS_BOOL,
    GxB_ONE_INT8,        GrB_ABS_INT8,      GrB_BNOT_INT8,
    GxB_ONE_INT16,       GrB_ABS_INT16,     GrB_BNOT_INT16,
    GxB_ONE_INT32,       GrB_ABS_INT32,     GrB_BNOT_INT32,
    GxB_ONE_INT64,       GrB_ABS_INT64,     GrB_BNOT_INT64,
    GxB_ONE_UINT8,       GrB_ABS_UINT8,     GrB_BNOT_UINT8,
    GxB_ONE_UINT16,      GrB_ABS_UINT16,    GrB_BNOT_UINT16,
    GxB_ONE_UINT32,      GrB_ABS_UINT32,    GrB_BNOT_UINT32,
    GxB_ONE_UINT64,      GrB_ABS_UINT64,    GrB_BNOT_UINT64,
    GxB_ONE_FP32,        GrB_ABS_FP32,
    GxB_ONE_FP64,        GrB_ABS_FP64,
    // complex unary operators:
    GxB_ONE_FC32,        // for complex types, z = abs(x)
    GxB_ONE_FC64,        // is real; listed below.

    // Boolean negation, z = !x, where both z and x are boolean.  There is no
    // suffix since z and x are only boolean.  This operator is identical to
    // GxB_LNOT_BOOL; it just has a different name.
    GrB_LNOT ;

// GxB_ABS is now in the v1.3 spec, the following names are historical:
GB_GLOBAL GrB_UnaryOp

    // z = abs(x)
    GxB_ABS_BOOL,
    GxB_ABS_INT8,
    GxB_ABS_INT16,
    GxB_ABS_INT32,
    GxB_ABS_INT64,
    GxB_ABS_UINT8,
    GxB_ABS_UINT16,
    GxB_ABS_UINT32,
    GxB_ABS_UINT64,
    GxB_ABS_FP32,
    GxB_ABS_FP64 ;

//------------------------------------------------------------------------------
// Unary operators for floating-point types only
//------------------------------------------------------------------------------

// The following floating-point unary operators and their C11 equivalents,
// are only defined for floating-point (real and complex) types.

GB_GLOBAL GrB_UnaryOp

    //--------------------------------------------------------------------------
    // z = f(x) where z and x have the same type (all 4 floating-point types)
    //--------------------------------------------------------------------------

    // z = sqrt (x)     z = log (x)         z = exp (x)         z = log2 (x)
    GxB_SQRT_FP32,      GxB_LOG_FP32,       GxB_EXP_FP32,       GxB_LOG2_FP32,
    GxB_SQRT_FP64,      GxB_LOG_FP64,       GxB_EXP_FP64,       GxB_LOG2_FP64,
    GxB_SQRT_FC32,      GxB_LOG_FC32,       GxB_EXP_FC32,       GxB_LOG2_FC32,
    GxB_SQRT_FC64,      GxB_LOG_FC64,       GxB_EXP_FC64,       GxB_LOG2_FC64,

    // z = sin (x)      z = cos (x)         z = tan (x)
    GxB_SIN_FP32,       GxB_COS_FP32,       GxB_TAN_FP32,
    GxB_SIN_FP64,       GxB_COS_FP64,       GxB_TAN_FP64,
    GxB_SIN_FC32,       GxB_COS_FC32,       GxB_TAN_FC32,
    GxB_SIN_FC64,       GxB_COS_FC64,       GxB_TAN_FC64,

    // z = acos (x)     z = asin (x)        z = atan (x)
    GxB_ACOS_FP32,      GxB_ASIN_FP32,      GxB_ATAN_FP32,
    GxB_ACOS_FP64,      GxB_ASIN_FP64,      GxB_ATAN_FP64,
    GxB_ACOS_FC32,      GxB_ASIN_FC32,      GxB_ATAN_FC32,
    GxB_ACOS_FC64,      GxB_ASIN_FC64,      GxB_ATAN_FC64,

    // z = sinh (x)     z = cosh (x)        z = tanh (x)
    GxB_SINH_FP32,      GxB_COSH_FP32,      GxB_TANH_FP32,
    GxB_SINH_FP64,      GxB_COSH_FP64,      GxB_TANH_FP64,
    GxB_SINH_FC32,      GxB_COSH_FC32,      GxB_TANH_FC32,
    GxB_SINH_FC64,      GxB_COSH_FC64,      GxB_TANH_FC64,

    // z = acosh (x)    z = asinh (x)       z = atanh (x)       z = signum (x)
    GxB_ACOSH_FP32,     GxB_ASINH_FP32,     GxB_ATANH_FP32,     GxB_SIGNUM_FP32,
    GxB_ACOSH_FP64,     GxB_ASINH_FP64,     GxB_ATANH_FP64,     GxB_SIGNUM_FP64,
    GxB_ACOSH_FC32,     GxB_ASINH_FC32,     GxB_ATANH_FC32,     GxB_SIGNUM_FC32,
    GxB_ACOSH_FC64,     GxB_ASINH_FC64,     GxB_ATANH_FC64,     GxB_SIGNUM_FC64,

    // z = ceil (x)     z = floor (x)       z = round (x)       z = trunc (x)
    GxB_CEIL_FP32,      GxB_FLOOR_FP32,     GxB_ROUND_FP32,     GxB_TRUNC_FP32,
    GxB_CEIL_FP64,      GxB_FLOOR_FP64,     GxB_ROUND_FP64,     GxB_TRUNC_FP64,
    GxB_CEIL_FC32,      GxB_FLOOR_FC32,     GxB_ROUND_FC32,     GxB_TRUNC_FC32,
    GxB_CEIL_FC64,      GxB_FLOOR_FC64,     GxB_ROUND_FC64,     GxB_TRUNC_FC64,

    // z = exp2 (x)     z = expm1 (x)       z = log10 (x)       z = log1p (x)
    GxB_EXP2_FP32,      GxB_EXPM1_FP32,     GxB_LOG10_FP32,     GxB_LOG1P_FP32,
    GxB_EXP2_FP64,      GxB_EXPM1_FP64,     GxB_LOG10_FP64,     GxB_LOG1P_FP64,
    GxB_EXP2_FC32,      GxB_EXPM1_FC32,     GxB_LOG10_FC32,     GxB_LOG1P_FC32,
    GxB_EXP2_FC64,      GxB_EXPM1_FC64,     GxB_LOG10_FC64,     GxB_LOG1P_FC64,

    //--------------------------------------------------------------------------
    // z = f(x) where z and x are the same type (floating-point real only)
    //--------------------------------------------------------------------------

    // z = lgamma (x)   z = tgamma (x)      z = erf (x)         z = erfc (x)
    GxB_LGAMMA_FP32,    GxB_TGAMMA_FP32,    GxB_ERF_FP32,       GxB_ERFC_FP32,
    GxB_LGAMMA_FP64,    GxB_TGAMMA_FP64,    GxB_ERF_FP64,       GxB_ERFC_FP64,

    // z = cbrt (x)
    GxB_CBRT_FP32,
    GxB_CBRT_FP64,

    // frexpx and frexpe return the mantissa and exponent, respectively,
    // from the C11 frexp function.  The exponent is returned as a
    // floating-point value, not an integer.

    // z = frexpx (x)   z = frexpe (x)
    GxB_FREXPX_FP32,    GxB_FREXPE_FP32,
    GxB_FREXPX_FP64,    GxB_FREXPE_FP64,

    //--------------------------------------------------------------------------
    // z = f(x) where z and x are the same type (complex only)
    //--------------------------------------------------------------------------

    // z = conj (x)
    GxB_CONJ_FC32,
    GxB_CONJ_FC64,

    //--------------------------------------------------------------------------
    // z = f(x) where z is real and x is complex:
    //--------------------------------------------------------------------------

    // z = creal (x)    z = cimag (x)       z = carg (x)       z = abs (x)
    GxB_CREAL_FC32,     GxB_CIMAG_FC32,     GxB_CARG_FC32,     GxB_ABS_FC32,
    GxB_CREAL_FC64,     GxB_CIMAG_FC64,     GxB_CARG_FC64,     GxB_ABS_FC64,

    //--------------------------------------------------------------------------
    // z = f(x) where z is bool and x is any floating-point type
    //--------------------------------------------------------------------------

    // z = isinf (x)
    GxB_ISINF_FP32,
    GxB_ISINF_FP64,
    GxB_ISINF_FC32,     // isinf (creal (x)) || isinf (cimag (x))
    GxB_ISINF_FC64,     // isinf (creal (x)) || isinf (cimag (x))

    // z = isnan (x)
    GxB_ISNAN_FP32,
    GxB_ISNAN_FP64,
    GxB_ISNAN_FC32,     // isnan (creal (x)) || isnan (cimag (x))
    GxB_ISNAN_FC64,     // isnan (creal (x)) || isnan (cimag (x))

    // z = isfinite (x)
    GxB_ISFINITE_FP32,
    GxB_ISFINITE_FP64,
    GxB_ISFINITE_FC32,  // isfinite (real (x)) && isfinite (cimag (x))
    GxB_ISFINITE_FC64 ; // isfinite (real (x)) && isfinite (cimag (x))

typedef void (*GxB_unary_function)  (void *, const void *) ;

//==============================================================================
// GrB_BinaryOp: binary operators
//==============================================================================

// GrB_BinaryOp: a function z=f(x,y).  The function f must have the signature:

//      void f (void *z, const void *x, const void *y) ;

// The pointers are void * but they are always of pointers to objects of type
// ztype, xtype, and ytype, respectively.  See Demo/usercomplex.c for examples.

typedef struct GB_BinaryOp_opaque *GrB_BinaryOp ;

//------------------------------------------------------------------------------
// built-in binary operators, z = f(x,y), where x,y,z all have the same type
//------------------------------------------------------------------------------

GB_GLOBAL GrB_BinaryOp

    // operators for all 13 types (including complex):

    // GxB_PAIR_T and GrB_ONEB_T are identical; the latter was added to the
    // v2.0 C API Specification.

    // z = x            z = y               z = 1               z = pow (x,y)
    GrB_FIRST_BOOL,     GrB_SECOND_BOOL,    GrB_ONEB_BOOL,      GxB_POW_BOOL,
    GrB_FIRST_INT8,     GrB_SECOND_INT8,    GrB_ONEB_INT8,      GxB_POW_INT8,
    GrB_FIRST_INT16,    GrB_SECOND_INT16,   GrB_ONEB_INT16,     GxB_POW_INT16,
    GrB_FIRST_INT32,    GrB_SECOND_INT32,   GrB_ONEB_INT32,     GxB_POW_INT32,
    GrB_FIRST_INT64,    GrB_SECOND_INT64,   GrB_ONEB_INT64,     GxB_POW_INT64,
    GrB_FIRST_UINT8,    GrB_SECOND_UINT8,   GrB_ONEB_UINT8,     GxB_POW_UINT8,
    GrB_FIRST_UINT16,   GrB_SECOND_UINT16,  GrB_ONEB_UINT16,    GxB_POW_UINT16,
    GrB_FIRST_UINT32,   GrB_SECOND_UINT32,  GrB_ONEB_UINT32,    GxB_POW_UINT32,
    GrB_FIRST_UINT64,   GrB_SECOND_UINT64,  GrB_ONEB_UINT64,    GxB_POW_UINT64,
    GrB_FIRST_FP32,     GrB_SECOND_FP32,    GrB_ONEB_FP32,      GxB_POW_FP32,
    GrB_FIRST_FP64,     GrB_SECOND_FP64,    GrB_ONEB_FP64,      GxB_POW_FP64,
    // complex:
    GxB_FIRST_FC32,     GxB_SECOND_FC32,    GxB_ONEB_FC32,      GxB_POW_FC32,
    GxB_FIRST_FC64,     GxB_SECOND_FC64,    GxB_ONEB_FC64,      GxB_POW_FC64,

    // z = x+y          z = x-y             z = x*y             z = x/y
    GrB_PLUS_BOOL,      GrB_MINUS_BOOL,     GrB_TIMES_BOOL,     GrB_DIV_BOOL,
    GrB_PLUS_INT8,      GrB_MINUS_INT8,     GrB_TIMES_INT8,     GrB_DIV_INT8,
    GrB_PLUS_INT16,     GrB_MINUS_INT16,    GrB_TIMES_INT16,    GrB_DIV_INT16,
    GrB_PLUS_INT32,     GrB_MINUS_INT32,    GrB_TIMES_INT32,    GrB_DIV_INT32,
    GrB_PLUS_INT64,     GrB_MINUS_INT64,    GrB_TIMES_INT64,    GrB_DIV_INT64,
    GrB_PLUS_UINT8,     GrB_MINUS_UINT8,    GrB_TIMES_UINT8,    GrB_DIV_UINT8,
    GrB_PLUS_UINT16,    GrB_MINUS_UINT16,   GrB_TIMES_UINT16,   GrB_DIV_UINT16,
    GrB_PLUS_UINT32,    GrB_MINUS_UINT32,   GrB_TIMES_UINT32,   GrB_DIV_UINT32,
    GrB_PLUS_UINT64,    GrB_MINUS_UINT64,   GrB_TIMES_UINT64,   GrB_DIV_UINT64,
    GrB_PLUS_FP32,      GrB_MINUS_FP32,     GrB_TIMES_FP32,     GrB_DIV_FP32,
    GrB_PLUS_FP64,      GrB_MINUS_FP64,     GrB_TIMES_FP64,     GrB_DIV_FP64,
    // complex:
    GxB_PLUS_FC32,      GxB_MINUS_FC32,     GxB_TIMES_FC32,     GxB_DIV_FC32,
    GxB_PLUS_FC64,      GxB_MINUS_FC64,     GxB_TIMES_FC64,     GxB_DIV_FC64,

    // z = y-x          z = y/x             z = 1               z = any(x,y)
    GxB_RMINUS_BOOL,    GxB_RDIV_BOOL,      GxB_PAIR_BOOL,      GxB_ANY_BOOL,
    GxB_RMINUS_INT8,    GxB_RDIV_INT8,      GxB_PAIR_INT8,      GxB_ANY_INT8,
    GxB_RMINUS_INT16,   GxB_RDIV_INT16,     GxB_PAIR_INT16,     GxB_ANY_INT16,
    GxB_RMINUS_INT32,   GxB_RDIV_INT32,     GxB_PAIR_INT32,     GxB_ANY_INT32,
    GxB_RMINUS_INT64,   GxB_RDIV_INT64,     GxB_PAIR_INT64,     GxB_ANY_INT64,
    GxB_RMINUS_UINT8,   GxB_RDIV_UINT8,     GxB_PAIR_UINT8,     GxB_ANY_UINT8,
    GxB_RMINUS_UINT16,  GxB_RDIV_UINT16,    GxB_PAIR_UINT16,    GxB_ANY_UINT16,
    GxB_RMINUS_UINT32,  GxB_RDIV_UINT32,    GxB_PAIR_UINT32,    GxB_ANY_UINT32,
    GxB_RMINUS_UINT64,  GxB_RDIV_UINT64,    GxB_PAIR_UINT64,    GxB_ANY_UINT64,
    GxB_RMINUS_FP32,    GxB_RDIV_FP32,      GxB_PAIR_FP32,      GxB_ANY_FP32,
    GxB_RMINUS_FP64,    GxB_RDIV_FP64,      GxB_PAIR_FP64,      GxB_ANY_FP64,
    // complex:
    GxB_RMINUS_FC32,    GxB_RDIV_FC32,      GxB_PAIR_FC32,      GxB_ANY_FC32,
    GxB_RMINUS_FC64,    GxB_RDIV_FC64,      GxB_PAIR_FC64,      GxB_ANY_FC64,

    // The GxB_IS* comparators z=f(x,y) return the same type as their
    // inputs.  Each of them compute z = (x OP y), where x, y, and z all have
    // the same type.  The value z is either 1 for true or 0 for false, but it
    // is a value with the same type as x and y.

    // z = (x == y)     z = (x != y)
    GxB_ISEQ_BOOL,      GxB_ISNE_BOOL,
    GxB_ISEQ_INT8,      GxB_ISNE_INT8,
    GxB_ISEQ_INT16,     GxB_ISNE_INT16,
    GxB_ISEQ_INT32,     GxB_ISNE_INT32,
    GxB_ISEQ_INT64,     GxB_ISNE_INT64,
    GxB_ISEQ_UINT8,     GxB_ISNE_UINT8,
    GxB_ISEQ_UINT16,    GxB_ISNE_UINT16,
    GxB_ISEQ_UINT32,    GxB_ISNE_UINT32,
    GxB_ISEQ_UINT64,    GxB_ISNE_UINT64,
    GxB_ISEQ_FP32,      GxB_ISNE_FP32,
    GxB_ISEQ_FP64,      GxB_ISNE_FP64,
    // complex:
    GxB_ISEQ_FC32,      GxB_ISNE_FC32,
    GxB_ISEQ_FC64,      GxB_ISNE_FC64,

    // z = (x > y)      z = (x < y)         z = (x >= y)     z = (x <= y)
    GxB_ISGT_BOOL,      GxB_ISLT_BOOL,      GxB_ISGE_BOOL,      GxB_ISLE_BOOL,
    GxB_ISGT_INT8,      GxB_ISLT_INT8,      GxB_ISGE_INT8,      GxB_ISLE_INT8,
    GxB_ISGT_INT16,     GxB_ISLT_INT16,     GxB_ISGE_INT16,     GxB_ISLE_INT16,
    GxB_ISGT_INT32,     GxB_ISLT_INT32,     GxB_ISGE_INT32,     GxB_ISLE_INT32,
    GxB_ISGT_INT64,     GxB_ISLT_INT64,     GxB_ISGE_INT64,     GxB_ISLE_INT64,
    GxB_ISGT_UINT8,     GxB_ISLT_UINT8,     GxB_ISGE_UINT8,     GxB_ISLE_UINT8,
    GxB_ISGT_UINT16,    GxB_ISLT_UINT16,    GxB_ISGE_UINT16,    GxB_ISLE_UINT16,
    GxB_ISGT_UINT32,    GxB_ISLT_UINT32,    GxB_ISGE_UINT32,    GxB_ISLE_UINT32,
    GxB_ISGT_UINT64,    GxB_ISLT_UINT64,    GxB_ISGE_UINT64,    GxB_ISLE_UINT64,
    GxB_ISGT_FP32,      GxB_ISLT_FP32,      GxB_ISGE_FP32,      GxB_ISLE_FP32,
    GxB_ISGT_FP64,      GxB_ISLT_FP64,      GxB_ISGE_FP64,      GxB_ISLE_FP64,

    // z = min(x,y)     z = max (x,y)
    GrB_MIN_BOOL,       GrB_MAX_BOOL,
    GrB_MIN_INT8,       GrB_MAX_INT8,
    GrB_MIN_INT16,      GrB_MAX_INT16,
    GrB_MIN_INT32,      GrB_MAX_INT32,
    GrB_MIN_INT64,      GrB_MAX_INT64,
    GrB_MIN_UINT8,      GrB_MAX_UINT8,
    GrB_MIN_UINT16,     GrB_MAX_UINT16,
    GrB_MIN_UINT32,     GrB_MAX_UINT32,
    GrB_MIN_UINT64,     GrB_MAX_UINT64,
    GrB_MIN_FP32,       GrB_MAX_FP32,
    GrB_MIN_FP64,       GrB_MAX_FP64,

    // Binary operators for each of the 11 real types:

    // The operators convert non-boolean types internally to boolean and return
    // a value 1 or 0 in the same type, for true or false.  Each computes z =
    // ((x != 0) OP (y != 0)), where x, y, and z all the same type.  These
    // operators are useful as multiplicative operators when combined with
    // non-boolean monoids of the same type.

    // z = (x || y)     z = (x && y)        z = (x != y)
    GxB_LOR_BOOL,       GxB_LAND_BOOL,      GxB_LXOR_BOOL,
    GxB_LOR_INT8,       GxB_LAND_INT8,      GxB_LXOR_INT8,
    GxB_LOR_INT16,      GxB_LAND_INT16,     GxB_LXOR_INT16,
    GxB_LOR_INT32,      GxB_LAND_INT32,     GxB_LXOR_INT32,
    GxB_LOR_INT64,      GxB_LAND_INT64,     GxB_LXOR_INT64,
    GxB_LOR_UINT8,      GxB_LAND_UINT8,     GxB_LXOR_UINT8,
    GxB_LOR_UINT16,     GxB_LAND_UINT16,    GxB_LXOR_UINT16,
    GxB_LOR_UINT32,     GxB_LAND_UINT32,    GxB_LXOR_UINT32,
    GxB_LOR_UINT64,     GxB_LAND_UINT64,    GxB_LXOR_UINT64,
    GxB_LOR_FP32,       GxB_LAND_FP32,      GxB_LXOR_FP32,
    GxB_LOR_FP64,       GxB_LAND_FP64,      GxB_LXOR_FP64,

    // Binary operators that operate only on boolean types: LOR, LAND, LXOR,
    // and LXNOR.  The naming convention differs (_BOOL is not appended to the
    // name).  They are the same as GxB_LOR_BOOL, GxB_LAND_BOOL, and
    // GxB_LXOR_BOOL, and GrB_EQ_BOOL, respectively.

    // z = (x || y)     z = (x && y)        z = (x != y)        z = (x == y)
    GrB_LOR,            GrB_LAND,           GrB_LXOR,           GrB_LXNOR,

    // Operators for floating-point reals:

    // z = atan2(x,y)   z = hypot(x,y)      z = fmod(x,y)   z = remainder(x,y)
    GxB_ATAN2_FP32,     GxB_HYPOT_FP32,     GxB_FMOD_FP32,  GxB_REMAINDER_FP32,
    GxB_ATAN2_FP64,     GxB_HYPOT_FP64,     GxB_FMOD_FP64,  GxB_REMAINDER_FP64,

    // z = ldexp(x,y)   z = copysign (x,y)
    GxB_LDEXP_FP32,     GxB_COPYSIGN_FP32,
    GxB_LDEXP_FP64,     GxB_COPYSIGN_FP64,

    // Bitwise operations on signed and unsigned integers: note that
    // bitwise operations on signed integers can lead to different results,
    // depending on your compiler; results are implementation-defined.

    // z = (x | y)      z = (x & y)         z = (x ^ y)        z = ~(x ^ y)
    GrB_BOR_INT8,       GrB_BAND_INT8,      GrB_BXOR_INT8,     GrB_BXNOR_INT8,
    GrB_BOR_INT16,      GrB_BAND_INT16,     GrB_BXOR_INT16,    GrB_BXNOR_INT16,
    GrB_BOR_INT32,      GrB_BAND_INT32,     GrB_BXOR_INT32,    GrB_BXNOR_INT32,
    GrB_BOR_INT64,      GrB_BAND_INT64,     GrB_BXOR_INT64,    GrB_BXNOR_INT64,
    GrB_BOR_UINT8,      GrB_BAND_UINT8,     GrB_BXOR_UINT8,    GrB_BXNOR_UINT8,
    GrB_BOR_UINT16,     GrB_BAND_UINT16,    GrB_BXOR_UINT16,   GrB_BXNOR_UINT16,
    GrB_BOR_UINT32,     GrB_BAND_UINT32,    GrB_BXOR_UINT32,   GrB_BXNOR_UINT32,
    GrB_BOR_UINT64,     GrB_BAND_UINT64,    GrB_BXOR_UINT64,   GrB_BXNOR_UINT64,

    // z = bitget(x,y)  z = bitset(x,y)     z = bitclr(x,y)
    GxB_BGET_INT8,      GxB_BSET_INT8,      GxB_BCLR_INT8,
    GxB_BGET_INT16,     GxB_BSET_INT16,     GxB_BCLR_INT16,
    GxB_BGET_INT32,     GxB_BSET_INT32,     GxB_BCLR_INT32,
    GxB_BGET_INT64,     GxB_BSET_INT64,     GxB_BCLR_INT64,
    GxB_BGET_UINT8,     GxB_BSET_UINT8,     GxB_BCLR_UINT8,
    GxB_BGET_UINT16,    GxB_BSET_UINT16,    GxB_BCLR_UINT16,
    GxB_BGET_UINT32,    GxB_BSET_UINT32,    GxB_BCLR_UINT32,
    GxB_BGET_UINT64,    GxB_BSET_UINT64,    GxB_BCLR_UINT64 ;

//------------------------------------------------------------------------------
// z=f(x,y) where z and x have the same type, but y is GrB_INT8
//------------------------------------------------------------------------------

    // z = bitshift (x,y) computes z = x left-shifted by y bits if y >= 0, or z
    // = x right-shifted by (-y) bits if y < 0.  z is equal to x if y is zero.
    // z and x have the same type, as given by the suffix on the operator name.
    // Since y must be signed, it cannot have the same type as x when x is
    // unsigned; it is always GrB_INT8 for all 8 versions of this operator.
    // The GxB_BSHIFT_* operators compute the arithmetic shift, and produce the
    // same results as the bitshift.m function, for all possible inputs.

GB_GLOBAL GrB_BinaryOp

    // z = bitshift(x,y)
    GxB_BSHIFT_INT8,
    GxB_BSHIFT_INT16,
    GxB_BSHIFT_INT32,
    GxB_BSHIFT_INT64,
    GxB_BSHIFT_UINT8,
    GxB_BSHIFT_UINT16,
    GxB_BSHIFT_UINT32,
    GxB_BSHIFT_UINT64 ;

//------------------------------------------------------------------------------
// z=f(x,y) where z is BOOL and the type of x,y is given by the suffix
//------------------------------------------------------------------------------

GB_GLOBAL GrB_BinaryOp

    // Six comparators z=f(x,y) return their result as boolean, but
    // where x and y have the same type.  The suffix in their names refers to
    // the type of x and y since z is always boolean.  If used as multiply
    // operators in a semiring, they can only be combined with boolean monoids.
    // The _BOOL versions of these operators give the same results as their
    // IS*_BOOL counterparts.  GrB_EQ_BOOL and GrB_LXNOR are identical.

    // z = (x == y)     z = (x != y)        z = (x > y)         z = (x < y)
    GrB_EQ_BOOL,        GrB_NE_BOOL,        GrB_GT_BOOL,        GrB_LT_BOOL,
    GrB_EQ_INT8,        GrB_NE_INT8,        GrB_GT_INT8,        GrB_LT_INT8,
    GrB_EQ_INT16,       GrB_NE_INT16,       GrB_GT_INT16,       GrB_LT_INT16,
    GrB_EQ_INT32,       GrB_NE_INT32,       GrB_GT_INT32,       GrB_LT_INT32,
    GrB_EQ_INT64,       GrB_NE_INT64,       GrB_GT_INT64,       GrB_LT_INT64,
    GrB_EQ_UINT8,       GrB_NE_UINT8,       GrB_GT_UINT8,       GrB_LT_UINT8,
    GrB_EQ_UINT16,      GrB_NE_UINT16,      GrB_GT_UINT16,      GrB_LT_UINT16,
    GrB_EQ_UINT32,      GrB_NE_UINT32,      GrB_GT_UINT32,      GrB_LT_UINT32,
    GrB_EQ_UINT64,      GrB_NE_UINT64,      GrB_GT_UINT64,      GrB_LT_UINT64,
    GrB_EQ_FP32,        GrB_NE_FP32,        GrB_GT_FP32,        GrB_LT_FP32,
    GrB_EQ_FP64,        GrB_NE_FP64,        GrB_GT_FP64,        GrB_LT_FP64,
    // complex:
    GxB_EQ_FC32,        GxB_NE_FC32,
    GxB_EQ_FC64,        GxB_NE_FC64,

    // z = (x >= y)     z = (x <= y)
    GrB_GE_BOOL,        GrB_LE_BOOL,
    GrB_GE_INT8,        GrB_LE_INT8,
    GrB_GE_INT16,       GrB_LE_INT16,
    GrB_GE_INT32,       GrB_LE_INT32,
    GrB_GE_INT64,       GrB_LE_INT64,
    GrB_GE_UINT8,       GrB_LE_UINT8,
    GrB_GE_UINT16,      GrB_LE_UINT16,
    GrB_GE_UINT32,      GrB_LE_UINT32,
    GrB_GE_UINT64,      GrB_LE_UINT64,
    GrB_GE_FP32,        GrB_LE_FP32,
    GrB_GE_FP64,        GrB_LE_FP64 ;

//------------------------------------------------------------------------------
// z=f(x,y) where z is complex and the type of x,y is given by the suffix
//------------------------------------------------------------------------------

GB_GLOBAL GrB_BinaryOp

    // z = cmplx (x,y)
    GxB_CMPLX_FP32,
    GxB_CMPLX_FP64 ;

//==============================================================================
// positional GrB_UnaryOp and GrB_BinaryOp operators
//==============================================================================

// Positional operators do not depend on the value of an entry, but its row or
// column index in the matrix instead.  For example, for an entry A(i,j),
// first_i(A(i,j),y) is equal to i.  These operators are useful for returning
// node id's as the result of a semiring operation.  If used as a mask, zero
// has a special value, and thus z=first_i1(A(i,j),j) returns i+1 instead of i.
// This can be useful when using a positional operator to construct a mask
// matrix or vector for another GraphBLAS operation.  It is also essential for
// the @GrB interface, since the user view of matrix indices in @GrB is
// 1-based, not 0-based.

// When applied to a vector, j is always equal to 0.  For a GxB_SCALAR,
// both i and j are always zero.

// GraphBLAS defines a GrB_Index as uint64_t, but these operators return a
// GrB_INT32 or GrB_INT64 type, which is more flexible to use because the
// result of this operator can be negated, to flag an entry for example.  The
// value -1 can be used to denote "no node" or "no position".  GrB_INT32 is
// useful for graphs smaller than 2^31 nodes.  If the row or column index
// exceeds INT32_MAX, the result is determined by the typecast from the
// 64-bit index to the smaller 32-bit index.

// Positional operators cannot be used to construct monoids.  They can be used
// as multiplicative operators in semirings, and as operators for GrB_eWise*,
// and GrB_apply (bind first or second).  For the latter, the operator cannot
// depend on the bound scalar.

// When used as multiplicative operators in a semiring, FIRSTJ and SECONDI
// are identical.  If C(i,j) += t is computed where t = A(i,k)*B(k,j), then
// t = k in both cases.  Likewise, FIRSTJ1 and SECONDI1 are identical.

GB_GLOBAL GrB_BinaryOp

    GxB_FIRSTI_INT32,   GxB_FIRSTI_INT64,    // z = first_i(A(i,j),y) == i
    GxB_FIRSTI1_INT32,  GxB_FIRSTI1_INT64,   // z = first_i1(A(i,j),y) == i+1
    GxB_FIRSTJ_INT32,   GxB_FIRSTJ_INT64,    // z = first_j(A(i,j),y) == j
    GxB_FIRSTJ1_INT32,  GxB_FIRSTJ1_INT64,   // z = first_j1(A(i,j),y) == j+1
    GxB_SECONDI_INT32,  GxB_SECONDI_INT64,   // z = second_i(x,B(i,j)) == i
    GxB_SECONDI1_INT32, GxB_SECONDI1_INT64,  // z = second_i1(x,B(i,j)) == i+1
    GxB_SECONDJ_INT32,  GxB_SECONDJ_INT64,   // z = second_j(x,B(i,j)) == j
    GxB_SECONDJ1_INT32, GxB_SECONDJ1_INT64 ; // z = second_j1(x,B(i,j)) == j+1

GB_GLOBAL GrB_UnaryOp

    GxB_POSITIONI_INT32,  GxB_POSITIONI_INT64,  // z=position_i(A(i,j)) == i
    GxB_POSITIONI1_INT32, GxB_POSITIONI1_INT64, // z=position_i1(A(i,j)) == i+1
    GxB_POSITIONJ_INT32,  GxB_POSITIONJ_INT64,  // z=position_j(A(i,j)) == j
    GxB_POSITIONJ1_INT32, GxB_POSITIONJ1_INT64 ;// z=position_j1(A(i,j)) == j+1

//==============================================================================
// special GrB_BinaryOp for build methods only
//==============================================================================

// In GrB*build* methods, passing dup as NULL means that no duplicates are
// tolerated.  If duplicates appear, an error is returned.  If dup is a binary
// operator, it is applied to reduce duplicates to a single value.  The
// GxB_IGNORE_DUP is a special case.  It is not an operator, but an indication
// that any duplicates are to be ignored.

GB_GLOBAL GrB_BinaryOp GxB_IGNORE_DUP ;

//==============================================================================
// About boolean and bitwise binary operators
//==============================================================================

// Some of the boolean operators compute the same thing with different names.
// For example, x*y and x&&y give the same results for boolean x and y.
// Operations such as x < y when x and y are boolean are treated as if true=1
// and false=0.  Below is the truth table for all binary operators with boolean
// inputs.  This table is defined by how C typecasts boolean values for
// non-boolean operations.  For example, if x, y, and z are boolean, x = true,
// and y = true, then z = x + y = true + true = true.  DIV (x/y) is defined
// below.  RDIV (y/x) is shown as \ in the table; it is the same as 2nd.

//  x y  1st 2nd min max +  -  *  /  or and xor eq ne > < ge le \ pow pair
//  0 0  0   0   0   0   0  0  0  0  0  0   0   1  0  0 0 1  1  0 1   1
//  0 1  0   1   0   1   1  1  0  0  1  0   1   0  1  0 1 0  1  1 0   1
//  1 0  1   0   0   1   1  1  0  1  1  0   1   0  1  1 0 1  0  0 1   1
//  1 1  1   1   1   1   1  0  1  1  1  1   0   1  0  0 0 1  1  1 1   1

// GraphBLAS includes a GrB_DIV_BOOL operator in its specification, but does
// not define what boolean "division" means.  SuiteSparse:GraphBLAS makes the
// following interpretation.

// GraphBLAS does not generate exceptions for divide-by-zero.  Floating-point
// divide-by-zero follows the IEEE 754 standard: 1/0 is +Inf, -1/0 is -Inf, and
// 0/0 is NaN.  For integer division by zero, if x is positive, x/0 is the
// largest integer, -x/0 is the integer minimum (zero for unsigned integers),
// and 0/0 is zero.  For example, for int8, 1/0 is 127, and -1/0 is -128.  For
// uint8, 1/0 is 255 and 0/0 is zero.

// Boolean division is treated as if it were an unsigned integer type with
// true=1 and false=0, and with the max and min value being 1 and 0.  As a
// result, GrB_IDENTITY_BOOL, GrB_AINV_BOOL, and GrB_MINV_BOOL all give the
// same result (z = x).

// With this convention for boolean "division", there are 11 unique binary
// operators that are purely boolean.  Other named *_BOOL operators are
// redundant but are included in GraphBLAS so that the name space of operators
// is complete.  Below is a list of all operators and their equivalents.

//                   x: 0 0 1 1
//                   y: 0 1 0 1
//                   z: see below
//
//      z = 0           0 0 0 0     (zero function, not predefined)
//      z = (x && y)    0 0 0 1     AND, MIN, TIMES
//      z = (x > y)     0 0 1 0     GT, ISGT, and set diff (x\y)
//      z = x           0 0 1 1     FIRST, DIV
//
//      z = (x < y)     0 1 0 0     LT, ISLT, and set diff (y\x)
//      z = y           0 1 0 1     SECOND, RDIV
//      z = (x != y)    0 1 1 0     XOR, MINUS, RMINUS, NE, ISNE
//      z = (x || y)    0 1 1 1     OR, MAX, PLUS
//
//      z = ~(x || y)   1 0 0 0     (nor(x,y) function, not predefined)
//      z = (x == y)    1 0 0 1     LXNOR, EQ, ISEQ
//      z = ~y          1 0 1 0     (not(y), not predefined)
//      z = (x >= y)    1 0 1 1     GE, ISGE, POW, and "x implies y"
//
//      z = ~x          1 1 0 0     (not(x), not predefined)
//      z = (x <= y)    1 1 0 1     LE, ISLE, and "y implies x"
//      z = ~(x && y)   1 1 1 0     (nand(x,y) function, not predefined)
//      z = 1           1 1 1 1     PAIR, ONEB
//
//      z = any(x,y)    0 . . 1     ANY (pick x or y arbitrarily)

// Four more that have no _BOOL suffix are also redundant with the operators
// of the form GxB_*_BOOL (GrB_LOR, GrB_LAND, GrB_LXOR, and GrB_LXNOR).

// Note that the boolean binary operator space is not complete.  Five other
// boolean functions could be pre-defined as well:  z = 0, nor(x,y),
// nand(x,y), not(x), and not(y).

// Four of the possible 16 bitwise operators are pre-defined: BOR, BAND,
// BXOR, and BXNOR.  This assumes that the computations for each bit are
// entirely independent (so BSHIFT would not fit in the table above).

typedef void (*GxB_binary_function) (void *, const void *, const void *) ;

//==============================================================================
// GxB_SelectOp: select operators (DEPRECATED: do not use)
//==============================================================================

// historical; use GrB_select and GrB_IndexUnaryOp instead:
typedef struct GB_SelectOp_opaque *GxB_SelectOp ;
GB_GLOBAL GxB_SelectOp GxB_TRIL, GxB_TRIU, GxB_DIAG, GxB_OFFDIAG, GxB_NONZERO,
    GxB_EQ_ZERO, GxB_GT_ZERO, GxB_GE_ZERO, GxB_LT_ZERO, GxB_LE_ZERO,
    GxB_NE_THUNK, GxB_EQ_THUNK, GxB_GT_THUNK, GxB_GE_THUNK, GxB_LT_THUNK,
    GxB_LE_THUNK ;

//==============================================================================
// GrB_IndexUnaryOp: a unary operator that depends on the row/col indices
//==============================================================================

// The indexop has the form z = f(aij, i, j, y) where aij is the numerical
// value of the A(i,j) entry, i and j are its row and column index, and y
// is a scalar.  For vectors, it has the form z = f(vi, i, 0, y).

typedef struct GB_IndexUnaryOp_opaque *GrB_IndexUnaryOp ;

typedef void (*GxB_index_unary_function)
(
    void *z,            // output value z, of type ztype
    const void *x,      // input value x of type xtype; value of v(i) or A(i,j)
    GrB_Index i,        // row index of A(i,j)
    GrB_Index j,        // column index of A(i,j), or zero for v(i)
    const void *y       // input scalar y
) ;

//==============================================================================
// GrB_IndexUnaryOp: a unary operator that depends on the row/col indices
//==============================================================================

// To facilitate computations with negative integers, the indices i and j are
// of type int64_t.  The scalar y has the type corresponding to the suffix
// of the name of the operator.

GB_GLOBAL GrB_IndexUnaryOp

    //--------------------------------------------------------------------------
    // Result has the integer type INT32 or INT64, the same as the suffix
    //--------------------------------------------------------------------------

    // These operators work on any data type, including user-defined.

    GrB_ROWINDEX_INT32,  GrB_ROWINDEX_INT64,    // (i+y): row index plus y
    GrB_COLINDEX_INT32,  GrB_COLINDEX_INT64,    // (j+y): col index plus y
    GrB_DIAGINDEX_INT32, GrB_DIAGINDEX_INT64,   // (j-(i+y)): diag index plus y

    //--------------------------------------------------------------------------
    // Result is bool, depending only on the indices i,j, and y
    //--------------------------------------------------------------------------

    // These operators work on any data type, including user-defined.
    // The scalar y is int64.

    GrB_TRIL,       // (j <= (i+y)): lower triangular part
    GrB_TRIU,       // (j >= (i+y)): upper triangular part
    GrB_DIAG,       // (j == (i+y)): diagonal
    GrB_OFFDIAG,    // (j != (i+y)): offdiagonal

    GrB_COLLE,      // (j <= y): columns 0:y
    GrB_COLGT,      // (j > y): columns y+1:ncols-1
    GrB_ROWLE,      // (i <= y): rows 0:y
    GrB_ROWGT,      // (i > y): rows y+1:nrows-1

    //--------------------------------------------------------------------------
    // Result is bool, depending only on the value aij
    //--------------------------------------------------------------------------

    // These operators work on matrices and vectors of any built-in type,
    // including complex types.  aij and the scalar y have the same type as the
    // operator suffix.

    // VALUEEQ: (aij == y)
    GrB_VALUEEQ_INT8,  GrB_VALUEEQ_UINT8,  GrB_VALUEEQ_FP32, GrB_VALUEEQ_BOOL,
    GrB_VALUEEQ_INT16, GrB_VALUEEQ_UINT16, GrB_VALUEEQ_FP64,
    GrB_VALUEEQ_INT32, GrB_VALUEEQ_UINT32, GxB_VALUEEQ_FC32,
    GrB_VALUEEQ_INT64, GrB_VALUEEQ_UINT64, GxB_VALUEEQ_FC64,

    // VALUENE: (aij != y)
    GrB_VALUENE_INT8,  GrB_VALUENE_UINT8,  GrB_VALUENE_FP32, GrB_VALUENE_BOOL,
    GrB_VALUENE_INT16, GrB_VALUENE_UINT16, GrB_VALUENE_FP64,
    GrB_VALUENE_INT32, GrB_VALUENE_UINT32, GxB_VALUENE_FC32,
    GrB_VALUENE_INT64, GrB_VALUENE_UINT64, GxB_VALUENE_FC64,

    // These operators work on matrices and vectors of any real (non-complex)
    // built-in type.

    // VALUELT: (aij < y)
    GrB_VALUELT_INT8,  GrB_VALUELT_UINT8,  GrB_VALUELT_FP32, GrB_VALUELT_BOOL,
    GrB_VALUELT_INT16, GrB_VALUELT_UINT16, GrB_VALUELT_FP64,
    GrB_VALUELT_INT32, GrB_VALUELT_UINT32,
    GrB_VALUELT_INT64, GrB_VALUELT_UINT64,

    // VALUELE: (aij <= y)
    GrB_VALUELE_INT8,  GrB_VALUELE_UINT8,  GrB_VALUELE_FP32, GrB_VALUELE_BOOL,
    GrB_VALUELE_INT16, GrB_VALUELE_UINT16, GrB_VALUELE_FP64,
    GrB_VALUELE_INT32, GrB_VALUELE_UINT32,
    GrB_VALUELE_INT64, GrB_VALUELE_UINT64,

    // VALUEGT: (aij > y)
    GrB_VALUEGT_INT8,  GrB_VALUEGT_UINT8,  GrB_VALUEGT_FP32, GrB_VALUEGT_BOOL,
    GrB_VALUEGT_INT16, GrB_VALUEGT_UINT16, GrB_VALUEGT_FP64,
    GrB_VALUEGT_INT32, GrB_VALUEGT_UINT32,
    GrB_VALUEGT_INT64, GrB_VALUEGT_UINT64,

    // VALUEGE: (aij >= y)
    GrB_VALUEGE_INT8,  GrB_VALUEGE_UINT8,  GrB_VALUEGE_FP32, GrB_VALUEGE_BOOL,
    GrB_VALUEGE_INT16, GrB_VALUEGE_UINT16, GrB_VALUEGE_FP64,
    GrB_VALUEGE_INT32, GrB_VALUEGE_UINT32,
    GrB_VALUEGE_INT64, GrB_VALUEGE_UINT64 ;

//==============================================================================
// GrB_Monoid
//==============================================================================

// A monoid is an associative operator z=op(x,y) where all three types of z, x,
// and y are identical.  The monoid also has an identity element, such that
// op(x,identity) = op(identity,x) = x.

typedef struct GB_Monoid_opaque *GrB_Monoid ;

//==============================================================================
// GrB_Semiring
//==============================================================================

typedef struct GB_Semiring_opaque *GrB_Semiring ;

//==============================================================================
// GrB_Scalar: a GraphBLAS scalar
//==============================================================================

typedef struct GB_Scalar_opaque *GxB_Scalar ;   // historical: use GrB_Scalar
typedef struct GB_Scalar_opaque *GrB_Scalar ;   // use this instead

//==============================================================================
// GrB_Vector: a GraphBLAS vector
//==============================================================================

typedef struct GB_Vector_opaque *GrB_Vector ;

//==============================================================================
// GrB_Matrix: a GraphBLAS matrix
//==============================================================================

typedef struct GB_Matrix_opaque *GrB_Matrix ;

//==============================================================================
// SuiteSparse:GraphBLAS options
//==============================================================================

// The following options modify how SuiteSparse:GraphBLAS stores and operates
// on its matrices.  The GrB_get/set methods allow the user to suggest how the
// internal representation of a matrix, or all matrices, should be held.  These
// options have no effect on the result (except for minor roundoff differences
// for floating-point types). They only affect the time and memory usage of the
// computations.

typedef enum            // for global options or matrix options
{

    //------------------------------------------------------------
    // GrB_get / GrB_set for GrB_Matrix and GrB_GLOBAL:
    //------------------------------------------------------------

    GxB_HYPER_SWITCH = 7000,    // switch to hypersparse (double value)
    GxB_HYPER_HASH = 7048,      // hyper_hash control (int64 value)
    GxB_BITMAP_SWITCH = 7001,   // switch to bitmap (double value)
    GxB_FORMAT = 7002,          // historical; use GrB_STORAGE_ORIENTATION_HINT

    //------------------------------------------------------------
    // GrB_get for GrB_GLOBAL:
    //------------------------------------------------------------

    GxB_MODE = 7003,                 // historical; use GrB_BLOCKING_MODE
    GxB_LIBRARY_NAME = 7004,         // historical; use GrB_NAME
    GxB_LIBRARY_VERSION = 7005,      // historical; use GrB_LIBRARY_VER_*
    GxB_LIBRARY_DATE = 7006,         // date of the library (char *)
    GxB_LIBRARY_ABOUT = 7007,        // about the library (char *)
    GxB_LIBRARY_URL = 7008,          // URL for the library (char *)
    GxB_LIBRARY_LICENSE = 7009,      // license of the library (char *)
    GxB_LIBRARY_COMPILE_DATE = 7010, // date library was compiled (char *)
    GxB_LIBRARY_COMPILE_TIME = 7011, // time library was compiled (char *)
    GxB_API_VERSION = 7012,          // historical; use GrB_API_VER_*
    GxB_API_DATE = 7013,             // date of the API (char *)
    GxB_API_ABOUT = 7014,            // about the API (char *)
    GxB_API_URL = 7015,              // URL for the API (char *)
    GxB_COMPILER_VERSION = 7016,     // compiler version (3 int's)
    GxB_COMPILER_NAME = 7017,        // compiler name (char *)
    GxB_LIBRARY_OPENMP = 7018,       // library compiled with OpenMP
    GxB_MALLOC_FUNCTION = 7037,      // malloc function pointer
    GxB_CALLOC_FUNCTION = 7038,      // calloc function pointer
    GxB_REALLOC_FUNCTION = 7039,     // realloc function pointer
    GxB_FREE_FUNCTION = 7040,        // free function pointer

    //------------------------------------------------------------
    // GrB_get / GrB_set for GrB_GLOBAL:
    //------------------------------------------------------------

    GxB_GLOBAL_NTHREADS = GxB_NTHREADS,  // max number of threads to use
    GxB_GLOBAL_CHUNK = GxB_CHUNK,        // chunk size for small problems.
    GxB_GLOBAL_GPU_ID = GxB_GPU_ID,      // which GPU to use (DRAFT)

    GxB_BURBLE = 7019,               // diagnostic output
    GxB_PRINTF = 7020,               // printf function diagnostic output
    GxB_FLUSH = 7021,                // flush function diagnostic output
    GxB_MEMORY_POOL = 7022,          // no longer used
    GxB_PRINT_1BASED = 7023,         // print matrices as 0-based or 1-based

    GxB_JIT_C_COMPILER_NAME = 7024,  // CPU JIT C compiler name
    GxB_JIT_C_COMPILER_FLAGS = 7025, // CPU JIT C compiler flags
    GxB_JIT_C_LINKER_FLAGS = 7026,   // CPU JIT C linker flags
    GxB_JIT_C_LIBRARIES = 7027,      // CPU JIT C libraries
    GxB_JIT_C_PREFACE = 7028,        // CPU JIT C preface
    GxB_JIT_C_CONTROL = 7029,        // CPU JIT C control
    GxB_JIT_CACHE_PATH = 7030,       // CPU/CUDA JIT path for compiled kernels
    GxB_JIT_C_CMAKE_LIBS = 7031,     // CPU JIT C libraries when using cmake
    GxB_JIT_USE_CMAKE = 7032,        // CPU JIT: use cmake or direct compile
    GxB_JIT_ERROR_LOG = 7033,        // CPU JIT: error log file

    GxB_JIT_CUDA_PREFACE = 7100,     // CUDA JIT C++ preface

    //------------------------------------------------------------
    // GrB_get for GrB_Matrix:
    //------------------------------------------------------------

    GxB_SPARSITY_STATUS = 7034,     // hyper, sparse, bitmap or full (1,2,4,8)
    GxB_IS_HYPER = 7035,            // historical; use GxB_SPARSITY_STATUS

    //------------------------------------------------------------
    // GrB_get/GrB_set for GrB_Matrix:
    //------------------------------------------------------------

    GxB_SPARSITY_CONTROL = 7036,    // sparsity control: 0 to 15; see below

} GxB_Option_Field ;

// for GxB_JIT_C_CONTROL:
typedef enum
{
    GxB_JIT_OFF = 0,    // do not use the JIT: free all JIT kernels if loaded
    GxB_JIT_PAUSE = 1,  // do not run JIT kernels but keep any loaded
    GxB_JIT_RUN = 2,    // run JIT kernels if already loaded; no load/compile
    GxB_JIT_LOAD = 3,   // able to load and run JIT kernels; may not compile
    GxB_JIT_ON = 4,     // full JIT: able to compile, load, and run
}
GxB_JIT_Control ;

// GxB_FORMAT is historical, but it can be by row or by column:
typedef enum
{
    GxB_BY_ROW = 0,     // CSR: compressed sparse row format
    GxB_BY_COL = 1,     // CSC: compressed sparse column format
    GxB_NO_FORMAT = -1  // format not defined
}
GxB_Format_Value ;

// The default format is by row.  These constants are defined as GB_GLOBAL
// const, so that if SuiteSparse:GraphBLAS is recompiled with a different
// default format, and the application is relinked but not recompiled, it will
// acquire the new default values.
GB_GLOBAL const GxB_Format_Value GxB_FORMAT_DEFAULT ;

// the default hyper_switch parameter
GB_GLOBAL const double GxB_HYPER_DEFAULT ;

// GxB_SPARSITY_CONTROL can be any sum or bitwise OR of these 4 values:
#define GxB_HYPERSPARSE 1   // store matrix in hypersparse form
#define GxB_SPARSE      2   // store matrix as sparse form (compressed vector)
#define GxB_BITMAP      4   // store matrix as a bitmap
#define GxB_FULL        8   // store matrix as full; all entries must be present

// size of b array for GxB_set/get (GxB_BITMAP_SWITCH, b)
#define GxB_NBITMAP_SWITCH 8    // size of bitmap_switch parameter array

// any sparsity value:
#define GxB_ANY_SPARSITY (GxB_HYPERSPARSE + GxB_SPARSE + GxB_BITMAP + GxB_FULL)

// the default sparsity control is any format:
#define GxB_AUTO_SPARSITY GxB_ANY_SPARSITY

// GrB_set (A, scontrol, GxB_SPARSITY_CONTROL) provides hints
// about which data structure GraphBLAS should use for the matrix A:
//
//      GxB_AUTO_SPARSITY: GraphBLAS selects automatically.
//      GxB_HYPERSPARSE: always hypersparse, taking O(nvals(A)) space.
//      GxB_SPARSE: always in a sparse struture: compressed-sparse row/column,
//          taking O(nrows+nvals(A)) space if stored by row, or
//          O(ncols+nvals(A)) if stored by column.
//      GxB_BITMAP: always in a bitmap struture, taking O(nrows*ncols) space.
//      GxB_FULL: always in a full structure, taking O(nrows*ncols) space,
//          unless not all entries are present, in which case the bitmap
//          storage is used.
//
// These options can be summed.  For example, to allow a matrix to be sparse
// or hypersparse, but not bitmap or full, use GxB_SPARSE + GxB_HYPERSPARSE.
// Since GxB_FULL can only be used when all entries are present, matrices with
// the just GxB_FULL control setting are stored in bitmap form if any entries
// are not present.
//
// Only the least 4 bits of the sparsity control are considered, so the
// formats can be bitwise negated.  For example, to allow for any format
// except full, use ~GxB_FULL.
//
// GrB_get (A, &sparsity, GxB_SPARSITY_STATUS) returns the
// current data structure currently used for the matrix A (either hypersparse,
// sparse, bitmap, or full).
//
// GrB_get (A, &scontrol, GxB_SPARSITY_CONTROL) returns the hint
// for how A should be stored (hypersparse, sparse, bitmap, or full, or any
// combination).

// GxB_HYPER_SWITCH:
//      If the matrix or vector structure can be sparse or hypersparse, the
//      GxB_HYPER_SWITCH parameter controls when each of these structures are
//      used.  The parameter is not used if the matrix or vector is full or
//      bitmap.
//
//      Let k be the actual number of non-empty vectors (with at least one
//      entry).  This value k is not dependent on whether or not the matrix is
//      stored in hypersparse structure.  Let n be the number of vectors (the #
//      of columns if CSC, or rows if CSR).  Let h be the value of the
//      GxB_HYPER_SWITCH setting of the matrix.
//
//      If a matrix is currently hypersparse, it can be converted to
//      non-hypersparse if (n <= 1  || k > 2*n*h).  Otherwise it stays
//      hypersparse.  If (n <= 1) the matrix is always stored as
//      non-hypersparse.
//
//      If currently non-hypersparse, it can be converted to hypersparse if (n
//      > 1 && k <= n*h).  Otherwise, it stays non-hypersparse.  If (n <= 1)
//      the matrix always remains non-hypersparse.
//
//      Setting GxB_HYPER_SWITCH to GxB_ALWAYS_HYPER or GxB_NEVER_HYPER ensures
//      a matrix always stays hypersparse, or always stays non-hypersparse,
//      respectively.

GB_GLOBAL const double GxB_ALWAYS_HYPER, GxB_NEVER_HYPER ;

//==============================================================================
// GxB_Context: for managing computational resources
//==============================================================================

typedef struct GB_Context_opaque *GxB_Context ;

// GxB_CONTEXT_WORLD is the default Context for all user threads.
GB_GLOBAL GxB_Context GxB_CONTEXT_WORLD ;

typedef enum
{
    GxB_CONTEXT_NTHREADS = GxB_NTHREADS,     // max number of threads to use.
                    // If <= 0, then one thread is used.

    GxB_CONTEXT_CHUNK = GxB_CHUNK,   // chunk size for small problems.
                    // If < 1, then the default is used.

    // GPU control (DRAFT: in progress, do not use)
    GxB_CONTEXT_GPU_ID      = GxB_GPU_ID,
}
GxB_Context_Field ;

//==============================================================================
// GrB_set and GrB_get
//==============================================================================

typedef struct GB_Global_opaque *GrB_Global ;
GB_GLOBAL const GrB_Global GrB_GLOBAL ;

typedef enum
{

    //--------------------------------------------------------------------------
    // GrB enums in the C API
    //--------------------------------------------------------------------------

    // GrB_Descriptor only:
    GrB_OUTP_FIELD = 0,     // descriptor for output of a method
    GrB_MASK_FIELD = 1,     // descriptor for the mask input of a method
    GrB_INP0_FIELD = 2,     // descriptor for the first input of a method
    GrB_INP1_FIELD = 3,     // descriptor for the second input of a method

    // all objects, including GrB_GLOBAL:
    GrB_NAME = 10,          // name of the object, as a string

    // GrB_GLOBAL only:
    GrB_LIBRARY_VER_MAJOR = 11,     // SuiteSparse:GraphBLAS version
    GrB_LIBRARY_VER_MINOR = 12,
    GrB_LIBRARY_VER_PATCH = 13,
    GrB_API_VER_MAJOR = 14,         // C API version
    GrB_API_VER_MINOR = 15,
    GrB_API_VER_PATCH = 16,
    GrB_BLOCKING_MODE = 17,         // GrB_Mode

    // GrB_GLOBAL, GrB_Matrix, GrB_Vector, GrB_Scalar:
    GrB_STORAGE_ORIENTATION_HINT = 100, // GrB_Orientation

    // GrB_Matrix, GrB_Vector, GrB_Scalar (and void * serialize):
    GrB_EL_TYPE_CODE = 102,         // a GrB_Type_code (see below)
    GrB_EL_TYPE_STRING = 106,       // name of the type

    // GrB_*Op, GrB_Monoid, and GrB_Semiring:
    GrB_INP0_TYPE_CODE = 103,       // GrB_Type_code
    GrB_INP1_TYPE_CODE = 104,
    GrB_OUTP_TYPE_CODE = 105,
    GrB_INP0_TYPE_STRING = 107,     // name of the type, as a string
    GrB_INP1_TYPE_STRING = 108,
    GrB_OUTP_TYPE_STRING = 109,

    // GrB_Type (readable only):
    GrB_SIZE = 110,                 // size of the type

    //--------------------------------------------------------------------------
    // SuiteSparse extensions:
    //--------------------------------------------------------------------------

    // GrB_Type, GrB_UnaryOp, GrB_BinaryOp, and GrB_IndexUnaryOp:
    GxB_JIT_C_NAME = 7041,          // C type or function name
    GxB_JIT_C_DEFINITION = 7042,    // C typedef or function definition

    // GrB_Monoid and GrB_Semiring:
    GxB_MONOID_IDENTITY = 7043,     // monoid identity value
    GxB_MONOID_TERMINAL = 7044,     // monoid terminal value
    GxB_MONOID_OPERATOR = 7045,     // monoid binary operator

    // GrB_Semiring only:
    GxB_SEMIRING_MONOID = 7046,     // semiring monoid
    GxB_SEMIRING_MULTIPLY = 7047,   // semiring multiplicative op
}
GrB_Field ;

typedef enum
{
    GrB_ROWMAJOR = 0,
    GrB_COLMAJOR = 1,
    GrB_BOTH     = 2,
    GrB_UNKNOWN  = 3,
}
GrB_Orientation ;

typedef enum
{
    GrB_UDT_CODE    = 0,        // user-defined type
    GrB_BOOL_CODE   = 1,        // GraphBLAS: GrB_BOOL      C: bool
    GrB_INT8_CODE   = 2,        // GraphBLAS: GrB_INT8      C: int8_t
    GrB_UINT8_CODE  = 3,        // GraphBLAS: GrB_UINT8     C: uint8_t
    GrB_INT16_CODE  = 4,        // GraphBLAS: GrB_INT16     C: int16_t
    GrB_UINT16_CODE = 5,        // GraphBLAS: GrB_UINT16    C: uint16_t
    GrB_INT32_CODE  = 6,        // GraphBLAS: GrB_INT32     C: int32_t
    GrB_UINT32_CODE = 7,        // GraphBLAS: GrB_UINT32    C: uint32_t
    GrB_INT64_CODE  = 8,        // GraphBLAS: GrB_INT64     C: int64_t
    GrB_UINT64_CODE = 9,        // GraphBLAS: GrB_UINT64    C: uint64_t
    GrB_FP32_CODE   = 10,       // GraphBLAS: GrB_FP32      C: float
    GrB_FP64_CODE   = 11,       // GraphBLAS: GrB_FP64      C: double
    GxB_FC32_CODE   = 7070,     // GraphBLAS: GxB_FC32      C: float complex
    GxB_FC64_CODE   = 7071,     // GraphBLAS: GxB_FC64      C: double complex
}
GrB_Type_Code ;

//==============================================================================
// GrB_wait: finish computations
//==============================================================================

typedef enum
{
    GrB_COMPLETE = 0,       // establishes a happens-before relation
    GrB_MATERIALIZE = 1     // object is complete
}
GrB_WaitMode ;

//==============================================================================
// GrB_extract: extract a submatrix or subvector
//==============================================================================

GB_GLOBAL const uint64_t *GrB_ALL ;

// These special values of ni and nj can be used for GrB_assign,
// GrB_extract, and GxB_subassign.
#define GxB_RANGE       (INT64_MAX)
#define GxB_STRIDE      (INT64_MAX-1)
#define GxB_BACKWARDS   (INT64_MAX-2)

// for the strided range begin:inc:end, I [GxB_BEGIN] is the value of begin, I
// [GxB_END] is the value end, I [GxB_INC] is the magnitude of the stride.  If
// the stride is negative, use ni = GxB_BACKWARDS.
#define GxB_BEGIN (0)
#define GxB_END   (1)
#define GxB_INC   (2)

//==============================================================================
// GrB_Monoid: built-in monoids
//==============================================================================

GB_GLOBAL GrB_Monoid

    //--------------------------------------------------------------------------
    // 10 MIN monoids: (not for complex types)
    //--------------------------------------------------------------------------

    // GxB_MIN monoids, historical, use GrB_MIN_MONOID_* instead:
    GxB_MIN_INT8_MONOID,        // identity: INT8_MAX     terminal: INT8_MIN
    GxB_MIN_INT16_MONOID,       // identity: INT16_MAX    terminal: INT16_MIN
    GxB_MIN_INT32_MONOID,       // identity: INT32_MAX    terminal: INT32_MIN
    GxB_MIN_INT64_MONOID,       // identity: INT64_MAX    terminal: INT32_MIN
    GxB_MIN_UINT8_MONOID,       // identity: UINT8_MAX    terminal: 0
    GxB_MIN_UINT16_MONOID,      // identity: UINT16_MAX   terminal: 0
    GxB_MIN_UINT32_MONOID,      // identity: UINT32_MAX   terminal: 0
    GxB_MIN_UINT64_MONOID,      // identity: UINT64_MAX   terminal: 0
    GxB_MIN_FP32_MONOID,        // identity: INFINITY     terminal: -INFINITY
    GxB_MIN_FP64_MONOID,        // identity: INFINITY     terminal: -INFINITY

    // preferred names from the v1.3 spec:
    GrB_MIN_MONOID_INT8,        // identity: INT8_MAX     terminal: INT8_MIN
    GrB_MIN_MONOID_INT16,       // identity: INT16_MAX    terminal: INT16_MIN
    GrB_MIN_MONOID_INT32,       // identity: INT32_MAX    terminal: INT32_MIN
    GrB_MIN_MONOID_INT64,       // identity: INT64_MAX    terminal: INT32_MIN
    GrB_MIN_MONOID_UINT8,       // identity: UINT8_MAX    terminal: 0
    GrB_MIN_MONOID_UINT16,      // identity: UINT16_MAX   terminal: 0
    GrB_MIN_MONOID_UINT32,      // identity: UINT32_MAX   terminal: 0
    GrB_MIN_MONOID_UINT64,      // identity: UINT64_MAX   terminal: 0
    GrB_MIN_MONOID_FP32,        // identity: INFINITY     terminal: -INFINITY
    GrB_MIN_MONOID_FP64,        // identity: INFINITY     terminal: -INFINITY

    //--------------------------------------------------------------------------
    // 10 MAX monoids:
    //--------------------------------------------------------------------------

    // GxB_MAX monoids, historical, use GrB_MAX_MONOID_* instead:
    GxB_MAX_INT8_MONOID,        // identity: INT8_MIN     terminal: INT8_MAX
    GxB_MAX_INT16_MONOID,       // identity: INT16_MIN    terminal: INT16_MAX
    GxB_MAX_INT32_MONOID,       // identity: INT32_MIN    terminal: INT32_MAX
    GxB_MAX_INT64_MONOID,       // identity: INT64_MIN    terminal: INT64_MAX
    GxB_MAX_UINT8_MONOID,       // identity: 0            terminal: UINT8_MAX
    GxB_MAX_UINT16_MONOID,      // identity: 0            terminal: UINT16_MAX
    GxB_MAX_UINT32_MONOID,      // identity: 0            terminal: UINT32_MAX
    GxB_MAX_UINT64_MONOID,      // identity: 0            terminal: UINT64_MAX
    GxB_MAX_FP32_MONOID,        // identity: -INFINITY    terminal: INFINITY
    GxB_MAX_FP64_MONOID,        // identity: -INFINITY    terminal: INFINITY

    // preferred names from the v1.3 spec:
    GrB_MAX_MONOID_INT8,        // identity: INT8_MIN     terminal: INT8_MAX
    GrB_MAX_MONOID_INT16,       // identity: INT16_MIN    terminal: INT16_MAX
    GrB_MAX_MONOID_INT32,       // identity: INT32_MIN    terminal: INT32_MAX
    GrB_MAX_MONOID_INT64,       // identity: INT64_MIN    terminal: INT64_MAX
    GrB_MAX_MONOID_UINT8,       // identity: 0            terminal: UINT8_MAX
    GrB_MAX_MONOID_UINT16,      // identity: 0            terminal: UINT16_MAX
    GrB_MAX_MONOID_UINT32,      // identity: 0            terminal: UINT32_MAX
    GrB_MAX_MONOID_UINT64,      // identity: 0            terminal: UINT64_MAX
    GrB_MAX_MONOID_FP32,        // identity: -INFINITY    terminal: INFINITY
    GrB_MAX_MONOID_FP64,        // identity: -INFINITY    terminal: INFINITY

    //--------------------------------------------------------------------------
    // 12 PLUS monoids:
    //--------------------------------------------------------------------------

    // GxB_PLUS monoids, historical, use GrB_PLUS_MONOID_* instead:
    GxB_PLUS_INT8_MONOID,       // identity: 0
    GxB_PLUS_INT16_MONOID,      // identity: 0
    GxB_PLUS_INT32_MONOID,      // identity: 0
    GxB_PLUS_INT64_MONOID,      // identity: 0
    GxB_PLUS_UINT8_MONOID,      // identity: 0
    GxB_PLUS_UINT16_MONOID,     // identity: 0
    GxB_PLUS_UINT32_MONOID,     // identity: 0
    GxB_PLUS_UINT64_MONOID,     // identity: 0
    GxB_PLUS_FP32_MONOID,       // identity: 0
    GxB_PLUS_FP64_MONOID,       // identity: 0

    // preferred names from the v1.3 spec:
    GrB_PLUS_MONOID_INT8,       // identity: 0
    GrB_PLUS_MONOID_INT16,      // identity: 0
    GrB_PLUS_MONOID_INT32,      // identity: 0
    GrB_PLUS_MONOID_INT64,      // identity: 0
    GrB_PLUS_MONOID_UINT8,      // identity: 0
    GrB_PLUS_MONOID_UINT16,     // identity: 0
    GrB_PLUS_MONOID_UINT32,     // identity: 0
    GrB_PLUS_MONOID_UINT64,     // identity: 0
    GrB_PLUS_MONOID_FP32,       // identity: 0
    GrB_PLUS_MONOID_FP64,       // identity: 0

    // complex monoids:
    GxB_PLUS_FC32_MONOID,       // identity: 0
    GxB_PLUS_FC64_MONOID,       // identity: 0

    //--------------------------------------------------------------------------
    // 12 TIMES monoids: identity value is 1, int* and uint* are terminal
    //--------------------------------------------------------------------------

    // GxB_TIMES monoids, historical, use GrB_TIMES_MONOID_* instead:
    GxB_TIMES_INT8_MONOID,      // identity: 1            terminal: 0
    GxB_TIMES_INT16_MONOID,     // identity: 1            terminal: 0
    GxB_TIMES_INT32_MONOID,     // identity: 1            terminal: 0
    GxB_TIMES_INT64_MONOID,     // identity: 1            terminal: 0
    GxB_TIMES_UINT8_MONOID,     // identity: 1            terminal: 0
    GxB_TIMES_UINT16_MONOID,    // identity: 1            terminal: 0
    GxB_TIMES_UINT32_MONOID,    // identity: 1            terminal: 0
    GxB_TIMES_UINT64_MONOID,    // identity: 1            terminal: 0
    GxB_TIMES_FP32_MONOID,      // identity: 1
    GxB_TIMES_FP64_MONOID,      // identity: 1

    // preferred names from the v1.3 spec:
    GrB_TIMES_MONOID_INT8,      // identity: 1            terminal: 0
    GrB_TIMES_MONOID_INT16,     // identity: 1            terminal: 0
    GrB_TIMES_MONOID_INT32,     // identity: 1            terminal: 0
    GrB_TIMES_MONOID_INT64,     // identity: 1            terminal: 0
    GrB_TIMES_MONOID_UINT8,     // identity: 1            terminal: 0
    GrB_TIMES_MONOID_UINT16,    // identity: 1            terminal: 0
    GrB_TIMES_MONOID_UINT32,    // identity: 1            terminal: 0
    GrB_TIMES_MONOID_UINT64,    // identity: 1            terminal: 0
    GrB_TIMES_MONOID_FP32,      // identity: 1
    GrB_TIMES_MONOID_FP64,      // identity: 1

    // complex monoids:
    GxB_TIMES_FC32_MONOID,      // identity: 1
    GxB_TIMES_FC64_MONOID,      // identity: 1

    //--------------------------------------------------------------------------
    // 13 ANY monoids:
    //--------------------------------------------------------------------------

    GxB_ANY_BOOL_MONOID,        // identity: any value    terminal: any value
    GxB_ANY_INT8_MONOID,        // identity: any value    terminal: any value
    GxB_ANY_INT16_MONOID,       // identity: any value    terminal: any value
    GxB_ANY_INT32_MONOID,       // identity: any value    terminal: any value
    GxB_ANY_INT64_MONOID,       // identity: any value    terminal: any value
    GxB_ANY_UINT8_MONOID,       // identity: any value    terminal: any value
    GxB_ANY_UINT16_MONOID,      // identity: any value    terminal: any value
    GxB_ANY_UINT32_MONOID,      // identity: any value    terminal: any value
    GxB_ANY_UINT64_MONOID,      // identity: any value    terminal: any value
    GxB_ANY_FP32_MONOID,        // identity: any value    terminal: any value
    GxB_ANY_FP64_MONOID,        // identity: any value    terminal: any value
    GxB_ANY_FC32_MONOID,        // identity: any value    terminal: any value
    GxB_ANY_FC64_MONOID,        // identity: any value    terminal: any value

    //--------------------------------------------------------------------------
    // 4 Boolean monoids: (see also the GxB_ANY_BOOL_MONOID above)
    //--------------------------------------------------------------------------

    // GxB_* boolean monoids, historical, use GrB_* instead:
    GxB_LOR_BOOL_MONOID,        // identity: false        terminal: true
    GxB_LAND_BOOL_MONOID,       // identity: true         terminal: false
    GxB_LXOR_BOOL_MONOID,       // identity: false
    GxB_LXNOR_BOOL_MONOID,      // identity: true
    GxB_EQ_BOOL_MONOID,         // (another name for GrB_LXNOR_MONOID_BOOL)

    // preferred names from the v1.3 spec:
    GrB_LOR_MONOID_BOOL,        // identity: false        terminal: true
    GrB_LAND_MONOID_BOOL,       // identity: true         terminal: false
    GrB_LXOR_MONOID_BOOL,       // identity: false
    GrB_LXNOR_MONOID_BOOL,      // identity: true

    //--------------------------------------------------------------------------
    // 16 Bitwise-or monoids:
    //--------------------------------------------------------------------------

    // BOR monoids (bitwise or):
    GxB_BOR_UINT8_MONOID,       // identity: 0   terminal: 0xFF
    GxB_BOR_UINT16_MONOID,      // identity: 0   terminal: 0xFFFF
    GxB_BOR_UINT32_MONOID,      // identity: 0   terminal: 0xFFFFFFFF
    GxB_BOR_UINT64_MONOID,      // identity: 0   terminal: 0xFFFFFFFFFFFFFFFF

    // BAND monoids (bitwise and):
    GxB_BAND_UINT8_MONOID,      // identity: 0xFF               terminal: 0
    GxB_BAND_UINT16_MONOID,     // identity: 0xFFFF             terminal: 0
    GxB_BAND_UINT32_MONOID,     // identity: 0xFFFFFFFF         terminal: 0
    GxB_BAND_UINT64_MONOID,     // identity: 0xFFFFFFFFFFFFFFFF terminal: 0

    // BXOR monoids (bitwise xor):
    GxB_BXOR_UINT8_MONOID,      // identity: 0
    GxB_BXOR_UINT16_MONOID,     // identity: 0
    GxB_BXOR_UINT32_MONOID,     // identity: 0
    GxB_BXOR_UINT64_MONOID,     // identity: 0

    // BXNOR monoids (bitwise xnor):
    GxB_BXNOR_UINT8_MONOID,     // identity: 0xFF
    GxB_BXNOR_UINT16_MONOID,    // identity: 0xFFFF
    GxB_BXNOR_UINT32_MONOID,    // identity: 0xFFFFFFFF
    GxB_BXNOR_UINT64_MONOID ;   // identity: 0xFFFFFFFFFFFFFFFF

//==============================================================================
// GrB_Semiring: built-in semirings
//==============================================================================

// Using built-in types and operators, SuiteSparse:GraphBLAS provides
// 1553 pre-defined, built-in semirings:

// 1000 semirings with a multiply operator TxT -> T where T is non-Boolean,
// from the complete cross product of:

//      5 monoids: MIN, MAX, PLUS, TIMES, ANY
//      20 multiply operators:
//          FIRST, SECOND, PAIR (=ONEB), MIN, MAX, PLUS, MINUS, TIMES, DIV,
//          RDIV, RMINUS
//          ISEQ, ISNE, ISGT, ISLT, ISGE, ISLE,
//          LOR, LAND, LXOR
//      10 non-Boolean real types, T
//
//      Note that min_pair, max_pair, times_pair are all identical to any_pair.
//      These 30 semirings are named below, but are internally remapped to
//      their corresponding any_pair semiring.

// 300 semirings with a comparator TxT -> bool, where T is
// non-Boolean, from the complete cross product of:

//      5 Boolean monoids: LAND, LOR, LXOR, EQ (=LXNOR), ANY
//      6 multiply operators: EQ, NE, GT, LT, GE, LE
//      10 non-Boolean real types, T

// 55 semirings with purely Boolean types, bool x bool -> bool, from the
// complete cross product of:

//      5 Boolean monoids LAND, LOR, LXOR, EQ (=LXNOR), ANY
//      11 multiply operators:
//          FIRST, SECOND, LOR, LAND, LXOR, EQ (=LXNOR), GT, LT, GE, LE,
//          PAIR (=ONEB)
//
//      Note that lor_pair, land_pair, and eq_pair are all identical to
//      any_pair.  These 3 semirings are named below, but are internally
//      remapped to any_pair_bool semiring.

// 54 complex semirings: TxT -> T where T is float complex or double complex:

//      3 complex monoids: PLUS, TIMES, ANY
//      9 complex multiply operators:
//          FIRST, SECOND, PAIR (=ONEB), PLUS, MINUS, TIMES, DIV, RDIV, RMINUS
//      2 complex types
//
//      Note that times_pair is identical to any_pair.
//      These 2 semirings are named below, but are internally remapped to
//      their corresponding any_pair semiring.

// 64 bitwise semirings: TxT -> T where T is an unsigned integer:

//      4 bitwise monoids: BOR, BAND, BXOR, BXNOR
//      4 bitwise multiply operators: BOR, BAND, BXOR, BXNOR
//      4 unsigned integer types: UINT8, UINT16, UINT32, UINT64

// 80 positional semirings: XxX -> T where T is int64 or int32, and the type of
// X is ignored:

//      5 monoids: MIN, MAX, PLUS, TIMES, ANY
//      8 multiply operators:
//          FIRSTI, FIRSTI1, FIRSTJ, FIRSTJ1,
//          SECONDI, SECONDI1, SECONDJ, SECONDJ1
//      2 types: int32, int64

// The ANY operator is also valid to use as a multiplicative operator in a
// semiring, but serves no purpose in that case.  The ANY operator is meant as
// a fast additive operator for a monoid, that terminates, or short-circuits,
// as soon as any value is found.  A valid user semiring can be constructed
// with ANY as the multiply operator, but they are not predefined below.

// Likewise, additional built-in operators can be used as multiplicative
// operators for floating-point semirings (POW, ATAN2, HYPOT, ...) and many
// more semirings can be constructed from bitwise monoids and many integer
// binary (non-bitwise) multiplicative operators, but these are not
// pre-defined.

// In the names below, each semiring has a name of the form GxB_add_mult_T
// where add is the additive monoid, mult is the multiply operator, and T is
// the type.  The type T is always the type of x and y for the z=mult(x,y)
// operator.  The monoid's three types and the ztype of the mult operator are
// always the same.  This is the type T for the first set, and Boolean for
// the second and third sets of semirngs.

// 1553 = 1000 + 300 + 55 + 54 + 64 + 80 semirings are named below, but 35 = 30
// + 3 + 2 are identical to the corresponding any_pair semirings of the same
// type.  For positional semirings, the mulitiply ops FIRSTJ and SECONDI are
// identical, as are FIRSTJ1 and SECONDI1.  These semirings still appear as
// predefined, for convenience.

GB_GLOBAL GrB_Semiring

//------------------------------------------------------------------------------
// 1000 non-Boolean semirings where all types are the same, given by suffix _T
//------------------------------------------------------------------------------

    // semirings with multiply op: z = FIRST (x,y), all types x,y,z the same:
    GxB_MIN_FIRST_INT8     , GxB_MAX_FIRST_INT8     , GxB_PLUS_FIRST_INT8    , GxB_TIMES_FIRST_INT8   , GxB_ANY_FIRST_INT8     ,
    GxB_MIN_FIRST_INT16    , GxB_MAX_FIRST_INT16    , GxB_PLUS_FIRST_INT16   , GxB_TIMES_FIRST_INT16  , GxB_ANY_FIRST_INT16    ,
    GxB_MIN_FIRST_INT32    , GxB_MAX_FIRST_INT32    , GxB_PLUS_FIRST_INT32   , GxB_TIMES_FIRST_INT32  , GxB_ANY_FIRST_INT32    ,
    GxB_MIN_FIRST_INT64    , GxB_MAX_FIRST_INT64    , GxB_PLUS_FIRST_INT64   , GxB_TIMES_FIRST_INT64  , GxB_ANY_FIRST_INT64    ,
    GxB_MIN_FIRST_UINT8    , GxB_MAX_FIRST_UINT8    , GxB_PLUS_FIRST_UINT8   , GxB_TIMES_FIRST_UINT8  , GxB_ANY_FIRST_UINT8    ,
    GxB_MIN_FIRST_UINT16   , GxB_MAX_FIRST_UINT16   , GxB_PLUS_FIRST_UINT16  , GxB_TIMES_FIRST_UINT16 , GxB_ANY_FIRST_UINT16   ,
    GxB_MIN_FIRST_UINT32   , GxB_MAX_FIRST_UINT32   , GxB_PLUS_FIRST_UINT32  , GxB_TIMES_FIRST_UINT32 , GxB_ANY_FIRST_UINT32   ,
    GxB_MIN_FIRST_UINT64   , GxB_MAX_FIRST_UINT64   , GxB_PLUS_FIRST_UINT64  , GxB_TIMES_FIRST_UINT64 , GxB_ANY_FIRST_UINT64   ,
    GxB_MIN_FIRST_FP32     , GxB_MAX_FIRST_FP32     , GxB_PLUS_FIRST_FP32    , GxB_TIMES_FIRST_FP32   , GxB_ANY_FIRST_FP32     ,
    GxB_MIN_FIRST_FP64     , GxB_MAX_FIRST_FP64     , GxB_PLUS_FIRST_FP64    , GxB_TIMES_FIRST_FP64   , GxB_ANY_FIRST_FP64     ,

    // semirings with multiply op: z = SECOND (x,y), all types x,y,z the same:
    GxB_MIN_SECOND_INT8    , GxB_MAX_SECOND_INT8    , GxB_PLUS_SECOND_INT8   , GxB_TIMES_SECOND_INT8  , GxB_ANY_SECOND_INT8    ,
    GxB_MIN_SECOND_INT16   , GxB_MAX_SECOND_INT16   , GxB_PLUS_SECOND_INT16  , GxB_TIMES_SECOND_INT16 , GxB_ANY_SECOND_INT16   ,
    GxB_MIN_SECOND_INT32   , GxB_MAX_SECOND_INT32   , GxB_PLUS_SECOND_INT32  , GxB_TIMES_SECOND_INT32 , GxB_ANY_SECOND_INT32   ,
    GxB_MIN_SECOND_INT64   , GxB_MAX_SECOND_INT64   , GxB_PLUS_SECOND_INT64  , GxB_TIMES_SECOND_INT64 , GxB_ANY_SECOND_INT64   ,
    GxB_MIN_SECOND_UINT8   , GxB_MAX_SECOND_UINT8   , GxB_PLUS_SECOND_UINT8  , GxB_TIMES_SECOND_UINT8 , GxB_ANY_SECOND_UINT8   ,
    GxB_MIN_SECOND_UINT16  , GxB_MAX_SECOND_UINT16  , GxB_PLUS_SECOND_UINT16 , GxB_TIMES_SECOND_UINT16, GxB_ANY_SECOND_UINT16  ,
    GxB_MIN_SECOND_UINT32  , GxB_MAX_SECOND_UINT32  , GxB_PLUS_SECOND_UINT32 , GxB_TIMES_SECOND_UINT32, GxB_ANY_SECOND_UINT32  ,
    GxB_MIN_SECOND_UINT64  , GxB_MAX_SECOND_UINT64  , GxB_PLUS_SECOND_UINT64 , GxB_TIMES_SECOND_UINT64, GxB_ANY_SECOND_UINT64  ,
    GxB_MIN_SECOND_FP32    , GxB_MAX_SECOND_FP32    , GxB_PLUS_SECOND_FP32   , GxB_TIMES_SECOND_FP32  , GxB_ANY_SECOND_FP32    ,
    GxB_MIN_SECOND_FP64    , GxB_MAX_SECOND_FP64    , GxB_PLUS_SECOND_FP64   , GxB_TIMES_SECOND_FP64  , GxB_ANY_SECOND_FP64    ,

    // semirings with multiply op: z = PAIR (x,y), all types x,y,z the same:
    // (note that min_pair, max_pair, times_pair are all identical to any_pair, and are marked below)
    GxB_MIN_PAIR_INT8  /**/, GxB_MAX_PAIR_INT8  /**/, GxB_PLUS_PAIR_INT8     , GxB_TIMES_PAIR_INT8  /**/, GxB_ANY_PAIR_INT8    ,
    GxB_MIN_PAIR_INT16 /**/, GxB_MAX_PAIR_INT16 /**/, GxB_PLUS_PAIR_INT16    , GxB_TIMES_PAIR_INT16 /**/, GxB_ANY_PAIR_INT16   ,
    GxB_MIN_PAIR_INT32 /**/, GxB_MAX_PAIR_INT32 /**/, GxB_PLUS_PAIR_INT32    , GxB_TIMES_PAIR_INT32 /**/, GxB_ANY_PAIR_INT32   ,
    GxB_MIN_PAIR_INT64 /**/, GxB_MAX_PAIR_INT64 /**/, GxB_PLUS_PAIR_INT64    , GxB_TIMES_PAIR_INT64 /**/, GxB_ANY_PAIR_INT64   ,
    GxB_MIN_PAIR_UINT8 /**/, GxB_MAX_PAIR_UINT8 /**/, GxB_PLUS_PAIR_UINT8    , GxB_TIMES_PAIR_UINT8 /**/, GxB_ANY_PAIR_UINT8   ,
    GxB_MIN_PAIR_UINT16/**/, GxB_MAX_PAIR_UINT16/**/, GxB_PLUS_PAIR_UINT16   , GxB_TIMES_PAIR_UINT16/**/, GxB_ANY_PAIR_UINT16  ,
    GxB_MIN_PAIR_UINT32/**/, GxB_MAX_PAIR_UINT32/**/, GxB_PLUS_PAIR_UINT32   , GxB_TIMES_PAIR_UINT32/**/, GxB_ANY_PAIR_UINT32  ,
    GxB_MIN_PAIR_UINT64/**/, GxB_MAX_PAIR_UINT64/**/, GxB_PLUS_PAIR_UINT64   , GxB_TIMES_PAIR_UINT64/**/, GxB_ANY_PAIR_UINT64  ,
    GxB_MIN_PAIR_FP32  /**/, GxB_MAX_PAIR_FP32  /**/, GxB_PLUS_PAIR_FP32     , GxB_TIMES_PAIR_FP32  /**/, GxB_ANY_PAIR_FP32    ,
    GxB_MIN_PAIR_FP64  /**/, GxB_MAX_PAIR_FP64  /**/, GxB_PLUS_PAIR_FP64     , GxB_TIMES_PAIR_FP64  /**/, GxB_ANY_PAIR_FP64    ,

    // semirings with multiply op: z = MIN (x,y), all types x,y,z the same:
    GxB_MIN_MIN_INT8       , GxB_MAX_MIN_INT8       , GxB_PLUS_MIN_INT8      , GxB_TIMES_MIN_INT8     , GxB_ANY_MIN_INT8       ,
    GxB_MIN_MIN_INT16      , GxB_MAX_MIN_INT16      , GxB_PLUS_MIN_INT16     , GxB_TIMES_MIN_INT16    , GxB_ANY_MIN_INT16      ,
    GxB_MIN_MIN_INT32      , GxB_MAX_MIN_INT32      , GxB_PLUS_MIN_INT32     , GxB_TIMES_MIN_INT32    , GxB_ANY_MIN_INT32      ,
    GxB_MIN_MIN_INT64      , GxB_MAX_MIN_INT64      , GxB_PLUS_MIN_INT64     , GxB_TIMES_MIN_INT64    , GxB_ANY_MIN_INT64      ,
    GxB_MIN_MIN_UINT8      , GxB_MAX_MIN_UINT8      , GxB_PLUS_MIN_UINT8     , GxB_TIMES_MIN_UINT8    , GxB_ANY_MIN_UINT8      ,
    GxB_MIN_MIN_UINT16     , GxB_MAX_MIN_UINT16     , GxB_PLUS_MIN_UINT16    , GxB_TIMES_MIN_UINT16   , GxB_ANY_MIN_UINT16     ,
    GxB_MIN_MIN_UINT32     , GxB_MAX_MIN_UINT32     , GxB_PLUS_MIN_UINT32    , GxB_TIMES_MIN_UINT32   , GxB_ANY_MIN_UINT32     ,
    GxB_MIN_MIN_UINT64     , GxB_MAX_MIN_UINT64     , GxB_PLUS_MIN_UINT64    , GxB_TIMES_MIN_UINT64   , GxB_ANY_MIN_UINT64     ,
    GxB_MIN_MIN_FP32       , GxB_MAX_MIN_FP32       , GxB_PLUS_MIN_FP32      , GxB_TIMES_MIN_FP32     , GxB_ANY_MIN_FP32       ,
    GxB_MIN_MIN_FP64       , GxB_MAX_MIN_FP64       , GxB_PLUS_MIN_FP64      , GxB_TIMES_MIN_FP64     , GxB_ANY_MIN_FP64       ,

    // semirings with multiply op: z = MAX (x,y), all types x,y,z the same:
    GxB_MIN_MAX_INT8       , GxB_MAX_MAX_INT8       , GxB_PLUS_MAX_INT8      , GxB_TIMES_MAX_INT8     , GxB_ANY_MAX_INT8       ,
    GxB_MIN_MAX_INT16      , GxB_MAX_MAX_INT16      , GxB_PLUS_MAX_INT16     , GxB_TIMES_MAX_INT16    , GxB_ANY_MAX_INT16      ,
    GxB_MIN_MAX_INT32      , GxB_MAX_MAX_INT32      , GxB_PLUS_MAX_INT32     , GxB_TIMES_MAX_INT32    , GxB_ANY_MAX_INT32      ,
    GxB_MIN_MAX_INT64      , GxB_MAX_MAX_INT64      , GxB_PLUS_MAX_INT64     , GxB_TIMES_MAX_INT64    , GxB_ANY_MAX_INT64      ,
    GxB_MIN_MAX_UINT8      , GxB_MAX_MAX_UINT8      , GxB_PLUS_MAX_UINT8     , GxB_TIMES_MAX_UINT8    , GxB_ANY_MAX_UINT8      ,
    GxB_MIN_MAX_UINT16     , GxB_MAX_MAX_UINT16     , GxB_PLUS_MAX_UINT16    , GxB_TIMES_MAX_UINT16   , GxB_ANY_MAX_UINT16     ,
    GxB_MIN_MAX_UINT32     , GxB_MAX_MAX_UINT32     , GxB_PLUS_MAX_UINT32    , GxB_TIMES_MAX_UINT32   , GxB_ANY_MAX_UINT32     ,
    GxB_MIN_MAX_UINT64     , GxB_MAX_MAX_UINT64     , GxB_PLUS_MAX_UINT64    , GxB_TIMES_MAX_UINT64   , GxB_ANY_MAX_UINT64     ,
    GxB_MIN_MAX_FP32       , GxB_MAX_MAX_FP32       , GxB_PLUS_MAX_FP32      , GxB_TIMES_MAX_FP32     , GxB_ANY_MAX_FP32       ,
    GxB_MIN_MAX_FP64       , GxB_MAX_MAX_FP64       , GxB_PLUS_MAX_FP64      , GxB_TIMES_MAX_FP64     , GxB_ANY_MAX_FP64       ,

    // semirings with multiply op: z = PLUS (x,y), all types x,y,z the same:
    GxB_MIN_PLUS_INT8      , GxB_MAX_PLUS_INT8      , GxB_PLUS_PLUS_INT8     , GxB_TIMES_PLUS_INT8    , GxB_ANY_PLUS_INT8      ,
    GxB_MIN_PLUS_INT16     , GxB_MAX_PLUS_INT16     , GxB_PLUS_PLUS_INT16    , GxB_TIMES_PLUS_INT16   , GxB_ANY_PLUS_INT16     ,
    GxB_MIN_PLUS_INT32     , GxB_MAX_PLUS_INT32     , GxB_PLUS_PLUS_INT32    , GxB_TIMES_PLUS_INT32   , GxB_ANY_PLUS_INT32     ,
    GxB_MIN_PLUS_INT64     , GxB_MAX_PLUS_INT64     , GxB_PLUS_PLUS_INT64    , GxB_TIMES_PLUS_INT64   , GxB_ANY_PLUS_INT64     ,
    GxB_MIN_PLUS_UINT8     , GxB_MAX_PLUS_UINT8     , GxB_PLUS_PLUS_UINT8    , GxB_TIMES_PLUS_UINT8   , GxB_ANY_PLUS_UINT8     ,
    GxB_MIN_PLUS_UINT16    , GxB_MAX_PLUS_UINT16    , GxB_PLUS_PLUS_UINT16   , GxB_TIMES_PLUS_UINT16  , GxB_ANY_PLUS_UINT16    ,
    GxB_MIN_PLUS_UINT32    , GxB_MAX_PLUS_UINT32    , GxB_PLUS_PLUS_UINT32   , GxB_TIMES_PLUS_UINT32  , GxB_ANY_PLUS_UINT32    ,
    GxB_MIN_PLUS_UINT64    , GxB_MAX_PLUS_UINT64    , GxB_PLUS_PLUS_UINT64   , GxB_TIMES_PLUS_UINT64  , GxB_ANY_PLUS_UINT64    ,
    GxB_MIN_PLUS_FP32      , GxB_MAX_PLUS_FP32      , GxB_PLUS_PLUS_FP32     , GxB_TIMES_PLUS_FP32    , GxB_ANY_PLUS_FP32      ,
    GxB_MIN_PLUS_FP64      , GxB_MAX_PLUS_FP64      , GxB_PLUS_PLUS_FP64     , GxB_TIMES_PLUS_FP64    , GxB_ANY_PLUS_FP64      ,

    // semirings with multiply op: z = MINUS (x,y), all types x,y,z the same:
    GxB_MIN_MINUS_INT8     , GxB_MAX_MINUS_INT8     , GxB_PLUS_MINUS_INT8    , GxB_TIMES_MINUS_INT8   , GxB_ANY_MINUS_INT8     ,
    GxB_MIN_MINUS_INT16    , GxB_MAX_MINUS_INT16    , GxB_PLUS_MINUS_INT16   , GxB_TIMES_MINUS_INT16  , GxB_ANY_MINUS_INT16    ,
    GxB_MIN_MINUS_INT32    , GxB_MAX_MINUS_INT32    , GxB_PLUS_MINUS_INT32   , GxB_TIMES_MINUS_INT32  , GxB_ANY_MINUS_INT32    ,
    GxB_MIN_MINUS_INT64    , GxB_MAX_MINUS_INT64    , GxB_PLUS_MINUS_INT64   , GxB_TIMES_MINUS_INT64  , GxB_ANY_MINUS_INT64    ,
    GxB_MIN_MINUS_UINT8    , GxB_MAX_MINUS_UINT8    , GxB_PLUS_MINUS_UINT8   , GxB_TIMES_MINUS_UINT8  , GxB_ANY_MINUS_UINT8    ,
    GxB_MIN_MINUS_UINT16   , GxB_MAX_MINUS_UINT16   , GxB_PLUS_MINUS_UINT16  , GxB_TIMES_MINUS_UINT16 , GxB_ANY_MINUS_UINT16   ,
    GxB_MIN_MINUS_UINT32   , GxB_MAX_MINUS_UINT32   , GxB_PLUS_MINUS_UINT32  , GxB_TIMES_MINUS_UINT32 , GxB_ANY_MINUS_UINT32   ,
    GxB_MIN_MINUS_UINT64   , GxB_MAX_MINUS_UINT64   , GxB_PLUS_MINUS_UINT64  , GxB_TIMES_MINUS_UINT64 , GxB_ANY_MINUS_UINT64   ,
    GxB_MIN_MINUS_FP32     , GxB_MAX_MINUS_FP32     , GxB_PLUS_MINUS_FP32    , GxB_TIMES_MINUS_FP32   , GxB_ANY_MINUS_FP32     ,
    GxB_MIN_MINUS_FP64     , GxB_MAX_MINUS_FP64     , GxB_PLUS_MINUS_FP64    , GxB_TIMES_MINUS_FP64   , GxB_ANY_MINUS_FP64     ,

    // semirings with multiply op: z = TIMES (x,y), all types x,y,z the same:
    GxB_MIN_TIMES_INT8     , GxB_MAX_TIMES_INT8     , GxB_PLUS_TIMES_INT8    , GxB_TIMES_TIMES_INT8   , GxB_ANY_TIMES_INT8     ,
    GxB_MIN_TIMES_INT16    , GxB_MAX_TIMES_INT16    , GxB_PLUS_TIMES_INT16   , GxB_TIMES_TIMES_INT16  , GxB_ANY_TIMES_INT16    ,
    GxB_MIN_TIMES_INT32    , GxB_MAX_TIMES_INT32    , GxB_PLUS_TIMES_INT32   , GxB_TIMES_TIMES_INT32  , GxB_ANY_TIMES_INT32    ,
    GxB_MIN_TIMES_INT64    , GxB_MAX_TIMES_INT64    , GxB_PLUS_TIMES_INT64   , GxB_TIMES_TIMES_INT64  , GxB_ANY_TIMES_INT64    ,
    GxB_MIN_TIMES_UINT8    , GxB_MAX_TIMES_UINT8    , GxB_PLUS_TIMES_UINT8   , GxB_TIMES_TIMES_UINT8  , GxB_ANY_TIMES_UINT8    ,
    GxB_MIN_TIMES_UINT16   , GxB_MAX_TIMES_UINT16   , GxB_PLUS_TIMES_UINT16  , GxB_TIMES_TIMES_UINT16 , GxB_ANY_TIMES_UINT16   ,
    GxB_MIN_TIMES_UINT32   , GxB_MAX_TIMES_UINT32   , GxB_PLUS_TIMES_UINT32  , GxB_TIMES_TIMES_UINT32 , GxB_ANY_TIMES_UINT32   ,
    GxB_MIN_TIMES_UINT64   , GxB_MAX_TIMES_UINT64   , GxB_PLUS_TIMES_UINT64  , GxB_TIMES_TIMES_UINT64 , GxB_ANY_TIMES_UINT64   ,
    GxB_MIN_TIMES_FP32     , GxB_MAX_TIMES_FP32     , GxB_PLUS_TIMES_FP32    , GxB_TIMES_TIMES_FP32   , GxB_ANY_TIMES_FP32     ,
    GxB_MIN_TIMES_FP64     , GxB_MAX_TIMES_FP64     , GxB_PLUS_TIMES_FP64    , GxB_TIMES_TIMES_FP64   , GxB_ANY_TIMES_FP64     ,

    // semirings with multiply op: z = DIV (x,y), all types x,y,z the same:
    GxB_MIN_DIV_INT8       , GxB_MAX_DIV_INT8       , GxB_PLUS_DIV_INT8      , GxB_TIMES_DIV_INT8     , GxB_ANY_DIV_INT8       ,
    GxB_MIN_DIV_INT16      , GxB_MAX_DIV_INT16      , GxB_PLUS_DIV_INT16     , GxB_TIMES_DIV_INT16    , GxB_ANY_DIV_INT16      ,
    GxB_MIN_DIV_INT32      , GxB_MAX_DIV_INT32      , GxB_PLUS_DIV_INT32     , GxB_TIMES_DIV_INT32    , GxB_ANY_DIV_INT32      ,
    GxB_MIN_DIV_INT64      , GxB_MAX_DIV_INT64      , GxB_PLUS_DIV_INT64     , GxB_TIMES_DIV_INT64    , GxB_ANY_DIV_INT64      ,
    GxB_MIN_DIV_UINT8      , GxB_MAX_DIV_UINT8      , GxB_PLUS_DIV_UINT8     , GxB_TIMES_DIV_UINT8    , GxB_ANY_DIV_UINT8      ,
    GxB_MIN_DIV_UINT16     , GxB_MAX_DIV_UINT16     , GxB_PLUS_DIV_UINT16    , GxB_TIMES_DIV_UINT16   , GxB_ANY_DIV_UINT16     ,
    GxB_MIN_DIV_UINT32     , GxB_MAX_DIV_UINT32     , GxB_PLUS_DIV_UINT32    , GxB_TIMES_DIV_UINT32   , GxB_ANY_DIV_UINT32     ,
    GxB_MIN_DIV_UINT64     , GxB_MAX_DIV_UINT64     , GxB_PLUS_DIV_UINT64    , GxB_TIMES_DIV_UINT64   , GxB_ANY_DIV_UINT64     ,
    GxB_MIN_DIV_FP32       , GxB_MAX_DIV_FP32       , GxB_PLUS_DIV_FP32      , GxB_TIMES_DIV_FP32     , GxB_ANY_DIV_FP32       ,
    GxB_MIN_DIV_FP64       , GxB_MAX_DIV_FP64       , GxB_PLUS_DIV_FP64      , GxB_TIMES_DIV_FP64     , GxB_ANY_DIV_FP64       ,

    // semirings with multiply op: z = RDIV (x,y), all types x,y,z the same:
    GxB_MIN_RDIV_INT8      , GxB_MAX_RDIV_INT8      , GxB_PLUS_RDIV_INT8     , GxB_TIMES_RDIV_INT8    , GxB_ANY_RDIV_INT8      ,
    GxB_MIN_RDIV_INT16     , GxB_MAX_RDIV_INT16     , GxB_PLUS_RDIV_INT16    , GxB_TIMES_RDIV_INT16   , GxB_ANY_RDIV_INT16     ,
    GxB_MIN_RDIV_INT32     , GxB_MAX_RDIV_INT32     , GxB_PLUS_RDIV_INT32    , GxB_TIMES_RDIV_INT32   , GxB_ANY_RDIV_INT32     ,
    GxB_MIN_RDIV_INT64     , GxB_MAX_RDIV_INT64     , GxB_PLUS_RDIV_INT64    , GxB_TIMES_RDIV_INT64   , GxB_ANY_RDIV_INT64     ,
    GxB_MIN_RDIV_UINT8     , GxB_MAX_RDIV_UINT8     , GxB_PLUS_RDIV_UINT8    , GxB_TIMES_RDIV_UINT8   , GxB_ANY_RDIV_UINT8     ,
    GxB_MIN_RDIV_UINT16    , GxB_MAX_RDIV_UINT16    , GxB_PLUS_RDIV_UINT16   , GxB_TIMES_RDIV_UINT16  , GxB_ANY_RDIV_UINT16    ,
    GxB_MIN_RDIV_UINT32    , GxB_MAX_RDIV_UINT32    , GxB_PLUS_RDIV_UINT32   , GxB_TIMES_RDIV_UINT32  , GxB_ANY_RDIV_UINT32    ,
    GxB_MIN_RDIV_UINT64    , GxB_MAX_RDIV_UINT64    , GxB_PLUS_RDIV_UINT64   , GxB_TIMES_RDIV_UINT64  , GxB_ANY_RDIV_UINT64    ,
    GxB_MIN_RDIV_FP32      , GxB_MAX_RDIV_FP32      , GxB_PLUS_RDIV_FP32     , GxB_TIMES_RDIV_FP32    , GxB_ANY_RDIV_FP32      ,
    GxB_MIN_RDIV_FP64      , GxB_MAX_RDIV_FP64      , GxB_PLUS_RDIV_FP64     , GxB_TIMES_RDIV_FP64    , GxB_ANY_RDIV_FP64      ,

    // semirings with multiply op: z = RMINUS (x,y), all types x,y,z the same:
    GxB_MIN_RMINUS_INT8    , GxB_MAX_RMINUS_INT8    , GxB_PLUS_RMINUS_INT8   , GxB_TIMES_RMINUS_INT8  , GxB_ANY_RMINUS_INT8    ,
    GxB_MIN_RMINUS_INT16   , GxB_MAX_RMINUS_INT16   , GxB_PLUS_RMINUS_INT16  , GxB_TIMES_RMINUS_INT16 , GxB_ANY_RMINUS_INT16   ,
    GxB_MIN_RMINUS_INT32   , GxB_MAX_RMINUS_INT32   , GxB_PLUS_RMINUS_INT32  , GxB_TIMES_RMINUS_INT32 , GxB_ANY_RMINUS_INT32   ,
    GxB_MIN_RMINUS_INT64   , GxB_MAX_RMINUS_INT64   , GxB_PLUS_RMINUS_INT64  , GxB_TIMES_RMINUS_INT64 , GxB_ANY_RMINUS_INT64   ,
    GxB_MIN_RMINUS_UINT8   , GxB_MAX_RMINUS_UINT8   , GxB_PLUS_RMINUS_UINT8  , GxB_TIMES_RMINUS_UINT8 , GxB_ANY_RMINUS_UINT8   ,
    GxB_MIN_RMINUS_UINT16  , GxB_MAX_RMINUS_UINT16  , GxB_PLUS_RMINUS_UINT16 , GxB_TIMES_RMINUS_UINT16, GxB_ANY_RMINUS_UINT16  ,
    GxB_MIN_RMINUS_UINT32  , GxB_MAX_RMINUS_UINT32  , GxB_PLUS_RMINUS_UINT32 , GxB_TIMES_RMINUS_UINT32, GxB_ANY_RMINUS_UINT32  ,
    GxB_MIN_RMINUS_UINT64  , GxB_MAX_RMINUS_UINT64  , GxB_PLUS_RMINUS_UINT64 , GxB_TIMES_RMINUS_UINT64, GxB_ANY_RMINUS_UINT64  ,
    GxB_MIN_RMINUS_FP32    , GxB_MAX_RMINUS_FP32    , GxB_PLUS_RMINUS_FP32   , GxB_TIMES_RMINUS_FP32  , GxB_ANY_RMINUS_FP32    ,
    GxB_MIN_RMINUS_FP64    , GxB_MAX_RMINUS_FP64    , GxB_PLUS_RMINUS_FP64   , GxB_TIMES_RMINUS_FP64  , GxB_ANY_RMINUS_FP64    ,

    // semirings with multiply op: z = ISEQ (x,y), all types x,y,z the same:
    GxB_MIN_ISEQ_INT8      , GxB_MAX_ISEQ_INT8      , GxB_PLUS_ISEQ_INT8     , GxB_TIMES_ISEQ_INT8    , GxB_ANY_ISEQ_INT8      ,
    GxB_MIN_ISEQ_INT16     , GxB_MAX_ISEQ_INT16     , GxB_PLUS_ISEQ_INT16    , GxB_TIMES_ISEQ_INT16   , GxB_ANY_ISEQ_INT16     ,
    GxB_MIN_ISEQ_INT32     , GxB_MAX_ISEQ_INT32     , GxB_PLUS_ISEQ_INT32    , GxB_TIMES_ISEQ_INT32   , GxB_ANY_ISEQ_INT32     ,
    GxB_MIN_ISEQ_INT64     , GxB_MAX_ISEQ_INT64     , GxB_PLUS_ISEQ_INT64    , GxB_TIMES_ISEQ_INT64   , GxB_ANY_ISEQ_INT64     ,
    GxB_MIN_ISEQ_UINT8     , GxB_MAX_ISEQ_UINT8     , GxB_PLUS_ISEQ_UINT8    , GxB_TIMES_ISEQ_UINT8   , GxB_ANY_ISEQ_UINT8     ,
    GxB_MIN_ISEQ_UINT16    , GxB_MAX_ISEQ_UINT16    , GxB_PLUS_ISEQ_UINT16   , GxB_TIMES_ISEQ_UINT16  , GxB_ANY_ISEQ_UINT16    ,
    GxB_MIN_ISEQ_UINT32    , GxB_MAX_ISEQ_UINT32    , GxB_PLUS_ISEQ_UINT32   , GxB_TIMES_ISEQ_UINT32  , GxB_ANY_ISEQ_UINT32    ,
    GxB_MIN_ISEQ_UINT64    , GxB_MAX_ISEQ_UINT64    , GxB_PLUS_ISEQ_UINT64   , GxB_TIMES_ISEQ_UINT64  , GxB_ANY_ISEQ_UINT64    ,
    GxB_MIN_ISEQ_FP32      , GxB_MAX_ISEQ_FP32      , GxB_PLUS_ISEQ_FP32     , GxB_TIMES_ISEQ_FP32    , GxB_ANY_ISEQ_FP32      ,
    GxB_MIN_ISEQ_FP64      , GxB_MAX_ISEQ_FP64      , GxB_PLUS_ISEQ_FP64     , GxB_TIMES_ISEQ_FP64    , GxB_ANY_ISEQ_FP64      ,

    // semirings with multiply op: z = ISNE (x,y), all types x,y,z the same:
    GxB_MIN_ISNE_INT8      , GxB_MAX_ISNE_INT8      , GxB_PLUS_ISNE_INT8     , GxB_TIMES_ISNE_INT8    , GxB_ANY_ISNE_INT8      ,
    GxB_MIN_ISNE_INT16     , GxB_MAX_ISNE_INT16     , GxB_PLUS_ISNE_INT16    , GxB_TIMES_ISNE_INT16   , GxB_ANY_ISNE_INT16     ,
    GxB_MIN_ISNE_INT32     , GxB_MAX_ISNE_INT32     , GxB_PLUS_ISNE_INT32    , GxB_TIMES_ISNE_INT32   , GxB_ANY_ISNE_INT32     ,
    GxB_MIN_ISNE_INT64     , GxB_MAX_ISNE_INT64     , GxB_PLUS_ISNE_INT64    , GxB_TIMES_ISNE_INT64   , GxB_ANY_ISNE_INT64     ,
    GxB_MIN_ISNE_UINT8     , GxB_MAX_ISNE_UINT8     , GxB_PLUS_ISNE_UINT8    , GxB_TIMES_ISNE_UINT8   , GxB_ANY_ISNE_UINT8     ,
    GxB_MIN_ISNE_UINT16    , GxB_MAX_ISNE_UINT16    , GxB_PLUS_ISNE_UINT16   , GxB_TIMES_ISNE_UINT16  , GxB_ANY_ISNE_UINT16    ,
    GxB_MIN_ISNE_UINT32    , GxB_MAX_ISNE_UINT32    , GxB_PLUS_ISNE_UINT32   , GxB_TIMES_ISNE_UINT32  , GxB_ANY_ISNE_UINT32    ,
    GxB_MIN_ISNE_UINT64    , GxB_MAX_ISNE_UINT64    , GxB_PLUS_ISNE_UINT64   , GxB_TIMES_ISNE_UINT64  , GxB_ANY_ISNE_UINT64    ,
    GxB_MIN_ISNE_FP32      , GxB_MAX_ISNE_FP32      , GxB_PLUS_ISNE_FP32     , GxB_TIMES_ISNE_FP32    , GxB_ANY_ISNE_FP32      ,
    GxB_MIN_ISNE_FP64      , GxB_MAX_ISNE_FP64      , GxB_PLUS_ISNE_FP64     , GxB_TIMES_ISNE_FP64    , GxB_ANY_ISNE_FP64      ,

    // semirings with multiply op: z = ISGT (x,y), all types x,y,z the same:
    GxB_MIN_ISGT_INT8      , GxB_MAX_ISGT_INT8      , GxB_PLUS_ISGT_INT8     , GxB_TIMES_ISGT_INT8    , GxB_ANY_ISGT_INT8      ,
    GxB_MIN_ISGT_INT16     , GxB_MAX_ISGT_INT16     , GxB_PLUS_ISGT_INT16    , GxB_TIMES_ISGT_INT16   , GxB_ANY_ISGT_INT16     ,
    GxB_MIN_ISGT_INT32     , GxB_MAX_ISGT_INT32     , GxB_PLUS_ISGT_INT32    , GxB_TIMES_ISGT_INT32   , GxB_ANY_ISGT_INT32     ,
    GxB_MIN_ISGT_INT64     , GxB_MAX_ISGT_INT64     , GxB_PLUS_ISGT_INT64    , GxB_TIMES_ISGT_INT64   , GxB_ANY_ISGT_INT64     ,
    GxB_MIN_ISGT_UINT8     , GxB_MAX_ISGT_UINT8     , GxB_PLUS_ISGT_UINT8    , GxB_TIMES_ISGT_UINT8   , GxB_ANY_ISGT_UINT8     ,
    GxB_MIN_ISGT_UINT16    , GxB_MAX_ISGT_UINT16    , GxB_PLUS_ISGT_UINT16   , GxB_TIMES_ISGT_UINT16  , GxB_ANY_ISGT_UINT16    ,
    GxB_MIN_ISGT_UINT32    , GxB_MAX_ISGT_UINT32    , GxB_PLUS_ISGT_UINT32   , GxB_TIMES_ISGT_UINT32  , GxB_ANY_ISGT_UINT32    ,
    GxB_MIN_ISGT_UINT64    , GxB_MAX_ISGT_UINT64    , GxB_PLUS_ISGT_UINT64   , GxB_TIMES_ISGT_UINT64  , GxB_ANY_ISGT_UINT64    ,
    GxB_MIN_ISGT_FP32      , GxB_MAX_ISGT_FP32      , GxB_PLUS_ISGT_FP32     , GxB_TIMES_ISGT_FP32    , GxB_ANY_ISGT_FP32      ,
    GxB_MIN_ISGT_FP64      , GxB_MAX_ISGT_FP64      , GxB_PLUS_ISGT_FP64     , GxB_TIMES_ISGT_FP64    , GxB_ANY_ISGT_FP64      ,

    // semirings with multiply op: z = ISLT (x,y), all types x,y,z the same:
    GxB_MIN_ISLT_INT8      , GxB_MAX_ISLT_INT8      , GxB_PLUS_ISLT_INT8     , GxB_TIMES_ISLT_INT8    , GxB_ANY_ISLT_INT8      ,
    GxB_MIN_ISLT_INT16     , GxB_MAX_ISLT_INT16     , GxB_PLUS_ISLT_INT16    , GxB_TIMES_ISLT_INT16   , GxB_ANY_ISLT_INT16     ,
    GxB_MIN_ISLT_INT32     , GxB_MAX_ISLT_INT32     , GxB_PLUS_ISLT_INT32    , GxB_TIMES_ISLT_INT32   , GxB_ANY_ISLT_INT32     ,
    GxB_MIN_ISLT_INT64     , GxB_MAX_ISLT_INT64     , GxB_PLUS_ISLT_INT64    , GxB_TIMES_ISLT_INT64   , GxB_ANY_ISLT_INT64     ,
    GxB_MIN_ISLT_UINT8     , GxB_MAX_ISLT_UINT8     , GxB_PLUS_ISLT_UINT8    , GxB_TIMES_ISLT_UINT8   , GxB_ANY_ISLT_UINT8     ,
    GxB_MIN_ISLT_UINT16    , GxB_MAX_ISLT_UINT16    , GxB_PLUS_ISLT_UINT16   , GxB_TIMES_ISLT_UINT16  , GxB_ANY_ISLT_UINT16    ,
    GxB_MIN_ISLT_UINT32    , GxB_MAX_ISLT_UINT32    , GxB_PLUS_ISLT_UINT32   , GxB_TIMES_ISLT_UINT32  , GxB_ANY_ISLT_UINT32    ,
    GxB_MIN_ISLT_UINT64    , GxB_MAX_ISLT_UINT64    , GxB_PLUS_ISLT_UINT64   , GxB_TIMES_ISLT_UINT64  , GxB_ANY_ISLT_UINT64    ,
    GxB_MIN_ISLT_FP32      , GxB_MAX_ISLT_FP32      , GxB_PLUS_ISLT_FP32     , GxB_TIMES_ISLT_FP32    , GxB_ANY_ISLT_FP32      ,
    GxB_MIN_ISLT_FP64      , GxB_MAX_ISLT_FP64      , GxB_PLUS_ISLT_FP64     , GxB_TIMES_ISLT_FP64    , GxB_ANY_ISLT_FP64      ,

    // semirings with multiply op: z = ISGE (x,y), all types x,y,z the same:
    GxB_MIN_ISGE_INT8      , GxB_MAX_ISGE_INT8      , GxB_PLUS_ISGE_INT8     , GxB_TIMES_ISGE_INT8    , GxB_ANY_ISGE_INT8      ,
    GxB_MIN_ISGE_INT16     , GxB_MAX_ISGE_INT16     , GxB_PLUS_ISGE_INT16    , GxB_TIMES_ISGE_INT16   , GxB_ANY_ISGE_INT16     ,
    GxB_MIN_ISGE_INT32     , GxB_MAX_ISGE_INT32     , GxB_PLUS_ISGE_INT32    , GxB_TIMES_ISGE_INT32   , GxB_ANY_ISGE_INT32     ,
    GxB_MIN_ISGE_INT64     , GxB_MAX_ISGE_INT64     , GxB_PLUS_ISGE_INT64    , GxB_TIMES_ISGE_INT64   , GxB_ANY_ISGE_INT64     ,
    GxB_MIN_ISGE_UINT8     , GxB_MAX_ISGE_UINT8     , GxB_PLUS_ISGE_UINT8    , GxB_TIMES_ISGE_UINT8   , GxB_ANY_ISGE_UINT8     ,
    GxB_MIN_ISGE_UINT16    , GxB_MAX_ISGE_UINT16    , GxB_PLUS_ISGE_UINT16   , GxB_TIMES_ISGE_UINT16  , GxB_ANY_ISGE_UINT16    ,
    GxB_MIN_ISGE_UINT32    , GxB_MAX_ISGE_UINT32    , GxB_PLUS_ISGE_UINT32   , GxB_TIMES_ISGE_UINT32  , GxB_ANY_ISGE_UINT32    ,
    GxB_MIN_ISGE_UINT64    , GxB_MAX_ISGE_UINT64    , GxB_PLUS_ISGE_UINT64   , GxB_TIMES_ISGE_UINT64  , GxB_ANY_ISGE_UINT64    ,
    GxB_MIN_ISGE_FP32      , GxB_MAX_ISGE_FP32      , GxB_PLUS_ISGE_FP32     , GxB_TIMES_ISGE_FP32    , GxB_ANY_ISGE_FP32      ,
    GxB_MIN_ISGE_FP64      , GxB_MAX_ISGE_FP64      , GxB_PLUS_ISGE_FP64     , GxB_TIMES_ISGE_FP64    , GxB_ANY_ISGE_FP64      ,

    // semirings with multiply op: z = ISLE (x,y), all types x,y,z the same:
    GxB_MIN_ISLE_INT8      , GxB_MAX_ISLE_INT8      , GxB_PLUS_ISLE_INT8     , GxB_TIMES_ISLE_INT8    , GxB_ANY_ISLE_INT8      ,
    GxB_MIN_ISLE_INT16     , GxB_MAX_ISLE_INT16     , GxB_PLUS_ISLE_INT16    , GxB_TIMES_ISLE_INT16   , GxB_ANY_ISLE_INT16     ,
    GxB_MIN_ISLE_INT32     , GxB_MAX_ISLE_INT32     , GxB_PLUS_ISLE_INT32    , GxB_TIMES_ISLE_INT32   , GxB_ANY_ISLE_INT32     ,
    GxB_MIN_ISLE_INT64     , GxB_MAX_ISLE_INT64     , GxB_PLUS_ISLE_INT64    , GxB_TIMES_ISLE_INT64   , GxB_ANY_ISLE_INT64     ,
    GxB_MIN_ISLE_UINT8     , GxB_MAX_ISLE_UINT8     , GxB_PLUS_ISLE_UINT8    , GxB_TIMES_ISLE_UINT8   , GxB_ANY_ISLE_UINT8     ,
    GxB_MIN_ISLE_UINT16    , GxB_MAX_ISLE_UINT16    , GxB_PLUS_ISLE_UINT16   , GxB_TIMES_ISLE_UINT16  , GxB_ANY_ISLE_UINT16    ,
    GxB_MIN_ISLE_UINT32    , GxB_MAX_ISLE_UINT32    , GxB_PLUS_ISLE_UINT32   , GxB_TIMES_ISLE_UINT32  , GxB_ANY_ISLE_UINT32    ,
    GxB_MIN_ISLE_UINT64    , GxB_MAX_ISLE_UINT64    , GxB_PLUS_ISLE_UINT64   , GxB_TIMES_ISLE_UINT64  , GxB_ANY_ISLE_UINT64    ,
    GxB_MIN_ISLE_FP32      , GxB_MAX_ISLE_FP32      , GxB_PLUS_ISLE_FP32     , GxB_TIMES_ISLE_FP32    , GxB_ANY_ISLE_FP32      ,
    GxB_MIN_ISLE_FP64      , GxB_MAX_ISLE_FP64      , GxB_PLUS_ISLE_FP64     , GxB_TIMES_ISLE_FP64    , GxB_ANY_ISLE_FP64      ,

    // semirings with multiply op: z = LOR (x,y), all types x,y,z the same:
    GxB_MIN_LOR_INT8       , GxB_MAX_LOR_INT8       , GxB_PLUS_LOR_INT8      , GxB_TIMES_LOR_INT8     , GxB_ANY_LOR_INT8       ,
    GxB_MIN_LOR_INT16      , GxB_MAX_LOR_INT16      , GxB_PLUS_LOR_INT16     , GxB_TIMES_LOR_INT16    , GxB_ANY_LOR_INT16      ,
    GxB_MIN_LOR_INT32      , GxB_MAX_LOR_INT32      , GxB_PLUS_LOR_INT32     , GxB_TIMES_LOR_INT32    , GxB_ANY_LOR_INT32      ,
    GxB_MIN_LOR_INT64      , GxB_MAX_LOR_INT64      , GxB_PLUS_LOR_INT64     , GxB_TIMES_LOR_INT64    , GxB_ANY_LOR_INT64      ,
    GxB_MIN_LOR_UINT8      , GxB_MAX_LOR_UINT8      , GxB_PLUS_LOR_UINT8     , GxB_TIMES_LOR_UINT8    , GxB_ANY_LOR_UINT8      ,
    GxB_MIN_LOR_UINT16     , GxB_MAX_LOR_UINT16     , GxB_PLUS_LOR_UINT16    , GxB_TIMES_LOR_UINT16   , GxB_ANY_LOR_UINT16     ,
    GxB_MIN_LOR_UINT32     , GxB_MAX_LOR_UINT32     , GxB_PLUS_LOR_UINT32    , GxB_TIMES_LOR_UINT32   , GxB_ANY_LOR_UINT32     ,
    GxB_MIN_LOR_UINT64     , GxB_MAX_LOR_UINT64     , GxB_PLUS_LOR_UINT64    , GxB_TIMES_LOR_UINT64   , GxB_ANY_LOR_UINT64     ,
    GxB_MIN_LOR_FP32       , GxB_MAX_LOR_FP32       , GxB_PLUS_LOR_FP32      , GxB_TIMES_LOR_FP32     , GxB_ANY_LOR_FP32       ,
    GxB_MIN_LOR_FP64       , GxB_MAX_LOR_FP64       , GxB_PLUS_LOR_FP64      , GxB_TIMES_LOR_FP64     , GxB_ANY_LOR_FP64       ,

    // semirings with multiply op: z = LAND (x,y), all types x,y,z the same:
    GxB_MIN_LAND_INT8      , GxB_MAX_LAND_INT8      , GxB_PLUS_LAND_INT8     , GxB_TIMES_LAND_INT8    , GxB_ANY_LAND_INT8      ,
    GxB_MIN_LAND_INT16     , GxB_MAX_LAND_INT16     , GxB_PLUS_LAND_INT16    , GxB_TIMES_LAND_INT16   , GxB_ANY_LAND_INT16     ,
    GxB_MIN_LAND_INT32     , GxB_MAX_LAND_INT32     , GxB_PLUS_LAND_INT32    , GxB_TIMES_LAND_INT32   , GxB_ANY_LAND_INT32     ,
    GxB_MIN_LAND_INT64     , GxB_MAX_LAND_INT64     , GxB_PLUS_LAND_INT64    , GxB_TIMES_LAND_INT64   , GxB_ANY_LAND_INT64     ,
    GxB_MIN_LAND_UINT8     , GxB_MAX_LAND_UINT8     , GxB_PLUS_LAND_UINT8    , GxB_TIMES_LAND_UINT8   , GxB_ANY_LAND_UINT8     ,
    GxB_MIN_LAND_UINT16    , GxB_MAX_LAND_UINT16    , GxB_PLUS_LAND_UINT16   , GxB_TIMES_LAND_UINT16  , GxB_ANY_LAND_UINT16    ,
    GxB_MIN_LAND_UINT32    , GxB_MAX_LAND_UINT32    , GxB_PLUS_LAND_UINT32   , GxB_TIMES_LAND_UINT32  , GxB_ANY_LAND_UINT32    ,
    GxB_MIN_LAND_UINT64    , GxB_MAX_LAND_UINT64    , GxB_PLUS_LAND_UINT64   , GxB_TIMES_LAND_UINT64  , GxB_ANY_LAND_UINT64    ,
    GxB_MIN_LAND_FP32      , GxB_MAX_LAND_FP32      , GxB_PLUS_LAND_FP32     , GxB_TIMES_LAND_FP32    , GxB_ANY_LAND_FP32      ,
    GxB_MIN_LAND_FP64      , GxB_MAX_LAND_FP64      , GxB_PLUS_LAND_FP64     , GxB_TIMES_LAND_FP64    , GxB_ANY_LAND_FP64      ,

    // semirings with multiply op: z = LXOR (x,y), all types x,y,z the same:
    GxB_MIN_LXOR_INT8      , GxB_MAX_LXOR_INT8      , GxB_PLUS_LXOR_INT8     , GxB_TIMES_LXOR_INT8    , GxB_ANY_LXOR_INT8      ,
    GxB_MIN_LXOR_INT16     , GxB_MAX_LXOR_INT16     , GxB_PLUS_LXOR_INT16    , GxB_TIMES_LXOR_INT16   , GxB_ANY_LXOR_INT16     ,
    GxB_MIN_LXOR_INT32     , GxB_MAX_LXOR_INT32     , GxB_PLUS_LXOR_INT32    , GxB_TIMES_LXOR_INT32   , GxB_ANY_LXOR_INT32     ,
    GxB_MIN_LXOR_INT64     , GxB_MAX_LXOR_INT64     , GxB_PLUS_LXOR_INT64    , GxB_TIMES_LXOR_INT64   , GxB_ANY_LXOR_INT64     ,
    GxB_MIN_LXOR_UINT8     , GxB_MAX_LXOR_UINT8     , GxB_PLUS_LXOR_UINT8    , GxB_TIMES_LXOR_UINT8   , GxB_ANY_LXOR_UINT8     ,
    GxB_MIN_LXOR_UINT16    , GxB_MAX_LXOR_UINT16    , GxB_PLUS_LXOR_UINT16   , GxB_TIMES_LXOR_UINT16  , GxB_ANY_LXOR_UINT16    ,
    GxB_MIN_LXOR_UINT32    , GxB_MAX_LXOR_UINT32    , GxB_PLUS_LXOR_UINT32   , GxB_TIMES_LXOR_UINT32  , GxB_ANY_LXOR_UINT32    ,
    GxB_MIN_LXOR_UINT64    , GxB_MAX_LXOR_UINT64    , GxB_PLUS_LXOR_UINT64   , GxB_TIMES_LXOR_UINT64  , GxB_ANY_LXOR_UINT64    ,
    GxB_MIN_LXOR_FP32      , GxB_MAX_LXOR_FP32      , GxB_PLUS_LXOR_FP32     , GxB_TIMES_LXOR_FP32    , GxB_ANY_LXOR_FP32      ,
    GxB_MIN_LXOR_FP64      , GxB_MAX_LXOR_FP64      , GxB_PLUS_LXOR_FP64     , GxB_TIMES_LXOR_FP64    , GxB_ANY_LXOR_FP64      ,

//------------------------------------------------------------------------------
// 300 semirings with a comparator TxT -> bool, where T is non-Boolean
//------------------------------------------------------------------------------

    // In the 4th column the GxB_EQ_*_* semirings could also be called
    // GxB_LXNOR_*_*, since the EQ and LXNOR boolean operators are identical
    // but those names are not included.

    // semirings with multiply op: z = EQ (x,y), where z is boolean and x,y are given by the suffix:
    GxB_LOR_EQ_INT8        , GxB_LAND_EQ_INT8       , GxB_LXOR_EQ_INT8       , GxB_EQ_EQ_INT8         , GxB_ANY_EQ_INT8        ,
    GxB_LOR_EQ_INT16       , GxB_LAND_EQ_INT16      , GxB_LXOR_EQ_INT16      , GxB_EQ_EQ_INT16        , GxB_ANY_EQ_INT16       ,
    GxB_LOR_EQ_INT32       , GxB_LAND_EQ_INT32      , GxB_LXOR_EQ_INT32      , GxB_EQ_EQ_INT32        , GxB_ANY_EQ_INT32       ,
    GxB_LOR_EQ_INT64       , GxB_LAND_EQ_INT64      , GxB_LXOR_EQ_INT64      , GxB_EQ_EQ_INT64        , GxB_ANY_EQ_INT64       ,
    GxB_LOR_EQ_UINT8       , GxB_LAND_EQ_UINT8      , GxB_LXOR_EQ_UINT8      , GxB_EQ_EQ_UINT8        , GxB_ANY_EQ_UINT8       ,
    GxB_LOR_EQ_UINT16      , GxB_LAND_EQ_UINT16     , GxB_LXOR_EQ_UINT16     , GxB_EQ_EQ_UINT16       , GxB_ANY_EQ_UINT16      ,
    GxB_LOR_EQ_UINT32      , GxB_LAND_EQ_UINT32     , GxB_LXOR_EQ_UINT32     , GxB_EQ_EQ_UINT32       , GxB_ANY_EQ_UINT32      ,
    GxB_LOR_EQ_UINT64      , GxB_LAND_EQ_UINT64     , GxB_LXOR_EQ_UINT64     , GxB_EQ_EQ_UINT64       , GxB_ANY_EQ_UINT64      ,
    GxB_LOR_EQ_FP32        , GxB_LAND_EQ_FP32       , GxB_LXOR_EQ_FP32       , GxB_EQ_EQ_FP32         , GxB_ANY_EQ_FP32        ,
    GxB_LOR_EQ_FP64        , GxB_LAND_EQ_FP64       , GxB_LXOR_EQ_FP64       , GxB_EQ_EQ_FP64         , GxB_ANY_EQ_FP64        ,

    // semirings with multiply op: z = NE (x,y), where z is boolean and x,y are given by the suffix:
    GxB_LOR_NE_INT8        , GxB_LAND_NE_INT8       , GxB_LXOR_NE_INT8       , GxB_EQ_NE_INT8         , GxB_ANY_NE_INT8        ,
    GxB_LOR_NE_INT16       , GxB_LAND_NE_INT16      , GxB_LXOR_NE_INT16      , GxB_EQ_NE_INT16        , GxB_ANY_NE_INT16       ,
    GxB_LOR_NE_INT32       , GxB_LAND_NE_INT32      , GxB_LXOR_NE_INT32      , GxB_EQ_NE_INT32        , GxB_ANY_NE_INT32       ,
    GxB_LOR_NE_INT64       , GxB_LAND_NE_INT64      , GxB_LXOR_NE_INT64      , GxB_EQ_NE_INT64        , GxB_ANY_NE_INT64       ,
    GxB_LOR_NE_UINT8       , GxB_LAND_NE_UINT8      , GxB_LXOR_NE_UINT8      , GxB_EQ_NE_UINT8        , GxB_ANY_NE_UINT8       ,
    GxB_LOR_NE_UINT16      , GxB_LAND_NE_UINT16     , GxB_LXOR_NE_UINT16     , GxB_EQ_NE_UINT16       , GxB_ANY_NE_UINT16      ,
    GxB_LOR_NE_UINT32      , GxB_LAND_NE_UINT32     , GxB_LXOR_NE_UINT32     , GxB_EQ_NE_UINT32       , GxB_ANY_NE_UINT32      ,
    GxB_LOR_NE_UINT64      , GxB_LAND_NE_UINT64     , GxB_LXOR_NE_UINT64     , GxB_EQ_NE_UINT64       , GxB_ANY_NE_UINT64      ,
    GxB_LOR_NE_FP32        , GxB_LAND_NE_FP32       , GxB_LXOR_NE_FP32       , GxB_EQ_NE_FP32         , GxB_ANY_NE_FP32        ,
    GxB_LOR_NE_FP64        , GxB_LAND_NE_FP64       , GxB_LXOR_NE_FP64       , GxB_EQ_NE_FP64         , GxB_ANY_NE_FP64        ,

    // semirings with multiply op: z = GT (x,y), where z is boolean and x,y are given by the suffix:
    GxB_LOR_GT_INT8        , GxB_LAND_GT_INT8       , GxB_LXOR_GT_INT8       , GxB_EQ_GT_INT8         , GxB_ANY_GT_INT8        ,
    GxB_LOR_GT_INT16       , GxB_LAND_GT_INT16      , GxB_LXOR_GT_INT16      , GxB_EQ_GT_INT16        , GxB_ANY_GT_INT16       ,
    GxB_LOR_GT_INT32       , GxB_LAND_GT_INT32      , GxB_LXOR_GT_INT32      , GxB_EQ_GT_INT32        , GxB_ANY_GT_INT32       ,
    GxB_LOR_GT_INT64       , GxB_LAND_GT_INT64      , GxB_LXOR_GT_INT64      , GxB_EQ_GT_INT64        , GxB_ANY_GT_INT64       ,
    GxB_LOR_GT_UINT8       , GxB_LAND_GT_UINT8      , GxB_LXOR_GT_UINT8      , GxB_EQ_GT_UINT8        , GxB_ANY_GT_UINT8       ,
    GxB_LOR_GT_UINT16      , GxB_LAND_GT_UINT16     , GxB_LXOR_GT_UINT16     , GxB_EQ_GT_UINT16       , GxB_ANY_GT_UINT16      ,
    GxB_LOR_GT_UINT32      , GxB_LAND_GT_UINT32     , GxB_LXOR_GT_UINT32     , GxB_EQ_GT_UINT32       , GxB_ANY_GT_UINT32      ,
    GxB_LOR_GT_UINT64      , GxB_LAND_GT_UINT64     , GxB_LXOR_GT_UINT64     , GxB_EQ_GT_UINT64       , GxB_ANY_GT_UINT64      ,
    GxB_LOR_GT_FP32        , GxB_LAND_GT_FP32       , GxB_LXOR_GT_FP32       , GxB_EQ_GT_FP32         , GxB_ANY_GT_FP32        ,
    GxB_LOR_GT_FP64        , GxB_LAND_GT_FP64       , GxB_LXOR_GT_FP64       , GxB_EQ_GT_FP64         , GxB_ANY_GT_FP64        ,

    // semirings with multiply op: z = LT (x,y), where z is boolean and x,y are given by the suffix:
    GxB_LOR_LT_INT8        , GxB_LAND_LT_INT8       , GxB_LXOR_LT_INT8       , GxB_EQ_LT_INT8         , GxB_ANY_LT_INT8        ,
    GxB_LOR_LT_INT16       , GxB_LAND_LT_INT16      , GxB_LXOR_LT_INT16      , GxB_EQ_LT_INT16        , GxB_ANY_LT_INT16       ,
    GxB_LOR_LT_INT32       , GxB_LAND_LT_INT32      , GxB_LXOR_LT_INT32      , GxB_EQ_LT_INT32        , GxB_ANY_LT_INT32       ,
    GxB_LOR_LT_INT64       , GxB_LAND_LT_INT64      , GxB_LXOR_LT_INT64      , GxB_EQ_LT_INT64        , GxB_ANY_LT_INT64       ,
    GxB_LOR_LT_UINT8       , GxB_LAND_LT_UINT8      , GxB_LXOR_LT_UINT8      , GxB_EQ_LT_UINT8        , GxB_ANY_LT_UINT8       ,
    GxB_LOR_LT_UINT16      , GxB_LAND_LT_UINT16     , GxB_LXOR_LT_UINT16     , GxB_EQ_LT_UINT16       , GxB_ANY_LT_UINT16      ,
    GxB_LOR_LT_UINT32      , GxB_LAND_LT_UINT32     , GxB_LXOR_LT_UINT32     , GxB_EQ_LT_UINT32       , GxB_ANY_LT_UINT32      ,
    GxB_LOR_LT_UINT64      , GxB_LAND_LT_UINT64     , GxB_LXOR_LT_UINT64     , GxB_EQ_LT_UINT64       , GxB_ANY_LT_UINT64      ,
    GxB_LOR_LT_FP32        , GxB_LAND_LT_FP32       , GxB_LXOR_LT_FP32       , GxB_EQ_LT_FP32         , GxB_ANY_LT_FP32        ,
    GxB_LOR_LT_FP64        , GxB_LAND_LT_FP64       , GxB_LXOR_LT_FP64       , GxB_EQ_LT_FP64         , GxB_ANY_LT_FP64        ,

    // semirings with multiply op: z = GE (x,y), where z is boolean and x,y are given by the suffix:
    GxB_LOR_GE_INT8        , GxB_LAND_GE_INT8       , GxB_LXOR_GE_INT8       , GxB_EQ_GE_INT8         , GxB_ANY_GE_INT8        ,
    GxB_LOR_GE_INT16       , GxB_LAND_GE_INT16      , GxB_LXOR_GE_INT16      , GxB_EQ_GE_INT16        , GxB_ANY_GE_INT16       ,
    GxB_LOR_GE_INT32       , GxB_LAND_GE_INT32      , GxB_LXOR_GE_INT32      , GxB_EQ_GE_INT32        , GxB_ANY_GE_INT32       ,
    GxB_LOR_GE_INT64       , GxB_LAND_GE_INT64      , GxB_LXOR_GE_INT64      , GxB_EQ_GE_INT64        , GxB_ANY_GE_INT64       ,
    GxB_LOR_GE_UINT8       , GxB_LAND_GE_UINT8      , GxB_LXOR_GE_UINT8      , GxB_EQ_GE_UINT8        , GxB_ANY_GE_UINT8       ,
    GxB_LOR_GE_UINT16      , GxB_LAND_GE_UINT16     , GxB_LXOR_GE_UINT16     , GxB_EQ_GE_UINT16       , GxB_ANY_GE_UINT16      ,
    GxB_LOR_GE_UINT32      , GxB_LAND_GE_UINT32     , GxB_LXOR_GE_UINT32     , GxB_EQ_GE_UINT32       , GxB_ANY_GE_UINT32      ,
    GxB_LOR_GE_UINT64      , GxB_LAND_GE_UINT64     , GxB_LXOR_GE_UINT64     , GxB_EQ_GE_UINT64       , GxB_ANY_GE_UINT64      ,
    GxB_LOR_GE_FP32        , GxB_LAND_GE_FP32       , GxB_LXOR_GE_FP32       , GxB_EQ_GE_FP32         , GxB_ANY_GE_FP32        ,
    GxB_LOR_GE_FP64        , GxB_LAND_GE_FP64       , GxB_LXOR_GE_FP64       , GxB_EQ_GE_FP64         , GxB_ANY_GE_FP64        ,

    // semirings with multiply op: z = LE (x,y), where z is boolean and x,y are given by the suffix:
    GxB_LOR_LE_INT8        , GxB_LAND_LE_INT8       , GxB_LXOR_LE_INT8       , GxB_EQ_LE_INT8         , GxB_ANY_LE_INT8        ,
    GxB_LOR_LE_INT16       , GxB_LAND_LE_INT16      , GxB_LXOR_LE_INT16      , GxB_EQ_LE_INT16        , GxB_ANY_LE_INT16       ,
    GxB_LOR_LE_INT32       , GxB_LAND_LE_INT32      , GxB_LXOR_LE_INT32      , GxB_EQ_LE_INT32        , GxB_ANY_LE_INT32       ,
    GxB_LOR_LE_INT64       , GxB_LAND_LE_INT64      , GxB_LXOR_LE_INT64      , GxB_EQ_LE_INT64        , GxB_ANY_LE_INT64       ,
    GxB_LOR_LE_UINT8       , GxB_LAND_LE_UINT8      , GxB_LXOR_LE_UINT8      , GxB_EQ_LE_UINT8        , GxB_ANY_LE_UINT8       ,
    GxB_LOR_LE_UINT16      , GxB_LAND_LE_UINT16     , GxB_LXOR_LE_UINT16     , GxB_EQ_LE_UINT16       , GxB_ANY_LE_UINT16      ,
    GxB_LOR_LE_UINT32      , GxB_LAND_LE_UINT32     , GxB_LXOR_LE_UINT32     , GxB_EQ_LE_UINT32       , GxB_ANY_LE_UINT32      ,
    GxB_LOR_LE_UINT64      , GxB_LAND_LE_UINT64     , GxB_LXOR_LE_UINT64     , GxB_EQ_LE_UINT64       , GxB_ANY_LE_UINT64      ,
    GxB_LOR_LE_FP32        , GxB_LAND_LE_FP32       , GxB_LXOR_LE_FP32       , GxB_EQ_LE_FP32         , GxB_ANY_LE_FP32        ,
    GxB_LOR_LE_FP64        , GxB_LAND_LE_FP64       , GxB_LXOR_LE_FP64       , GxB_EQ_LE_FP64         , GxB_ANY_LE_FP64        ,

//------------------------------------------------------------------------------
// 55 semirings with purely Boolean types, bool x bool -> bool
//------------------------------------------------------------------------------

    // Note that lor_pair, land_pair, and eq_pair are all identical to any_pair.
    // These 3 are marked below.  GxB_EQ_*_BOOL could be called
    // GxB_LXNOR_*_BOOL, and GxB_*_EQ_BOOL could be called GxB_*_LXNOR_BOOL,
    // but those names are not included.

    // purely boolean semirings in the form GxB_(add monoid)_(multiply operator)_BOOL:
    GxB_LOR_FIRST_BOOL     , GxB_LAND_FIRST_BOOL    , GxB_LXOR_FIRST_BOOL    , GxB_EQ_FIRST_BOOL      , GxB_ANY_FIRST_BOOL     ,
    GxB_LOR_SECOND_BOOL    , GxB_LAND_SECOND_BOOL   , GxB_LXOR_SECOND_BOOL   , GxB_EQ_SECOND_BOOL     , GxB_ANY_SECOND_BOOL    ,
    GxB_LOR_PAIR_BOOL/**/  , GxB_LAND_PAIR_BOOL/**/ , GxB_LXOR_PAIR_BOOL     , GxB_EQ_PAIR_BOOL/**/   , GxB_ANY_PAIR_BOOL      ,
    GxB_LOR_LOR_BOOL       , GxB_LAND_LOR_BOOL      , GxB_LXOR_LOR_BOOL      , GxB_EQ_LOR_BOOL        , GxB_ANY_LOR_BOOL       ,
    GxB_LOR_LAND_BOOL      , GxB_LAND_LAND_BOOL     , GxB_LXOR_LAND_BOOL     , GxB_EQ_LAND_BOOL       , GxB_ANY_LAND_BOOL      ,
    GxB_LOR_LXOR_BOOL      , GxB_LAND_LXOR_BOOL     , GxB_LXOR_LXOR_BOOL     , GxB_EQ_LXOR_BOOL       , GxB_ANY_LXOR_BOOL      ,
    GxB_LOR_EQ_BOOL        , GxB_LAND_EQ_BOOL       , GxB_LXOR_EQ_BOOL       , GxB_EQ_EQ_BOOL         , GxB_ANY_EQ_BOOL        ,
    GxB_LOR_GT_BOOL        , GxB_LAND_GT_BOOL       , GxB_LXOR_GT_BOOL       , GxB_EQ_GT_BOOL         , GxB_ANY_GT_BOOL        ,
    GxB_LOR_LT_BOOL        , GxB_LAND_LT_BOOL       , GxB_LXOR_LT_BOOL       , GxB_EQ_LT_BOOL         , GxB_ANY_LT_BOOL        ,
    GxB_LOR_GE_BOOL        , GxB_LAND_GE_BOOL       , GxB_LXOR_GE_BOOL       , GxB_EQ_GE_BOOL         , GxB_ANY_GE_BOOL        ,
    GxB_LOR_LE_BOOL        , GxB_LAND_LE_BOOL       , GxB_LXOR_LE_BOOL       , GxB_EQ_LE_BOOL         , GxB_ANY_LE_BOOL        ,

//------------------------------------------------------------------------------
// 54 complex semirings
//------------------------------------------------------------------------------

    // 3 monoids (plus, times, any), 2 types (FC32 and FC64), and 9
    // multiplicative operators.

    // Note that times_pair is identical to any_pair.
    // These 2 are marked below.

    GxB_PLUS_FIRST_FC32    , GxB_TIMES_FIRST_FC32   , GxB_ANY_FIRST_FC32     ,
    GxB_PLUS_FIRST_FC64    , GxB_TIMES_FIRST_FC64   , GxB_ANY_FIRST_FC64     ,

    GxB_PLUS_SECOND_FC32   , GxB_TIMES_SECOND_FC32  , GxB_ANY_SECOND_FC32    ,
    GxB_PLUS_SECOND_FC64   , GxB_TIMES_SECOND_FC64  , GxB_ANY_SECOND_FC64    ,

    GxB_PLUS_PAIR_FC32     , GxB_TIMES_PAIR_FC32/**/, GxB_ANY_PAIR_FC32      ,
    GxB_PLUS_PAIR_FC64     , GxB_TIMES_PAIR_FC64/**/, GxB_ANY_PAIR_FC64      ,

    GxB_PLUS_PLUS_FC32     , GxB_TIMES_PLUS_FC32    , GxB_ANY_PLUS_FC32      ,
    GxB_PLUS_PLUS_FC64     , GxB_TIMES_PLUS_FC64    , GxB_ANY_PLUS_FC64      ,

    GxB_PLUS_MINUS_FC32    , GxB_TIMES_MINUS_FC32   , GxB_ANY_MINUS_FC32     ,
    GxB_PLUS_MINUS_FC64    , GxB_TIMES_MINUS_FC64   , GxB_ANY_MINUS_FC64     ,

    GxB_PLUS_TIMES_FC32    , GxB_TIMES_TIMES_FC32   , GxB_ANY_TIMES_FC32     ,
    GxB_PLUS_TIMES_FC64    , GxB_TIMES_TIMES_FC64   , GxB_ANY_TIMES_FC64     ,

    GxB_PLUS_DIV_FC32      , GxB_TIMES_DIV_FC32     , GxB_ANY_DIV_FC32       ,
    GxB_PLUS_DIV_FC64      , GxB_TIMES_DIV_FC64     , GxB_ANY_DIV_FC64       ,

    GxB_PLUS_RDIV_FC32     , GxB_TIMES_RDIV_FC32    , GxB_ANY_RDIV_FC32      ,
    GxB_PLUS_RDIV_FC64     , GxB_TIMES_RDIV_FC64    , GxB_ANY_RDIV_FC64      ,

    GxB_PLUS_RMINUS_FC32   , GxB_TIMES_RMINUS_FC32  , GxB_ANY_RMINUS_FC32    ,
    GxB_PLUS_RMINUS_FC64   , GxB_TIMES_RMINUS_FC64  , GxB_ANY_RMINUS_FC64    ,

//------------------------------------------------------------------------------
// 64 bitwise semirings
//------------------------------------------------------------------------------

    // monoids: (BOR, BAND, BXOR, BXNOR) x
    // mult:    (BOR, BAND, BXOR, BXNOR) x
    // types:   (UINT8, UINT16, UINT32, UINT64)

    GxB_BOR_BOR_UINT8      , GxB_BOR_BOR_UINT16     , GxB_BOR_BOR_UINT32     , GxB_BOR_BOR_UINT64     ,
    GxB_BOR_BAND_UINT8     , GxB_BOR_BAND_UINT16    , GxB_BOR_BAND_UINT32    , GxB_BOR_BAND_UINT64    ,
    GxB_BOR_BXOR_UINT8     , GxB_BOR_BXOR_UINT16    , GxB_BOR_BXOR_UINT32    , GxB_BOR_BXOR_UINT64    ,
    GxB_BOR_BXNOR_UINT8    , GxB_BOR_BXNOR_UINT16   , GxB_BOR_BXNOR_UINT32   , GxB_BOR_BXNOR_UINT64   ,

    GxB_BAND_BOR_UINT8     , GxB_BAND_BOR_UINT16    , GxB_BAND_BOR_UINT32    , GxB_BAND_BOR_UINT64    ,
    GxB_BAND_BAND_UINT8    , GxB_BAND_BAND_UINT16   , GxB_BAND_BAND_UINT32   , GxB_BAND_BAND_UINT64   ,
    GxB_BAND_BXOR_UINT8    , GxB_BAND_BXOR_UINT16   , GxB_BAND_BXOR_UINT32   , GxB_BAND_BXOR_UINT64   ,
    GxB_BAND_BXNOR_UINT8   , GxB_BAND_BXNOR_UINT16  , GxB_BAND_BXNOR_UINT32  , GxB_BAND_BXNOR_UINT64  ,

    GxB_BXOR_BOR_UINT8     , GxB_BXOR_BOR_UINT16    , GxB_BXOR_BOR_UINT32    , GxB_BXOR_BOR_UINT64    ,
    GxB_BXOR_BAND_UINT8    , GxB_BXOR_BAND_UINT16   , GxB_BXOR_BAND_UINT32   , GxB_BXOR_BAND_UINT64   ,
    GxB_BXOR_BXOR_UINT8    , GxB_BXOR_BXOR_UINT16   , GxB_BXOR_BXOR_UINT32   , GxB_BXOR_BXOR_UINT64   ,
    GxB_BXOR_BXNOR_UINT8   , GxB_BXOR_BXNOR_UINT16  , GxB_BXOR_BXNOR_UINT32  , GxB_BXOR_BXNOR_UINT64  ,

    GxB_BXNOR_BOR_UINT8    , GxB_BXNOR_BOR_UINT16   , GxB_BXNOR_BOR_UINT32   , GxB_BXNOR_BOR_UINT64   ,
    GxB_BXNOR_BAND_UINT8   , GxB_BXNOR_BAND_UINT16  , GxB_BXNOR_BAND_UINT32  , GxB_BXNOR_BAND_UINT64  ,
    GxB_BXNOR_BXOR_UINT8   , GxB_BXNOR_BXOR_UINT16  , GxB_BXNOR_BXOR_UINT32  , GxB_BXNOR_BXOR_UINT64  ,
    GxB_BXNOR_BXNOR_UINT8  , GxB_BXNOR_BXNOR_UINT16 , GxB_BXNOR_BXNOR_UINT32 , GxB_BXNOR_BXNOR_UINT64 ,

//------------------------------------------------------------------------------
// 80 positional semirings
//------------------------------------------------------------------------------

    // monoids: (MIN, MAX, ANY, PLUS, TIMES) x
    // mult:    (FIRSTI, FIRSTI1, FIRSTJ, FIRSTJ1, SECONDI, SECONDI1, SECONDJ, SECONDJ1)
    // types:   (INT32, INT64)

    GxB_MIN_FIRSTI_INT32,     GxB_MIN_FIRSTI_INT64,
    GxB_MAX_FIRSTI_INT32,     GxB_MAX_FIRSTI_INT64,
    GxB_ANY_FIRSTI_INT32,     GxB_ANY_FIRSTI_INT64,
    GxB_PLUS_FIRSTI_INT32,    GxB_PLUS_FIRSTI_INT64,
    GxB_TIMES_FIRSTI_INT32,   GxB_TIMES_FIRSTI_INT64,

    GxB_MIN_FIRSTI1_INT32,    GxB_MIN_FIRSTI1_INT64,
    GxB_MAX_FIRSTI1_INT32,    GxB_MAX_FIRSTI1_INT64,
    GxB_ANY_FIRSTI1_INT32,    GxB_ANY_FIRSTI1_INT64,
    GxB_PLUS_FIRSTI1_INT32,   GxB_PLUS_FIRSTI1_INT64,
    GxB_TIMES_FIRSTI1_INT32,  GxB_TIMES_FIRSTI1_INT64,

    GxB_MIN_FIRSTJ_INT32,     GxB_MIN_FIRSTJ_INT64,
    GxB_MAX_FIRSTJ_INT32,     GxB_MAX_FIRSTJ_INT64,
    GxB_ANY_FIRSTJ_INT32,     GxB_ANY_FIRSTJ_INT64,
    GxB_PLUS_FIRSTJ_INT32,    GxB_PLUS_FIRSTJ_INT64,
    GxB_TIMES_FIRSTJ_INT32,   GxB_TIMES_FIRSTJ_INT64,

    GxB_MIN_FIRSTJ1_INT32,    GxB_MIN_FIRSTJ1_INT64,
    GxB_MAX_FIRSTJ1_INT32,    GxB_MAX_FIRSTJ1_INT64,
    GxB_ANY_FIRSTJ1_INT32,    GxB_ANY_FIRSTJ1_INT64,
    GxB_PLUS_FIRSTJ1_INT32,   GxB_PLUS_FIRSTJ1_INT64,
    GxB_TIMES_FIRSTJ1_INT32,  GxB_TIMES_FIRSTJ1_INT64,

    GxB_MIN_SECONDI_INT32,    GxB_MIN_SECONDI_INT64,
    GxB_MAX_SECONDI_INT32,    GxB_MAX_SECONDI_INT64,
    GxB_ANY_SECONDI_INT32,    GxB_ANY_SECONDI_INT64,
    GxB_PLUS_SECONDI_INT32,   GxB_PLUS_SECONDI_INT64,
    GxB_TIMES_SECONDI_INT32,  GxB_TIMES_SECONDI_INT64,

    GxB_MIN_SECONDI1_INT32,   GxB_MIN_SECONDI1_INT64,
    GxB_MAX_SECONDI1_INT32,   GxB_MAX_SECONDI1_INT64,
    GxB_ANY_SECONDI1_INT32,   GxB_ANY_SECONDI1_INT64,
    GxB_PLUS_SECONDI1_INT32,  GxB_PLUS_SECONDI1_INT64,
    GxB_TIMES_SECONDI1_INT32, GxB_TIMES_SECONDI1_INT64,

    GxB_MIN_SECONDJ_INT32,    GxB_MIN_SECONDJ_INT64,
    GxB_MAX_SECONDJ_INT32,    GxB_MAX_SECONDJ_INT64,
    GxB_ANY_SECONDJ_INT32,    GxB_ANY_SECONDJ_INT64,
    GxB_PLUS_SECONDJ_INT32,   GxB_PLUS_SECONDJ_INT64,
    GxB_TIMES_SECONDJ_INT32,  GxB_TIMES_SECONDJ_INT64,

    GxB_MIN_SECONDJ1_INT32,   GxB_MIN_SECONDJ1_INT64,
    GxB_MAX_SECONDJ1_INT32,   GxB_MAX_SECONDJ1_INT64,
    GxB_ANY_SECONDJ1_INT32,   GxB_ANY_SECONDJ1_INT64,
    GxB_PLUS_SECONDJ1_INT32,  GxB_PLUS_SECONDJ1_INT64,
    GxB_TIMES_SECONDJ1_INT32, GxB_TIMES_SECONDJ1_INT64 ;

//------------------------------------------------------------------------------
// GrB_* semirings
//------------------------------------------------------------------------------

// The v1.3 C API for GraphBLAS adds the following 124 predefined semirings,
// with GrB_* names.  They are identical to 124 GxB_* semirings defined above,
// with the same name, except that GrB_LXNOR_LOR_SEMIRING_BOOL is identical to
// GxB_EQ_LOR_BOOL (since GrB_EQ_BOOL == GrB_LXNOR).  The old names are listed
// below alongside each new name; the new GrB_* names are preferred.

// 12 kinds of GrB_* semirings are available for all 10 real non-boolean types:

    // PLUS_TIMES, PLUS_MIN,
    // MIN_PLUS, MIN_TIMES, MIN_FIRST, MIN_SECOND, MIN_MAX,
    // MAX_PLUS, MAX_TIMES, MAX_FIRST, MAX_SECOND, MAX_MIN

// and 4 semirings for boolean only:

    // LOR_LAND, LAND_LOR, LXOR_LAND, LXNOR_LOR.

// GxB_* semirings corresponding to the equivalent GrB_* semiring are
// historical.

GB_GLOBAL GrB_Semiring

    //--------------------------------------------------------------------------
    // 20 semirings with PLUS monoids
    //--------------------------------------------------------------------------

    // PLUS_TIMES semirings for all 10 real, non-boolean types:
    GrB_PLUS_TIMES_SEMIRING_INT8,       // GxB_PLUS_TIMES_INT8
    GrB_PLUS_TIMES_SEMIRING_INT16,      // GxB_PLUS_TIMES_INT16
    GrB_PLUS_TIMES_SEMIRING_INT32,      // GxB_PLUS_TIMES_INT32
    GrB_PLUS_TIMES_SEMIRING_INT64,      // GxB_PLUS_TIMES_INT64
    GrB_PLUS_TIMES_SEMIRING_UINT8,      // GxB_PLUS_TIMES_UINT8
    GrB_PLUS_TIMES_SEMIRING_UINT16,     // GxB_PLUS_TIMES_UINT16
    GrB_PLUS_TIMES_SEMIRING_UINT32,     // GxB_PLUS_TIMES_UINT32
    GrB_PLUS_TIMES_SEMIRING_UINT64,     // GxB_PLUS_TIMES_UINT64
    GrB_PLUS_TIMES_SEMIRING_FP32,       // GxB_PLUS_TIMES_FP32
    GrB_PLUS_TIMES_SEMIRING_FP64,       // GxB_PLUS_TIMES_FP64

    // PLUS_MIN semirings for all 10 real, non-boolean types:
    GrB_PLUS_MIN_SEMIRING_INT8,         // GxB_PLUS_MIN_INT8
    GrB_PLUS_MIN_SEMIRING_INT16,        // GxB_PLUS_MIN_INT16
    GrB_PLUS_MIN_SEMIRING_INT32,        // GxB_PLUS_MIN_INT32
    GrB_PLUS_MIN_SEMIRING_INT64,        // GxB_PLUS_MIN_INT64
    GrB_PLUS_MIN_SEMIRING_UINT8,        // GxB_PLUS_MIN_UINT8
    GrB_PLUS_MIN_SEMIRING_UINT16,       // GxB_PLUS_MIN_UINT16
    GrB_PLUS_MIN_SEMIRING_UINT32,       // GxB_PLUS_MIN_UINT32
    GrB_PLUS_MIN_SEMIRING_UINT64,       // GxB_PLUS_MIN_UINT64
    GrB_PLUS_MIN_SEMIRING_FP32,         // GxB_PLUS_MIN_FP32
    GrB_PLUS_MIN_SEMIRING_FP64,         // GxB_PLUS_MIN_FP64

    //--------------------------------------------------------------------------
    // 50 semirings with MIN monoids
    //--------------------------------------------------------------------------

    // MIN_PLUS semirings for all 10 real, non-boolean types:
    GrB_MIN_PLUS_SEMIRING_INT8,         // GxB_MIN_PLUS_INT8
    GrB_MIN_PLUS_SEMIRING_INT16,        // GxB_MIN_PLUS_INT16
    GrB_MIN_PLUS_SEMIRING_INT32,        // GxB_MIN_PLUS_INT32
    GrB_MIN_PLUS_SEMIRING_INT64,        // GxB_MIN_PLUS_INT64
    GrB_MIN_PLUS_SEMIRING_UINT8,        // GxB_MIN_PLUS_UINT8
    GrB_MIN_PLUS_SEMIRING_UINT16,       // GxB_MIN_PLUS_UINT16
    GrB_MIN_PLUS_SEMIRING_UINT32,       // GxB_MIN_PLUS_UINT32
    GrB_MIN_PLUS_SEMIRING_UINT64,       // GxB_MIN_PLUS_UINT64
    GrB_MIN_PLUS_SEMIRING_FP32,         // GxB_MIN_PLUS_FP32
    GrB_MIN_PLUS_SEMIRING_FP64,         // GxB_MIN_PLUS_FP64

    // MIN_TIMES semirings for all 10 real, non-boolean types:
    GrB_MIN_TIMES_SEMIRING_INT8,        // GxB_MIN_TIMES_INT8
    GrB_MIN_TIMES_SEMIRING_INT16,       // GxB_MIN_TIMES_INT16
    GrB_MIN_TIMES_SEMIRING_INT32,       // GxB_MIN_TIMES_INT32
    GrB_MIN_TIMES_SEMIRING_INT64,       // GxB_MIN_TIMES_INT64
    GrB_MIN_TIMES_SEMIRING_UINT8,       // GxB_MIN_TIMES_UINT8
    GrB_MIN_TIMES_SEMIRING_UINT16,      // GxB_MIN_TIMES_UINT16
    GrB_MIN_TIMES_SEMIRING_UINT32,      // GxB_MIN_TIMES_UINT32
    GrB_MIN_TIMES_SEMIRING_UINT64,      // GxB_MIN_TIMES_UINT64
    GrB_MIN_TIMES_SEMIRING_FP32,        // GxB_MIN_TIMES_FP32
    GrB_MIN_TIMES_SEMIRING_FP64,        // GxB_MIN_TIMES_FP64

    // MIN_FIRST semirings for all 10 real, non-boolean types:
    GrB_MIN_FIRST_SEMIRING_INT8,        // GxB_MIN_FIRST_INT8
    GrB_MIN_FIRST_SEMIRING_INT16,       // GxB_MIN_FIRST_INT16
    GrB_MIN_FIRST_SEMIRING_INT32,       // GxB_MIN_FIRST_INT32
    GrB_MIN_FIRST_SEMIRING_INT64,       // GxB_MIN_FIRST_INT64
    GrB_MIN_FIRST_SEMIRING_UINT8,       // GxB_MIN_FIRST_UINT8
    GrB_MIN_FIRST_SEMIRING_UINT16,      // GxB_MIN_FIRST_UINT16
    GrB_MIN_FIRST_SEMIRING_UINT32,      // GxB_MIN_FIRST_UINT32
    GrB_MIN_FIRST_SEMIRING_UINT64,      // GxB_MIN_FIRST_UINT64
    GrB_MIN_FIRST_SEMIRING_FP32,        // GxB_MIN_FIRST_FP32
    GrB_MIN_FIRST_SEMIRING_FP64,        // GxB_MIN_FIRST_FP64

    // MIN_SECOND semirings for all 10 real, non-boolean types:
    GrB_MIN_SECOND_SEMIRING_INT8,       // GxB_MIN_SECOND_INT8
    GrB_MIN_SECOND_SEMIRING_INT16,      // GxB_MIN_SECOND_INT16
    GrB_MIN_SECOND_SEMIRING_INT32,      // GxB_MIN_SECOND_INT32
    GrB_MIN_SECOND_SEMIRING_INT64,      // GxB_MIN_SECOND_INT64
    GrB_MIN_SECOND_SEMIRING_UINT8,      // GxB_MIN_SECOND_UINT8
    GrB_MIN_SECOND_SEMIRING_UINT16,     // GxB_MIN_SECOND_UINT16
    GrB_MIN_SECOND_SEMIRING_UINT32,     // GxB_MIN_SECOND_UINT32
    GrB_MIN_SECOND_SEMIRING_UINT64,     // GxB_MIN_SECOND_UINT64
    GrB_MIN_SECOND_SEMIRING_FP32,       // GxB_MIN_SECOND_FP32
    GrB_MIN_SECOND_SEMIRING_FP64,       // GxB_MIN_SECOND_FP64

    // MIN_MAX semirings for all 10 real, non-boolean types:
    GrB_MIN_MAX_SEMIRING_INT8,          // GxB_MIN_MAX_INT8
    GrB_MIN_MAX_SEMIRING_INT16,         // GxB_MIN_MAX_INT16
    GrB_MIN_MAX_SEMIRING_INT32,         // GxB_MIN_MAX_INT32
    GrB_MIN_MAX_SEMIRING_INT64,         // GxB_MIN_MAX_INT64
    GrB_MIN_MAX_SEMIRING_UINT8,         // GxB_MIN_MAX_UINT8
    GrB_MIN_MAX_SEMIRING_UINT16,        // GxB_MIN_MAX_UINT16
    GrB_MIN_MAX_SEMIRING_UINT32,        // GxB_MIN_MAX_UINT32
    GrB_MIN_MAX_SEMIRING_UINT64,        // GxB_MIN_MAX_UINT64
    GrB_MIN_MAX_SEMIRING_FP32,          // GxB_MIN_MAX_FP32
    GrB_MIN_MAX_SEMIRING_FP64,          // GxB_MIN_MAX_FP64

    //--------------------------------------------------------------------------
    // 50 semirings with MAX monoids
    //--------------------------------------------------------------------------

    // MAX_PLUS semirings for all 10 real, non-boolean types
    GrB_MAX_PLUS_SEMIRING_INT8,         // GxB_MAX_PLUS_INT8
    GrB_MAX_PLUS_SEMIRING_INT16,        // GxB_MAX_PLUS_INT16
    GrB_MAX_PLUS_SEMIRING_INT32,        // GxB_MAX_PLUS_INT32
    GrB_MAX_PLUS_SEMIRING_INT64,        // GxB_MAX_PLUS_INT64
    GrB_MAX_PLUS_SEMIRING_UINT8,        // GxB_MAX_PLUS_UINT8
    GrB_MAX_PLUS_SEMIRING_UINT16,       // GxB_MAX_PLUS_UINT16
    GrB_MAX_PLUS_SEMIRING_UINT32,       // GxB_MAX_PLUS_UINT32
    GrB_MAX_PLUS_SEMIRING_UINT64,       // GxB_MAX_PLUS_UINT64
    GrB_MAX_PLUS_SEMIRING_FP32,         // GxB_MAX_PLUS_FP32
    GrB_MAX_PLUS_SEMIRING_FP64,         // GxB_MAX_PLUS_FP64

    // MAX_TIMES semirings for all 10 real, non-boolean types:
    GrB_MAX_TIMES_SEMIRING_INT8,        // GxB_MAX_TIMES_INT8
    GrB_MAX_TIMES_SEMIRING_INT16,       // GxB_MAX_TIMES_INT16
    GrB_MAX_TIMES_SEMIRING_INT32,       // GxB_MAX_TIMES_INT32
    GrB_MAX_TIMES_SEMIRING_INT64,       // GxB_MAX_TIMES_INT64
    GrB_MAX_TIMES_SEMIRING_UINT8,       // GxB_MAX_TIMES_UINT8
    GrB_MAX_TIMES_SEMIRING_UINT16,      // GxB_MAX_TIMES_UINT16
    GrB_MAX_TIMES_SEMIRING_UINT32,      // GxB_MAX_TIMES_UINT32
    GrB_MAX_TIMES_SEMIRING_UINT64,      // GxB_MAX_TIMES_UINT64
    GrB_MAX_TIMES_SEMIRING_FP32,        // GxB_MAX_TIMES_FP32
    GrB_MAX_TIMES_SEMIRING_FP64,        // GxB_MAX_TIMES_FP64

    // MAX_FIRST semirings for all 10 real, non-boolean types:
    GrB_MAX_FIRST_SEMIRING_INT8,        // GxB_MAX_FIRST_INT8
    GrB_MAX_FIRST_SEMIRING_INT16,       // GxB_MAX_FIRST_INT16
    GrB_MAX_FIRST_SEMIRING_INT32,       // GxB_MAX_FIRST_INT32
    GrB_MAX_FIRST_SEMIRING_INT64,       // GxB_MAX_FIRST_INT64
    GrB_MAX_FIRST_SEMIRING_UINT8,       // GxB_MAX_FIRST_UINT8
    GrB_MAX_FIRST_SEMIRING_UINT16,      // GxB_MAX_FIRST_UINT16
    GrB_MAX_FIRST_SEMIRING_UINT32,      // GxB_MAX_FIRST_UINT32
    GrB_MAX_FIRST_SEMIRING_UINT64,      // GxB_MAX_FIRST_UINT64
    GrB_MAX_FIRST_SEMIRING_FP32,        // GxB_MAX_FIRST_FP32
    GrB_MAX_FIRST_SEMIRING_FP64,        // GxB_MAX_FIRST_FP64

    // MAX_SECOND semirings for all 10 real, non-boolean types:
    GrB_MAX_SECOND_SEMIRING_INT8,       // GxB_MAX_SECOND_INT8
    GrB_MAX_SECOND_SEMIRING_INT16,      // GxB_MAX_SECOND_INT16
    GrB_MAX_SECOND_SEMIRING_INT32,      // GxB_MAX_SECOND_INT32
    GrB_MAX_SECOND_SEMIRING_INT64,      // GxB_MAX_SECOND_INT64
    GrB_MAX_SECOND_SEMIRING_UINT8,      // GxB_MAX_SECOND_UINT8
    GrB_MAX_SECOND_SEMIRING_UINT16,     // GxB_MAX_SECOND_UINT16
    GrB_MAX_SECOND_SEMIRING_UINT32,     // GxB_MAX_SECOND_UINT32
    GrB_MAX_SECOND_SEMIRING_UINT64,     // GxB_MAX_SECOND_UINT64
    GrB_MAX_SECOND_SEMIRING_FP32,       // GxB_MAX_SECOND_FP32
    GrB_MAX_SECOND_SEMIRING_FP64,       // GxB_MAX_SECOND_FP64

    // MAX_MIN semirings for all 10 real, non-boolean types:
    GrB_MAX_MIN_SEMIRING_INT8,          // GxB_MAX_MIN_INT8
    GrB_MAX_MIN_SEMIRING_INT16,         // GxB_MAX_MIN_INT16
    GrB_MAX_MIN_SEMIRING_INT32,         // GxB_MAX_MIN_INT32
    GrB_MAX_MIN_SEMIRING_INT64,         // GxB_MAX_MIN_INT64
    GrB_MAX_MIN_SEMIRING_UINT8,         // GxB_MAX_MIN_UINT8
    GrB_MAX_MIN_SEMIRING_UINT16,        // GxB_MAX_MIN_UINT16
    GrB_MAX_MIN_SEMIRING_UINT32,        // GxB_MAX_MIN_UINT32
    GrB_MAX_MIN_SEMIRING_UINT64,        // GxB_MAX_MIN_UINT64
    GrB_MAX_MIN_SEMIRING_FP32,          // GxB_MAX_MIN_FP32
    GrB_MAX_MIN_SEMIRING_FP64,          // GxB_MAX_MIN_FP64

    //--------------------------------------------------------------------------
    // 4 boolean semirings:
    //--------------------------------------------------------------------------

    GrB_LOR_LAND_SEMIRING_BOOL,         // GxB_LOR_LAND_BOOL
    GrB_LAND_LOR_SEMIRING_BOOL,         // GxB_LAND_LOR_BOOL
    GrB_LXOR_LAND_SEMIRING_BOOL,        // GxB_LXOR_LAND_BOOL
    GrB_LXNOR_LOR_SEMIRING_BOOL ;       // GxB_EQ_LOR_BOOL (note EQ == LXNOR)

//==============================================================================
// GxB_fprint and GxB_print: print the contents of a GraphBLAS object
//==============================================================================

typedef enum
{
    GxB_SILENT = 0,     // nothing is printed, just check the object
    GxB_SUMMARY = 1,    // print a terse summary
    GxB_SHORT = 2,      // short description, about 30 entries of a matrix
    GxB_COMPLETE = 3,   // print the entire contents of the object
    GxB_SHORT_VERBOSE = 4,    // GxB_SHORT but with "%.15g" for doubles
    GxB_COMPLETE_VERBOSE = 5  // GxB_COMPLETE but with "%.15g" for doubles
}
GxB_Print_Level ;

//==============================================================================
// GrB import/export
//==============================================================================

// The GrB C API specification supports 3 formats:
typedef enum
{
    GrB_CSR_FORMAT = 0,     // CSR format (equiv to GxB_SPARSE with GxB_BY_ROW)
    GrB_CSC_FORMAT = 1,     // CSC format (equiv to GxB_SPARSE with GxB_BY_COL)
    GrB_COO_FORMAT = 2      // triplet format (like input to GrB*build)
}
GrB_Format ;

//==============================================================================
// serialize/deserialize compression levels
//==============================================================================

// Currently implemented: no compression, LZ4, LZ4HC, and ZSTD
#define GxB_COMPRESSION_NONE -1     // no compression
#define GxB_COMPRESSION_DEFAULT 0   // ZSTD (level 1)
#define GxB_COMPRESSION_LZ4   1000  // LZ4
#define GxB_COMPRESSION_LZ4HC 2000  // LZ4HC, with default level 9
#define GxB_COMPRESSION_ZSTD  3000  // ZSTD, with default level 1

//==============================================================================
//=== GraphBLAS functions ======================================================
//==============================================================================

// All user-callable GrB* and GxB* functions in GraphBLAS are defined below.
// The GB_CUDA_FOLDER flag is only meant for use by the C++ functions in
// GraphBLAS/CUDA, since they do not need access these definitions.  User
// applications have access to these methods.

#if defined ( __cplusplus )
extern "C"
{
#endif

#ifndef GB_CUDA_FOLDER

//==============================================================================
// GrB_init / GrB_finalize
//==============================================================================

GrB_Info GrB_init           // start up GraphBLAS
(
    GrB_Mode mode           // blocking or non-blocking mode, no GPU
) ;

GrB_Info GxB_init           // start up GraphBLAS and also define malloc, etc
(
    GrB_Mode mode,          // blocking or non-blocking mode,
                            // with or without GPU
    // pointers to memory management functions
    void * (* user_malloc_function  ) (size_t),
    void * (* user_calloc_function  ) (size_t, size_t),
    void * (* user_realloc_function ) (void *, size_t),
    void   (* user_free_function    ) (void *)
) ;

GrB_Info GrB_finalize (void) ;     // finish GraphBLAS

//==============================================================================
// GrB_getVersion: GraphBLAS C API version
//==============================================================================

// GrB_getVersion provides a runtime access of the C API Version.  Can also be
// done with two calls to GrB_Global_get_INT32 with the v2.1 C API:
//
//      GrB_get (GrB_GLOBAL, &version,    GrB_API_VER_MAJOR) ;
//      GrB_get (GrB_GLOBAL, &subversion, GrB_API_VER_MINOR) ;

GrB_Info GrB_getVersion         // runtime access to C API version number
(
    unsigned int *version,      // returns GRB_VERSION
    unsigned int *subversion    // returns GRB_SUBVERSION
) ;

//==============================================================================
// GrB_Descriptor: the GraphBLAS descriptor
//==============================================================================

GrB_Info GrB_Descriptor_new     // create a new descriptor
(
    GrB_Descriptor *descriptor  // handle of descriptor to create
) ;

// historical methods; use GrB_set and GrB_get instead:
GrB_Info GrB_Descriptor_set (GrB_Descriptor, GrB_Desc_Field, GrB_Desc_Value) ;
GrB_Info GxB_Descriptor_get (GrB_Desc_Value *, GrB_Descriptor, GrB_Desc_Field) ;
GrB_Info GxB_Desc_set       (GrB_Descriptor, GrB_Desc_Field, ...) ;
GrB_Info GxB_Desc_set_INT32 (GrB_Descriptor, GrB_Desc_Field, int32_t) ;
GrB_Info GxB_Desc_set_FP64  (GrB_Descriptor, GrB_Desc_Field, double) ;
GrB_Info GxB_Desc_get       (GrB_Descriptor, GrB_Desc_Field, ...) ;
GrB_Info GxB_Desc_get_INT32 (GrB_Descriptor, GrB_Desc_Field, int32_t *) ;
GrB_Info GxB_Desc_get_FP64  (GrB_Descriptor, GrB_Desc_Field, double *) ;

GrB_Info GrB_Descriptor_free    // free a descriptor
(
    GrB_Descriptor *descriptor  // handle of descriptor to free
) ;

//==============================================================================
// helper macros for polymorphic functions
//==============================================================================

#define GB_CAT(w,x,y,z) w ## x ## y ## z
#define GB_CONCAT(w,x,y,z) GB_CAT (w, x, y, z)

// methods for C scalars of various types
#define GB_CASES(prefix,func)                                           \
              bool         : GB_CONCAT ( prefix, _, func, _BOOL   ),    \
              int8_t       : GB_CONCAT ( prefix, _, func, _INT8   ),    \
              int16_t      : GB_CONCAT ( prefix, _, func, _INT16  ),    \
              int32_t      : GB_CONCAT ( prefix, _, func, _INT32  ),    \
              int64_t      : GB_CONCAT ( prefix, _, func, _INT64  ),    \
              uint8_t      : GB_CONCAT ( prefix, _, func, _UINT8  ),    \
              uint16_t     : GB_CONCAT ( prefix, _, func, _UINT16 ),    \
              uint32_t     : GB_CONCAT ( prefix, _, func, _UINT32 ),    \
              uint64_t     : GB_CONCAT ( prefix, _, func, _UINT64 ),    \
              float        : GB_CONCAT ( prefix, _, func, _FP32   ),    \
              double       : GB_CONCAT ( prefix, _, func, _FP64   ),    \
              GxB_FC32_t   : GB_CONCAT ( GxB   , _, func, _FC32   ),    \
              GxB_FC64_t   : GB_CONCAT ( GxB   , _, func, _FC64   ),    \
        const void       * : GB_CONCAT ( prefix, _, func, _UDT    ),    \
              void       * : GB_CONCAT ( prefix, _, func, _UDT    )

// methods for C arrays of various types
#define GB_PCASES(prefix,func)                                          \
        const bool       * : GB_CONCAT ( prefix, _, func, _BOOL   ),    \
              bool       * : GB_CONCAT ( prefix, _, func, _BOOL   ),    \
        const int8_t     * : GB_CONCAT ( prefix, _, func, _INT8   ),    \
              int8_t     * : GB_CONCAT ( prefix, _, func, _INT8   ),    \
        const int16_t    * : GB_CONCAT ( prefix, _, func, _INT16  ),    \
              int16_t    * : GB_CONCAT ( prefix, _, func, _INT16  ),    \
        const int32_t    * : GB_CONCAT ( prefix, _, func, _INT32  ),    \
              int32_t    * : GB_CONCAT ( prefix, _, func, _INT32  ),    \
        const int64_t    * : GB_CONCAT ( prefix, _, func, _INT64  ),    \
              int64_t    * : GB_CONCAT ( prefix, _, func, _INT64  ),    \
        const uint8_t    * : GB_CONCAT ( prefix, _, func, _UINT8  ),    \
              uint8_t    * : GB_CONCAT ( prefix, _, func, _UINT8  ),    \
        const uint16_t   * : GB_CONCAT ( prefix, _, func, _UINT16 ),    \
              uint16_t   * : GB_CONCAT ( prefix, _, func, _UINT16 ),    \
        const uint32_t   * : GB_CONCAT ( prefix, _, func, _UINT32 ),    \
              uint32_t   * : GB_CONCAT ( prefix, _, func, _UINT32 ),    \
        const uint64_t   * : GB_CONCAT ( prefix, _, func, _UINT64 ),    \
              uint64_t   * : GB_CONCAT ( prefix, _, func, _UINT64 ),    \
        const float      * : GB_CONCAT ( prefix, _, func, _FP32   ),    \
              float      * : GB_CONCAT ( prefix, _, func, _FP32   ),    \
        const double     * : GB_CONCAT ( prefix, _, func, _FP64   ),    \
              double     * : GB_CONCAT ( prefix, _, func, _FP64   ),    \
        const GxB_FC32_t * : GB_CONCAT ( GxB   , _, func, _FC32   ),    \
              GxB_FC32_t * : GB_CONCAT ( GxB   , _, func, _FC32   ),    \
        const GxB_FC64_t * : GB_CONCAT ( GxB   , _, func, _FC64   ),    \
              GxB_FC64_t * : GB_CONCAT ( GxB   , _, func, _FC64   ),    \
        const void       * : GB_CONCAT ( prefix, _, func, _UDT    ),    \
              void       * : GB_CONCAT ( prefix, _, func, _UDT    )

//==============================================================================
// GrB_Type: data types
//==============================================================================

//------------------------------------------------------------------------------
// GrB_Type_new:  create a new type
//------------------------------------------------------------------------------

GrB_Info GrB_Type_new           // create a new GraphBLAS type
(
    GrB_Type *type,             // handle of user type to create
    size_t sizeof_ctype         // size = sizeof (ctype) of the C type
) ;

// GxB_Type_new creates a type with a name and definition that are known to
// GraphBLAS, as strings.  The type_name is any valid string (max length of 128
// characters, including the required null-terminating character) that may
// appear as the name of a C type created by a C "typedef" statement.  It must
// not contain any white-space characters.  Example, creating a type of size
// 16*4+4 = 68 bytes, with a 4-by-4 dense float array and a 32-bit integer:
//
//      typedef struct { float x [4][4] ; int color ; } myquaternion ;
//      GrB_Type MyQtype ;
//      GxB_Type_new (&MyQtype, sizeof (myquaternion), "myquaternion",
//          "typedef struct { float x [4][4] ; int color ; } myquaternion ;") ;
//
// The type_name and type_defn are both null-terminated strings.  The two
// strings are required for best performance in the JIT (both on the CPU and
// GPU).  User defined types created by GrB_Type_new will not work with a JIT.
//
// At most GxB_MAX_NAME_LEN characters are accessed in type_name; characters
// beyond that limit are silently ignored.

GrB_Info GxB_Type_new           // create a new named GraphBLAS type
(
    GrB_Type *type,             // handle of user type to create
    size_t sizeof_ctype,        // size = sizeof (ctype) of the C type
    const char *type_name,      // name of the type (max 128 characters)
    const char *type_defn       // typedef for the type (no max length)
) ;

// historical; use GrB_get instead:
GrB_Info GxB_Type_name (char *type_name, const GrB_Type type) ;
GrB_Info GxB_Type_size (size_t *size, const GrB_Type type) ;

GrB_Info GxB_Type_from_name     // return the built-in GrB_Type from a name
(
    GrB_Type *type,             // built-in type, or NULL if user-defined.
                                // The name can be the underlying C type
                                // ("int8_t") or the GraphBLAS name
                                // ("GrB_INT8").
    const char *type_name       // array of size at least GxB_MAX_NAME_LEN
) ;

GrB_Info GrB_Type_free          // free a user-defined type
(
    GrB_Type *type              // handle of user-defined type to free
) ;

//==============================================================================
// GrB_UnaryOp: unary operators
//==============================================================================

// GrB_UnaryOp_new creates a user-defined unary op (with no name or defn)
GrB_Info GrB_UnaryOp_new            // create a new user-defined unary operator
(
    GrB_UnaryOp *unaryop,           // handle for the new unary operator
    GxB_unary_function function,    // pointer to the unary function
    GrB_Type ztype,                 // type of output z
    GrB_Type xtype                  // type of input x
) ;

// GxB_UnaryOp_new creates a named and defined user-defined unary op.
GrB_Info GxB_UnaryOp_new            // create a new user-defined unary operator
(
    GrB_UnaryOp *unaryop,           // handle for the new unary operator
    GxB_unary_function function,    // pointer to the unary function
    GrB_Type ztype,                 // type of output z
    GrB_Type xtype,                 // type of input x
    const char *unop_name,          // name of the user function
    const char *unop_defn           // definition of the user function
) ;

// historical; use GrB_get instead:
GrB_Info GxB_UnaryOp_ztype (GrB_Type *ztype, GrB_UnaryOp unaryop) ;
GrB_Info GxB_UnaryOp_ztype_name (char *type_name, const GrB_UnaryOp unaryop) ;
GrB_Info GxB_UnaryOp_xtype (GrB_Type *xtype, GrB_UnaryOp unaryop) ;
GrB_Info GxB_UnaryOp_xtype_name (char *type_name, const GrB_UnaryOp unaryop) ;

GrB_Info GrB_UnaryOp_free           // free a user-created unary operator
(
    GrB_UnaryOp *unaryop            // handle of unary operator to free
) ;

//==============================================================================
// GrB_BinaryOp: binary operators
//==============================================================================

// GrB_BinaryOp_new creates a user-defined binary op (no name or defn)
GrB_Info GrB_BinaryOp_new
(
    GrB_BinaryOp *binaryop,         // handle for the new binary operator
    GxB_binary_function function,   // pointer to the binary function
    GrB_Type ztype,                 // type of output z
    GrB_Type xtype,                 // type of input x
    GrB_Type ytype                  // type of input y
) ;

// GxB_BinaryOp_new creates a named and defined user-defined binary op.
GrB_Info GxB_BinaryOp_new
(
    GrB_BinaryOp *op,               // handle for the new binary operator
    GxB_binary_function function,   // pointer to the binary function
    GrB_Type ztype,                 // type of output z
    GrB_Type xtype,                 // type of input x
    GrB_Type ytype,                 // type of input y
    const char *binop_name,         // name of the user function
    const char *binop_defn          // definition of the user function
) ;

// historical; use GrB_get instead:
GrB_Info GxB_BinaryOp_ztype (GrB_Type *ztype, GrB_BinaryOp op) ;
GrB_Info GxB_BinaryOp_ztype_name (char *type_name, const GrB_BinaryOp op) ;
GrB_Info GxB_BinaryOp_xtype (GrB_Type *xtype, GrB_BinaryOp op) ;
GrB_Info GxB_BinaryOp_xtype_name (char *type_name, const GrB_BinaryOp op) ;
GrB_Info GxB_BinaryOp_ytype (GrB_Type *ytype, GrB_BinaryOp op) ;
GrB_Info GxB_BinaryOp_ytype_name (char *type_name, const GrB_BinaryOp op) ;

GrB_Info GrB_BinaryOp_free          // free a user-created binary operator
(
    GrB_BinaryOp *binaryop          // handle of binary operator to free
) ;

//==============================================================================
// GxB_SelectOp: select operators (DEPRECATED: do not use)
//==============================================================================

// historical; use GrB_select and GrB_IndexUnaryOp instead:
GrB_Info GxB_SelectOp_xtype (GrB_Type *xtype, GxB_SelectOp selectop) ;
GrB_Info GxB_SelectOp_ttype (GrB_Type *ttype, GxB_SelectOp selectop) ;

//==============================================================================
// GrB_IndexUnaryOp: a unary operator that depends on the row/col indices
//==============================================================================

// GrB_IndexUnaryOp_new creates a user-defined unary op (no name or defn)

GrB_Info GrB_IndexUnaryOp_new       // create a new user-defined IndexUnary op
(
    GrB_IndexUnaryOp *op,           // handle for the new IndexUnary operator
    GxB_index_unary_function function,    // pointer to IndexUnary function
    GrB_Type ztype,                 // type of output z
    GrB_Type xtype,                 // type of input x (the A(i,j) entry)
    GrB_Type ytype                  // type of input y (the scalar)
) ;

GrB_Info GxB_IndexUnaryOp_new   // create a named user-created IndexUnaryOp
(
    GrB_IndexUnaryOp *op,           // handle for the new IndexUnary operator
    GxB_index_unary_function function,    // pointer to index_unary function
    GrB_Type ztype,                 // type of output z
    GrB_Type xtype,                 // type of input x (the A(i,j) entry)
    GrB_Type ytype,                 // type of input y (the scalar)
    const char *idxop_name,         // name of the user function
    const char *idxop_defn          // definition of the user function
) ;

// historical; use GrB_get instead:
GrB_Info GxB_IndexUnaryOp_ztype_name (char *, const GrB_IndexUnaryOp op) ;
GrB_Info GxB_IndexUnaryOp_xtype_name (char *, const GrB_IndexUnaryOp op) ;
GrB_Info GxB_IndexUnaryOp_ytype_name (char *, const GrB_IndexUnaryOp op) ;

GrB_Info GrB_IndexUnaryOp_free  // free a user-created IndexUnaryOp
(
    GrB_IndexUnaryOp *op        // handle of IndexUnary to free
) ;

//==============================================================================
// GrB_Monoid
//==============================================================================

GrB_Info GrB_Monoid_new_BOOL        // create a new boolean monoid
(
    GrB_Monoid *monoid,             // handle of monoid to create
    GrB_BinaryOp op,                // binary operator of the monoid
    bool identity                   // identity value of the monoid
) ;

GrB_Info GrB_Monoid_new_INT8        // create a new int8 monoid
(
    GrB_Monoid *monoid,             // handle of monoid to create
    GrB_BinaryOp op,                // binary operator of the monoid
    int8_t identity                 // identity value of the monoid
) ;

GrB_Info GrB_Monoid_new_UINT8       // create a new uint8 monoid
(
    GrB_Monoid *monoid,             // handle of monoid to create
    GrB_BinaryOp op,                // binary operator of the monoid
    uint8_t identity                // identity value of the monoid
) ;

GrB_Info GrB_Monoid_new_INT16       // create a new int16 monoid
(
    GrB_Monoid *monoid,             // handle of monoid to create
    GrB_BinaryOp op,                // binary operator of the monoid
    int16_t identity                // identity value of the monoid
) ;

GrB_Info GrB_Monoid_new_UINT16      // create a new uint16 monoid
(
    GrB_Monoid *monoid,             // handle of monoid to create
    GrB_BinaryOp op,                // binary operator of the monoid
    uint16_t identity               // identity value of the monoid
) ;

GrB_Info GrB_Monoid_new_INT32       // create a new int32 monoid
(
    GrB_Monoid *monoid,             // handle of monoid to create
    GrB_BinaryOp op,                // binary operator of the monoid
    int32_t identity                // identity value of the monoid
) ;

GrB_Info GrB_Monoid_new_UINT32      // create a new uint32 monoid
(
    GrB_Monoid *monoid,             // handle of monoid to create
    GrB_BinaryOp op,                // binary operator of the monoid
    uint32_t identity               // identity value of the monoid
) ;

GrB_Info GrB_Monoid_new_INT64       // create a new int64 monoid
(
    GrB_Monoid *monoid,             // handle of monoid to create
    GrB_BinaryOp op,                // binary operator of the monoid
    int64_t identity                // identity value of the monoid
) ;

GrB_Info GrB_Monoid_new_UINT64      // create a new uint64 monoid
(
    GrB_Monoid *monoid,             // handle of monoid to create
    GrB_BinaryOp op,                // binary operator of the monoid
    uint64_t identity               // identity value of the monoid
) ;

GrB_Info GrB_Monoid_new_FP32        // create a new float monoid
(
    GrB_Monoid *monoid,             // handle of monoid to create
    GrB_BinaryOp op,                // binary operator of the monoid
    float identity                  // identity value of the monoid
) ;

GrB_Info GrB_Monoid_new_FP64        // create a new double monoid
(
    GrB_Monoid *monoid,             // handle of monoid to create
    GrB_BinaryOp op,                // binary operator of the monoid
    double identity                 // identity value of the monoid
) ;

GrB_Info GxB_Monoid_new_FC32        // create a new float complex monoid
(
    GrB_Monoid *monoid,             // handle of monoid to create
    GrB_BinaryOp op,                // binary operator of the monoid
    GxB_FC32_t identity             // identity value of the monoid
) ;

GrB_Info GxB_Monoid_new_FC64        // create a new double complex monoid
(
    GrB_Monoid *monoid,             // handle of monoid to create
    GrB_BinaryOp op,                // binary operator of the monoid
    GxB_FC64_t identity             // identity value of the monoid
) ;

GrB_Info GrB_Monoid_new_UDT         // create a monoid with a user-defined type
(
    GrB_Monoid *monoid,             // handle of monoid to create
    GrB_BinaryOp op,                // binary operator of the monoid
    void *identity                  // identity value of the monoid
) ;

// Type-generic method for creating a new monoid:

/*
GrB_Info GrB_Monoid_new             // create a monoid
(
    GrB_Monoid *monoid,             // handle of monoid to create
    GrB_BinaryOp op,          // binary operator of the monoid
    <type> identity           // identity value of the monoid
) ;
*/

#if GxB_STDC_VERSION >= 201112L
#define GrB_Monoid_new(monoid,op,identity)      \
    _Generic                                    \
    (                                           \
        (identity),                             \
            GB_CASES (GrB, Monoid_new)          \
    )                                           \
    (monoid, op, identity)
#endif

// GxB_Monoid_terminal_new is identical to GrB_Monoid_new, except that a
// terminal value can be specified.  The terminal may be NULL, which indicates
// no terminal value (and in this case, it is identical to GrB_Monoid_new).
// The terminal value, if not NULL, must have the same type as the identity.

GrB_Info GxB_Monoid_terminal_new_BOOL        // create a new boolean monoid
(
    GrB_Monoid *monoid,             // handle of monoid to create
    GrB_BinaryOp op,                // binary operator of the monoid
    bool identity,                  // identity value of the monoid
    bool terminal                   // terminal value of the monoid
) ;

GrB_Info GxB_Monoid_terminal_new_INT8        // create a new int8 monoid
(
    GrB_Monoid *monoid,             // handle of monoid to create
    GrB_BinaryOp op,                // binary operator of the monoid
    int8_t identity,                // identity value of the monoid
    int8_t terminal                 // terminal value of the monoid
) ;

GrB_Info GxB_Monoid_terminal_new_UINT8       // create a new uint8 monoid
(
    GrB_Monoid *monoid,             // handle of monoid to create
    GrB_BinaryOp op,                // binary operator of the monoid
    uint8_t identity,               // identity value of the monoid
    uint8_t terminal                // terminal value of the monoid
) ;

GrB_Info GxB_Monoid_terminal_new_INT16       // create a new int16 monoid
(
    GrB_Monoid *monoid,             // handle of monoid to create
    GrB_BinaryOp op,                // binary operator of the monoid
    int16_t identity,               // identity value of the monoid
    int16_t terminal                // terminal value of the monoid
) ;

GrB_Info GxB_Monoid_terminal_new_UINT16      // create a new uint16 monoid
(
    GrB_Monoid *monoid,             // handle of monoid to create
    GrB_BinaryOp op,                // binary operator of the monoid
    uint16_t identity,              // identity value of the monoid
    uint16_t terminal               // terminal value of the monoid
) ;

GrB_Info GxB_Monoid_terminal_new_INT32       // create a new int32 monoid
(
    GrB_Monoid *monoid,             // handle of monoid to create
    GrB_BinaryOp op,                // binary operator of the monoid
    int32_t identity,               // identity value of the monoid
    int32_t terminal                // terminal value of the monoid
) ;

GrB_Info GxB_Monoid_terminal_new_UINT32      // create a new uint32 monoid
(
    GrB_Monoid *monoid,             // handle of monoid to create
    GrB_BinaryOp op,                // binary operator of the monoid
    uint32_t identity,              // identity value of the monoid
    uint32_t terminal               // terminal value of the monoid
) ;

GrB_Info GxB_Monoid_terminal_new_INT64       // create a new int64 monoid
(
    GrB_Monoid *monoid,             // handle of monoid to create
    GrB_BinaryOp op,                // binary operator of the monoid
    int64_t identity,               // identity value of the monoid
    int64_t terminal                // terminal value of the monoid
) ;

GrB_Info GxB_Monoid_terminal_new_UINT64      // create a new uint64 monoid
(
    GrB_Monoid *monoid,             // handle of monoid to create
    GrB_BinaryOp op,                // binary operator of the monoid
    uint64_t identity,              // identity value of the monoid
    uint64_t terminal               // terminal value of the monoid
) ;

GrB_Info GxB_Monoid_terminal_new_FP32        // create a new float monoid
(
    GrB_Monoid *monoid,             // handle of monoid to create
    GrB_BinaryOp op,                // binary operator of the monoid
    float identity,                 // identity value of the monoid
    float terminal                  // terminal value of the monoid
) ;

GrB_Info GxB_Monoid_terminal_new_FP64        // create a new double monoid
(
    GrB_Monoid *monoid,             // handle of monoid to create
    GrB_BinaryOp op,                // binary operator of the monoid
    double identity,                // identity value of the monoid
    double terminal                 // terminal value of the monoid
) ;

GrB_Info GxB_Monoid_terminal_new_FC32   // create a new float complex monoid
(
    GrB_Monoid *monoid,             // handle of monoid to create
    GrB_BinaryOp op,                // binary operator of the monoid
    GxB_FC32_t identity,            // identity value of the monoid
    GxB_FC32_t terminal             // terminal value of the monoid
) ;

GrB_Info GxB_Monoid_terminal_new_FC64   // create a new double complex monoid
(
    GrB_Monoid *monoid,             // handle of monoid to create
    GrB_BinaryOp op,                // binary operator of the monoid
    GxB_FC64_t identity,            // identity value of the monoid
    GxB_FC64_t terminal             // terminal value of the monoid
) ;

GrB_Info GxB_Monoid_terminal_new_UDT    // create a monoid with a user type
(
    GrB_Monoid *monoid,             // handle of monoid to create
    GrB_BinaryOp op,                // binary operator of the monoid
    void *identity,                 // identity value of the monoid
    void *terminal                  // terminal value of the monoid
) ;

// Type-generic method for creating a new monoid with a terminal value:

/*
GrB_Info GxB_Monoid_terminal_new             // create a monoid
(
    GrB_Monoid *monoid,             // handle of monoid to create
    GrB_BinaryOp op,                // binary operator of the monoid
    <type> identity,                // identity value of the monoid
    <type> terminal                 // terminal value of the monoid
) ;
*/

#if GxB_STDC_VERSION >= 201112L
#define GxB_Monoid_terminal_new(monoid,op,identity,terminal)    \
    _Generic                                                    \
    (                                                           \
        (identity),                                             \
            GB_CASES (GxB, Monoid_terminal_new)                 \
    )                                                           \
    (monoid, op, identity, terminal)
#endif

// historical; use GrB_get instead:
GrB_Info GxB_Monoid_operator (GrB_BinaryOp *op, GrB_Monoid monoid) ;
GrB_Info GxB_Monoid_identity (void *identity, GrB_Monoid monoid) ;
GrB_Info GxB_Monoid_terminal (bool *, void *terminal, GrB_Monoid monoid) ;

GrB_Info GrB_Monoid_free            // free a user-created monoid
(
    GrB_Monoid *monoid              // handle of monoid to free
) ;

//==============================================================================
// GrB_Semiring
//==============================================================================

GrB_Info GrB_Semiring_new           // create a semiring
(
    GrB_Semiring *semiring,         // handle of semiring to create
    GrB_Monoid add,                 // add monoid of the semiring
    GrB_BinaryOp multiply           // multiply operator of the semiring
) ;

// historical; use GrB_get instead:
GrB_Info GxB_Semiring_add (GrB_Monoid *add, GrB_Semiring semiring) ;
GrB_Info GxB_Semiring_multiply (GrB_BinaryOp *multiply, GrB_Semiring semiring) ;

GrB_Info GrB_Semiring_free          // free a user-created semiring
(
    GrB_Semiring *semiring          // handle of semiring to free
) ;

//==============================================================================
// GrB_Scalar: a GraphBLAS scalar
//==============================================================================

// These methods create, free, copy, and clear a GrB_Scalar.  The nvals,
// and type methods return basic information about a GrB_Scalar.

GrB_Info GrB_Scalar_new     // create a new GrB_Scalar with no entry
(
    GrB_Scalar *s,          // handle of GrB_Scalar to create
    GrB_Type type           // type of GrB_Scalar to create
) ;

GrB_Info GrB_Scalar_dup     // make an exact copy of a GrB_Scalar
(
    GrB_Scalar *s,          // handle of output GrB_Scalar to create
    const GrB_Scalar t      // input GrB_Scalar to copy
) ;

GrB_Info GrB_Scalar_clear   // clear a GrB_Scalar of its entry
(                           // type remains unchanged.
    GrB_Scalar s            // GrB_Scalar to clear
) ;

GrB_Info GrB_Scalar_nvals   // get the number of entries in a GrB_Scalar
(
    GrB_Index *nvals,       // GrB_Scalar has nvals entries (0 or 1)
    const GrB_Scalar s      // GrB_Scalar to query
) ;

// historical; use GrB_get instead:
GrB_Info GxB_Scalar_type (GrB_Type *type, const GrB_Scalar s) ;
GrB_Info GxB_Scalar_type_name (char *type_name, const GrB_Scalar s) ;

GrB_Info GxB_Scalar_memoryUsage  // return # of bytes used for a scalar
(
    size_t *size,           // # of bytes used by the scalar s
    const GrB_Scalar s      // GrB_Scalar to query
) ;

GrB_Info GrB_Scalar_free    // free a GrB_Scalar
(
    GrB_Scalar *s           // handle of GrB_Scalar to free
) ;

// historical names identical to GrB_Scalar_methods above:
GrB_Info GxB_Scalar_new   (GrB_Scalar *s, GrB_Type type) ;
GrB_Info GxB_Scalar_dup   (GrB_Scalar *s, const GrB_Scalar t) ;
GrB_Info GxB_Scalar_clear (GrB_Scalar s) ;
GrB_Info GxB_Scalar_nvals (GrB_Index *nvals, const GrB_Scalar s) ;
GrB_Info GxB_Scalar_free  (GrB_Scalar *s) ;

//------------------------------------------------------------------------------
// GrB_Scalar_setElement
//------------------------------------------------------------------------------

// Set a single GrB_Scalar s, from a user scalar x: s = x, typecasting from the
// type of x to the type of w as needed.

GrB_Info GrB_Scalar_setElement_BOOL     // s = x
(
    GrB_Scalar s,                       // GrB_Scalar to modify
    bool x                              // user scalar to assign to s
) ;

GrB_Info GrB_Scalar_setElement_INT8     // s = x
(
    GrB_Scalar s,                       // GrB_Scalar to modify
    int8_t x                            // user scalar to assign to s
) ;

GrB_Info GrB_Scalar_setElement_UINT8    // s = x
(
    GrB_Scalar s,                       // GrB_Scalar to modify
    uint8_t x                           // user scalar to assign to s
) ;

GrB_Info GrB_Scalar_setElement_INT16    // s = x
(
    GrB_Scalar s,                       // GrB_Scalar to modify
    int16_t x                           // user scalar to assign to s
) ;

GrB_Info GrB_Scalar_setElement_UINT16   // s = x
(
    GrB_Scalar s,                       // GrB_Scalar to modify
    uint16_t x                          // user scalar to assign to s
) ;

GrB_Info GrB_Scalar_setElement_INT32    // s = x
(
    GrB_Scalar s,                       // GrB_Scalar to modify
    int32_t x                           // user scalar to assign to s
) ;

GrB_Info GrB_Scalar_setElement_UINT32   // s = x
(
    GrB_Scalar s,                       // GrB_Scalar to modify
    uint32_t x                          // user scalar to assign to s
) ;

GrB_Info GrB_Scalar_setElement_INT64    // s = x
(
    GrB_Scalar s,                       // GrB_Scalar to modify
    int64_t x                           // user scalar to assign to s
) ;

GrB_Info GrB_Scalar_setElement_UINT64   // s = x
(
    GrB_Scalar s,                       // GrB_Scalar to modify
    uint64_t x                          // user scalar to assign to s
) ;

GrB_Info GrB_Scalar_setElement_FP32     // s = x
(
    GrB_Scalar s,                       // GrB_Scalar to modify
    float x                             // user scalar to assign to s
) ;

GrB_Info GrB_Scalar_setElement_FP64     // s = x
(
    GrB_Scalar s,                       // GrB_Scalar to modify
    double x                            // user scalar to assign to s
) ;

GrB_Info GxB_Scalar_setElement_FC32     // s = x
(
    GrB_Scalar s,                       // GrB_Scalar to modify
    GxB_FC32_t x                        // user scalar to assign to s
) ;

GrB_Info GxB_Scalar_setElement_FC64     // s = x
(
    GrB_Scalar s,                       // GrB_Scalar to modify
    GxB_FC64_t x                        // user scalar to assign to s
) ;

GrB_Info GrB_Scalar_setElement_UDT      // s = x
(
    GrB_Scalar s,                       // GrB_Scalar to modify
    void *x                             // user scalar to assign to s
) ;

// historical names identical to GrB_Scalar_methods above:
GrB_Info GxB_Scalar_setElement_BOOL   (GrB_Scalar s, bool     x) ;
GrB_Info GxB_Scalar_setElement_INT8   (GrB_Scalar s, int8_t   x) ;
GrB_Info GxB_Scalar_setElement_INT16  (GrB_Scalar s, int16_t  x) ;
GrB_Info GxB_Scalar_setElement_INT32  (GrB_Scalar s, int32_t  x) ;
GrB_Info GxB_Scalar_setElement_INT64  (GrB_Scalar s, int64_t  x) ;
GrB_Info GxB_Scalar_setElement_UINT8  (GrB_Scalar s, uint8_t  x) ;
GrB_Info GxB_Scalar_setElement_UINT16 (GrB_Scalar s, uint16_t x) ;
GrB_Info GxB_Scalar_setElement_UINT32 (GrB_Scalar s, uint32_t x) ;
GrB_Info GxB_Scalar_setElement_UINT64 (GrB_Scalar s, uint64_t x) ;
GrB_Info GxB_Scalar_setElement_FP32   (GrB_Scalar s, float    x) ;
GrB_Info GxB_Scalar_setElement_FP64   (GrB_Scalar s, double   x) ;
GrB_Info GxB_Scalar_setElement_UDT    (GrB_Scalar s, void    *x) ;

// Type-generic version:  x can be any supported C type or void * for a
// user-defined type.

/*
GrB_Info GrB_Scalar_setElement          // s = x
(
    GrB_Scalar s,                       // GrB_Scalar to modify
    <type> x                            // user scalar to assign to s
) ;
*/

#if GxB_STDC_VERSION >= 201112L
#define GrB_Scalar_setElement(s,x)                  \
    _Generic                                        \
    (                                               \
        (x),                                        \
            GB_CASES (GrB, Scalar_setElement)       \
    )                                               \
    (s, x)

// historical; use GrB_Scalar_setElement instead:
#define GxB_Scalar_setElement(s,x) GrB_Scalar_setElement (s, x)
#endif

//------------------------------------------------------------------------------
// GrB_Scalar_extractElement
//------------------------------------------------------------------------------

// Extract a single entry from a GrB_Scalar, x = s, typecasting from the type
// of s to the type of x as needed.

GrB_Info GrB_Scalar_extractElement_BOOL     // x = s
(
    bool *x,                        // user scalar extracted
    const GrB_Scalar s              // GrB_Scalar to extract an entry from
) ;

GrB_Info GrB_Scalar_extractElement_INT8     // x = s
(
    int8_t *x,                      // user scalar extracted
    const GrB_Scalar s              // GrB_Scalar to extract an entry from
) ;

GrB_Info GrB_Scalar_extractElement_UINT8    // x = s
(
    uint8_t *x,                     // user scalar extracted
    const GrB_Scalar s              // GrB_Scalar to extract an entry from
) ;

GrB_Info GrB_Scalar_extractElement_INT16    // x = s
(
    int16_t *x,                     // user scalar extracted
    const GrB_Scalar s              // GrB_Scalar to extract an entry from
) ;

GrB_Info GrB_Scalar_extractElement_UINT16   // x = s
(
    uint16_t *x,                    // user scalar extracted
    const GrB_Scalar s              // GrB_Scalar to extract an entry from
) ;

GrB_Info GrB_Scalar_extractElement_INT32    // x = s
(
    int32_t *x,                     // user scalar extracted
    const GrB_Scalar s              // GrB_Scalar to extract an entry from
) ;

GrB_Info GrB_Scalar_extractElement_UINT32   // x = s
(
    uint32_t *x,                    // user scalar extracted
    const GrB_Scalar s              // GrB_Scalar to extract an entry from
) ;

GrB_Info GrB_Scalar_extractElement_INT64    // x = s
(
    int64_t *x,                     // user scalar extracted
    const GrB_Scalar s              // GrB_Scalar to extract an entry from
) ;

GrB_Info GrB_Scalar_extractElement_UINT64   // x = s
(
    uint64_t *x,                    // user scalar extracted
    const GrB_Scalar s              // GrB_Scalar to extract an entry from
) ;

GrB_Info GrB_Scalar_extractElement_FP32     // x = s
(
    float *x,                       // user scalar extracted
    const GrB_Scalar s              // GrB_Scalar to extract an entry from
) ;

GrB_Info GrB_Scalar_extractElement_FP64     // x = s
(
    double *x,                      // user scalar extracted
    const GrB_Scalar s              // GrB_Scalar to extract an entry from
) ;

GrB_Info GxB_Scalar_extractElement_FC32     // x = s
(
    GxB_FC32_t *x,                  // user scalar extracted
    const GrB_Scalar s              // GrB_Scalar to extract an entry from
) ;

GrB_Info GxB_Scalar_extractElement_FC64     // x = s
(
    GxB_FC64_t *x,                  // user scalar extracted
    const GrB_Scalar s              // GrB_Scalar to extract an entry from
) ;

GrB_Info GrB_Scalar_extractElement_UDT      // x = s
(
    void *x,                        // user scalar extracted
    const GrB_Scalar s              // GrB_Scalar to extract an entry from
) ;

// historical names identical to GrB_Scalar_methods above:
GrB_Info GxB_Scalar_extractElement_BOOL   (bool     *x, const GrB_Scalar s) ;
GrB_Info GxB_Scalar_extractElement_INT8   (int8_t   *x, const GrB_Scalar s) ;
GrB_Info GxB_Scalar_extractElement_INT16  (int16_t  *x, const GrB_Scalar s) ;
GrB_Info GxB_Scalar_extractElement_INT32  (int32_t  *x, const GrB_Scalar s) ;
GrB_Info GxB_Scalar_extractElement_INT64  (int64_t  *x, const GrB_Scalar s) ;
GrB_Info GxB_Scalar_extractElement_UINT8  (uint8_t  *x, const GrB_Scalar s) ;
GrB_Info GxB_Scalar_extractElement_UINT16 (uint16_t *x, const GrB_Scalar s) ;
GrB_Info GxB_Scalar_extractElement_UINT32 (uint32_t *x, const GrB_Scalar s) ;
GrB_Info GxB_Scalar_extractElement_UINT64 (uint64_t *x, const GrB_Scalar s) ;
GrB_Info GxB_Scalar_extractElement_FP32   (float    *x, const GrB_Scalar s) ;
GrB_Info GxB_Scalar_extractElement_FP64   (double   *x, const GrB_Scalar s) ;
GrB_Info GxB_Scalar_extractElement_UDT    (void     *x, const GrB_Scalar s) ;

// Type-generic version:  x can be a pointer to any supported C type or void *
// for a user-defined type.

/*

GrB_Info GrB_Scalar_extractElement  // x = s
(
    <type> *x,                      // user scalar extracted
    const GrB_Scalar s              // GrB_Scalar to extract an entry from
) ;

*/

#if GxB_STDC_VERSION >= 201112L
#define GrB_Scalar_extractElement(x,s)                  \
    _Generic                                            \
    (                                                   \
        (x),                                            \
            GB_PCASES (GrB, Scalar_extractElement)      \
    )                                                   \
    (x, s)

// historical; use GrB_Scalar_extractElement instead:
#define GxB_Scalar_extractElement(x,s) GrB_Scalar_extractElement (x, s)
#endif

//==============================================================================
// GrB_Vector: a GraphBLAS vector
//==============================================================================

// These methods create, free, copy, and clear a vector.  The size, nvals,
// and type methods return basic information about a vector.

GrB_Info GrB_Vector_new     // create a new vector with no entries
(
    GrB_Vector *v,          // handle of vector to create
    GrB_Type type,          // type of vector to create
    GrB_Index n             // vector dimension is n-by-1
                            // (n must be <= GrB_INDEX_MAX+1)
) ;

GrB_Info GrB_Vector_dup     // make an exact copy of a vector
(
    GrB_Vector *w,          // handle of output vector to create
    const GrB_Vector u      // input vector to copy
) ;

GrB_Info GrB_Vector_clear   // clear a vector of all entries;
(                           // type and dimension remain unchanged.
    GrB_Vector v            // vector to clear
) ;

GrB_Info GrB_Vector_size    // get the dimension of a vector
(
    GrB_Index *n,           // vector dimension is n-by-1
    const GrB_Vector v      // vector to query
) ;

GrB_Info GrB_Vector_nvals   // get the number of entries in a vector
(
    GrB_Index *nvals,       // vector has nvals entries
    const GrB_Vector v      // vector to query
) ;

// historical; use GrB_get instead:
GrB_Info GxB_Vector_type (GrB_Type *type, const GrB_Vector v) ;
GrB_Info GxB_Vector_type_name (char *type_name, const GrB_Vector v) ;

GrB_Info GxB_Vector_memoryUsage  // return # of bytes used for a vector
(
    size_t *size,           // # of bytes used by the vector v
    const GrB_Vector v      // vector to query
) ;

GrB_Info GxB_Vector_iso     // return iso status of a vector
(
    bool *iso,              // true if the vector is iso-valued
    const GrB_Vector v      // vector to query
) ;

GrB_Info GrB_Vector_free    // free a vector
(
    GrB_Vector *v           // handle of vector to free
) ;

//------------------------------------------------------------------------------
// GrB_Vector_build
//------------------------------------------------------------------------------

// GrB_Vector_build:  w = sparse (I,1,X), but using any
// associative operator to assemble duplicate entries.

GrB_Info GrB_Vector_build_BOOL      // build a vector from (I,X) tuples
(
    GrB_Vector w,                   // vector to build
    const GrB_Index *Ilist,         // array of row indices of tuples
    const bool *X,                  // array of values of tuples
    GrB_Index nvals,                // number of tuples
    const GrB_BinaryOp dup          // binary function to assemble duplicates
) ;

GrB_Info GrB_Vector_build_INT8      // build a vector from (I,X) tuples
(
    GrB_Vector w,                   // vector to build
    const GrB_Index *Ilist,         // array of row indices of tuples
    const int8_t *X,                // array of values of tuples
    GrB_Index nvals,                // number of tuples
    const GrB_BinaryOp dup          // binary function to assemble duplicates
) ;

GrB_Info GrB_Vector_build_UINT8     // build a vector from (I,X) tuples
(
    GrB_Vector w,                   // vector to build
    const GrB_Index *Ilist,         // array of row indices of tuples
    const uint8_t *X,               // array of values of tuples
    GrB_Index nvals,                // number of tuples
    const GrB_BinaryOp dup          // binary function to assemble duplicates
) ;

GrB_Info GrB_Vector_build_INT16     // build a vector from (I,X) tuples
(
    GrB_Vector w,                   // vector to build
    const GrB_Index *Ilist,         // array of row indices of tuples
    const int16_t *X,               // array of values of tuples
    GrB_Index nvals,                // number of tuples
    const GrB_BinaryOp dup          // binary function to assemble duplicates
) ;

GrB_Info GrB_Vector_build_UINT16    // build a vector from (I,X) tuples
(
    GrB_Vector w,                   // vector to build
    const GrB_Index *Ilist,         // array of row indices of tuples
    const uint16_t *X,              // array of values of tuples
    GrB_Index nvals,                // number of tuples
    const GrB_BinaryOp dup          // binary function to assemble duplicates
) ;

GrB_Info GrB_Vector_build_INT32     // build a vector from (I,X) tuples
(
    GrB_Vector w,                   // vector to build
    const GrB_Index *Ilist,         // array of row indices of tuples
    const int32_t *X,               // array of values of tuples
    GrB_Index nvals,                // number of tuples
    const GrB_BinaryOp dup          // binary function to assemble duplicates
) ;

GrB_Info GrB_Vector_build_UINT32    // build a vector from (I,X) tuples
(
    GrB_Vector w,                   // vector to build
    const GrB_Index *Ilist,         // array of row indices of tuples
    const uint32_t *X,              // array of values of tuples
    GrB_Index nvals,                // number of tuples
    const GrB_BinaryOp dup          // binary function to assemble duplicates
) ;

GrB_Info GrB_Vector_build_INT64     // build a vector from (I,X) tuples
(
    GrB_Vector w,                   // vector to build
    const GrB_Index *Ilist,         // array of row indices of tuples
    const int64_t *X,               // array of values of tuples
    GrB_Index nvals,                // number of tuples
    const GrB_BinaryOp dup          // binary function to assemble duplicates
) ;

GrB_Info GrB_Vector_build_UINT64    // build a vector from (I,X) tuples
(
    GrB_Vector w,                   // vector to build
    const GrB_Index *Ilist,         // array of row indices of tuples
    const uint64_t *X,              // array of values of tuples
    GrB_Index nvals,                // number of tuples
    const GrB_BinaryOp dup          // binary function to assemble duplicates
) ;

GrB_Info GrB_Vector_build_FP32      // build a vector from (I,X) tuples
(
    GrB_Vector w,                   // vector to build
    const GrB_Index *Ilist,         // array of row indices of tuples
    const float *X,                 // array of values of tuples
    GrB_Index nvals,                // number of tuples
    const GrB_BinaryOp dup          // binary function to assemble duplicates
) ;

GrB_Info GrB_Vector_build_FP64      // build a vector from (I,X) tuples
(
    GrB_Vector w,                   // vector to build
    const GrB_Index *Ilist,         // array of row indices of tuples
    const double *X,                // array of values of tuples
    GrB_Index nvals,                // number of tuples
    const GrB_BinaryOp dup          // binary function to assemble duplicates
) ;

GrB_Info GxB_Vector_build_FC32      // build a vector from (I,X) tuples
(
    GrB_Vector w,                   // vector to build
    const GrB_Index *Ilist,         // array of row indices of tuples
    const GxB_FC32_t *X,            // array of values of tuples
    GrB_Index nvals,                // number of tuples
    const GrB_BinaryOp dup          // binary function to assemble duplicates
) ;

GrB_Info GxB_Vector_build_FC64      // build a vector from (I,X) tuples
(
    GrB_Vector w,                   // vector to build
    const GrB_Index *Ilist,         // array of row indices of tuples
    const GxB_FC64_t *X,            // array of values of tuples
    GrB_Index nvals,                // number of tuples
    const GrB_BinaryOp dup          // binary function to assemble duplicates
) ;

GrB_Info GrB_Vector_build_UDT       // build a vector from (I,X) tuples
(
    GrB_Vector w,                   // vector to build
    const GrB_Index *Ilist,         // array of row indices of tuples
    const void *X,                  // array of values of tuples
    GrB_Index nvals,                // number of tuples
    const GrB_BinaryOp dup          // binary function to assemble duplicates
) ;

GrB_Info GxB_Vector_build_Scalar    // build a vector from (i,scalar) tuples
(
    GrB_Vector w,                   // vector to build
    const GrB_Index *Ilist,         // array of row indices of tuples
    GrB_Scalar scalar,              // value for all tuples
    GrB_Index nvals                 // number of tuples
) ;

// Type-generic version:  X can be a pointer to any supported C type or void *
// for a user-defined type.

/*
GrB_Info GrB_Vector_build           // build a vector from (I,X) tuples
(
    GrB_Vector w,                   // vector to build
    const GrB_Index *Ilist,         // array of row indices of tuples
    const <type> *X,                // array of values of tuples
    GrB_Index nvals,                // number of tuples
    const GrB_BinaryOp dup          // binary function to assemble duplicates
) ;
*/

#if GxB_STDC_VERSION >= 201112L
#define GrB_Vector_build(w,Ilist,X,nvals,dup)           \
    _Generic                                            \
    (                                                   \
        (X),                                            \
            GB_PCASES (GrB, Vector_build)               \
    )                                                   \
    (w, Ilist, ((const void *) (X)), nvals, dup)
#endif

//------------------------------------------------------------------------------
// GrB_Vector_setElement
//------------------------------------------------------------------------------

// Set a single scalar in a vector, w(i) = x, typecasting from the type of x to
// the type of w as needed.

GrB_Info GrB_Vector_setElement_BOOL     // w(i) = x
(
    GrB_Vector w,                       // vector to modify
    bool x,                             // scalar to assign to w(i)
    GrB_Index i                         // row index
) ;

GrB_Info GrB_Vector_setElement_INT8     // w(i) = x
(
    GrB_Vector w,                       // vector to modify
    int8_t x,                           // scalar to assign to w(i)
    GrB_Index i                         // row index
) ;

GrB_Info GrB_Vector_setElement_UINT8    // w(i) = x
(
    GrB_Vector w,                       // vector to modify
    uint8_t x,                          // scalar to assign to w(i)
    GrB_Index i                         // row index
) ;

GrB_Info GrB_Vector_setElement_INT16    // w(i) = x
(
    GrB_Vector w,                       // vector to modify
    int16_t x,                          // scalar to assign to w(i)
    GrB_Index i                         // row index
) ;

GrB_Info GrB_Vector_setElement_UINT16   // w(i) = x
(
    GrB_Vector w,                       // vector to modify
    uint16_t x,                         // scalar to assign to w(i)
    GrB_Index i                         // row index
) ;

GrB_Info GrB_Vector_setElement_INT32    // w(i) = x
(
    GrB_Vector w,                       // vector to modify
    int32_t x,                          // scalar to assign to w(i)
    GrB_Index i                         // row index
) ;

GrB_Info GrB_Vector_setElement_UINT32   // w(i) = x
(
    GrB_Vector w,                       // vector to modify
    uint32_t x,                         // scalar to assign to w(i)
    GrB_Index i                         // row index
) ;

GrB_Info GrB_Vector_setElement_INT64    // w(i) = x
(
    GrB_Vector w,                       // vector to modify
    int64_t x,                          // scalar to assign to w(i)
    GrB_Index i                         // row index
) ;

GrB_Info GrB_Vector_setElement_UINT64   // w(i) = x
(
    GrB_Vector w,                       // vector to modify
    uint64_t x,                         // scalar to assign to w(i)
    GrB_Index i                         // row index
) ;

GrB_Info GrB_Vector_setElement_FP32     // w(i) = x
(
    GrB_Vector w,                       // vector to modify
    float x,                            // scalar to assign to w(i)
    GrB_Index i                         // row index
) ;

GrB_Info GrB_Vector_setElement_FP64     // w(i) = x
(
    GrB_Vector w,                       // vector to modify
    double x,                           // scalar to assign to w(i)
    GrB_Index i                         // row index
) ;

GrB_Info GxB_Vector_setElement_FC32     // w(i) = x
(
    GrB_Vector w,                       // vector to modify
    GxB_FC32_t x,                       // scalar to assign to w(i)
    GrB_Index i                         // row index
) ;

GrB_Info GxB_Vector_setElement_FC64     // w(i) = x
(
    GrB_Vector w,                       // vector to modify
    GxB_FC64_t x,                       // scalar to assign to w(i)
    GrB_Index i                         // row index
) ;

GrB_Info GrB_Vector_setElement_UDT      // w(i) = x
(
    GrB_Vector w,                       // vector to modify
    void *x,                            // scalar to assign to w(i)
    GrB_Index i                         // row index
) ;

GrB_Info GrB_Vector_setElement_Scalar   // w(i) = x
(
    GrB_Vector w,                       // vector to modify
    GrB_Scalar x,                       // scalar to assign to w(i)
    GrB_Index i                         // row index
) ;

// Type-generic version:  x can be any supported C type or void * for a
// user-defined type.

/*
GrB_Info GrB_Vector_setElement          // w(i) = x
(
    GrB_Vector w,                       // vector to modify
    <type> x,                           // scalar to assign to w(i)
    GrB_Index i                         // row index
) ;
*/

#if GxB_STDC_VERSION >= 201112L
#define GrB_Vector_setElement(w,x,i)                        \
    _Generic                                                \
    (                                                       \
        (x),                                                \
            GB_CASES (GrB, Vector_setElement),              \
            default:  GrB_Vector_setElement_Scalar          \
    )                                                       \
    (w, x, i)
#endif

//------------------------------------------------------------------------------
// GrB_Vector_extractElement
//------------------------------------------------------------------------------

// Extract a single entry from a vector, x = v(i), typecasting from the type of
// v to the type of x as needed.

GrB_Info GrB_Vector_extractElement_BOOL     // x = v(i)
(
    bool *x,                        // scalar extracted
    const GrB_Vector v,             // vector to extract an entry from
    GrB_Index i                     // row index
) ;

GrB_Info GrB_Vector_extractElement_INT8     // x = v(i)
(
    int8_t *x,                      // scalar extracted
    const GrB_Vector v,             // vector to extract an entry from
    GrB_Index i                     // row index
) ;

GrB_Info GrB_Vector_extractElement_UINT8    // x = v(i)
(
    uint8_t *x,                     // scalar extracted
    const GrB_Vector v,             // vector to extract an entry from
    GrB_Index i                     // row index
) ;

GrB_Info GrB_Vector_extractElement_INT16    // x = v(i)
(
    int16_t *x,                     // scalar extracted
    const GrB_Vector v,             // vector to extract an entry from
    GrB_Index i                     // row index
) ;

GrB_Info GrB_Vector_extractElement_UINT16   // x = v(i)
(
    uint16_t *x,                    // scalar extracted
    const GrB_Vector v,             // vector to extract an entry from
    GrB_Index i                     // row index
) ;

GrB_Info GrB_Vector_extractElement_INT32    // x = v(i)
(
    int32_t *x,                     // scalar extracted
    const GrB_Vector v,             // vector to extract an entry from
    GrB_Index i                     // row index
) ;

GrB_Info GrB_Vector_extractElement_UINT32   // x = v(i)
(
    uint32_t *x,                    // scalar extracted
    const GrB_Vector v,             // vector to extract an entry from
    GrB_Index i                     // row index
) ;

GrB_Info GrB_Vector_extractElement_INT64    // x = v(i)
(
    int64_t *x,                     // scalar extracted
    const GrB_Vector v,             // vector to extract an entry from
    GrB_Index i                     // row index
) ;

GrB_Info GrB_Vector_extractElement_UINT64   // x = v(i)
(
    uint64_t *x,                    // scalar extracted
    const GrB_Vector v,             // vector to extract an entry from
    GrB_Index i                     // row index
) ;

GrB_Info GrB_Vector_extractElement_FP32     // x = v(i)
(
    float *x,                       // scalar extracted
    const GrB_Vector v,             // vector to extract an entry from
    GrB_Index i                     // row index
) ;

GrB_Info GrB_Vector_extractElement_FP64     // x = v(i)
(
    double *x,                      // scalar extracted
    const GrB_Vector v,             // vector to extract an entry from
    GrB_Index i                     // row index
) ;

GrB_Info GxB_Vector_extractElement_FC32     // x = v(i)
(
    GxB_FC32_t *x,                  // scalar extracted
    const GrB_Vector v,             // vector to extract an entry from
    GrB_Index i                     // row index
) ;

GrB_Info GxB_Vector_extractElement_FC64     // x = v(i)
(
    GxB_FC64_t *x,                  // scalar extracted
    const GrB_Vector v,             // vector to extract an entry from
    GrB_Index i                     // row index
) ;

GrB_Info GrB_Vector_extractElement_UDT      // x = v(i)
(
    void *x,                        // scalar extracted
    const GrB_Vector v,             // vector to extract an entry from
    GrB_Index i                     // row index
) ;

GrB_Info GrB_Vector_extractElement_Scalar   // x = v(i)
(
    GrB_Scalar x,                   // scalar extracted
    const GrB_Vector v,             // vector to extract an entry from
    GrB_Index i                     // row index
) ;

// Type-generic version:  x can be a pointer to any supported C type or void *
// for a user-defined type.

/*
GrB_Info GrB_Vector_extractElement  // x = v(i)
(
    <type> *x,                      // scalar extracted
    const GrB_Vector v,             // vector to extract an entry from
    GrB_Index i                     // row index
) ;
*/

#if GxB_STDC_VERSION >= 201112L
#define GrB_Vector_extractElement(x,v,i)                        \
    _Generic                                                    \
    (                                                           \
        (x),                                                    \
            GB_PCASES (GrB, Vector_extractElement),             \
            default:  GrB_Vector_extractElement_Scalar          \
    )                                                           \
    (x, v, i)
#endif

// GxB_Vector_isStoredElement determines if v(i) is present in the structure
// of the vector v, as a stored element.  It does not return the value.  It
// returns GrB_SUCCESS if the element is present, or GrB_NO_VALUE otherwise.

GrB_Info GxB_Vector_isStoredElement // determine if v(i) is a stored element
(
    const GrB_Vector v,             // vector to check
    GrB_Index i                     // row index
) ;

//------------------------------------------------------------------------------
// GrB_Vector_removeElement
//------------------------------------------------------------------------------

// GrB_Vector_removeElement (v,i) removes the element v(i) from the vector v.

GrB_Info GrB_Vector_removeElement
(
    GrB_Vector v,                   // vector to remove an element from
    GrB_Index i                     // index
) ;

//------------------------------------------------------------------------------
// GrB_Vector_extractTuples
//------------------------------------------------------------------------------

// Extracts all tuples from a vector, like [I,~,X] = find (v).  If
// any parameter I and/or X is NULL, then that component is not extracted.  For
// example, to extract just the row indices, pass I as non-NULL, and X as NULL.
// This is like [I,~,~] = find (v).

GrB_Info GrB_Vector_extractTuples_BOOL      // [I,~,X] = find (v)
(
    GrB_Index *Ilist,           // array for returning row indices of tuples
    bool *X,                    // array for returning values of tuples
    GrB_Index *nvals,           // I, X size on input; # tuples on output
    const GrB_Vector v          // vector to extract tuples from
) ;

GrB_Info GrB_Vector_extractTuples_INT8      // [I,~,X] = find (v)
(
    GrB_Index *Ilist,           // array for returning row indices of tuples
    int8_t *X,                  // array for returning values of tuples
    GrB_Index *nvals,           // I, X size on input; # tuples on output
    const GrB_Vector v          // vector to extract tuples from
) ;

GrB_Info GrB_Vector_extractTuples_UINT8     // [I,~,X] = find (v)
(
    GrB_Index *Ilist,           // array for returning row indices of tuples
    uint8_t *X,                 // array for returning values of tuples
    GrB_Index *nvals,           // I, X size on input; # tuples on output
    const GrB_Vector v          // vector to extract tuples from
) ;

GrB_Info GrB_Vector_extractTuples_INT16     // [I,~,X] = find (v)
(
    GrB_Index *Ilist,           // array for returning row indices of tuples
    int16_t *X,                 // array for returning values of tuples
    GrB_Index *nvals,           // I, X size on input; # tuples on output
    const GrB_Vector v          // vector to extract tuples from
) ;

GrB_Info GrB_Vector_extractTuples_UINT16    // [I,~,X] = find (v)
(
    GrB_Index *Ilist,           // array for returning row indices of tuples
    uint16_t *X,                // array for returning values of tuples
    GrB_Index *nvals,           // I, X size on input; # tuples on output
    const GrB_Vector v          // vector to extract tuples from
) ;

GrB_Info GrB_Vector_extractTuples_INT32     // [I,~,X] = find (v)
(
    GrB_Index *Ilist,           // array for returning row indices of tuples
    int32_t *X,                 // array for returning values of tuples
    GrB_Index *nvals,           // I, X size on input; # tuples on output
    const GrB_Vector v          // vector to extract tuples from
) ;

GrB_Info GrB_Vector_extractTuples_UINT32    // [I,~,X] = find (v)
(
    GrB_Index *Ilist,           // array for returning row indices of tuples
    uint32_t *X,                // array for returning values of tuples
    GrB_Index *nvals,           // I, X size on input; # tuples on output
    const GrB_Vector v          // vector to extract tuples from
) ;

GrB_Info GrB_Vector_extractTuples_INT64     // [I,~,X] = find (v)
(
    GrB_Index *Ilist,           // array for returning row indices of tuples
    int64_t *X,                 // array for returning values of tuples
    GrB_Index *nvals,           // I, X size on input; # tuples on output
    const GrB_Vector v          // vector to extract tuples from
) ;

GrB_Info GrB_Vector_extractTuples_UINT64    // [I,~,X] = find (v)
(
    GrB_Index *Ilist,           // array for returning row indices of tuples
    uint64_t *X,                // array for returning values of tuples
    GrB_Index *nvals,           // I, X size on input; # tuples on output
    const GrB_Vector v          // vector to extract tuples from
) ;

GrB_Info GrB_Vector_extractTuples_FP32      // [I,~,X] = find (v)
(
    GrB_Index *Ilist,           // array for returning row indices of tuples
    float *X,                   // array for returning values of tuples
    GrB_Index *nvals,           // I, X size on input; # tuples on output
    const GrB_Vector v          // vector to extract tuples from
) ;

GrB_Info GrB_Vector_extractTuples_FP64      // [I,~,X] = find (v)
(
    GrB_Index *Ilist,           // array for returning row indices of tuples
    double *X,                  // array for returning values of tuples
    GrB_Index *nvals,           // I, X size on input; # tuples on output
    const GrB_Vector v          // vector to extract tuples from
) ;

GrB_Info GxB_Vector_extractTuples_FC32      // [I,~,X] = find (v)
(
    GrB_Index *Ilist,           // array for returning row indices of tuples
    GxB_FC32_t *X,              // array for returning values of tuples
    GrB_Index *nvals,           // I, X size on input; # tuples on output
    const GrB_Vector v          // vector to extract tuples from
) ;

GrB_Info GxB_Vector_extractTuples_FC64      // [I,~,X] = find (v)
(
    GrB_Index *Ilist,           // array for returning row indices of tuples
    GxB_FC64_t *X,              // array for returning values of tuples
    GrB_Index *nvals,           // I, X size on input; # tuples on output
    const GrB_Vector v          // vector to extract tuples from
) ;

GrB_Info GrB_Vector_extractTuples_UDT       // [I,~,X] = find (v)
(
    GrB_Index *Ilist,           // array for returning row indices of tuples
    void *X,                    // array for returning values of tuples
    GrB_Index *nvals,           // I, X size on input; # tuples on output
    const GrB_Vector v          // vector to extract tuples from
) ;

// Type-generic version:  X can be a pointer to any supported C type or void *
// for a user-defined type.

/*
GrB_Info GrB_Vector_extractTuples           // [I,~,X] = find (v)
(
    GrB_Index *Ilist,           // array for returning row indices of tuples
    <type> *X,                  // array for returning values of tuples
    GrB_Index *nvals,           // I, X size on input; # tuples on output
    const GrB_Vector v          // vector to extract tuples from
) ;
*/

#if GxB_STDC_VERSION >= 201112L
#define GrB_Vector_extractTuples(Ilist,X,nvals,v)       \
    _Generic                                            \
    (                                                   \
        (X),                                            \
            GB_PCASES (GrB, Vector_extractTuples)       \
    )                                                   \
    (Ilist, X, nvals, v)
#endif

//==============================================================================
// GrB_Matrix: a GraphBLAS matrix
//==============================================================================

// These methods create, free, copy, and clear a matrix.  The nrows, ncols,
// nvals, and type methods return basic information about a matrix.

GrB_Info GrB_Matrix_new     // create a new matrix with no entries
(
    GrB_Matrix *A,          // handle of matrix to create
    GrB_Type type,          // type of matrix to create
    GrB_Index nrows,        // matrix dimension is nrows-by-ncols
    GrB_Index ncols         // (nrows and ncols must be <= GrB_INDEX_MAX+1)
) ;

GrB_Info GrB_Matrix_dup     // make an exact copy of a matrix
(
    GrB_Matrix *C,          // handle of output matrix to create
    const GrB_Matrix A      // input matrix to copy
) ;

GrB_Info GrB_Matrix_clear   // clear a matrix of all entries;
(                           // type and dimensions remain unchanged
    GrB_Matrix A            // matrix to clear
) ;

GrB_Info GrB_Matrix_nrows   // get the number of rows of a matrix
(
    GrB_Index *nrows,       // matrix has nrows rows
    const GrB_Matrix A      // matrix to query
) ;

GrB_Info GrB_Matrix_ncols   // get the number of columns of a matrix
(
    GrB_Index *ncols,       // matrix has ncols columns
    const GrB_Matrix A      // matrix to query
) ;

GrB_Info GrB_Matrix_nvals   // get the number of entries in a matrix
(
    GrB_Index *nvals,       // matrix has nvals entries
    const GrB_Matrix A      // matrix to query
) ;

// historical; use GrB_get instead:
GrB_Info GxB_Matrix_type (GrB_Type *type, const GrB_Matrix A) ;
GrB_Info GxB_Matrix_type_name (char *type_name, const GrB_Matrix A) ;

GrB_Info GxB_Matrix_memoryUsage  // return # of bytes used for a matrix
(
    size_t *size,           // # of bytes used by the matrix A
    const GrB_Matrix A      // matrix to query
) ;

GrB_Info GxB_Matrix_iso     // return iso status of a matrix
(
    bool *iso,              // true if the matrix is iso-valued
    const GrB_Matrix A      // matrix to query
) ;

GrB_Info GrB_Matrix_free    // free a matrix
(
    GrB_Matrix *A           // handle of matrix to free
) ;

//------------------------------------------------------------------------------
// GrB_Matrix_build
//------------------------------------------------------------------------------

// GrB_Matrix_build:  C = sparse (I,J,X), but using any
// associative operator to assemble duplicate entries.

GrB_Info GrB_Matrix_build_BOOL      // build a matrix from (I,J,X) tuples
(
    GrB_Matrix C,                   // matrix to build
    const GrB_Index *Ilist,         // array of row indices of tuples
    const GrB_Index *J,             // array of column indices of tuples
    const bool *X,                  // array of values of tuples
    GrB_Index nvals,                // number of tuples
    const GrB_BinaryOp dup          // binary function to assemble duplicates
) ;

GrB_Info GrB_Matrix_build_INT8      // build a matrix from (I,J,X) tuples
(
    GrB_Matrix C,                   // matrix to build
    const GrB_Index *Ilist,         // array of row indices of tuples
    const GrB_Index *J,             // array of column indices of tuples
    const int8_t *X,                // array of values of tuples
    GrB_Index nvals,                // number of tuples
    const GrB_BinaryOp dup          // binary function to assemble duplicates
) ;

GrB_Info GrB_Matrix_build_UINT8     // build a matrix from (I,J,X) tuples
(
    GrB_Matrix C,                   // matrix to build
    const GrB_Index *Ilist,         // array of row indices of tuples
    const GrB_Index *J,             // array of column indices of tuples
    const uint8_t *X,               // array of values of tuples
    GrB_Index nvals,                // number of tuples
    const GrB_BinaryOp dup          // binary function to assemble duplicates
) ;

GrB_Info GrB_Matrix_build_INT16     // build a matrix from (I,J,X) tuples
(
    GrB_Matrix C,                   // matrix to build
    const GrB_Index *Ilist,         // array of row indices of tuples
    const GrB_Index *J,             // array of column indices of tuples
    const int16_t *X,               // array of values of tuples
    GrB_Index nvals,                // number of tuples
    const GrB_BinaryOp dup          // binary function to assemble duplicates
) ;

GrB_Info GrB_Matrix_build_UINT16    // build a matrix from (I,J,X) tuples
(
    GrB_Matrix C,                   // matrix to build
    const GrB_Index *Ilist,         // array of row indices of tuples
    const GrB_Index *J,             // array of column indices of tuples
    const uint16_t *X,              // array of values of tuples
    GrB_Index nvals,                // number of tuples
    const GrB_BinaryOp dup          // binary function to assemble duplicates
) ;

GrB_Info GrB_Matrix_build_INT32     // build a matrix from (I,J,X) tuples
(
    GrB_Matrix C,                   // matrix to build
    const GrB_Index *Ilist,         // array of row indices of tuples
    const GrB_Index *J,             // array of column indices of tuples
    const int32_t *X,               // array of values of tuples
    GrB_Index nvals,                // number of tuples
    const GrB_BinaryOp dup          // binary function to assemble duplicates
) ;

GrB_Info GrB_Matrix_build_UINT32    // build a matrix from (I,J,X) tuples
(
    GrB_Matrix C,                   // matrix to build
    const GrB_Index *Ilist,         // array of row indices of tuples
    const GrB_Index *J,             // array of column indices of tuples
    const uint32_t *X,              // array of values of tuples
    GrB_Index nvals,                // number of tuples
    const GrB_BinaryOp dup          // binary function to assemble duplicates
) ;

GrB_Info GrB_Matrix_build_INT64     // build a matrix from (I,J,X) tuples
(
    GrB_Matrix C,                   // matrix to build
    const GrB_Index *Ilist,         // array of row indices of tuples
    const GrB_Index *J,             // array of column indices of tuples
    const int64_t *X,               // array of values of tuples
    GrB_Index nvals,                // number of tuples
    const GrB_BinaryOp dup          // binary function to assemble duplicates
) ;

GrB_Info GrB_Matrix_build_UINT64    // build a matrix from (I,J,X) tuples
(
    GrB_Matrix C,                   // matrix to build
    const GrB_Index *Ilist,         // array of row indices of tuples
    const GrB_Index *J,             // array of column indices of tuples
    const uint64_t *X,              // array of values of tuples
    GrB_Index nvals,                // number of tuples
    const GrB_BinaryOp dup          // binary function to assemble duplicates
) ;

GrB_Info GrB_Matrix_build_FP32      // build a matrix from (I,J,X) tuples
(
    GrB_Matrix C,                   // matrix to build
    const GrB_Index *Ilist,         // array of row indices of tuples
    const GrB_Index *J,             // array of column indices of tuples
    const float *X,                 // array of values of tuples
    GrB_Index nvals,                // number of tuples
    const GrB_BinaryOp dup          // binary function to assemble duplicates
) ;

GrB_Info GrB_Matrix_build_FP64      // build a matrix from (I,J,X) tuples
(
    GrB_Matrix C,                   // matrix to build
    const GrB_Index *Ilist,         // array of row indices of tuples
    const GrB_Index *J,             // array of column indices of tuples
    const double *X,                // array of values of tuples
    GrB_Index nvals,                // number of tuples
    const GrB_BinaryOp dup          // binary function to assemble duplicates
) ;

GrB_Info GxB_Matrix_build_FC32      // build a matrix from (I,J,X) tuples
(
    GrB_Matrix C,                   // matrix to build
    const GrB_Index *Ilist,         // array of row indices of tuples
    const GrB_Index *J,             // array of column indices of tuples
    const GxB_FC32_t *X,            // array of values of tuples
    GrB_Index nvals,                // number of tuples
    const GrB_BinaryOp dup          // binary function to assemble duplicates
) ;

GrB_Info GxB_Matrix_build_FC64      // build a matrix from (I,J,X) tuples
(
    GrB_Matrix C,                   // matrix to build
    const GrB_Index *Ilist,         // array of row indices of tuples
    const GrB_Index *J,             // array of column indices of tuples
    const GxB_FC64_t *X,            // array of values of tuples
    GrB_Index nvals,                // number of tuples
    const GrB_BinaryOp dup          // binary function to assemble duplicates
) ;

GrB_Info GrB_Matrix_build_UDT       // build a matrix from (I,J,X) tuples
(
    GrB_Matrix C,                   // matrix to build
    const GrB_Index *Ilist,         // array of row indices of tuples
    const GrB_Index *J,             // array of column indices of tuples
    const void *X,                  // array of values of tuples
    GrB_Index nvals,                // number of tuples
    const GrB_BinaryOp dup          // binary function to assemble duplicates
) ;

GrB_Info GxB_Matrix_build_Scalar    // build a matrix from (I,J,scalar) tuples
(
    GrB_Matrix C,                   // matrix to build
    const GrB_Index *Ilist,         // array of row indices of tuples
    const GrB_Index *J,             // array of column indices of tuples
    GrB_Scalar scalar,              // value for all tuples
    GrB_Index nvals                 // number of tuples
) ;

// Type-generic version:  X can be a pointer to any supported C type or void *
// for a user-defined type.

/*
GrB_Info GrB_Matrix_build           // build a matrix from (I,J,X) tuples
(
    GrB_Matrix C,                   // matrix to build
    const GrB_Index *Ilist,         // array of row indices of tuples
    const GrB_Index *J,             // array of column indices of tuples
    const <type> *X,                // array of values of tuples
    GrB_Index nvals,                // number of tuples
    const GrB_BinaryOp dup          // binary function to assemble duplicates
) ;
*/

#if GxB_STDC_VERSION >= 201112L
#define GrB_Matrix_build(C,Ilist,J,X,nvals,dup)         \
    _Generic                                            \
    (                                                   \
        (X),                                            \
            GB_PCASES (GrB, Matrix_build)               \
    )                                                   \
    (C, Ilist, J, ((const void *) (X)), nvals, dup)
#endif

//------------------------------------------------------------------------------
// GrB_Matrix_setElement
//------------------------------------------------------------------------------

// Set a single entry in a matrix, C(i,j) = x, typecasting
// from the type of x to the type of C, as needed.

GrB_Info GrB_Matrix_setElement_BOOL     // C (i,j) = x
(
    GrB_Matrix C,                       // matrix to modify
    bool x,                             // scalar to assign to C(i,j)
    GrB_Index i,                        // row index
    GrB_Index j                         // column index
) ;

GrB_Info GrB_Matrix_setElement_INT8     // C (i,j) = x
(
    GrB_Matrix C,                       // matrix to modify
    int8_t x,                           // scalar to assign to C(i,j)
    GrB_Index i,                        // row index
    GrB_Index j                         // column index
) ;

GrB_Info GrB_Matrix_setElement_UINT8    // C (i,j) = x
(
    GrB_Matrix C,                       // matrix to modify
    uint8_t x,                          // scalar to assign to C(i,j)
    GrB_Index i,                        // row index
    GrB_Index j                         // column index
) ;

GrB_Info GrB_Matrix_setElement_INT16    // C (i,j) = x
(
    GrB_Matrix C,                       // matrix to modify
    int16_t x,                          // scalar to assign to C(i,j)
    GrB_Index i,                        // row index
    GrB_Index j                         // column index
) ;

GrB_Info GrB_Matrix_setElement_UINT16   // C (i,j) = x
(
    GrB_Matrix C,                       // matrix to modify
    uint16_t x,                         // scalar to assign to C(i,j)
    GrB_Index i,                        // row index
    GrB_Index j                         // column index
) ;

GrB_Info GrB_Matrix_setElement_INT32    // C (i,j) = x
(
    GrB_Matrix C,                       // matrix to modify
    int32_t x,                          // scalar to assign to C(i,j)
    GrB_Index i,                        // row index
    GrB_Index j                         // column index
) ;

GrB_Info GrB_Matrix_setElement_UINT32   // C (i,j) = x
(
    GrB_Matrix C,                       // matrix to modify
    uint32_t x,                         // scalar to assign to C(i,j)
    GrB_Index i,                        // row index
    GrB_Index j                         // column index
) ;

GrB_Info GrB_Matrix_setElement_INT64    // C (i,j) = x
(
    GrB_Matrix C,                       // matrix to modify
    int64_t x,                          // scalar to assign to C(i,j)
    GrB_Index i,                        // row index
    GrB_Index j                         // column index
) ;

GrB_Info GrB_Matrix_setElement_UINT64   // C (i,j) = x
(
    GrB_Matrix C,                       // matrix to modify
    uint64_t x,                         // scalar to assign to C(i,j)
    GrB_Index i,                        // row index
    GrB_Index j                         // column index
) ;

GrB_Info GrB_Matrix_setElement_FP32     // C (i,j) = x
(
    GrB_Matrix C,                       // matrix to modify
    float x,                            // scalar to assign to C(i,j)
    GrB_Index i,                        // row index
    GrB_Index j                         // column index
) ;

GrB_Info GrB_Matrix_setElement_FP64     // C (i,j) = x
(
    GrB_Matrix C,                       // matrix to modify
    double x,                           // scalar to assign to C(i,j)
    GrB_Index i,                        // row index
    GrB_Index j                         // column index
) ;

GrB_Info GxB_Matrix_setElement_FC32     // C (i,j) = x
(
    GrB_Matrix C,                       // matrix to modify
    GxB_FC32_t x,                       // scalar to assign to C(i,j)
    GrB_Index i,                        // row index
    GrB_Index j                         // column index
) ;

GrB_Info GxB_Matrix_setElement_FC64     // C (i,j) = x
(
    GrB_Matrix C,                       // matrix to modify
    GxB_FC64_t x,                       // scalar to assign to C(i,j)
    GrB_Index i,                        // row index
    GrB_Index j                         // column index
) ;

GrB_Info GrB_Matrix_setElement_UDT      // C (i,j) = x
(
    GrB_Matrix C,                       // matrix to modify
    void *x,                            // scalar to assign to C(i,j)
    GrB_Index i,                        // row index
    GrB_Index j                         // column index
) ;

GrB_Info GrB_Matrix_setElement_Scalar   // C (i,j) = x
(
    GrB_Matrix C,                       // matrix to modify
    GrB_Scalar x,                       // scalar to assign to C(i,j)
    GrB_Index i,                        // row index
    GrB_Index j                         // column index
) ;

// Type-generic version:  x can be any supported C type or void * for a
// user-defined type.

/*
GrB_Info GrB_Matrix_setElement          // C (i,j) = x
(
    GrB_Matrix C,                       // matrix to modify
    <type> x,                           // scalar to assign to C(i,j)
    GrB_Index i,                        // row index
    GrB_Index j                         // column index
) ;
*/

#if GxB_STDC_VERSION >= 201112L
#define GrB_Matrix_setElement(C,x,i,j)                      \
    _Generic                                                \
    (                                                       \
        (x),                                                \
            GB_CASES (GrB, Matrix_setElement),              \
            default:  GrB_Matrix_setElement_Scalar          \
    )                                                       \
    (C, x, i, j)
#endif

//------------------------------------------------------------------------------
// GrB_Matrix_extractElement
//------------------------------------------------------------------------------

// Extract a single entry from a matrix, x = A(i,j), typecasting from the type
// of A to the type of x, as needed.

GrB_Info GrB_Matrix_extractElement_BOOL     // x = A(i,j)
(
    bool *x,                            // extracted scalar
    const GrB_Matrix A,                 // matrix to extract a scalar from
    GrB_Index i,                        // row index
    GrB_Index j                         // column index
) ;

GrB_Info GrB_Matrix_extractElement_INT8     // x = A(i,j)
(
    int8_t *x,                          // extracted scalar
    const GrB_Matrix A,                 // matrix to extract a scalar from
    GrB_Index i,                        // row index
    GrB_Index j                         // column index
) ;

GrB_Info GrB_Matrix_extractElement_UINT8    // x = A(i,j)
(
    uint8_t *x,                         // extracted scalar
    const GrB_Matrix A,                 // matrix to extract a scalar from
    GrB_Index i,                        // row index
    GrB_Index j                         // column index
) ;

GrB_Info GrB_Matrix_extractElement_INT16    // x = A(i,j)
(
    int16_t *x,                         // extracted scalar
    const GrB_Matrix A,                 // matrix to extract a scalar from
    GrB_Index i,                        // row index
    GrB_Index j                         // column index
) ;

GrB_Info GrB_Matrix_extractElement_UINT16   // x = A(i,j)
(
    uint16_t *x,                        // extracted scalar
    const GrB_Matrix A,                 // matrix to extract a scalar from
    GrB_Index i,                        // row index
    GrB_Index j                         // column index
) ;

GrB_Info GrB_Matrix_extractElement_INT32    // x = A(i,j)
(
    int32_t *x,                         // extracted scalar
    const GrB_Matrix A,                 // matrix to extract a scalar from
    GrB_Index i,                        // row index
    GrB_Index j                         // column index
) ;

GrB_Info GrB_Matrix_extractElement_UINT32   // x = A(i,j)
(
    uint32_t *x,                        // extracted scalar
    const GrB_Matrix A,                 // matrix to extract a scalar from
    GrB_Index i,                        // row index
    GrB_Index j                         // column index
) ;

GrB_Info GrB_Matrix_extractElement_INT64    // x = A(i,j)
(
    int64_t *x,                         // extracted scalar
    const GrB_Matrix A,                 // matrix to extract a scalar from
    GrB_Index i,                        // row index
    GrB_Index j                         // column index
) ;

GrB_Info GrB_Matrix_extractElement_UINT64   // x = A(i,j)
(
    uint64_t *x,                        // extracted scalar
    const GrB_Matrix A,                 // matrix to extract a scalar from
    GrB_Index i,                        // row index
    GrB_Index j                         // column index
) ;

GrB_Info GrB_Matrix_extractElement_FP32     // x = A(i,j)
(
    float *x,                           // extracted scalar
    const GrB_Matrix A,                 // matrix to extract a scalar from
    GrB_Index i,                        // row index
    GrB_Index j                         // column index
) ;

GrB_Info GrB_Matrix_extractElement_FP64     // x = A(i,j)
(
    double *x,                          // extracted scalar
    const GrB_Matrix A,                 // matrix to extract a scalar from
    GrB_Index i,                        // row index
    GrB_Index j                         // column index
) ;

GrB_Info GxB_Matrix_extractElement_FC32     // x = A(i,j)
(
    GxB_FC32_t *x,                      // extracted scalar
    const GrB_Matrix A,                 // matrix to extract a scalar from
    GrB_Index i,                        // row index
    GrB_Index j                         // column index
) ;

GrB_Info GxB_Matrix_extractElement_FC64     // x = A(i,j)
(
    GxB_FC64_t *x,                      // extracted scalar
    const GrB_Matrix A,                 // matrix to extract a scalar from
    GrB_Index i,                        // row index
    GrB_Index j                         // column index
) ;

GrB_Info GrB_Matrix_extractElement_UDT      // x = A(i,j)
(
    void *x,                            // extracted scalar
    const GrB_Matrix A,                 // matrix to extract a scalar from
    GrB_Index i,                        // row index
    GrB_Index j                         // column index
) ;

GrB_Info GrB_Matrix_extractElement_Scalar   // x = A(i,j)
(
    GrB_Scalar x,                       // extracted scalar
    const GrB_Matrix A,                 // matrix to extract a scalar from
    GrB_Index i,                        // row index
    GrB_Index j                         // column index
) ;

// Type-generic version:  x can be a pointer to any supported C type or void *
// for a user-defined type.

/*
GrB_Info GrB_Matrix_extractElement      // x = A(i,j)
(
    <type> *x,                          // extracted scalar
    const GrB_Matrix A,                 // matrix to extract a scalar from
    GrB_Index i,                        // row index
    GrB_Index j                         // column index
) ;
*/

#if GxB_STDC_VERSION >= 201112L
#define GrB_Matrix_extractElement(x,A,i,j)                      \
    _Generic                                                    \
    (                                                           \
        (x),                                                    \
            GB_PCASES (GrB, Matrix_extractElement),             \
            default:  GrB_Matrix_extractElement_Scalar          \
    )                                                           \
    (x, A, i, j)
#endif

// GxB_Matrix_isStoredElement determines if A(i,j) is present in the structure
// of the matrix A, as a stored element.  It does not return the value.  It
// returns GrB_SUCCESS if the element is present, or GrB_NO_VALUE otherwise.

GrB_Info GxB_Matrix_isStoredElement // determine if A(i,j) is a stored element
(
    const GrB_Matrix A,                 // matrix to check
    GrB_Index i,                        // row index
    GrB_Index j                         // column index
) ;

//------------------------------------------------------------------------------
// GrB_Matrix_removeElement
//------------------------------------------------------------------------------

// GrB_Matrix_removeElement (A,i,j) removes the entry A(i,j) from the matrix A.

GrB_Info GrB_Matrix_removeElement
(
    GrB_Matrix C,                   // matrix to remove entry from
    GrB_Index i,                    // row index
    GrB_Index j                     // column index
) ;

//------------------------------------------------------------------------------
// GrB_Matrix_extractTuples
//------------------------------------------------------------------------------

// Extracts all tuples from a matrix, like [I,J,X] = find (A).  If
// any parameter I, J and/or X is NULL, then that component is not extracted.
// For example, to extract just the row and col indices, pass I and J as
// non-NULL, and X as NULL.  This is like [I,J,~] = find (A).

GrB_Info GrB_Matrix_extractTuples_BOOL      // [I,J,X] = find (A)
(
    GrB_Index *Ilist,           // array for returning row indices of tuples
    GrB_Index *J,               // array for returning col indices of tuples
    bool *X,                    // array for returning values of tuples
    GrB_Index *nvals,           // I,J,X size on input; # tuples on output
    const GrB_Matrix A          // matrix to extract tuples from
) ;

GrB_Info GrB_Matrix_extractTuples_INT8      // [I,J,X] = find (A)
(
    GrB_Index *Ilist,           // array for returning row indices of tuples
    GrB_Index *J,               // array for returning col indices of tuples
    int8_t *X,                  // array for returning values of tuples
    GrB_Index *nvals,           // I,J,X size on input; # tuples on output
    const GrB_Matrix A          // matrix to extract tuples from
) ;

GrB_Info GrB_Matrix_extractTuples_UINT8     // [I,J,X] = find (A)
(
    GrB_Index *Ilist,           // array for returning row indices of tuples
    GrB_Index *J,               // array for returning col indices of tuples
    uint8_t *X,                 // array for returning values of tuples
    GrB_Index *nvals,           // I,J,X size on input; # tuples on output
    const GrB_Matrix A          // matrix to extract tuples from
) ;

GrB_Info GrB_Matrix_extractTuples_INT16     // [I,J,X] = find (A)
(
    GrB_Index *Ilist,           // array for returning row indices of tuples
    GrB_Index *J,               // array for returning col indices of tuples
    int16_t *X,                 // array for returning values of tuples
    GrB_Index *nvals,           // I,J,X size on input; # tuples on output
    const GrB_Matrix A          // matrix to extract tuples from
) ;

GrB_Info GrB_Matrix_extractTuples_UINT16    // [I,J,X] = find (A)
(
    GrB_Index *Ilist,           // array for returning row indices of tuples
    GrB_Index *J,               // array for returning col indices of tuples
    uint16_t *X,                // array for returning values of tuples
    GrB_Index *nvals,           // I,J,X size on input; # tuples on output
    const GrB_Matrix A          // matrix to extract tuples from
) ;

GrB_Info GrB_Matrix_extractTuples_INT32     // [I,J,X] = find (A)
(
    GrB_Index *Ilist,           // array for returning row indices of tuples
    GrB_Index *J,               // array for returning col indices of tuples
    int32_t *X,                 // array for returning values of tuples
    GrB_Index *nvals,           // I,J,X size on input; # tuples on output
    const GrB_Matrix A          // matrix to extract tuples from
) ;

GrB_Info GrB_Matrix_extractTuples_UINT32    // [I,J,X] = find (A)
(
    GrB_Index *Ilist,           // array for returning row indices of tuples
    GrB_Index *J,               // array for returning col indices of tuples
    uint32_t *X,                // array for returning values of tuples
    GrB_Index *nvals,           // I,J,X size on input; # tuples on output
    const GrB_Matrix A          // matrix to extract tuples from
) ;

GrB_Info GrB_Matrix_extractTuples_INT64     // [I,J,X] = find (A)
(
    GrB_Index *Ilist,           // array for returning row indices of tuples
    GrB_Index *J,               // array for returning col indices of tuples
    int64_t *X,                 // array for returning values of tuples
    GrB_Index *nvals,           // I,J,X size on input; # tuples on output
    const GrB_Matrix A          // matrix to extract tuples from
) ;

GrB_Info GrB_Matrix_extractTuples_UINT64    // [I,J,X] = find (A)
(
    GrB_Index *Ilist,           // array for returning row indices of tuples
    GrB_Index *J,               // array for returning col indices of tuples
    uint64_t *X,                // array for returning values of tuples
    GrB_Index *nvals,           // I,J,X size on input; # tuples on output
    const GrB_Matrix A          // matrix to extract tuples from
) ;

GrB_Info GrB_Matrix_extractTuples_FP32      // [I,J,X] = find (A)
(
    GrB_Index *Ilist,           // array for returning row indices of tuples
    GrB_Index *J,               // array for returning col indices of tuples
    float *X,                   // array for returning values of tuples
    GrB_Index *nvals,           // I,J,X size on input; # tuples on output
    const GrB_Matrix A          // matrix to extract tuples from
) ;

GrB_Info GrB_Matrix_extractTuples_FP64      // [I,J,X] = find (A)
(
    GrB_Index *Ilist,           // array for returning row indices of tuples
    GrB_Index *J,               // array for returning col indices of tuples
    double *X,                  // array for returning values of tuples
    GrB_Index *nvals,           // I,J,X size on input; # tuples on output
    const GrB_Matrix A          // matrix to extract tuples from
) ;

GrB_Info GxB_Matrix_extractTuples_FC32      // [I,J,X] = find (A)
(
    GrB_Index *Ilist,           // array for returning row indices of tuples
    GrB_Index *J,               // array for returning col indices of tuples
    GxB_FC32_t *X,              // array for returning values of tuples
    GrB_Index *nvals,           // I,J,X size on input; # tuples on output
    const GrB_Matrix A          // matrix to extract tuples from
) ;

GrB_Info GxB_Matrix_extractTuples_FC64      // [I,J,X] = find (A)
(
    GrB_Index *Ilist,           // array for returning row indices of tuples
    GrB_Index *J,               // array for returning col indices of tuples
    GxB_FC64_t *X,              // array for returning values of tuples
    GrB_Index *nvals,           // I,J,X size on input; # tuples on output
    const GrB_Matrix A          // matrix to extract tuples from
) ;

GrB_Info GrB_Matrix_extractTuples_UDT       // [I,J,X] = find (A)
(
    GrB_Index *Ilist,           // array for returning row indices of tuples
    GrB_Index *J,               // array for returning col indices of tuples
    void *X,                    // array for returning values of tuples
    GrB_Index *nvals,           // I,J,X size on input; # tuples on output
    const GrB_Matrix A          // matrix to extract tuples from
) ;

// Type-generic version:  X can be a pointer to any supported C type or void *
// for a user-defined type.

/*
GrB_Info GrB_Matrix_extractTuples           // [I,J,X] = find (A)
(
    GrB_Index *Ilist,           // array for returning row indices of tuples
    GrB_Index *J,               // array for returning col indices of tuples
    <type> *X,                  // array for returning values of tuples
    GrB_Index *nvals,           // I,J,X size on input; # tuples on output
    const GrB_Matrix A          // matrix to extract tuples from
) ;
*/

#if GxB_STDC_VERSION >= 201112L
#define GrB_Matrix_extractTuples(Ilist,J,X,nvals,A)     \
    _Generic                                            \
    (                                                   \
        (X),                                            \
            GB_PCASES (GrB, Matrix_extractTuples)       \
    )                                                   \
    (Ilist, J, X, nvals, A)
#endif

//------------------------------------------------------------------------------
// GxB_Matrix_concat and GxB_Matrix_split
//------------------------------------------------------------------------------

// GxB_Matrix_concat concatenates an array of matrices (Tiles) into a single
// GrB_Matrix C.

// Tiles is an m-by-n dense array of matrices held in row-major format, where
// Tiles [i*n+j] is the (i,j)th tile, and where m > 0 and n > 0 must hold.  Let
// A{i,j} denote the (i,j)th tile.  The matrix C is constructed by
// concatenating these tiles together, as:

//  C = [ A{0,0}   A{0,1}   A{0,2}   ... A{0,n-1}
//        A{1,0}   A{1,1}   A{1,2}   ... A{1,n-1}
//        ...
//        A{m-1,0} A{m-1,1} A{m-1,2} ... A{m-1,n-1} ]

// On input, the matrix C must already exist.  Any existing entries in C are
// discarded.  C must have dimensions nrows by ncols where nrows is the sum of
// # of rows in the matrices A{i,0} for all i, and ncols is the sum of the # of
// columns in the matrices A{0,j} for all j.  All matrices in any given tile
// row i must have the same number of rows (that is, nrows(A{i,0}) must equal
// nrows(A{i,j}) for all j), and all matrices in any given tile column j must
// have the same number of columns (that is, ncols(A{0,j}) must equal
// ncols(A{i,j}) for all i).

// The type of C is unchanged, and all matrices A{i,j} are typecasted into the
// type of C.  Any settings made to C by GrB_set (format by row
// or by column, bitmap switch, hyper switch, and sparsity control) are
// unchanged.

GrB_Info GxB_Matrix_concat          // concatenate a 2D array of matrices
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix *Tiles,        // 2D row-major array of size m-by-n
    const GrB_Index m,
    const GrB_Index n,
    const GrB_Descriptor desc       // unused, except threading control
) ;

// GxB_Matrix_split does the opposite of GxB_Matrix_concat.  It splits a single
// input matrix A into a 2D array of tiles.  On input, the Tiles array must be
// a non-NULL pointer to a previously allocated array of size at least m*n
// where both m and n must be > 0.  The Tiles_nrows array has size m, and
// Tiles_ncols has size n.  The (i,j)th tile has dimension
// Tiles_nrows[i]-by-Tiles_ncols[j].  The sum of Tiles_nrows [0:m-1] must equal
// the number of rows of A, and the sum of Tiles_ncols [0:n-1] must equal the
// number of columns of A.  The type of each tile is the same as the type of A;
// no typecasting is done.

GrB_Info GxB_Matrix_split           // split a matrix into 2D array of matrices
(
    GrB_Matrix *Tiles,              // 2D row-major array of size m-by-n
    const GrB_Index m,
    const GrB_Index n,
    const GrB_Index *Tile_nrows,    // array of size m
    const GrB_Index *Tile_ncols,    // array of size n
    const GrB_Matrix A,             // input matrix to split
    const GrB_Descriptor desc       // unused, except threading control
) ;

//------------------------------------------------------------------------------
// GxB_Matrix_diag, GxB_Vector_diag, GrB_Matrix_diag
//------------------------------------------------------------------------------

// GrB_Matrix_diag constructs a new matrix from a vector.  Let n be the length
// of the v vector, from GrB_Vector_size (&n, v).  If k = 0, then C is an
// n-by-n diagonal matrix with the entries from v along the main diagonal of C,
// with C(i,i) = v(i).  If k is nonzero, C is square with dimension n+abs(k).
// If k is positive, it denotes diagonals above the main diagonal, with
// C(i,i+k) = v(i).  If k is negative, it denotes diagonals below the main
// diagonal of C, with C(i-k,i) = v(i).  C is constructed with the same type
// as v.

GrB_Info GrB_Matrix_diag    // build a diagonal matrix from a vector
(
    GrB_Matrix *C,                  // output matrix
    const GrB_Vector v,             // input vector
    int64_t k
) ;

// GrB_Matrix_diag is like GxB_Matrix_diag (&C, v, k, NULL), except that C must
// already exist on input, of the correct size.  Any existing entries in C are
// discarded.  The type of C is preserved, so that if the type of C and v
// differ, the entries are typecasted into the type of C.  Any settings made to
// C by GrB_set (format by row or by column, bitmap switch, hyper
// switch, and sparsity control) are unchanged.

GrB_Info GxB_Matrix_diag    // construct a diagonal matrix from a vector
(
    GrB_Matrix C,                   // output matrix
    const GrB_Vector v,             // input vector
    int64_t k,
    const GrB_Descriptor desc       // to specify # of threads
) ;

// GxB_Vector_diag extracts a vector v from an input matrix A, which may be
// rectangular.  If k = 0, the main diagonal of A is extracted; k > 0 denotes
// diagonals above the main diagonal of A, and k < 0 denotes diagonals below
// the main diagonal of A.  Let A have dimension m-by-n.  If k is in the range
// 0 to n-1, then v has length min(m,n-k).  If k is negative and in the range
// -1 to -m+1, then v has length min(m+k,n).  If k is outside these ranges,
// v has length 0 (this is not an error).

// v must already exist on input, of the correct length; that is
// GrB_Vector_size (&len,v) must return len = 0 if k >= n or k <= -m, len =
// min(m,n-k) if k is in the range 0 to n-1, and len = min(m+k,n) if k is in
// the range -1 to -m+1.  Any existing entries in v are discarded.  The type of
// v is preserved, so that if the type of A and v differ, the entries are
// typecasted into the type of v.  Any settings made to v by
// GrB_set (bitmap switch and sparsity control) are unchanged.

GrB_Info GxB_Vector_diag    // extract a diagonal from a matrix, as a vector
(
    GrB_Vector v,                   // output vector
    const GrB_Matrix A,             // input matrix
    int64_t k,
    const GrB_Descriptor desc       // unused, except threading control
) ;

//==============================================================================
// GxB_Context: for managing computational resources
//==============================================================================

GrB_Info GxB_Context_new            // create a new Context
(
    GxB_Context *Context            // handle of Context to create
) ;

GrB_Info GxB_Context_free           // free a Context
(
    GxB_Context *Context            // handle of Context to free
) ;

GrB_Info GxB_Context_engage         // engage a Context
(
    GxB_Context Context             // Context to engage
) ;

GrB_Info GxB_Context_disengage      // disengage a Context
(
    GxB_Context Context             // Context to disengage
) ;

//==============================================================================
// GxB_set and GxB_get: historical; use GrB_set and GrB_get instead
//==============================================================================

// historical: use GrB_set and GrB_get instead of these methods:
GrB_Info GxB_Matrix_Option_set       (GrB_Matrix, GxB_Option_Field, ...) ;
GrB_Info GxB_Matrix_Option_set_INT32 (GrB_Matrix, GxB_Option_Field, int32_t) ;
GrB_Info GxB_Matrix_Option_set_FP64  (GrB_Matrix, GxB_Option_Field, double) ;
GrB_Info GxB_Matrix_Option_get       (GrB_Matrix, GxB_Option_Field, ...) ;
GrB_Info GxB_Matrix_Option_get_INT32 (GrB_Matrix, GxB_Option_Field, int32_t *) ;
GrB_Info GxB_Matrix_Option_get_FP64  (GrB_Matrix, GxB_Option_Field, double *) ;
GrB_Info GxB_Vector_Option_set       (GrB_Vector, GxB_Option_Field, ...) ;
GrB_Info GxB_Vector_Option_set_INT32 (GrB_Vector, GxB_Option_Field, int32_t) ;
GrB_Info GxB_Vector_Option_set_FP64  (GrB_Vector, GxB_Option_Field, double) ;
GrB_Info GxB_Vector_Option_get       (GrB_Vector, GxB_Option_Field, ...) ;
GrB_Info GxB_Vector_Option_get_INT32 (GrB_Vector, GxB_Option_Field, int32_t *) ;
GrB_Info GxB_Vector_Option_get_FP64  (GrB_Vector, GxB_Option_Field, double *) ;
GrB_Info GxB_Global_Option_set             (GxB_Option_Field, ...) ;
GrB_Info GxB_Global_Option_set_INT32       (GxB_Option_Field, int32_t) ;
GrB_Info GxB_Global_Option_set_FP64        (GxB_Option_Field, double) ;
GrB_Info GxB_Global_Option_set_FP64_ARRAY  (GxB_Option_Field, double *) ;
GrB_Info GxB_Global_Option_set_INT64_ARRAY (GxB_Option_Field, int64_t *) ;
GrB_Info GxB_Global_Option_set_CHAR        (GxB_Option_Field, const char *) ;
GrB_Info GxB_Global_Option_set_FUNCTION    (GxB_Option_Field, void *) ;
GrB_Info GxB_Global_Option_get             (GxB_Option_Field, ...) ;
GrB_Info GxB_Global_Option_get_INT32       (GxB_Option_Field, int32_t *) ;
GrB_Info GxB_Global_Option_get_FP64        (GxB_Option_Field, double *) ;
GrB_Info GxB_Global_Option_get_INT64       (GxB_Option_Field, int64_t *) ;
GrB_Info GxB_Global_Option_get_CHAR        (GxB_Option_Field, const char **) ;
GrB_Info GxB_Global_Option_get_FUNCTION    (GxB_Option_Field, void **) ;
GrB_Info GxB_Context_set_INT32 (GxB_Context, GxB_Context_Field, int32_t) ;
GrB_Info GxB_Context_set_FP64  (GxB_Context, GxB_Context_Field, double) ;
GrB_Info GxB_Context_set       (GxB_Context, GxB_Context_Field, ...) ;
GrB_Info GxB_Context_get_INT32 (GxB_Context, GxB_Context_Field, int32_t *) ;
GrB_Info GxB_Context_get_FP64  (GxB_Context, GxB_Context_Field, double *) ;
GrB_Info GxB_Context_get       (GxB_Context, GxB_Context_Field, ...) ;

#if GxB_STDC_VERSION >= 201112L
#define GxB_set(arg1,...)                                       \
    _Generic                                                    \
    (                                                           \
        (arg1),                                                 \
            default:           GxB_Global_Option_set ,          \
            GxB_Option_Field : GxB_Global_Option_set ,          \
            GrB_Vector       : GxB_Vector_Option_set ,          \
            GrB_Matrix       : GxB_Matrix_Option_set ,          \
            GrB_Descriptor   : GxB_Desc_set          ,          \
            GxB_Context      : GxB_Context_set                  \
    )                                                           \
    (arg1, __VA_ARGS__)

#define GxB_get(arg1,...)                                       \
    _Generic                                                    \
    (                                                           \
        (arg1),                                                 \
            default:           GxB_Global_Option_get ,          \
            GxB_Option_Field : GxB_Global_Option_get ,          \
            GrB_Vector       : GxB_Vector_Option_get ,          \
            GrB_Matrix       : GxB_Matrix_Option_get ,          \
            GrB_Descriptor   : GxB_Desc_get          ,          \
            GxB_Context      : GxB_Context_get                  \
    )                                                           \
    (arg1, __VA_ARGS__)
#endif

//==============================================================================
// GrB_set and GrB_get
//==============================================================================

//------------------------------------------------------------------------------
// GrB_get: get a scalar, string, enum, size, or void * from an object
//------------------------------------------------------------------------------

GrB_Info GrB_Scalar_get_Scalar (GrB_Scalar, GrB_Scalar, GrB_Field) ;
GrB_Info GrB_Scalar_get_String (GrB_Scalar, char *    , GrB_Field) ;
GrB_Info GrB_Scalar_get_INT32  (GrB_Scalar, int32_t * , GrB_Field) ;
GrB_Info GrB_Scalar_get_SIZE   (GrB_Scalar, size_t *  , GrB_Field) ;
GrB_Info GrB_Scalar_get_VOID   (GrB_Scalar, void *    , GrB_Field) ;

GrB_Info GrB_Vector_get_Scalar (GrB_Vector, GrB_Scalar, GrB_Field) ;
GrB_Info GrB_Vector_get_String (GrB_Vector, char *    , GrB_Field) ;
GrB_Info GrB_Vector_get_INT32  (GrB_Vector, int32_t * , GrB_Field) ;
GrB_Info GrB_Vector_get_SIZE   (GrB_Vector, size_t *  , GrB_Field) ;
GrB_Info GrB_Vector_get_VOID   (GrB_Vector, void *    , GrB_Field) ;

GrB_Info GrB_Matrix_get_Scalar (GrB_Matrix, GrB_Scalar, GrB_Field) ;
GrB_Info GrB_Matrix_get_String (GrB_Matrix, char *    , GrB_Field) ;
GrB_Info GrB_Matrix_get_INT32  (GrB_Matrix, int32_t * , GrB_Field) ;
GrB_Info GrB_Matrix_get_SIZE   (GrB_Matrix, size_t *  , GrB_Field) ;
GrB_Info GrB_Matrix_get_VOID   (GrB_Matrix, void *    , GrB_Field) ;

GrB_Info GxB_Serialized_get_Scalar (const void *, GrB_Scalar, GrB_Field, size_t) ;
GrB_Info GxB_Serialized_get_String (const void *, char *    , GrB_Field, size_t) ;
GrB_Info GxB_Serialized_get_INT32  (const void *, int32_t * , GrB_Field, size_t) ;
GrB_Info GxB_Serialized_get_SIZE   (const void *, size_t *  , GrB_Field, size_t) ;
GrB_Info GxB_Serialized_get_VOID   (const void *, void *    , GrB_Field, size_t) ;

GrB_Info GrB_UnaryOp_get_Scalar (GrB_UnaryOp, GrB_Scalar, GrB_Field) ;
GrB_Info GrB_UnaryOp_get_String (GrB_UnaryOp, char *    , GrB_Field) ;
GrB_Info GrB_UnaryOp_get_INT32  (GrB_UnaryOp, int32_t * , GrB_Field) ;
GrB_Info GrB_UnaryOp_get_SIZE   (GrB_UnaryOp, size_t *  , GrB_Field) ;
GrB_Info GrB_UnaryOp_get_VOID   (GrB_UnaryOp, void *    , GrB_Field) ;

GrB_Info GrB_IndexUnaryOp_get_Scalar (GrB_IndexUnaryOp, GrB_Scalar, GrB_Field) ;
GrB_Info GrB_IndexUnaryOp_get_String (GrB_IndexUnaryOp, char *    , GrB_Field) ;
GrB_Info GrB_IndexUnaryOp_get_INT32  (GrB_IndexUnaryOp, int32_t * , GrB_Field) ;
GrB_Info GrB_IndexUnaryOp_get_SIZE   (GrB_IndexUnaryOp, size_t *  , GrB_Field) ;
GrB_Info GrB_IndexUnaryOp_get_VOID   (GrB_IndexUnaryOp, void *    , GrB_Field) ;

GrB_Info GrB_BinaryOp_get_Scalar (GrB_BinaryOp, GrB_Scalar, GrB_Field) ;
GrB_Info GrB_BinaryOp_get_String (GrB_BinaryOp, char *    , GrB_Field) ;
GrB_Info GrB_BinaryOp_get_INT32  (GrB_BinaryOp, int32_t * , GrB_Field) ;
GrB_Info GrB_BinaryOp_get_SIZE   (GrB_BinaryOp, size_t *  , GrB_Field) ;
GrB_Info GrB_BinaryOp_get_VOID   (GrB_BinaryOp, void *    , GrB_Field) ;

GrB_Info GrB_Monoid_get_Scalar (GrB_Monoid, GrB_Scalar, GrB_Field) ;
GrB_Info GrB_Monoid_get_String (GrB_Monoid, char *    , GrB_Field) ;
GrB_Info GrB_Monoid_get_INT32  (GrB_Monoid, int32_t * , GrB_Field) ;
GrB_Info GrB_Monoid_get_SIZE   (GrB_Monoid, size_t *  , GrB_Field) ;
GrB_Info GrB_Monoid_get_VOID   (GrB_Monoid, void *    , GrB_Field) ;

GrB_Info GrB_Semiring_get_Scalar (GrB_Semiring, GrB_Scalar, GrB_Field) ;
GrB_Info GrB_Semiring_get_String (GrB_Semiring, char *    , GrB_Field) ;
GrB_Info GrB_Semiring_get_INT32  (GrB_Semiring, int32_t * , GrB_Field) ;
GrB_Info GrB_Semiring_get_SIZE   (GrB_Semiring, size_t *  , GrB_Field) ;
GrB_Info GrB_Semiring_get_VOID   (GrB_Semiring, void *    , GrB_Field) ;

GrB_Info GrB_Descriptor_get_Scalar (GrB_Descriptor, GrB_Scalar, GrB_Field) ;
GrB_Info GrB_Descriptor_get_String (GrB_Descriptor, char *    , GrB_Field) ;
GrB_Info GrB_Descriptor_get_INT32  (GrB_Descriptor, int32_t * , GrB_Field) ;
GrB_Info GrB_Descriptor_get_SIZE   (GrB_Descriptor, size_t *  , GrB_Field) ;
GrB_Info GrB_Descriptor_get_VOID   (GrB_Descriptor, void *    , GrB_Field) ;

GrB_Info GrB_Type_get_Scalar (GrB_Type, GrB_Scalar, GrB_Field) ;
GrB_Info GrB_Type_get_String (GrB_Type, char *    , GrB_Field) ;
GrB_Info GrB_Type_get_INT32  (GrB_Type, int32_t * , GrB_Field) ;
GrB_Info GrB_Type_get_SIZE   (GrB_Type, size_t *  , GrB_Field) ;
GrB_Info GrB_Type_get_VOID   (GrB_Type, void *    , GrB_Field) ;

GrB_Info GrB_Global_get_Scalar (GrB_Global, GrB_Scalar, GrB_Field) ;
GrB_Info GrB_Global_get_String (GrB_Global, char *    , GrB_Field) ;
GrB_Info GrB_Global_get_INT32  (GrB_Global, int32_t * , GrB_Field) ;
GrB_Info GrB_Global_get_SIZE   (GrB_Global, size_t *  , GrB_Field) ;
GrB_Info GrB_Global_get_VOID   (GrB_Global, void *    , GrB_Field) ;

GrB_Info GxB_Context_get_Scalar (GxB_Context, GrB_Scalar, GrB_Field) ;
GrB_Info GxB_Context_get_String (GxB_Context, char *    , GrB_Field) ;
GrB_Info GxB_Context_get_INT    (GxB_Context, int32_t * , GrB_Field) ;
GrB_Info GxB_Context_get_SIZE   (GxB_Context, size_t *  , GrB_Field) ;
GrB_Info GxB_Context_get_VOID   (GxB_Context, void *    , GrB_Field) ;

// GrB_get (object, value, field):
#if GxB_STDC_VERSION >= 201112L
#define GrB_get(object,value,...)                                           \
    _Generic                                                                \
    (                                                                       \
        (object),                                                           \
            GrB_Scalar :                                                    \
                _Generic                                                    \
                (                                                           \
                    (value),                                                \
                        GrB_Scalar  : GrB_Scalar_get_Scalar ,               \
                        char *      : GrB_Scalar_get_String ,               \
                        int32_t *   : GrB_Scalar_get_INT32  ,               \
                        size_t *    : GrB_Scalar_get_SIZE   ,               \
                        void *      : GrB_Scalar_get_VOID                   \
                ) ,                                                         \
            GrB_Vector :                                                    \
                _Generic                                                    \
                (                                                           \
                    (value),                                                \
                        GrB_Scalar  : GrB_Vector_get_Scalar ,               \
                        char *      : GrB_Vector_get_String ,               \
                        int32_t *   : GrB_Vector_get_INT32  ,               \
                        size_t *    : GrB_Vector_get_SIZE   ,               \
                        void *      : GrB_Vector_get_VOID                   \
                ) ,                                                         \
            GrB_Matrix :                                                    \
                _Generic                                                    \
                (                                                           \
                    (value),                                                \
                        GrB_Scalar  : GrB_Matrix_get_Scalar ,               \
                        char *      : GrB_Matrix_get_String ,               \
                        int32_t *   : GrB_Matrix_get_INT32  ,               \
                        size_t *    : GrB_Matrix_get_SIZE   ,               \
                        void *      : GrB_Matrix_get_VOID                   \
                ) ,                                                         \
            GrB_UnaryOp :                                                   \
                _Generic                                                    \
                (                                                           \
                    (value),                                                \
                        GrB_Scalar  : GrB_UnaryOp_get_Scalar ,              \
                        char *      : GrB_UnaryOp_get_String ,              \
                        int32_t *   : GrB_UnaryOp_get_INT32  ,              \
                        size_t *    : GrB_UnaryOp_get_SIZE   ,              \
                        void *      : GrB_UnaryOp_get_VOID                  \
                ) ,                                                         \
            GrB_IndexUnaryOp :                                              \
                _Generic                                                    \
                (                                                           \
                    (value),                                                \
                        GrB_Scalar  : GrB_IndexUnaryOp_get_Scalar ,         \
                        char *      : GrB_IndexUnaryOp_get_String ,         \
                        int32_t *   : GrB_IndexUnaryOp_get_INT32  ,         \
                        size_t *    : GrB_IndexUnaryOp_get_SIZE   ,         \
                        void *      : GrB_IndexUnaryOp_get_VOID             \
                ) ,                                                         \
            GrB_BinaryOp :                                                  \
                _Generic                                                    \
                (                                                           \
                    (value),                                                \
                        GrB_Scalar  : GrB_BinaryOp_get_Scalar ,             \
                        char *      : GrB_BinaryOp_get_String ,             \
                        int32_t *   : GrB_BinaryOp_get_INT32  ,             \
                        size_t *    : GrB_BinaryOp_get_SIZE   ,             \
                        void *      : GrB_BinaryOp_get_VOID                 \
                ) ,                                                         \
            GrB_Monoid :                                                    \
                _Generic                                                    \
                (                                                           \
                    (value),                                                \
                        GrB_Scalar  : GrB_Monoid_get_Scalar ,               \
                        char *      : GrB_Monoid_get_String ,               \
                        int32_t *   : GrB_Monoid_get_INT32  ,               \
                        size_t *    : GrB_Monoid_get_SIZE   ,               \
                        void *      : GrB_Monoid_get_VOID                   \
                ) ,                                                         \
            GrB_Semiring :                                                  \
                _Generic                                                    \
                (                                                           \
                    (value),                                                \
                        GrB_Scalar  : GrB_Semiring_get_Scalar ,             \
                        char *      : GrB_Semiring_get_String ,             \
                        int32_t *   : GrB_Semiring_get_INT32  ,             \
                        size_t *    : GrB_Semiring_get_SIZE   ,             \
                        void *      : GrB_Semiring_get_VOID                 \
                ) ,                                                         \
            GrB_Type :                                                      \
                _Generic                                                    \
                (                                                           \
                    (value),                                                \
                        GrB_Scalar  : GrB_Type_get_Scalar ,                 \
                        char *      : GrB_Type_get_String ,                 \
                        int32_t *   : GrB_Type_get_INT32  ,                 \
                        size_t *    : GrB_Type_get_SIZE   ,                 \
                        void *      : GrB_Type_get_VOID                     \
                ) ,                                                         \
            GrB_Descriptor :                                                \
                _Generic                                                    \
                (                                                           \
                    (value),                                                \
                        GrB_Scalar  : GrB_Descriptor_get_Scalar ,           \
                        char *      : GrB_Descriptor_get_String ,           \
                        int32_t *   : GrB_Descriptor_get_INT32  ,           \
                        size_t *    : GrB_Descriptor_get_SIZE   ,           \
                        void *      : GrB_Descriptor_get_VOID               \
                ) ,                                                         \
            GrB_Global :                                                    \
                _Generic                                                    \
                (                                                           \
                    (value),                                                \
                        GrB_Scalar  : GrB_Global_get_Scalar ,               \
                        char *      : GrB_Global_get_String ,               \
                        int32_t *   : GrB_Global_get_INT32  ,               \
                        size_t *    : GrB_Global_get_SIZE   ,               \
                        void *      : GrB_Global_get_VOID                   \
                ) ,                                                         \
            GxB_Context :                                                   \
                _Generic                                                    \
                (                                                           \
                    (value),                                                \
                        GrB_Scalar  : GxB_Context_get_Scalar ,              \
                        char *      : GxB_Context_get_String ,              \
                        int32_t *   : GxB_Context_get_INT    ,              \
                        size_t *    : GxB_Context_get_SIZE   ,              \
                        void *      : GxB_Context_get_VOID                  \
                ) ,                                                         \
            const void *:                                                   \
                _Generic                                                    \
                (                                                           \
                    (value),                                                \
                        GrB_Scalar  : GxB_Serialized_get_Scalar ,           \
                        char *      : GxB_Serialized_get_String ,           \
                        int32_t *   : GxB_Serialized_get_INT32  ,           \
                        size_t *    : GxB_Serialized_get_SIZE   ,           \
                        void *      : GxB_Serialized_get_VOID               \
                ) ,                                                         \
            void *:                                                         \
                _Generic                                                    \
                (                                                           \
                    (value),                                                \
                        GrB_Scalar  : GxB_Serialized_get_Scalar ,           \
                        char *      : GxB_Serialized_get_String ,           \
                        int32_t *   : GxB_Serialized_get_INT32  ,           \
                        size_t *    : GxB_Serialized_get_SIZE   ,           \
                        void *      : GxB_Serialized_get_VOID               \
                )                                                           \
    ) (object, value, __VA_ARGS__)
#endif

//------------------------------------------------------------------------------
// GrB_set: set a scalar, string, enum, size, or void * of an object
//------------------------------------------------------------------------------

GrB_Info GrB_Scalar_set_Scalar (GrB_Scalar, GrB_Scalar, GrB_Field) ;
GrB_Info GrB_Scalar_set_String (GrB_Scalar, char *    , GrB_Field) ;
GrB_Info GrB_Scalar_set_INT32  (GrB_Scalar, int32_t   , GrB_Field) ;
GrB_Info GrB_Scalar_set_VOID   (GrB_Scalar, void *    , GrB_Field, size_t) ;

GrB_Info GrB_Vector_set_Scalar (GrB_Vector, GrB_Scalar, GrB_Field) ;
GrB_Info GrB_Vector_set_String (GrB_Vector, char *    , GrB_Field) ;
GrB_Info GrB_Vector_set_INT32  (GrB_Vector, int32_t   , GrB_Field) ;
GrB_Info GrB_Vector_set_VOID   (GrB_Vector, void *    , GrB_Field, size_t) ;

GrB_Info GrB_Matrix_set_Scalar (GrB_Matrix, GrB_Scalar, GrB_Field) ;
GrB_Info GrB_Matrix_set_String (GrB_Matrix, char *    , GrB_Field) ;
GrB_Info GrB_Matrix_set_INT32  (GrB_Matrix, int32_t   , GrB_Field) ;
GrB_Info GrB_Matrix_set_VOID   (GrB_Matrix, void *    , GrB_Field, size_t) ;

GrB_Info GrB_UnaryOp_set_Scalar (GrB_UnaryOp, GrB_Scalar, GrB_Field) ;
GrB_Info GrB_UnaryOp_set_String (GrB_UnaryOp, char *    , GrB_Field) ;
GrB_Info GrB_UnaryOp_set_INT32  (GrB_UnaryOp, int32_t   , GrB_Field) ;
GrB_Info GrB_UnaryOp_set_VOID   (GrB_UnaryOp, void *    , GrB_Field, size_t) ;

GrB_Info GrB_IndexUnaryOp_set_Scalar (GrB_IndexUnaryOp, GrB_Scalar, GrB_Field) ;
GrB_Info GrB_IndexUnaryOp_set_String (GrB_IndexUnaryOp, char *    , GrB_Field) ;
GrB_Info GrB_IndexUnaryOp_set_INT32  (GrB_IndexUnaryOp, int32_t   , GrB_Field) ;
GrB_Info GrB_IndexUnaryOp_set_VOID   (GrB_IndexUnaryOp, void *    , GrB_Field,
                                                                    size_t) ;

GrB_Info GrB_BinaryOp_set_Scalar (GrB_BinaryOp, GrB_Scalar, GrB_Field) ;
GrB_Info GrB_BinaryOp_set_String (GrB_BinaryOp, char *    , GrB_Field) ;
GrB_Info GrB_BinaryOp_set_INT32  (GrB_BinaryOp, int32_t   , GrB_Field) ;
GrB_Info GrB_BinaryOp_set_VOID   (GrB_BinaryOp, void *    , GrB_Field, size_t) ;

GrB_Info GrB_Monoid_set_Scalar (GrB_Monoid, GrB_Scalar, GrB_Field) ;
GrB_Info GrB_Monoid_set_String (GrB_Monoid, char *    , GrB_Field) ;
GrB_Info GrB_Monoid_set_INT32  (GrB_Monoid, int32_t   , GrB_Field) ;
GrB_Info GrB_Monoid_set_VOID   (GrB_Monoid, void *    , GrB_Field, size_t) ;

GrB_Info GrB_Semiring_set_Scalar (GrB_Semiring, GrB_Scalar, GrB_Field) ;
GrB_Info GrB_Semiring_set_String (GrB_Semiring, char *    , GrB_Field) ;
GrB_Info GrB_Semiring_set_INT32  (GrB_Semiring, int32_t   , GrB_Field) ;
GrB_Info GrB_Semiring_set_VOID   (GrB_Semiring, void *    , GrB_Field, size_t) ;

GrB_Info GrB_Descriptor_set_Scalar (GrB_Descriptor, GrB_Scalar, GrB_Field) ;
GrB_Info GrB_Descriptor_set_String (GrB_Descriptor, char *    , GrB_Field) ;
GrB_Info GrB_Descriptor_set_INT32  (GrB_Descriptor, int32_t   , GrB_Field) ;
GrB_Info GrB_Descriptor_set_VOID   (GrB_Descriptor, void *    , GrB_Field,
                                                                size_t) ;

GrB_Info GrB_Type_set_Scalar (GrB_Type, GrB_Scalar, GrB_Field) ;
GrB_Info GrB_Type_set_String (GrB_Type, char *    , GrB_Field) ;
GrB_Info GrB_Type_set_INT32  (GrB_Type, int32_t   , GrB_Field) ;
GrB_Info GrB_Type_set_VOID   (GrB_Type, void *    , GrB_Field, size_t) ;

GrB_Info GrB_Global_set_Scalar (GrB_Global, GrB_Scalar, GrB_Field) ;
GrB_Info GrB_Global_set_String (GrB_Global, char *    , GrB_Field) ;
GrB_Info GrB_Global_set_INT32  (GrB_Global, int32_t   , GrB_Field) ;
GrB_Info GrB_Global_set_VOID   (GrB_Global, void *    , GrB_Field, size_t) ;

GrB_Info GxB_Context_set_Scalar (GxB_Context, GrB_Scalar, GrB_Field) ;
GrB_Info GxB_Context_set_String (GxB_Context, char *    , GrB_Field) ;
GrB_Info GxB_Context_set_INT    (GxB_Context, int32_t   , GrB_Field) ;
GrB_Info GxB_Context_set_VOID   (GxB_Context, void *    , GrB_Field, size_t) ;

// GrB_set (object, value, field) or (object, value, field, size) for _VOID
#if GxB_STDC_VERSION >= 201112L
#define GrB_set(object,value,...)                                           \
    _Generic                                                                \
    (                                                                       \
        (object),                                                           \
            GrB_Scalar :                                                    \
                _Generic                                                    \
                (                                                           \
                    (value),                                                \
                        GrB_Scalar  : GrB_Scalar_set_Scalar ,               \
                        char *      : GrB_Scalar_set_String ,               \
                        int32_t     : GrB_Scalar_set_INT32  ,               \
                        void *      : GrB_Scalar_set_VOID                   \
                ) ,                                                         \
            GrB_Vector :                                                    \
                _Generic                                                    \
                (                                                           \
                    (value),                                                \
                        GrB_Scalar  : GrB_Vector_set_Scalar ,               \
                        char *      : GrB_Vector_set_String ,               \
                        int32_t     : GrB_Vector_set_INT32  ,               \
                        void *      : GrB_Vector_set_VOID                   \
                ) ,                                                         \
            GrB_Matrix :                                                    \
                _Generic                                                    \
                (                                                           \
                    (value),                                                \
                        GrB_Scalar  : GrB_Matrix_set_Scalar ,               \
                        char *      : GrB_Matrix_set_String ,               \
                        int32_t     : GrB_Matrix_set_INT32  ,               \
                        void *      : GrB_Matrix_set_VOID                   \
                ) ,                                                         \
            GrB_UnaryOp :                                                   \
                _Generic                                                    \
                (                                                           \
                    (value),                                                \
                        GrB_Scalar  : GrB_UnaryOp_set_Scalar ,              \
                        char *      : GrB_UnaryOp_set_String ,              \
                        int32_t     : GrB_UnaryOp_set_INT32  ,              \
                        void *      : GrB_UnaryOp_set_VOID                  \
                ) ,                                                         \
            GrB_IndexUnaryOp :                                              \
                _Generic                                                    \
                (                                                           \
                    (value),                                                \
                        GrB_Scalar  : GrB_IndexUnaryOp_set_Scalar ,         \
                        char *      : GrB_IndexUnaryOp_set_String ,         \
                        int32_t     : GrB_IndexUnaryOp_set_INT32  ,         \
                        void *      : GrB_IndexUnaryOp_set_VOID             \
                ) ,                                                         \
            GrB_BinaryOp :                                                  \
                _Generic                                                    \
                (                                                           \
                    (value),                                                \
                        GrB_Scalar  : GrB_BinaryOp_set_Scalar ,             \
                        char *      : GrB_BinaryOp_set_String ,             \
                        int32_t     : GrB_BinaryOp_set_INT32  ,             \
                        void *      : GrB_BinaryOp_set_VOID                 \
                ) ,                                                         \
            GrB_Monoid :                                                    \
                _Generic                                                    \
                (                                                           \
                    (value),                                                \
                        GrB_Scalar  : GrB_Monoid_set_Scalar ,               \
                        char *      : GrB_Monoid_set_String ,               \
                        int32_t     : GrB_Monoid_set_INT32  ,               \
                        void *      : GrB_Monoid_set_VOID                   \
                ) ,                                                         \
            GrB_Semiring :                                                  \
                _Generic                                                    \
                (                                                           \
                    (value),                                                \
                        GrB_Scalar  : GrB_Semiring_set_Scalar ,             \
                        char *      : GrB_Semiring_set_String ,             \
                        int32_t     : GrB_Semiring_set_INT32  ,             \
                        void *      : GrB_Semiring_set_VOID                 \
                ) ,                                                         \
            GrB_Type :                                                      \
                _Generic                                                    \
                (                                                           \
                    (value),                                                \
                        GrB_Scalar  : GrB_Type_set_Scalar ,                 \
                        char *      : GrB_Type_set_String ,                 \
                        int32_t     : GrB_Type_set_INT32  ,                 \
                        void *      : GrB_Type_set_VOID                     \
                ) ,                                                         \
            GrB_Descriptor :                                                \
                _Generic                                                    \
                (                                                           \
                    (value),                                                \
                        GrB_Scalar  : GrB_Descriptor_set_Scalar ,           \
                        char *      : GrB_Descriptor_set_String ,           \
                        int32_t     : GrB_Descriptor_set_INT32  ,           \
                        void *      : GrB_Descriptor_set_VOID               \
                ) ,                                                         \
            GrB_Global :                                                    \
                _Generic                                                    \
                (                                                           \
                    (value),                                                \
                        GrB_Scalar  : GrB_Global_set_Scalar ,               \
                        char *      : GrB_Global_set_String ,               \
                        int32_t     : GrB_Global_set_INT32  ,               \
                        void *      : GrB_Global_set_VOID                   \
                ) ,                                                         \
            GxB_Context :                                                   \
                _Generic                                                    \
                (                                                           \
                    (value),                                                \
                        GrB_Scalar  : GxB_Context_set_Scalar ,              \
                        char *      : GxB_Context_set_String ,              \
                        int32_t     : GxB_Context_set_INT    ,              \
                        void *      : GxB_Context_set_VOID                  \
                )                                                           \
    ) (object, value, __VA_ARGS__)
#endif

//==============================================================================
// GrB_free: free any GraphBLAS object
//==============================================================================

#if GxB_STDC_VERSION >= 201112L
#define GrB_free(object)                                \
    _Generic                                            \
    (                                                   \
        (object),                                       \
            GrB_Type         *: GrB_Type_free         , \
            GrB_UnaryOp      *: GrB_UnaryOp_free      , \
            GrB_BinaryOp     *: GrB_BinaryOp_free     , \
            GrB_IndexUnaryOp *: GrB_IndexUnaryOp_free , \
            GrB_Monoid       *: GrB_Monoid_free       , \
            GrB_Semiring     *: GrB_Semiring_free     , \
            GrB_Scalar       *: GrB_Scalar_free       , \
            GrB_Vector       *: GrB_Vector_free       , \
            GrB_Matrix       *: GrB_Matrix_free       , \
            GrB_Descriptor   *: GrB_Descriptor_free   , \
            GxB_Context      *: GxB_Context_free      , \
            GxB_Iterator     *: GxB_Iterator_free       \
    )                                                   \
    (object)
#endif

//==============================================================================
// GrB_wait: finish computations
//==============================================================================

// Finish all pending work in a specific object.

GrB_Info GrB_Type_wait         (GrB_Type       type    , GrB_WaitMode waitmode);
GrB_Info GrB_UnaryOp_wait      (GrB_UnaryOp    op      , GrB_WaitMode waitmode);
GrB_Info GrB_BinaryOp_wait     (GrB_BinaryOp   op      , GrB_WaitMode waitmode);
GrB_Info GrB_IndexUnaryOp_wait (GrB_IndexUnaryOp op    , GrB_WaitMode waitmode);
GrB_Info GrB_Monoid_wait       (GrB_Monoid     monoid  , GrB_WaitMode waitmode);
GrB_Info GrB_Semiring_wait     (GrB_Semiring   semiring, GrB_WaitMode waitmode);
GrB_Info GrB_Descriptor_wait   (GrB_Descriptor desc    , GrB_WaitMode waitmode);
GrB_Info GrB_Scalar_wait       (GrB_Scalar     s       , GrB_WaitMode waitmode);
GrB_Info GrB_Vector_wait       (GrB_Vector     v       , GrB_WaitMode waitmode);
GrB_Info GrB_Matrix_wait       (GrB_Matrix     A       , GrB_WaitMode waitmode);
GrB_Info GxB_Context_wait      (GxB_Context    Context , GrB_WaitMode waitmode);

// GrB_wait (object,waitmode) polymorphic function:
#if GxB_STDC_VERSION >= 201112L
#define GrB_wait(object,waitmode)                       \
    _Generic                                            \
    (                                                   \
        (object),                                       \
            GrB_Type         : GrB_Type_wait         ,  \
            GrB_UnaryOp      : GrB_UnaryOp_wait      ,  \
            GrB_BinaryOp     : GrB_BinaryOp_wait     ,  \
            GrB_IndexUnaryOp : GrB_IndexUnaryOp_wait ,  \
            GrB_Monoid       : GrB_Monoid_wait       ,  \
            GrB_Semiring     : GrB_Semiring_wait     ,  \
            GrB_Scalar       : GrB_Scalar_wait       ,  \
            GrB_Vector       : GrB_Vector_wait       ,  \
            GrB_Matrix       : GrB_Matrix_wait       ,  \
            GxB_Context      : GxB_Context_wait      ,  \
            GrB_Descriptor   : GrB_Descriptor_wait      \
    )                                                   \
    (object, waitmode)
#endif

// NOTE: GxB_Scalar_wait is historical; use GrB_Scalar_wait instead
GrB_Info GxB_Scalar_wait (GrB_Scalar *s) ;

//==============================================================================
// GrB_error: error handling
//==============================================================================

// Each GraphBLAS method and operation returns a GrB_Info error code.
// GrB_error returns additional information on the error in a thread-safe
// null-terminated string.  The string returned by GrB_error is owned by
// the GraphBLAS library and must not be free'd.

GrB_Info GrB_Type_error         (const char **error, const GrB_Type      type) ;
GrB_Info GrB_UnaryOp_error      (const char **error, const GrB_UnaryOp     op) ;
GrB_Info GrB_BinaryOp_error     (const char **error, const GrB_BinaryOp    op) ;
GrB_Info GrB_IndexUnaryOp_error (const char **error, const GrB_IndexUnaryOp op) ;
GrB_Info GrB_Monoid_error       (const char **error, const GrB_Monoid monoid) ;
GrB_Info GrB_Semiring_error     (const char **error, const GrB_Semiring semiring) ;
GrB_Info GrB_Scalar_error       (const char **error, const GrB_Scalar     s) ;
GrB_Info GrB_Vector_error       (const char **error, const GrB_Vector     v) ;
GrB_Info GrB_Matrix_error       (const char **error, const GrB_Matrix     A) ;
GrB_Info GrB_Descriptor_error   (const char **error, const GrB_Descriptor d) ;
// GxB_Scalar_error is historical: use GrB_Scalar_error instead
GrB_Info GxB_Scalar_error       (const char **error, const GrB_Scalar     s) ;
GrB_Info GxB_Context_error      (const char **error, const GxB_Context Context);

// GrB_error (error,object) polymorphic function:
#if GxB_STDC_VERSION >= 201112L
#define GrB_error(error,object)                                 \
    _Generic                                                    \
    (                                                           \
        (object),                                               \
                  GrB_Type         : GrB_Type_error         ,   \
                  GrB_UnaryOp      : GrB_UnaryOp_error      ,   \
                  GrB_BinaryOp     : GrB_BinaryOp_error     ,   \
                  GrB_IndexUnaryOp : GrB_IndexUnaryOp_error ,   \
                  GrB_Monoid       : GrB_Monoid_error       ,   \
                  GrB_Semiring     : GrB_Semiring_error     ,   \
                  GrB_Scalar       : GrB_Scalar_error       ,   \
                  GrB_Vector       : GrB_Vector_error       ,   \
                  GrB_Matrix       : GrB_Matrix_error       ,   \
                  GxB_Context      : GxB_Context_error      ,   \
                  GrB_Descriptor   : GrB_Descriptor_error       \
    )                                                           \
    (error, object)
#endif

//==============================================================================
// GrB_mxm, vxm, mxv: matrix multiplication over a semiring
//==============================================================================

GrB_Info GrB_mxm                    // C<Mask> = accum (C, A*B)
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix Mask,          // optional mask for C, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C,T)
    const GrB_Semiring semiring,    // defines '+' and '*' for A*B
    const GrB_Matrix A,             // first input:  matrix A
    const GrB_Matrix B,             // second input: matrix B
    const GrB_Descriptor desc       // descriptor for C, Mask, A, and B
) ;

GrB_Info GrB_vxm                    // w'<Mask> = accum (w, u'*A)
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector mask,          // optional mask for w, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for z=accum(w,t)
    const GrB_Semiring semiring,    // defines '+' and '*' for u'*A
    const GrB_Vector u,             // first input:  vector u
    const GrB_Matrix A,             // second input: matrix A
    const GrB_Descriptor desc       // descriptor for w, mask, and A
) ;

GrB_Info GrB_mxv                    // w<Mask> = accum (w, A*u)
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector mask,          // optional mask for w, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for z=accum(w,t)
    const GrB_Semiring semiring,    // defines '+' and '*' for A*B
    const GrB_Matrix A,             // first input:  matrix A
    const GrB_Vector u,             // second input: vector u
    const GrB_Descriptor desc       // descriptor for w, mask, and A
) ;

//==============================================================================
// GrB_eWiseMult: element-wise matrix and vector operations, set intersection
//==============================================================================

// GrB_eWiseMult computes C<Mask> = accum (C, A.*B), where ".*" is the Hadamard
// product, and where pairs of elements in two matrices (or vectors) are
// pairwise "multiplied" with C(i,j) = mult (A(i,j),B(i,j)).

GrB_Info GrB_Vector_eWiseMult_Semiring       // w<Mask> = accum (w, u.*v)
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector mask,          // optional mask for w, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for z=accum(w,t)
    const GrB_Semiring semiring,    // defines '.*' for t=u.*v
    const GrB_Vector u,             // first input:  vector u
    const GrB_Vector v,             // second input: vector v
    const GrB_Descriptor desc       // descriptor for w and mask
) ;

GrB_Info GrB_Vector_eWiseMult_Monoid         // w<Mask> = accum (w, u.*v)
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector mask,          // optional mask for w, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for z=accum(w,t)
    const GrB_Monoid monoid,        // defines '.*' for t=u.*v
    const GrB_Vector u,             // first input:  vector u
    const GrB_Vector v,             // second input: vector v
    const GrB_Descriptor desc       // descriptor for w and mask
) ;

GrB_Info GrB_Vector_eWiseMult_BinaryOp       // w<Mask> = accum (w, u.*v)
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector mask,          // optional mask for w, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for z=accum(w,t)
    const GrB_BinaryOp mult,        // defines '.*' for t=u.*v
    const GrB_Vector u,             // first input:  vector u
    const GrB_Vector v,             // second input: vector v
    const GrB_Descriptor desc       // descriptor for w and mask
) ;

GrB_Info GrB_Matrix_eWiseMult_Semiring       // C<Mask> = accum (C, A.*B)
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix Mask,          // optional mask for C, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C,T)
    const GrB_Semiring semiring,    // defines '.*' for T=A.*B
    const GrB_Matrix A,             // first input:  matrix A
    const GrB_Matrix B,             // second input: matrix B
    const GrB_Descriptor desc       // descriptor for C, Mask, A, and B
) ;

GrB_Info GrB_Matrix_eWiseMult_Monoid         // C<Mask> = accum (C, A.*B)
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix Mask,          // optional mask for C, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C,T)
    const GrB_Monoid monoid,        // defines '.*' for T=A.*B
    const GrB_Matrix A,             // first input:  matrix A
    const GrB_Matrix B,             // second input: matrix B
    const GrB_Descriptor desc       // descriptor for C, Mask, A, and B
) ;

GrB_Info GrB_Matrix_eWiseMult_BinaryOp       // C<Mask> = accum (C, A.*B)
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix Mask,          // optional mask for C, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C,T)
    const GrB_BinaryOp mult,        // defines '.*' for T=A.*B
    const GrB_Matrix A,             // first input:  matrix A
    const GrB_Matrix B,             // second input: matrix B
    const GrB_Descriptor desc       // descriptor for C, Mask, A, and B
) ;

// All 6 of the above type-specific functions are captured in a single
// type-generic function, GrB_eWiseMult:

#if GxB_STDC_VERSION >= 201112L
#define GrB_eWiseMult(C,Mask,accum,op,A,B,desc)                              \
    _Generic                                                                 \
    (                                                                        \
        (C),                                                                 \
            GrB_Matrix :                                                     \
                _Generic                                                     \
                (                                                            \
                    (op),                                                    \
                              GrB_Semiring : GrB_Matrix_eWiseMult_Semiring , \
                              GrB_Monoid   : GrB_Matrix_eWiseMult_Monoid   , \
                              GrB_BinaryOp : GrB_Matrix_eWiseMult_BinaryOp   \
                ),                                                           \
            GrB_Vector :                                                     \
                _Generic                                                     \
                (                                                            \
                    (op),                                                    \
                              GrB_Semiring : GrB_Vector_eWiseMult_Semiring , \
                              GrB_Monoid   : GrB_Vector_eWiseMult_Monoid   , \
                              GrB_BinaryOp : GrB_Vector_eWiseMult_BinaryOp   \
                )                                                            \
    )                                                                        \
    (C, Mask, accum, op, A, B, desc)
#endif

//==============================================================================
// GrB_eWiseAdd: element-wise matrix and vector operations, set union
//==============================================================================

// GrB_eWiseAdd computes C<Mask> = accum (C, A+B), where pairs of elements in
// two matrices (or two vectors) are pairwise "added".

GrB_Info GrB_Vector_eWiseAdd_Semiring       // w<mask> = accum (w, u+v)
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector mask,          // optional mask for w, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for z=accum(w,t)
    const GrB_Semiring semiring,    // defines '+' for t=u+v
    const GrB_Vector u,             // first input:  vector u
    const GrB_Vector v,             // second input: vector v
    const GrB_Descriptor desc       // descriptor for w and mask
) ;

GrB_Info GrB_Vector_eWiseAdd_Monoid         // w<mask> = accum (w, u+v)
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector mask,          // optional mask for w, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for z=accum(w,t)
    const GrB_Monoid monoid,        // defines '+' for t=u+v
    const GrB_Vector u,             // first input:  vector u
    const GrB_Vector v,             // second input: vector v
    const GrB_Descriptor desc       // descriptor for w and mask
) ;

GrB_Info GrB_Vector_eWiseAdd_BinaryOp       // w<mask> = accum (w, u+v)
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector mask,          // optional mask for w, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for z=accum(w,t)
    const GrB_BinaryOp add,         // defines '+' for t=u+v
    const GrB_Vector u,             // first input:  vector u
    const GrB_Vector v,             // second input: vector v
    const GrB_Descriptor desc       // descriptor for w and mask
) ;

GrB_Info GrB_Matrix_eWiseAdd_Semiring       // C<Mask> = accum (C, A+B)
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix Mask,          // optional mask for C, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C,T)
    const GrB_Semiring semiring,    // defines '+' for T=A+B
    const GrB_Matrix A,             // first input:  matrix A
    const GrB_Matrix B,             // second input: matrix B
    const GrB_Descriptor desc       // descriptor for C, Mask, A, and B
) ;

GrB_Info GrB_Matrix_eWiseAdd_Monoid         // C<Mask> = accum (C, A+B)
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix Mask,          // optional mask for C, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C,T)
    const GrB_Monoid monoid,        // defines '+' for T=A+B
    const GrB_Matrix A,             // first input:  matrix A
    const GrB_Matrix B,             // second input: matrix B
    const GrB_Descriptor desc       // descriptor for C, Mask, A, and B
) ;

GrB_Info GrB_Matrix_eWiseAdd_BinaryOp       // C<Mask> = accum (C, A+B)
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix Mask,          // optional mask for C, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C,T)
    const GrB_BinaryOp add,         // defines '+' for T=A+B
    const GrB_Matrix A,             // first input:  matrix A
    const GrB_Matrix B,             // second input: matrix B
    const GrB_Descriptor desc       // descriptor for C, Mask, A, and B
) ;

#if GxB_STDC_VERSION >= 201112L
#define GrB_eWiseAdd(C,Mask,accum,op,A,B,desc)                              \
    _Generic                                                                \
    (                                                                       \
        (C),                                                                \
            GrB_Matrix :                                                    \
                _Generic                                                    \
                (                                                           \
                    (op),                                                   \
                              GrB_Semiring : GrB_Matrix_eWiseAdd_Semiring , \
                              GrB_Monoid   : GrB_Matrix_eWiseAdd_Monoid   , \
                              GrB_BinaryOp : GrB_Matrix_eWiseAdd_BinaryOp   \
                ),                                                          \
            GrB_Vector :                                                    \
                _Generic                                                    \
                (                                                           \
                    (op),                                                   \
                              GrB_Semiring : GrB_Vector_eWiseAdd_Semiring , \
                              GrB_Monoid   : GrB_Vector_eWiseAdd_Monoid   , \
                              GrB_BinaryOp : GrB_Vector_eWiseAdd_BinaryOp   \
                )                                                           \
    )                                                                       \
    (C, Mask, accum, op, A, B, desc)
#endif

//==============================================================================
// GxB_eWiseUnion: a variant of GrB_eWiseAdd
//==============================================================================

// GxB_eWiseUnion is a variant of eWiseAdd.  The methods create a result with
// the same sparsity structure.  They differ when an entry is present in A but
// not B, or in B but not A.

// eWiseAdd does the following, for a matrix, where "+" is the add binary op:

//      if A(i,j) and B(i,j) are both present:
//          C(i,j) = A(i,j) + B(i,j)
//      else if A(i,j) is present but not B(i,j)
//          C(i,j) = A(i,j)
//      else if B(i,j) is present but not A(i,j)
//          C(i,j) = B(i,j)

// by contrast, eWiseUnion always applies the operator:

//      if A(i,j) and B(i,j) are both present:
//          C(i,j) = A(i,j) + B(i,j)
//      else if A(i,j) is present but not B(i,j)
//          C(i,j) = A(i,j) + beta
//      else if B(i,j) is present but not A(i,j)
//          C(i,j) = alpha + B(i,j)

GrB_Info GxB_Vector_eWiseUnion      // w<mask> = accum (w, u+v)
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector mask,          // optional mask for w, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for z=accum(w,t)
    const GrB_BinaryOp add,         // defines '+' for t=u+v
    const GrB_Vector u,             // first input:  vector u
    const GrB_Scalar alpha,
    const GrB_Vector v,             // second input: vector v
    const GrB_Scalar beta,
    const GrB_Descriptor desc       // descriptor for w and mask
) ;

GrB_Info GxB_Matrix_eWiseUnion      // C<M> = accum (C, A+B)
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix Mask,          // optional mask for C, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C,T)
    const GrB_BinaryOp add,         // defines '+' for T=A+B
    const GrB_Matrix A,             // first input:  matrix A
    const GrB_Scalar alpha,
    const GrB_Matrix B,             // second input: matrix B
    const GrB_Scalar beta,
    const GrB_Descriptor desc       // descriptor for C, M, A, and B
) ;

#if GxB_STDC_VERSION >= 201112L
#define GxB_eWiseUnion(C,Mask,accum,op,A,alpha,B,beta,desc)                 \
    _Generic                                                                \
    (                                                                       \
        (C),                                                                \
                  GrB_Matrix : GxB_Matrix_eWiseUnion ,                      \
                  GrB_Vector : GxB_Vector_eWiseUnion                        \
    )                                                                       \
    (C, Mask, accum, op, A, alpha, B, beta, desc)
#endif

//==============================================================================
// GrB_extract: extract a submatrix or subvector
//==============================================================================

GrB_Info GrB_Vector_extract         // w<mask> = accum (w, u(I))
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector mask,          // optional mask for w, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for z=accum(w,t)
    const GrB_Vector u,             // first input:  vector u
    const GrB_Index *Ilist,         // row indices
    GrB_Index ni,                   // number of row indices
    const GrB_Descriptor desc       // descriptor for w and mask
) ;

GrB_Info GrB_Matrix_extract         // C<Mask> = accum (C, A(I,J))
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix Mask,          // optional mask for C, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C,T)
    const GrB_Matrix A,             // first input:  matrix A
    const GrB_Index *Ilist,         // row indices
    GrB_Index ni,                   // number of row indices
    const GrB_Index *J,             // column indices
    GrB_Index nj,                   // number of column indices
    const GrB_Descriptor desc       // descriptor for C, Mask, and A
) ;

GrB_Info GrB_Col_extract            // w<mask> = accum (w, A(I,j))
(
    GrB_Vector w,                   // input/output matrix for results
    const GrB_Vector mask,          // optional mask for w, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for z=accum(w,t)
    const GrB_Matrix A,             // first input:  matrix A
    const GrB_Index *Ilist,         // row indices
    GrB_Index ni,                   // number of row indices
    GrB_Index j,                    // column index
    const GrB_Descriptor desc       // descriptor for w, mask, and A
) ;

// GrB_extract is a generic interface to the following functions:

// GrB_Vector_extract (w,mask,acc,u,I,ni,d)      // w<m>    = acc (w, u(I))
// GrB_Col_extract    (w,mask,acc,A,I,ni,j,d)    // w<m>    = acc (w, A(I,j))
// GrB_Matrix_extract (C,Mask,acc,A,I,ni,J,nj,d) // C<Mask> = acc (C, A(I,J))

#if GxB_STDC_VERSION >= 201112L
#define GrB_extract(arg1,Mask,accum,arg4,...) \
    _Generic                                                    \
    (                                                           \
        (arg1),                                                 \
            GrB_Vector :                                        \
                _Generic                                        \
                (                                               \
                    (arg4),                                     \
                              GrB_Vector : GrB_Vector_extract , \
                              GrB_Matrix : GrB_Col_extract      \
                ),                                              \
            GrB_Matrix : GrB_Matrix_extract                     \
    )                                                           \
    (arg1, Mask, accum, arg4, __VA_ARGS__)
#endif

//==============================================================================
// GxB_subassign: matrix and vector subassign: C(I,J)<Mask> = accum (C(I,J), A)
//==============================================================================

// Assign entries in a matrix or vector; C(I,J) = A.

// Each GxB_subassign function is very similar to its corresponding GrB_assign
// function in the spec, but they differ in two ways: (1) the mask in
// GxB_subassign has the same size as w(I) for vectors and C(I,J) for matrices,
// and (2) they differ in the GrB_REPLACE option.  See the user guide for
// details.

// In GraphBLAS notation, the two methods can be described as follows:

// matrix and vector subassign: C(I,J)<Mask> = accum (C(I,J), A)
// matrix and vector    assign: C<Mask>(I,J) = accum (C(I,J), A)

// --- assign ------------------------------------------------------------------
//
// GrB_Matrix_assign      C<M>(I,J) += A        M same size as matrix C.
//                                              A is |I|-by-|J|
//
// GrB_Vector_assign      w<m>(I)   += u        m same size as column vector w.
//                                              u is |I|-by-1
//
// GrB_Row_assign         C<m'>(i,J) += u'      m is a column vector the same
//                                              size as a row of C.
//                                              u is |J|-by-1, i is a scalar.
//
// GrB_Col_assign         C<m>(I,j) += u        m is a column vector the same
//                                              size as a column of C.
//                                              u is |I|-by-1, j is a scalar.
//
// --- subassign ---------------------------------------------------------------
//
// GxB_Matrix_subassign   C(I,J)<M> += A        M same size as matrix A.
//                                              A is |I|-by-|J|
//
// GxB_Vector_subassign   w(I)<m>   += u        m same size as column vector u.
//                                              u is |I|-by-1
//
// GxB_Row_subassign      C(i,J)<m'> += u'      m same size as column vector u.
//                                              u is |J|-by-1, i is a scalar.
//
// GxB_Col_subassign      C(I,j)<m> += u        m same size as column vector u.
//                                              u is |I|-by-1, j is a scalar.

GrB_Info GxB_Vector_subassign       // w(I)<mask> = accum (w(I),u)
(
    GrB_Vector w,                   // input/output matrix for results
    const GrB_Vector mask,          // optional mask for w(I), unused if NULL
    const GrB_BinaryOp accum,       // optional accum for z=accum(w(I),t)
    const GrB_Vector u,             // first input:  vector u
    const GrB_Index *Ilist,         // row indices
    GrB_Index ni,                   // number of row indices
    const GrB_Descriptor desc       // descriptor for w(I) and mask
) ;

GrB_Info GxB_Matrix_subassign       // C(I,J)<Mask> = accum (C(I,J),A)
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix Mask,          // optional mask for C(I,J), unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C(I,J),T)
    const GrB_Matrix A,             // first input:  matrix A
    const GrB_Index *Ilist,         // row indices
    GrB_Index ni,                   // number of row indices
    const GrB_Index *J,             // column indices
    GrB_Index nj,                   // number of column indices
    const GrB_Descriptor desc       // descriptor for C(I,J), Mask, and A
) ;

GrB_Info GxB_Col_subassign          // C(I,j)<mask> = accum (C(I,j),u)
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Vector mask,          // optional mask for C(I,j), unused if NULL
    const GrB_BinaryOp accum,       // optional accum for z=accum(C(I,j),t)
    const GrB_Vector u,             // input vector
    const GrB_Index *Ilist,         // row indices
    GrB_Index ni,                   // number of row indices
    GrB_Index j,                    // column index
    const GrB_Descriptor desc       // descriptor for C(I,j) and mask
) ;

GrB_Info GxB_Row_subassign          // C(i,J)<mask'> = accum (C(i,J),u')
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Vector mask,          // optional mask for C(i,J), unused if NULL
    const GrB_BinaryOp accum,       // optional accum for z=accum(C(i,J),t)
    const GrB_Vector u,             // input vector
    GrB_Index i,                    // row index
    const GrB_Index *J,             // column indices
    GrB_Index nj,                   // number of column indices
    const GrB_Descriptor desc       // descriptor for C(i,J) and mask
) ;

//------------------------------------------------------------------------------
// GxB_Vector_subassign_[SCALAR]:  scalar expansion assignment to subvector
//------------------------------------------------------------------------------

// Assigns a single scalar to a subvector, w(I)<mask> = accum(w(I),x).  The
// scalar x is implicitly expanded into a vector u of size ni-by-1, with each
// entry in u equal to x, and then w(I)<mask> = accum(w(I),u) is done.

GrB_Info GxB_Vector_subassign_BOOL  // w(I)<mask> = accum (w(I),x)
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector mask,          // optional mask for w(I), unused if NULL
    const GrB_BinaryOp accum,       // optional accum for z=accum(w(I),x)
    bool x,                         // scalar to assign to w(I)
    const GrB_Index *Ilist,         // row indices
    GrB_Index ni,                   // number of row indices
    const GrB_Descriptor desc       // descriptor for w(I) and mask
) ;

GrB_Info GxB_Vector_subassign_INT8  // w(I)<mask> = accum (w(I),x)
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector mask,          // optional mask for w(I), unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(w(I),x)
    int8_t x,                       // scalar to assign to w(I)
    const GrB_Index *Ilist,         // row indices
    GrB_Index ni,                   // number of row indices
    const GrB_Descriptor desc       // descriptor for w(I) and mask
) ;

GrB_Info GxB_Vector_subassign_UINT8 // w(I)<mask> = accum (w(I),x)
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector mask,          // optional mask for w(I), unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(w(I),x)
    uint8_t x,                      // scalar to assign to w(I)
    const GrB_Index *Ilist,         // row indices
    GrB_Index ni,                   // number of row indices
    const GrB_Descriptor desc       // descriptor for w(I) and mask
) ;

GrB_Info GxB_Vector_subassign_INT16 // w(I)<mask> = accum (w(I),x)
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector mask,          // optional mask for w(I), unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(w(I),x)
    int16_t x,                      // scalar to assign to w(I)
    const GrB_Index *Ilist,         // row indices
    GrB_Index ni,                   // number of row indices
    const GrB_Descriptor desc       // descriptor for w(I) and mask
) ;

GrB_Info GxB_Vector_subassign_UINT16   // w(I)<mask> = accum (w(I),x)
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector mask,          // optional mask for w(I), unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(w(I),x)
    uint16_t x,                     // scalar to assign to w(I)
    const GrB_Index *Ilist,         // row indices
    GrB_Index ni,                   // number of row indices
    const GrB_Descriptor desc       // descriptor for w(I) and mask
) ;

GrB_Info GxB_Vector_subassign_INT32    // w(I)<mask> = accum (w(I),x)
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector mask,          // optional mask for w(I), unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(w(I),x)
    int32_t x,                      // scalar to assign to w(I)
    const GrB_Index *Ilist,         // row indices
    GrB_Index ni,                   // number of row indices
    const GrB_Descriptor desc       // descriptor for w(I) and mask
) ;

GrB_Info GxB_Vector_subassign_UINT32   // w(I)<mask> = accum (w(I),x)
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector mask,          // optional mask for w(I), unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(w(I),x)
    uint32_t x,                     // scalar to assign to w(I)
    const GrB_Index *Ilist,         // row indices
    GrB_Index ni,                   // number of row indices
    const GrB_Descriptor desc       // descriptor for w(I) and mask
) ;

GrB_Info GxB_Vector_subassign_INT64    // w(I)<mask> = accum (w(I),x)
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector mask,          // optional mask for w(I), unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(w(I),x)
    int64_t x,                      // scalar to assign to w(I)
    const GrB_Index *Ilist,         // row indices
    GrB_Index ni,                   // number of row indices
    const GrB_Descriptor desc       // descriptor for w(I) and mask
) ;

GrB_Info GxB_Vector_subassign_UINT64   // w(I)<mask> = accum (w(I),x)
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector mask,          // optional mask for w(I), unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(w(I),x)
    uint64_t x,                     // scalar to assign to w(I)
    const GrB_Index *Ilist,         // row indices
    GrB_Index ni,                   // number of row indices
    const GrB_Descriptor desc       // descriptor for w(I) and mask
) ;

GrB_Info GxB_Vector_subassign_FP32     // w(I)<mask> = accum (w(I),x)
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector mask,          // optional mask for w(I), unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(w(I),x)
    float x,                        // scalar to assign to w(I)
    const GrB_Index *Ilist,         // row indices
    GrB_Index ni,                   // number of row indices
    const GrB_Descriptor desc       // descriptor for w(I) and mask
) ;

GrB_Info GxB_Vector_subassign_FP64     // w(I)<mask> = accum (w(I),x)
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector mask,          // optional mask for w(I), unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(w(I),x)
    double x,                       // scalar to assign to w(I)
    const GrB_Index *Ilist,         // row indices
    GrB_Index ni,                   // number of row indices
    const GrB_Descriptor desc       // descriptor for w(I) and mask
) ;

GrB_Info GxB_Vector_subassign_FC32     // w(I)<mask> = accum (w(I),x)
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector mask,          // optional mask for w(I), unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(w(I),x)
    GxB_FC32_t x,                   // scalar to assign to w(I)
    const GrB_Index *Ilist,         // row indices
    GrB_Index ni,                   // number of row indices
    const GrB_Descriptor desc       // descriptor for w(I) and mask
) ;

GrB_Info GxB_Vector_subassign_FC64     // w(I)<mask> = accum (w(I),x)
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector mask,          // optional mask for w(I), unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(w(I),x)
    GxB_FC64_t x,                   // scalar to assign to w(I)
    const GrB_Index *Ilist,         // row indices
    GrB_Index ni,                   // number of row indices
    const GrB_Descriptor desc       // descriptor for w(I) and mask
) ;

GrB_Info GxB_Vector_subassign_UDT      // w(I)<mask> = accum (w(I),x)
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector mask,          // optional mask for w(I), unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(w(I),x)
    void *x,                        // scalar to assign to w(I)
    const GrB_Index *Ilist,         // row indices
    GrB_Index ni,                   // number of row indices
    const GrB_Descriptor desc       // descriptor for w(I) and mask
) ;

GrB_Info GxB_Vector_subassign_Scalar   // w(I)<mask> = accum (w(I),x)
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector mask,          // optional mask for w(I), unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(w(I),x)
    GrB_Scalar x,                   // scalar to assign to w(I)
    const GrB_Index *Ilist,         // row indices
    GrB_Index ni,                   // number of row indices
    const GrB_Descriptor desc       // descriptor for w(I) and mask
) ;

//------------------------------------------------------------------------------
// GxB_Matrix_subassign_[SCALAR]:  scalar expansion assignment to submatrix
//------------------------------------------------------------------------------

// Assigns a single scalar to a submatrix, C(I,J)<Mask> = accum(C(I,J),x).  The
// scalar x is implicitly expanded into a matrix A of size ni-by-nj, with each
// entry in A equal to x, and then C(I,J)<Mask> = accum(C(I,J),A) is done.

GrB_Info GxB_Matrix_subassign_BOOL  // C(I,J)<Mask> = accum (C(I,J),x)
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix Mask,          // optional mask for C(I,J), unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C(I,J),x)
    bool x,                         // scalar to assign to C(I,J)
    const GrB_Index *Ilist,         // row indices
    GrB_Index ni,                   // number of row indices
    const GrB_Index *J,             // column indices
    GrB_Index nj,                   // number of column indices
    const GrB_Descriptor desc       // descriptor for C(I,J) and Mask
) ;

GrB_Info GxB_Matrix_subassign_INT8  // C(I,J)<Mask> = accum (C(I,J),x)
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix Mask,          // optional mask for C(I,J), unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C(I,J),x)
    int8_t x,                       // scalar to assign to C(I,J)
    const GrB_Index *Ilist,         // row indices
    GrB_Index ni,                   // number of row indices
    const GrB_Index *J,             // column indices
    GrB_Index nj,                   // number of column indices
    const GrB_Descriptor desc       // descriptor for C(I,J) and Mask
) ;

GrB_Info GxB_Matrix_subassign_UINT8 // C(I,J)<Mask> = accum (C(I,J),x)
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix Mask,          // optional mask for C(I,J), unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C(I,J),x)
    uint8_t x,                      // scalar to assign to C(I,J)
    const GrB_Index *Ilist,         // row indices
    GrB_Index ni,                   // number of row indices
    const GrB_Index *J,             // column indices
    GrB_Index nj,                   // number of column indices
    const GrB_Descriptor desc       // descriptor for C(I,J) and Mask
) ;

GrB_Info GxB_Matrix_subassign_INT16 // C(I,J)<Mask> = accum (C(I,J),x)
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix Mask,          // optional mask for C(I,J), unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C(I,J),x)
    int16_t x,                      // scalar to assign to C(I,J)
    const GrB_Index *Ilist,         // row indices
    GrB_Index ni,                   // number of row indices
    const GrB_Index *J,             // column indices
    GrB_Index nj,                   // number of column indices
    const GrB_Descriptor desc       // descriptor for C(I,J) and Mask
) ;

GrB_Info GxB_Matrix_subassign_UINT16   // C(I,J)<Mask> = accum (C(I,J),x)
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix Mask,          // optional mask for C(I,J), unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C(I,J),x)
    uint16_t x,                     // scalar to assign to C(I,J)
    const GrB_Index *Ilist,         // row indices
    GrB_Index ni,                   // number of row indices
    const GrB_Index *J,             // column indices
    GrB_Index nj,                   // number of column indices
    const GrB_Descriptor desc       // descriptor for C(I,J) and Mask
) ;

GrB_Info GxB_Matrix_subassign_INT32    // C(I,J)<Mask> = accum (C(I,J),x)
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix Mask,          // optional mask for C(I,J), unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C(I,J),x)
    int32_t x,                      // scalar to assign to C(I,J)
    const GrB_Index *Ilist,         // row indices
    GrB_Index ni,                   // number of row indices
    const GrB_Index *J,             // column indices
    GrB_Index nj,                   // number of column indices
    const GrB_Descriptor desc       // descriptor for C(I,J) and Mask
) ;

GrB_Info GxB_Matrix_subassign_UINT32   // C(I,J)<Mask> = accum (C(I,J),x)
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix Mask,          // optional mask for C(I,J), unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C(I,J),x)
    uint32_t x,                     // scalar to assign to C(I,J)
    const GrB_Index *Ilist,         // row indices
    GrB_Index ni,                   // number of row indices
    const GrB_Index *J,             // column indices
    GrB_Index nj,                   // number of column indices
    const GrB_Descriptor desc       // descriptor for C(I,J) and Mask
) ;

GrB_Info GxB_Matrix_subassign_INT64    // C(I,J)<Mask> = accum (C(I,J),x)
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix Mask,          // optional mask for C(I,J), unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C(I,J),x)
    int64_t x,                      // scalar to assign to C(I,J)
    const GrB_Index *Ilist,         // row indices
    GrB_Index ni,                   // number of row indices
    const GrB_Index *J,             // column indices
    GrB_Index nj,                   // number of column indices
    const GrB_Descriptor desc       // descriptor for C(I,J) and Mask
) ;

GrB_Info GxB_Matrix_subassign_UINT64   // C(I,J)<Mask> = accum (C(I,J),x)
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix Mask,          // optional mask for C(I,J), unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C(I,J),x)
    uint64_t x,                     // scalar to assign to C(I,J)
    const GrB_Index *Ilist,         // row indices
    GrB_Index ni,                   // number of row indices
    const GrB_Index *J,             // column indices
    GrB_Index nj,                   // number of column indices
    const GrB_Descriptor desc       // descriptor for C(I,J) and Mask
) ;

GrB_Info GxB_Matrix_subassign_FP32     // C(I,J)<Mask> = accum (C(I,J),x)
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix Mask,          // optional mask for C(I,J), unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C(I,J),x)
    float x,                        // scalar to assign to C(I,J)
    const GrB_Index *Ilist,         // row indices
    GrB_Index ni,                   // number of row indices
    const GrB_Index *J,             // column indices
    GrB_Index nj,                   // number of column indices
    const GrB_Descriptor desc       // descriptor for C(I,J) and Mask
) ;

GrB_Info GxB_Matrix_subassign_FP64     // C(I,J)<Mask> = accum (C(I,J),x)
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix Mask,          // optional mask for C(I,J), unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C(I,J),x)
    double x,                       // scalar to assign to C(I,J)
    const GrB_Index *Ilist,         // row indices
    GrB_Index ni,                   // number of row indices
    const GrB_Index *J,             // column indices
    GrB_Index nj,                   // number of column indices
    const GrB_Descriptor desc       // descriptor for C(I,J) and Mask
) ;

GrB_Info GxB_Matrix_subassign_FC32     // C(I,J)<Mask> = accum (C(I,J),x)
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix Mask,          // optional mask for C(I,J), unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C(I,J),x)
    GxB_FC32_t x,                   // scalar to assign to C(I,J)
    const GrB_Index *Ilist,         // row indices
    GrB_Index ni,                   // number of row indices
    const GrB_Index *J,             // column indices
    GrB_Index nj,                   // number of column indices
    const GrB_Descriptor desc       // descriptor for C(I,J) and Mask
) ;

GrB_Info GxB_Matrix_subassign_FC64     // C(I,J)<Mask> = accum (C(I,J),x)
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix Mask,          // optional mask for C(I,J), unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C(I,J),x)
    GxB_FC64_t x,                   // scalar to assign to C(I,J)
    const GrB_Index *Ilist,         // row indices
    GrB_Index ni,                   // number of row indices
    const GrB_Index *J,             // column indices
    GrB_Index nj,                   // number of column indices
    const GrB_Descriptor desc       // descriptor for C(I,J) and Mask
) ;

GrB_Info GxB_Matrix_subassign_UDT      // C(I,J)<Mask> = accum (C(I,J),x)
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix Mask,          // optional mask for C(I,J), unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C(I,J),x)
    void *x,                        // scalar to assign to C(I,J)
    const GrB_Index *Ilist,         // row indices
    GrB_Index ni,                   // number of row indices
    const GrB_Index *J,             // column indices
    GrB_Index nj,                   // number of column indices
    const GrB_Descriptor desc       // descriptor for C(I,J) and Mask
) ;

GrB_Info GxB_Matrix_subassign_Scalar   // C(I,J)<Mask> = accum (C(I,J),x)
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix Mask,          // optional mask for C(I,J), unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C(I,J),x)
    GrB_Scalar x,                   // scalar to assign to C(I,J)
    const GrB_Index *Ilist,         // row indices
    GrB_Index ni,                   // number of row indices
    const GrB_Index *J,             // column indices
    GrB_Index nj,                   // number of column indices
    const GrB_Descriptor desc       // descriptor for C(I,J) and Mask
) ;

//------------------------------------------------------------------------------
// GxB_subassign: generic submatrix/subvector assignment
//------------------------------------------------------------------------------

// GxB_subassign is a generic function that provides access to all specific
// GxB_*_subassign* functions:

// GxB_Vector_subassign   (w,m,acc,u,I,ni,d)      // w(I)<m>    = acc(w(I),u)
// GxB_Matrix_subassign   (C,M,acc,A,I,ni,J,nj,d) // C(I,J)<M>  = acc(C(I,J),A)
// GxB_Col_subassign      (C,m,acc,u,I,ni,j,d)    // C(I,j)<m>  = acc(C(I,j),u)
// GxB_Row_subassign      (C,m,acc,u,i,J,nj,d)    // C(i,J)<m'> = acc(C(i,J),u')
// GxB_Vector_subassign_T (w,m,acc,x,I,ni,d)      // w(I)<m>    = acc(w(I),x)
// GxB_Matrix_subassign_T (C,M,acc,x,I,ni,J,nj,d) // C(I,J)<M>  = acc(C(I,J),x)

#if GxB_STDC_VERSION >= 201112L
#define GxB_subassign(arg1,Mask,accum,arg4,arg5,...)                    \
    _Generic                                                            \
    (                                                                   \
        (arg1),                                                         \
        GrB_Vector :                                                    \
            _Generic                                                    \
            (                                                           \
                (arg4),                                                 \
                    GB_CASES (GxB, Vector_subassign) ,                  \
                    GrB_Scalar : GxB_Vector_subassign_Scalar,           \
                    default:  GxB_Vector_subassign                      \
            ),                                                          \
        default:                                                        \
            _Generic                                                    \
            (                                                           \
                (arg4),                                                 \
                    GB_CASES (GxB, Matrix_subassign) ,                  \
                    GrB_Scalar : GxB_Matrix_subassign_Scalar,           \
                    GrB_Vector :                                        \
                        _Generic                                        \
                        (                                               \
                            (arg5),                                     \
                                const GrB_Index *: GxB_Col_subassign ,  \
                                      GrB_Index *: GxB_Col_subassign ,  \
                                default:           GxB_Row_subassign    \
                        ),                                              \
                    default:  GxB_Matrix_subassign                      \
            )                                                           \
    )                                                                   \
    (arg1, Mask, accum, arg4, arg5, __VA_ARGS__)
#endif

//==============================================================================
// GrB_assign: matrix and vector assign: C<Mask>(I,J) = accum (C(I,J), A)
//==============================================================================

// Assign entries in a matrix or vector; C(I,J) = A.
// Each of these can be used with their generic name, GrB_assign.

GrB_Info GrB_Vector_assign          // w<mask>(I) = accum (w(I),u)
(
    GrB_Vector w,                   // input/output matrix for results
    const GrB_Vector mask,          // optional mask for w, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for z=accum(w(I),t)
    const GrB_Vector u,             // first input:  vector u
    const GrB_Index *Ilist,         // row indices
    GrB_Index ni,                   // number of row indices
    const GrB_Descriptor desc       // descriptor for w and mask
) ;

GrB_Info GrB_Matrix_assign          // C<Mask>(I,J) = accum (C(I,J),A)
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix Mask,          // optional mask for C, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C(I,J),T)
    const GrB_Matrix A,             // first input:  matrix A
    const GrB_Index *Ilist,         // row indices
    GrB_Index ni,                   // number of row indices
    const GrB_Index *J,             // column indices
    GrB_Index nj,                   // number of column indices
    const GrB_Descriptor desc       // descriptor for C, Mask, and A
) ;

GrB_Info GrB_Col_assign             // C<mask>(I,j) = accum (C(I,j),u)
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Vector mask,          // optional mask for C(:,j), unused if NULL
    const GrB_BinaryOp accum,       // optional accum for z=accum(C(I,j),t)
    const GrB_Vector u,             // input vector
    const GrB_Index *Ilist,         // row indices
    GrB_Index ni,                   // number of row indices
    GrB_Index j,                    // column index
    const GrB_Descriptor desc       // descriptor for C(:,j) and mask
) ;

GrB_Info GrB_Row_assign             // C<mask'>(i,J) = accum (C(i,J),u')
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Vector mask,          // optional mask for C(i,:), unused if NULL
    const GrB_BinaryOp accum,       // optional accum for z=accum(C(i,J),t)
    const GrB_Vector u,             // input vector
    GrB_Index i,                    // row index
    const GrB_Index *J,             // column indices
    GrB_Index nj,                   // number of column indices
    const GrB_Descriptor desc       // descriptor for C(i,:) and mask
) ;

//------------------------------------------------------------------------------
// GrB_Vector_assign_[SCALAR]:  scalar expansion assignment to subvector
//------------------------------------------------------------------------------

// Assigns a single scalar to a subvector, w<mask>(I) = accum(w(I),x).  The
// scalar x is implicitly expanded into a vector u of size ni-by-1, with each
// entry in u equal to x, and then w<mask>(I) = accum(w(I),u) is done.

GrB_Info GrB_Vector_assign_BOOL     // w<mask>(I) = accum (w(I),x)
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector mask,          // optional mask for w, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for z=accum(w(I),x)
    bool x,                         // scalar to assign to w(I)
    const GrB_Index *Ilist,         // row indices
    GrB_Index ni,                   // number of row indices
    const GrB_Descriptor desc       // descriptor for w and mask
) ;

GrB_Info GrB_Vector_assign_INT8     // w<mask>(I) = accum (w(I),x)
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector mask,          // optional mask for w, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(w(I),x)
    int8_t x,                       // scalar to assign to w(I)
    const GrB_Index *Ilist,         // row indices
    GrB_Index ni,                   // number of row indices
    const GrB_Descriptor desc       // descriptor for w and mask
) ;

GrB_Info GrB_Vector_assign_UINT8    // w<mask>(I) = accum (w(I),x)
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector mask,          // optional mask for w, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(w(I),x)
    uint8_t x,                      // scalar to assign to w(I)
    const GrB_Index *Ilist,         // row indices
    GrB_Index ni,                   // number of row indices
    const GrB_Descriptor desc       // descriptor for w and mask
) ;

GrB_Info GrB_Vector_assign_INT16    // w<mask>(I) = accum (w(I),x)
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector mask,          // optional mask for w, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(w(I),x)
    int16_t x,                      // scalar to assign to w(I)
    const GrB_Index *Ilist,         // row indices
    GrB_Index ni,                   // number of row indices
    const GrB_Descriptor desc       // descriptor for w and mask
) ;

GrB_Info GrB_Vector_assign_UINT16   // w<mask>(I) = accum (w(I),x)
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector mask,          // optional mask for w, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(w(I),x)
    uint16_t x,                     // scalar to assign to w(I)
    const GrB_Index *Ilist,         // row indices
    GrB_Index ni,                   // number of row indices
    const GrB_Descriptor desc       // descriptor for w and mask
) ;

GrB_Info GrB_Vector_assign_INT32    // w<mask>(I) = accum (w(I),x)
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector mask,          // optional mask for w, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(w(I),x)
    int32_t x,                      // scalar to assign to w(I)
    const GrB_Index *Ilist,         // row indices
    GrB_Index ni,                   // number of row indices
    const GrB_Descriptor desc       // descriptor for w and mask
) ;

GrB_Info GrB_Vector_assign_UINT32   // w<mask>(I) = accum (w(I),x)
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector mask,          // optional mask for w, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(w(I),x)
    uint32_t x,                     // scalar to assign to w(I)
    const GrB_Index *Ilist,         // row indices
    GrB_Index ni,                   // number of row indices
    const GrB_Descriptor desc       // descriptor for w and mask
) ;

GrB_Info GrB_Vector_assign_INT64    // w<mask>(I) = accum (w(I),x)
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector mask,          // optional mask for w, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(w(I),x)
    int64_t x,                      // scalar to assign to w(I)
    const GrB_Index *Ilist,         // row indices
    GrB_Index ni,                   // number of row indices
    const GrB_Descriptor desc       // descriptor for w and mask
) ;

GrB_Info GrB_Vector_assign_UINT64   // w<mask>(I) = accum (w(I),x)
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector mask,          // optional mask for w, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(w(I),x)
    uint64_t x,                     // scalar to assign to w(I)
    const GrB_Index *Ilist,         // row indices
    GrB_Index ni,                   // number of row indices
    const GrB_Descriptor desc       // descriptor for w and mask
) ;

GrB_Info GrB_Vector_assign_FP32     // w<mask>(I) = accum (w(I),x)
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector mask,          // optional mask for w, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(w(I),x)
    float x,                        // scalar to assign to w(I)
    const GrB_Index *Ilist,         // row indices
    GrB_Index ni,                   // number of row indices
    const GrB_Descriptor desc       // descriptor for w and mask
) ;

GrB_Info GrB_Vector_assign_FP64     // w<mask>(I) = accum (w(I),x)
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector mask,          // optional mask for w, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(w(I),x)
    double x,                       // scalar to assign to w(I)
    const GrB_Index *Ilist,         // row indices
    GrB_Index ni,                   // number of row indices
    const GrB_Descriptor desc       // descriptor for w and mask
) ;

GrB_Info GxB_Vector_assign_FC32     // w<mask>(I) = accum (w(I),x)
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector mask,          // optional mask for w, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(w(I),x)
    GxB_FC32_t x,                   // scalar to assign to w(I)
    const GrB_Index *Ilist,         // row indices
    GrB_Index ni,                   // number of row indices
    const GrB_Descriptor desc       // descriptor for w and mask
) ;

GrB_Info GxB_Vector_assign_FC64     // w<mask>(I) = accum (w(I),x)
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector mask,          // optional mask for w, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(w(I),x)
    GxB_FC64_t x,                   // scalar to assign to w(I)
    const GrB_Index *Ilist,         // row indices
    GrB_Index ni,                   // number of row indices
    const GrB_Descriptor desc       // descriptor for w and mask
) ;

GrB_Info GrB_Vector_assign_UDT      // w<mask>(I) = accum (w(I),x)
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector mask,          // optional mask for w, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(w(I),x)
    void *x,                        // scalar to assign to w(I)
    const GrB_Index *Ilist,         // row indices
    GrB_Index ni,                   // number of row indices
    const GrB_Descriptor desc       // descriptor for w and mask
) ;

GrB_Info GrB_Vector_assign_Scalar   // w<mask>(I) = accum (w(I),x)
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector mask,          // optional mask for w, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(w(I),x)
    GrB_Scalar x,                   // scalar to assign to w(I)
    const GrB_Index *Ilist,         // row indices
    GrB_Index ni,                   // number of row indices
    const GrB_Descriptor desc       // descriptor for w and mask
) ;

//------------------------------------------------------------------------------
// GrB_Matrix_assign_[SCALAR]:  scalar expansion assignment to submatrix
//------------------------------------------------------------------------------

// Assigns a single scalar to a submatrix, C<Mask>(I,J) = accum(C(I,J),x).  The
// scalar x is implicitly expanded into a matrix A of size ni-by-nj, with each
// entry in A equal to x, and then C<Mask>(I,J) = accum(C(I,J),A) is done.

GrB_Info GrB_Matrix_assign_BOOL     // C<Mask>(I,J) = accum (C(I,J),x)
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix Mask,          // optional mask for C, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C(I,J),x)
    bool x,                         // scalar to assign to C(I,J)
    const GrB_Index *Ilist,         // row indices
    GrB_Index ni,                   // number of row indices
    const GrB_Index *J,             // column indices
    GrB_Index nj,                   // number of column indices
    const GrB_Descriptor desc       // descriptor for C and Mask
) ;

GrB_Info GrB_Matrix_assign_INT8     // C<Mask>(I,J) = accum (C(I,J),x)
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix Mask,          // optional mask for C, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C(I,J),x)
    int8_t x,                       // scalar to assign to C(I,J)
    const GrB_Index *Ilist,         // row indices
    GrB_Index ni,                   // number of row indices
    const GrB_Index *J,             // column indices
    GrB_Index nj,                   // number of column indices
    const GrB_Descriptor desc       // descriptor for C and Mask
) ;

GrB_Info GrB_Matrix_assign_UINT8    // C<Mask>(I,J) = accum (C(I,J),x)
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix Mask,          // optional mask for C, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C(I,J),x)
    uint8_t x,                      // scalar to assign to C(I,J)
    const GrB_Index *Ilist,         // row indices
    GrB_Index ni,                   // number of row indices
    const GrB_Index *J,             // column indices
    GrB_Index nj,                   // number of column indices
    const GrB_Descriptor desc       // descriptor for C and Mask
) ;

GrB_Info GrB_Matrix_assign_INT16    // C<Mask>(I,J) = accum (C(I,J),x)
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix Mask,          // optional mask for C, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C(I,J),x)
    int16_t x,                      // scalar to assign to C(I,J)
    const GrB_Index *Ilist,         // row indices
    GrB_Index ni,                   // number of row indices
    const GrB_Index *J,             // column indices
    GrB_Index nj,                   // number of column indices
    const GrB_Descriptor desc       // descriptor for C and Mask
) ;

GrB_Info GrB_Matrix_assign_UINT16   // C<Mask>(I,J) = accum (C(I,J),x)
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix Mask,          // optional mask for C, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C(I,J),x)
    uint16_t x,                     // scalar to assign to C(I,J)
    const GrB_Index *Ilist,         // row indices
    GrB_Index ni,                   // number of row indices
    const GrB_Index *J,             // column indices
    GrB_Index nj,                   // number of column indices
    const GrB_Descriptor desc       // descriptor for C and Mask
) ;

GrB_Info GrB_Matrix_assign_INT32    // C<Mask>(I,J) = accum (C(I,J),x)
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix Mask,          // optional mask for C, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C(I,J),x)
    int32_t x,                      // scalar to assign to C(I,J)
    const GrB_Index *Ilist,         // row indices
    GrB_Index ni,                   // number of row indices
    const GrB_Index *J,             // column indices
    GrB_Index nj,                   // number of column indices
    const GrB_Descriptor desc       // descriptor for C and Mask
) ;

GrB_Info GrB_Matrix_assign_UINT32   // C<Mask>(I,J) = accum (C(I,J),x)
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix Mask,          // optional mask for C, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C(I,J),x)
    uint32_t x,                     // scalar to assign to C(I,J)
    const GrB_Index *Ilist,         // row indices
    GrB_Index ni,                   // number of row indices
    const GrB_Index *J,             // column indices
    GrB_Index nj,                   // number of column indices
    const GrB_Descriptor desc       // descriptor for C and Mask
) ;

GrB_Info GrB_Matrix_assign_INT64    // C<Mask>(I,J) = accum (C(I,J),x)
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix Mask,          // optional mask for C, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C(I,J),x)
    int64_t x,                      // scalar to assign to C(I,J)
    const GrB_Index *Ilist,         // row indices
    GrB_Index ni,                   // number of row indices
    const GrB_Index *J,             // column indices
    GrB_Index nj,                   // number of column indices
    const GrB_Descriptor desc       // descriptor for C and Mask
) ;

GrB_Info GrB_Matrix_assign_UINT64   // C<Mask>(I,J) = accum (C(I,J),x)
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix Mask,          // optional mask for C, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C(I,J),x)
    uint64_t x,                     // scalar to assign to C(I,J)
    const GrB_Index *Ilist,         // row indices
    GrB_Index ni,                   // number of row indices
    const GrB_Index *J,             // column indices
    GrB_Index nj,                   // number of column indices
    const GrB_Descriptor desc       // descriptor for C and Mask
) ;

GrB_Info GrB_Matrix_assign_FP32     // C<Mask>(I,J) = accum (C(I,J),x)
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix Mask,          // optional mask for C, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C(I,J),x)
    float x,                        // scalar to assign to C(I,J)
    const GrB_Index *Ilist,         // row indices
    GrB_Index ni,                   // number of row indices
    const GrB_Index *J,             // column indices
    GrB_Index nj,                   // number of column indices
    const GrB_Descriptor desc       // descriptor for C and Mask
) ;

GrB_Info GrB_Matrix_assign_FP64     // C<Mask>(I,J) = accum (C(I,J),x)
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix Mask,          // optional mask for C, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C(I,J),x)
    double x,                       // scalar to assign to C(I,J)
    const GrB_Index *Ilist,         // row indices
    GrB_Index ni,                   // number of row indices
    const GrB_Index *J,             // column indices
    GrB_Index nj,                   // number of column indices
    const GrB_Descriptor desc       // descriptor for C and Mask
) ;

GrB_Info GxB_Matrix_assign_FC32     // C<Mask>(I,J) = accum (C(I,J),x)
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix Mask,          // optional mask for C, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C(I,J),x)
    GxB_FC32_t x,                   // scalar to assign to C(I,J)
    const GrB_Index *Ilist,         // row indices
    GrB_Index ni,                   // number of row indices
    const GrB_Index *J,             // column indices
    GrB_Index nj,                   // number of column indices
    const GrB_Descriptor desc       // descriptor for C and Mask
) ;

GrB_Info GxB_Matrix_assign_FC64     // C<Mask>(I,J) = accum (C(I,J),x)
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix Mask,          // optional mask for C, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C(I,J),x)
    GxB_FC64_t x,                   // scalar to assign to C(I,J)
    const GrB_Index *Ilist,         // row indices
    GrB_Index ni,                   // number of row indices
    const GrB_Index *J,             // column indices
    GrB_Index nj,                   // number of column indices
    const GrB_Descriptor desc       // descriptor for C and Mask
) ;

GrB_Info GrB_Matrix_assign_UDT      // C<Mask>(I,J) = accum (C(I,J),x)
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix Mask,          // optional mask for C, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C(I,J),x)
    void *x,                        // scalar to assign to C(I,J)
    const GrB_Index *Ilist,         // row indices
    GrB_Index ni,                   // number of row indices
    const GrB_Index *J,             // column indices
    GrB_Index nj,                   // number of column indices
    const GrB_Descriptor desc       // descriptor for C and Mask
) ;

GrB_Info GrB_Matrix_assign_Scalar   // C<Mask>(I,J) = accum (C(I,J),x)
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix Mask,          // optional mask for C, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C(I,J),x)
    GrB_Scalar x,                   // scalar to assign to C(I,J)
    const GrB_Index *Ilist,         // row indices
    GrB_Index ni,                   // number of row indices
    const GrB_Index *J,             // column indices
    GrB_Index nj,                   // number of column indices
    const GrB_Descriptor desc       // descriptor for C and Mask
) ;

//------------------------------------------------------------------------------
// GrB_assign: generic submatrix/subvector assignment
//------------------------------------------------------------------------------

// GrB_assign is a generic function that provides access to all specific
// GrB_*_assign* functions:

// GrB_Vector_assign_T (w,m,acc,x,I,ni,d)      // w<m>(I)    = acc(w(I),x)
// GrB_Vector_assign   (w,m,acc,u,I,ni,d)      // w<m>(I)    = acc(w(I),u)
// GrB_Matrix_assign_T (C,M,acc,x,I,ni,J,nj,d) // C<M>(I,J)  = acc(C(I,J),x)
// GrB_Col_assign      (C,m,acc,u,I,ni,j,d)    // C<m>(I,j)  = acc(C(I,j),u)
// GrB_Row_assign      (C,m,acc,u,i,J,nj,d)    // C<m'>(i,J) = acc(C(i,J),u')
// GrB_Matrix_assign   (C,M,acc,A,I,ni,J,nj,d) // C<M>(I,J)  = acc(C(I,J),A)

#if GxB_STDC_VERSION >= 201112L
#define GrB_assign(arg1,Mask,accum,arg4,arg5,...)                       \
    _Generic                                                            \
    (                                                                   \
        (arg1),                                                         \
            GrB_Vector :                                                \
                _Generic                                                \
                (                                                       \
                    (arg4),                                             \
                        GB_CASES (GrB, Vector_assign) ,                 \
                        GrB_Scalar : GrB_Vector_assign_Scalar ,         \
                        default:  GrB_Vector_assign                     \
                ),                                                      \
            default:                                                    \
                _Generic                                                \
                (                                                       \
                    (arg4),                                             \
                        GB_CASES (GrB, Matrix_assign) ,                 \
                        GrB_Scalar : GrB_Matrix_assign_Scalar ,         \
                        GrB_Vector :                                    \
                            _Generic                                    \
                            (                                           \
                                (arg5),                                 \
                                const GrB_Index *: GrB_Col_assign ,     \
                                      GrB_Index *: GrB_Col_assign ,     \
                                default:           GrB_Row_assign       \
                            ),                                          \
                        default:  GrB_Matrix_assign                     \
                )                                                       \
    )                                                                   \
    (arg1, Mask, accum, arg4, arg5, __VA_ARGS__)
#endif

//==============================================================================
// GrB_apply: matrix and vector apply
//==============================================================================

// Apply a unary, index_unary, or binary operator to entries in a matrix or
// vector, C<M> = accum (C, op (A)).

GrB_Info GrB_Vector_apply           // w<mask> = accum (w, op(u))
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector mask,          // optional mask for w, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for z=accum(w,t)
    const GrB_UnaryOp op,           // operator to apply to the entries
    const GrB_Vector u,             // first input:  vector u
    const GrB_Descriptor desc       // descriptor for w and mask
) ;

GrB_Info GrB_Matrix_apply           // C<Mask> = accum (C, op(A)) or op(A')
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix Mask,          // optional mask for C, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C,T)
    const GrB_UnaryOp op,           // operator to apply to the entries
    const GrB_Matrix A,             // first input:  matrix A
    const GrB_Descriptor desc       // descriptor for C, mask, and A
) ;

//-------------------------------------------
// vector apply: binaryop variants (bind 1st)
//-------------------------------------------

// Apply a binary operator to the entries in a vector, binding the first
// input to a scalar x, w<mask> = accum (w, op (x,u)).

GrB_Info GrB_Vector_apply_BinaryOp1st_Scalar    // w<mask> = accum (w, op(x,u))
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector mask,          // optional mask for w, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for z=accum(w,t)
    const GrB_BinaryOp op,          // operator to apply to the entries
    const GrB_Scalar x,             // first input:  scalar x
    const GrB_Vector u,             // second input: vector u
    const GrB_Descriptor desc       // descriptor for w and mask
) ;

// historical: identical to GxB_Vector_apply_BinaryOp1st
GrB_Info GxB_Vector_apply_BinaryOp1st           // w<mask> = accum (w, op(x,u))
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector mask,          // optional mask for w, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for z=accum(w,t)
    const GrB_BinaryOp op,          // operator to apply to the entries
    const GrB_Scalar x,             // first input:  scalar x
    const GrB_Vector u,             // second input: vector u
    const GrB_Descriptor desc       // descriptor for w and mask
) ;

GrB_Info GrB_Vector_apply_BinaryOp1st_BOOL      // w<mask> = accum (w, op(x,u))
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector mask,          // optional mask for w, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for z=accum(w,t)
    const GrB_BinaryOp op,          // operator to apply to the entries
    bool x,                         // first input:  scalar x
    const GrB_Vector u,             // second input: vector u
    const GrB_Descriptor desc       // descriptor for w and mask
) ;

GrB_Info GrB_Vector_apply_BinaryOp1st_INT8      // w<mask> = accum (w, op(x,u))
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector mask,          // optional mask for w, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for z=accum(w,t)
    const GrB_BinaryOp op,          // operator to apply to the entries
    int8_t x,                       // first input:  scalar x
    const GrB_Vector u,             // second input: vector u
    const GrB_Descriptor desc       // descriptor for w and mask
) ;

GrB_Info GrB_Vector_apply_BinaryOp1st_INT16     // w<mask> = accum (w, op(x,u))
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector mask,          // optional mask for w, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for z=accum(w,t)
    const GrB_BinaryOp op,          // operator to apply to the entries
    int16_t x,                      // first input:  scalar x
    const GrB_Vector u,             // second input: vector u
    const GrB_Descriptor desc       // descriptor for w and mask
) ;

GrB_Info GrB_Vector_apply_BinaryOp1st_INT32     // w<mask> = accum (w, op(x,u))
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector mask,          // optional mask for w, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for z=accum(w,t)
    const GrB_BinaryOp op,          // operator to apply to the entries
    int32_t x,                      // first input:  scalar x
    const GrB_Vector u,             // second input: vector u
    const GrB_Descriptor desc       // descriptor for w and mask
) ;

GrB_Info GrB_Vector_apply_BinaryOp1st_INT64     // w<mask> = accum (w, op(x,u))
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector mask,          // optional mask for w, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for z=accum(w,t)
    const GrB_BinaryOp op,          // operator to apply to the entries
    int64_t x,                      // first input:  scalar x
    const GrB_Vector u,             // second input: vector u
    const GrB_Descriptor desc       // descriptor for w and mask
) ;

GrB_Info GrB_Vector_apply_BinaryOp1st_UINT8     // w<mask> = accum (w, op(x,u))
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector mask,          // optional mask for w, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for z=accum(w,t)
    const GrB_BinaryOp op,          // operator to apply to the entries
    uint8_t x,                      // first input:  scalar x
    const GrB_Vector u,             // second input: vector u
    const GrB_Descriptor desc       // descriptor for w and mask
) ;

GrB_Info GrB_Vector_apply_BinaryOp1st_UINT16    // w<mask> = accum (w, op(x,u))
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector mask,          // optional mask for w, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for z=accum(w,t)
    const GrB_BinaryOp op,          // operator to apply to the entries
    uint16_t x,                     // first input:  scalar x
    const GrB_Vector u,             // second input: vector u
    const GrB_Descriptor desc       // descriptor for w and mask
) ;

GrB_Info GrB_Vector_apply_BinaryOp1st_UINT32    // w<mask> = accum (w, op(x,u))
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector mask,          // optional mask for w, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for z=accum(w,t)
    const GrB_BinaryOp op,          // operator to apply to the entries
    uint32_t x,                     // first input:  scalar x
    const GrB_Vector u,             // second input: vector u
    const GrB_Descriptor desc       // descriptor for w and mask
) ;

GrB_Info GrB_Vector_apply_BinaryOp1st_UINT64    // w<mask> = accum (w, op(x,u))
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector mask,          // optional mask for w, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for z=accum(w,t)
    const GrB_BinaryOp op,          // operator to apply to the entries
    uint64_t x,                     // first input:  scalar x
    const GrB_Vector u,             // second input: vector u
    const GrB_Descriptor desc       // descriptor for w and mask
) ;

GrB_Info GrB_Vector_apply_BinaryOp1st_FP32      // w<mask> = accum (w, op(x,u))
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector mask,          // optional mask for w, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for z=accum(w,t)
    const GrB_BinaryOp op,          // operator to apply to the entries
    float x,                        // first input:  scalar x
    const GrB_Vector u,             // second input: vector u
    const GrB_Descriptor desc       // descriptor for w and mask
) ;

GrB_Info GrB_Vector_apply_BinaryOp1st_FP64      // w<mask> = accum (w, op(x,u))
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector mask,          // optional mask for w, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for z=accum(w,t)
    const GrB_BinaryOp op,          // operator to apply to the entries
    double x,                       // first input:  scalar x
    const GrB_Vector u,             // second input: vector u
    const GrB_Descriptor desc       // descriptor for w and mask
) ;

GrB_Info GxB_Vector_apply_BinaryOp1st_FC32      // w<mask> = accum (w, op(x,u))
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector mask,          // optional mask for w, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for z=accum(w,t)
    const GrB_BinaryOp op,          // operator to apply to the entries
    GxB_FC32_t x,                   // first input:  scalar x
    const GrB_Vector u,             // second input: vector u
    const GrB_Descriptor desc       // descriptor for w and mask
) ;

GrB_Info GxB_Vector_apply_BinaryOp1st_FC64      // w<mask> = accum (w, op(x,u))
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector mask,          // optional mask for w, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for z=accum(w,t)
    const GrB_BinaryOp op,          // operator to apply to the entries
    GxB_FC64_t x,                   // first input:  scalar x
    const GrB_Vector u,             // second input: vector u
    const GrB_Descriptor desc       // descriptor for w and mask
) ;

GrB_Info GrB_Vector_apply_BinaryOp1st_UDT       // w<mask> = accum (w, op(x,u))
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector mask,          // optional mask for w, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for z=accum(w,t)
    const GrB_BinaryOp op,          // operator to apply to the entries
    const void *x,                  // first input:  scalar x
    const GrB_Vector u,             // second input: vector u
    const GrB_Descriptor desc       // descriptor for w and mask
) ;

//-------------------------------------------
// vector apply: binaryop variants (bind 2nd)
//-------------------------------------------

// Apply a binary operator to the entries in a vector, binding the second
// input to a scalar y, w<mask> = accum (w, op (u,y)).

GrB_Info GrB_Vector_apply_BinaryOp2nd_Scalar    // w<mask> = accum (w, op(u,y))
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector mask,          // optional mask for w, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for z=accum(w,t)
    const GrB_BinaryOp op,          // operator to apply to the entries
    const GrB_Vector u,             // first input:  vector u
    const GrB_Scalar y,             // second input: scalar y
    const GrB_Descriptor desc       // descriptor for w and mask
) ;

// historical: identical to GrB_Vector_apply_BinaryOp2nd_Scalar
GrB_Info GxB_Vector_apply_BinaryOp2nd           // w<mask> = accum (w, op(u,y))
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector mask,          // optional mask for w, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for z=accum(w,t)
    const GrB_BinaryOp op,          // operator to apply to the entries
    const GrB_Vector u,             // first input:  vector u
    const GrB_Scalar y,             // second input: scalar y
    const GrB_Descriptor desc       // descriptor for w and mask
) ;

GrB_Info GrB_Vector_apply_BinaryOp2nd_BOOL      // w<mask> = accum (w, op(u,y))
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector mask,          // optional mask for w, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for z=accum(w,t)
    const GrB_BinaryOp op,          // operator to apply to the entries
    const GrB_Vector u,             // first input:  vector u
    bool y,                         // second input: scalar y
    const GrB_Descriptor desc       // descriptor for w and mask
) ;

GrB_Info GrB_Vector_apply_BinaryOp2nd_INT8      // w<mask> = accum (w, op(u,y))
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector mask,          // optional mask for w, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for z=accum(w,t)
    const GrB_BinaryOp op,          // operator to apply to the entries
    const GrB_Vector u,             // first input:  vector u
    int8_t y,                       // second input: scalar y
    const GrB_Descriptor desc       // descriptor for w and mask
) ;

GrB_Info GrB_Vector_apply_BinaryOp2nd_INT16     // w<mask> = accum (w, op(u,y))
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector mask,          // optional mask for w, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for z=accum(w,t)
    const GrB_BinaryOp op,          // operator to apply to the entries
    const GrB_Vector u,             // first input:  vector u
    int16_t y,                      // second input: scalar y
    const GrB_Descriptor desc       // descriptor for w and mask
) ;

GrB_Info GrB_Vector_apply_BinaryOp2nd_INT32     // w<mask> = accum (w, op(u,y))
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector mask,          // optional mask for w, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for z=accum(w,t)
    const GrB_BinaryOp op,          // operator to apply to the entries
    const GrB_Vector u,             // first input:  vector u
    int32_t y,                      // second input: scalar y
    const GrB_Descriptor desc       // descriptor for w and mask
) ;

GrB_Info GrB_Vector_apply_BinaryOp2nd_INT64     // w<mask> = accum (w, op(u,y))
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector mask,          // optional mask for w, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for z=accum(w,t)
    const GrB_BinaryOp op,          // operator to apply to the entries
    const GrB_Vector u,             // first input:  vector u
    int64_t y,                      // second input: scalar y
    const GrB_Descriptor desc       // descriptor for w and mask
) ;

GrB_Info GrB_Vector_apply_BinaryOp2nd_UINT8     // w<mask> = accum (w, op(u,y))
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector mask,          // optional mask for w, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for z=accum(w,t)
    const GrB_BinaryOp op,          // operator to apply to the entries
    const GrB_Vector u,             // first input:  vector u
    uint8_t y,                      // second input: scalar y
    const GrB_Descriptor desc       // descriptor for w and mask
) ;

GrB_Info GrB_Vector_apply_BinaryOp2nd_UINT16    // w<mask> = accum (w, op(u,y))
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector mask,          // optional mask for w, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for z=accum(w,t)
    const GrB_BinaryOp op,          // operator to apply to the entries
    const GrB_Vector u,             // first input:  vector u
    uint16_t y,                     // second input: scalar y
    const GrB_Descriptor desc       // descriptor for w and mask
) ;

GrB_Info GrB_Vector_apply_BinaryOp2nd_UINT32    // w<mask> = accum (w, op(u,y))
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector mask,          // optional mask for w, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for z=accum(w,t)
    const GrB_BinaryOp op,          // operator to apply to the entries
    const GrB_Vector u,             // first input:  vector u
    uint32_t y,                     // second input: scalar y
    const GrB_Descriptor desc       // descriptor for w and mask
) ;

GrB_Info GrB_Vector_apply_BinaryOp2nd_UINT64    // w<mask> = accum (w, op(u,y))
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector mask,          // optional mask for w, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for z=accum(w,t)
    const GrB_BinaryOp op,          // operator to apply to the entries
    const GrB_Vector u,             // first input:  vector u
    uint64_t y,                     // second input: scalar y
    const GrB_Descriptor desc       // descriptor for w and mask
) ;

GrB_Info GrB_Vector_apply_BinaryOp2nd_FP32      // w<mask> = accum (w, op(u,y))
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector mask,          // optional mask for w, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for z=accum(w,t)
    const GrB_BinaryOp op,          // operator to apply to the entries
    const GrB_Vector u,             // first input:  vector u
    float y,                        // second input: scalar y
    const GrB_Descriptor desc       // descriptor for w and mask
) ;

GrB_Info GrB_Vector_apply_BinaryOp2nd_FP64      // w<mask> = accum (w, op(u,y))
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector mask,          // optional mask for w, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for z=accum(w,t)
    const GrB_BinaryOp op,          // operator to apply to the entries
    const GrB_Vector u,             // first input:  vector u
    double y,                       // second input: scalar y
    const GrB_Descriptor desc       // descriptor for w and mask
) ;

GrB_Info GxB_Vector_apply_BinaryOp2nd_FC32      // w<mask> = accum (w, op(u,y))
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector mask,          // optional mask for w, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for z=accum(w,t)
    const GrB_BinaryOp op,          // operator to apply to the entries
    const GrB_Vector u,             // first input:  vector u
    GxB_FC32_t y,                   // second input: scalar y
    const GrB_Descriptor desc       // descriptor for w and mask
) ;

GrB_Info GxB_Vector_apply_BinaryOp2nd_FC64      // w<mask> = accum (w, op(u,y))
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector mask,          // optional mask for w, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for z=accum(w,t)
    const GrB_BinaryOp op,          // operator to apply to the entries
    const GrB_Vector u,             // first input:  vector u
    GxB_FC64_t y,                   // second input: scalar y
    const GrB_Descriptor desc       // descriptor for w and mask
) ;

GrB_Info GrB_Vector_apply_BinaryOp2nd_UDT       // w<mask> = accum (w, op(u,y))
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector mask,          // optional mask for w, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for z=accum(w,t)
    const GrB_BinaryOp op,          // operator to apply to the entries
    const GrB_Vector u,             // first input:  vector u
    const void *y,                  // second input: scalar y
    const GrB_Descriptor desc       // descriptor for w and mask
) ;

//-------------------------------------------
// vector apply: IndexUnaryOp variants
//-------------------------------------------

// Apply a GrB_IndexUnaryOp to the entries in a vector

GrB_Info GrB_Vector_apply_IndexOp_Scalar    // w<mask> = accum (w, op(u))
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector mask,          // optional mask for w, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for z=accum(w,t)
    const GrB_IndexUnaryOp op,      // operator to apply to the entries
    const GrB_Vector u,             // first input:  vector u
    const GrB_Scalar y,             // second input: scalar y
    const GrB_Descriptor desc       // descriptor for w and mask
) ;

GrB_Info GrB_Vector_apply_IndexOp_BOOL      // w<mask> = accum (w, op(u))
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector mask,          // optional mask for w, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for z=accum(w,t)
    const GrB_IndexUnaryOp op,      // operator to apply to the entries
    const GrB_Vector u,             // first input:  vector u
    bool y,                         // second input: scalar y
    const GrB_Descriptor desc       // descriptor for w and mask
) ;

GrB_Info GrB_Vector_apply_IndexOp_INT8      // w<mask> = accum (w, op(u))
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector mask,          // optional mask for w, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for z=accum(w,t)
    const GrB_IndexUnaryOp op,      // operator to apply to the entries
    const GrB_Vector u,             // first input:  vector u
    int8_t y,                       // second input: scalar y
    const GrB_Descriptor desc       // descriptor for w and mask
) ;

GrB_Info GrB_Vector_apply_IndexOp_INT16     // w<mask> = accum (w, op(u))
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector mask,          // optional mask for w, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for z=accum(w,t)
    const GrB_IndexUnaryOp op,      // operator to apply to the entries
    const GrB_Vector u,             // first input:  vector u
    int16_t y,                      // second input: scalar y
    const GrB_Descriptor desc       // descriptor for w and mask
) ;

GrB_Info GrB_Vector_apply_IndexOp_INT32     // w<mask> = accum (w, op(u))
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector mask,          // optional mask for w, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for z=accum(w,t)
    const GrB_IndexUnaryOp op,      // operator to apply to the entries
    const GrB_Vector u,             // first input:  vector u
    int32_t y,                      // second input: scalar y
    const GrB_Descriptor desc       // descriptor for w and mask
) ;

GrB_Info GrB_Vector_apply_IndexOp_INT64     // w<mask> = accum (w, op(u))
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector mask,          // optional mask for w, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for z=accum(w,t)
    const GrB_IndexUnaryOp op,      // operator to apply to the entries
    const GrB_Vector u,             // first input:  vector u
    int64_t y,                      // second input: scalar y
    const GrB_Descriptor desc       // descriptor for w and mask
) ;

GrB_Info GrB_Vector_apply_IndexOp_UINT8     // w<mask> = accum (w, op(u))
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector mask,          // optional mask for w, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for z=accum(w,t)
    const GrB_IndexUnaryOp op,      // operator to apply to the entries
    const GrB_Vector u,             // first input:  vector u
    uint8_t y,                      // second input: scalar y
    const GrB_Descriptor desc       // descriptor for w and mask
) ;

GrB_Info GrB_Vector_apply_IndexOp_UINT16    // w<mask> = accum (w, op(u))
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector mask,          // optional mask for w, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for z=accum(w,t)
    const GrB_IndexUnaryOp op,      // operator to apply to the entries
    const GrB_Vector u,             // first input:  vector u
    uint16_t y,                     // second input: scalar y
    const GrB_Descriptor desc       // descriptor for w and mask
) ;

GrB_Info GrB_Vector_apply_IndexOp_UINT32    // w<mask> = accum (w, op(u))
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector mask,          // optional mask for w, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for z=accum(w,t)
    const GrB_IndexUnaryOp op,      // operator to apply to the entries
    const GrB_Vector u,             // first input:  vector u
    uint32_t y,                     // second input: scalar y
    const GrB_Descriptor desc       // descriptor for w and mask
) ;

GrB_Info GrB_Vector_apply_IndexOp_UINT64    // w<mask> = accum (w, op(u))
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector mask,          // optional mask for w, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for z=accum(w,t)
    const GrB_IndexUnaryOp op,      // operator to apply to the entries
    const GrB_Vector u,             // first input:  vector u
    uint64_t y,                     // second input: scalar y
    const GrB_Descriptor desc       // descriptor for w and mask
) ;

GrB_Info GrB_Vector_apply_IndexOp_FP32      // w<mask> = accum (w, op(u))
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector mask,          // optional mask for w, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for z=accum(w,t)
    const GrB_IndexUnaryOp op,      // operator to apply to the entries
    const GrB_Vector u,             // first input:  vector u
    float y,                        // second input: scalar y
    const GrB_Descriptor desc       // descriptor for w and mask
) ;

GrB_Info GrB_Vector_apply_IndexOp_FP64      // w<mask> = accum (w, op(u))
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector mask,          // optional mask for w, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for z=accum(w,t)
    const GrB_IndexUnaryOp op,      // operator to apply to the entries
    const GrB_Vector u,             // first input:  vector u
    double y,                       // second input: scalar y
    const GrB_Descriptor desc       // descriptor for w and mask
) ;

GrB_Info GxB_Vector_apply_IndexOp_FC32      // w<mask> = accum (w, op(u))
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector mask,          // optional mask for w, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for z=accum(w,t)
    const GrB_IndexUnaryOp op,      // operator to apply to the entries
    const GrB_Vector u,             // first input:  vector u
    GxB_FC32_t y,                   // second input: scalar y
    const GrB_Descriptor desc       // descriptor for w and mask
) ;

GrB_Info GxB_Vector_apply_IndexOp_FC64      // w<mask> = accum (w, op(u))
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector mask,          // optional mask for w, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for z=accum(w,t)
    const GrB_IndexUnaryOp op,      // operator to apply to the entries
    const GrB_Vector u,             // first input:  vector u
    GxB_FC64_t y,                   // second input: scalar y
    const GrB_Descriptor desc       // descriptor for w and mask
) ;

GrB_Info GrB_Vector_apply_IndexOp_UDT       // w<mask> = accum (w, op(u))
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector mask,          // optional mask for w, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for z=accum(w,t)
    const GrB_IndexUnaryOp op,      // operator to apply to the entries
    const GrB_Vector u,             // first input:  vector u
    const void *y,                  // second input: scalar y
    const GrB_Descriptor desc       // descriptor for w and mask
) ;

//-------------------------------------------
// matrix apply: binaryop variants (bind 1st)
//-------------------------------------------

// Apply a binary operator to the entries in a matrix, binding the first input
// to a scalar x, C<Mask> = accum (C, op (x,A)), or op(x,A').

GrB_Info GrB_Matrix_apply_BinaryOp1st_Scalar    // C<M>=accum(C,op(x,A))
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix Mask,          // optional mask for C, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C,T)
    const GrB_BinaryOp op,          // operator to apply to the entries
    const GrB_Scalar x,             // first input:  scalar x
    const GrB_Matrix A,             // second input: matrix A
    const GrB_Descriptor desc       // descriptor for C, mask, and A
) ;

// historical: identical to GrB_Matrix_apply_BinaryOp1st_Scalar
GrB_Info GxB_Matrix_apply_BinaryOp1st           // C<M>=accum(C,op(x,A))
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix Mask,          // optional mask for C, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C,T)
    const GrB_BinaryOp op,          // operator to apply to the entries
    const GrB_Scalar x,             // first input:  scalar x
    const GrB_Matrix A,             // second input: matrix A
    const GrB_Descriptor desc       // descriptor for C, mask, and A
) ;

GrB_Info GrB_Matrix_apply_BinaryOp1st_BOOL      // C<M>=accum(C,op(x,A))
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix Mask,          // optional mask for C, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C,T)
    const GrB_BinaryOp op,          // operator to apply to the entries
    bool x,                         // first input:  scalar x
    const GrB_Matrix A,             // second input: matrix A
    const GrB_Descriptor desc       // descriptor for C, mask, and A
) ;

GrB_Info GrB_Matrix_apply_BinaryOp1st_INT8      // C<M>=accum(C,op(x,A))
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix Mask,          // optional mask for C, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C,T)
    const GrB_BinaryOp op,          // operator to apply to the entries
    int8_t x,                       // first input:  scalar x
    const GrB_Matrix A,             // second input: matrix A
    const GrB_Descriptor desc       // descriptor for C, mask, and A
) ;

GrB_Info GrB_Matrix_apply_BinaryOp1st_INT16     // C<M>=accum(C,op(x,A))
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix Mask,          // optional mask for C, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C,T)
    const GrB_BinaryOp op,          // operator to apply to the entries
    int16_t x,                      // first input:  scalar x
    const GrB_Matrix A,             // second input: matrix A
    const GrB_Descriptor desc       // descriptor for C, mask, and A
) ;

GrB_Info GrB_Matrix_apply_BinaryOp1st_INT32     // C<M>=accum(C,op(x,A))
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix Mask,          // optional mask for C, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C,T)
    const GrB_BinaryOp op,          // operator to apply to the entries
    int32_t x,                      // first input:  scalar x
    const GrB_Matrix A,             // second input: matrix A
    const GrB_Descriptor desc       // descriptor for C, mask, and A
) ;

GrB_Info GrB_Matrix_apply_BinaryOp1st_INT64     // C<M>=accum(C,op(x,A))
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix Mask,          // optional mask for C, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C,T)
    const GrB_BinaryOp op,          // operator to apply to the entries
    int64_t x,                      // first input:  scalar x
    const GrB_Matrix A,             // second input: matrix A
    const GrB_Descriptor desc       // descriptor for C, mask, and A
) ;

GrB_Info GrB_Matrix_apply_BinaryOp1st_UINT8      // C<M>=accum(C,op(x,A))
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix Mask,          // optional mask for C, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C,T)
    const GrB_BinaryOp op,          // operator to apply to the entries
    uint8_t x,                      // first input:  scalar x
    const GrB_Matrix A,             // second input: matrix A
    const GrB_Descriptor desc       // descriptor for C, mask, and A
) ;

GrB_Info GrB_Matrix_apply_BinaryOp1st_UINT16     // C<M>=accum(C,op(x,A))
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix Mask,          // optional mask for C, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C,T)
    const GrB_BinaryOp op,          // operator to apply to the entries
    uint16_t x,                     // first input:  scalar x
    const GrB_Matrix A,             // second input: matrix A
    const GrB_Descriptor desc       // descriptor for C, mask, and A
) ;

GrB_Info GrB_Matrix_apply_BinaryOp1st_UINT32     // C<M>=accum(C,op(x,A))
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix Mask,          // optional mask for C, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C,T)
    const GrB_BinaryOp op,          // operator to apply to the entries
    uint32_t x,                     // first input:  scalar x
    const GrB_Matrix A,             // second input: matrix A
    const GrB_Descriptor desc       // descriptor for C, mask, and A
) ;

GrB_Info GrB_Matrix_apply_BinaryOp1st_UINT64     // C<M>=accum(C,op(x,A))
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix Mask,          // optional mask for C, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C,T)
    const GrB_BinaryOp op,          // operator to apply to the entries
    uint64_t x,                     // first input:  scalar x
    const GrB_Matrix A,             // second input: matrix A
    const GrB_Descriptor desc       // descriptor for C, mask, and A
) ;

GrB_Info GrB_Matrix_apply_BinaryOp1st_FP32      // C<M>=accum(C,op(x,A))
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix Mask,          // optional mask for C, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C,T)
    const GrB_BinaryOp op,          // operator to apply to the entries
    float x,                        // first input:  scalar x
    const GrB_Matrix A,             // second input: matrix A
    const GrB_Descriptor desc       // descriptor for C, mask, and A
) ;

GrB_Info GrB_Matrix_apply_BinaryOp1st_FP64      // C<M>=accum(C,op(x,A))
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix Mask,          // optional mask for C, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C,T)
    const GrB_BinaryOp op,          // operator to apply to the entries
    double x,                       // first input:  scalar x
    const GrB_Matrix A,             // second input: matrix A
    const GrB_Descriptor desc       // descriptor for C, mask, and A
) ;

GrB_Info GxB_Matrix_apply_BinaryOp1st_FC32      // C<M>=accum(C,op(x,A))
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix Mask,          // optional mask for C, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C,T)
    const GrB_BinaryOp op,          // operator to apply to the entries
    GxB_FC32_t x,                   // first input:  scalar x
    const GrB_Matrix A,             // second input: matrix A
    const GrB_Descriptor desc       // descriptor for C, mask, and A
) ;

GrB_Info GxB_Matrix_apply_BinaryOp1st_FC64      // C<M>=accum(C,op(x,A))
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix Mask,          // optional mask for C, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C,T)
    const GrB_BinaryOp op,          // operator to apply to the entries
    GxB_FC64_t x,                   // first input:  scalar x
    const GrB_Matrix A,             // second input: matrix A
    const GrB_Descriptor desc       // descriptor for C, mask, and A
) ;

GrB_Info GrB_Matrix_apply_BinaryOp1st_UDT       // C<M>=accum(C,op(x,A))
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix Mask,          // optional mask for C, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C,T)
    const GrB_BinaryOp op,          // operator to apply to the entries
    const void *x,                  // first input:  scalar x
    const GrB_Matrix A,             // second input: matrix A
    const GrB_Descriptor desc       // descriptor for C, mask, and A
) ;

//-------------------------------------------
// matrix apply: binaryop variants (bind 2nd)
//-------------------------------------------

// Apply a binary operator to the entries in a matrix, binding the second input
// to a scalar y, C<Mask> = accum (C, op (A,y)), or op(A',y).

GrB_Info GrB_Matrix_apply_BinaryOp2nd_Scalar    // C<M>=accum(C,op(A,y))
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix Mask,          // optional mask for C, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C,T)
    const GrB_BinaryOp op,          // operator to apply to the entries
    const GrB_Matrix A,             // first input:  matrix A
    const GrB_Scalar y,             // second input: scalar y
    const GrB_Descriptor desc       // descriptor for C, mask, and A
) ;

// historical: identical to GrB_Matrix_apply_BinaryOp2nd_Scalar
GrB_Info GxB_Matrix_apply_BinaryOp2nd           // C<M>=accum(C,op(A,y))
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix Mask,          // optional mask for C, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C,T)
    const GrB_BinaryOp op,          // operator to apply to the entries
    const GrB_Matrix A,             // first input:  matrix A
    const GrB_Scalar y,             // second input: scalar y
    const GrB_Descriptor desc       // descriptor for C, mask, and A
) ;

GrB_Info GrB_Matrix_apply_BinaryOp2nd_BOOL      // C<M>=accum(C,op(A,y))
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix Mask,          // optional mask for C, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C,T)
    const GrB_BinaryOp op,          // operator to apply to the entries
    const GrB_Matrix A,             // first input:  matrix A
    bool y,                         // second input: scalar y
    const GrB_Descriptor desc       // descriptor for C, mask, and A
) ;

GrB_Info GrB_Matrix_apply_BinaryOp2nd_INT8      // C<M>=accum(C,op(A,y))
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix Mask,          // optional mask for C, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C,T)
    const GrB_BinaryOp op,          // operator to apply to the entries
    const GrB_Matrix A,             // first input:  matrix A
    int8_t y,                       // second input: scalar y
    const GrB_Descriptor desc       // descriptor for C, mask, and A
) ;

GrB_Info GrB_Matrix_apply_BinaryOp2nd_INT16     // C<M>=accum(C,op(A,y))
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix Mask,          // optional mask for C, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C,T)
    const GrB_BinaryOp op,          // operator to apply to the entries
    const GrB_Matrix A,             // first input:  matrix A
    int16_t y,                      // second input: scalar y
    const GrB_Descriptor desc       // descriptor for C, mask, and A
) ;

GrB_Info GrB_Matrix_apply_BinaryOp2nd_INT32     // C<M>=accum(C,op(A,y))
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix Mask,          // optional mask for C, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C,T)
    const GrB_BinaryOp op,          // operator to apply to the entries
    const GrB_Matrix A,             // first input:  matrix A
    int32_t y,                      // second input: scalar y
    const GrB_Descriptor desc       // descriptor for C, mask, and A
) ;

GrB_Info GrB_Matrix_apply_BinaryOp2nd_INT64     // C<M>=accum(C,op(A,y))
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix Mask,          // optional mask for C, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C,T)
    const GrB_BinaryOp op,          // operator to apply to the entries
    const GrB_Matrix A,             // first input:  matrix A
    int64_t y,                      // second input: scalar y
    const GrB_Descriptor desc       // descriptor for C, mask, and A
) ;

GrB_Info GrB_Matrix_apply_BinaryOp2nd_UINT8      // C<M>=accum(C,op(A,y))
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix Mask,          // optional mask for C, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C,T)
    const GrB_BinaryOp op,          // operator to apply to the entries
    const GrB_Matrix A,             // first input:  matrix A
    uint8_t y,                      // second input: scalar y
    const GrB_Descriptor desc       // descriptor for C, mask, and A
) ;

GrB_Info GrB_Matrix_apply_BinaryOp2nd_UINT16     // C<M>=accum(C,op(A,y))
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix Mask,          // optional mask for C, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C,T)
    const GrB_BinaryOp op,          // operator to apply to the entries
    const GrB_Matrix A,             // first input:  matrix A
    uint16_t y,                     // second input: scalar y
    const GrB_Descriptor desc       // descriptor for C, mask, and A
) ;

GrB_Info GrB_Matrix_apply_BinaryOp2nd_UINT32     // C<M>=accum(C,op(A,y))
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix Mask,          // optional mask for C, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C,T)
    const GrB_BinaryOp op,          // operator to apply to the entries
    const GrB_Matrix A,             // first input:  matrix A
    uint32_t y,                     // second input: scalar y
    const GrB_Descriptor desc       // descriptor for C, mask, and A
) ;

GrB_Info GrB_Matrix_apply_BinaryOp2nd_UINT64     // C<M>=accum(C,op(A,y))
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix Mask,          // optional mask for C, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C,T)
    const GrB_BinaryOp op,          // operator to apply to the entries
    const GrB_Matrix A,             // first input:  matrix A
    uint64_t y,                     // second input: scalar y
    const GrB_Descriptor desc       // descriptor for C, mask, and A
) ;

GrB_Info GrB_Matrix_apply_BinaryOp2nd_FP32      // C<M>=accum(C,op(A,y))
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix Mask,          // optional mask for C, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C,T)
    const GrB_BinaryOp op,          // operator to apply to the entries
    const GrB_Matrix A,             // first input:  matrix A
    float y,                        // second input: scalar y
    const GrB_Descriptor desc       // descriptor for C, mask, and A
) ;

GrB_Info GrB_Matrix_apply_BinaryOp2nd_FP64      // C<M>=accum(C,op(A,y))
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix Mask,          // optional mask for C, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C,T)
    const GrB_BinaryOp op,          // operator to apply to the entries
    const GrB_Matrix A,             // first input:  matrix A
    double y,                       // second input: scalar y
    const GrB_Descriptor desc       // descriptor for C, mask, and A
) ;

GrB_Info GxB_Matrix_apply_BinaryOp2nd_FC32      // C<M>=accum(C,op(A,y))
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix Mask,          // optional mask for C, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C,T)
    const GrB_BinaryOp op,          // operator to apply to the entries
    const GrB_Matrix A,             // first input:  matrix A
    GxB_FC32_t y,                   // second input: scalar y
    const GrB_Descriptor desc       // descriptor for C, mask, and A
) ;

GrB_Info GxB_Matrix_apply_BinaryOp2nd_FC64      // C<M>=accum(C,op(A,y))
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix Mask,          // optional mask for C, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C,T)
    const GrB_BinaryOp op,          // operator to apply to the entries
    const GrB_Matrix A,             // first input:  matrix A
    GxB_FC64_t y,                   // second input: scalar y
    const GrB_Descriptor desc       // descriptor for C, mask, and A
) ;

GrB_Info GrB_Matrix_apply_BinaryOp2nd_UDT       // C<M>=accum(C,op(A,y))
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix Mask,          // optional mask for C, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C,T)
    const GrB_BinaryOp op,          // operator to apply to the entries
    const GrB_Matrix A,             // first input:  matrix A
    const void *y,                  // second input: scalar y
    const GrB_Descriptor desc       // descriptor for C, mask, and A
) ;

//-------------------------------------------
// matrix apply: IndexUnaryOp variants
//-------------------------------------------

// Apply a GrB_IndexUnaryOp to the entries in a matrix.

GrB_Info GrB_Matrix_apply_IndexOp_Scalar    // C<M>=accum(C,op(A))
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix Mask,          // optional mask for C, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C,T)
    const GrB_IndexUnaryOp op,      // operator to apply to the entries
    const GrB_Matrix A,             // first input:  matrix A
    const GrB_Scalar y,             // second input: scalar y
    const GrB_Descriptor desc       // descriptor for C, mask, and A
) ;

GrB_Info GrB_Matrix_apply_IndexOp_BOOL      // C<M>=accum(C,op(A))
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix Mask,          // optional mask for C, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C,T)
    const GrB_IndexUnaryOp op,      // operator to apply to the entries
    const GrB_Matrix A,             // first input:  matrix A
    bool y,                         // second input: scalar y
    const GrB_Descriptor desc       // descriptor for C, mask, and A
) ;

GrB_Info GrB_Matrix_apply_IndexOp_INT8      // C<M>=accum(C,op(A))
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix Mask,          // optional mask for C, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C,T)
    const GrB_IndexUnaryOp op,      // operator to apply to the entries
    const GrB_Matrix A,             // first input:  matrix A
    int8_t y,                       // second input: scalar y
    const GrB_Descriptor desc       // descriptor for C, mask, and A
) ;

GrB_Info GrB_Matrix_apply_IndexOp_INT16     // C<M>=accum(C,op(A))
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix Mask,          // optional mask for C, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C,T)
    const GrB_IndexUnaryOp op,      // operator to apply to the entries
    const GrB_Matrix A,             // first input:  matrix A
    int16_t y,                      // second input: scalar y
    const GrB_Descriptor desc       // descriptor for C, mask, and A
) ;

GrB_Info GrB_Matrix_apply_IndexOp_INT32     // C<M>=accum(C,op(A))
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix Mask,          // optional mask for C, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C,T)
    const GrB_IndexUnaryOp op,      // operator to apply to the entries
    const GrB_Matrix A,             // first input:  matrix A
    int32_t y,                      // second input: scalar y
    const GrB_Descriptor desc       // descriptor for C, mask, and A
) ;

GrB_Info GrB_Matrix_apply_IndexOp_INT64     // C<M>=accum(C,op(A))
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix Mask,          // optional mask for C, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C,T)
    const GrB_IndexUnaryOp op,      // operator to apply to the entries
    const GrB_Matrix A,             // first input:  matrix A
    int64_t y,                      // second input: scalar y
    const GrB_Descriptor desc       // descriptor for C, mask, and A
) ;

GrB_Info GrB_Matrix_apply_IndexOp_UINT8      // C<M>=accum(C,op(A))
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix Mask,          // optional mask for C, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C,T)
    const GrB_IndexUnaryOp op,      // operator to apply to the entries
    const GrB_Matrix A,             // first input:  matrix A
    uint8_t y,                      // second input: scalar y
    const GrB_Descriptor desc       // descriptor for C, mask, and A
) ;

GrB_Info GrB_Matrix_apply_IndexOp_UINT16     // C<M>=accum(C,op(A))
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix Mask,          // optional mask for C, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C,T)
    const GrB_IndexUnaryOp op,      // operator to apply to the entries
    const GrB_Matrix A,             // first input:  matrix A
    uint16_t y,                     // second input: scalar y
    const GrB_Descriptor desc       // descriptor for C, mask, and A
) ;

GrB_Info GrB_Matrix_apply_IndexOp_UINT32     // C<M>=accum(C,op(A))
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix Mask,          // optional mask for C, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C,T)
    const GrB_IndexUnaryOp op,      // operator to apply to the entries
    const GrB_Matrix A,             // first input:  matrix A
    uint32_t y,                     // second input: scalar y
    const GrB_Descriptor desc       // descriptor for C, mask, and A
) ;

GrB_Info GrB_Matrix_apply_IndexOp_UINT64     // C<M>=accum(C,op(A))
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix Mask,          // optional mask for C, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C,T)
    const GrB_IndexUnaryOp op,      // operator to apply to the entries
    const GrB_Matrix A,             // first input:  matrix A
    uint64_t y,                     // second input: scalar y
    const GrB_Descriptor desc       // descriptor for C, mask, and A
) ;

GrB_Info GrB_Matrix_apply_IndexOp_FP32      // C<M>=accum(C,op(A))
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix Mask,          // optional mask for C, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C,T)
    const GrB_IndexUnaryOp op,      // operator to apply to the entries
    const GrB_Matrix A,             // first input:  matrix A
    float y,                        // second input: scalar y
    const GrB_Descriptor desc       // descriptor for C, mask, and A
) ;

GrB_Info GrB_Matrix_apply_IndexOp_FP64      // C<M>=accum(C,op(A))
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix Mask,          // optional mask for C, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C,T)
    const GrB_IndexUnaryOp op,      // operator to apply to the entries
    const GrB_Matrix A,             // first input:  matrix A
    double y,                       // second input: scalar y
    const GrB_Descriptor desc       // descriptor for C, mask, and A
) ;

GrB_Info GxB_Matrix_apply_IndexOp_FC32      // C<M>=accum(C,op(A))
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix Mask,          // optional mask for C, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C,T)
    const GrB_IndexUnaryOp op,      // operator to apply to the entries
    const GrB_Matrix A,             // first input:  matrix A
    GxB_FC32_t y,                   // second input: scalar y
    const GrB_Descriptor desc       // descriptor for C, mask, and A
) ;

GrB_Info GxB_Matrix_apply_IndexOp_FC64      // C<M>=accum(C,op(A))
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix Mask,          // optional mask for C, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C,T)
    const GrB_IndexUnaryOp op,      // operator to apply to the entries
    const GrB_Matrix A,             // first input:  matrix A
    GxB_FC64_t y,                   // second input: scalar y
    const GrB_Descriptor desc       // descriptor for C, mask, and A
) ;

GrB_Info GrB_Matrix_apply_IndexOp_UDT       // C<M>=accum(C,op(A))
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix Mask,          // optional mask for C, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C,T)
    const GrB_IndexUnaryOp op,      // operator to apply to the entries
    const GrB_Matrix A,             // first input:  matrix A
    const void *y,                  // second input: scalar y
    const GrB_Descriptor desc       // descriptor for C, mask, and A
) ;

//------------------------------------------------------------------------------
// GrB_apply: generic matrix/vector apply
//------------------------------------------------------------------------------

// GrB_apply is a generic function for applying a unary operator to a matrix
// or vector and provides access to these functions:

// GrB_Vector_apply (w,mask,acc,op,u,d)  // w<mask> = accum (w, op(u))
// GrB_Matrix_apply (C,Mask,acc,op,A,d)  // C<Mask> = accum (C, op(A))

// GrB_Vector_apply                  (w,m,acc,unop ,u,d)
// GrB_Vector_apply_BinaryOp1st_TYPE (w,m,acc,binop,x,u,d)
// GrB_Vector_apply_BinaryOp2nd_TYPE (w,m,acc,binop,u,y,d)
// GrB_Vector_apply_IndexOp_TYPE     (w,m,acc,idxop,u,y,d)

// GrB_Matrix_apply                  (C,M,acc,unop ,A,d)
// GrB_Matrix_apply_BinaryOp1st_TYPE (C,M,acc,binop,x,A,d)
// GrB_Matrix_apply_BinaryOp2nd_TYPE (C,M,acc,binop,A,y,d)
// GrB_Matrix_apply_IndexOp_TYPE     (C,M,acc,idxop,A,y,d)

#if GxB_STDC_VERSION >= 201112L

#define GB_BIND(kind,x,y,...)                                                \
    _Generic                                                                 \
    (                                                                        \
        (x),                                                                 \
        GrB_Scalar: GB_CONCAT ( GrB,_,kind,_apply_BinaryOp1st_Scalar),       \
        GB_CASES (GrB, GB_CONCAT ( kind, _apply_BinaryOp1st,, )) ,           \
        default:                                                             \
            _Generic                                                         \
            (                                                                \
                (y),                                                         \
                GB_CASES (GrB, GB_CONCAT ( kind , _apply_BinaryOp2nd,, )),   \
                default:  GB_CONCAT ( GrB,_,kind,_apply_BinaryOp2nd_Scalar)  \
            )                                                                \
    )

#define GB_IDXOP(kind,A,y,...)                                               \
    _Generic                                                                 \
    (                                                                        \
        (y),                                                                 \
            GB_CASES (GrB, GB_CONCAT ( kind, _apply_IndexOp,, )),            \
            default:  GB_CONCAT ( GrB, _, kind, _apply_IndexOp_Scalar)       \
    )

#define GrB_apply(C,Mask,accum,op,...)                                       \
    _Generic                                                                 \
    (                                                                        \
        (C),                                                                 \
            GrB_Vector :                                                     \
                _Generic                                                     \
                (                                                            \
                    (op),                                                    \
                        GrB_UnaryOp  : GrB_Vector_apply ,                    \
                        GrB_BinaryOp : GB_BIND (Vector, __VA_ARGS__),        \
                        GrB_IndexUnaryOp : GB_IDXOP (Vector, __VA_ARGS__)    \
                ),                                                           \
            GrB_Matrix :                                                     \
                _Generic                                                     \
                (                                                            \
                    (op),                                                    \
                        GrB_UnaryOp  : GrB_Matrix_apply ,                    \
                        GrB_BinaryOp : GB_BIND (Matrix, __VA_ARGS__),        \
                        GrB_IndexUnaryOp : GB_IDXOP (Matrix, __VA_ARGS__)    \
                )                                                            \
    )                                                                        \
    (C, Mask, accum, op, __VA_ARGS__)
#endif

//==============================================================================
// GrB_select: matrix and vector selection using an IndexUnaryOp
//==============================================================================

//-------------------------------------------
// vector select using an IndexUnaryOp
//-------------------------------------------

GrB_Info GrB_Vector_select_Scalar   // w<mask> = accum (w, op(u))
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector mask,          // optional mask for w, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for z=accum(w,t)
    const GrB_IndexUnaryOp op,      // operator to apply to the entries
    const GrB_Vector u,             // first input:  vector u
    const GrB_Scalar y,             // second input: scalar y
    const GrB_Descriptor desc       // descriptor for w and mask
) ;

GrB_Info GrB_Vector_select_BOOL      // w<mask> = accum (w, op(u))
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector mask,          // optional mask for w, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for z=accum(w,t)
    const GrB_IndexUnaryOp op,      // operator to apply to the entries
    const GrB_Vector u,             // first input:  vector u
    bool y,                         // second input: scalar y
    const GrB_Descriptor desc       // descriptor for w and mask
) ;

GrB_Info GrB_Vector_select_INT8      // w<mask> = accum (w, op(u))
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector mask,          // optional mask for w, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for z=accum(w,t)
    const GrB_IndexUnaryOp op,      // operator to apply to the entries
    const GrB_Vector u,             // first input:  vector u
    int8_t y,                       // second input: scalar y
    const GrB_Descriptor desc       // descriptor for w and mask
) ;

GrB_Info GrB_Vector_select_INT16     // w<mask> = accum (w, op(u))
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector mask,          // optional mask for w, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for z=accum(w,t)
    const GrB_IndexUnaryOp op,      // operator to apply to the entries
    const GrB_Vector u,             // first input:  vector u
    int16_t y,                      // second input: scalar y
    const GrB_Descriptor desc       // descriptor for w and mask
) ;

GrB_Info GrB_Vector_select_INT32     // w<mask> = accum (w, op(u))
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector mask,          // optional mask for w, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for z=accum(w,t)
    const GrB_IndexUnaryOp op,      // operator to apply to the entries
    const GrB_Vector u,             // first input:  vector u
    int32_t y,                      // second input: scalar y
    const GrB_Descriptor desc       // descriptor for w and mask
) ;

GrB_Info GrB_Vector_select_INT64     // w<mask> = accum (w, op(u))
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector mask,          // optional mask for w, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for z=accum(w,t)
    const GrB_IndexUnaryOp op,      // operator to apply to the entries
    const GrB_Vector u,             // first input:  vector u
    int64_t y,                      // second input: scalar y
    const GrB_Descriptor desc       // descriptor for w and mask
) ;

GrB_Info GrB_Vector_select_UINT8     // w<mask> = accum (w, op(u))
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector mask,          // optional mask for w, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for z=accum(w,t)
    const GrB_IndexUnaryOp op,      // operator to apply to the entries
    const GrB_Vector u,             // first input:  vector u
    uint8_t y,                      // second input: scalar y
    const GrB_Descriptor desc       // descriptor for w and mask
) ;

GrB_Info GrB_Vector_select_UINT16    // w<mask> = accum (w, op(u))
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector mask,          // optional mask for w, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for z=accum(w,t)
    const GrB_IndexUnaryOp op,      // operator to apply to the entries
    const GrB_Vector u,             // first input:  vector u
    uint16_t y,                     // second input: scalar y
    const GrB_Descriptor desc       // descriptor for w and mask
) ;

GrB_Info GrB_Vector_select_UINT32    // w<mask> = accum (w, op(u))
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector mask,          // optional mask for w, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for z=accum(w,t)
    const GrB_IndexUnaryOp op,      // operator to apply to the entries
    const GrB_Vector u,             // first input:  vector u
    uint32_t y,                     // second input: scalar y
    const GrB_Descriptor desc       // descriptor for w and mask
) ;

GrB_Info GrB_Vector_select_UINT64    // w<mask> = accum (w, op(u))
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector mask,          // optional mask for w, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for z=accum(w,t)
    const GrB_IndexUnaryOp op,      // operator to apply to the entries
    const GrB_Vector u,             // first input:  vector u
    uint64_t y,                     // second input: scalar y
    const GrB_Descriptor desc       // descriptor for w and mask
) ;

GrB_Info GrB_Vector_select_FP32      // w<mask> = accum (w, op(u))
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector mask,          // optional mask for w, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for z=accum(w,t)
    const GrB_IndexUnaryOp op,      // operator to apply to the entries
    const GrB_Vector u,             // first input:  vector u
    float y,                        // second input: scalar y
    const GrB_Descriptor desc       // descriptor for w and mask
) ;

GrB_Info GrB_Vector_select_FP64      // w<mask> = accum (w, op(u))
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector mask,          // optional mask for w, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for z=accum(w,t)
    const GrB_IndexUnaryOp op,      // operator to apply to the entries
    const GrB_Vector u,             // first input:  vector u
    double y,                       // second input: scalar y
    const GrB_Descriptor desc       // descriptor for w and mask
) ;

GrB_Info GxB_Vector_select_FC32      // w<mask> = accum (w, op(u))
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector mask,          // optional mask for w, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for z=accum(w,t)
    const GrB_IndexUnaryOp op,      // operator to apply to the entries
    const GrB_Vector u,             // first input:  vector u
    GxB_FC32_t y,                   // second input: scalar y
    const GrB_Descriptor desc       // descriptor for w and mask
) ;

GrB_Info GxB_Vector_select_FC64      // w<mask> = accum (w, op(u))
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector mask,          // optional mask for w, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for z=accum(w,t)
    const GrB_IndexUnaryOp op,      // operator to apply to the entries
    const GrB_Vector u,             // first input:  vector u
    GxB_FC64_t y,                   // second input: scalar y
    const GrB_Descriptor desc       // descriptor for w and mask
) ;

GrB_Info GrB_Vector_select_UDT       // w<mask> = accum (w, op(u))
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector mask,          // optional mask for w, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for z=accum(w,t)
    const GrB_IndexUnaryOp op,      // operator to apply to the entries
    const GrB_Vector u,             // first input:  vector u
    const void *y,                  // second input: scalar y
    const GrB_Descriptor desc       // descriptor for w and mask
) ;

//-------------------------------------------
// matrix select using an IndexUnaryOp
//-------------------------------------------

GrB_Info GrB_Matrix_select_Scalar   // C<M>=accum(C,op(A))
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix Mask,          // optional mask for C, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C,T)
    const GrB_IndexUnaryOp op,      // operator to apply to the entries
    const GrB_Matrix A,             // first input:  matrix A
    const GrB_Scalar y,             // second input: scalar y
    const GrB_Descriptor desc       // descriptor for C, mask, and A
) ;

GrB_Info GrB_Matrix_select_BOOL     // C<M>=accum(C,op(A))
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix Mask,          // optional mask for C, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C,T)
    const GrB_IndexUnaryOp op,      // operator to apply to the entries
    const GrB_Matrix A,             // first input:  matrix A
    bool y,                         // second input: scalar y
    const GrB_Descriptor desc       // descriptor for C, mask, and A
) ;

GrB_Info GrB_Matrix_select_INT8     // C<M>=accum(C,op(A))
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix Mask,          // optional mask for C, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C,T)
    const GrB_IndexUnaryOp op,      // operator to apply to the entries
    const GrB_Matrix A,             // first input:  matrix A
    int8_t y,                       // second input: scalar y
    const GrB_Descriptor desc       // descriptor for C, mask, and A
) ;

GrB_Info GrB_Matrix_select_INT16    // C<M>=accum(C,op(A))
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix Mask,          // optional mask for C, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C,T)
    const GrB_IndexUnaryOp op,      // operator to apply to the entries
    const GrB_Matrix A,             // first input:  matrix A
    int16_t y,                      // second input: scalar y
    const GrB_Descriptor desc       // descriptor for C, mask, and A
) ;

GrB_Info GrB_Matrix_select_INT32    // C<M>=accum(C,op(A))
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix Mask,          // optional mask for C, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C,T)
    const GrB_IndexUnaryOp op,      // operator to apply to the entries
    const GrB_Matrix A,             // first input:  matrix A
    int32_t y,                      // second input: scalar y
    const GrB_Descriptor desc       // descriptor for C, mask, and A
) ;

GrB_Info GrB_Matrix_select_INT64    // C<M>=accum(C,op(A))
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix Mask,          // optional mask for C, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C,T)
    const GrB_IndexUnaryOp op,      // operator to apply to the entries
    const GrB_Matrix A,             // first input:  matrix A
    int64_t y,                      // second input: scalar y
    const GrB_Descriptor desc       // descriptor for C, mask, and A
) ;

GrB_Info GrB_Matrix_select_UINT8    // C<M>=accum(C,op(A))
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix Mask,          // optional mask for C, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C,T)
    const GrB_IndexUnaryOp op,      // operator to apply to the entries
    const GrB_Matrix A,             // first input:  matrix A
    uint8_t y,                      // second input: scalar y
    const GrB_Descriptor desc       // descriptor for C, mask, and A
) ;

GrB_Info GrB_Matrix_select_UINT16   // C<M>=accum(C,op(A))
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix Mask,          // optional mask for C, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C,T)
    const GrB_IndexUnaryOp op,      // operator to apply to the entries
    const GrB_Matrix A,             // first input:  matrix A
    uint16_t y,                     // second input: scalar y
    const GrB_Descriptor desc       // descriptor for C, mask, and A
) ;

GrB_Info GrB_Matrix_select_UINT32   // C<M>=accum(C,op(A))
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix Mask,          // optional mask for C, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C,T)
    const GrB_IndexUnaryOp op,      // operator to apply to the entries
    const GrB_Matrix A,             // first input:  matrix A
    uint32_t y,                     // second input: scalar y
    const GrB_Descriptor desc       // descriptor for C, mask, and A
) ;

GrB_Info GrB_Matrix_select_UINT64   // C<M>=accum(C,op(A))
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix Mask,          // optional mask for C, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C,T)
    const GrB_IndexUnaryOp op,      // operator to apply to the entries
    const GrB_Matrix A,             // first input:  matrix A
    uint64_t y,                     // second input: scalar y
    const GrB_Descriptor desc       // descriptor for C, mask, and A
) ;

GrB_Info GrB_Matrix_select_FP32     // C<M>=accum(C,op(A))
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix Mask,          // optional mask for C, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C,T)
    const GrB_IndexUnaryOp op,      // operator to apply to the entries
    const GrB_Matrix A,             // first input:  matrix A
    float y,                        // second input: scalar y
    const GrB_Descriptor desc       // descriptor for C, mask, and A
) ;

GrB_Info GrB_Matrix_select_FP64     // C<M>=accum(C,op(A))
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix Mask,          // optional mask for C, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C,T)
    const GrB_IndexUnaryOp op,      // operator to apply to the entries
    const GrB_Matrix A,             // first input:  matrix A
    double y,                       // second input: scalar y
    const GrB_Descriptor desc       // descriptor for C, mask, and A
) ;

GrB_Info GxB_Matrix_select_FC32      // C<M>=accum(C,op(A))
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix Mask,          // optional mask for C, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C,T)
    const GrB_IndexUnaryOp op,      // operator to apply to the entries
    const GrB_Matrix A,             // first input:  matrix A
    GxB_FC32_t y,                   // second input: scalar y
    const GrB_Descriptor desc       // descriptor for C, mask, and A
) ;

GrB_Info GxB_Matrix_select_FC64      // C<M>=accum(C,op(A))
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix Mask,          // optional mask for C, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C,T)
    const GrB_IndexUnaryOp op,      // operator to apply to the entries
    const GrB_Matrix A,             // first input:  matrix A
    GxB_FC64_t y,                   // second input: scalar y
    const GrB_Descriptor desc       // descriptor for C, mask, and A
) ;

GrB_Info GrB_Matrix_select_UDT      // C<M>=accum(C,op(A))
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix Mask,          // optional mask for C, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C,T)
    const GrB_IndexUnaryOp op,      // operator to apply to the entries
    const GrB_Matrix A,             // first input:  matrix A
    const void *y,                  // second input: scalar y
    const GrB_Descriptor desc       // descriptor for C, mask, and A
) ;

// GrB_select is a generic method that applies an IndexUnaryOp to
// a matrix or vector, using any type of the scalar y.

// GrB_Vector_select_TYPE (w,m,acc,idxop,u,y,d)
// GrB_Matrix_select_TYPE (C,M,acc,idxop,A,y,d)

#if GxB_STDC_VERSION >= 201112L
#define GrB_select(C,Mask,accum,op,x,y,d)                               \
    _Generic                                                            \
    (                                                                   \
        (C),                                                            \
            GrB_Vector :                                                \
                _Generic                                                \
                (                                                       \
                    (y),                                                \
                        GB_CASES (GrB, Vector_select),                  \
                        default:  GrB_Vector_select_Scalar              \
                ),                                                      \
            GrB_Matrix :                                                \
                _Generic                                                \
                (                                                       \
                    (y),                                                \
                        GB_CASES (GrB, Matrix_select),                  \
                        default:  GrB_Matrix_select_Scalar              \
                )                                                       \
    )                                                                   \
    (C, Mask, accum, op, x, y, d)
#endif

//==============================================================================
// GxB_select: matrix and vector selection (DEPRECATED: use GrB_select instead)
//==============================================================================

GrB_Info GxB_Vector_select (GrB_Vector w, const GrB_Vector mask,
    const GrB_BinaryOp accum, const GxB_SelectOp op, const GrB_Vector u,
    const GrB_Scalar Thunk, const GrB_Descriptor desc) ;
GrB_Info GxB_Matrix_select (GrB_Matrix C, const GrB_Matrix Mask,
    const GrB_BinaryOp accum, const GxB_SelectOp op, const GrB_Matrix A,
    const GrB_Scalar Thunk, const GrB_Descriptor desc) ;
#if GxB_STDC_VERSION >= 201112L
#define GxB_select(C,Mask,accum,op,A,Thunk,desc) _Generic \
((C), GrB_Vector:GxB_Vector_select , GrB_Matrix:GxB_Matrix_select ) \
(C, Mask, accum, op, A, Thunk, desc)
#endif

//==============================================================================
// GrB_reduce: matrix and vector reduction
//==============================================================================

// Reduce the entries in a matrix to a vector, a column vector t such that
// t(i) = sum (A (i,:)), and where "sum" is a commutative and associative
// monoid with an identity value.  A can be transposed, which reduces down the
// columns instead of the rows.

// For GrB_Matrix_reduce_BinaryOp, the GrB_BinaryOp op must correspond to a
// known built-in monoid:
//
//      operator                data-types (all built-in)
//      ----------------------  ---------------------------
//      MIN, MAX                INT*, UINT*, FP*
//      TIMES, PLUS             INT*, UINT*, FP*, FC*
//      ANY                     INT*, UINT*, FP*, FC*, BOOL
//      LOR, LAND, LXOR, EQ     BOOL
//      BOR, BAND, BXOR, BXNOR  UINT*

GrB_Info GrB_Matrix_reduce_Monoid   // w<mask> = accum (w,reduce(A))
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector mask,          // optional mask for w, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for z=accum(w,t)
    const GrB_Monoid monoid,        // reduce operator for t=reduce(A)
    const GrB_Matrix A,             // first input:  matrix A
    const GrB_Descriptor desc       // descriptor for w, mask, and A
) ;

GrB_Info GrB_Matrix_reduce_BinaryOp // w<mask> = accum (w,reduce(A))
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector mask,          // optional mask for w, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for z=accum(w,t)
    const GrB_BinaryOp op,          // reduce operator for t=reduce(A)
    const GrB_Matrix A,             // first input:  matrix A
    const GrB_Descriptor desc       // descriptor for w, mask, and A
) ;

//------------------------------------------------------------------------------
// reduce a vector to a scalar
//------------------------------------------------------------------------------

// Reduce entries in a vector to a scalar, c = accum (c, reduce_to_scalar(u))

GrB_Info GrB_Vector_reduce_BOOL     // c = accum (c, reduce_to_scalar (u))
(
    bool *c,                        // result scalar
    const GrB_BinaryOp accum,       // optional accum for c=accum(c,t)
    const GrB_Monoid monoid,        // monoid to do the reduction
    const GrB_Vector u,             // vector to reduce
    const GrB_Descriptor desc
) ;

GrB_Info GrB_Vector_reduce_INT8     // c = accum (c, reduce_to_scalar (u))
(
    int8_t *c,                      // result scalar
    const GrB_BinaryOp accum,       // optional accum for c=accum(c,t)
    const GrB_Monoid monoid,        // monoid to do the reduction
    const GrB_Vector u,             // vector to reduce
    const GrB_Descriptor desc
) ;

GrB_Info GrB_Vector_reduce_UINT8    // c = accum (c, reduce_to_scalar (u))
(
    uint8_t *c,                     // result scalar
    const GrB_BinaryOp accum,       // optional accum for c=accum(c,t)
    const GrB_Monoid monoid,        // monoid to do the reduction
    const GrB_Vector u,             // vector to reduce
    const GrB_Descriptor desc
) ;

GrB_Info GrB_Vector_reduce_INT16    // c = accum (c, reduce_to_scalar (u))
(
    int16_t *c,                     // result scalar
    const GrB_BinaryOp accum,       // optional accum for c=accum(c,t)
    const GrB_Monoid monoid,        // monoid to do the reduction
    const GrB_Vector u,             // vector to reduce
    const GrB_Descriptor desc
) ;

GrB_Info GrB_Vector_reduce_UINT16   // c = accum (c, reduce_to_scalar (u))
(
    uint16_t *c,                    // result scalar
    const GrB_BinaryOp accum,       // optional accum for c=accum(c,t)
    const GrB_Monoid monoid,        // monoid to do the reduction
    const GrB_Vector u,             // vector to reduce
    const GrB_Descriptor desc
) ;

GrB_Info GrB_Vector_reduce_INT32    // c = accum (c, reduce_to_scalar (u))
(
    int32_t *c,                     // result scalar
    const GrB_BinaryOp accum,       // optional accum for c=accum(c,t)
    const GrB_Monoid monoid,        // monoid to do the reduction
    const GrB_Vector u,             // vector to reduce
    const GrB_Descriptor desc
) ;

GrB_Info GrB_Vector_reduce_UINT32   // c = accum (c, reduce_to_scalar (u))
(
    uint32_t *c,                    // result scalar
    const GrB_BinaryOp accum,       // optional accum for c=accum(c,t)
    const GrB_Monoid monoid,        // monoid to do the reduction
    const GrB_Vector u,             // vector to reduce
    const GrB_Descriptor desc
) ;

GrB_Info GrB_Vector_reduce_INT64    // c = accum (c, reduce_to_scalar (u))
(
    int64_t *c,                     // result scalar
    const GrB_BinaryOp accum,       // optional accum for c=accum(c,t)
    const GrB_Monoid monoid,        // monoid to do the reduction
    const GrB_Vector u,             // vector to reduce
    const GrB_Descriptor desc
) ;

GrB_Info GrB_Vector_reduce_UINT64   // c = accum (c, reduce_to_scalar (u))
(
    uint64_t *c,                    // result scalar
    const GrB_BinaryOp accum,       // optional accum for c=accum(c,t)
    const GrB_Monoid monoid,        // monoid to do the reduction
    const GrB_Vector u,             // vector to reduce
    const GrB_Descriptor desc
) ;

GrB_Info GrB_Vector_reduce_FP32     // c = accum (c, reduce_to_scalar (u))
(
    float *c,                       // result scalar
    const GrB_BinaryOp accum,       // optional accum for c=accum(c,t)
    const GrB_Monoid monoid,        // monoid to do the reduction
    const GrB_Vector u,             // vector to reduce
    const GrB_Descriptor desc
) ;

GrB_Info GrB_Vector_reduce_FP64     // c = accum (c, reduce_to_scalar (u))
(
    double *c,                      // result scalar
    const GrB_BinaryOp accum,       // optional accum for c=accum(c,t)
    const GrB_Monoid monoid,        // monoid to do the reduction
    const GrB_Vector u,             // vector to reduce
    const GrB_Descriptor desc
) ;

GrB_Info GxB_Vector_reduce_FC32     // c = accum (c, reduce_to_scalar (u))
(
    GxB_FC32_t *c,                  // result scalar
    const GrB_BinaryOp accum,       // optional accum for c=accum(c,t)
    const GrB_Monoid monoid,        // monoid to do the reduction
    const GrB_Vector u,             // vector to reduce
    const GrB_Descriptor desc
) ;

GrB_Info GxB_Vector_reduce_FC64     // c = accum (c, reduce_to_scalar (u))
(
    GxB_FC64_t *c,                  // result scalar
    const GrB_BinaryOp accum,       // optional accum for c=accum(c,t)
    const GrB_Monoid monoid,        // monoid to do the reduction
    const GrB_Vector u,             // vector to reduce
    const GrB_Descriptor desc
) ;

GrB_Info GrB_Vector_reduce_UDT      // c = accum (c, reduce_to_scalar (u))
(
    void *c,                        // result scalar
    const GrB_BinaryOp accum,       // optional accum for c=accum(c,t)
    const GrB_Monoid monoid,        // monoid to do the reduction
    const GrB_Vector u,             // vector to reduce
    const GrB_Descriptor desc
) ;

GrB_Info GrB_Vector_reduce_Monoid_Scalar   // c = accum(c,reduce_to_scalar(u))
(
    GrB_Scalar c,                   // result scalar
    const GrB_BinaryOp accum,       // optional accum for c=accum(c,t)
    const GrB_Monoid monoid,        // monoid to do the reduction
    const GrB_Vector u,             // vector to reduce
    const GrB_Descriptor desc
) ;

GrB_Info GrB_Vector_reduce_BinaryOp_Scalar
(
    GrB_Scalar c,                   // result scalar
    const GrB_BinaryOp accum,       // optional accum for c=accum(c,t)
    const GrB_BinaryOp op,          // binary op to do the reduction
    const GrB_Vector u,             // vector to reduce
    const GrB_Descriptor desc
) ;

//------------------------------------------------------------------------------
// reduce a matrix to a scalar
//------------------------------------------------------------------------------

// Reduce entries in a matrix to a scalar, c = accum (c, reduce_to_scalar(A))

GrB_Info GrB_Matrix_reduce_BOOL     // c = accum (c, reduce_to_scalar (A))
(
    bool *c,                        // result scalar
    const GrB_BinaryOp accum,       // optional accum for c=accum(c,t)
    const GrB_Monoid monoid,        // monoid to do the reduction
    const GrB_Matrix A,             // matrix to reduce
    const GrB_Descriptor desc
) ;

GrB_Info GrB_Matrix_reduce_INT8     // c = accum (c, reduce_to_scalar (A))
(
    int8_t *c,                      // result scalar
    const GrB_BinaryOp accum,       // optional accum for c=accum(c,t)
    const GrB_Monoid monoid,        // monoid to do the reduction
    const GrB_Matrix A,             // matrix to reduce
    const GrB_Descriptor desc
) ;

GrB_Info GrB_Matrix_reduce_UINT8    // c = accum (c, reduce_to_scalar (A))
(
    uint8_t *c,                     // result scalar
    const GrB_BinaryOp accum,       // optional accum for c=accum(c,t)
    const GrB_Monoid monoid,        // monoid to do the reduction
    const GrB_Matrix A,             // matrix to reduce
    const GrB_Descriptor desc
) ;

GrB_Info GrB_Matrix_reduce_INT16    // c = accum (c, reduce_to_scalar (A))
(
    int16_t *c,                     // result scalar
    const GrB_BinaryOp accum,       // optional accum for c=accum(c,t)
    const GrB_Monoid monoid,        // monoid to do the reduction
    const GrB_Matrix A,             // matrix to reduce
    const GrB_Descriptor desc
) ;

GrB_Info GrB_Matrix_reduce_UINT16   // c = accum (c, reduce_to_scalar (A))
(
    uint16_t *c,                    // result scalar
    const GrB_BinaryOp accum,       // optional accum for c=accum(c,t)
    const GrB_Monoid monoid,        // monoid to do the reduction
    const GrB_Matrix A,             // matrix to reduce
    const GrB_Descriptor desc
) ;

GrB_Info GrB_Matrix_reduce_INT32    // c = accum (c, reduce_to_scalar (A))
(
    int32_t *c,                     // result scalar
    const GrB_BinaryOp accum,       // optional accum for c=accum(c,t)
    const GrB_Monoid monoid,        // monoid to do the reduction
    const GrB_Matrix A,             // matrix to reduce
    const GrB_Descriptor desc
) ;

GrB_Info GrB_Matrix_reduce_UINT32   // c = accum (c, reduce_to_scalar (A))
(
    uint32_t *c,                    // result scalar
    const GrB_BinaryOp accum,       // optional accum for c=accum(c,t)
    const GrB_Monoid monoid,        // monoid to do the reduction
    const GrB_Matrix A,             // matrix to reduce
    const GrB_Descriptor desc
) ;

GrB_Info GrB_Matrix_reduce_INT64    // c = accum (c, reduce_to_scalar (A))
(
    int64_t *c,                     // result scalar
    const GrB_BinaryOp accum,       // optional accum for c=accum(c,t)
    const GrB_Monoid monoid,        // monoid to do the reduction
    const GrB_Matrix A,             // matrix to reduce
    const GrB_Descriptor desc
) ;

GrB_Info GrB_Matrix_reduce_UINT64   // c = accum (c, reduce_to_scalar (A))
(
    uint64_t *c,                    // result scalar
    const GrB_BinaryOp accum,       // optional accum for c=accum(c,t)
    const GrB_Monoid monoid,        // monoid to do the reduction
    const GrB_Matrix A,             // matrix to reduce
    const GrB_Descriptor desc
) ;

GrB_Info GrB_Matrix_reduce_FP32     // c = accum (c, reduce_to_scalar (A))
(
    float *c,                       // result scalar
    const GrB_BinaryOp accum,       // optional accum for c=accum(c,t)
    const GrB_Monoid monoid,        // monoid to do the reduction
    const GrB_Matrix A,             // matrix to reduce
    const GrB_Descriptor desc
) ;

GrB_Info GrB_Matrix_reduce_FP64     // c = accum (c, reduce_to_scalar (A))
(
    double *c,                      // result scalar
    const GrB_BinaryOp accum,       // optional accum for c=accum(c,t)
    const GrB_Monoid monoid,        // monoid to do the reduction
    const GrB_Matrix A,             // matrix to reduce
    const GrB_Descriptor desc
) ;

GrB_Info GxB_Matrix_reduce_FC32     // c = accum (c, reduce_to_scalar (A))
(
    GxB_FC32_t *c,                  // result scalar
    const GrB_BinaryOp accum,       // optional accum for c=accum(c,t)
    const GrB_Monoid monoid,        // monoid to do the reduction
    const GrB_Matrix A,             // matrix to reduce
    const GrB_Descriptor desc
) ;

GrB_Info GxB_Matrix_reduce_FC64     // c = accum (c, reduce_to_scalar (A))
(
    GxB_FC64_t *c,                  // result scalar
    const GrB_BinaryOp accum,       // optional accum for c=accum(c,t)
    const GrB_Monoid monoid,        // monoid to do the reduction
    const GrB_Matrix A,             // matrix to reduce
    const GrB_Descriptor desc
) ;

GrB_Info GrB_Matrix_reduce_UDT      // c = accum (c, reduce_to_scalar (A))
(
    void *c,                        // result scalar
    const GrB_BinaryOp accum,       // optional accum for c=accum(c,t)
    const GrB_Monoid monoid,        // monoid to do the reduction
    const GrB_Matrix A,             // matrix to reduce
    const GrB_Descriptor desc
) ;

GrB_Info GrB_Matrix_reduce_Monoid_Scalar   // c = accum(c,reduce_to_scalar(A))
(
    GrB_Scalar c,                   // result scalar
    const GrB_BinaryOp accum,       // optional accum for c=accum(c,t)
    const GrB_Monoid monoid,        // monoid to do the reduction
    const GrB_Matrix A,             // matrix to reduce
    const GrB_Descriptor desc
) ;

GrB_Info GrB_Matrix_reduce_BinaryOp_Scalar
(
    GrB_Scalar S,                   // result scalar
    const GrB_BinaryOp accum,       // optional accum for c=accum(c,t)
    const GrB_BinaryOp op,          // binary op to do the reduction
    const GrB_Matrix A,             // matrix to reduce
    const GrB_Descriptor desc
) ;

//------------------------------------------------------------------------------
// GrB_reduce: generic matrix/vector reduction to a vector or scalar
//------------------------------------------------------------------------------

// GrB_reduce is a generic function that provides access to all GrB_*reduce*
// functions:

// reduce matrix to vector:
// GrB_Matrix_reduce_Monoid   (w,mask,acc,mo,A,d) // w<mask> = acc (w,reduce(A))
// GrB_Matrix_reduce_BinaryOp (w,mask,acc,op,A,d) // w<mask> = acc (w,reduce(A))

// reduce matrix to scalar:
// GrB_Vector_reduce_[SCALAR] (c,acc,monoid,u,d)  // c = acc (c,reduce(u))
// GrB_Matrix_reduce_[SCALAR] (c,acc,monoid,A,d)  // c = acc (c,reduce(A))
// GrB_Vector_reduce_Monoid_Scalar (s,acc,monoid,u,d)   // s = acc (s,reduce(u))
// GrB_Matrix_reduce_Monoid_Scalar (s,acc,monoid,A,d)   // s = acc (s,reduce(A))
// GrB_Vector_reduce_BinaryOp_Scalar (s,acc,op,u,d)     // s = acc (s,reduce(u))
// GrB_Matrix_reduce_BinaryOp_Scalar (s,acc,op,A,d)     // s = acc (s,reduce(A))

#if GxB_STDC_VERSION >= 201112L
#define GB_REDUCE_TO_SCALAR(kind,c,op)                                         \
    _Generic                                                                   \
    (                                                                          \
        (c),                                                                   \
            GB_PCASES (GrB, GB_CONCAT ( kind, _reduce,, )),                    \
            default:                                                           \
                _Generic                                                       \
                (                                                              \
                    (op),                                                      \
                        GrB_BinaryOp :                                         \
                                GB_CONCAT (GrB,_,kind,_reduce_BinaryOp_Scalar),\
                        default:  GB_CONCAT (GrB,_,kind,_reduce_Monoid_Scalar) \
                )                                                              \
    )

#define GrB_reduce(arg1,arg2,arg3,arg4,...)                                 \
    _Generic                                                                \
    (                                                                       \
        (arg4),                                                             \
                  GrB_Vector   : GB_REDUCE_TO_SCALAR (Vector, arg1, arg3),  \
                  GrB_Matrix   : GB_REDUCE_TO_SCALAR (Matrix, arg1, arg3),  \
                  GrB_Monoid   : GrB_Matrix_reduce_Monoid   ,               \
                  GrB_BinaryOp : GrB_Matrix_reduce_BinaryOp                 \
    )                                                                       \
    (arg1, arg2, arg3, arg4, __VA_ARGS__)
#endif

//==============================================================================
// GrB_transpose: matrix transpose
//==============================================================================

GrB_Info GrB_transpose              // C<Mask> = accum (C, A')
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix Mask,          // optional mask for C, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C,T)
    const GrB_Matrix A,             // first input:  matrix A
    const GrB_Descriptor desc       // descriptor for C, Mask, and A
) ;

//==============================================================================
// GrB_kronecker:  Kronecker product
//==============================================================================

// GxB_kron is historical; use GrB_kronecker instead
GrB_Info GxB_kron                   // C<Mask> = accum(C,kron(A,B)) (historical)
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix Mask,          // optional mask for C, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C,T)
    const GrB_BinaryOp op,          // defines '*' for T=kron(A,B)
    const GrB_Matrix A,             // first input:  matrix A
    const GrB_Matrix B,             // second input: matrix B
    const GrB_Descriptor desc       // descriptor for C, Mask, A, and B
) ;

GrB_Info GrB_Matrix_kronecker_BinaryOp  // C<M> = accum (C, kron(A,B))
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix M,             // optional mask for C, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C,T)
    const GrB_BinaryOp op,          // defines '*' for T=kron(A,B)
    const GrB_Matrix A,             // first input:  matrix A
    const GrB_Matrix B,             // second input: matrix B
    const GrB_Descriptor desc       // descriptor for C, M, A, and B
) ;

GrB_Info GrB_Matrix_kronecker_Monoid  // C<M> = accum (C, kron(A,B))
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix M,             // optional mask for C, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C,T)
    const GrB_Monoid monoid,        // defines '*' for T=kron(A,B)
    const GrB_Matrix A,             // first input:  matrix A
    const GrB_Matrix B,             // second input: matrix B
    const GrB_Descriptor desc       // descriptor for C, M, A, and B
) ;

GrB_Info GrB_Matrix_kronecker_Semiring  // C<M> = accum (C, kron(A,B))
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix M,             // optional mask for C, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C,T)
    const GrB_Semiring semiring,    // defines '*' for T=kron(A,B)
    const GrB_Matrix A,             // first input:  matrix A
    const GrB_Matrix B,             // second input: matrix B
    const GrB_Descriptor desc       // descriptor for C, M, A, and B
) ;

#if GxB_STDC_VERSION >= 201112L
#define GrB_kronecker(C,Mask,accum,op,A,B,desc)                     \
    _Generic                                                        \
    (                                                               \
        (op),                                                       \
                  GrB_Semiring : GrB_Matrix_kronecker_Semiring ,    \
                  GrB_Monoid   : GrB_Matrix_kronecker_Monoid   ,    \
                  GrB_BinaryOp : GrB_Matrix_kronecker_BinaryOp      \
    )                                                               \
    (C, Mask, accum, op, A, B, desc)
#endif

//==============================================================================
// GrB_*_resize:  change the size of a matrix or vector
//==============================================================================

// If the dimensions decrease, entries that fall outside the resized matrix or
// vector are deleted.

GrB_Info GrB_Matrix_resize      // change the size of a matrix
(
    GrB_Matrix C,               // matrix to modify
    GrB_Index nrows_new,        // new number of rows in matrix
    GrB_Index ncols_new         // new number of columns in matrix
) ;

GrB_Info GrB_Vector_resize      // change the size of a vector
(
    GrB_Vector w,               // vector to modify
    GrB_Index nrows_new         // new number of rows in vector
) ;

// GxB_*_resize are identical to the GrB_*resize methods above
GrB_Info GxB_Matrix_resize      // change the size of a matrix (historical)
(
    GrB_Matrix C,               // matrix to modify
    GrB_Index nrows_new,        // new number of rows in matrix
    GrB_Index ncols_new         // new number of columns in matrix
) ;

GrB_Info GxB_Vector_resize      // change the size of a vector (historical)
(
    GrB_Vector w,               // vector to modify
    GrB_Index nrows_new         // new number of rows in vector
) ;

// GxB_resize is a generic function for resizing a matrix or vector:

// GrB_Vector_resize (u,nrows_new)
// GrB_Matrix_resize (A,nrows_new,ncols_new)

#if GxB_STDC_VERSION >= 201112L
#define GxB_resize(arg1,...)                                \
    _Generic                                                \
    (                                                       \
        (arg1),                                             \
              GrB_Vector : GrB_Vector_resize ,              \
              GrB_Matrix : GrB_Matrix_resize                \
    )                                                       \
    (arg1, __VA_ARGS__)
#endif

//------------------------------------------------------------------------------
// GxB_Matrix_Pending:  Checks to see if matrix has pending operations
//------------------------------------------------------------------------------

GrB_Info GxB_Matrix_Pending
(
	GrB_Matrix A,                   // matrix to query
	bool *pending                   // are there any pending operations
) ;

//==============================================================================
// GxB_fprint and GxB_print: print the contents of a GraphBLAS object
//==============================================================================

// GxB_fprint (object, GxB_Print_Level pr, FILE *f) prints the contents of any
// of the 9 GraphBLAS objects to the file f, and also does an extensive test on
// the object to determine if it is valid.  It returns one of the following
// error conditions:
//
//      GrB_SUCCESS               object is valid
//      GrB_UNINITIALIZED_OBJECT  object is not initialized
//      GrB_INVALID_OBJECT        object is not valid
//      GrB_NULL_POINTER          object is a NULL pointer
//      GrB_INVALID_VALUE         fprintf returned an I/O error; see the ANSI C
//                                errno or GrB_error( )for details.
//
// GxB_fprint does not modify the status of any object.  If a matrix or vector
// has not been completed, the pending computations are guaranteed to *not* be
// performed by GxB_fprint.  The reason is simple.  It is possible for a bug in
// the user application (such as accessing memory outside the bounds of an
// array) to mangle the internal content of a GraphBLAS object, and GxB_fprint
// can be a helpful tool to track down this bug.  If GxB_fprint attempted to
// complete any computations prior to printing or checking the contents of the
// matrix or vector, then further errors could occur, including a segfault.
//
// The type-specific functions include an additional argument, the name string.
// The name is printed at the beginning of the display (assuming pr is not
// GxB_SILENT) so that the object can be more easily identified in the output.
// For the type-generic methods GxB_fprint and GxB_print, the name string is
// the variable name of the object itself.
//
// If f is NULL, stdout is used; this is not an error condition.  If pr is
// outside the bounds 0 to 3, negative values are treated as GxB_SILENT, and
// values > 3 are treated as GxB_COMPLETE.  If name is NULL, it is treated as
// the empty string.
//
// GxB_print (object, GxB_Print_Level pr) is the same as GxB_fprint, except
// that it prints the contents with printf instead of fprintf to a file f.
//
// The exact content and format of what is printed is implementation-dependent,
// and will change from version to version of SuiteSparse:GraphBLAS.  Do not
// attempt to rely on the exact content or format by trying to parse the
// resulting output via another program.  The intent of these functions is to
// produce a report of the object for visual inspection.

GrB_Info GxB_Type_fprint            // print and check a GrB_Type
(
    GrB_Type type,                  // object to print and check
    const char *name,               // name of the object
    GxB_Print_Level pr,             // print level
    FILE *f                         // file for output
) ;

GrB_Info GxB_UnaryOp_fprint         // print and check a GrB_UnaryOp
(
    GrB_UnaryOp unaryop,            // object to print and check
    const char *name,               // name of the object
    GxB_Print_Level pr,             // print level
    FILE *f                         // file for output
) ;

GrB_Info GxB_BinaryOp_fprint        // print and check a GrB_BinaryOp
(
    GrB_BinaryOp binaryop,          // object to print and check
    const char *name,               // name of the object
    GxB_Print_Level pr,             // print level
    FILE *f                         // file for output
) ;

GrB_Info GxB_IndexUnaryOp_fprint    // print and check a GrB_IndexUnaryOp
(
    GrB_IndexUnaryOp op,            // object to print and check
    const char *name,               // name of the object
    GxB_Print_Level pr,             // print level
    FILE *f                         // file for output
) ;

GrB_Info GxB_SelectOp_fprint (GxB_SelectOp op, const char *name,
    GxB_Print_Level pr, FILE *f) ;

GrB_Info GxB_Monoid_fprint          // print and check a GrB_Monoid
(
    GrB_Monoid monoid,              // object to print and check
    const char *name,               // name of the object
    GxB_Print_Level pr,             // print level
    FILE *f                         // file for output
) ;

GrB_Info GxB_Semiring_fprint        // print and check a GrB_Semiring
(
    GrB_Semiring semiring,          // object to print and check
    const char *name,               // name of the object
    GxB_Print_Level pr,             // print level
    FILE *f                         // file for output
) ;

GrB_Info GxB_Descriptor_fprint      // print and check a GrB_Descriptor
(
    GrB_Descriptor descriptor,      // object to print and check
    const char *name,               // name of the object
    GxB_Print_Level pr,             // print level
    FILE *f                         // file for output
) ;

GrB_Info GxB_Matrix_fprint          // print and check a GrB_Matrix
(
    GrB_Matrix A,                   // object to print and check
    const char *name,               // name of the object
    GxB_Print_Level pr,             // print level
    FILE *f                         // file for output
) ;

GrB_Info GxB_Vector_fprint          // print and check a GrB_Vector
(
    GrB_Vector v,                   // object to print and check
    const char *name,               // name of the object
    GxB_Print_Level pr,             // print level
    FILE *f                         // file for output
) ;

GrB_Info GxB_Scalar_fprint          // print and check a GrB_Scalar
(
    GrB_Scalar s,                   // object to print and check
    const char *name,               // name of the object
    GxB_Print_Level pr,             // print level
    FILE *f                         // file for output
) ;

GrB_Info GxB_Context_fprint         // print and check a GxB_Context
(
    GxB_Context Context,            // object to print and check
    const char *name,               // name of the object
    GxB_Print_Level pr,             // print level
    FILE *f                         // file for output
) ;

#if GxB_STDC_VERSION >= 201112L
#define GxB_fprint(object,pr,f)                                 \
    _Generic                                                    \
    (                                                           \
        (object),                                               \
                  GrB_Type         : GxB_Type_fprint         ,  \
                  GrB_UnaryOp      : GxB_UnaryOp_fprint      ,  \
                  GrB_BinaryOp     : GxB_BinaryOp_fprint     ,  \
                  GrB_IndexUnaryOp : GxB_IndexUnaryOp_fprint ,  \
                  GxB_SelectOp     : GxB_SelectOp_fprint     ,  \
                  GrB_Monoid       : GxB_Monoid_fprint       ,  \
                  GrB_Semiring     : GxB_Semiring_fprint     ,  \
                  GrB_Scalar       : GxB_Scalar_fprint       ,  \
                  GrB_Vector       : GxB_Vector_fprint       ,  \
                  GrB_Matrix       : GxB_Matrix_fprint       ,  \
                  GrB_Descriptor   : GxB_Descriptor_fprint   ,  \
                  GxB_Context      : GxB_Context_fprint         \
    )                                                           \
    (object, GB_STR(object), pr, f)

#define GxB_print(object,pr) GxB_fprint(object,pr,NULL)
#endif

//==============================================================================
// Matrix and vector import/export/pack/unpack
//==============================================================================

// The import/export/pack/unpack functions allow the user application to create
// a GrB_Matrix or GrB_Vector object, and to extract its contents, faster and
// with less memory overhead than the GrB_*_build and GrB_*_extractTuples
// functions.

// The semantics of import/export/pack/unpack are the same as the "move
// constructor" in C++.  On import, the user provides a set of arrays that have
// been previously allocated via the ANSI C malloc function.  The arrays define
// the content of the matrix or vector.  Unlike GrB_*_build, the GraphBLAS
// library then takes ownership of the user's input arrays and may either (a)
// incorporate them into its internal data structure for the new GrB_Matrix or
// GrB_Vector, potentially creating the GrB_Matrix or GrB_Vector in constant
// time with no memory copying performed, or (b) if the library does not
// support the import format directly, then it may convert the input to its
// internal format, and then free the user's input arrays.  GraphBLAS may also
// choose to use a mix of the two strategies.  In either case, the input arrays
// are no longer "owned" by the user application.  If A is a GrB_Matrix created
// by an import/pack, the user input arrays are freed no later than GrB_free
// (&A), and may be freed earlier, at the discretion of the GraphBLAS library.
// The data structure of the GrB_Matrix and GrB_Vector remain opaque.

// The export/unpack of a GrB_Matrix or GrB_Vector is symmetric with the import
// operation.  The export is destructive, where the GrB_Matrix or GrB_Vector no
// longer exists when the export completes.  The GrB_Matrix or GrB_Vector
// exists after an unpack operation, just with no entries.  In both export and
// unpack, the user is returned several arrays that contain the matrix or
// vector in the requested format.  Ownership of these arrays is given to the
// user application, which is then responsible for freeing them via the ANSI C
// free function.  If the output format is supported by the GraphBLAS library,
// then these arrays may be returned to the user application in O(1) time and
// with no memory copying performed.  Otherwise, the GraphBLAS library will
// create the output arrays for the user (via the ANSI C malloc function), fill
// them with the GrB_Matrix or GrB_Vector data, and then return the newly
// allocated arrays to the user.

// Eight different formats are provided for import/export.  For each format,
// the Ax array has a C-type <type> corresponding to one of the 13 built-in
// types in GraphBLAS (bool, int*_t, uint*_t, float, double, float complex, or
// double complex), or a user-defined type.

// On import/pack, the required user arrays Ah, Ap, Ab, Ai, Aj, and/or Ax must
// be non-NULL pointers to memory space allocated by the ANSI C malloc (or
// calloc, or realloc), unless nzmax is zero (in which case the Ab, Ai, Aj, Ax,
// vb, vi, and vx arrays may all be NULL).  For the import, A (or GrB_Vector v)
// is undefined on input, just like GrB_*_new, the GrB_Matrix.  If the import
// is successful, the GrB_Matrix A or GrB_Vector v is created, and the pointers
// to the user input arrays have been set to NULL.  These user arrays have
// either been incorporated directly into the GrB_Matrix A or GrB_Vector v, in
// which case the user input arrays will eventually be freed by GrB_free (&A),
// or their contents have been copied and the arrays freed.  This decision is
// made by the GraphBLAS library itself, and the user application has no
// control over this decision.

// If any of the arrays Ab, Aj, Ai, Ax, vb, vi, or vx have zero size (with
// nzmax of zero), they are allowed to be be NULL pointers on input.

// A matrix or vector may be "iso", where all entries present in the pattern
// have the same value.  In this case, the boolean iso flag is true, and the
// corresponding numerical array (Ax for matrices, vx for vectors, below) need
// be only large enough to hold a single value.

// No error checking is performed on the content of the user input arrays.  If
// the user input arrays do not conform to the precise specifications above,
// results are undefined.  No typecasting of the values of the matrix or vector
// entries is performed on import or export.

// SuiteSparse:GraphBLAS supports all eight formats natively (CSR, CSC,
// HyperCSR, and HyperCSC, BitmapR, BitmapC, FullR, FullC).  For vectors, only
// CSC, BitmapC, and FullC formats are used.  On import, the all eight formats
// take O(1) time and memory to import.  On export, if the GrB_Matrix or
// GrB_Vector is already in this particular format, then the export takes O(1)
// time and no memory copying is performed.

// If the import is not successful, the GxB_Matrix_import_* functions return A
// as NULL, GxB_Vector_import returns v as NULL, and the user input arrays are
// neither modified nor freed.  They are still owned by the user application.

// If the input data is untrusted, use the following descriptor setting for
// GxB_Matrix_import* and GxB_Matrix_pack*.  The import/pack will be slower,
// but secure.  GrB_Matrix_import uses the slow, secure method, since it has
// no descriptor input.
//
//      GxB_set (desc, GxB_IMPORT, GxB_SECURE_IMPORT) ;

// As of v5.2.0, GxB_*import* and GxB_*export* are declared historical.  Use
// GxB_*pack* and GxB_*unpack* instead.  The GxB import/export will be kept
// but only documented here, not in the User Guide.

//------------------------------------------------------------------------------
// GxB_Matrix_pack_CSR: pack a CSR matrix
//------------------------------------------------------------------------------

GrB_Info GxB_Matrix_import_CSR  // historical: use GxB_Matrix_pack_CSR
(
    GrB_Matrix *A,      // handle of matrix to create
    GrB_Type type,      // type of matrix to create
    GrB_Index nrows,    // number of rows of the matrix
    GrB_Index ncols,    // number of columns of the matrix
    GrB_Index **Ap,     // row "pointers", Ap_size >= (nrows+1)* sizeof(int64_t)
    GrB_Index **Aj,     // column indices, Aj_size >= nvals(A) * sizeof(int64_t)
    void **Ax,          // values, Ax_size >= nvals(A) * (type size)
                        // or Ax_size >= (type size), if iso is true
    GrB_Index Ap_size,  // size of Ap in bytes
    GrB_Index Aj_size,  // size of Aj in bytes
    GrB_Index Ax_size,  // size of Ax in bytes
    bool iso,           // if true, A is iso
    bool jumbled,       // if true, indices in each row may be unsorted
    const GrB_Descriptor desc
) ;

GrB_Info GxB_Matrix_pack_CSR      // pack a CSR matrix
(
    GrB_Matrix A,       // matrix to create (type, nrows, ncols unchanged)
    GrB_Index **Ap,     // row "pointers", Ap_size >= (nrows+1)* sizeof(int64_t)
    GrB_Index **Aj,     // column indices, Aj_size >= nvals(A) * sizeof(int64_t)
    void **Ax,          // values, Ax_size >= nvals(A) * (type size)
                        // or Ax_size >= (type size), if iso is true
    GrB_Index Ap_size,  // size of Ap in bytes
    GrB_Index Aj_size,  // size of Aj in bytes
    GrB_Index Ax_size,  // size of Ax in bytes
    bool iso,           // if true, A is iso
    bool jumbled,       // if true, indices in each row may be unsorted
    const GrB_Descriptor desc
) ;

    // CSR:  an nrows-by-ncols matrix with nvals entries in CSR format consists
    // of 3 arrays, where nvals = Ap [nrows]:
    //
    //          GrB_Index Ap [nrows+1], Aj [nvals] ; <type> Ax [nvals] ;
    //
    //      The column indices of entries in the ith row of the matrix are held
    //      in Aj [Ap [i] ... Ap[i+1]], and the corresponding values are held
    //      in the same positions in Ax.  Column indices must be in the range 0
    //      to ncols-1.  If jumbled is false, the column indices must appear in
    //      sorted order within each row.  No duplicate column indices may
    //      appear in any row.  Ap [0] must equal zero, and Ap [nrows] must
    //      equal nvals.  The Ap array must be of size nrows+1 (or larger), and
    //      the Aj and Ax arrays must have size at least nvals.  If nvals is
    //      zero, then the Aj and Ax arrays need not be present and can be
    //      NULL.

//------------------------------------------------------------------------------
// GxB_Matrix_pack_CSC: pack a CSC matrix
//------------------------------------------------------------------------------

GrB_Info GxB_Matrix_import_CSC  // historical: use GxB_Matrix_pack_CSC
(
    GrB_Matrix *A,      // handle of matrix to create
    GrB_Type type,      // type of matrix to create
    GrB_Index nrows,    // number of rows of the matrix
    GrB_Index ncols,    // number of columns of the matrix
    GrB_Index **Ap,     // col "pointers", Ap_size >= (ncols+1)*sizeof(int64_t)
    GrB_Index **Ai,     // row indices, Ai_size >= nvals(A)*sizeof(int64_t)
    void **Ax,          // values, Ax_size >= nvals(A) * (type size)
                        // or Ax_size >= (type size), if iso is true
    GrB_Index Ap_size,  // size of Ap in bytes
    GrB_Index Ai_size,  // size of Ai in bytes
    GrB_Index Ax_size,  // size of Ax in bytes
    bool iso,           // if true, A is iso
    bool jumbled,       // if true, indices in each column may be unsorted
    const GrB_Descriptor desc
) ;

GrB_Info GxB_Matrix_pack_CSC      // pack a CSC matrix
(
    GrB_Matrix A,       // matrix to create (type, nrows, ncols unchanged)
    GrB_Index **Ap,     // col "pointers", Ap_size >= (ncols+1)*sizeof(int64_t)
    GrB_Index **Ai,     // row indices, Ai_size >= nvals(A)*sizeof(int64_t)
    void **Ax,          // values, Ax_size >= nvals(A) * (type size)
                        // or Ax_size >= (type size), if iso is true
    GrB_Index Ap_size,  // size of Ap in bytes
    GrB_Index Ai_size,  // size of Ai in bytes
    GrB_Index Ax_size,  // size of Ax in bytes
    bool iso,           // if true, A is iso
    bool jumbled,       // if true, indices in each column may be unsorted
    const GrB_Descriptor desc
) ;

    // CSC:  an nrows-by-ncols matrix with nvals entries in CSC format consists
    // of 3 arrays, where nvals = Ap [ncols]:
    //
    //          GrB_Index Ap [ncols+1], Ai [nvals] ; <type> Ax [nvals] ;
    //
    //      The row indices of entries in the jth column of the matrix are held
    //      in Ai [Ap [j] ... Ap[j+1]], and the corresponding values are held
    //      in the same positions in Ax.  Row indices must be in the range 0 to
    //      nrows-1.  If jumbled is false, the row indices must appear in
    //      sorted order within each column.  No duplicate row indices may
    //      appear in any column.  Ap [0] must equal zero, and Ap [ncols] must
    //      equal nvals.  The Ap array must be of size ncols+1 (or larger), and
    //      the Ai and Ax arrays must have size at least nvals.  If nvals is
    //      zero, then the Ai and Ax arrays need not be present and can be
    //      NULL.

//------------------------------------------------------------------------------
// GxB_Matrix_pack_HyperCSR: pack a hypersparse CSR matrix
//------------------------------------------------------------------------------

GrB_Info GxB_Matrix_import_HyperCSR // historical: use GxB_Matrix_pack_HyperCSR
(
    GrB_Matrix *A,      // handle of matrix to create
    GrB_Type type,      // type of matrix to create
    GrB_Index nrows,    // number of rows of the matrix
    GrB_Index ncols,    // number of columns of the matrix
    GrB_Index **Ap,     // row "pointers", Ap_size >= (nvec+1)*sizeof(int64_t)
    GrB_Index **Ah,     // row indices, Ah_size >= nvec*sizeof(int64_t)
    GrB_Index **Aj,     // column indices, Aj_size >= nvals(A)*sizeof(int64_t)
    void **Ax,          // values, Ax_size >= nvals(A) * (type size)
                        // or Ax_size >= (type size), if iso is true
    GrB_Index Ap_size,  // size of Ap in bytes
    GrB_Index Ah_size,  // size of Ah in bytes
    GrB_Index Aj_size,  // size of Aj in bytes
    GrB_Index Ax_size,  // size of Ax in bytes
    bool iso,           // if true, A is iso
    GrB_Index nvec,     // number of rows that appear in Ah
    bool jumbled,       // if true, indices in each row may be unsorted
    const GrB_Descriptor desc
) ;

GrB_Info GxB_Matrix_pack_HyperCSR      // pack a hypersparse CSR matrix
(
    GrB_Matrix A,       // matrix to create (type, nrows, ncols unchanged)
    GrB_Index **Ap,     // row "pointers", Ap_size >= (nvec+1)*sizeof(int64_t)
    GrB_Index **Ah,     // row indices, Ah_size >= nvec*sizeof(int64_t)
    GrB_Index **Aj,     // column indices, Aj_size >= nvals(A)*sizeof(int64_t)
    void **Ax,          // values, Ax_size >= nvals(A) * (type size)
                        // or Ax_size >= (type size), if iso is true
    GrB_Index Ap_size,  // size of Ap in bytes
    GrB_Index Ah_size,  // size of Ah in bytes
    GrB_Index Aj_size,  // size of Aj in bytes
    GrB_Index Ax_size,  // size of Ax in bytes
    bool iso,           // if true, A is iso
    GrB_Index nvec,     // number of rows that appear in Ah
    bool jumbled,       // if true, indices in each row may be unsorted
    const GrB_Descriptor desc
) ;

    // HyperCSR: an nrows-by-ncols matrix with nvals entries and nvec
    // rows that may have entries in HyperCSR format consists of 4 arrays,
    // where nvals = Ap [nvec]:
    //
    //          GrB_Index Ah [nvec], Ap [nvec+1], Aj [nvals] ;
    //          <type> Ax [nvals] ;
    //
    //      The Aj and Ax arrays are the same for a matrix in CSR or HyperCSR
    //      format.  Only Ap and Ah differ.
    //
    //      The Ah array is a list of the row indices of rows that appear in
    //      the matrix.  It
    //      must appear in sorted order, and no duplicates may appear.  If i =
    //      Ah [k] is the kth row, then the column indices of the ith
    //      row appear in Aj [Ap [k] ... Ap [k+1]], and the corresponding
    //      values appear in the same locations in Ax.  Column indices must be
    //      in the range 0 to ncols-1, and must appear in sorted order within
    //      each row.  No duplicate column indices may appear in any row.  nvec
    //      may be zero, to denote an array with no entries.  The Ah array must
    //      be of size at least nvec, Ap must be of size at least nvec+1, and
    //      Aj and Ax must be at least of size nvals.  If nvals is zero, then
    //      the Aj and Ax arrays need not be present and can be NULL.

//------------------------------------------------------------------------------
// GxB_Matrix_pack_HyperCSC: pack a hypersparse CSC matrix
//------------------------------------------------------------------------------

GrB_Info GxB_Matrix_import_HyperCSC // historical: use GxB_Matrix_pack_HyperCSC
(
    GrB_Matrix *A,      // handle of matrix to create
    GrB_Type type,      // type of matrix to create
    GrB_Index nrows,    // number of rows of the matrix
    GrB_Index ncols,    // number of columns of the matrix
    GrB_Index **Ap,     // col "pointers", Ap_size >= (nvec+1)*sizeof(int64_t)
    GrB_Index **Ah,     // column indices, Ah_size >= nvec*sizeof(int64_t)
    GrB_Index **Ai,     // row indices, Ai_size >= nvals(A)*sizeof(int64_t)
    void **Ax,          // values, Ax_size >= nvals(A)*(type size)
                        // or Ax_size >= (type size), if iso is true
    GrB_Index Ap_size,  // size of Ap in bytes
    GrB_Index Ah_size,  // size of Ah in bytes
    GrB_Index Ai_size,  // size of Ai in bytes
    GrB_Index Ax_size,  // size of Ax in bytes
    bool iso,           // if true, A is iso
    GrB_Index nvec,     // number of columns that appear in Ah
    bool jumbled,       // if true, indices in each column may be unsorted
    const GrB_Descriptor desc
) ;

GrB_Info GxB_Matrix_pack_HyperCSC      // pack a hypersparse CSC matrix
(
    GrB_Matrix A,       // matrix to create (type, nrows, ncols unchanged)
    GrB_Index **Ap,     // col "pointers", Ap_size >= (nvec+1)*sizeof(int64_t)
    GrB_Index **Ah,     // column indices, Ah_size >= nvec*sizeof(int64_t)
    GrB_Index **Ai,     // row indices, Ai_size >= nvals(A)*sizeof(int64_t)
    void **Ax,          // values, Ax_size >= nvals(A)*(type size)
                        // or Ax_size >= (type size), if iso is true
    GrB_Index Ap_size,  // size of Ap in bytes
    GrB_Index Ah_size,  // size of Ah in bytes
    GrB_Index Ai_size,  // size of Ai in bytes
    GrB_Index Ax_size,  // size of Ax in bytes
    bool iso,           // if true, A is iso
    GrB_Index nvec,     // number of columns that appear in Ah
    bool jumbled,       // if true, indices in each column may be unsorted
    const GrB_Descriptor desc
) ;

    // HyperCSC: an nrows-by-ncols matrix with nvals entries and nvec
    // columns that may have entries in HyperCSC format consists of 4 arrays,
    // where nvals = Ap [nvec]:
    //
    //
    //          GrB_Index Ah [nvec], Ap [nvec+1], Ai [nvals] ;
    //          <type> Ax [nvals] ;
    //
    //      The Ai and Ax arrays are the same for a matrix in CSC or HyperCSC
    //      format.  Only Ap and Ah differ.
    //
    //      The Ah array is a list of the column indices of non-empty columns.
    //      It must appear in sorted order, and no duplicates may appear.  If j
    //      = Ah [k] is the kth non-empty column, then the row indices of the
    //      jth column appear in Ai [Ap [k] ... Ap [k+1]], and the
    //      corresponding values appear in the same locations in Ax.  Row
    //      indices must be in the range 0 to nrows-1, and must appear in
    //      sorted order within each column.  No duplicate row indices may
    //      appear in any column.  nvec may be zero, to denote an array with no
    //      entries.  The Ah array must be of size at least nvec, Ap must be of
    //      size at least nvec+1, and Ai and Ax must be at least of size nvals.
    //      If nvals is zero, then the Ai and Ax arrays need not be present and
    //      can be NULL.

//------------------------------------------------------------------------------
// GxB_Matrix_pack_BitmapR: pack a bitmap matrix, held by row
//------------------------------------------------------------------------------

GrB_Info GxB_Matrix_import_BitmapR // historical: use GxB_Matrix_pack_BitmapR
(
    GrB_Matrix *A,      // handle of matrix to create
    GrB_Type type,      // type of matrix to create
    GrB_Index nrows,    // number of rows of the matrix
    GrB_Index ncols,    // number of columns of the matrix
    int8_t **Ab,        // bitmap, Ab_size >= nrows*ncols
    void **Ax,          // values, Ax_size >= nrows*ncols * (type size)
                        // or Ax_size >= (type size), if iso is true
    GrB_Index Ab_size,  // size of Ab in bytes
    GrB_Index Ax_size,  // size of Ax in bytes
    bool iso,           // if true, A is iso
    GrB_Index nvals,    // # of entries in bitmap
    const GrB_Descriptor desc
) ;

GrB_Info GxB_Matrix_pack_BitmapR  // pack a bitmap matrix, held by row
(
    GrB_Matrix A,       // matrix to create (type, nrows, ncols unchanged)
    int8_t **Ab,        // bitmap, Ab_size >= nrows*ncols
    void **Ax,          // values, Ax_size >= nrows*ncols * (type size)
                        // or Ax_size >= (type size), if iso is true
    GrB_Index Ab_size,  // size of Ab in bytes
    GrB_Index Ax_size,  // size of Ax in bytes
    bool iso,           // if true, A is iso
    GrB_Index nvals,    // # of entries in bitmap
    const GrB_Descriptor desc
) ;

    // BitmapR: a dense format, but able to represent sparsity structure of A.
    //
    //          int8_t Ab [nrows*ncols] ;
    //          <type> Ax [nrows*ncols] ;
    //
    //      Ab and Ax are both of size nrows*ncols.  Ab [i*ncols+j] = 1 if the
    //      A(i,j) entry is present with value Ax [i*ncols+j], or 0 if A(i,j)
    //      is not present.  nvals must equal the number of 1's in the Ab
    //      array.

//------------------------------------------------------------------------------
// GxB_Matrix_pack_BitmapC: pack a bitmap matrix, held by column
//------------------------------------------------------------------------------

GrB_Info GxB_Matrix_import_BitmapC // historical: use GxB_Matrix_pack_BitmapC
(
    GrB_Matrix *A,      // handle of matrix to create
    GrB_Type type,      // type of matrix to create
    GrB_Index nrows,    // number of rows of the matrix
    GrB_Index ncols,    // number of columns of the matrix
    int8_t **Ab,        // bitmap, Ab_size >= nrows*ncols
    void **Ax,          // values, Ax_size >= nrows*ncols * (type size)
                        // or Ax_size >= (type size), if iso is true
    GrB_Index Ab_size,  // size of Ab in bytes
    GrB_Index Ax_size,  // size of Ax in bytes
    bool iso,           // if true, A is iso
    GrB_Index nvals,    // # of entries in bitmap
    const GrB_Descriptor desc
) ;

GrB_Info GxB_Matrix_pack_BitmapC  // pack a bitmap matrix, held by column
(
    GrB_Matrix A,       // matrix to create (type, nrows, ncols unchanged)
    int8_t **Ab,        // bitmap, Ab_size >= nrows*ncols
    void **Ax,          // values, Ax_size >= nrows*ncols * (type size)
                        // or Ax_size >= (type size), if iso is true
    GrB_Index Ab_size,  // size of Ab in bytes
    GrB_Index Ax_size,  // size of Ax in bytes
    bool iso,           // if true, A is iso
    GrB_Index nvals,    // # of entries in bitmap
    const GrB_Descriptor desc
) ;

    // BitmapC: a dense format, but able to represent sparsity structure of A.
    //
    //          int8_t Ab [nrows*ncols] ;
    //          <type> Ax [nrows*ncols] ;
    //
    //      Ab and Ax are both of size nrows*ncols.  Ab [i+j*nrows] = 1 if the
    //      A(i,j) entry is present with value Ax [i+j*nrows], or 0 if A(i,j)
    //      is not present.  nvals must equal the number of 1's in the Ab
    //      array.

//------------------------------------------------------------------------------
// GxB_Matrix_pack_FullR:  pack a full matrix, held by row
//------------------------------------------------------------------------------

GrB_Info GxB_Matrix_import_FullR // historical: use GxB_Matrix_pack_FullR
(
    GrB_Matrix *A,      // handle of matrix to create
    GrB_Type type,      // type of matrix to create
    GrB_Index nrows,    // number of rows of the matrix
    GrB_Index ncols,    // number of columns of the matrix
    void **Ax,          // values, Ax_size >= nrows*ncols * (type size)
                        // or Ax_size >= (type size), if iso is true
    GrB_Index Ax_size,  // size of Ax in bytes
    bool iso,           // if true, A is iso
    const GrB_Descriptor desc
) ;

GrB_Info GxB_Matrix_pack_FullR  // pack a full matrix, held by row
(
    GrB_Matrix A,       // matrix to create (type, nrows, ncols unchanged)
    void **Ax,          // values, Ax_size >= nrows*ncols * (type size)
                        // or Ax_size >= (type size), if iso is true
    GrB_Index Ax_size,  // size of Ax in bytes
    bool iso,           // if true, A is iso
    const GrB_Descriptor desc
) ;

    // FullR: an nrows-by-ncols full matrix held in row-major order:
    //
    //  <type> Ax [nrows*ncols] ;
    //
    //      Ax is an array of size nrows*ncols, where A(i,j) is held in
    //      Ax [i*ncols+j].  All entries in A are present.

//------------------------------------------------------------------------------
// GxB_Matrix_pack_FullC: pack a full matrix, held by column
//------------------------------------------------------------------------------

GrB_Info GxB_Matrix_import_FullC // historical: use GxB_Matrix_pack_FullC
(
    GrB_Matrix *A,      // handle of matrix to create
    GrB_Type type,      // type of matrix to create
    GrB_Index nrows,    // number of rows of the matrix
    GrB_Index ncols,    // number of columns of the matrix
    void **Ax,          // values, Ax_size >= nrows*ncols * (type size)
                        // or Ax_size >= (type size), if iso is true
    GrB_Index Ax_size,  // size of Ax in bytes
    bool iso,           // if true, A is iso
    const GrB_Descriptor desc
) ;

GrB_Info GxB_Matrix_pack_FullC  // pack a full matrix, held by column
(
    GrB_Matrix A,       // matrix to create (type, nrows, ncols unchanged)
    void **Ax,          // values, Ax_size >= nrows*ncols * (type size)
                        // or Ax_size >= (type size), if iso is true
    GrB_Index Ax_size,  // size of Ax in bytes
    bool iso,           // if true, A is iso
    const GrB_Descriptor desc
) ;

    // FullC: an nrows-by-ncols full matrix held in column-major order:
    //
    //  <type> Ax [nrows*ncols] ;
    //
    //      Ax is an array of size nrows*ncols, where A(i,j) is held in
    //      Ax [i+j*nrows].  All entries in A are present.

//------------------------------------------------------------------------------
// GxB_Vector_pack_CSC: import/pack a vector in CSC format
//------------------------------------------------------------------------------

GrB_Info GxB_Vector_import_CSC // historical: use GxB_Vector_pack_CSC
(
    GrB_Vector *v,      // handle of vector to create
    GrB_Type type,      // type of vector to create
    GrB_Index n,        // vector length
    GrB_Index **vi,     // indices, vi_size >= nvals(v) * sizeof(int64_t)
    void **vx,          // values, vx_size >= nvals(v) * (type size)
                        // or vx_size >= (type size), if iso is true
    GrB_Index vi_size,  // size of vi in bytes
    GrB_Index vx_size,  // size of vx in bytes
    bool iso,           // if true, v is iso
    GrB_Index nvals,    // # of entries in vector
    bool jumbled,       // if true, indices may be unsorted
    const GrB_Descriptor desc
) ;

GrB_Info GxB_Vector_pack_CSC  // pack a vector in CSC format
(
    GrB_Vector v,       // vector to create (type and length unchanged)
    GrB_Index **vi,     // indices, vi_size >= nvals(v) * sizeof(int64_t)
    void **vx,          // values, vx_size >= nvals(v) * (type size)
                        // or vx_size >= (type size), if iso is true
    GrB_Index vi_size,  // size of vi in bytes
    GrB_Index vx_size,  // size of vx in bytes
    bool iso,           // if true, v is iso
    GrB_Index nvals,    // # of entries in vector
    bool jumbled,       // if true, indices may be unsorted
    const GrB_Descriptor desc
) ;

    // The GrB_Vector is treated as if it was a single column of an n-by-1
    // matrix in CSC format, except that no vp array is required.  If nvals is
    // zero, then the vi and vx arrays need not be present and can be NULL.

//------------------------------------------------------------------------------
// GxB_Vector_pack_Bitmap: pack a vector in bitmap format
//------------------------------------------------------------------------------

GrB_Info GxB_Vector_import_Bitmap // historical: GxB_Vector_pack_Bitmap
(
    GrB_Vector *v,      // handle of vector to create
    GrB_Type type,      // type of vector to create
    GrB_Index n,        // vector length
    int8_t **vb,        // bitmap, vb_size >= n
    void **vx,          // values, vx_size >= n * (type size)
                        // or vx_size >= (type size), if iso is true
    GrB_Index vb_size,  // size of vb in bytes
    GrB_Index vx_size,  // size of vx in bytes
    bool iso,           // if true, v is iso
    GrB_Index nvals,    // # of entries in bitmap
    const GrB_Descriptor desc
) ;

GrB_Info GxB_Vector_pack_Bitmap // pack a bitmap vector
(
    GrB_Vector v,       // vector to create (type and length unchanged)
    int8_t **vb,        // bitmap, vb_size >= n
    void **vx,          // values, vx_size >= n * (type size)
                        // or vx_size >= (type size), if iso is true
    GrB_Index vb_size,  // size of vb in bytes
    GrB_Index vx_size,  // size of vx in bytes
    bool iso,           // if true, v is iso
    GrB_Index nvals,    // # of entries in bitmap
    const GrB_Descriptor desc
) ;

    // The GrB_Vector is treated as if it was a single column of an n-by-1
    // matrix in BitmapC format.

//------------------------------------------------------------------------------
// GxB_Vector_pack_Full: pack a vector in full format
//------------------------------------------------------------------------------

GrB_Info GxB_Vector_import_Full // historical: use GxB_Vector_pack_Full
(
    GrB_Vector *v,      // handle of vector to create
    GrB_Type type,      // type of vector to create
    GrB_Index n,        // vector length
    void **vx,          // values, vx_size >= nvals(v) * (type size)
                        // or vx_size >= (type size), if iso is true
    GrB_Index vx_size,  // size of vx in bytes
    bool iso,           // if true, v is iso
    const GrB_Descriptor desc
) ;

GrB_Info GxB_Vector_pack_Full // pack a full vector
(
    GrB_Vector v,       // vector to create (type and length unchanged)
    void **vx,          // values, vx_size >= nvals(v) * (type size)
                        // or vx_size >= (type size), if iso is true
    GrB_Index vx_size,  // size of vx in bytes
    bool iso,           // if true, v is iso
    const GrB_Descriptor desc
) ;

    // The GrB_Vector is treated as if it was a single column of an n-by-1
    // matrix in FullC format.

//------------------------------------------------------------------------------
// GxB* export/unpack
//------------------------------------------------------------------------------

// The GxB_*_export/unpack functions are symmetric with the GxB_*_import/pack
// functions.  The export/unpack functions force completion of any pending
// operations, prior to the export, except if the only pending operation is to
// unjumble the matrix.
//
// If there are no entries in the matrix or vector, then the index arrays (Ai,
// Aj, or vi) and value arrays (Ax or vx) are returned as NULL.  This is not an
// error condition.
//
// A GrB_Matrix may be exported/unpacked in any one of four different formats.
// On successful export, the input GrB_Matrix A is freed, and the output arrays
// Ah, Ap, Ai, Aj, and/or Ax are returned to the user application as arrays
// allocated by the ANSI C malloc function.  The four formats are the same as
// the import formats for GxB_Matrix_import/pack.
//
// If jumbled is NULL on input, this indicates to GxB_*export/unpack* that the
// exported/unpacked matrix cannot be returned in a jumbled format.  In this
// case, if the matrix is jumbled, it is sorted before exporting it to the
// caller.
//
// If iso is NULL on input, this indicates to the export/unpack methods that
// the exported/unpacked matrix cannot be returned in a iso format, with an Ax
// array with just one entry.  In this case, if the matrix is iso, it is
// expanded before exporting/unpacking it to the caller.
//
// For the export/unpack*Full* methods, all entries in the matrix or must be
// present.  That is, GrB_*_nvals must report nvals equal to nrows*ncols or a
// matrix.  If this condition does not hold, the matrix/vector is not exported,
// and GrB_INVALID_VALUE is returned.
//
// If the export/unpack is not successful, the export/unpack functions do not
// modify matrix or vector and the user arrays are returned as NULL.

GrB_Info GxB_Matrix_export_CSR  // historical: use GxB_Matrix_unpack_CSR
(
    GrB_Matrix *A,      // handle of matrix to export and free
    GrB_Type *type,     // type of matrix exported
    GrB_Index *nrows,   // number of rows of the matrix
    GrB_Index *ncols,   // number of columns of the matrix
    GrB_Index **Ap,     // row "pointers"
    GrB_Index **Aj,     // column indices
    void **Ax,          // values
    GrB_Index *Ap_size, // size of Ap in bytes
    GrB_Index *Aj_size, // size of Aj in bytes
    GrB_Index *Ax_size, // size of Ax in bytes
    bool *iso,          // if true, A is iso
    bool *jumbled,      // if true, indices in each row may be unsorted
    const GrB_Descriptor desc
) ;

GrB_Info GxB_Matrix_unpack_CSR  // unpack a CSR matrix
(
    GrB_Matrix A,       // matrix to unpack (type, nrows, ncols unchanged)
    GrB_Index **Ap,     // row "pointers"
    GrB_Index **Aj,     // column indices
    void **Ax,          // values
    GrB_Index *Ap_size, // size of Ap in bytes
    GrB_Index *Aj_size, // size of Aj in bytes
    GrB_Index *Ax_size, // size of Ax in bytes
    bool *iso,          // if true, A is iso
    bool *jumbled,      // if true, indices in each row may be unsorted
    const GrB_Descriptor desc
) ;

GrB_Info GxB_Matrix_export_CSC  // historical: use GxB_Matrix_unpack_CSC
(
    GrB_Matrix *A,      // handle of matrix to export and free
    GrB_Type *type,     // type of matrix exported
    GrB_Index *nrows,   // number of rows of the matrix
    GrB_Index *ncols,   // number of columns of the matrix
    GrB_Index **Ap,     // column "pointers"
    GrB_Index **Ai,     // row indices
    void **Ax,          // values
    GrB_Index *Ap_size, // size of Ap in bytes
    GrB_Index *Ai_size, // size of Ai in bytes
    GrB_Index *Ax_size, // size of Ax in bytes
    bool *iso,          // if true, A is iso
    bool *jumbled,      // if true, indices in each column may be unsorted
    const GrB_Descriptor desc
) ;

GrB_Info GxB_Matrix_unpack_CSC  // unpack a CSC matrix
(
    GrB_Matrix A,       // matrix to unpack (type, nrows, ncols unchanged)
    GrB_Index **Ap,     // column "pointers"
    GrB_Index **Ai,     // row indices
    void **Ax,          // values
    GrB_Index *Ap_size, // size of Ap in bytes
    GrB_Index *Ai_size, // size of Ai in bytes
    GrB_Index *Ax_size, // size of Ax in bytes
    bool *iso,          // if true, A is iso
    bool *jumbled,      // if true, indices in each column may be unsorted
    const GrB_Descriptor desc
) ;

GrB_Info GxB_Matrix_export_HyperCSR //historical: use GxB_Matrix_unpack_HyperCSR
(
    GrB_Matrix *A,      // handle of matrix to export and free
    GrB_Type *type,     // type of matrix exported
    GrB_Index *nrows,   // number of rows of the matrix
    GrB_Index *ncols,   // number of columns of the matrix
    GrB_Index **Ap,     // row "pointers"
    GrB_Index **Ah,     // row indices
    GrB_Index **Aj,     // column indices
    void **Ax,          // values
    GrB_Index *Ap_size, // size of Ap in bytes
    GrB_Index *Ah_size, // size of Ah in bytes
    GrB_Index *Aj_size, // size of Aj in bytes
    GrB_Index *Ax_size, // size of Ax in bytes
    bool *iso,          // if true, A is iso
    GrB_Index *nvec,    // number of rows that appear in Ah
    bool *jumbled,      // if true, indices in each row may be unsorted
    const GrB_Descriptor desc
) ;

GrB_Info GxB_Matrix_unpack_HyperCSR  // unpack a hypersparse CSR matrix
(
    GrB_Matrix A,       // matrix to unpack (type, nrows, ncols unchanged)
    GrB_Index **Ap,     // row "pointers"
    GrB_Index **Ah,     // row indices
    GrB_Index **Aj,     // column indices
    void **Ax,          // values
    GrB_Index *Ap_size, // size of Ap in bytes
    GrB_Index *Ah_size, // size of Ah in bytes
    GrB_Index *Aj_size, // size of Aj in bytes
    GrB_Index *Ax_size, // size of Ax in bytes
    bool *iso,          // if true, A is iso
    GrB_Index *nvec,    // number of rows that appear in Ah
    bool *jumbled,      // if true, indices in each row may be unsorted
    const GrB_Descriptor desc
) ;

GrB_Info GxB_Matrix_export_HyperCSC //historical: use GxB_Matrix_unpack_HyperCSC
(
    GrB_Matrix *A,      // handle of matrix to export and free
    GrB_Type *type,     // type of matrix exported
    GrB_Index *nrows,   // number of rows of the matrix
    GrB_Index *ncols,   // number of columns of the matrix
    GrB_Index **Ap,     // column "pointers"
    GrB_Index **Ah,     // column indices
    GrB_Index **Ai,     // row indices
    void **Ax,          // values
    GrB_Index *Ap_size, // size of Ap in bytes
    GrB_Index *Ah_size, // size of Ah in bytes
    GrB_Index *Ai_size, // size of Ai in bytes
    GrB_Index *Ax_size, // size of Ax in bytes
    bool *iso,          // if true, A is iso
    GrB_Index *nvec,    // number of columns that appear in Ah
    bool *jumbled,      // if true, indices in each column may be unsorted
    const GrB_Descriptor desc
) ;

GrB_Info GxB_Matrix_unpack_HyperCSC  // unpack a hypersparse CSC matrix
(
    GrB_Matrix A,       // matrix to unpack (type, nrows, ncols unchanged)
    GrB_Index **Ap,     // column "pointers"
    GrB_Index **Ah,     // column indices
    GrB_Index **Ai,     // row indices
    void **Ax,          // values
    GrB_Index *Ap_size, // size of Ap in bytes
    GrB_Index *Ah_size, // size of Ah in bytes
    GrB_Index *Ai_size, // size of Ai in bytes
    GrB_Index *Ax_size, // size of Ax in bytes
    bool *iso,          // if true, A is iso
    GrB_Index *nvec,    // number of columns that appear in Ah
    bool *jumbled,      // if true, indices in each column may be unsorted
    const GrB_Descriptor desc
) ;

GrB_Info GxB_Matrix_export_BitmapR // historical: use GxB_Matrix_unpack_BitmapR
(
    GrB_Matrix *A,      // handle of matrix to export and free
    GrB_Type *type,     // type of matrix exported
    GrB_Index *nrows,   // number of rows of the matrix
    GrB_Index *ncols,   // number of columns of the matrix
    int8_t **Ab,        // bitmap
    void **Ax,          // values
    GrB_Index *Ab_size, // size of Ab in bytes
    GrB_Index *Ax_size, // size of Ax in bytes
    bool *iso,          // if true, A is iso
    GrB_Index *nvals,   // # of entries in bitmap
    const GrB_Descriptor desc
) ;

GrB_Info GxB_Matrix_unpack_BitmapR  // unpack a bitmap matrix, by row
(
    GrB_Matrix A,       // matrix to unpack (type, nrows, ncols unchanged)
    int8_t **Ab,        // bitmap
    void **Ax,          // values
    GrB_Index *Ab_size, // size of Ab in bytes
    GrB_Index *Ax_size, // size of Ax in bytes
    bool *iso,          // if true, A is iso
    GrB_Index *nvals,   // # of entries in bitmap
    const GrB_Descriptor desc
) ;

GrB_Info GxB_Matrix_export_BitmapC // historical: use GxB_Matrix_unpack_BitmapC
(
    GrB_Matrix *A,      // handle of matrix to export and free
    GrB_Type *type,     // type of matrix exported
    GrB_Index *nrows,   // number of rows of the matrix
    GrB_Index *ncols,   // number of columns of the matrix
    int8_t **Ab,        // bitmap
    void **Ax,          // values
    GrB_Index *Ab_size, // size of Ab in bytes
    GrB_Index *Ax_size, // size of Ax in bytes
    bool *iso,          // if true, A is iso
    GrB_Index *nvals,   // # of entries in bitmap
    const GrB_Descriptor desc
) ;

GrB_Info GxB_Matrix_unpack_BitmapC  // unpack a bitmap matrix, by col
(
    GrB_Matrix A,       // matrix to unpack (type, nrows, ncols unchanged)
    int8_t **Ab,        // bitmap
    void **Ax,          // values
    GrB_Index *Ab_size, // size of Ab in bytes
    GrB_Index *Ax_size, // size of Ax in bytes
    bool *iso,          // if true, A is iso
    GrB_Index *nvals,   // # of entries in bitmap
    const GrB_Descriptor desc
) ;

GrB_Info GxB_Matrix_export_FullR // historical: use GxB_Matrix_unpack_FullR
(
    GrB_Matrix *A,      // handle of matrix to export and free
    GrB_Type *type,     // type of matrix exported
    GrB_Index *nrows,   // number of rows of the matrix
    GrB_Index *ncols,   // number of columns of the matrix
    void **Ax,          // values
    GrB_Index *Ax_size, // size of Ax in bytes
    bool *iso,          // if true, A is iso
    const GrB_Descriptor desc
) ;

GrB_Info GxB_Matrix_unpack_FullR  // unpack a full matrix, by row
(
    GrB_Matrix A,       // matrix to unpack (type, nrows, ncols unchanged)
    void **Ax,          // values
    GrB_Index *Ax_size, // size of Ax in bytes
    bool *iso,          // if true, A is iso
    const GrB_Descriptor desc
) ;

GrB_Info GxB_Matrix_export_FullC // historical: use GxB_Matrix_unpack_FullC
(
    GrB_Matrix *A,      // handle of matrix to export and free
    GrB_Type *type,     // type of matrix exported
    GrB_Index *nrows,   // number of rows of the matrix
    GrB_Index *ncols,   // number of columns of the matrix
    void **Ax,          // values
    GrB_Index *Ax_size, // size of Ax in bytes
    bool *iso,          // if true, A is iso
    const GrB_Descriptor desc
) ;

GrB_Info GxB_Matrix_unpack_FullC  // unpack a full matrix, by column
(
    GrB_Matrix A,       // matrix to unpack (type, nrows, ncols unchanged)
    void **Ax,          // values
    GrB_Index *Ax_size, // size of Ax in bytes
    bool *iso,          // if true, A is iso
    const GrB_Descriptor desc
) ;

GrB_Info GxB_Vector_export_CSC // historical: use GxB_Vector_unpack_CSC
(
    GrB_Vector *v,      // handle of vector to export and free
    GrB_Type *type,     // type of vector exported
    GrB_Index *n,       // length of the vector
    GrB_Index **vi,     // indices
    void **vx,          // values
    GrB_Index *vi_size, // size of vi in bytes
    GrB_Index *vx_size, // size of vx in bytes
    bool *iso,          // if true, v is iso
    GrB_Index *nvals,   // # of entries in vector
    bool *jumbled,      // if true, indices may be unsorted
    const GrB_Descriptor desc
) ;

GrB_Info GxB_Vector_unpack_CSC  // unpack a CSC vector
(
    GrB_Vector v,       // vector to unpack (type and length unchanged)
    GrB_Index **vi,     // indices
    void **vx,          // values
    GrB_Index *vi_size, // size of vi in bytes
    GrB_Index *vx_size, // size of vx in bytes
    bool *iso,          // if true, v is iso
    GrB_Index *nvals,   // # of entries in vector
    bool *jumbled,      // if true, indices may be unsorted
    const GrB_Descriptor desc
) ;

GrB_Info GxB_Vector_export_Bitmap // historical: use GxB_Vector_unpack_Bitmap
(
    GrB_Vector *v,      // handle of vector to export and free
    GrB_Type *type,     // type of vector exported
    GrB_Index *n,       // length of the vector
    int8_t **vb,        // bitmap
    void **vx,          // values
    GrB_Index *vb_size, // size of vb in bytes
    GrB_Index *vx_size, // size of vx in bytes
    bool *iso,          // if true, v is iso
    GrB_Index *nvals,    // # of entries in bitmap
    const GrB_Descriptor desc
) ;

GrB_Info GxB_Vector_unpack_Bitmap   // unpack a bitmap vector
(
    GrB_Vector v,       // vector to unpack (type and length unchanged)
    int8_t **vb,        // bitmap
    void **vx,          // values
    GrB_Index *vb_size, // size of vb in bytes
    GrB_Index *vx_size, // size of vx in bytes
    bool *iso,          // if true, v is iso
    GrB_Index *nvals,    // # of entries in bitmap
    const GrB_Descriptor desc
) ;

GrB_Info GxB_Vector_export_Full // historical: use GxB_Vector_unpack_Full
(
    GrB_Vector *v,      // handle of vector to export and free
    GrB_Type *type,     // type of vector exported
    GrB_Index *n,       // length of the vector
    void **vx,          // values
    GrB_Index *vx_size, // size of vx in bytes
    bool *iso,          // if true, v is iso
    const GrB_Descriptor desc
) ;

GrB_Info GxB_Vector_unpack_Full   // unpack a full vector
(
    GrB_Vector v,       // vector to unpack (type and length unchanged)
    void **vx,          // values
    GrB_Index *vx_size, // size of vx in bytes
    bool *iso,          // if true, v is iso
    const GrB_Descriptor desc
) ;

//------------------------------------------------------------------------------
// GxB hyper_hash pack/unpack
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS v7.3.0 adds a new internal component to the
// hypersparse matrix format: the hyper_hash GrB_Matrix A->Y.  The matrix
// provides a fast lookup into the hyperlist Ah.

// GxB_unpack_HyperHash unpacks the hyper_hash from the hypersparse matrix A.
// Normally, this method is called immediately before calling one of the four
// methods GxB_Matrix_(export/unpack)_Hyper(CSR/CSC).  For example, to unpack
// then pack a hypersparse CSC matrix:

//      GrB_Matrix Y = NULL ;
//
//      // to unpack all of A:
//      GxB_unpack_HyperHash (A, &Y, desc) ;    // first unpack A->Y into Y
//      GxB_Matrix_unpack_HyperCSC (A,          // then unpack the rest of A
//          &Ap, &Ah, &Ai, &Ax, &Ap_size, &Ah_size, &Ai_size, &Ax_size,
//          &iso, &nvec, &jumbled, descriptor) ;
//
//      // use the unpacked contents of A here, but do not change Ah or nvec.
//      ...
//
//      // to pack the data back into A:
//      GxB_Matrix_pack_HyperCSC (A, ...) ;     // pack most of A, except A->Y
//      GxB_pack_HyperHash (A, &Y, desc) ;      // then pack A->Y

// The same process is used with GxB_Matrix_unpack_HyperCSR,
// an the GxB_Matrix_export_Hyper* and GxB_Matrix_import_Hyper* methods.

// If A is not hypersparse on input to GxB_unpack_HyperHash, or if A is
// hypersparse but does yet not have a hyper_hash, then Y is returned as NULL.
// This is not an error condition, and GrB_SUCCESS is returned.  The hyper_hash
// of a hypersparse matrix A is a matrix that provides quick access to the
// inverse of Ah.  It is not always needed and may not be present.  It is left
// as pending work to be computed when needed.  GrB_Matrix_wait (A) will ensure
// that the hyper_hash is constructed, if A is hypersparse.

// If Y is moved from A and returned as non-NULL to the caller, then it is
// the responsibility of the user application to free it, or to re-pack it back
// into A via GxB_pack_HyperHash, as shown in the example above.

// If this method is called to remove the hyper_hash Y from the hypersparse
// matrix A, and then GrB_Matrix_wait (A) is called, a new hyper_hash matrix is
// constructed for A.

GrB_Info GxB_unpack_HyperHash       // move A->Y into Y
(
    GrB_Matrix A,                   // matrix to modify
    GrB_Matrix *Y,                  // hyper_hash matrix to move from A
    const GrB_Descriptor desc       // unused
) ;

// GxB_pack_HyperHash assigns the input Y matrix as the A->Y hyper_hash of the
// hypersparse matrix A.  Normally, this method is called immediately after
// calling one of the four methods GxB_Matrix_(import/pack)_Hyper(CSR/CSC).

// If A is not hypersparse on input to GxB_pack_HyperHash, or if A already has
// a hyper_hash matrix, or if Y is NULL on input, then nothing happens and Y is
// unchanged.  This is not an error condition and this method returns
// GrB_SUCCESS.  In this case, if Y is non-NULL after calling this method, it
// owned by the user application and freeing it is the responsibility of the
// user application.

// Basic checks are perfomed on Y: Y must have the right dimensions:  if A is
// HyperCSR and m-by-n with nvec vectors present in Ah, then Y must be n-by-v
// where v is a power of 2; if A is HyperCSR and m-by-n, then Y must be m-by-v.
// nvals(Y) must equal nvec.  Y must be sparse, held by column, and have type
// int64.  It cannot have any pending work.  It cannot have a hyper_hash
// of its own.  If any of these conditions hold, GrB_INVALID is returned and
// A and Y are unchanged.

// If Y is moved into A as its hyper_hash, then the caller's Y is set to NULL
// to indicate that it has been moved into A.  It is no longer owned by the
// caller, but is instead an opaque component of the A matrix.  It will be
// freed by SuiteSparse:GraphBLAS if A is modified or freed.

// Results are undefined if the input Y was not created by GxB_unpack_HyperHash
// (see the example above) or if the Ah contents or nvec of the matrix A are
// modified after they were exported/unpacked by
// GxB_Matrix_(export/unpack)_Hyper(CSR/CSC).

GrB_Info GxB_pack_HyperHash         // move Y into A->Y
(
    GrB_Matrix A,                   // matrix to modify
    GrB_Matrix *Y,                  // hyper_hash matrix to pack into A
    const GrB_Descriptor desc       // unused
) ;

//==============================================================================
// GrB import/export
//==============================================================================

// The GrB_Matrix_import method copies from user-provided arrays into an
// opaque GrB_Matrix and GrB_Matrix_export copies data out, from an opaque
// GrB_Matrix into user-provided arrays.  Unlike the GxB pack/unpack methods,
// memory is not handed off between the user application and GraphBLAS.

// These methods are much slower than the GxB pack/unpack methods, since they
// require a copy of the data to be made.  GrB_Matrix_import also must assume
// its input data cannot be trusted, and so it does extensive checks.  The GxB
// pack takes O(1) time in all cases (unless it is told the input data is
// untrusted, via the descriptor).  GxB unpack takes O(1) time unless the
// matrix is exported in a different format than it currently has.

// No typecasting of the values is done on import or export.

GrB_Info GrB_Matrix_import_BOOL     // import a GrB_BOOL matrix
(
    GrB_Matrix *A,          // handle of matrix to create
    GrB_Type type,          // type of matrix to create (must be GrB_BOOL)
    GrB_Index nrows,        // number of rows of the matrix
    GrB_Index ncols,        // number of columns of the matrix
    const GrB_Index *Ap,    // pointers for CSR, CSC, column indices for COO
    const GrB_Index *Ai,    // row indices for CSR, CSC
    const bool *Ax,         // values
    GrB_Index Ap_len,       // number of entries in Ap (not # of bytes)
    GrB_Index Ai_len,       // number of entries in Ai (not # of bytes)
    GrB_Index Ax_len,       // number of entries in Ax (not # of bytes)
    GrB_Format format       // import format
) ;

GrB_Info GrB_Matrix_import_INT8     // import a GrB_INT8 matrix
(
    GrB_Matrix *A,          // handle of matrix to create
    GrB_Type type,          // type of matrix to create (must be GrB_iNT8)
    GrB_Index nrows,        // number of rows of the matrix
    GrB_Index ncols,        // number of columns of the matrix
    const GrB_Index *Ap,    // pointers for CSR, CSC, column indices for COO
    const GrB_Index *Ai,    // row indices for CSR, CSC
    const int8_t *Ax,       // values
    GrB_Index Ap_len,       // number of entries in Ap (not # of bytes)
    GrB_Index Ai_len,       // number of entries in Ai (not # of bytes)
    GrB_Index Ax_len,       // number of entries in Ax (not # of bytes)
    GrB_Format format       // import format
) ;

GrB_Info GrB_Matrix_import_INT16    // import a GrB_INT16 matrix
(
    GrB_Matrix *A,          // handle of matrix to create
    GrB_Type type,          // type of matrix to create (must be GrB_INT16)
    GrB_Index nrows,        // number of rows of the matrix
    GrB_Index ncols,        // number of columns of the matrix
    const GrB_Index *Ap,    // pointers for CSR, CSC, column indices for COO
    const GrB_Index *Ai,    // row indices for CSR, CSC
    const int16_t *Ax,      // values
    GrB_Index Ap_len,       // number of entries in Ap (not # of bytes)
    GrB_Index Ai_len,       // number of entries in Ai (not # of bytes)
    GrB_Index Ax_len,       // number of entries in Ax (not # of bytes)
    GrB_Format format       // import format
) ;

GrB_Info GrB_Matrix_import_INT32    // import a GrB_INT32 matrix
(
    GrB_Matrix *A,          // handle of matrix to create
    GrB_Type type,          // type of matrix to create (must be GrB_INT32)
    GrB_Index nrows,        // number of rows of the matrix
    GrB_Index ncols,        // number of columns of the matrix
    const GrB_Index *Ap,    // pointers for CSR, CSC, column indices for COO
    const GrB_Index *Ai,    // row indices for CSR, CSC
    const int32_t *Ax,      // values
    GrB_Index Ap_len,       // number of entries in Ap (not # of bytes)
    GrB_Index Ai_len,       // number of entries in Ai (not # of bytes)
    GrB_Index Ax_len,       // number of entries in Ax (not # of bytes)
    GrB_Format format       // import format
) ;

GrB_Info GrB_Matrix_import_INT64    // import a GrB_INT64 matrix
(
    GrB_Matrix *A,          // handle of matrix to create
    GrB_Type type,          // type of matrix to create (must be GrB_INT64)
    GrB_Index nrows,        // number of rows of the matrix
    GrB_Index ncols,        // number of columns of the matrix
    const GrB_Index *Ap,    // pointers for CSR, CSC, column indices for COO
    const GrB_Index *Ai,    // row indices for CSR, CSC
    const int64_t *Ax,      // values
    GrB_Index Ap_len,       // number of entries in Ap (not # of bytes)
    GrB_Index Ai_len,       // number of entries in Ai (not # of bytes)
    GrB_Index Ax_len,       // number of entries in Ax (not # of bytes)
    GrB_Format format       // import format
) ;

GrB_Info GrB_Matrix_import_UINT8    // import a GrB_UINT8 matrix
(
    GrB_Matrix *A,          // handle of matrix to create
    GrB_Type type,          // type of matrix to create (must be GrB_UINT8)
    GrB_Index nrows,        // number of rows of the matrix
    GrB_Index ncols,        // number of columns of the matrix
    const GrB_Index *Ap,    // pointers for CSR, CSC, column indices for COO
    const GrB_Index *Ai,    // row indices for CSR, CSC
    const uint8_t *Ax,      // values
    GrB_Index Ap_len,       // number of entries in Ap (not # of bytes)
    GrB_Index Ai_len,       // number of entries in Ai (not # of bytes)
    GrB_Index Ax_len,       // number of entries in Ax (not # of bytes)
    GrB_Format format       // import format
) ;

GrB_Info GrB_Matrix_import_UINT16   // import a GrB_UINT16 matrix
(
    GrB_Matrix *A,          // handle of matrix to create
    GrB_Type type,          // type of matrix to create (must be GrB_UINT16)
    GrB_Index nrows,        // number of rows of the matrix
    GrB_Index ncols,        // number of columns of the matrix
    const GrB_Index *Ap,    // pointers for CSR, CSC, column indices for COO
    const GrB_Index *Ai,    // row indices for CSR, CSC
    const uint16_t *Ax,     // values
    GrB_Index Ap_len,       // number of entries in Ap (not # of bytes)
    GrB_Index Ai_len,       // number of entries in Ai (not # of bytes)
    GrB_Index Ax_len,       // number of entries in Ax (not # of bytes)
    GrB_Format format       // import format
) ;

GrB_Info GrB_Matrix_import_UINT32   // import a GrB_UINT32 matrix
(
    GrB_Matrix *A,          // handle of matrix to create
    GrB_Type type,          // type of matrix to create (must be GrB_UINT32)
    GrB_Index nrows,        // number of rows of the matrix
    GrB_Index ncols,        // number of columns of the matrix
    const GrB_Index *Ap,    // pointers for CSR, CSC, column indices for COO
    const GrB_Index *Ai,    // row indices for CSR, CSC
    const uint32_t *Ax,     // values
    GrB_Index Ap_len,       // number of entries in Ap (not # of bytes)
    GrB_Index Ai_len,       // number of entries in Ai (not # of bytes)
    GrB_Index Ax_len,       // number of entries in Ax (not # of bytes)
    GrB_Format format       // import format
) ;

GrB_Info GrB_Matrix_import_UINT64   // import a GrB_UINT64 matrix
(
    GrB_Matrix *A,          // handle of matrix to create
    GrB_Type type,          // type of matrix to create (must be GrB_UINT64)
    GrB_Index nrows,        // number of rows of the matrix
    GrB_Index ncols,        // number of columns of the matrix
    const GrB_Index *Ap,    // pointers for CSR, CSC, column indices for COO
    const GrB_Index *Ai,    // row indices for CSR, CSC
    const uint64_t *Ax,     // values
    GrB_Index Ap_len,       // number of entries in Ap (not # of bytes)
    GrB_Index Ai_len,       // number of entries in Ai (not # of bytes)
    GrB_Index Ax_len,       // number of entries in Ax (not # of bytes)
    GrB_Format format       // import format
) ;

GrB_Info GrB_Matrix_import_FP32   // import a GrB_FP32 matrix
(
    GrB_Matrix *A,          // handle of matrix to create
    GrB_Type type,          // type of matrix to create (must be GrB_FP32)
    GrB_Index nrows,        // number of rows of the matrix
    GrB_Index ncols,        // number of columns of the matrix
    const GrB_Index *Ap,    // pointers for CSR, CSC, column indices for COO
    const GrB_Index *Ai,    // row indices for CSR, CSC
    const float *Ax,        // values
    GrB_Index Ap_len,       // number of entries in Ap (not # of bytes)
    GrB_Index Ai_len,       // number of entries in Ai (not # of bytes)
    GrB_Index Ax_len,       // number of entries in Ax (not # of bytes)
    GrB_Format format       // import format
) ;

GrB_Info GrB_Matrix_import_FP64   // import a GrB_FP64 matrix
(
    GrB_Matrix *A,          // handle of matrix to create
    GrB_Type type,          // type of matrix to create (must be GrB_FP64)
    GrB_Index nrows,        // number of rows of the matrix
    GrB_Index ncols,        // number of columns of the matrix
    const GrB_Index *Ap,    // pointers for CSR, CSC, column indices for COO
    const GrB_Index *Ai,    // row indices for CSR, CSC
    const double *Ax,       // values
    GrB_Index Ap_len,       // number of entries in Ap (not # of bytes)
    GrB_Index Ai_len,       // number of entries in Ai (not # of bytes)
    GrB_Index Ax_len,       // number of entries in Ax (not # of bytes)
    GrB_Format format       // import format
) ;

GrB_Info GxB_Matrix_import_FC32   // import a GxB_FC32 matrix
(
    GrB_Matrix *A,          // handle of matrix to create
    GrB_Type type,          // type of matrix to create (must be GxB_FC32)
    GrB_Index nrows,        // number of rows of the matrix
    GrB_Index ncols,        // number of columns of the matrix
    const GrB_Index *Ap,    // pointers for CSR, CSC, column indices for COO
    const GrB_Index *Ai,    // row indices for CSR, CSC
    const GxB_FC32_t *Ax,   // values
    GrB_Index Ap_len,       // number of entries in Ap (not # of bytes)
    GrB_Index Ai_len,       // number of entries in Ai (not # of bytes)
    GrB_Index Ax_len,       // number of entries in Ax (not # of bytes)
    GrB_Format format       // import format
) ;

GrB_Info GxB_Matrix_import_FC64   // import a GxB_FC64 matrix
(
    GrB_Matrix *A,          // handle of matrix to create
    GrB_Type type,          // type of matrix to create (must be GxB_FC64)
    GrB_Index nrows,        // number of rows of the matrix
    GrB_Index ncols,        // number of columns of the matrix
    const GrB_Index *Ap,    // pointers for CSR, CSC, column indices for COO
    const GrB_Index *Ai,    // row indices for CSR, CSC
    const GxB_FC64_t *Ax,   // values
    GrB_Index Ap_len,       // number of entries in Ap (not # of bytes)
    GrB_Index Ai_len,       // number of entries in Ai (not # of bytes)
    GrB_Index Ax_len,       // number of entries in Ax (not # of bytes)
    GrB_Format format       // import format
) ;

GrB_Info GrB_Matrix_import_UDT    // import a matrix with a user-defined type
(
    GrB_Matrix *A,          // handle of matrix to create
    GrB_Type type,          // type of matrix to create
    GrB_Index nrows,        // number of rows of the matrix
    GrB_Index ncols,        // number of columns of the matrix
    const GrB_Index *Ap,    // pointers for CSR, CSC, column indices for COO
    const GrB_Index *Ai,    // row indices for CSR, CSC
    const void *Ax,         // values (must match the type parameter)
    GrB_Index Ap_len,       // number of entries in Ap (not # of bytes)
    GrB_Index Ai_len,       // number of entries in Ai (not # of bytes)
    GrB_Index Ax_len,       // number of entries in Ax (not # of bytes)
    GrB_Format format       // import format
) ;

#if GxB_STDC_VERSION >= 201112L
#define GrB_Matrix_import(A,type,nrows,ncols,Ap,Ai,Ax,Ap_len,Ai_len,Ax_len,fmt)\
    _Generic                                                    \
    (                                                           \
        (Ax),                                                   \
            GB_PCASES (GrB, Matrix_import)                      \
    )                                                           \
    (A, type, nrows, ncols, Ap, Ai, Ax, Ap_len, Ai_len, Ax_len, fmt)
#endif

// For GrB_Matrix_export_T: on input, Ap_len, Ai_len, and Ax_len are
// the size of the 3 arrays Ap, Ai, and Ax, in terms of the # of entries.
// On output, these 3 values are modified to be the # of entries copied
// into those 3 arrays.

GrB_Info GrB_Matrix_export_BOOL     // export a GrB_BOOL matrix
(
    GrB_Index *Ap,          // pointers for CSR, CSC, column indices for COO
    GrB_Index *Ai,          // col indices for CSR/COO, row indices for CSC
    bool *Ax,               // values (must match the type of A)
    GrB_Index *Ap_len,      // number of entries in Ap (not # of bytes)
    GrB_Index *Ai_len,      // number of entries in Ai (not # of bytes)
    GrB_Index *Ax_len,      // number of entries in Ax (not # of bytes)
    GrB_Format format,      // export format
    GrB_Matrix A            // matrix to export (must be of type GrB_BOOL)
) ;

GrB_Info GrB_Matrix_export_INT8     // export a GrB_INT8 matrix
(
    GrB_Index *Ap,          // pointers for CSR, CSC, column indices for COO
    GrB_Index *Ai,          // col indices for CSR/COO, row indices for CSC
    int8_t *Ax,             // values (must match the type of A)
    GrB_Index *Ap_len,      // number of entries in Ap (not # of bytes)
    GrB_Index *Ai_len,      // number of entries in Ai (not # of bytes)
    GrB_Index *Ax_len,      // number of entries in Ax (not # of bytes)
    GrB_Format format,      // export format
    GrB_Matrix A            // matrix to export (must be of type GrB_INT8)
) ;

GrB_Info GrB_Matrix_export_INT16     // export a GrB_INT16 matrix
(
    GrB_Index *Ap,          // pointers for CSR, CSC, column indices for COO
    GrB_Index *Ai,          // col indices for CSR/COO, row indices for CSC
    int16_t *Ax,            // values (must match the type of A)
    GrB_Index *Ap_len,      // number of entries in Ap (not # of bytes)
    GrB_Index *Ai_len,      // number of entries in Ai (not # of bytes)
    GrB_Index *Ax_len,      // number of entries in Ax (not # of bytes)
    GrB_Format format,      // export format
    GrB_Matrix A            // matrix to export (must be of type GrB_INT16)
) ;

GrB_Info GrB_Matrix_export_INT32     // export a GrB_INT32 matrix
(
    GrB_Index *Ap,          // pointers for CSR, CSC, column indices for COO
    GrB_Index *Ai,          // col indices for CSR/COO, row indices for CSC
    int32_t *Ax,            // values (must match the type of A)
    GrB_Index *Ap_len,      // number of entries in Ap (not # of bytes)
    GrB_Index *Ai_len,      // number of entries in Ai (not # of bytes)
    GrB_Index *Ax_len,      // number of entries in Ax (not # of bytes)
    GrB_Format format,      // export format
    GrB_Matrix A            // matrix to export (must be of type GrB_INT32)
) ;

GrB_Info GrB_Matrix_export_INT64     // export a GrB_INT64 matrix
(
    GrB_Index *Ap,          // pointers for CSR, CSC, column indices for COO
    GrB_Index *Ai,          // col indices for CSR/COO, row indices for CSC
    int64_t *Ax,            // values (must match the type of A)
    GrB_Index *Ap_len,      // number of entries in Ap (not # of bytes)
    GrB_Index *Ai_len,      // number of entries in Ai (not # of bytes)
    GrB_Index *Ax_len,      // number of entries in Ax (not # of bytes)
    GrB_Format format,      // export format
    GrB_Matrix A            // matrix to export (must be of type GrB_INT64)
) ;

GrB_Info GrB_Matrix_export_UINT8     // export a GrB_UINT8 matrix
(
    GrB_Index *Ap,          // pointers for CSR, CSC, column indices for COO
    GrB_Index *Ai,          // col indices for CSR/COO, row indices for CSC
    uint8_t *Ax,            // values (must match the type of A)
    GrB_Index *Ap_len,      // number of entries in Ap (not # of bytes)
    GrB_Index *Ai_len,      // number of entries in Ai (not # of bytes)
    GrB_Index *Ax_len,      // number of entries in Ax (not # of bytes)
    GrB_Format format,      // export format
    GrB_Matrix A            // matrix to export (must be of type GrB_UINT8)
) ;

GrB_Info GrB_Matrix_export_UINT16     // export a GrB_UINT16 matrix
(
    GrB_Index *Ap,          // pointers for CSR, CSC, column indices for COO
    GrB_Index *Ai,          // col indices for CSR/COO, row indices for CSC
    uint16_t *Ax,           // values (must match the type of A)
    GrB_Index *Ap_len,      // number of entries in Ap (not # of bytes)
    GrB_Index *Ai_len,      // number of entries in Ai (not # of bytes)
    GrB_Index *Ax_len,      // number of entries in Ax (not # of bytes)
    GrB_Format format,      // export format
    GrB_Matrix A            // matrix to export (must be of type GrB_UINT16)
) ;

GrB_Info GrB_Matrix_export_UINT32     // export a GrB_UINT32 matrix
(
    GrB_Index *Ap,          // pointers for CSR, CSC, column indices for COO
    GrB_Index *Ai,          // col indices for CSR/COO, row indices for CSC
    uint32_t *Ax,           // values (must match the type of A)
    GrB_Index *Ap_len,      // number of entries in Ap (not # of bytes)
    GrB_Index *Ai_len,      // number of entries in Ai (not # of bytes)
    GrB_Index *Ax_len,      // number of entries in Ax (not # of bytes)
    GrB_Format format,      // export format
    GrB_Matrix A            // matrix to export (must be of type GrB_UINT32)
) ;

GrB_Info GrB_Matrix_export_UINT64     // export a GrB_UINT64 matrix
(
    GrB_Index *Ap,          // pointers for CSR, CSC, column indices for COO
    GrB_Index *Ai,          // col indices for CSR/COO, row indices for CSC
    uint64_t *Ax,           // values (must match the type of A)
    GrB_Index *Ap_len,      // number of entries in Ap (not # of bytes)
    GrB_Index *Ai_len,      // number of entries in Ai (not # of bytes)
    GrB_Index *Ax_len,      // number of entries in Ax (not # of bytes)
    GrB_Format format,      // export format
    GrB_Matrix A            // matrix to export (must be of type GrB_UINT64)
) ;

GrB_Info GrB_Matrix_export_FP32     // export a GrB_FP32 matrix
(
    GrB_Index *Ap,          // pointers for CSR, CSC, column indices for COO
    GrB_Index *Ai,          // col indices for CSR/COO, row indices for CSC
    float *Ax,              // values (must match the type of A)
    GrB_Index *Ap_len,      // number of entries in Ap (not # of bytes)
    GrB_Index *Ai_len,      // number of entries in Ai (not # of bytes)
    GrB_Index *Ax_len,      // number of entries in Ax (not # of bytes)
    GrB_Format format,      // export format
    GrB_Matrix A            // matrix to export (must be of type GrB_FP32)
) ;

GrB_Info GrB_Matrix_export_FP64     // export a GrB_FP64 matrix
(
    GrB_Index *Ap,          // pointers for CSR, CSC, column indices for COO
    GrB_Index *Ai,          // col indices for CSR/COO, row indices for CSC
    double *Ax,             // values (must match the type of A)
    GrB_Index *Ap_len,      // number of entries in Ap (not # of bytes)
    GrB_Index *Ai_len,      // number of entries in Ai (not # of bytes)
    GrB_Index *Ax_len,      // number of entries in Ax (not # of bytes)
    GrB_Format format,      // export format
    GrB_Matrix A            // matrix to export (must be of type GrB_FP64)
) ;

GrB_Info GxB_Matrix_export_FC32     // export a GrB_FC32 matrix
(
    GrB_Index *Ap,          // pointers for CSR, CSC, column indices for COO
    GrB_Index *Ai,          // col indices for CSR/COO, row indices for CSC
    GxB_FC32_t *Ax,         // values (must match the type of A)
    GrB_Index *Ap_len,      // number of entries in Ap (not # of bytes)
    GrB_Index *Ai_len,      // number of entries in Ai (not # of bytes)
    GrB_Index *Ax_len,      // number of entries in Ax (not # of bytes)
    GrB_Format format,      // export format
    GrB_Matrix A            // matrix to export (must be of type GrB_FC32)
) ;

GrB_Info GxB_Matrix_export_FC64     // export a GrB_FC64 matrix
(
    GrB_Index *Ap,          // pointers for CSR, CSC, column indices for COO
    GrB_Index *Ai,          // col indices for CSR/COO, row indices for CSC
    GxB_FC64_t *Ax,         // values (must match the type of A)
    GrB_Index *Ap_len,      // number of entries in Ap (not # of bytes)
    GrB_Index *Ai_len,      // number of entries in Ai (not # of bytes)
    GrB_Index *Ax_len,      // number of entries in Ax (not # of bytes)
    GrB_Format format,      // export format
    GrB_Matrix A            // matrix to export (must be of type GrB_FC64)
) ;

GrB_Info GrB_Matrix_export_UDT      // export a matrix with a user-defined type
(
    GrB_Index *Ap,          // pointers for CSR, CSC, column indices for COO
    GrB_Index *Ai,          // col indices for CSR/COO, row indices for CSC
    void *Ax,               // values (must match the type of A)
    GrB_Index *Ap_len,      // number of entries in Ap (not # of bytes)
    GrB_Index *Ai_len,      // number of entries in Ai (not # of bytes)
    GrB_Index *Ax_len,      // number of entries in Ax (not # of bytes)
    GrB_Format format,      // export format
    GrB_Matrix A            // matrix to export
) ;

#if GxB_STDC_VERSION >= 201112L
#define GrB_Matrix_export(Ap,Ai,Ax,Ap_len,Ai_len,Ax_len,fmt,A)  \
    _Generic                                                    \
    (                                                           \
        (Ax),                                                   \
            GB_PCASES (GrB, Matrix_export)                      \
    )                                                           \
    (Ap, Ai, Ax, Ap_len, Ai_len, Ax_len, fmt, A)
#endif

GrB_Info GrB_Matrix_exportSize  // determine sizes of user arrays for export
(
    GrB_Index *Ap_len,      // # of entries required for Ap (not # of bytes)
    GrB_Index *Ai_len,      // # of entries required for Ai (not # of bytes)
    GrB_Index *Ax_len,      // # of entries required for Ax (not # of bytes)
    GrB_Format format,      // export format
    GrB_Matrix A            // matrix to export
) ;

GrB_Info GrB_Matrix_exportHint  // suggest the best export format
(
    GrB_Format *format,     // export format
    GrB_Matrix A            // matrix to export
) ;

//==============================================================================
// serialize/deserialize
//==============================================================================

// GxB_Matrix_serialize copies the contents of a GrB_Matrix into a single array
// of bytes (the "blob").  The contents of the blob are implementation
// dependent.  The blob can be saved to a file, or sent across a communication
// channel, and then a GrB_Matrix can be reconstructed from the blob, even on
// another process or another machine, using the same version of
// SuiteSparse:GraphBLAS (v5.2.0 or later).  The goal is that future versions
// of SuiteSparse:GraphBLAS should be able to read in the blob as well, and
// reconstruct a matrix.  The matrix can be reconstructed from the blob using
// GxB_Matrix_deserialize.  The blob is compressed, by default, and
// uncompressed by GxB_Matrix_deserialize.

// GrB_Matrix_serialize/deserialize are slightly different from their GxB*
// counterparts.  The blob is allocated by GxB_Matrix_serialize, and must be
// freed by the same free() method passed to GxB_init (or the C11 free()
// if GrB_init was used).  By contrast, the GrB* methods require the user
// application to pass in a preallocated blob to GrB_Matrix_serialize, whose
// size can be given by GrB_Matrix_serializeSize (as a loose upper bound).

// The GrB* and GxB* methods can be mixed.  GrB_Matrix_serialize and
// GxB_Matrix_serialize construct the same blob (assuming they are given the
// same # of threads to do the work).  Both GrB_Matrix_deserialize and
// GxB_Matrix_deserialize can deserialize a blob coming from either
// GrB_Matrix_serialize or GxB_Matrix_serialize.

// Deserialization of untrusted data is a common security problem; see
// https://cwe.mitre.org/data/definitions/502.html. The deserialization methods
// below do a few basic checks so that no out-of-bounds access occurs during
// deserialization, but the output matrix itself may still be corrupted.  If
// the data is untrusted, use this to check the matrix:
//      GxB_Matrix_fprint (A, "A deserialized", GrB_SILENT, NULL)

// Example usage:

/*
    //--------------------------------------------------------------------------
    // using GxB serialize/deserialize
    //--------------------------------------------------------------------------

    // Given a GrB_Matrix A: assuming a user-defined type:
    void *blob ;
    GrB_Index blob_size ;
    GxB_Matrix_serialize (&blob, &blob_size, A, NULL) ;
    FILE *f = fopen ("myblob", "w") ;
    fwrite (blob_size, sizeof (size_t), 1, f) ;
    fwrite (blob, sizeof (uint8_t), blob_size, f) ;
    fclose (f) ;
    GrB_Matrix_free (&A) ;
    // B is a copy of A
    GxB_Matrix_deserialize (&B, MyQtype, blob, blob_size, NULL) ;
    GrB_Matrix_free (&B) ;
    free (blob) ;
    GrB_finalize ( ) ;

    // --- in another process, to recreate the GrB_Matrix A:
    GrB_init (GrB_NONBLOCKING) ;
    FILE *f = fopen ("myblob", "r") ;
    fread (&blob_size, sizeof (size_t), 1, f) ;
    blob = malloc (blob_size) ;
    fread (blob, sizeof (uint8_t), blob_size, f) ;
    fclose (f) ;
    char type_name [GxB_MAX_NAME_LEN] ;
    GrB_get (blob, type_name, GxB_JIT_C_NAME, blob_size) ;
    printf ("blob type is: %s\n", type_name) ;
    GrB_Type user_type = NULL ;
    if (strncmp (type_name, "myquaternion", GxB_MAX_NAME_LEN) == 0)
        user_type = MyQtype ;
    GxB_Matrix_deserialize (&A, user_type, blob, blob_size, NULL) ;
    free (blob) ;               // note, freed by the user, not GraphBLAS

    //--------------------------------------------------------------------------
    // using GrB serialize/deserialize
    //--------------------------------------------------------------------------

    // Given a GrB_Matrix A: assuming a user-defined type, MyQType:
    void *blob = NULL ;
    GrB_Index blob_size = 0 ;
    GrB_Matrix A, B = NULL ;
    // construct a matrix A, then serialized it:
    GrB_Matrix_serializeSize (&blob_size, A) ;      // loose upper bound
    blob = malloc (blob_size) ;
    GrB_Matrix_serialize (blob, &blob_size, A) ;    // returns actual size
    blob = realloc (blob, blob_size) ;              // user can shrink the blob
    FILE *f = fopen ("myblob", "w") ;
    fwrite (blob_size, sizeof (size_t), 1, f) ;
    fwrite (blob, sizeof (uint8_t), blob_size, f) ;
    fclose (f) ;
    GrB_Matrix_free (&A) ;
    // B is a copy of A:
    GrB_Matrix_deserialize (&B, MyQtype, blob, blob_size) ;
    GrB_Matrix_free (&B) ;
    free (blob) ;
    GrB_finalize ( ) ;

    // --- in another process, to recreate the GrB_Matrix A:
    GrB_init (GrB_NONBLOCKING) ;
    FILE *f = fopen ("myblob", "r") ;
    fread (&blob_size, sizeof (size_t), 1, f) ;
    blob = malloc (blob_size) ;
    fread (blob, sizeof (uint8_t), blob_size, f) ;
    fclose (f) ;
    // the user must know the type of A is MyQType
    GrB_Matrix_deserialize (&A, MyQtype, blob, blob_size) ;
    free (blob) ;
*/

// Most of the above methods have a level parameter that controls the tradeoff
// between run time and the amount of compression obtained.  Higher levels
// result in a more compact result, at the cost of higher run time:

//  LZ4     no level setting
//  LZ4HC   1: fast, 9: default, 9: max
//  ZSTD:   1: fast, 1: default, 19: max

// For all methods, a level of zero results in the default level setting.
// These settings can be added, so to use LZ4HC at level 5, use method =
// GxB_COMPRESSION_LZ4HC + 5.

// If the level setting is out of range, the default is used for that method.
// If the method is negative, no compression is performed.  If the method is
// positive but unrecognized, the default is used (GxB_COMPRESSION_ZSTD,
// level 1).

GrB_Info GxB_Matrix_serialize       // serialize a GrB_Matrix to a blob
(
    // output:
    void **blob_handle,             // the blob, allocated on output
    GrB_Index *blob_size_handle,    // size of the blob on output
    // input:
    GrB_Matrix A,                   // matrix to serialize
    const GrB_Descriptor desc       // descriptor to select compression method
                                    // and to control # of threads used
) ;

GrB_Info GrB_Matrix_serialize       // serialize a GrB_Matrix to a blob
(
    // output:
    void *blob,                     // the blob, already allocated in input
    // input/output:
    GrB_Index *blob_size_handle,    // size of the blob on input.  On output,
                                    // the # of bytes used in the blob.
    // input:
    GrB_Matrix A                    // matrix to serialize
) ;

GrB_Info GxB_Vector_serialize       // serialize a GrB_Vector to a blob
(
    // output:
    void **blob_handle,             // the blob, allocated on output
    GrB_Index *blob_size_handle,    // size of the blob on output
    // input:
    GrB_Vector u,                   // vector to serialize
    const GrB_Descriptor desc       // descriptor to select compression method
                                    // and to control # of threads used
) ;

GrB_Info GrB_Matrix_serializeSize   // estimate the size of a blob
(
    // output:
    GrB_Index *blob_size_handle,    // upper bound on the required size of the
                                    // blob on output.
    // input:
    GrB_Matrix A                    // matrix to serialize
) ;

// The GrB* and GxB* deserialize methods are nearly identical.  The GxB*
// deserialize methods simply add the descriptor, which allows for optional
// control of the # of threads used to deserialize the blob.

GrB_Info GxB_Matrix_deserialize     // deserialize blob into a GrB_Matrix
(
    // output:
    GrB_Matrix *C,      // output matrix created from the blob
    // input:
    GrB_Type type,      // type of the matrix C.  Required if the blob holds a
                        // matrix of user-defined type.  May be NULL if blob
                        // holds a built-in type; otherwise must match the
                        // type of C.
    const void *blob,       // the blob
    GrB_Index blob_size,    // size of the blob
    const GrB_Descriptor desc       // to control # of threads used
) ;

GrB_Info GrB_Matrix_deserialize     // deserialize blob into a GrB_Matrix
(
    // output:
    GrB_Matrix *C,      // output matrix created from the blob
    // input:
    GrB_Type type,      // type of the matrix C.  Required if the blob holds a
                        // matrix of user-defined type.  May be NULL if blob
                        // holds a built-in type; otherwise must match the
                        // type of C.
    const void *blob,       // the blob
    GrB_Index blob_size     // size of the blob
) ;

GrB_Info GxB_Vector_deserialize     // deserialize blob into a GrB_Vector
(
    // output:
    GrB_Vector *w,      // output vector created from the blob
    // input:
    GrB_Type type,      // type of the vector w.  Required if the blob holds a
                        // vector of user-defined type.  May be NULL if blob
                        // holds a built-in type; otherwise must match the
                        // type of w.
    const void *blob,       // the blob
    GrB_Index blob_size,    // size of the blob
    const GrB_Descriptor desc       // to control # of threads used
) ;

// historical; use GrB_get with GxB_JIT_C_NAME instead.
GrB_Info GxB_deserialize_type_name (char *, const void *, GrB_Index) ;

//==============================================================================
// GxB_Vector_sort and GxB_Matrix_sort: sort a matrix or vector
//==============================================================================

GrB_Info GxB_Vector_sort
(
    // output:
    GrB_Vector w,           // vector of sorted values
    GrB_Vector p,           // vector containing the permutation
    // input
    GrB_BinaryOp op,        // comparator op
    GrB_Vector u,           // vector to sort
    const GrB_Descriptor desc
) ;

GrB_Info GxB_Matrix_sort
(
    // output:
    GrB_Matrix C,           // matrix of sorted values
    GrB_Matrix P,           // matrix containing the permutations
    // input
    GrB_BinaryOp op,        // comparator op
    GrB_Matrix A,           // matrix to sort
    const GrB_Descriptor desc
) ;

#define GxB_sort(arg1,...)                                  \
    _Generic                                                \
    (                                                       \
        (arg1),                                             \
              GrB_Vector : GxB_Vector_sort ,                \
              GrB_Matrix : GxB_Matrix_sort                  \
    )                                                       \
    (arg1, __VA_ARGS__)

//==============================================================================
// GxB_Matrix_reshape and GxB_Matrix_reshapeDup:  reshape a matrix
//==============================================================================

// GxB_Matrix_reshape changes the dimensions of a matrix, reshaping the entries
// by row or by column.

// For example, if C is 3-by-4 on input, and is reshaped by column to have
// dimensions 2-by-6:

//      C on input      C on output (by_col true)
//      00 01 02 03     00 20 11 02 22 13
//      10 11 12 13     10 01 21 12 03 23
//      20 21 22 23

// If the same C on input is reshaped by row to dimensions 2-by-6:

//      C on input      C on output (by_col false)
//      00 01 02 03     00 01 02 03 10 11
//      10 11 12 13     12 13 20 21 22 23
//      20 21 22 23

// If the input matrix is nrows-by-ncols, and the size of the reshaped matrix
// is nrows_new-by-ncols_new, then nrows*ncols must equal nrows_new*ncols_new.
// The format of the input matrix (by row or by column) is unchanged; this
// format need not match the by_col input parameter.

GrB_Info GxB_Matrix_reshape     // reshape a GrB_Matrix in place
(
    // input/output:
    GrB_Matrix C,               // input/output matrix, reshaped in place
    // input:
    bool by_col,                // true if reshape by column, false if by row
    GrB_Index nrows_new,        // new number of rows of C
    GrB_Index ncols_new,        // new number of columns of C
    const GrB_Descriptor desc   // to control # of threads used
) ;

// GxB_Matrix_reshapeDup reshapes a matrix into another matrix.

// If the input matrix A is nrows-by-ncols, and the size of the newly-created
// matrix C is nrows_new-by-ncols_new, then nrows*ncols must equal
// nrows_new*ncols_new.  The format of the input matrix A (by row or by column)
// determines the format of the output matrix C, which need not match the
// by_col input parameter.

GrB_Info GxB_Matrix_reshapeDup // reshape a GrB_Matrix into another GrB_Matrix
(
    // output:
    GrB_Matrix *C,              // newly created output matrix, not in place
    // input:
    GrB_Matrix A,               // input matrix, not modified
    bool by_col,                // true if reshape by column, false if by row
    GrB_Index nrows_new,        // number of rows of C
    GrB_Index ncols_new,        // number of columns of C
    const GrB_Descriptor desc   // to control # of threads used
) ;

//==============================================================================
// GxB_Iterator: an object that iterates over the entries of a matrix or vector
//==============================================================================

/* Example usage:

single thread iteration of a whole matrix, one row at a time (in the
outer loop), and one entry at a time within the row (in the inner loop):

    // create an iterator
    GxB_Iterator iterator ;
    GxB_Iterator_new (&iterator) ;
    // attach it to the matrix A, known to be type GrB_FP64
    GrB_Info info = GxB_rowIterator_attach (iterator, A, NULL) ;
    if (info < 0) { handle the failure ... }
    // seek to A(0,:)
    info = GxB_rowIterator_seekRow (iterator, 0) ;
    while (info != GxB_EXHAUSTED)
    {
        // iterate over entries in A(i,:)
        GrB_Index i = GxB_rowIterator_getRowIndex (iterator) ;
        while (info == GrB_SUCCESS)
        {
            // get the entry A(i,j)
            GrB_Index j = GxB_rowIterator_getColIndex (iterator) ;
            double  aij = GxB_Iterator_get_FP64 (iterator) ;
            // move to the next entry in A(i,:)
            info = GxB_rowIterator_nextCol (iterator) ;
        }
        // move to the next row, A(i+1,:)
        info = GxB_rowIterator_nextRow (iterator) ;
    }
    GrB_free (&iterator) ;

parallel iteration using 4 threads (work may be imbalanced however):

    GrB_Index nrows ;
    GrB_wait (A, GrB_MATERIALIZE) ;     // this is essential
    GrB_Matrix_nrows (&nrows, A) ;
    #pragma omp parallel for num_threads(4)
    for (int tid = 0 ; tid < 4 ; tid++)
    {
        // thread tid operates on A(row1:row2-1,:)
        GrB_Index row1 = tid * (nrows / 4) ;
        GrB_Index row2 = (tid == 3) ? nrows : ((tid+1) * (nrows / 4)) ;
        GxB_Iterator iterator ;
        GxB_Iterator_new (&iterator) ;
        GrB_Info info = GxB_rowIterator_attach (iterator, A, NULL) ;
        if (info < 0) { handle the failure ... }
        // seek to A(row1,:)
        info = GxB_rowIterator_seekRow (iterator, row1) ;
        while (info != GxB_EXHAUSTED)
        {
            // iterate over entries in A(i,:)
            GrB_Index i = GxB_rowIterator_getRowIndex (iterator) ;
            if (i >= row2) break ;
            while (info == GrB_SUCCESS)
            {
                // get the entry A(i,j)
                GrB_Index j = GxB_rowIterator_getColIndex (iterator) ;
                double  aij = GxB_Iterator_get_FP64 (iterator) ;
                // move to the next entry in A(i,:)
                info = GxB_rowIterator_nextCol (iterator) ;
            }
            // move to the next row, A(i+1,:)
            info = GxB_rowIterator_nextRow (iterator) ;
        }
        GrB_free (&iterator) ;
    }

    In the parallel example above, a more balanced work distribution can be
    obtained by first computing the row degree via GrB_mxv (see LAGraph), and
    then compute the cumulative sum (ideally in parallel).  Next, partition the
    cumulative sum into one part per thread via binary search, and divide the
    rows into parts accordingly.

*/

//------------------------------------------------------------------------------
// GxB_Iterator: definition and new/free methods
//------------------------------------------------------------------------------

// The contents of an iterator must not be directly accessed by the user
// application.  Only the functions and macros provided here may access
// "iterator->..." contents.  The iterator is defined here only so that macros
// can be used to speed up the use of the iterator methods.  User applications
// must not use "iterator->..." directly.

struct GB_Iterator_opaque
{
    // these components change as the iterator moves (via seek or next):
    int64_t pstart ;            // the start of the current vector
    int64_t pend ;              // the end of the current vector
    int64_t p ;                 // position of the current entry
    int64_t k ;                 // the current vector

    // only changes when the iterator is created:
    size_t header_size ;        // size of this iterator object

    // these components only change when the iterator is attached:
    int64_t pmax ;              // avlen*avdim for bitmap; nvals(A) otherwise
    int64_t avlen ;             // length of each vector in the matrix
    int64_t avdim ;             // number of vectors in the matrix dimension
    int64_t anvec ;             // # of vectors present in the matrix
    const int64_t *GB_restrict Ap ;      // pointers for sparse and hypersparse
    const int64_t *GB_restrict Ah ;      // vector names for hypersparse
    const int8_t  *GB_restrict Ab ;      // bitmap
    const int64_t *GB_restrict Ai ;      // indices for sparse and hypersparse
    const void    *GB_restrict Ax ;      // values for all 4 data structures
    size_t type_size ;          // size of the type of A
    int A_sparsity ;            // sparse, hyper, bitmap, or full
    bool iso ;                  // true if A is iso-valued, false otherwise
    bool by_col ;               // true if A is held by column, false if by row
} ;

typedef struct GB_Iterator_opaque *GxB_Iterator ;

// GxB_Iterator_new: create a new iterator, not attached to any matrix/vector
GrB_Info GxB_Iterator_new (GxB_Iterator *iterator) ;

// GxB_Iterator_free: free an iterator
GrB_Info GxB_Iterator_free (GxB_Iterator *iterator) ;

//==============================================================================
// GB_Iterator_*: implements user-callable GxB_*Iterator_* methods
//==============================================================================

// GB_* methods are not user-callable.  These methods appear here so that the
// iterator methods can be done via macros.

//------------------------------------------------------------------------------
// GB_Iterator_attach: attach a row/col/entry iterator to a matrix
//------------------------------------------------------------------------------

GrB_Info GB_Iterator_attach
(
    GxB_Iterator iterator,      // iterator to attach to the matrix A
    GrB_Matrix A,               // matrix to attach
    GxB_Format_Value format,    // by row, by col, or by entry (GxB_NO_FORMAT)
    GrB_Descriptor desc
) ;

//------------------------------------------------------------------------------
// GB_Iterator_rc_seek: seek a row/col iterator to a particular vector
//------------------------------------------------------------------------------

GrB_Info GB_Iterator_rc_seek
(
    GxB_Iterator iterator,
    GrB_Index j,
    bool jth_vector
) ;

//------------------------------------------------------------------------------
// GB_Iterator_rc_bitmap_next: move a row/col iterator to next entry in bitmap
//------------------------------------------------------------------------------

GrB_Info GB_Iterator_rc_bitmap_next (GxB_Iterator iterator) ;

//------------------------------------------------------------------------------
// GB_Iterator_rc_knext: move a row/col iterator to the next vector
//------------------------------------------------------------------------------

#define GB_Iterator_rc_knext(iterator)                                      \
(                                                                           \
    /* move to the next vector, and check if iterator is exhausted */       \
    (++(iterator->k) >= iterator->anvec) ?                                  \
    (                                                                       \
        /* iterator is at the end of the matrix */                          \
        iterator->pstart = 0,                                               \
        iterator->pend = 0,                                                 \
        iterator->p = 0,                                                    \
        iterator->k = iterator->anvec,                                      \
        GxB_EXHAUSTED                                                       \
    )                                                                       \
    :                                                                       \
    (                                                                       \
        /* find first entry in vector, and pstart/pend for this vector */   \
        (iterator->A_sparsity <= GxB_SPARSE) ?                              \
        (                                                                   \
            /* matrix is sparse or hypersparse */                           \
            iterator->pstart = iterator->Ap [iterator->k],                  \
            iterator->pend = iterator->Ap [iterator->k+1],                  \
            iterator->p = iterator->pstart,                                 \
            ((iterator->p >= iterator->pend) ? GrB_NO_VALUE : GrB_SUCCESS)  \
        )                                                                   \
        :                                                                   \
        (                                                                   \
            /* matrix is bitmap or full */                                  \
            iterator->pstart += iterator->avlen,                            \
            iterator->pend += iterator->avlen,                              \
            iterator->p = iterator->pstart,                                 \
            (iterator->A_sparsity <= GxB_BITMAP) ?                          \
            (                                                               \
                /* matrix is bitmap */                                      \
                GB_Iterator_rc_bitmap_next (iterator)                       \
            )                                                               \
            :                                                               \
            (                                                               \
                /* matrix is full */                                        \
                ((iterator->p >= iterator->pend) ? GrB_NO_VALUE : GrB_SUCCESS) \
            )                                                               \
        )                                                                   \
    )                                                                       \
)

//------------------------------------------------------------------------------
// GB_Iterator_rc_inext: move a row/col iterator the next entry in the vector
//------------------------------------------------------------------------------

#define GB_Iterator_rc_inext(iterator)                                      \
(                                                                           \
    /* move to the next entry in the vector */                              \
    (++(iterator->p) >= iterator->pend) ?                                   \
    (                                                                       \
        /* no more entries in the current vector */                         \
        GrB_NO_VALUE                                                        \
    )                                                                       \
    :                                                                       \
    (                                                                       \
        (iterator->A_sparsity == GxB_BITMAP) ?                              \
        (                                                                   \
            /* the matrix is in bitmap form */                              \
            GB_Iterator_rc_bitmap_next (iterator)                           \
        )                                                                   \
        :                                                                   \
        (                                                                   \
            GrB_SUCCESS                                                     \
        )                                                                   \
    )                                                                       \
)


//------------------------------------------------------------------------------
// GB_Iterator_rc_getj: get index of current vector for row/col iterator
//------------------------------------------------------------------------------

#define GB_Iterator_rc_getj(iterator)                                       \
(                                                                           \
    (iterator->k >= iterator->anvec) ?                                      \
    (                                                                       \
        /* iterator is past the end of the matrix */                        \
        iterator->avdim                                                     \
    )                                                                       \
    :                                                                       \
    (                                                                       \
        (iterator->A_sparsity == GxB_HYPERSPARSE) ?                         \
        (                                                                   \
            /* return the name of kth vector: j = Ah [k] if it appears */   \
            iterator->Ah [iterator->k]                                      \
        )                                                                   \
        :                                                                   \
        (                                                                   \
            /* return the kth vector: j = k */                              \
            iterator->k                                                     \
        )                                                                   \
    )                                                                       \
)

//------------------------------------------------------------------------------
// GB_Iterator_rc_geti: return index of current entry for row/col iterator
//------------------------------------------------------------------------------

#define GB_Iterator_rc_geti(iterator)                                       \
(                                                                           \
    (iterator->Ai != NULL) ?                                                \
    (                                                                       \
        iterator->Ai [iterator->p]                                          \
    )                                                                       \
    :                                                                       \
    (                                                                       \
        (iterator->p - iterator->pstart)                                    \
    )                                                                       \
)

//==============================================================================
// GxB_rowIterator_*: iterate over the rows of a matrix
//==============================================================================

#undef GxB_rowIterator_attach
#undef GxB_rowIterator_kount
#undef GxB_rowIterator_seekRow
#undef GxB_rowIterator_kseek
#undef GxB_rowIterator_nextRow
#undef GxB_rowIterator_nextCol
#undef GxB_rowIterator_getRowIndex
#undef GxB_rowIterator_getColIndex

//------------------------------------------------------------------------------
// GxB_rowIterator_attach: attach a row iterator to a matrix
//------------------------------------------------------------------------------

// On input, the iterator must already exist, having been created by
// GxB_Iterator_new.

// GxB_rowIterator_attach attaches a row iterator to a matrix.  If the iterator
// is already attached to a matrix, it is detached and then attached to the
// given matrix A.

// The following error conditions are returned:
// GrB_NULL_POINTER:    if the iterator or A are NULL.
// GrB_INVALID_OBJECT:  if the matrix A is invalid.
// GrB_NOT_IMPLEMENTED: if the matrix A cannot be iterated by row.
// GrB_OUT_OF_MEMORY:   if the method runs out of memory.

// If successful, the row iterator is attached to the matrix, but not to any
// specific row.  Use GxB_rowIterator_*seek* to move the iterator to a row.

GrB_Info GxB_rowIterator_attach
(
    GxB_Iterator iterator,
    GrB_Matrix A,
    GrB_Descriptor desc
) ;

#define GxB_rowIterator_attach(iterator, A, desc)                           \
(                                                                           \
    GB_Iterator_attach (iterator, A, GxB_BY_ROW, desc)                      \
)

//------------------------------------------------------------------------------
// GxB_rowIterator_kount: upper bound on the # of nonempty rows of a matrix
//------------------------------------------------------------------------------

// On input, the row iterator must be attached to a matrix, but need not be at
// any specific row; results are undefined if this condition is not met.

// GxB_rowIterator_kount returns an upper bound on the # of non-empty rows of a
// matrix.  A GraphBLAS library may always return this as simply nrows(A), but
// in some libraries, it may be a value between the # of rows with at least one
// entry, and nrows(A), inclusive.  Any value in this range is a valid return
// value from this function.

// For SuiteSparse:GraphBLAS: If A is m-by-n, and sparse, bitmap, or full, then
// kount == m.  If A is hypersparse, kount is the # of vectors held in the data
// structure for the matrix, some of which may be empty, and kount <= m.

GrB_Index GxB_rowIterator_kount (GxB_Iterator iterator) ;

#define GxB_rowIterator_kount(iterator)                                     \
(                                                                           \
    (iterator)->anvec                                                       \
)

//------------------------------------------------------------------------------
// GxB_rowIterator_seekRow:  move a row iterator to a different row of a matrix
//------------------------------------------------------------------------------

// On input, the row iterator must be attached to a matrix, but need not be at
// any specific row; results are undefined if this condition is not met.

// GxB_rowIterator_seekRow moves a row iterator to the first entry of A(row,:).
// If A(row,:) has no entries, the iterator may move to the first entry of next
// nonempty row i for some i > row.  The row index can be determined by
// GxB_rowIterator_getRowIndex.

// For SuiteSparse:GraphBLAS: If the matrix is hypersparse, and the row
// does not appear in the hyperlist, then the iterator is moved to the first
// row after the given row that does appear in the hyperlist.

// The method is always successful; the following are conditions are returned:
// GxB_EXHAUSTED:   if the row index is >= nrows(A); the row iterator is
//                  exhausted, but is still attached to the matrix.
// GrB_NO_VALUE:    if the row index is valid but A(row,:) has no entries; the
//                  row iterator is positioned at A(row,:).
// GrB_SUCCESS:     if the row index is valid and A(row,:) has at least one
//                  entry. The row iterator is positioned at A(row,:).
//                  GxB_rowIterator_get* can be used to return the indices of
//                  the first entry in A(row,:), and GxB_Iterator_get* can
//                  return its value.

GrB_Info GxB_rowIterator_seekRow (GxB_Iterator iterator, GrB_Index row) ;

#define GxB_rowIterator_seekRow(iterator, row)                              \
(                                                                           \
    GB_Iterator_rc_seek (iterator, row, false)                              \
)

//------------------------------------------------------------------------------
// GxB_rowIterator_kseek:  move a row iterator to a different row of a matrix
//------------------------------------------------------------------------------

// On input, the row iterator must be attached to a matrix, but need not be at
// any specific row; results are undefined if this condition is not met.

// GxB_rowIterator_kseek is identical to GxB_rowIterator_seekRow, except for
// how the row index is specified.  The row is the kth non-empty row of A.
// More precisely, k is in the range 0 to kount-1, where kount is the value
// returned by GxB_rowIterator_kount.

GrB_Info GxB_rowIterator_kseek (GxB_Iterator iterator, GrB_Index k) ;

#define GxB_rowIterator_kseek(iterator, k)                                  \
(                                                                           \
    GB_Iterator_rc_seek (iterator, k, true)                                 \
)

//------------------------------------------------------------------------------
// GxB_rowIterator_nextRow: move a row iterator to the next row of a matrix
//------------------------------------------------------------------------------

// On input, the row iterator must already be attached to a matrix via a prior
// call to GxB_rowIterator_attach, and the iterator must be at a specific row,
// via a prior call to GxB_rowIterator_*seek* or GxB_rowIterator_nextRow;
// results are undefined if this condition is not met.

// If the the row iterator is currently at A(row,:), it is moved to A(row+1,:),
// or to the first non-empty row after A(row,:), at the discretion of this
// method.  That is, empty rows may be skipped.

// The method is always successful, and the return conditions are identical to
// the return conditions of GxB_rowIterator_seekRow.

GrB_Info GxB_rowIterator_nextRow (GxB_Iterator iterator) ;

#define GxB_rowIterator_nextRow(iterator)                                   \
(                                                                           \
    GB_Iterator_rc_knext (iterator)                                         \
)

//------------------------------------------------------------------------------
// GxB_rowIterator_nextCol: move a row iterator to the next entry in A(row,:)
//------------------------------------------------------------------------------

// On input, the row iterator must already be attached to a matrix via a prior
// call to GxB_rowIterator_attach, and the iterator must be at a specific row,
// via a prior call to GxB_rowIterator_*seek* or GxB_rowIterator_nextRow;
// results are undefined if this condition is not met.

// The method is always successful, and returns the following conditions:
// GrB_NO_VALUE:    If the iterator is already exhausted, or if there is no
//                  entry in the current A(row,;),
// GrB_SUCCESS:     If the row iterator has been moved to the next entry in
//                  A(row,:).

GrB_Info GxB_rowIterator_nextCol (GxB_Iterator iterator) ;

#define GxB_rowIterator_nextCol(iterator)                                   \
(                                                                           \
    GB_Iterator_rc_inext ((iterator))                                       \
)

//------------------------------------------------------------------------------
// GxB_rowIterator_getRowIndex: get current row index of a row iterator
//------------------------------------------------------------------------------

// On input, the iterator must be already successfully attached to matrix as a
// row iterator; results are undefined if this condition is not met.

// The method returns nrows(A) if the iterator is exhausted, or the current
// row index otherwise.  There need not be any entry in the current row.
// Zero is returned if the iterator is attached to the matrix but
// GxB_rowIterator_*seek* has not been called, but this does not mean the
// iterator is positioned at row zero.

GrB_Index GxB_rowIterator_getRowIndex (GxB_Iterator iterator) ;

#define GxB_rowIterator_getRowIndex(iterator)                               \
(                                                                           \
    GB_Iterator_rc_getj ((iterator))                                        \
)

//------------------------------------------------------------------------------
// GxB_rowIterator_getColIndex: get current column index of a row iterator
//------------------------------------------------------------------------------

// On input, the iterator must be already successfully attached to matrix as a
// row iterator, and in addition, the row iterator must be positioned at a
// valid entry present in the matrix.  That is, the last call to
// GxB_rowIterator_*seek* or GxB_rowIterator_*next*, must have returned
// GrB_SUCCESS.  Results are undefined if this condition is not met.

GrB_Index GxB_rowIterator_getColIndex (GxB_Iterator iterator) ;

#define GxB_rowIterator_getColIndex(iterator)                               \
(                                                                           \
    GB_Iterator_rc_geti ((iterator))                                        \
)

//==============================================================================
// GxB_colIterator_*: iterate over columns of a matrix
//==============================================================================

// The column iterator is analoguous to the row iterator.

#undef GxB_colIterator_attach
#undef GxB_colIterator_kount
#undef GxB_colIterator_seekCol
#undef GxB_colIterator_kseek
#undef GxB_colIterator_nextCol
#undef GxB_colIterator_nextRow
#undef GxB_colIterator_getColIndex
#undef GxB_colIterator_getRowIndex

// GxB_colIterator_attach: attach a column iterator to a matrix
GrB_Info GxB_colIterator_attach
(
    GxB_Iterator iterator,
    GrB_Matrix A,
    GrB_Descriptor desc
) ;
#define GxB_colIterator_attach(iterator, A, desc)                           \
(                                                                           \
    GB_Iterator_attach (iterator, A, GxB_BY_COL, desc)                      \
)

// GxB_colIterator_kount: return # of nonempty columns of the matrix
GrB_Index GxB_colIterator_kount (GxB_Iterator iterator) ;
#define GxB_colIterator_kount(iterator)                                     \
(                                                                           \
    (iterator)->anvec                                                       \
)

// GxB_colIterator_seekCol: move a column iterator to A(:,col)
GrB_Info GxB_colIterator_seekCol (GxB_Iterator iterator, GrB_Index col) ;
#define GxB_colIterator_seekCol(iterator, col)                              \
(                                                                           \
    GB_Iterator_rc_seek (iterator, col, false)                              \
)

// GxB_colIterator_kseek: move a column iterator to kth non-empty column of A
GrB_Info GxB_colIterator_kseek (GxB_Iterator iterator, GrB_Index k) ;
#define GxB_colIterator_kseek(iterator, k)                                  \
(                                                                           \
    GB_Iterator_rc_seek (iterator, k, true)                                 \
)

// GxB_colIterator_nextCol: move a column iterator to first entry of next column
GrB_Info GxB_colIterator_nextCol (GxB_Iterator iterator) ;
#define GxB_colIterator_nextCol(iterator)                                   \
(                                                                           \
    GB_Iterator_rc_knext ((iterator))                                       \
)

// GxB_colIterator_nextRow: move a column iterator to next entry in column
GrB_Info GxB_colIterator_nextRow (GxB_Iterator iterator) ;
#define GxB_colIterator_nextRow(iterator)                                   \
(                                                                           \
    GB_Iterator_rc_inext ((iterator))                                       \
)

// GxB_colIterator_getColIndex: return the column index of current entry
GrB_Index GxB_colIterator_getColIndex (GxB_Iterator iterator) ;
#define GxB_colIterator_getColIndex(iterator)                               \
(                                                                           \
    GB_Iterator_rc_getj ((iterator))                                        \
)

// GxB_colIterator_getRowIndex: return the row index of current entry
GrB_Index GxB_colIterator_getRowIndex (GxB_Iterator iterator) ;
#define GxB_colIterator_getRowIndex(iterator)                               \
(                                                                           \
    GB_Iterator_rc_geti ((iterator))                                        \
)

//==============================================================================
// GxB_Matrix_Iterator_*: iterate over the entries of a matrix
//==============================================================================

// Example usage:

// single thread iteration of a whole matrix, one entry at at time

/*
    // create an iterator
    GxB_Iterator iterator ;
    GxB_Iterator_new (&iterator) ;
    // attach it to the matrix A, known to be type GrB_FP64
    GrB_Info info = GxB_Matrix_Iterator_attach (iterator, A, NULL) ;
    if (info < 0) { handle the failure ... }
    // seek to the first entry
    info = GxB_Matrix_Iterator_seek (iterator, 0) ;
    while (info != GxB_EXHAUSTED)
    {
        // get the entry A(i,j)
        GrB_Index i, j ;
        GxB_Matrix_Iterator_getIndex (iterator, &i, &j) ;
        double aij = GxB_Iterator_get_FP64 (iterator) ;
        // move to the next entry in A
        info = GxB_Matrix_Iterator_next (iterator) ;
    }
    GrB_free (&iterator) ;
*/

//------------------------------------------------------------------------------
// GxB_Matrix_Iterator_attach: attach an entry iterator to a matrix
//------------------------------------------------------------------------------

// On input, the iterator must already exist, having been created by
// GxB_Iterator_new.

// GxB_Matrix_Iterator_attach attaches an entry iterator to a matrix.  If the
// iterator is already attached to a matrix, it is detached and then attached
// to the given matrix A.

// The following error conditions are returned:
// GrB_NULL_POINTER:    if the iterator or A are NULL.
// GrB_INVALID_OBJECT:  if the matrix A is invalid.
// GrB_OUT_OF_MEMORY:   if the method runs out of memory.

// If successful, the entry iterator is attached to the matrix, but not to any
// specific entry.  Use GxB_Matrix_Iterator_*seek* to move the iterator to a
// particular entry.

GrB_Info GxB_Matrix_Iterator_attach
(
    GxB_Iterator iterator,
    GrB_Matrix A,
    GrB_Descriptor desc
) ;

//------------------------------------------------------------------------------
// GxB_Matrix_Iterator_getpmax: return the range of the iterator
//------------------------------------------------------------------------------

// On input, the entry iterator must be already attached to a matrix via
// GxB_Matrix_Iterator_attach; results are undefined if this condition is not
// met.

// Entries in a matrix are given an index p, ranging from 0 to pmax-1, where
// pmax >= nvals(A).  For sparse, hypersparse, and full matrices, pmax is equal
// to nvals(A).  For an m-by-n bitmap matrix, pmax=m*n, or pmax=0 if the
// matrix has no entries.

GrB_Index GxB_Matrix_Iterator_getpmax (GxB_Iterator iterator) ;

//------------------------------------------------------------------------------
// GxB_Matrix_Iterator_seek: seek to a specific entry
//------------------------------------------------------------------------------

// On input, the entry iterator must be already attached to a matrix via
// GxB_Matrix_Iterator_attach; results are undefined if this condition is not
// met.

// The input p is in range 0 to pmax-1, which points to an entry in the matrix,
// or p >= pmax if the iterator is exhausted, where pmax is the return value
// from GxB_Matrix_Iterator_getpmax.

// Returns GrB_SUCCESS if the iterator is at an entry that exists in the
// matrix, or GxB_EXHAUSTED if the iterator is exhausted.

GrB_Info GxB_Matrix_Iterator_seek (GxB_Iterator iterator, GrB_Index p) ;

//------------------------------------------------------------------------------
// GxB_Matrix_Iterator_next: move to the next entry of a matrix
//------------------------------------------------------------------------------

// On input, the entry iterator must be already attached to a matrix via
// GxB_Matrix_Iterator_attach, and the position of the iterator must also have
// been defined by a prior call to GxB_Matrix_Iterator_seek or
// GxB_Matrix_Iterator_next.  Results are undefined if these conditions are not
// met.

// Returns GrB_SUCCESS if the iterator is at an entry that exists in the
// matrix, or GxB_EXHAUSTED if the iterator is exhausted.

GrB_Info GxB_Matrix_Iterator_next (GxB_Iterator iterator) ;

//------------------------------------------------------------------------------
// GxB_Matrix_Iterator_getp: get the current position of a matrix iterator
//------------------------------------------------------------------------------

// On input, the entry iterator must be already attached to a matrix via
// GxB_Matrix_Iterator_attach, and the position of the iterator must also have
// been defined by a prior call to GxB_Matrix_Iterator_seek or
// GxB_Matrix_Iterator_next.  Results are undefined if these conditions are not
// met.

GrB_Index GxB_Matrix_Iterator_getp (GxB_Iterator iterator) ;

//------------------------------------------------------------------------------
// GxB_Matrix_Iterator_getIndex: get the row and column index of a matrix entry
//------------------------------------------------------------------------------

// On input, the entry iterator must be already attached to a matrix via
// GxB_Matrix_Iterator_attach, and the position of the iterator must also have
// been defined by a prior call to GxB_Matrix_Iterator_seek or
// GxB_Matrix_Iterator_next, with a return value of GrB_SUCCESS.  Results are
// undefined if these conditions are not met.

void GxB_Matrix_Iterator_getIndex
(
    GxB_Iterator iterator,
    GrB_Index *row,
    GrB_Index *col
) ;

//==============================================================================
// GxB_Vector_Iterator_*: iterate over the entries of a vector
//==============================================================================

/* Example usage:

single thread iteration of a whole vector, one entry at at time

    // create an iterator
    GxB_Iterator iterator ;
    GxB_Iterator_new (&iterator) ;
    // attach it to the vector v, known to be type GrB_FP64
    GrB_Info info = GxB_Vector_Iterator_attach (iterator, v, NULL) ;
    if (info < 0) { handle the failure ... }
    // seek to the first entry
    info = GxB_Vector_Iterator_seek (iterator, 0) ;
    while (info != GxB_EXHAUSTED)
    {
        // get the entry v(i)
        GrB_Index i = GxB_Vector_Iterator_getIndex (iterator) ;
        double vi = GxB_Iterator_get_FP64 (iterator) ;
        // move to the next entry in v
        info = GxB_Vector_Iterator_next (iterator) ;
    }
    GrB_free (&iterator) ;

*/

#undef GxB_Vector_Iterator_getpmax
#undef GxB_Vector_Iterator_seek
#undef GxB_Vector_Iterator_next
#undef GxB_Vector_Iterator_getp
#undef GxB_Vector_Iterator_getIndex

//------------------------------------------------------------------------------
// GxB_Vector_Iterator_attach: attach an iterator to a vector
//------------------------------------------------------------------------------

// On input, the iterator must already exist, having been created by
// GxB_Iterator_new.

// GxB_Vector_Iterator_attach attaches an iterator to a vector.  If the
// iterator is already attached to a vector or matrix, it is detached and then
// attached to the given vector v.

// The following error conditions are returned:
// GrB_NULL_POINTER:    if the iterator or v are NULL.
// GrB_INVALID_OBJECT:  if the vector v is invalid.
// GrB_OUT_OF_MEMORY:   if the method runs out of memory.

// If successful, the iterator is attached to the vector, but not to any
// specific entry.  Use GxB_Vector_Iterator_seek to move the iterator to a
// particular entry.

GrB_Info GxB_Vector_Iterator_attach
(
    GxB_Iterator iterator,
    GrB_Vector v,
    GrB_Descriptor desc
) ;

//------------------------------------------------------------------------------
// GxB_Vector_Iterator_getpmax: return the range of the vector iterator
//------------------------------------------------------------------------------

// On input, the iterator must be already attached to a vector via
// GxB_Vector_Iterator_attach; results are undefined if this condition is not
// met.

// Entries in a vector are given an index p, ranging from 0 to pmax-1, where
// pmax >= nvals(v).  For sparse and full vectors, pmax is equal to nvals(v).
// For a size-m bitmap vector, pmax=m, or pmax=0 if the vector has no entries.

GrB_Index GxB_Vector_Iterator_getpmax (GxB_Iterator iterator) ;

#define GxB_Vector_Iterator_getpmax(iterator)                               \
(                                                                           \
    (iterator->pmax)                                                        \
)

//------------------------------------------------------------------------------
// GxB_Vector_Iterator_seek: seek to a specific entry in the vector
//------------------------------------------------------------------------------

// On input, the iterator must be already attached to a vector via
// GxB_Vector_Iterator_attach; results are undefined if this condition is not
// met.

// The input p is in range 0 to pmax-1, which points to an entry in the vector,
// or p >= pmax if the iterator is exhausted, where pmax is the return value
// from GxB_Vector_Iterator_getpmax.

// Returns GrB_SUCCESS if the iterator is at an entry that exists in the
// vector, or GxB_EXHAUSTED if the iterator is exhausted.

GrB_Info GB_Vector_Iterator_bitmap_seek (GxB_Iterator iterator,
    GrB_Index unused) ; // unused parameter to be removed in v8.x

GrB_Info GxB_Vector_Iterator_seek (GxB_Iterator iterator, GrB_Index p) ;

#define GB_Vector_Iterator_seek(iterator, q)                                \
(                                                                           \
    (q >= iterator->pmax) ?                                                 \
    (                                                                       \
        /* the iterator is exhausted */                                     \
        iterator->p = iterator->pmax,                                       \
        GxB_EXHAUSTED                                                       \
    )                                                                       \
    :                                                                       \
    (                                                                       \
        /* seek to an arbitrary position in the vector */                   \
        iterator->p = q,                                                    \
        (iterator->A_sparsity == GxB_BITMAP) ?                              \
        (                                                                   \
            GB_Vector_Iterator_bitmap_seek (iterator, 0)                    \
        )                                                                   \
        :                                                                   \
        (                                                                   \
            GrB_SUCCESS                                                     \
        )                                                                   \
    )                                                                       \
)

#define GxB_Vector_Iterator_seek(iterator, p)                               \
(                                                                           \
    GB_Vector_Iterator_seek (iterator, p)                                   \
)

//------------------------------------------------------------------------------
// GxB_Vector_Iterator_next: move to the next entry of a vector
//------------------------------------------------------------------------------

// On input, the iterator must be already attached to a vector via
// GxB_Vector_Iterator_attach, and the position of the iterator must also have
// been defined by a prior call to GxB_Vector_Iterator_seek or
// GxB_Vector_Iterator_next.  Results are undefined if these conditions are not
// met.

// Returns GrB_SUCCESS if the iterator is at an entry that exists in the
// vector, or GxB_EXHAUSTED if the iterator is exhausted.

GrB_Info GxB_Vector_Iterator_next (GxB_Iterator iterator) ;

#define GB_Vector_Iterator_next(iterator)                                   \
(                                                                           \
    /* move to the next entry */                                            \
    (++(iterator->p) >= iterator->pmax) ?                                   \
    (                                                                       \
        /* the iterator is exhausted */                                     \
        iterator->p = iterator->pmax,                                       \
        GxB_EXHAUSTED                                                       \
    )                                                                       \
    :                                                                       \
    (                                                                       \
        (iterator->A_sparsity == GxB_BITMAP) ?                              \
        (                                                                   \
            /* bitmap: seek to the next entry present in the bitmap */      \
            GB_Vector_Iterator_bitmap_seek (iterator, 0)                    \
        )                                                                   \
        :                                                                   \
        (                                                                   \
            /* other formats: already at the next entry */                  \
            GrB_SUCCESS                                                     \
        )                                                                   \
    )                                                                       \
)

#define GxB_Vector_Iterator_next(iterator)                                  \
(                                                                           \
    GB_Vector_Iterator_next (iterator)                                      \
)

//------------------------------------------------------------------------------
// GxB_Vector_Iterator_getp: get the current position of a vector iterator
//------------------------------------------------------------------------------

// On input, the iterator must be already attached to a vector via
// GxB_Vector_Iterator_attach, and the position of the iterator must also have
// been defined by a prior call to GxB_Vector_Iterator_seek or
// GxB_Vector_Iterator_next.  Results are undefined if these conditions are not
// met.

GrB_Index GxB_Vector_Iterator_getp (GxB_Iterator iterator) ;

#define GxB_Vector_Iterator_getp(iterator)                                  \
(                                                                           \
    (iterator->p)                                                           \
)

//------------------------------------------------------------------------------
// GxB_Vector_Iterator_getIndex: get the index of a vector entry
//------------------------------------------------------------------------------

// On input, the iterator must be already attached to a vector via
// GxB_Vector_Iterator_attach, and the position of the iterator must also have
// been defined by a prior call to GxB_Vector_Iterator_seek or
// GxB_Vector_Iterator_next, with a return value of GrB_SUCCESS.  Results are
// undefined if these conditions are not met.

GrB_Index GxB_Vector_Iterator_getIndex (GxB_Iterator iterator) ;

#define GxB_Vector_Iterator_getIndex(iterator)                              \
(                                                                           \
    ((iterator->Ai != NULL) ? iterator->Ai [iterator->p] : iterator->p)     \
)

//==============================================================================
// GxB_Iterator_get_TYPE: get value of the current entry for any iterator
//==============================================================================

// On input, the prior call to GxB_*Iterator_*seek*, or GxB_*Iterator_*next*
// must have returned GrB_SUCCESS, indicating that the iterator is at a valid
// current entry for either a matrix or vector.

// Returns the value of the current entry at the position determined by the
// iterator.  No typecasting is permitted; the method name must match the
// type of the matrix or vector.

#undef GxB_Iterator_get_BOOL
#undef GxB_Iterator_get_INT8
#undef GxB_Iterator_get_INT16
#undef GxB_Iterator_get_INT32
#undef GxB_Iterator_get_INT64
#undef GxB_Iterator_get_UINT8
#undef GxB_Iterator_get_UINT16
#undef GxB_Iterator_get_UINT32
#undef GxB_Iterator_get_UINT64
#undef GxB_Iterator_get_FP32
#undef GxB_Iterator_get_FP64
#undef GxB_Iterator_get_FC32
#undef GxB_Iterator_get_FC64
#undef GxB_Iterator_get_UDT

bool       GxB_Iterator_get_BOOL   (GxB_Iterator iterator) ;
int8_t     GxB_Iterator_get_INT8   (GxB_Iterator iterator) ;
int16_t    GxB_Iterator_get_INT16  (GxB_Iterator iterator) ;
int32_t    GxB_Iterator_get_INT32  (GxB_Iterator iterator) ;
int64_t    GxB_Iterator_get_INT64  (GxB_Iterator iterator) ;
uint8_t    GxB_Iterator_get_UINT8  (GxB_Iterator iterator) ;
uint16_t   GxB_Iterator_get_UINT16 (GxB_Iterator iterator) ;
uint32_t   GxB_Iterator_get_UINT32 (GxB_Iterator iterator) ;
uint64_t   GxB_Iterator_get_UINT64 (GxB_Iterator iterator) ;
float      GxB_Iterator_get_FP32   (GxB_Iterator iterator) ;
double     GxB_Iterator_get_FP64   (GxB_Iterator iterator) ;
GxB_FC32_t GxB_Iterator_get_FC32   (GxB_Iterator iterator) ;
GxB_FC64_t GxB_Iterator_get_FC64   (GxB_Iterator iterator) ;
void       GxB_Iterator_get_UDT    (GxB_Iterator iterator,
                                              void *value) ;

#define GB_Iterator_get(iterator, type)                                     \
(                                                                           \
    (((type *) (iterator)->Ax) [(iterator)->iso ? 0 : (iterator)->p])       \
)

#define GxB_Iterator_get_BOOL(iterator)   GB_Iterator_get (iterator, bool)
#define GxB_Iterator_get_INT8(iterator)   GB_Iterator_get (iterator, int8_t)
#define GxB_Iterator_get_INT16(iterator)  GB_Iterator_get (iterator, int16_t)
#define GxB_Iterator_get_INT32(iterator)  GB_Iterator_get (iterator, int32_t)
#define GxB_Iterator_get_INT64(iterator)  GB_Iterator_get (iterator, int64_t)
#define GxB_Iterator_get_UINT8(iterator)  GB_Iterator_get (iterator, uint8_t)
#define GxB_Iterator_get_UINT16(iterator) GB_Iterator_get (iterator, uint16_t)
#define GxB_Iterator_get_UINT32(iterator) GB_Iterator_get (iterator, uint32_t)
#define GxB_Iterator_get_UINT64(iterator) GB_Iterator_get (iterator, uint64_t)
#define GxB_Iterator_get_FP32(iterator)   GB_Iterator_get (iterator, float)
#define GxB_Iterator_get_FP64(iterator)   GB_Iterator_get (iterator, double)
#define GxB_Iterator_get_FC32(iterator)   GB_Iterator_get (iterator, GxB_FC32_t)
#define GxB_Iterator_get_FC64(iterator)   GB_Iterator_get (iterator, GxB_FC64_t)

#define GxB_Iterator_get_UDT(iterator, value)                               \
(                                                                           \
    (void) memcpy ((void *) value, ((const uint8_t *) ((iterator)->Ax)) +   \
        ((iterator)->iso ? 0 : ((iterator)->type_size * (iterator)->p)),    \
        (iterator)->type_size)                                              \
)

#endif  // GB_CUDA_FOLDER
#if defined ( __cplusplus )
}
#endif
#endif

