//------------------------------------------------------------------------------
// LAGraph.h: user-visible include file for LAGraph
//------------------------------------------------------------------------------

// LAGraph, (c) 2019-2023 by The LAGraph Contributors, All Rights Reserved.
// SPDX-License-Identifier: BSD-2-Clause
//
// For additional details (including references to third party source code and
// other files) see the LICENSE file or contact permission@sei.cmu.edu. See
// Contributors.txt for a full list of contributors. Created, in part, with
// funding and support from the U.S. Government (see Acknowledgments.txt file).
// DM22-0790

//------------------------------------------------------------------------------

// LAGraph is a package of graph algorithms based on GraphBLAS.  GraphBLAS
// defines a set of sparse matrix operations on an extended algebra of
// semirings, using an almost unlimited variety of operators and types.  When
// applied to sparse adjacency matrices, these algebraic operations are
// equivalent to computations on graphs.  GraphBLAS provides a powerful and
// expressive framework creating graph algorithms based on the elegant
// mathematics of sparse matrix operations on a semiring.

// However, GraphBLAS itself does not have graph algorithms.  The purpose of
// LAGraph is to provide a robust, easy-to-use high-performance library of
// graph algorithms that rely on GraphBLAS.

//------------------------------------------------------------------------------

#ifndef LAGRAPH_H
#define LAGRAPH_H

//==============================================================================
// LAGraph version
//==============================================================================

// See also the LAGraph_Version utility method, which returns these values.
// These definitions are derived from LAGraph/CMakeLists.txt.

#define LAGRAPH_DATE "Feb 20, 2025"
#define LAGRAPH_VERSION_MAJOR  1
#define LAGRAPH_VERSION_MINOR  1
#define LAGRAPH_VERSION_UPDATE 5

//==============================================================================
// include files and helper macros
//==============================================================================

#include <GraphBLAS.h>
#if defined ( _OPENMP )
    #include <omp.h>
#endif

// LAGRAPH_MIN/MAX: suitable for integers, and non-NaN floating point
#define LAGRAPH_MIN(x,y) (((x) < (y)) ? (x) : (y))
#define LAGRAPH_MAX(x,y) (((x) > (y)) ? (x) : (y))

//==============================================================================
// GraphBLAS platform specifics
//==============================================================================

// GraphBLAS C API specification, OpenMP, and vanilla vs
// SuiteSparse:GraphBLAS GxB extensions.

#if ( GRB_VERSION < 2 )
    #error "The GraphBLAS library must support the v2.0 C API Specification"
#endif

#if ( _MSC_VER && !__INTEL_COMPILER && LG_DLL )
    #ifdef LG_LIBRARY
        // compiling LAGraph itself, exporting symbols to user apps
        #define LAGRAPH_PUBLIC __declspec ( dllexport )
    #else
        // compiling the user application, importing symbols from LAGraph
        #define LAGRAPH_PUBLIC __declspec ( dllimport )
    #endif
#else
    // for other compilers
    #define LAGRAPH_PUBLIC
#endif

#if defined ( __cplusplus )
    // C++ does not have the restrict keyword
    #define LAGRAPH_RESTRICT
#elif ( _MSC_VER && !__INTEL_COMPILER )
    // Microsoft Visual Studio uses __restrict instead of restrict for C
    #define LAGRAPH_RESTRICT __restrict
#else
    // use the restrict keyword for ANSI C99 compilers
    #define LAGRAPH_RESTRICT restrict
#endif

// vanilla vs SuiteSparse:
#if !defined ( LAGRAPH_VANILLA )
    // by default, set LAGRAPH_VANILLA to false
    #define LAGRAPH_VANILLA 0
#endif

#if ( !LAGRAPH_VANILLA ) && defined ( GxB_SUITESPARSE_GRAPHBLAS )
    // use SuiteSparse, and its GxB* extensions
    #if GxB_IMPLEMENTATION < GxB_VERSION (7,1,0)
    #error "If using SuiteSparse::GraphBLAS, version 7.1.0 or later is required"
    #endif
    #define LAGRAPH_SUITESPARSE 1
#else
    // use any GraphBLAS library (possibly SuiteSparse) but with no GxB*
    #define LAGRAPH_SUITESPARSE 0
#endif

// maximum length of the name of a GrB type, including the null-terminator
#if LAGRAPH_SUITESPARSE
#define LAGRAPH_MAX_NAME_LEN GxB_MAX_NAME_LEN
#else
#define LAGRAPH_MAX_NAME_LEN 128
#endif

//==============================================================================
// LAGraph error handling: return values and msg string
//==============================================================================

/**
 * Nearly all LAGraph methods return an int to denote their status, and
 * have a final string (msg) that captures any error messages.
 *
 * LAGraph has a single function that does not follow this rule.
 * @sphinxref{LAGraph_WallClockTime} has no error handling mechanism (it
 * returns a value of type double, and does not have an final msg string
 * parameter.
 *
 * All other methods return an int to denote their status:  zero if they are
 * successful (which is the value of GrB_SUCCESS), negative on error, or
 * positive for an informational value (such as GrB_NO_VALUE).  Integers in the
 * range -999 to 999 are reserved for GraphBLAS GrB_Info return values:
 *
 * \rst_star{
 *  successful results:
 *    - GrB_SUCCESS = 0             // all is well
 *    - GrB_NO_VALUE = 1            // A(i,j) requested but not there
 *
 *  errors:
 *    - GrB_UNINITIALIZED_OBJECT = -1   // object has not been initialized
 *    - GrB_NULL_POINTER = -2           // input pointer is NULL
 *    - GrB_INVALID_VALUE = -3          // generic error; some value is bad
 *    - GrB_INVALID_INDEX = -4          // row or column index is out of bounds
 *    - GrB_DOMAIN_MISMATCH = -5        // object domains are not compatible
 *    - GrB_DIMENSION_MISMATCH = -6     // matrix dimensions do not match
 *    - GrB_OUTPUT_NOT_EMPTY = -7       // output matrix already has values
 *    - GrB_NOT_IMPLEMENTED = -8        // method not implemented
 *    - GrB_PANIC = -101                // unknown error
 *    - GrB_OUT_OF_MEMORY = -102        // out of memory
 *    - GrB_INSUFFICIENT_SPACE = -103,  // output array not large enough
 *    - GrB_INVALID_OBJECT = -104       // object is corrupted
 *    - GrB_INDEX_OUT_OF_BOUNDS = -105  // row or col index out of bounds
 *    - GrB_EMPTY_OBJECT = -106         // an object does not contain a value
 * }
 * LAGraph returns any errors it receives from GraphBLAS, and also uses the
 * GrB_* error codes in these cases:
 *    - GrB_INVALID_INDEX: if a source node id is out of range
 *    - GrB_INVALID_VALUE: if an enum to select an option is out of range
 *    - GrB_NOT_IMPLEMENTED: if a type is not supported, or when SuiteSparse
 *          GraphBLAS is required.
 *
 * Summary of return values for all LAGraph functions that return int:
 *    - GrB_SUCCESS if successful
 *    - a negative GrB_Info value on error (in range -999 to -1)
 *    - a positive GrB_Info value if successful but with extra information
 *      (in range 1 to 999)
 *    - -1999 to -1000: a common LAGraph-specific error, see list above
 *    - 1000 to 1999: if successful, with extra LAGraph-specific information
 *    - <= -2000: an LAGraph error specific to a particular LAGraph method
 *    - >= 2000: an LAGraph warning specific to a particular LAGraph method
 *
 * Many LAGraph methods share common error cases, described below.  These
 * return values are in the range -1000 to -1999.  Return values of -2000 or
 * greater may be used by specific LAGraph methods, which denote errors not in
 * the following list:
 * \rst_star{
 *    - LAGRAPH_INVALID_GRAPH (-1000):
 *          The input graph is invalid; the details are given in the error msg
 *          string returned by the method.
 *    - LAGRAPH_SYMMETRIC_STRUCTURE_REQUIRED (-1001):
 *          The method requires an undirected graph, or a directed graph with
 *          an adjacency matrix that is known to have a symmetric structure.
 *          LAGraph_Cached_IsSymmetricStructure can be used to determine this
 *          cached property.
 *    - LAGRAPH_IO_ERROR (-1002):
 *          A file input or output method failed, or an input file has an
 *          incorrect format that cannot be parsed.
 *    - LAGRAPH_NOT_CACHED (-1003):
 *          Some methods require one or more cached properties to be computed
 *          before calling them (AT, out_degree, or in_degree.  Use
 *          LAGraph_Cached_AT, LAGraph_Cached_OutDegree, and/or
 *          LAGraph_Cached_InDegree to compute them.
 *    - LAGRAPH_NO_SELF_EDGES_ALLOWED (-1004):
 *          Some methods requires that the graph have no self edges, which
 *          correspond to the entries on the diagonal of the adjacency matrix.
 *          If the G->nself_edges cached property is nonzero or unknown, this
 *          error condition is returned.  Use LAGraph_Cached_NSelfEdges to
 *          compute G->nself_edges, or LAGraph_DeleteSelfEdges to remove all
 *          diagonal entries from G->A.
 *    - LAGRAPH_CONVERGENCE_FAILURE (-1005):
 *          An iterative process failed to converge to a good solution.
 *    - LAGRAPH_CACHE_NOT_NEEDED (1000):
 *          This is a warning, not an error.  It is returned by
 *          LAGraph_Cached_* methods when asked to compute cached properties
 *          that are not needed.  These include G->AT and G->in_degree for an
 *          undirected graph.
 * }
 */
#define LAGRAPH_RETURN_VALUES

#define LAGRAPH_INVALID_GRAPH                   (-1000)
#define LAGRAPH_SYMMETRIC_STRUCTURE_REQUIRED    (-1001)
#define LAGRAPH_IO_ERROR                        (-1002)
#define LAGRAPH_NOT_CACHED                      (-1003)
#define LAGRAPH_NO_SELF_EDGES_ALLOWED           (-1004)
#define LAGRAPH_CONVERGENCE_FAILURE             (-1005)
#define LAGRAPH_CACHE_NOT_NEEDED                ( 1000)

/**
 * All LAGraph functions (except for @sphinxref{LAGraph_WallClockTime})
 * have a final msg parameter that is a pointer to a user-allocated string in
 * which an algorithm-specific error message can be returned.  If msg is NULL,
 * no error message is returned.  This is not itself an error condition, it
 * just indicates that the caller does not need the message returned.  If the
 * message string is provided but no error occurs, an empty string is returned.
 *
 * LAGRAPH_MSG_LEN is the minimum required length of a message string.
 *
 * For example, the following call computes the breadth-first-search of an
 * LAGraph_Graph G, starting at a given source node.  It returns a status of
 * zero if it succeeds and a negative value on failure.
 *
 *      GrB_Vector level, parent ;
 *      char msg [LAGRAPH_MSG_LEN] ;
 *      int status = LAGr_BreadthFirstSearch (&level, &parent, G, src, msg) ;
 *      if (status < 0)
 *      {
 *          printf ("status %d, error: %s\n", status, msg) ;
 *          ... take corrective action here ...
 *      }
 *
 * Error handling is simplified by the @sphinxref{LAGRAPH_TRY} / LAGRAPH_CATCH
 * mechanism described below.  For example, assuming the user application
 * #defines a single LAGRAPH_CATCH mechanism for all error handling, the
 * above example would become:
 *
 *      GrB_Vector level, parent ;
 *      char msg [LAGRAPH_MSG_LEN] ;
 *      #define LAGRAPH_CATCH(status)                           \
 *      {                                                       \
 *          printf ("status %d, error: %s\n", status, msg) ;    \
 *          ... take corrective action here ...                 \
 *      }
 *      ...
 *      LAGRAPH_TRY (LAGr_BreadthFirstSearch (&level, &parent, G, src, msg)) ;
 *
 * The advantage of the second use case is that the error-handling block of
 * code needs to be written only once.
 */
#define LAGRAPH_MSG_LEN 256

//------------------------------------------------------------------------------
// LAGRAPH_TRY: try an LAGraph method and check for errors
//------------------------------------------------------------------------------

/** LAGRAPH_TRY: try an LAGraph method and check for errors.
 *
 * In a robust application, the return values from each call to LAGraph and
 * GraphBLAS should be checked, and corrective action should be taken if an
 * error occurs.  The LAGRAPH_TRY and @sphinxref{GRB_TRY} macros assist in this
 * effort.
 *
 * LAGraph and GraphBLAS are written in C, and so they cannot rely on the
 * try/catch mechanism of C++.  To accomplish a similar goal, each LAGraph file
 * must `#define` its own file-specific macro called LAGRAPH_CATCH.  The typical
 * usage of macro is to free any temporary matrices/vectors or workspace when
 * an error occurs, and then "throw" the error by returning to the caller.  A
 * user application may also `#define LAGRAPH_CATCH` and use these macros.
 * The LAGRAPH_CATCH macro takes a single argument, which is the return value
 * from an LAGraph method.
 *
 * A typical example of a user function that calls LAGraph might #define
 * LAGRAPH_CATCH as follows.  Suppose workvector is a GrB_vector used for
 * computations internal to the mybfs function, and W is a (double *) space
 * created by malloc.
 *
 *      // an example user-defined LAGRAPH_CATCH macro, which prints the error
 *      // then frees any workspace or results, and returns to the caller:
 *      #define LAGRAPH_CATCH(status)                                   \
 *      {                                                               \
 *          printf ("LAGraph error: (%d): file: %s, line: %d\n%s\n",    \
 *              status, __FILE__, __LINE__, msg) ;                      \
 *          GrB_free (*parent) ;                                        \
 *          GrB_free (workvector) ;                                     \
 *          LAGraph_Free ((void **) &W, NULL) ;                         \
 *          return (status) ;                                           \
 *      }
 *
 *      // an example user function that uses LAGRAPH_TRY / LAGRAPH_CATCH
 *      int mybfs (LAGraph_Graph G, GrB_Vector *parent, int64_t src)
 *      {
 *          GrB_Vector workvector = NULL ;
 *          double *W = NULL ;
 *          char msg [LAGRAPH_MSG_LEN] ;
 *          (*parent) = NULL ;
 *          LAGRAPH_TRY (LAGr_BreadthFirstSearch (NULL, parent, G, src, true,
 *              msg)) ;
 *          // ...
 *          return (GrB_SUCCESS) ;
 *      }
 *
 * LAGRAPH_TRY is defined as follows:
 *
 *      #define LAGRAPH_TRY(LAGraph_method)             \
 *      {                                               \
 *          int LG_status = LAGraph_method ;            \
 *          if (LG_status < GrB_SUCCESS)                \
 *          {                                           \
 *              LAGRAPH_CATCH (LG_status) ;             \
 *          }                                           \
 *      }
 */

#define LAGRAPH_TRY(LAGraph_method)             \
{                                               \
    int LG_status = LAGraph_method ;            \
    if (LG_status < GrB_SUCCESS)                \
    {                                           \
        LAGRAPH_CATCH (LG_status) ;             \
    }                                           \
}

//------------------------------------------------------------------------------
// GRB_TRY: try a GraphBLAS method and check for errors
//------------------------------------------------------------------------------

/** GRB_TRY: LAGraph provides a similar functionality as @sphinxref{LAGRAPH_TRY}
 * for calling GraphBLAS methods, with the GRB_TRY macro.  GraphBLAS returns info
 * = 0 (GrB_SUCCESS) or 1 (GrB_NO_VALUE) on success, and a value < 0 on failure.
 * The user application must `#define GRB_CATCH` to use GRB_TRY.
 *
 * GraphBLAS and LAGraph both use the convention that negative values are
 * errors, and the LAGraph_status is a superset of the GrB_Info enum.  As a
 * result, the user can define LAGRAPH_CATCH and GRB_CATCH as the same
 * operation.  The main difference between the two would be the error message
 * string.  For LAGraph, the string is the last parameter, and LAGRAPH_CATCH
 * can optionally print it out.  For GraphBLAS, the GrB_error mechanism can
 * return a string.
 *
 * GRB_TRY is defined as follows:
 *
 *      #define GRB_TRY(GrB_method)                     \
 *      {                                               \
 *          GrB_Info LG_GrB_Info = GrB_method ;         \
 *          if (LG_GrB_Info < GrB_SUCCESS)              \
 *          {                                           \
 *              GRB_CATCH (LG_GrB_Info) ;               \
 *          }                                           \
 *      }
 *
 */

#define GRB_TRY(GrB_method)                     \
{                                               \
    GrB_Info LG_GrB_Info = GrB_method ;         \
    if (LG_GrB_Info < GrB_SUCCESS)              \
    {                                           \
        GRB_CATCH (LG_GrB_Info) ;               \
    }                                           \
}

//==============================================================================
// for C++ applications:
//==============================================================================

#if defined ( __cplusplus )
extern "C"
{
#endif

//==============================================================================
// LAGraph memory management
//==============================================================================

// LAGraph provides wrappers for the malloc/calloc/realloc/free set of memory
// management functions, initialized by LAGraph_Init or LAGr_Init.  By default,
// the following are pointers to the ANSI C11 malloc/calloc/realloc/free
// functions.

LAGRAPH_PUBLIC extern void * (* LAGraph_Malloc_function  ) (size_t)         ;
LAGRAPH_PUBLIC extern void * (* LAGraph_Calloc_function  ) (size_t, size_t) ;
LAGRAPH_PUBLIC extern void * (* LAGraph_Realloc_function ) (void *, size_t) ;
LAGRAPH_PUBLIC extern void   (* LAGraph_Free_function    ) (void *)         ;

//------------------------------------------------------------------------------
// LAGraph_Malloc:  allocate a block of memory (wrapper for malloc)
//------------------------------------------------------------------------------

/** LAGraph_Malloc: allocates a block of memory.  Uses the ANSI C11 malloc
 * function if LAGraph_Init was used, or the user_malloc_function passed to
 * LAGr_Init.
 *
 * @param[out] p        handle to allocated memory.
 * @param[in] nitems    number of items to allocate, each of size size_of_item.
 * @param[in] size_of_item  bytes allocted = nitems*size_of_item.
 * @param[in,out] msg   any error messages.
 *
 * @retval GrB_SUCCESS if successful.
 * @retval GrB_OUT_OF_MEMORY if out of memory.
 * @retval GrB_NULL_POINTER if p is NULL on input.
 */

LAGRAPH_PUBLIC
int LAGraph_Malloc
(
    // output:
    void **p,               // pointer to allocated block of memory
    // input:
    size_t nitems,          // number of items
    size_t size_of_item,    // size of each item
    char *msg
) ;

//------------------------------------------------------------------------------
// LAGraph_Calloc:  allocate a block of memory (wrapper for calloc)
//------------------------------------------------------------------------------

/** LAGraph_Calloc: allocates a block of memory and sets it to zero.  Uses the
 * ANSI C11 malloc and memset functions if LAGraph_Init was used or if NULL
 * was passed to LAGr_Init for the user_calloc_function, or the non-NULL
 * user_calloc_function passed to LAGr_Init otherwise.
 *
 * @param[out] p        handle to allocated memory.
 * @param[in] nitems    number of items to allocate, each of size size_of_item.
 * @param[in] size_of_item  bytes allocted = nitems*size_of_item.
 * @param[in,out] msg   any error messages.
 *
 * @retval GrB_SUCCESS if successful.
 * @retval GrB_OUT_OF_MEMORY if out of memory.
 * @retval GrB_NULL_POINTER if p is NULL on input.
 */

LAGRAPH_PUBLIC
int LAGraph_Calloc
(
    // output:
    void **p,               // pointer to allocated block of memory
    // input:
    size_t nitems,          // number of items
    size_t size_of_item,    // size of each item
    char *msg
) ;

//------------------------------------------------------------------------------
// LAGraph_Realloc: reallocate a block of memory (wrapper for realloc)
//------------------------------------------------------------------------------

/** LAGraph_Realloc: reallocates a block of memory.  Uses the ANSI C11 malloc,
 * memcpy, and free functions if LAGraph_Init was used or if NULL was passed to
 * LAGr_Init for the user_realloc_function, or the non-NULL
 * user_realloc_function passed to LAGr_Init otherwise.  Note that unlike the
 * ANSI C11 realloc function, this function requires the old size of block to
 * be provided on input (nitems_old).  This size must be exact; behavior is
 * undefined if the incorrect size is given.  The value nitems_old*size_of_item
 * must be the same as nitems*size_of_item when the block of memory was
 * allocated by LAGraph_Malloc or LAGraph_Calloc, or equal to
 * nitems_new*size_of_item from the last call to LAGraph_Realloc.
 *
 * @param[in,out] p     handle to allocated memory.
 * @param[in] nitems_new    new number of items to allocate.
 * @param[in] nitems_old    prior number of items allocated.
 * @param[in] size_of_item  size of each item.
 * @param[in,out] msg   any error messages.
 *
 * @retval GrB_SUCCESS if successful.
 * @retval GrB_OUT_OF_MEMORY if out of memory.
 * @retval GrB_NULL_POINTER if p is NULL on input.
 */

LAGRAPH_PUBLIC
int LAGraph_Realloc
(
    // input/output:
    void **p,               // old block to reallocate
    // input:
    size_t nitems_new,      // new number of items in the object
    size_t nitems_old,      // old number of items in the object
    size_t size_of_item,    // size of each item
    char *msg
) ;

//------------------------------------------------------------------------------
// LAGraph_Free:  free a block of memory (wrapper for free)
//------------------------------------------------------------------------------

/** LAGraph_Free: frees a block of memory.  The block must have been previously
 * allocated by LAGraph_Malloc or LAGraph_Calloc.  If LAGraph_Malloc (&p, ...)
 * is the pointer to the allocated block of memory, LAGraph_Free (&p, ...) is
 * the method to free it.  The parameter is passed as &p so that p can be set
 * to NULL on return, to guard against double-free.  LAGraph_Free does nothing
 * if &p or p are NULL on input; this is not an error.
 *
 * @param[in,out] p     handle to allocated memory.
 * @param[in,out] msg   any error messages.
 *
 * @retval GrB_SUCCESS in all cases.
 */

LAGRAPH_PUBLIC
int LAGraph_Free            // free a block of memory and set p to NULL
(
    // input/output:
    void **p,               // pointer to object to free, does nothing if NULL
    char *msg
) ;

//==============================================================================
// LAGraph data structures
//==============================================================================

// In addition to relying on underlying GraphBLAS objects (GrB_Matrix,
// GrB_Vector, GrB_Descriptor, ...), LAGraph adds the LAGraph_Graph.  This
// object contains a representation of a graph and its associated data.  Unlike
// the GrB_* objects, the LAGraph_Graph is not opaque.

// LAGRAPH_UNKNOWN is used for all scalars whose value is not known
#define LAGRAPH_UNKNOWN (-1)

//------------------------------------------------------------------------------
// LAGraph_Kind: the kind of a graph
//------------------------------------------------------------------------------

/** LAGraph_Kind: the kind of a graph.  Currently, only two types of graphs are
 * supported: undirected graphs and directed graphs.  Edge weights are assumed
 * to be present.  Unweighted graphs can be represented by setting all entries
 * present in the sparsity structure to the same value, typically 1.
 * Additional types of graphs will be added in the future.
 */

typedef enum
{
    LAGraph_ADJACENCY_UNDIRECTED = 0, ///< undirected graph.
        ///< G->A is square and symmetric; both upper and lower triangular parts
        ///< are present.  A(i,j) is the edge (i,j).  Results are undefined if
        ///< A is unsymmetric.

    LAGraph_ADJACENCY_DIRECTED = 1,   ///< directed graph.
        ///< G->A is square; A(i,j) is the edge (i,j).

    // possible future kinds of graphs:
    // LAGraph_ADJACENCY_UNDIRECTED_UNWEIGHTED
    // LAGraph_ADJACENCY_DIRECTED_UNWEIGHTED
    // LAGraph_ADJACENCY_UNDIRECTED_TRIL
    // LAGraph_ADJACENCY_UNDIRECTED_TRIU
    // LAGraph_BIPARTITE
    // LAGraph_BIPARTITE_DIRECTED
    // LAGraph_BIPARTITE_UNDIRECTED
    // LAGraph_INCIDENCE_*
    // LAGraph_MULTIGRAPH_*
    // LAGraph_HYPERGRAPH
    // LAGraph_HYPERGRAPH_DIRECTED
    // ...

    LAGraph_KIND_UNKNOWN = LAGRAPH_UNKNOWN  ///< unknown kind of graph (-1).
}
LAGraph_Kind ;

//------------------------------------------------------------------------------
// LAGraph_Boolean: true, false, or unknown
//------------------------------------------------------------------------------

/** LAGraph_Boolean: a boolean value (true or false) or unknown (-1).
 */

typedef enum
{
    LAGraph_FALSE = 0,      ///< the Boolean value is known to be false.
    LAGraph_TRUE = 1,       ///< the Boolean value is known to be true.
    LAGraph_BOOLEAN_UNKNOWN = LAGRAPH_UNKNOWN   ///< Boolean value is unknown.
}
LAGraph_Boolean ;

//------------------------------------------------------------------------------
// LAGraph_State: value, bound, or unknown
//------------------------------------------------------------------------------

/** LAGraph_State describes the status of a cached property of a graph.  If the
 * cached property is computed in floating-point arithmetic, it may have been
 * computed with roundoff error, but it may still be declared as "value" if the
 * roundoff error is expected to be small, or if the cached property was
 * computed as carefully as possible (to within reasonable roundoff error).
 * The "bound" state indicates that the cached property is an upper or lower
 * bound, depending on the particular cached property.  If computed in
 * floating-point arithmetic, an "upper bound" cached property may be actually
 * slightly lower than the actual upper bound, because of floating-point
 * roundoff.
 */

typedef enum
{
    LAGraph_VALUE = 0,  ///< cached property is a known value.
    LAGraph_BOUND = 1,  ///< cached property is a bound.
        ///< The bound is upper or lower, depending on the particular cached
        ///< property.
    LAGraph_STATE_UNKNOWN = LAGRAPH_UNKNOWN,    ///< the property is unknown.
}
LAGraph_State ;

//------------------------------------------------------------------------------
// LAGraph_Graph: the primary graph data structure of LAGraph
//------------------------------------------------------------------------------

/** LAGraph_Graph: a representation of a graph.
 *
 * The LAGraph_Graph G contains a GrB_Matrix G->A as its primary component.
 * For graphs represented with adjacency matrices, A(i,j) denotes the edge
 * (i,j).  Unlike GrB_* objects in GraphBLAS, the LAGraph_Graph data structure
 * is not opaque.  User applications have full access to its contents.
 *
 * An LAGraph_Graph G contains two kinds of components:
 *  1. Primary components of the graph, which fully define the graph.
 *  2. Cached properties of the graph, which can be recreated any time.
 */

// (1) primary components:
//      A           the adjacency matrix of the graph
//      kind        the kind of graph (undirected, directed, bipartite, ...)
// (2) cached properties:
//      AT          AT = A'
//      out_degree  out_degree(i) = # of entries in A(i,:)
//      in_degree   in_degree(j) = # of entries in A(:,j)
//      is_symmetric_structure: true if the structure of A is symmetric
//      nself_edges the number of entries on the diagonal of A
//      emin        minimum edge weight
//      emax        maximum edge weight

struct LAGraph_Graph_struct
{

    //--------------------------------------------------------------------------
    // primary components of the graph
    //--------------------------------------------------------------------------

    /** @name Primary Components */
    //@{

    GrB_Matrix  A ;         ///< the adjacency matrix of the graph
    LAGraph_Kind kind ;     ///< the kind of graph

    //@}

    // possible future components:
    // multigraph ..
    // GrB_Matrix *Amult ; // array of size nmatrices
    // int nmatrices ;
    // GrB_Vector NodeWeights ;

    //--------------------------------------------------------------------------
    // cached properties of the graph
    //--------------------------------------------------------------------------

    /** @name Cached Properties
     *
     * All of these components may be deleted or set to 'unknown' at any time.
     * For example, if AT is NULL, then the transpose of A has not been
     * computed.  A scalar cached property of type LAGraph_Boolean would be set
     * to LAGRAPH_UNKNOWN to denote that its value is unknown.
     *
     * If present, the cached properties must be valid and accurate.  If the
     * graph changes, these cached properties can either be recomputed or
     * deleted to denote the fact that they are unknown.  This choice is up to
     * individual LAGraph methods and utilities.
     *
     * LAGraph methods can set non-scalar cached properties only if they are
     * constructing the graph.  They cannot modify them or create them if the
     * graph is declared as a read-only object in the parameter list of the
     * method.
     */

    //@{

    GrB_Matrix AT ; ///< AT = A', the transpose of A, with the same type.

    GrB_Vector out_degree ; ///< a GrB_INT64 vector of length m, if A is m-by-n.
           ///< where out_degree(i) is the number of entries in A(i,:).  If
           ///< out_degree is sparse and the entry out_degree(i) is not present,
           ///< then it is assumed to be zero.

    GrB_Vector in_degree ;  ///< a GrB_INT64 vector of length n, if A is m-by-n.
            ///< where in_degree(j) is the number of entries in A(:,j).  If
            ///< in_degree is sparse and the entry in_degree(j) is not present,
            ///< then it is assumed to be zero.  If A is known to have a
            ///< symmetric structure, the convention is that the degree is held
            ///< in G->out_degree, and in G->in_degree is left as NULL.

    // FUTURE: If G is held as an incidence matrix, then G->A might be
    // rectangular, in the future, but the graph G may have a symmetric
    // structure anyway.

    LAGraph_Boolean is_symmetric_structure ;    ///< For an undirected
            ///< graph, this cached property will always be implicitly true and
            ///< can be ignored.  The matrix A for a directed weighted graph
            ///< will typically be unsymmetric, but might have a symmetric
            ///< structure.  In that case, this scalar cached property can be
            ///< set to true. By default, this cached property is set to
            ///< LAGRAPH_UNKNOWN.

    int64_t nself_edges ; ///< number of entries on the diagonal of A, or
            ///< LAGRAPH_UNKNOWN if unknown.  For the adjacency matrix of a
            ///< directed or undirected graph, this is the number of self-edges
            ///< in the graph.

    GrB_Scalar emin ;   ///< minimum edge weight: value, lower bound, or unknown
    LAGraph_State emin_state ;
            ///< - VALUE: emin is equal to the smallest entry, min(G->A)
            ///< - BOUND: emin <= min(G->A)
            ///< - UNKNOWN: emin is unknown

    GrB_Scalar emax ;   ///< maximum edge weight: value, upper bound, or unknown
    LAGraph_State emax_state ;
            ///< - VALUE: emax is equal to the largest entry, max(G->A)
            ///< - BOUND: emax >= max(G->A)
            ///< - UNKNOWN: emax is unknown

    //@}

    // FUTURE: possible future cached properties:
    // Some algorithms may want to know if the graph has any edge weights
    // exactly equal to zero.  In some cases, this can be inferred from the
    // emin/emax bounds, or it can be indicated via the following cached
    // property:
    // LAGraph_Boolean nonzero ;  // If true, then all entries in
            // G->A are known to be nonzero.  If false, G->A may contain
            // entries in its structure that are identically equal to zero.  If
            // unknown, then G->A may or may not have entries equal to zero.
    // other edge weight metrics: median, standard deviation....  Might be
    // useful for computing Delta for a Basic SSSP.
    // GrB_Vector row_sum, col_sum ;
    // row_sum(i) = sum(abs(A(i,:))), regardless of kind
    // col_sum(j) = sum(abs(A(:,j))), regardless of kind
    // LAGraph_Boolean connected ;   // true if G is a connected graph
} ;

typedef struct LAGraph_Graph_struct *LAGraph_Graph ;

//==============================================================================
// LAGraph utilities
//==============================================================================

//------------------------------------------------------------------------------
// LAGraph_Init: start GraphBLAS and LAGraph
//------------------------------------------------------------------------------

/** LAGraph_Init: initializes GraphBLAS and LAGraph.  This method must be
 * called before calling any other GrB* or LAGraph* method.  It initializes
 * GraphBLAS with GrB_init and then performs LAGraph-specific initializations.
 * In particular, the LAGraph semirings listed below are created.  GrB_init can
 * also safely be called before calling @sphinxref{LAGr_Init} or LAGraph_Init.
 *
 * @param[in,out] msg   any error messages.
 *
 * @retval GrB_SUCCESS if successful.
 * @retval GrB_INVALID_VALUE if LAGraph_Init or LAGr_Init has already been
 *      called by the user application.
 * @returns any GraphBLAS errors that may have been encountered.
 */

LAGRAPH_PUBLIC
int LAGraph_Init
(
    char *msg
) ;

// FUTURE: include these as built-in semirings in v2.1 C API, Table 3.9:

// LAGraph semirings, created by LAGraph_Init or LAGr_Init:
LAGRAPH_PUBLIC extern GrB_Semiring

    // LAGraph_plus_first_T: using the GrB_PLUS_MONOID_T monoid and the
    // corresponding GrB_FIRST_T multiplicative operator.
    LAGraph_plus_first_int8   ,
    LAGraph_plus_first_int16  ,
    LAGraph_plus_first_int32  ,
    LAGraph_plus_first_int64  ,
    LAGraph_plus_first_uint8  ,
    LAGraph_plus_first_uint16 ,
    LAGraph_plus_first_uint32 ,
    LAGraph_plus_first_uint64 ,
    LAGraph_plus_first_fp32   ,
    LAGraph_plus_first_fp64   ,

    // LAGraph_plus_second_T: using the GrB_PLUS_MONOID_T monoid and the
    // corresponding GrB_SECOND_T multiplicative operator.
    LAGraph_plus_second_int8   ,
    LAGraph_plus_second_int16  ,
    LAGraph_plus_second_int32  ,
    LAGraph_plus_second_int64  ,
    LAGraph_plus_second_uint8  ,
    LAGraph_plus_second_uint16 ,
    LAGraph_plus_second_uint32 ,
    LAGraph_plus_second_uint64 ,
    LAGraph_plus_second_fp32   ,
    LAGraph_plus_second_fp64   ,

    // LAGraph_plus_one_T: using the GrB_PLUS_MONOID_T monoid and the
    // corresponding GrB_ONEB_T multiplicative operator.
    LAGraph_plus_one_int8   ,
    LAGraph_plus_one_int16  ,
    LAGraph_plus_one_int32  ,
    LAGraph_plus_one_int64  ,
    LAGraph_plus_one_uint8  ,
    LAGraph_plus_one_uint16 ,
    LAGraph_plus_one_uint32 ,
    LAGraph_plus_one_uint64 ,
    LAGraph_plus_one_fp32   ,
    LAGraph_plus_one_fp64   ,

    // LAGraph_any_one_T: using the GrB_MIN_MONOID_T for non-boolean types or
    // GrB_LOR_MONOID_BOOL for boolean, and the GrB_ONEB_T multiplicative op.
    // These semirings are very useful for unweighted graphs, or for algorithms
    // that operate only on the sparsity structure of unweighted graphs.
    LAGraph_any_one_bool   ,    // (or, true) semiring
    LAGraph_any_one_int8   ,    // (min, 1) semiring
    LAGraph_any_one_int16  ,
    LAGraph_any_one_int32  ,
    LAGraph_any_one_int64  ,
    LAGraph_any_one_uint8  ,
    LAGraph_any_one_uint16 ,
    LAGraph_any_one_uint32 ,
    LAGraph_any_one_uint64 ,
    LAGraph_any_one_fp32   ,
    LAGraph_any_one_fp64   ;

//------------------------------------------------------------------------------
// LAGraph_Version: determine the version of LAGraph
//------------------------------------------------------------------------------

/** LAGraph_Version: determines the version of LAGraph.  The version number and
 * date can also be obtained via compile-time constants from LAGraph.h.
 * However, it is possible to compile a user application that #includes one
 * version of LAGraph.h and then links with another version of the LAGraph
 * library later on, so the version number and date may differ from the
 * compile-time constants.
 *
 * The LAGraph_Version method allows the library itself to be queried, after it
 * is linked in with the user application.
 *
 * The version_number array is set to LAGRAPH_VERSION_MAJOR,
 * LAGRAPH_VERSION_MINOR, and LAGRAPH_VERSION_UPDATE, in that order.  The
 * LAGRAPH_DATE string is copied into the user-provided version_date string,
 * and is null-terminated.
 *
 * @param[out] version_number   an array of size 3; with the major, minor,
 *      and update versions of LAGraph, in that order.
 * @param[out] version_date     an array of size >= LAGraph_MSG_LEN, returned
 *      with the date of this version of LAGraph.
 * @param[in,out] msg   any error messages.
 *
 * @retval GrB_SUCCESS if successful.
 * @retval GrB_NULL_POINTER if version_number or version_date are NULL.
 */

LAGRAPH_PUBLIC
int LAGraph_Version
(
    // output:
    int version_number [3], // user-provided array of size 3
    char *version_date,     // user-provided array of size >= LAGRAPH_MSG_LEN
    char *msg
) ;

//------------------------------------------------------------------------------
// LAGraph_Finalize: finish LAGraph
//------------------------------------------------------------------------------

/** LAGraph_Finalize: finish LAGraph and GraphBLAS.  Must be called as the last
 * LAGraph method.  It calls GrB_finalize and frees any LAGraph objects created
 * by @sphinxref{LAGraph_Init} or @sphinxref{LAGr_Init}.  After calling this
 * method, no LAGraph or GraphBLAS methods may be used.
 *
 * @param[in,out] msg   any error messages.
 *
 * @retval GrB_SUCCESS if successful.
 * @returns any GraphBLAS errors that may have been encountered.
 */

LAGRAPH_PUBLIC
int LAGraph_Finalize
(
    char *msg
) ;

//------------------------------------------------------------------------------
// LAGraph_New: create a new graph
//------------------------------------------------------------------------------

/** LAGraph_New: creates a new graph G.  The cached properties G->AT,
 * G->out_degree, and G->in_degree are set to NULL, and scalar cached
 * properties are set to LAGRAPH_UNKNOWN.
 *
 * @param[out] G        handle to the newly created graph, as &G.
 * @param[in,out] A     adjacency matrix.  A is moved into G as G->A, and A
 *                      itself is set to NULL to denote that is now a part of
 *                      G.  That is, { G->A = A ; A = NULL ; } is performed.
 *                      When G is deleted, G->A is freed.  If A is NULL, the
 *                      graph is invalid until G->A is set.
 * @param[in] kind      the kind of graph to create.  This may be
 *                      LAGRAPH_UNKNOWN, which must then be revised later
 *                      before the graph can be used.
 * @param[in,out] msg   any error messages.
 *
 * @retval GrB_SUCCESS if successful.
 * @retval GrB_NULL_POINTER if G is null.
 * @returns any GraphBLAS errors that may have been encountered.
 */

LAGRAPH_PUBLIC
int LAGraph_New
(
    // output:
    LAGraph_Graph *G,   // the graph to create, NULL if failure
    // input/output:
    GrB_Matrix    *A,   // the adjacency matrix of the graph, may be NULL.
    // input:
    LAGraph_Kind kind,  // the kind of graph.
    char *msg
) ;

//------------------------------------------------------------------------------
// LAGraph_Delete: free a graph and all its contents
//------------------------------------------------------------------------------

/** LAGraph_Delete: frees a graph G.  The adjacency matrix G->A and the cached
 * properties G->AT, G->out_degree, and G->in_degree are all freed.
 *
 * @param[in,out] G     handle to the graph to be free. *G is NULL on output.
 *                      To keep G->A while deleting the graph G, use:
 *                      { A = G->A ; G->A = NULL ; LAGraph_Delete (&G, msg) ; }
 * @param[in,out] msg   any error messages.
 *
 * @retval GrB_SUCCESS if successful.
 * @returns any GraphBLAS errors that may have been encountered.
 */

LAGRAPH_PUBLIC
int LAGraph_Delete
(
    // input/output:
    LAGraph_Graph *G,   // the graph to delete; G set to NULL on output.
    char *msg
) ;

//------------------------------------------------------------------------------
// LAGraph_DeleteCached: free any internal cached properties of a graph
//------------------------------------------------------------------------------

/** LAGraph_DeleteCached: frees all cached properies of a graph G.  The graph is
 * still valid.  This method should be used if G->A changes, since such changes
 * will normally invalidate G->AT, G->out_degree, and/or G->in_degree.
 *
 * @param[in,out] G     handle to the graph to modified.  The graph G remains
 *                      valid on output, but with all cached properties freed.
 *                      G may be NULL, in which case nothing is done.
 * @param[in,out] msg   any error messages.
 *
 * @retval GrB_SUCCESS if successful.
 * @returns any GraphBLAS errors that may have been encountered.
 */

LAGRAPH_PUBLIC
int LAGraph_DeleteCached
(
    // input/output:
    LAGraph_Graph G,    // G stays valid, only cached properties are freed
    char *msg
) ;

//------------------------------------------------------------------------------
// LAGraph_Cached_AT: construct G->AT for a graph
//------------------------------------------------------------------------------

/** LAGraph_Cached_AT: constructs G->AT, the transpose of G->A.  This matrix is
 * required by some of the algorithms.  Basic algorithms may construct G->AT if
 * they require it.  The matrix G->AT is then available for subsequent use.  If
 * G->A changes, G->AT should be freed and recomputed.  If G->AT already
 * exists, it is left unchanged (even if it is not equal to the transpose of
 * G->A).  As a result, if G->A changes, G->AT should be explictly freed.
 *
 * @param[in,out] G     graph for which G->AT is computed.
 * @param[in,out] msg   any error messages.
 *
 * @retval GrB_SUCCESS if successful.
 * @retval GrB_NULL_POINTER if G is NULL.
 * @retval LAGRAPH_CACHE_NOT_NEEDED if G->kind is LAGraph_ADJACENCY_UNDIRECTED.
 * @retval LAGRAPH_INVALID_GRAPH if G is invalid (G->A missing, or G->kind
 *      not a recognized kind).
 * @returns any GraphBLAS errors that may have been encountered.
 */

LAGRAPH_PUBLIC
int LAGraph_Cached_AT
(
    // input/output:
    LAGraph_Graph G,    // graph for which to compute G->AT
    char *msg
) ;

//------------------------------------------------------------------------------
// LAGraph_Cached_IsSymmetricStructure: determine G->is_symmetric_structure
//------------------------------------------------------------------------------

/** LAGraph_Cached_IsSymmetricStructure: determines if the sparsity structure of
 * G->A is symmetric (ignoring its values).  If G->kind denotes that the graph
 * is undirected, this cached property is implicitly true (and not checked).
 * Otherwise, this method determines if the structure of G->A for a directed
 * graph G has a symmetric sparsity structure.  No work is performed if the
 * cached property is already known.
 *
 * @param[in,out] G     graph for which G->is_symmetric_structure is computed.
 * @param[in,out] msg   any error messages.
 *
 * @retval GrB_SUCCESS if successful.
 * @retval GrB_NULL_POINTER if G is NULL.
 * @retval LAGRAPH_INVALID_GRAPH if G is invalid (G->A missing, or G->kind
 *      not a recognized kind).
 * @returns any GraphBLAS errors that may have been encountered.
 */

LAGRAPH_PUBLIC
int LAGraph_Cached_IsSymmetricStructure
(
    // input/output:
    LAGraph_Graph G,    // graph to determine the symmetry of structure of A
    char *msg
) ;

//------------------------------------------------------------------------------
// LAGraph_Cached_OutDegree: determine G->out_degree
//------------------------------------------------------------------------------

/** LAGraph_Cached_OutDegree: computes G->out_degree.  No work is performed if
 * it already exists in G.
 *
 * @param[in,out] G     graph for which G->out_degree is computed.
 * @param[in,out] msg   any error messages.
 *
 * @retval GrB_SUCCESS if successful.
 * @retval GrB_NULL_POINTER if G is NULL.
 * @retval LAGRAPH_INVALID_GRAPH if G is invalid (G->A missing, or G->kind
 *      not a recognized kind).
 * @returns any GraphBLAS errors that may have been encountered.
 */

LAGRAPH_PUBLIC
int LAGraph_Cached_OutDegree
(
    // input/output:
    LAGraph_Graph G,    // graph to determine G->out_degree
    char *msg
) ;

//------------------------------------------------------------------------------
// LAGraph_Cached_InDegree: determine G->in_degree
//------------------------------------------------------------------------------

/** LAGraph_Cached_InDegree computes G->in_degree.  No work is performed if it
 * already exists in G.  If G is undirected, G->in_degree is never computed and
 * remains NULL (the method returns LAGRAPH_CACHE_NOT_NEEDED).  No work is
 * performed if it is already exists in G.
 *
 * Performance note: for SuiteSparse:GraphBLAS, if G->A is held by row (the
 * default format), then computing G->in_degree is fastest if G->AT is known.
 * If G->AT will be needed anyway, compute it first with LAGraph_Cached_AT, and
 * then call LAGraph_Cached_Indegree.  This is optional; if G->AT is not known,
 * then G->in_degree is computed from G->A instead.
 *
 * @param[in,out] G     graph for which G->in_degree is computed.
 * @param[in,out] msg   any error messages.
 *
 * @retval GrB_SUCCESS if successful.
 * @retval GrB_NULL_POINTER if G is NULL.
 * @retval LAGRAPH_CACHE_NOT_NEEDED if G->kind is LAGraph_ADJACENCY_UNDIRECTED.
 * @retval LAGRAPH_INVALID_GRAPH if G is invalid (G->A missing, or G->kind
 *      not a recognized kind).
 * @returns any GraphBLAS errors that may have been encountered.
 */

LAGRAPH_PUBLIC
int LAGraph_Cached_InDegree
(
    // input/output:
    LAGraph_Graph G,    // graph to determine G->in_degree
    char *msg
) ;

//------------------------------------------------------------------------------
// LAGraph_Cached_NSelfEdges: determine G->nself_edges
//------------------------------------------------------------------------------

/** LAGraph_Cached_NSelfEdges: computes G->nself_edges, the number of diagonal
 * entries that appear in the G->A matrix.  For an undirected or directed graph
 * with an adjacency matrix G->A, these are the number of self-edges in G.  No
 * work is performed it is already computed.
 *
 * @param[in,out] G     graph for which G->nself_edges is computed.
 * @param[in,out] msg   any error messages.
 *
 * @retval GrB_SUCCESS if successful.
 * @retval GrB_NULL_POINTER if G is NULL.
 * @retval LAGRAPH_INVALID_GRAPH if G is invalid (G->A missing, or G->kind
 *      not a recognized kind).
 * @returns any GraphBLAS errors that may have been encountered.
 */

LAGRAPH_PUBLIC
int LAGraph_Cached_NSelfEdges
(
    // input/output:
    LAGraph_Graph G,    // graph to compute G->nself_edges
    char *msg
) ;

//------------------------------------------------------------------------------
// LAGraph_Cached_EMin: determine G->emin
//------------------------------------------------------------------------------

/** LAGraph_Cached_EMin: computes the G->emin, the smallest entry in G->A.
 * Not computed if G->emin already exists.
 *
 * @param[in,out] G     graph for which G->emin is computed.
 * @param[in,out] msg   any error messages.
 *
 * @retval GrB_SUCCESS if successful.
 * @retval GrB_NOT_IMPLEMENTED if G does not have a built-in real type:
 *      GrB_(BOOL, INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64,
 *      FP32, OR FP64).
 * @retval GrB_NULL_POINTER if G is NULL.
 * @retval LAGRAPH_INVALID_GRAPH if G is invalid (G->A missing, or G->kind
 *      not a recognized kind).
 * @returns any GraphBLAS errors that may have been encountered.
 */

LAGRAPH_PUBLIC
int LAGraph_Cached_EMin
(
    // input/output:
    LAGraph_Graph G,    // graph to determine G->emin
    char *msg
) ;

//------------------------------------------------------------------------------
// LAGraph_Cached_EMax: determine G->emax
//------------------------------------------------------------------------------

/** LAGraph_Cached_EMax: computes the G->emax, the largest entry in G->A.
 * Not computed if G->emax already exists.
 *
 * @param[in,out] G     graph for which G->emax is computed.
 * @param[in,out] msg   any error messages.
 *
 * @retval GrB_SUCCESS if successful.
 * @retval GrB_NOT_IMPLEMENTED if G does not have a built-in real type:
 *      GrB_(BOOL, INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64,
 *      FP32, OR FP64).
 * @retval GrB_NULL_POINTER if G is NULL.
 * @retval LAGRAPH_INVALID_GRAPH if G is invalid (G->A missing, or G->kind
 *      not a recognized kind).
 * @returns any GraphBLAS errors that may have been encountered.
 */

LAGRAPH_PUBLIC
int LAGraph_Cached_EMax
(
    // input/output:
    LAGraph_Graph G,    // graph to determine G->emax
    char *msg
) ;

//------------------------------------------------------------------------------
// LAGraph_DeleteSelfEdges: remove all diagonal entries from G->A
//------------------------------------------------------------------------------

/** LAGraph_DeleteSelfEdges: removes any diagonal entries from G->A.  Most
 * cached properties are cleared or set to LAGRAPH_UNKNOWN.  G->nself_edges is
 * set to zero, and G->is_symmetric_structure is left unchanged.
 *
 * @param[in,out] G     graph for which G->A is modified.
 * @param[in,out] msg   any error messages.
 *
 * @retval GrB_SUCCESS if successful.
 * @retval GrB_NULL_POINTER if G is NULL.
 * @retval LAGRAPH_INVALID_GRAPH if G is invalid (G->A missing, or G->kind
 *      not a recognized kind).
 * @returns any GraphBLAS errors that may have been encountered.
 */

LAGRAPH_PUBLIC
int LAGraph_DeleteSelfEdges
(
    // input/output:
    LAGraph_Graph G,    // diagonal entries removed, most cached properties
                        // cleared
    char *msg
) ;

//------------------------------------------------------------------------------
// LAGraph_CheckGraph: determine if a graph is valid
//------------------------------------------------------------------------------

/** LAGraph_CheckGraph: determines if a graph is valid.  Only basic checks are
 * performed on the cached properties, taking O(1) time.
 *
 * @param[in] G         graph to check.
 * @param[in,out] msg   any error messages.
 *
 * @retval GrB_SUCCESS if successful.
 * @retval GrB_NULL_POINTER if G is NULL.
 * @retval LAGRAPH_INVALID_GRAPH if G is invalid: G->A missing, G->kind
 *      not a recognized kind, G->AT present but has the wrong dimensions or
 *      its type does not match G->A, G->in_degree/out_degree present but
 *      with the wrong dimension or type (in/out_degree must be GrB_INT64).
 * @returns any GraphBLAS errors that may have been encountered.
 */

LAGRAPH_PUBLIC
int LAGraph_CheckGraph
(
    // input/output:
    LAGraph_Graph G,    // graph to check
    char *msg
) ;

//------------------------------------------------------------------------------
// LAGraph_GetNumThreads: determine # of OpenMP threads to use
//------------------------------------------------------------------------------

/** LAGraph_GetNumThreads determines the current number of OpenMP threads that
 * can be used.  See LAGraph_SetNumThreads for a description of nthreads_outer
 * and nthreads_inner.
 *
 * @param[out] nthreads_outer   number of threads for outer region.
 * @param[out] nthreads_inner   number of threads for inner region,
 *                              or for the underlying GraphBLAS library.
 * @param[in,out] msg           any error messages.
 *
 * @retval GrB_SUCCESS if successful.
 * @retval GrB_NULL_POINTER if nthreads_outer or nthreads_inner are NULL.
 */

LAGRAPH_PUBLIC
int LAGraph_GetNumThreads
(
    // output:
    int *nthreads_outer,    // for outer region for nested parallelism
    int *nthreads_inner,    // for inner region of nested parallelism, or for
                            // the underlying GraphBLAS library
    char *msg
) ;

//------------------------------------------------------------------------------
// LAGraph_SetNumThreads: set # of OpenMP threads to use
//------------------------------------------------------------------------------

/** LAGraph_SetNumThreads sets the current number of OpenMP threads that can be
 * used by LAGraph and GraphBLAS.  Two thread counts can be controlled:
 *
 * @param[in] nthreads_outer
 *    number of threads to be used in outer regions of a
 *    nested parallel construct assuming that nthreads_inner is used in the
 *    inner region.  The total number of threads used for an entire nested
 *    region in LAGraph is given by nthreads_outer*nthreads_inner.  This
 *    product is also the # of threads that a flat parallel region in LAGraph
 *    may use.
 * @param[in] nthreads_inner
 *    number of threads to be used in an inner region of a
 *    nested parallel construct, or for the # of threads to be used in each
 *    call to the underlying GraphBLAS library.
 * @param[in,out] msg           any error messages.
 *
 * @retval GrB_SUCCESS if successful.
 * @returns any GraphBLAS errors that may have been encountered.
 */

LAGRAPH_PUBLIC
int LAGraph_SetNumThreads
(
    // input:
    int nthreads_outer,
    int nthreads_inner,
    char *msg
) ;

//------------------------------------------------------------------------------
// LAGraph_WallClockTime: return the current wall clock time
//------------------------------------------------------------------------------

/** LAGraph_WallClockTime returns the current wall clock time.  Normally, this
 * is simply a wrapper for omp_get_wtime, if OpenMP is in use.  Otherwise, an
 * OS-specific timing function is called.  Note that unlike all other LAGraph
 * functions, this function does not return an error condition, nor does it
 * have a msg string parameter.  Instead, it returns the current wall clock
 * time (in seconds) since some fixed point in the past.
 *
 * @returns the current wall clock time.
 */

LAGRAPH_PUBLIC
double LAGraph_WallClockTime     // returns omp_get_wtime(), or other timer
(
    void
) ;

//------------------------------------------------------------------------------
// LAGraph_MMRead: read a matrix in MatrixMarket format
//------------------------------------------------------------------------------

/** LAGraph_MMRead: reads a matrix in MatrixMarket format.
 * The file format used here is compatible with all variations of the Matrix
 * Market "coordinate" and "array" format (http://www.nist.gov/MatrixMarket),
 * for sparse and dense matrices respectively.  The format is fully described
 * in <a href="https://github.com/GraphBLAS/LAGraph/blob/stable/papers/MatrixMarket.pdf">
 * LAGraph/Doc/MatrixMarket.pdf</a>, and summarized here (with extensions for LAGraph).
 *
 * \rst_star{
 * First Line
 * ----------
 *
 * The first line of the file starts with ``%%MatrixMarket``, with the following
 * format::
 *
 *      %%MatrixMarket matrix <fmt> <type> <storage>
 *
 * <fmt>
 *      One of:
 *
 *      - coordinate : sparse matrix in triplet form
 *      - array : dense matrix in column-major form
 *
 *      Both formats are returned as a GrB_Matrix.
 *
 *      If not present, defaults to coordinate.
 *
 * <type>
 *      One of:
 *
 *      - real : returns as GrB_FP64
 *      - integer : returns as GrB_INT64
 *      - pattern : returns as GrB_BOOL
 *      - complex : *currently not supported*
 *
 *      The return type can be modified by the ``%%GraphBLAS``
 *      structured comment described below.
 *
 *      If not present, defaults to real.
 *
 * <storage>
 *      One of:
 *
 *      - general
 *            the matrix has no symmetry properties (or at least none that
 *            were exploited when the file was created).
 *      - Hermitian
 *            square complex matrix with A(i,j) = conj (A(j,i)).  All
 *            entries on the diagonal are real.  Each off-diagonal entry in the file
 *            creates two entries in the GrB_Matrix that is returned.
 *      - symmetric
 *            A(i,j) == A(j,i).  Only entries on or below the diagonal
 *            appear in the file.  Each off-diagonal entry in the file creates two entries
 *            in the GrB_Matrix that is returned.
 *      - skew-symmetric
 *            A(i,j) == -A(i,j).  There are no entries on the
 *            diagonal.  Only entries below the diagonal appear in the file.  Each
 *            off-diagonal entry in the file creates two entries in the GrB_Matrix that is
 *            returned.
 *
 *      The Matrix Market format is case-insensitive, so "hermitian" and
 *      "Hermitian" are treated the same.
 *
 *      If not present, defaults to general.
 *
 * Not all combinations are permitted.  Only the following are meaningful:
 *
 * (1) (coordinate or array) x (real, integer, or complex) x (general, symmetric, or skew-symmetric)
 * (2) (coordinate or array) x (complex) x (Hermitian)
 * (3) (coodinate) x (pattern) x (general or symmetric)
 *
 * Second Line
 * -----------
 *
 * The second line is an optional extension to the Matrix Market format::
 *
 *      %%GraphBLAS type <entrytype>
 *
 * <entrytype>
 *      One of the 11 built-in types (bool, int8_t, int16_t,
 *      int32_t, int64_t, uint8_t, uint16_t, uint32_t, uint64_t, float, or
 *      double.
 *
 * If this second line is included, it overrides the default GraphBLAS
 * types for the Matrix Market <type> on line one of the file: real,
 * pattern, and integer.  The Matrix Market complex <type> is not yet
 * supported.
 *
 * Other Lines
 * -----------
 *
 * Any other lines starting with "%" are treated as comments, and are ignored.
 * Comments may be interspersed throughout the file.  Blank lines are ignored.
 * The Matrix Market header is optional in this routine (it is not optional in
 * the Matrix Market format).  The remaining lines are space delimited,
 * and free format (one or more spaces can
 * appear, and each field has arbitrary width).
 *
 * Coordinate Format
 * -----------------
 * For coordinate format, the first non-comment line must appear,
 * and it must contain three integers::
 *
 *      nrows ncols nvals
 *
 * For example, a 5-by-12 matrix with 42 entries would have::
 *
 *      5 12 42
 *
 * Each of the remaining lines defines one entry.  The order is arbitrary.  If
 * the Matrix Market <type> is real or integer, each line contains three
 * numbers: row index, column index, and value.  For example, if A(3,4) is
 * equal to 5.77, a line::
 *
 *      3 4 5.77
 *
 * would appear in the file.  The indices in the Matrix Market are 1-based, so
 * this entry becomes A(2,3) in the GrB_Matrix returned to the caller.  If the
 * <type> is pattern, then only the row and column index appears.  If <type> is
 * complex, four values appear.  If A(8,4) has a real part of 6.2 and an
 * imaginary part of -9.3, then the line is::
 *
 *      8 4 6.2 -9.3
 *
 * and since the file is 1-based but a GraphBLAS matrix is always 0-based, one
 * is subtracted from the row and column indices in the file, so this entry
 * becomes A(7,3).  Note however that LAGraph does not yet support complex
 * types.
 *
 * Array Format
 * ------------
 * For array format, the first non-comment line must appear, and it
 * must contain just two integers::
 *
 *      nrows ncols
 *
 * A 5-by-12 matrix would have this as the first non-comment line after the
 * header::
 *
 *      5 12
 *
 * Each of the remaining lines defines one entry, in column major order.  If
 * the <type> is real or integer, this is the value of the entry.  An entry if
 * <type> of complex consists of two values, the real and imaginary part (not
 * yet supported).  The <type> cannot be pattern in this case.
 *
 * Infinity & Not-A-Number
 * -----------------------
 * For both coordinate and array formats, real and complex values may use the
 * terms ``INF``, ``+INF``, ``-INF``, and ``NAN`` to represent floating-point
 * infinity and NaN values, in either upper or lower case.
 * }
 *
 * According to the Matrix Market format, entries are always listed in
 * column-major order.  This rule is follwed by @sphinxref{LAGraph_MMWrite}.
 * However, LAGraph_MMRead can read the entries in any order.
 *
 * @param[out] A        handle of the matrix to create.
 * @param[in,out]  f    handle to an open file to read from.
 * @param[in,out] msg   any error messages.
 *
 * @retval GrB_SUCCESS if successful.
 * @retval GrB_NULL_POINTER if A or f are NULL.
 * @retval LAGRAPH_IO_ERROR if the file could not
 *      be read or contains a matrix with an invalid Matrix Market format.
 * @retval GrB_NOT_IMPLEMENTED if the type is not supported.  Complex types
 *      (GxB_FC32 and GxB_FC64 in SuiteSparse:GraphBLAS) are not yet supported.
 * @returns any GraphBLAS errors that may have been encountered.
 */

// FUTURE: add support for user-defined types. Perhaps LAGr_MMRead (...) with
// an extra parameter: pointer to function that reads a single UDT scalar
// from the file and returns the UDT scalar itself.  Or LAGraph_MMRead_UDT.

LAGRAPH_PUBLIC
int LAGraph_MMRead
(
    // output:
    GrB_Matrix *A,  // handle of matrix to create
    // input:
    FILE *f,        // file to read from, already open
    char *msg
) ;

//------------------------------------------------------------------------------
// LAGraph_MMWrite: write a matrix in MatrixMarket format
//------------------------------------------------------------------------------

/** LAGraph_MMWrite: writes a matrix in MatrixMarket format.  Refer to
 * @sphinxref{LAGraph_MMRead} for a description of the output file format.  The
 * MatrixMarket header line always appears, followed by the second line
 * containing the GraphBLAS type:
 *
 *      %%GraphBLAS type <entrytype>
 *
 * @param[in] A         matrix to write.
 * @param[in,out] f     handle to an open file to write to.
 * @param[in] fcomments optional handle to an open file containing comments; may be NULL.
 * @param[in,out] msg   any error messages.
 *
 * @retval GrB_SUCCESS if successful.
 * @retval GrB_NULL_POINTER if A or f are NULL.
 * @retval LAGRAPH_IO_ERROR if the file could not be written to.
 * @retval GrB_NOT_IMPLEMENTED if the type is not supported.  Complex types
 *      (GxB_FC32 and GxB_FC64 in SuiteSparse:GraphBLAS) are not yet supported.
 * @returns any GraphBLAS errors that may have been encountered.
 */

// FUTURE: add support for user-defined types.  Perhaps as LAGr_MMWrite, which
// has an extra parameter: a pointer to a function that takes a UDT scalar as
// input, and writes it to the file.  Or call it LAGraph_MMWrite_UDT.

LAGRAPH_PUBLIC
int LAGraph_MMWrite
(
    // input:
    GrB_Matrix A,       // matrix to write to the file
    FILE *f,            // file to write it to, must be already open
    FILE *fcomments,    // optional file with extra comments, may be NULL
    char *msg
) ;

//------------------------------------------------------------------------------
// LAGraph_Matrix_Structure: return the structure of a matrix
//------------------------------------------------------------------------------

/** LAGraph_Matrix_Structure: returns the sparsity structure of a matrix A as a
 * boolean (GrB_BOOL) matrix C.  If A(i,j) appears in the sparsity structure of
 * A, then C(i,j) is set to true.  The sparsity structure of A and C are
 * identical.
 *
 * @param[out] C    A boolean matrix with same structure of A, with C(i,j)
 *                  true if A(i,j) appears in the sparsity structure of A.
 * @param[in]  A    matrix to compute the structure for.
 * @param[in,out] msg   any error messages.
 *
 * @retval GrB_SUCCESS if successful.
 * @retval GrB_NULL_POINTER if A or C are NULL.
 * @returns any GraphBLAS errors that may have been encountered.
 */

LAGRAPH_PUBLIC
int LAGraph_Matrix_Structure
(
    // output:
    GrB_Matrix *C,      // the structure of A
    // input:
    GrB_Matrix A,
    char *msg
) ;

//------------------------------------------------------------------------------
// LAGraph_Vector_Structure: return the structure of a vector
//------------------------------------------------------------------------------

/** LAGraph_Vector_Structure: returns the sparsity structure of a vector u as a
 * boolean (GrB_BOOL) vector w.  If u(i) appears in the sparsity structure of
 * u, then w(i) is set to true.  The sparsity structure of u and w are
 * identical.
 *
 * @param[out] w    A boolean vector with same structure of u, with w(i)
 *                  true if u(i,j) appears in the sparsity structure of u.
 * @param[in]  u    vector to compute the structure for.
 * @param[in,out] msg   any error messages.
 *
 * @retval GrB_SUCCESS if successful.
 * @retval GrB_NULL_POINTER if w or u are NULL.
 * @returns any GraphBLAS errors that may have been encountered.
 */

LAGRAPH_PUBLIC
int LAGraph_Vector_Structure
(
    // output:
    GrB_Vector *w,  // the structure of u
    // input:
    GrB_Vector u,
    char *msg
) ;

//------------------------------------------------------------------------------
// LAGraph_NameOfType: return the name of a type
//------------------------------------------------------------------------------

/** LAGraph_NameOfType returns the name of a GraphBLAS type as a string.  The
 * names for the 11 built-in types (GrB_BOOL, GrB_INT8, etc) correspond to the
 * names of the corresponding C types (bool, int8_t, etc).
 *
 * @param[out] name     name of the type: user provided array of size at
 *                      least LAGRAPH_MAX_NAME_LEN.
 * @param[in]  type     GraphBLAS type to find the name of.
 * @param[in,out] msg   any error messages.
 *
 * @retval GrB_SUCCESS if successful.
 * @retval GrB_NULL_POINTER if name or type are NULL.
 * @returns any GraphBLAS errors that may have been encountered.
 */

LAGRAPH_PUBLIC
int LAGraph_NameOfType
(
    // output:
    char *name,     // name of type: user array of size LAGRAPH_MAX_NAME_LEN
    // input:
    GrB_Type type,  // GraphBLAS type
    char *msg
) ;

//------------------------------------------------------------------------------
// LAGraph_TypeFromName: return a GrB_Type from its name
//------------------------------------------------------------------------------

/** LAGraph_TypeFromName: returns the GrB_Type corresponding to its name.  That
 * is, given the string "bool", this method returns GrB_BOOL.
 *
 * @param[out] type     GraphBLAS type corresponding to the given name string.
 * @param[in]  name     name of the type: a null-terminated string.
 * @param[in,out] msg   any error messages.
 *
 * @retval GrB_SUCCESS if successful.
 * @retval GrB_NULL_POINTER if name or type are NULL.
 * @returns any GraphBLAS errors that may have been encountered.
 */

LAGRAPH_PUBLIC
int LAGraph_TypeFromName
(
    // output:
    GrB_Type *type, // GraphBLAS type
    // input:
    char *name,     // name of the type: a null-terminated string
    char *msg
) ;

//------------------------------------------------------------------------------
// LAGraph_SizeOfType: return sizeof(...) of a GraphBLAS GrB_Type
//------------------------------------------------------------------------------

/** LAGraph_SizeOfType: returns sizeof(...) of a GraphBLAS GrB_Type.  For
 * example, if given the GrB_Type of GrB_FP64, the value sizeof(double) is
 * returned.
 *
 * @param[out] size     size of the type
 * @param[in]  type     GraphBLAS type to find the size of.
 * @param[in,out] msg   any error messages.
 *
 * @retval GrB_SUCCESS if successful.
 * @retval GrB_NULL_POINTER if size or type are NULL.
 * @returns any GraphBLAS errors that may have been encountered.
 */

LAGRAPH_PUBLIC
int LAGraph_SizeOfType
(
    // output:
    size_t *size,   // size of the type
    // input:
    GrB_Type type,  // GraphBLAS type
    char *msg
) ;

//------------------------------------------------------------------------------
// LAGraph_Matrix_TypeName: return the name of the GrB_Type of a GrB_Matrix
//------------------------------------------------------------------------------

/** LAGraph_Matrix_TypeName: returns the name of the GrB_Type of a GrB_Matrix.
 *
 * @param[out] name     name of the type of the matrix: user provided array of
 *                      size at least LAGRAPH_MAX_NAME_LEN.
 * @param[in]  A        GraphBLAS matrix to find the type name of.
 * @param[in,out] msg   any error messages.
 *
 * @retval GrB_SUCCESS if successful.
 * @retval GrB_NULL_POINTER if name or A are NULL.
 * @returns any GraphBLAS errors that may have been encountered.
 */

LAGRAPH_PUBLIC
int LAGraph_Matrix_TypeName
(
    // output:
    char *name,     // name of the type of the matrix A (user-provided array
                    // of size at least LAGRAPH_MAX_NAME_LEN).
    // input:
    GrB_Matrix A,   // matrix to query
    char *msg
) ;

//------------------------------------------------------------------------------
// LAGraph_Vector_TypeName: return the name of the GrB_Type of a GrB_Vector
//------------------------------------------------------------------------------

/** LAGraph_Vector_TypeName: returns the name of the GrB_Type of a GrB_Vector.
 *
 * @param[out] name     name of the type of the vector: user provided array of
 *                      size at least LAGRAPH_MAX_NAME_LEN.
 * @param[in]  v        GraphBLAS vector to find the type name of.
 * @param[in,out] msg   any error messages.
 *
 * @retval GrB_SUCCESS if successful.
 * @retval GrB_NULL_POINTER if name or v are NULL.
 * @returns any GraphBLAS errors that may have been encountered.
 */

LAGRAPH_PUBLIC
int LAGraph_Vector_TypeName
(
    // output:
    char *name,     // name of the type of the vector v (user-provided array
                    // of size at least LAGRAPH_MAX_NAME_LEN).
    // input:
    GrB_Vector v,   // vector to query
    char *msg
) ;

//------------------------------------------------------------------------------
// LAGraph_Scalar_TypeName: return the name of the GrB_Type of a GrB_Scalar
//------------------------------------------------------------------------------

/** LAGraph_Scalar_TypeName: returns the name of the GrB_Type of a GrB_Scalar.
 *
 * @param[out] name     name of the type of the scalar: user provided array of
 *                      size at least LAGRAPH_MAX_NAME_LEN.
 * @param[in]  s        GraphBLAS scalar to find the type name of.
 * @param[in,out] msg   any error messages.
 *
 * @retval GrB_SUCCESS if successful.
 * @retval GrB_NULL_POINTER if name or s are NULL.
 * @returns any GraphBLAS errors that may have been encountered.
 */

LAGRAPH_PUBLIC
int LAGraph_Scalar_TypeName
(
    // output:
    char *name,     // name of the type of the scalar s (user-provided array
                    // of size at least LAGRAPH_MAX_NAME_LEN).
    // input:
    GrB_Scalar s,   // scalar to query
    char *msg
) ;

//------------------------------------------------------------------------------
// LAGraph_PrintLevel:  control printing in LAGraph_*_Print methods
//------------------------------------------------------------------------------

/** LAGraph_PrintLevel:  an enum to control how much to print in
 * LAGraph_*_Print methods.
 */

typedef enum
{
    LAGraph_SILENT = 0,     ///< nothing is printed.
    LAGraph_SUMMARY = 1,    ///< print a terse summary.
    LAGraph_SHORT = 2,      ///< short description, about 30 entries.
    LAGraph_COMPLETE = 3,   ///< print the entire contents of the object.
    LAGraph_SHORT_VERBOSE = 4,    ///< short, but with "%.15g" for doubles.
    LAGraph_COMPLETE_VERBOSE = 5  ///< complete, but "%.15g" for doubles.
    // FUTURE:
    // LAGraph_SHORT_ARRAY ...
    // LAGraph_COMPLETE_ARRAY ...
    //
    //      .  3.3 . 9 .
    //     99  32  . . .
    //
    // LAGraph_SHORT_STRUCTURE ...
    // LAGraph_COMPLETE_STRUCTURE ...
    //
    //      . x . x x x .
    //      x x . x x x .
    //      . x . x . x .
    //      . . . . x x .
    // FUTURE: # of digits per entry? automatic digits per column?
}
LAGraph_PrintLevel ;

//------------------------------------------------------------------------------
// LAGraph_Graph_Print: print the contents of a graph
//------------------------------------------------------------------------------

/** LAGraph_Graph_Print: prints the contents of a graph to a file in a human-
 * readable form.  This method is not meant for saving a graph to a file; see
 * @sphinxref{LAGraph_MMWrite} for that method.
 *
 * @param[in] G         graph to display.
 * @param[in] pr        print level.
 * @param[in,out] f     handle to an open file to write to.
 * @param[in,out] msg   any error messages.
 *
 * @retval GrB_SUCCESS if successful.
 * @retval GrB_NULL_POINTER if G or f are NULL.
 * @retval LAGRAPH_INVALID_GRAPH if G is invalid
 *              (@sphinxref{LAGraph_CheckGraph} failed).
 * @retval GrB_NOT_IMPLEMENTED if G->A has a user-defined type.
 * @retval LAGRAPH_IO_ERROR if the file could not be written to.
 * @returns any GraphBLAS errors that may have been encountered.
 */

LAGRAPH_PUBLIC
int LAGraph_Graph_Print
(
    // input:
    const LAGraph_Graph G,  // graph to display
    LAGraph_PrintLevel pr,  // print level (0 to 5)
    FILE *f,                // file to write to, must already be open
    char *msg
) ;

//------------------------------------------------------------------------------
// LAGraph_Matrix_Print: pretty-print a matrix
//------------------------------------------------------------------------------

/** LAGraph_Matrix_Print displays a matrix in a human-readable form.  This
 * method is not meant for saving a GrB_Matrix to a file; see
 * @sphinxref{LAGraph_MMWrite} for that method.
 *
 * @param[in] A         matrix to display.
 * @param[in] pr        print level.
 * @param[in,out] f     handle to an open file to write to.
 * @param[in,out] msg   any error messages.
 *
 * @retval GrB_SUCCESS if successful.
 * @retval GrB_NULL_POINTER if A or f are NULL.
 * @retval GrB_NOT_IMPLEMENTED if A has a user-defined type.
 * @retval LAGRAPH_IO_ERROR if the file could not be written to.
 * @returns any GraphBLAS errors that may have been encountered.
 */

LAGRAPH_PUBLIC
int LAGraph_Matrix_Print
(
    // input:
    const GrB_Matrix A,     // matrix to pretty-print to the file
    LAGraph_PrintLevel pr,  // print level (0 to 5)
    FILE *f,            // file to write it to, must be already open; use
                        // stdout or stderr to print to those locations.
    char *msg
) ;

//------------------------------------------------------------------------------
// LAGraph_Vector_Print: pretty-print a matrix
//------------------------------------------------------------------------------

/** LAGraph_Vector_Print displays a vector in a human-readable form.  This
 * method is not meant for saving a GrB_Vector to a file.  To perform that
 * operation, copy the GrB_Vector into an n-by-1 GrB_Matrix and use
 * @sphinxref{LAGraph_MMWrite}.
 *
 * @param[in] v         vector to display.
 * @param[in] pr        print level.
 * @param[in,out] f     handle to an open file to write to.
 * @param[in,out] msg   any error messages.
 *
 * @retval GrB_SUCCESS if successful.
 * @retval GrB_NULL_POINTER if v or f are NULL.
 * @retval GrB_NOT_IMPLEMENTED if v has a user-defined type.
 * @retval LAGRAPH_IO_ERROR if the file could not be written to.
 * @returns any GraphBLAS errors that may have been encountered.
 */

LAGRAPH_PUBLIC
int LAGraph_Vector_Print
(
    // input:
    const GrB_Vector v,     // vector to pretty-print to the file
    LAGraph_PrintLevel pr,  // print level (0 to 5)
    FILE *f,            // file to write it to, must be already open; use
                        // stdout or stderr to print to those locations.
    char *msg
) ;

//------------------------------------------------------------------------------
// LAGraph_Matrix_IsEqual: compare for exact equality
//------------------------------------------------------------------------------

/** LAGraph_Matrix_IsEqual compares two matrices for exact equality.  If the
 * two matrices have different data types, the result is always false (no
 * typecasting is performed).  Only the 11 built-in GrB* types are supported.
 * If both A and B are NULL, the return value is true.  If A and/or B are
 * floating-point types and contain NaN's, result is false.
 *
 * @param[out] result   true if A and B are exactly equal, false otherwise.
 * @param[in] A         matrix to compare.
 * @param[in] B         matrix to compare.
 * @param[in,out] msg   any error messages.
 *
 * @retval GrB_SUCCESS if successful.
 * @retval GrB_NULL_POINTER if result is NULL.
 * @retval GrB_NOT_IMPLEMENTED if A or B has a user-defined type.
 * @returns any GraphBLAS errors that may have been encountered.
 */

LAGRAPH_PUBLIC
int LAGraph_Matrix_IsEqual
(
    // output:
    bool *result,       // true if A == B, false if A != B or error
    // input:
    const GrB_Matrix A,
    const GrB_Matrix B,
    char *msg
) ;

//------------------------------------------------------------------------------
// LAGraph_Matrix_IsEqualOp: check if two matrices are equal with given op
//------------------------------------------------------------------------------

/** LAGraph_Matrix_IsEqualOp compares two matrices using the given binary
 * operator.  The op may be built-in or user-defined.  The two matrices may
 * have different types and still be determined to be equal.  To be equal, two
 * matrices must have the same sparsity structure, and op(aij,bij) must return
 * true for all pairs of entries aij and bij that appear in the structure of
 * both A and B.  The matrices A and/or B can have any type, as long as they
 * are valid inputs to the op.  If both A and B are NULL, the return value is
 * true.
 *
 * @param[out] result   true if A and B are equal (per the op), false otherwise.
 * @param[in] A         matrix to compare.
 * @param[in] B         matrix to compare.
 * @param[in] op        operator for the comparison.
 * @param[in,out] msg   any error messages.
 *
 * @retval GrB_SUCCESS if successful.
 * @retval GrB_NULL_POINTER if result or op are NULL.
 * @returns any GraphBLAS errors that may have been encountered.
 */

LAGRAPH_PUBLIC
int LAGraph_Matrix_IsEqualOp
(
    // output:
    bool *result,           // true if A == B, false if A != B or error
    // input:
    const GrB_Matrix A,
    const GrB_Matrix B,
    const GrB_BinaryOp op,        // comparator to use
    char *msg
) ;

//------------------------------------------------------------------------------
// LAGraph_Vector_IsEqual: check if two vectors are equal
//------------------------------------------------------------------------------

/** LAGraph_Vector_IsEqual compares two vectors for exact equality.  If the
 * two vectors have different data types, the result is always false (no
 * typecasting is performed).  Only the 11 built-in GrB* types are supported.
 * If both u and v are NULL, the return value is true.  If u and/or v are
 * floating-point types and contain NaN's, result is false.
 *
 * @param[out] result   true if u and v are exactly equal, false otherwise.
 * @param[in] u         vector to compare.
 * @param[in] v         vector to compare.
 * @param[in,out] msg   any error messages.
 *
 * @retval GrB_SUCCESS if successful.
 * @retval GrB_NULL_POINTER if result is NULL.
 * @retval GrB_NOT_IMPLEMENTED if u or v has a user-defined type.
 * @returns any GraphBLAS errors that may have been encountered.
 */

LAGRAPH_PUBLIC
int LAGraph_Vector_IsEqual
(
    // output:
    bool *result,           // true if A == B, false if A != B or error
    // input:
    const GrB_Vector u,
    const GrB_Vector v,
    char *msg
) ;

//------------------------------------------------------------------------------
// LAGraph_Vector_IsEqualOp: check if two vectors are equal with given op
//------------------------------------------------------------------------------

/** LAGraph_Vector_IsEqualOp compares two vectors using the given binary
 * operator.  The op may be built-in or user-defined.  The two vectors may
 * have different types and still be determined to be equal.  To be equal, two
 * vectors must have the same sparsity structure, and op(ui,vi) must return
 * true for all pairs of entries ui and vi that appear in the structure of
 * both u and v.  The vectors u and/or v can have any type, as long as they
 * are valid inputs to the op.  If both u and v are NULL, the return value is
 * true.
 *
 * @param[out] result   true if u and v are equal (per the op), false otherwise.
 * @param[in] u         vector to compare.
 * @param[in] v         vector to compare.
 * @param[in] op        operator for the comparison.
 * @param[in,out] msg   any error messages.
 *
 * @retval GrB_SUCCESS if successful.
 * @retval GrB_NULL_POINTER if result or op are NULL.
 * @returns any GraphBLAS errors that may have been encountered.
 */

LAGRAPH_PUBLIC
int LAGraph_Vector_IsEqualOp
(
    // output:
    bool *result,           // true if u == v, false if u != v or error
    // input:
    const GrB_Vector u,
    const GrB_Vector v,
    const GrB_BinaryOp op,        // comparator to use
    char *msg
) ;

//==============================================================================
// LAGraph Basic algorithms
//==============================================================================

// Basic algorithm are meant to be easy to use.  They may encompass many
// underlying Advanced algorithms, each with various parameters that may be
// controlled.  For the Basic API, these parameters are determined
// automatically.  Cached graph properties may be determined, and as a result,
// the graph G is both an input and an output of these methods, since they may
// be modified.

// LAGraph Basic algorithms are named with the LAGraph_* prefix.

//------------------------------------------------------------------------------
// LAGraph_TriangleCount
//------------------------------------------------------------------------------

/** LAGraph_TriangleCount: count the triangles in a graph.  This is a Basic
 * algorithm (G->nself_edges, G->out_degree, G->is_symmetric_structure are
 * computed, if not present).
 *
 * @param[out]    ntriangles    the number of triangles in G.
 * @param[in,out] G             the graph, which must by undirected, or
 *                              directed but with a symmetric structure.
 *                              No self loops can be present.
 * @param[in,out] msg           any error messages.
 *
 * @retval GrB_SUCCESS if successful.
 * @retval GrB_NULL_POINTER if G or ntriangles are NULL.
 * @retval LAGRAPH_INVALID_GRAPH if G is invalid
 *              (@sphinxref{LAGraph_CheckGraph} failed).
 * @retval LAGRAPH_NO_SELF_EDGES_ALLOWED if G has any self-edges.
 * @retval LAGRAPH_SYMMETRIC_STRUCTURE_REQUIRED if G is directed with an
 *      unsymmetric G->A matrix.
 * @returns any GraphBLAS errors that may have been encountered.
 */

LAGRAPH_PUBLIC
int LAGraph_TriangleCount
(
    // output:
    uint64_t *ntriangles,   // # of triangles
    // input/output:
    LAGraph_Graph  G,
    char *msg
) ;

//==============================================================================
// LAGraph Advanced algorithms and utilities
//==============================================================================

// The Advanced algorithms require the caller to select the algorithm and choose
// any parameter settings.  G is not modified, and so it is an input-only
// parameter to these methods.  If an Advanced algorithm requires a cached
// graph property to be computed, it must be computed prior to calling the
// Advanced method.

// Advanced algorithms are named with the LAGr_* prefix, to distinguish them
// from Basic algorithms.

//------------------------------------------------------------------------------
// LAGr_Init: start GraphBLAS and LAGraph, and set malloc/etc functions
//------------------------------------------------------------------------------

/** LAGr_Init: initializes GraphBLAS and LAGraph.  LAGr_Init is identical to
 * @sphinxref{LAGraph_Init}, except that it allows the user application to
 * specify the GraphBLAS mode.  It also provides four memory management
 * functions, replacing the standard `malloc`, `calloc`, `realloc`, and `free`.
 * The functions `user_malloc_function`, `user_calloc_function`,
 * `user_realloc_function`, and `user_free_function` have the same signature as
 * the ANSI C malloc, calloc, realloc, and free functions, respectively.  Only
 * user_malloc_function and user_free_function are required.
 * user_calloc_function may be NULL, in which case `LAGraph_Calloc` uses
 * `LAGraph_Malloc` and `memset`.  Likewise, user_realloc_function may be NULL,
 * in which case `LAGraph_Realloc` uses `LAGraph_Malloc`, `memcpy`, and
 * `LAGraph_Free`.
 *
 * @param[in] mode                      the mode for GrB_Init
 * @param[in] user_malloc_function      pointer to a malloc function
 * @param[in] user_calloc_function      pointer to a calloc function, or NULL
 * @param[in] user_realloc_function     pointer to a realalloc function, or NULL
 * @param[in] user_free_function        pointer to a free function
 * @param[in,out] msg                   any error messages.
 *
 * @retval GrB_SUCCESS if successful.
 * @retval GrB_INVALID_VALUE if LAGraph_Init or LAGr_Init has already been
 *      called by the user application.
 * @retval GrB_NULL_POINTER if user_malloc_function or user_free_function
 *      are NULL.
 * @returns any GraphBLAS errors that may have been encountered.
 */

LAGRAPH_PUBLIC
int LAGr_Init
(
    // input:
    GrB_Mode mode,      // mode for GrB_Init
    void * (* user_malloc_function  ) (size_t),
    void * (* user_calloc_function  ) (size_t, size_t),
    void * (* user_realloc_function ) (void *, size_t),
    void   (* user_free_function    ) (void *),
    char *msg
) ;

//------------------------------------------------------------------------------
// LAGr_SortByDegree: sort a graph by its row or column degree
//------------------------------------------------------------------------------

/** LAGr_SortByDegree sorts the nodes of a graph by their out or in degrees.
 * The graph G->A itself is not changed.  Refer to LAGr_TriangleCount for an
 * example of how to permute G->A after calling this function.  The output &P
 * must be freed by LAGraph_Free.  This method requires G->out_degree or
 * G->in_degree to already be computed.
 *
 * @param[out] P        permutation of the integers 0..n-1.
 * @param[in] G         graph of n nodes.
 * @param[in] byout     if true, sort by out-degree, else sort by in-degree.
 * @param[in] ascending if true, sort in ascending order, else descending.
 * @param[in,out] msg   any error messages.
 *
 * @retval GrB_NULL_POINTER if P or G are NULL.
 * @retval LAGRAPH_NOT_CACHED if G->in_degree or G->out_degree is not computed
 *      (whichever one is required).
 * @retval LAGRAPH_INVALID_GRAPH if G is invalid
 *              (@sphinxref{LAGraph_CheckGraph} failed).
 * @returns any GraphBLAS errors that may have been encountered.
 */

LAGRAPH_PUBLIC
int LAGr_SortByDegree
(
    // output:
    int64_t **P,            // permutation vector of size n
    // input:
    const LAGraph_Graph G,  // graph of n nodes
    bool byout,             // if true, sort G->out_degree, else G->in_degree
    bool ascending,         // sort in ascending or descending order
    char *msg
) ;

//------------------------------------------------------------------------------
// LAGr_SampleDegree: sample the degree median and mean
//------------------------------------------------------------------------------

/** LAGr_SampleDegree computes an estimate of the median and mean of the out or
 * in degree, by randomly sampling the G->out_degree or G->in_degree vector.
 * This method requires G->out_degree or G->in_degree to already be computed.
 *
 * @param[out] sample_mean      sampled mean of the degree.
 * @param[out] sample_median    sampled median of the degree.
 * @param[in] G         graph to sample.
 * @param[in] byout     if true, sample out-degree, else sample in-degree.
 * @param[in] nsamples  number of samples to take.
 * @param[in] seed      random number seed.
 * @param[in,out] msg   any error messages.
 *
 * @retval GrB_NULL_POINTER if sample_mean, sample_median, or G are NULL.
 * @retval LAGRAPH_NOT_CACHED if G->in_degree or G->out_degree is not computed
 *      (whichever one is required).
 * @retval LAGRAPH_INVALID_GRAPH if G is invalid
 *              (@sphinxref{LAGraph_CheckGraph} failed).
 * @returns any GraphBLAS errors that may have been encountered.
 */

LAGRAPH_PUBLIC
int LAGr_SampleDegree
(
    // output:
    double *sample_mean,    // sampled mean degree
    double *sample_median,  // sampled median degree
    // input:
    const LAGraph_Graph G,  // graph of n nodes
    bool byout,             // if true, sample G->out_degree, else G->in_degree
    int64_t nsamples,       // number of samples
    uint64_t seed,          // random number seed
    char *msg
) ;

//------------------------------------------------------------------------------
// LAGr_BreadthFirstSearch: breadth-first search
//------------------------------------------------------------------------------

/** LAGr_BreadthFirstSearch: breadth-first search of a graph, computing the
 * breadth-first-search tree and/or the level of the nodes encountered.  This
 * is an Advanced algorithm.  G->AT and G->out_degree are required to use the
 * fastest push/pull method when using SuiteSparse:GraphBLAS.  If these cached
 * properties are not present, or if a vanilla GraphBLAS library is being used,
 * then a push-only method is used (which can be slower).  G is not modified;
 * that is, G->AT and G->out_degree are not computed if not already cached.
 *
 * @param[out]    level      If non-NULL on input, on successful return, it
 *                           contains the levels of each node reached. The
 *                           src node is assigned level 0. If a node i is
 *                           not reached, level(i) is not present.  The level
 *                           vector is not computed if NULL.
 * @param[out]    parent     If non-NULL on input, on successful return, it
 *                           contains parent node IDs for each node
 *                           reached, where parent(i) is the node ID of the
 *                           parent of node i.  The src node will have itself
 *                           as its parent. If a node i is not reached,
 *                           parent(i) is not present.  The parent vector is
 *                           not computed if NULL.
 * @param[in]     G          The graph, directed or undirected.
 * @param[in]     src        The index of the src node (0-based)
 * @param[in,out] msg        any error messages.
 *
 * @retval GrB_SUCCESS if successful.
 * @retval GrB_INVALID_INDEX if src is invalid.
 * @retval GrB_NULL_POINTER if both level and parent are NULL, or if
 *      G is NULL.
 * @retval LAGRAPH_INVALID_GRAPH Graph is invalid
 *              (@sphinxref{LAGraph_CheckGraph} failed).
 * @returns any GraphBLAS errors that may have been encountered.
 */

LAGRAPH_PUBLIC
int LAGr_BreadthFirstSearch
(
    // output:
    GrB_Vector *level,
    GrB_Vector *parent,
    // input:
    const LAGraph_Graph G,
    GrB_Index src,
    char *msg
) ;

//------------------------------------------------------------------------------
// LAGr_ConnectedComponents: connected components of an undirected graph
//------------------------------------------------------------------------------

/** LAGr_ConnectedComponents: connected components of an undirected graph.
 * This is an Advanced algorithm (G->is_symmetric_structure must be known).
 *
 * @param[out] component    component(i)=s if node i is in the component whose
 *                          representative node is s.  If node i has no edges,
 *                          it is placed in its own component, and thus the
 *                          component vector is always dense.
 * @param[in] G             input graph to find the components for.
 *                          The graph must be undirected, or
 *                          G->is_symmetric_structure must be true.
 * @param[in,out] msg       any error messages.
 *
 * @retval GrB_SUCCESS if successful.
 * @retval GrB_NULL_POINTER if G or component are NULL.
 * @retval LAGRAPH_INVALID_GRAPH Graph is invalid
 *              (@sphinxref{LAGraph_CheckGraph} failed).
 * @retval LAGRAPH_SYMMETRIC_STRUCTURE_REQUIRED if G is directed with an
 *      unsymmetric G->A matrix.
 * @returns any GraphBLAS errors that may have been encountered.
 */

LAGRAPH_PUBLIC
int LAGr_ConnectedComponents
(
    // output:
    GrB_Vector *component,
    // input:
    const LAGraph_Graph G,  // input graph
    char *msg
) ;

//------------------------------------------------------------------------------
// LAGr_SingleSourceShortestPath: single-source shortest paths
//------------------------------------------------------------------------------

/** LAGr_SingleSourceShortestPath: single-source shortest paths.  This is an
 * Advanced algorithm (G->emin is required for best performance).  The graph G
 * must have an adjacency matrix of type GrB_INT32, GrB_INT64, GrB_UINT32,
 * GrB_UINT64, GrB_FP32, or GrB_FP64.  If G->A has any other type,
 * GrB_NOT_IMPLEMENTED is returned.
 *
 * @param[out] path_length  path_length (i) is the length of the shortest
 *     path from the source node to node i.  The path_length vector is dense.
 *     If node (i) is not reachable from the src node, then path_length (i) is
 *     set to INFINITY for GrB_FP32 and FP32, or the maximum integer for
 *     GrB_INT32, INT64, UINT32, or UINT64.
 * @param[in] G         input graph.
 * @param[in] src       source node.
 * @param[in] Delta     for delta stepping.
 * @param[in,out] msg   any error messages.
 *
 * @retval GrB_SUCCESS if successful.
 * @retval GrB_NULL_POINTER if G or path_length are NULL.
 * @retval GrB_INVALID_INDEX if src is invalid.
 * @retval GrB_EMPTY_OBJECT if Delta does not contain a value.
 * @retval GrB_NOT_IMPLEMENTED if the type is not supported.
 * @retval LAGRAPH_INVALID_GRAPH Graph is invalid
 *              (@sphinxref{LAGraph_CheckGraph} failed).
 * @returns any GraphBLAS errors that may have been encountered.
 */

// FUTURE: add a Basic algorithm that computes G->emin, G->emax, and then uses
// that information to compute an appropriate (estimated) Delta.

LAGRAPH_PUBLIC
int LAGr_SingleSourceShortestPath
(
    // output:
    GrB_Vector *path_length,
    // input:
    const LAGraph_Graph G,
    GrB_Index src,
    GrB_Scalar Delta,           // delta value for delta stepping
    char *msg
) ;

//------------------------------------------------------------------------------
// LAGr_Betweenness: betweeness centrality metric
//------------------------------------------------------------------------------

/** LAGr_Betweenness: betweeness centrality metric.  This methods computes an
 * approximation of the betweeness-centrality metric of all nodes in the graph.
 * Only a few given source nodes are used for the approximation.  This is an
 * Advanced algorithm (G->AT is required).
 *
 * @param[out] centrality   centrality(i) is the metric for node i.
 * @param[in] G         input graph.
 * @param[in] sources   source vertices to compute shortest paths, size ns
 * @param[in] ns        number of source vertices.
 * @param[in,out] msg   any error messages.
 *
 * @retval GrB_SUCCESS if successful.
 * @retval GrB_NULL_POINTER if G, centrality, and/our sources are NULL.
 * @retval GrB_INVALID_INDEX if any source node is invalid.
 * @retval LAGRAPH_INVALID_GRAPH Graph is invalid
 *              (@sphinxref{LAGraph_CheckGraph} failed).
 * @retval LAGRAPH_NOT_CACHED if G->AT is required but not present.
 * @returns any GraphBLAS errors that may have been encountered.
 */

// FUTURE: create a Basic algorithm that randomly selects source nodes, or
// computes the exact centrality with all nodes as sources (which is very
// costly).


LAGRAPH_PUBLIC
int LAGr_Betweenness
(
    // output:
    GrB_Vector *centrality,     // centrality(i): betweeness centrality of i
    // input:
    const LAGraph_Graph G,      // input graph
    const GrB_Index *sources,   // source vertices to compute shortest paths
    int32_t ns,                 // number of source vertices
    char *msg
) ;

//------------------------------------------------------------------------------
// LAGr_PageRank: PageRank of a graph.
//------------------------------------------------------------------------------

/** LAGr_PageRank: computes the standard PageRank of a directed graph G.  Sinks
 * (nodes with no out-going edges) are properly handled.  This method should be
 * used for production, not for the GAP benchmark.  This is an Advanced
 * algorithm (G->AT and G->out_degree are required).
 *
 * @param[out] centrality   centrality(i) is the PageRank of node i.
 * @param[out] iters        number of iterations taken.
 * @param[in] G             input graph.
 * @param[in] damping       damping factor (typically 0.85).
 * @param[in] tol           stopping tolerance (typically 1e-4).
 * @param[in] itermax       maximum number of iterations (typically 100).
 * @param[in,out] msg       any error messages.
 *
 * @retval GrB_SUCCESS if successful.
 * @retval GrB_NULL_POINTER if G, centrality, and/our iters are NULL.
 * @retval LAGRAPH_NOT_CACHED if G->AT is required but not present,
 *      or if G->out_degree is not present.
 * @retval LAGRAPH_INVALID_GRAPH Graph is invalid
 *              (@sphinxref{LAGraph_CheckGraph} failed).
 * @returns any GraphBLAS errors that may have been encountered.
 */

LAGRAPH_PUBLIC
int LAGr_PageRank
(
    // output:
    GrB_Vector *centrality,
    int *iters,
    // input:
    const LAGraph_Graph G,
    float damping,
    float tol,
    int itermax,
    char *msg
) ;

//------------------------------------------------------------------------------
// LAGr_PageRankGAP: GAP-style PageRank of a graph (for GAP benchmarking only)
//------------------------------------------------------------------------------

// LAGr_PageRankGAP: computes the GAP PageRank of a directed graph G.  Sinks
// (nodes with no out-going edges) are NOT properly handled.  This method
// should be NOT be used for production.  It is intended for the GAP benchmark
// only.  This is an Advanced algorithm (G->AT and G->out_degree are required).
// The parameters and return values are the same as LAGr_PageRank.

LAGRAPH_PUBLIC
int LAGr_PageRankGAP
(
    // output:
    GrB_Vector *centrality,
    int *iters,
    // input:
    const LAGraph_Graph G,
    float damping,
    float tol,
    int itermax,
    char *msg
) ;

//------------------------------------------------------------------------------
// LAGr_TriangleCount: triangle counting
//------------------------------------------------------------------------------

/** LAGr_TriangleCount_Method: an enum to select the method used to count the
 * number of triangles.
 */

typedef enum
{
    LAGr_TriangleCount_AutoMethod = 0,  ///< auto selection of method
    LAGr_TriangleCount_Burkhardt = 1,   ///< sum (sum ((A^2) .* A)) / 6
    LAGr_TriangleCount_Cohen = 2,       ///< sum (sum ((L * U) .* A)) / 2
    LAGr_TriangleCount_Sandia_LL = 3,   ///< sum (sum ((L * L) .* L))
    LAGr_TriangleCount_Sandia_UU = 4,   ///< sum (sum ((U * U) .* U))
    LAGr_TriangleCount_Sandia_LUT = 5,  ///< sum (sum ((L * U') .* L))
    LAGr_TriangleCount_Sandia_ULT = 6,  ///< sum (sum ((U * L') .* U))
}
LAGr_TriangleCount_Method ;

/** LAGr_TriangleCount_Presort: an enum to control if/how the matrix is sorted
 * prior to counting triangles.
 */

typedef enum
{
    LAGr_TriangleCount_NoSort = 2,      ///< no sort
    LAGr_TriangleCount_Ascending = 1,   ///< sort by degree, ascending.
    LAGr_TriangleCount_Descending = -1, ///< sort by degree, descending.
    LAGr_TriangleCount_AutoSort = 0,    ///< auto selection of presort:
        ///< No presort is done for the Burkhardt or Cohen methods, and
        ///< no sort is done for the Sandia_* methods if the sampled mean
        ///< out-degree is <= 4 * the sample median out-degree.
        ///< Otherwise: sort in ascending order for Sandia_LL and Sandia_LUT,
        ///< descending ordering for Sandia_UU and Sandia_ULT.  On output,
        ///< presort is modified to reflect the sorting method used (NoSort,
        ///< Ascending, or Descending).
}
LAGr_TriangleCount_Presort ;

/** LAGr_TriangleCount: count the triangles in a graph (advanced API).
 *
 * @param[out] ntriangles   the number of triangles in G.
 * @param[in]  G            The graph, which must be undirected or have
 *                          G->is_symmetric_structure true, with no self loops.
 *                          G->nself_edges, G->out_degree, and
 *                          G->is_symmetric_structure are required.
 * @param[in,out] method    specifies which algorithm to use, and returns
 *                          the method chosen.  If NULL, the AutoMethod is
 *                          used, and the method is not reported.  Also see the
 *                          LAGr_TriangleCount_Method enum description.
 * @param[in,out] presort   controls the presort of the graph, and returns the
 *                          presort chosen.  If NULL, the AutoSort is used, and
 *                          the presort method is not reported.  Also see the
 *                          description of the LAGr_TriangleCount_Presort enum.
 * @param[in,out] msg       any error messages.
 *
 * @retval GrB_SUCCESS if successful.
 * @retval GrB_NULL_POINTER if G or ntriangles are NULL.
 * @retval LAGRAPH_INVALID_GRAPH Graph is invalid
 *              (@sphinxref{LAGraph_CheckGraph} failed).
 * @retval LAGRAPH_NO_SELF_EDGES_ALLOWED if G has any self-edges, or if
 *      G->nself_edges is not computed.
 * @retval LAGRAPH_SYMMETRIC_STRUCTURE_REQUIRED if G is directed with an
 *      unsymmetric G->A matrix.
 * @retval LAGRAPH_NOT_CACHED if G->out_degree is not present in G.
 * @retval GrB_INVALID_VALUE method or presort are invalid.
 * @returns any GraphBLAS errors that may have been encountered.
 */

LAGRAPH_PUBLIC
int LAGr_TriangleCount
(
    // output:
    uint64_t *ntriangles,
    // input:
    const LAGraph_Graph G,
    LAGr_TriangleCount_Method *method,
    LAGr_TriangleCount_Presort *presort,
    char *msg
) ;

#if defined ( __cplusplus )
}
#endif

#endif
