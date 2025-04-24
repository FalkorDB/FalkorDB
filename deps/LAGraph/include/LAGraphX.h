//------------------------------------------------------------------------------
// LAGraphX.h: include file for LAGraph experimental code
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

#ifndef LAGRAPHX_H
#define LAGRAPHX_H

#include <GraphBLAS.h>
#include <LAGraph.h>

#if ( _MSC_VER && !__INTEL_COMPILER && LGX_DLL )
    #ifdef LGX_LIBRARY
        // compiling LAGraph itself, exporting symbols to user apps
        #define LAGRAPHX_PUBLIC __declspec ( dllexport )
    #else
        // compiling the user application, importing symbols from LAGraph
        #define LAGRAPHX_PUBLIC __declspec ( dllimport )
    #endif
#else
    // for other compilers
    #define LAGRAPHX_PUBLIC
#endif

//==============================================================================
// for C++ applications:
//==============================================================================

#if defined ( __cplusplus )
extern "C"
{
#endif

//==============================================================================
// Experimental methods: in experimental/algorithm and experimental/utility
//==============================================================================

// Do not rely on these in production.  These methods are still under
// development, and is intended only for illustration or testing, not
// benchmarking.  Do not use for benchmarking without asking the authors.

//------------------------------------------------------------------------------
// LAGraph_Random_*: Random number generator
//------------------------------------------------------------------------------

LAGRAPHX_PUBLIC
int LAGraph_Random_Init
(
    char *msg
) ;
LAGRAPHX_PUBLIC
int LAGraph_Random_Finalize
(
    char *msg
) ;

#if defined ( COVERAGE )
// for testing only
LAGRAPHX_PUBLIC extern bool random_hack ;
#endif

LAGRAPHX_PUBLIC
int LAGraph_Random_Seed     // construct a random seed vector
(
    // input/output
    GrB_Vector Seed,    // vector of random number seeds, normally GrB_UINT64
    // input
    uint64_t seed,      // scalar input seed
    char *msg
) ;

LAGRAPHX_PUBLIC
int LAGraph_Random_Next     // advance to next random vector
(
    // input/output
    GrB_Vector Seed,
    char *msg
) ;

LAGRAPHX_PUBLIC
GrB_Info LAGraph_Random_Matrix    // random matrix of any built-in type
(
    // output
    GrB_Matrix *A,      // A is constructed on output
    // input
    GrB_Type type,      // type of matrix to construct
    GrB_Index nrows,    // # of rows of A
    GrB_Index ncols,    // # of columns of A
    double density,     // density: build a sparse matrix with
                        // density*nrows*cols values if not INFINITY;
                        // build a dense matrix if INFINITY.
    uint64_t seed,      // random number seed
    char *msg
) ;

//****************************************************************************
// binary file I/O
//****************************************************************************

// The LAGraph *.lagraph file consists of an ASCII JSON header, followed by
// one or more serialized "blobs" created by GrB_Matrix_serialize (or
// GxB_Matrix_serialize if using SuiteSparse:GraphBLAS).  The file can only be
// read back into LAGraph when using the same GraphBLAS library used to create
// it.

// To create a binary file containing one or more GrB_Matrix objects, the user
// application must first open the file f, create the ascii JSON header with
// LAGraph_SWrite_Header*, and then write one or more binary serialized
// GrB_Matrix blobs from  using LAGraph_SWrite_Matrix.

// Example:

/*
    // serialize the matrices A (of type GrB_FP64) and B (of type GrB_BOOL)
    void *Ablob, *Bblob ;
    GrB_Index Ablob_size, Bblob_size ;
    GxB_Matrix_serialize (&Ablob, &Ablob_size, A, NULL) ;
    GxB_Matrix_serialize (&Bblob, &Bblob_size, B, NULL) ;

    // open the file and write the JSON header
    FILE *f = fopen ("mymatrices.lagraph", "w") ;
    LAGraph_SWrite_HeaderStart (f, "mystuff", msg) ;
    LAGraph_SWrite_HeaderItem (f, LAGraph_matrix_kind, "A", "double", 0,
        Ablob_size, msg) ;
    LAGraph_SWrite_HeaderItem (f, LAGraph_matrix_kind, "B", "bool", 0,
        Bblob_size, msg) ;
    LAGraph_SWrite_HeaderEnd (f, msg) ;

    // write the matrices in binary
    LAGraph_SWrite_Item (f, Ablob, Ablob_size, msg) ;
    LAGraph_SWrite_Item (f, Bblob, Bblob_size, msg) ;

    fclose (f) ;
*/

typedef enum
{
    LAGraph_unknown_kind = -1,  // unknown kind
    LAGraph_matrix_kind = 0,    // a serialized GrB_Matrix
    LAGraph_vector_kind = 1,    // a serialized GrB_Vector (SS:GrB only)
    LAGraph_text_kind = 2,      // text (char *), possibly compressed
}
LAGraph_Contents_kind ;

typedef struct
{
    // serialized matrix/vector, or pointer to text, and its size
    void *blob ;
    size_t blob_size ;

    // kind of item: matrix, vector, text, or unknown
    LAGraph_Contents_kind kind ;

    // if kind is text: compression used
    // -1: none, 0: default for library, 1000: LZ4, 200x: LZ4HC:x
    int compression ;

    // name of the object
    char name [LAGRAPH_MAX_NAME_LEN+4] ;

    // if kind is matrix or vector: type name
    char type_name [LAGRAPH_MAX_NAME_LEN+4] ;
}
LAGraph_Contents ;

LAGRAPHX_PUBLIC
int LAGraph_SWrite_HeaderStart  // write the first part of the JSON header
(
    FILE *f,                    // file to write to
    const char *name,           // name of this collection of matrices
    char *msg
) ;

LAGRAPHX_PUBLIC
int LAGraph_SWrite_HeaderItem   // write a single item to the JSON header
(
    // inputs:
    FILE *f,                    // file to write to
    LAGraph_Contents_kind kind, // matrix, vector, or text
    const char *name,           // name of the matrix/vector/text; matrices from
                                // sparse.tamu.edu use the form "Group/Name"
    const char *type,           // name of type of the matrix/vector
    int compression,            // text compression method
    GrB_Index blob_size,        // exact size of serialized blob for this item
    char *msg
) ;

LAGRAPHX_PUBLIC
int LAGraph_SWrite_HeaderItem   // write a single item to the JSON header
(
    // inputs:
    FILE *f,                    // file to write to
    LAGraph_Contents_kind kind, // matrix, vector, or text
    const char *name,           // name of the matrix/vector/text; matrices from
                                // sparse.tamu.edu use the form "Group/Name"
    const char *type,           // name of type of the matrix/vector
    // todo: text not yet supported by LAGraph_SWrithe_HeaderItem
    int compression,            // text compression method
    GrB_Index blob_size,        // exact size of serialized blob for this item
    char *msg
) ;

LAGRAPHX_PUBLIC
int LAGraph_SWrite_HeaderEnd    // write the end of the JSON header
(
    FILE *f,                    // file to write to
    char *msg
) ;

LAGRAPHX_PUBLIC
int LAGraph_SWrite_Item  // write the serialized blob of a matrix/vector/text
(
    // input:
    FILE *f,                // file to write to
    const void *blob,       // serialized blob from G*B_Matrix_serialize
    GrB_Index blob_size,    // exact size of the serialized blob
    char *msg
) ;

LAGRAPHX_PUBLIC
int LAGraph_SRead   // read a set of matrices from a *.lagraph file
(
    FILE *f,                        // file to read from
    // output
    char **collection,              // name of collection (allocated string)
    LAGraph_Contents **Contents,    // array contents of contents
    GrB_Index *ncontents,           // # of items in the Contents array
    char *msg
) ;

LAGRAPHX_PUBLIC
void LAGraph_SFreeContents      // free the Contents returned by LAGraph_SRead
(
    // input/output
    LAGraph_Contents **Contents,    // array of size ncontents
    GrB_Index ncontents
) ;

LAGRAPHX_PUBLIC
int LAGraph_SSaveSet            // save a set of matrices from a *.lagraph file
(
    // inputs:
    char *filename,             // name of file to write to
    GrB_Matrix *Set,            // array of GrB_Matrix of size nmatrices
    GrB_Index nmatrices,        // # of matrices to write to *.lagraph file
//  todo: handle vectors and text in LAGraph_SSaveSet
    char *collection,           // name of this collection of matrices
    char *msg
) ;

int LAGraph_SLoadSet            // load a set of matrices from a *.lagraph file
(
    // input:
    char *filename,             // name of file to read from
    // outputs:
    GrB_Matrix **Set_handle,        // array of GrB_Matrix of size nmatrices
    GrB_Index *nmatrices_handle,    // # of matrices loaded from *.lagraph file
//  todo: handle vectors and text in LAGraph_SLoadSet
//  GrB_Vector **Set_handle,        // array of GrB_Vector of size nvector
//  GrB_Index **nvectors_handle,    // # of vectors loaded from *.lagraph file
//  char **Text_handle,             // array of pointers to (char *) strings
//  GrB_Index **ntext_handle,       // # of texts loaded from *.lagraph file
    char **collection_handle,   // name of this collection of matrices
    char *msg
) ;

LAGRAPHX_PUBLIC
void LAGraph_SFreeSet           // free a set of matrices
(
    // input/output
    GrB_Matrix **Set_handle,    // array of GrB_Matrix of size nmatrices
    GrB_Index nmatrices         // # of matrices in the set
) ;

//****************************************************************************
// Algorithms
//****************************************************************************

//****************************************************************************
/**
 * Given a symmetric graph A with no-self edges, compute all k-trusses of A.
 *
 * @param[out]  Cset    size n, output k-truss subgraphs.
 * @param[out]  kmax    smallest k where k-truss is empty
 * @param[out]  ntris   Array of size n (on input), ntris [k] is num triangles in k-truss
 * @param[out]  nedges  Array of size n (on input), nedges [k] is num edges in k-truss
 * @param[out]  nstepss Array of size n (on input), nstepss [k] is num steps for k-truss
 * @param[in]   G       input graph, A, not modified.  Must be undirected
 *                      or directed with symmetric structure, no self edges.
 * @param[in,out] msg   any error messages.
 *
 * @retval GrB_SUCCESS      if completed successfully (equal or not)
 * @retval GrB_NULL_POINTER if kmax, ntris, nedges, nsteps is NULL
 */
LAGRAPHX_PUBLIC
int LAGraph_AllKTruss   // compute all k-trusses of a graph
(
    // outputs
    GrB_Matrix *Cset,   // size n, output k-truss subgraphs
    int64_t *kmax,      // smallest k where k-truss is empty
    int64_t *ntris,     // size max(n,4), ntris [k] is #triangles in k-truss
    int64_t *nedges,    // size max(n,4), nedges [k] is #edges in k-truss
    int64_t *nstepss,   // size max(n,4), nstepss [k] is #steps for k-truss
    // input
    LAGraph_Graph G,    // input graph
    char *msg
) ;

//****************************************************************************
/**
 * Given an undirected graph G with no-self edges, LAGraph_KTruss finds the
 * k-truss subgraph of G.
 *
 * @param[out]  C       k-truss subgraph, of type GrB_UINT32
 * @param[in]   G       input graph, not modified
 * @param[in]   k       the truss to find
 * @param[in,out] msg   any error messages.
 *
 * @retval GrB_SUCCESS      if completed successfully (equal or not)
 * @retval GrB_NULL_POINTER if C or C_type is NULL
 * @return Any GraphBLAS errors that may have been encountered
 */
LAGRAPHX_PUBLIC
int LAGraph_KTruss      // compute the k-truss of a graph
(
    // outputs:
    GrB_Matrix *C,      // output k-truss subgraph, C
    // inputs:
    LAGraph_Graph G,    // input graph
    uint32_t k,         // find the k-truss, where k >= 3
    char *msg
) ;

//****************************************************************************
// Connected components
//****************************************************************************

/**
 * Determine connected components in an undirected graph.
 *
 * @param[out] result    array of component identifiers for each vertex (allocated
 *                       by the algorithm, ownership returned to caller).
 * @param[in]  A         the graph (symmetric)
 * @param[in]  sanitize  If true, test to ensure A is symmetric
 * @param[in,out] msg    any error messages.
 *
 * @retval GrB_SUCCESS      if completed successfully
 * @retval GrB_NULL_POINTER if result is NULL
 */
LAGRAPHX_PUBLIC
int LAGraph_cc_lacc (
    GrB_Vector *result,
    GrB_Matrix A,
    bool sanitize,
    char *msg
) ;

//****************************************************************************
// Bellman Ford variants
//****************************************************************************

/**
 * Bellman-Ford single source shortest paths, returning just the shortest path
 * lengths.
 *
 * @param[out]  pd_output    the pointer to the vector of distance (created internally)
 * @param[in]   A            matrix for the graph
 * @param[in]   s            index of the source
 *
 * @retval GrB_SUCCESS        if completed successfully
 * @retval GrB_NULL_POINTER   If pd_output or A is NULL
 * @retval GrB_INVALID_VALUE  if A is not square, s is not a valid vertex index
 * @retval GrB_NO_VALUE       if A has a negative weight cycle
 */
LAGRAPHX_PUBLIC
GrB_Info LAGraph_BF_basic
(
    GrB_Vector *pd_output,
    const GrB_Matrix A,
    const GrB_Index s
) ;

/**
 * Bellman-Ford single source shortest paths, returning just the shortest path
 * lengths.
 *
 * @param[out]  pd_output    the pointer to the vector of distance (created internally)
 * @param[in]   A            matrix for the graph (optional-ish)
 * @param[in]   AT           transpose of A (optional-ish)
 * @param[in]   s            index of the source
 *
 * @retval GrB_SUCCESS        if completed successfully
 * @retval GrB_NULL_POINTER   If pd_output is NULL or both A and AT are NULL
 * @retval GrB_INVALID_VALUE  if A is not square, s is not a valid vertex index
 * @retval GrB_NO_VALUE       if A has a negative weight cycle
 *
 */
LAGRAPHX_PUBLIC
GrB_Info LAGraph_BF_basic_pushpull
(
    GrB_Vector *pd_output,
    const GrB_Matrix A,
    const GrB_Matrix AT,
    const GrB_Index s
) ;

/**
 * Bellman-Ford single source shortest paths, returning just the shortest path
 * lengths.
 *
 * @param[out]  pd_output    the pointer to the vector of distance (created internally)
 * @param[in]   AT           transposed adjacency matrix for the graph
 * @param[in]   s            index of the source
 *
 * @retval GrB_SUCCESS        if completed successfully
 * @retval GrB_NULL_POINTER   If pd_output or AT is NULL
 * @retval GrB_INVALID_VALUE  if A is not square, s is not a valid vertex index
 * @retval GrB_NO_VALUE       if A has a negative weight cycle
 *
 */
LAGRAPHX_PUBLIC
GrB_Info LAGraph_BF_basic_mxv
(
    GrB_Vector *pd_output,      //the pointer to the vector of distance
    const GrB_Matrix AT,        //transposed adjacency matrix for the graph
    const GrB_Index s           //given index of the source
) ;

/**
 * Bellman-Ford single source shortest paths, returning both the path lengths
 * and the shortest-path tree.
 *
 * @param[out]  pd_output    the pointer to the vector of distance (created internally)
 * @param[out]  ppi_output   the pointer to the vector of parent (created internally)
 * @param[out]  ph_output    the pointer to the vector of hops (created internally)
 * @param[in]   A            adjacency matrix for the graph
 * @param[in]   s            index of the source
 *
 * @retval GrB_SUCCESS        if completed successfully
 * @retval GrB_NULL_POINTER   If pd_output, ppi_output, ph_output, or A is NULL
 * @retval GrB_INVALID_VALUE  if A is not square, s is not a valid vertex index
 * @retval GrB_OUT_OF_MEMORY  if allocation fails
 * @retval GrB_NO_VALUE       if A has a negative weight cycle
 *
 */
LAGRAPHX_PUBLIC
GrB_Info LAGraph_BF_full
(
    GrB_Vector *pd_output,
    GrB_Vector *ppi_output,
    GrB_Vector *ph_output,
    const GrB_Matrix A,
    const GrB_Index s
) ;

/**
 * Bellman-Ford single source shortest paths, returning both the path lengths
 * and the shortest-path tree.
 *
 * @param[out]  pd_output    the pointer to the vector of distance (created internally)
 * @param[out]  ppi_output   the pointer to the vector of parent (created internally)
 * @param[out]  ph_output    the pointer to the vector of hops (created internally)
 * @param[in]   A            adjacency matrix for the graph
 * @param[in]   s            index of the source
 *
 * @retval GrB_SUCCESS        if completed successfully
 * @retval GrB_NULL_POINTER   If pd_output, ppi_output, ph_output, or A is NULL
 * @retval GrB_INVALID_VALUE  if A is not square, s is not a valid vertex index
 * @retval GrB_OUT_OF_MEMORY  if allocation fails
 * @retval GrB_NO_VALUE       if A has a negative weight cycle
 *
 */
LAGRAPHX_PUBLIC
GrB_Info LAGraph_BF_full1
(
    GrB_Vector *pd_output,
    GrB_Vector *ppi_output,
    GrB_Vector *ph_output,
    const GrB_Matrix A,
    const GrB_Index s
) ;

/**
 * Bellman-Ford single source shortest paths, returning both the path lengths
 * and the shortest-path tree.
 *
 * @param[out]  pd_output    the pointer to the vector of distance (created internally)
 * @param[out]  ppi_output   the pointer to the vector of parent (created internally)
 * @param[out]  ph_output    the pointer to the vector of hops (created internally)
 * @param[in]   A            adjacency matrix for the graph
 * @param[in]   s            index of the source
 *
 * @retval GrB_SUCCESS        if completed successfully
 * @retval GrB_NULL_POINTER   If pd_output, ppi_output, ph_output, or A is NULL
 * @retval GrB_INVALID_VALUE  if A is not square, s is not a valid vertex index
 * @retval GrB_OUT_OF_MEMORY  if allocation fails
 * @retval GrB_NO_VALUE       if A has a negative weight cycle
 *
 */
LAGRAPHX_PUBLIC
GrB_Info LAGraph_BF_full1a
(
    GrB_Vector *pd_output,
    GrB_Vector *ppi_output,
    GrB_Vector *ph_output,
    const GrB_Matrix A,
    const GrB_Index s
) ;

/**
 * Bellman-Ford single source shortest paths, returning both the path lengths
 * and the shortest-path tree.
 *
 * @param[out]  pd_output    the pointer to the vector of distance (created internally)
 * @param[out]  ppi_output   the pointer to the vector of parent (created internally)
 * @param[out]  ph_output    the pointer to the vector of hops (created internally)
 * @param[in]   A            adjacency matrix for the graph
 * @param[in]   s            index of the source
 *
 * @retval GrB_SUCCESS        if completed successfully
 * @retval GrB_NULL_POINTER   If pd_output, ppi_output, ph_output, or A is NULL
 * @retval GrB_INVALID_VALUE  if A is not square, s is not a valid vertex index
 * @retval GrB_OUT_OF_MEMORY  if allocation fails
 * @retval GrB_NO_VALUE       if A has a negative weight cycle
 *
 */
LAGRAPHX_PUBLIC
GrB_Info LAGraph_BF_full2
(
    GrB_Vector *pd_output,      //the pointer to the vector of distance
    GrB_Vector *ppi_output,     //the pointer to the vector of parent
    GrB_Vector *ph_output,      //the pointer to the vector of hops
    const GrB_Matrix A,         //matrix for the graph
    const GrB_Index s           //given index of the source
) ;

/**
 * Bellman-Ford single source shortest paths, returning both the path lengths
 * and the shortest-path tree.
 *
 * @param[out]  pd_output    the pointer to the vector of distance (created internally)
 * @param[out]  ppi_output   the pointer to the vector of parent (created internally)
 * @param[out]  ph_output    the pointer to the vector of hops (created internally)
 * @param[in]   AT           transpose of the adjacency matrix for the graph
 * @param[in]   s            index of the source
 *
 * @retval GrB_SUCCESS        if completed successfully
 * @retval GrB_NULL_POINTER   If pd_output, ppi_output, ph_output, or AT is NULL
 * @retval GrB_INVALID_VALUE  if A is not square, s is not a valid vertex index
 * @retval GrB_OUT_OF_MEMORY  if allocation fails
 * @retval GrB_NO_VALUE       if A has a negative weight cycle
 *
 */
LAGRAPHX_PUBLIC
GrB_Info LAGraph_BF_full_mxv
(
    GrB_Vector *pd_output,
    GrB_Vector *ppi_output,
    GrB_Vector *ph_output,
    const GrB_Matrix AT,
    const GrB_Index s
) ;

/**
 * Bellman-Ford single source shortest paths, returning both the path lengths
 * and the shortest-path tree (integer weights).
 *
 * @param[out]  pd       pointer to distance vector d, d(k) = shortest distance
 *                       between s and k if k is reachable from s
 * @param[out]  ppi      pointer to parent index vector pi, pi(k) = parent of
 *                       node k in the shortest path tree
 * @param[in]   s        index of the source
 * @param[in]   n        number of nodes
 * @param[in]   nz       number of edges
 * @param[in]   Ilist    row index vector (size n)
 * @param[in]   J        column index vector (size nz)
 * @param[in]   W        weight vector (size nz), W(i) = weight of edge 
 *                       (Ilist(i),J(i))
 *
 * @retval GrB_SUCCESS        if completed successfully
 * @retval GrB_NULL_POINTER   If pd, ppi, Ilist, J, or W is NULL
 * @retval GrB_INVALID_VALUE  if s is not a valid vertex index
 * @retval GrB_OUT_OF_MEMORY  if allocation fails.
 * @retval GrB_NO_VALUE       if A has a negative weight cycle
 *
 */
LAGRAPHX_PUBLIC
GrB_Info LAGraph_BF_pure_c
(
    int32_t **pd,

    int64_t **ppi,

    const int64_t s,
    const int64_t n,
    const int64_t nz,
    const int64_t *Ilist,
    const int64_t *J,
    const int32_t *W
) ;

/**
 * Bellman-Ford single source shortest paths, returning both the path lengths
 * and the shortest-path tree (double weights).
 *
 * @param[out]  pd       pointer to distance vector d, d(k) = shortest distance
 *                       between s and k if k is reachable from s
 * @param[out]  ppi      pointer to parent index vector pi, pi(k) = parent of
 *                       node k in the shortest path tree
 * @param[in]   s        index of the source
 * @param[in]   n        number of nodes
 * @param[in]   nz       number of edges
 * @param[in]   Ilist    row index vector (size n)
 * @param[in]   J        column index vector (size nz)
 * @param[in]   W        weight vector (size nz), W(i) = weight of edge
 *                       (Ilist(i),J(i))
 *
 * @retval GrB_SUCCESS        if completed successfully
 * @retval GrB_NULL_POINTER   If pd, ppi, Ilist, J, or W is NULL
 * @retval GrB_INVALID_VALUE  if s is not a valid vertex index
 * @retval GrB_OUT_OF_MEMORY  if allocation fails.
 * @retval GrB_NO_VALUE       if A has a negative weight cycle
 *
 */
LAGRAPHX_PUBLIC
GrB_Info LAGraph_BF_pure_c_double
(
    double **pd,

    int64_t **ppi,

    const int64_t s,
    const int64_t n,
    const int64_t nz,
    const int64_t *Ilist,
    const int64_t *J,
    const double  *W
) ;

//****************************************************************************
/**
 * Community detection using label propagation algorithm
 *
 * @param[out]  CDLP_handle  community vector
 * @param[in]   A            adjacency matrix for the graph
 * @param[in]   symmetric    denote whether the matrix is symmetric
 * @param[in]   sanitize     if true, verify that A is binary
 * @param[in]   itermax      max number of iterations (0 computes nothing)
 * @param[out]  t            array of two doubles allocated by caller:
 *                           [0]=sanitize time, [1]=cdlp time in seconds
 * @param[in,out] msg        any error messages.
 *
 * @retval GrB_SUCCESS        if completed successfully
 * @retval GrB_NULL_POINTER   If t or CDLP_handle is NULL
 * @retval GrB_INVALID_OBJECT If A is not stored in CSR format
 * @retval GrB_OUT_OF_MEMORY  if allocation fails.
 * @retval GrB_NO_VALUE       if A has a negative weight cycle
 */
LAGRAPHX_PUBLIC
int LAGraph_cdlp
(
    GrB_Vector *CDLP_handle,
    const GrB_Matrix A,
    bool symmetric,
    bool sanitize,
    int itermax,
    double *t,
    char *msg
) ;

//****************************************************************************
/**
 * Sparse deep neural network inference. Performs ReLU inference using input
 * feature vectors Y0.
 *
 * @param[out]  Yhandle      Y, created on output
 * @param[in]   W            W [0..nlayers-1], each nneurons-by-nneurons
 * @param[in]   Bias         Bias [0..nlayers-1], diagonal nneurons-by-nneurons
 * @param[in]   nlayers      number of layers
 * @param[in]   Y0           input features: nfeatures-by-nneurons
 *
 * @retval GrB_SUCCESS         if completed successfully
 * @retval GrB_NOT_IMPLEMENTED vanilla version has not been implemented yet
 * @retval GrB_NULL_POINTER    If Yhandle, W, Bias, or Y0 is NULL
 * @retval GrB_DOMAIN_MISMATCH if type of Y0 is not FP32 or FP64, or the types of
 *                             W or Bias arent the same as Y0
 */
LAGRAPHX_PUBLIC
GrB_Info LAGraph_dnn
(
    // output
    GrB_Matrix *Yhandle,
    // input: not modified
    GrB_Matrix *W,
    GrB_Matrix *Bias,
    int nlayers,
    GrB_Matrix Y0
) ;

//****************************************************************************
/**
 * Compute all-pairs shortest paths using Floyd-Warshall method
 *
 * @param[in]   G       input graph, with edge weights
 * @param[out]  D       output graph, created on output
 * @param[out]  D_type  type of scalar stored in D (see source for explanation)
 *
 * @retval GrB_SUCCESS         if completed successfully
 * @retval GrB_NOT_IMPLEMENTED vanilla version has not been implemented yet
 * @retval GrB_NULL_POINTER    If D or D_type is NULL
 * @retval GrB_INVALID_VALUE   If G is not square
 */
LAGRAPHX_PUBLIC
GrB_Info LAGraph_FW
(
    const GrB_Matrix G,
    GrB_Matrix *D,
    GrB_Type   *D_type
) ;

//****************************************************************************
/**
 * Compute the local clustering coefficient for all nodes in a graph.
 *
 * @param[out]  LCC_handle   output vector holding coefficients
 * @param[in]   A            adjacency matrix for the graph
 * @param[in]   symmetric    denote whether the matrix is symmetric
 * @param[in]   sanitize     if true, verify that A is binary
 * @param[out]  t            array of two doubles
 *                           [0]=sanitize time, [1]=lcc time in seconds
 * @param[in,out] msg        any error messages.
 *
 * @retval GrB_SUCCESS        if completed successfully
 * @retval GrB_NOT_IMPLEMENTED vanilla version has not been implemented yet
 * @retval GrB_NULL_POINTER   If LCC_handle or LCC_type is NULL
 * @retval GrB_INVALID_VALUE  If A is not stored in CSR format
 */
LAGRAPHX_PUBLIC
int LAGraph_lcc            // compute lcc for all nodes in A
(
    GrB_Vector *LCC_handle,     // output vector
    const GrB_Matrix A,         // input matrix
    bool symmetric,             // if true, the matrix is symmetric
    bool sanitize,              // if true, ensure A is binary
    double t [2],               // t [0] = sanitize time, t [1] = lcc time,
                                // in seconds
    char *msg
) ;

//****************************************************************************

LAGRAPHX_PUBLIC
int LAGraph_msf
(
    GrB_Matrix *result, // output: an unsymmetrical matrix, the spanning forest
    GrB_Matrix A,       // input matrix
    bool sanitize,      // if true, ensure A is symmetric
    char *msg
) ;

//****************************************************************************

LAGRAPHX_PUBLIC
int LAGraph_scc (
    GrB_Vector *result,     // output: array of component identifiers
    GrB_Matrix A,           // input matrix
    char *msg
) ;

//****************************************************************************
LAGRAPHX_PUBLIC
int LAGraph_VertexCentrality_Triangle       // vertex triangle-centrality
(
    // outputs:
    GrB_Vector *centrality,     // centrality(i): triangle centrality of i
    uint64_t *ntriangles,       // # of triangles in the graph
    // inputs:
    int method,                 // 0, 1, 2, or 3
    LAGraph_Graph G,            // input graph
    char *msg
) ;

//****************************************************************************
LAGRAPHX_PUBLIC
int LAGraph_MaximalIndependentSet       // maximal independent set
(
    // outputs:
    GrB_Vector *mis,            // mis(i) = true if i is in the set
    // inputs:
    LAGraph_Graph G,            // input graph
    uint64_t seed,              // random number seed
    GrB_Vector ignore_node,     // if NULL, no nodes are ignored.  Otherwise
                                // ignore_node(i) = true if node i is to be
                                // ignored, and not treated as a candidate
                                // added to maximal independent set.
    char *msg
) ;

LAGRAPHX_PUBLIC
int LG_CC_FastSV5           // SuiteSparse:GraphBLAS method, with GxB extensions
(
    // output
    GrB_Vector *component,  // output: array of component identifiers
    // inputs
    LAGraph_Graph G,        // input graph, modified then restored
    char *msg
) ;

//------------------------------------------------------------------------------
// kcore algorithms
//------------------------------------------------------------------------------

LAGRAPHX_PUBLIC
int LAGraph_KCore_All
(
    // outputs:
    GrB_Vector *decomp,     // kcore decomposition
    uint64_t *kmax,
    // inputs:
    LAGraph_Graph G,            // input graph
    char *msg
) ;

LAGRAPHX_PUBLIC
int LAGraph_KCore
(
    // outputs:
    GrB_Vector *decomp,     // kcore decomposition
    // inputs:
    LAGraph_Graph G,        // input graph
    uint64_t k,             //k level to compare to
    char *msg
) ;

LAGRAPHX_PUBLIC
int LAGraph_KCore_Decompose
(
    // outputs:
    GrB_Matrix *D,              // kcore decomposition
    // inputs:
    LAGraph_Graph G,            // input graph
    GrB_Vector decomp,         // input decomposition matrix
    uint64_t k,
    char *msg
) ;

//------------------------------------------------------------------------------
// counting graphlets
//------------------------------------------------------------------------------

LAGRAPHX_PUBLIC
int LAGraph_FastGraphletTransform
(
    // outputs:
    GrB_Matrix *F_net,  // 16-by-n matrix of graphlet counts
    // inputs:
    LAGraph_Graph G,
    bool compute_d_15,  // probably this makes most sense
    char *msg
) ;

LAGRAPHX_PUBLIC
int LAGraph_SquareClustering
(
    // outputs:
    GrB_Vector *square_clustering,
    // inputs:
    LAGraph_Graph G,
    char *msg
) ;

//------------------------------------------------------------------------------
// a simple example of an algorithm
//------------------------------------------------------------------------------

LAGRAPHX_PUBLIC
int LAGraph_HelloWorld // a simple algorithm, just for illustration
(
    // output
    GrB_Matrix *Yhandle,    // Y, created on output
    // input: not modified
    LAGraph_Graph G,
    char *msg
) ;

#if defined ( __cplusplus )
}
#endif

#endif
