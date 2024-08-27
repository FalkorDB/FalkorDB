/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "entities/node.h"
#include "entities/edge.h"
#include "delta_matrix/delta_matrix.h"
#include "delta_matrix/delta_matrix_iter.h"

// Checks if X represents edge ID
#define SINGLE_EDGE(x) !((x) & MSB_MASK)

typedef struct MultiEdgeIterator MultiEdgeIterator;
typedef bool (*IterFunc)(MultiEdgeIterator *, NodeID *, NodeID *, EdgeID *);

// A relation type is defined via two matrices:
//   1. uint64 connecting source nodes to destination nodes
//   2. multi-edge, O[meid,e] = true
typedef struct {
	Delta_Matrix R;      // relation matrix
	Delta_Matrix E;      // multi-edge matrix
	uint64_t row_id;     // multi-edge id
	uint64_t *freelist;  // multi-edge deleted ids
} MultiEdgeMatrix;

// Multi edge matrix iterator
struct MultiEdgeIterator {
    MultiEdgeMatrix *M;          // multi-edge matrix
    Delta_MatrixTupleIter r_it;  // relation matrix iterator
    Delta_MatrixTupleIter e_it;  // multi-edge matrix iterator
    EdgeID edge_id;              // edge id
    NodeID src;                  // src id
    NodeID dest;                 // dest id
    IterFunc iter_func;          // iteration strategy
};

// attach iterator by source range
void MultiEdgeIterator_AttachSourceRange
(
    MultiEdgeIterator *it,  // multi-edge iterator
    MultiEdgeMatrix *M,     // multi-edge matrix
    NodeID min_src_id,      // minimum source id
    NodeID max_src_id,      // maximum source id
    bool transposed         // transpose to traverse incoming edges
);

// attach iterator by source and destination
void MultiEdgeIterator_AttachSourceDest
(
    MultiEdgeIterator *it,  // multi-edge iterator
    MultiEdgeMatrix *M,     // multi-edge matrix
    NodeID src_id,          // source id
    NodeID dest_id          // dest id
);

// get the next edge
bool MultiEdgeIterator_next
(
    MultiEdgeIterator *it,  // multi-edge iterator
    NodeID *src,            // [out] source id
    NodeID *dest,           // [out] dest id
    EdgeID *edge_id         // [out] edge id
);

// whether iterator is attached to a matrix
bool MultiEdgeIterator_is_attached
(
    MultiEdgeIterator *it,  // multi-edge iterator
    MultiEdgeMatrix *M      // multi-edge matrix
);

// init new multi-edge matrix
void MultiEdgeMatrix_init
(
    MultiEdgeMatrix *M,  // multi-edge matrix
    GrB_Index nrows,     // # of rows
    GrB_Index ncols,     // # of columns
    GrB_Index me_nrows,  // # of multi-edge rows
    GrB_Index me_ncols   // # of multi-edge columns
);

// create edge between src and dest
void MultiEdgeMatrix_FormConnection
(
    MultiEdgeMatrix *M,  // multi-edge matrix
    NodeID src,          // source id
	NodeID dest,         // dest id
	EdgeID edge_id       // edge id
);

// bulk create edges
void MultiEdgeMatrix_FormConnections
(
	MultiEdgeMatrix *M,  // multi-edge matrix
	Edge **edges         // edges to create
);

// free multi-edge matrix
void MultiEdgeMatrix_free
(
    MultiEdgeMatrix *M  // multi-edge matrix
);