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

typedef struct RelationIterator RelationIterator;
typedef bool (*IterFunc)(RelationIterator *, NodeID *, NodeID *, EdgeID *);

typedef struct RelationMatrix *RelationMatrix;

// A relation type is defined via two matrices:
//   1. uint64 connecting source nodes to destination nodes
//   2. multi-edge, E[i,j] = true where R[s, d] = i 
//      means multiple edges exist between s and d
struct RelationMatrix {
	Delta_Matrix R;      // relation matrix
	Delta_Matrix E;      // multi-edge matrix
	uint64_t row_id;     // multi-edge id
	uint64_t *freelist;  // multi-edge deleted row ids
};

// Relationship matrix iterator
struct RelationIterator {
	RelationMatrix M;            // relation matrix
	Delta_MatrixTupleIter r_it;  // relation matrix iterator
	Delta_MatrixTupleIter e_it;  // multi-edge iterator
	EdgeID edge_id;              // edge id
	NodeID src;                  // src id
	NodeID dest;                 // dest id
	IterFunc iter_func;          // iteration strategy
};

// attach iterator by source range
void RelationIterator_AttachSourceRange
(
	RelationIterator *it,   // relation iterator
	RelationMatrix M,       // relation matrix
	NodeID min_src_id,      // minimum source id
	NodeID max_src_id,      // maximum source id
	bool transposed         // transpose to traverse incoming edges
);

// attach iterator by source and destination
void RelationIterator_AttachSourceDest
(
	RelationIterator *it,   // relation iterator
	RelationMatrix M,       // relation matrix
	NodeID src_id,          // source id
	NodeID dest_id          // dest id
);

// advance iterator
bool RelationIterator_next
(
	RelationIterator *it,   // relation iterator
	NodeID *src,            // [optional out] source id
	NodeID *dest,           // [optional out] dest id
	EdgeID *edge_id         // [optional out] edge id
);

// whether iterator is attached to a matrix
bool RelationIterator_is_attached
(
	const RelationIterator *it,   // relation iterator
	const RelationMatrix M        // relation matrix
);

// init new relation matrix
RelationMatrix RelationMatrix_new
(
	GrB_Index nrows,  // # of rows
	GrB_Index ncols   // # of columns
);

// create edge between src and dest
void RelationMatrix_FormConnection
(
	RelationMatrix M,    // relation matrix
	NodeID src,          // source id
	NodeID dest,         // dest id
	EdgeID edge_id       // edge id
);

// bulk create edges
void RelationMatrix_FormConnections
(
	RelationMatrix M,    // relation matrix
	const Edge **edges   // edges to create
);

// checks to see if matrix has pending operations
bool RelationMatrix_pending
(
	RelationMatrix M    // relation matrix
);

// free relation matrix
void RelationMatrix_free
(
	RelationMatrix *M   // relation matrix
);