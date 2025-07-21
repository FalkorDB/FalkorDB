/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
*/

#pragma once

#include <pthread.h>

#include "rax.h"
#include "entities/node.h"
#include "entities/edge.h"
#include "../redismodule.h"
#include "graph_statistics.h"
#include "rg_matrix/rg_matrix.h"
#include "../util/datablock/datablock.h"
#include "../util/datablock/datablock_iterator.h"
#include "../../deps/GraphBLAS/Include/GraphBLAS.h"

#define GRAPH_DEFAULT_RELATION_TYPE_CAP 16  // default number of different relationship types a graph can hold before resizing.
#define GRAPH_DEFAULT_LABEL_CAP 16          // default number of different labels a graph can hold before resizing.
#define GRAPH_NO_LABEL -1                   // labels are numbered [0-N], -1 represents no label.
#define GRAPH_UNKNOWN_LABEL -2              // labels are numbered [0-N], -2 represents an unknown relation.
#define GRAPH_NO_RELATION -1                // relations are numbered [0-N], -1 represents no relation.
#define GRAPH_UNKNOWN_RELATION -2           // relations are numbered [0-N], -2 represents an unknown relation.

typedef enum {
	GRAPH_EDGE_DIR_INCOMING,
	GRAPH_EDGE_DIR_OUTGOING,
	GRAPH_EDGE_DIR_BOTH,
} GRAPH_EDGE_DIR;

typedef enum {
	SYNC_POLICY_UNKNOWN,
	SYNC_POLICY_FLUSH_RESIZE,
	SYNC_POLICY_RESIZE,
	SYNC_POLICY_NOP,
} MATRIX_POLICY;

// forward declaration of Graph struct
typedef struct Graph Graph;
// typedef for synchronization function pointer
typedef void (*SyncMatrixFunc)(const Graph *, RG_Matrix);

struct Graph {
	int reserved_node_count;           // number of nodes not commited yet
	DataBlock *nodes;                  // graph nodes stored in blocks
	DataBlock *edges;                  // graph edges stored in blocks
	RG_Matrix adjacency_matrix;        // adjacency matrix, holds all graph connections
	RG_Matrix *labels;                 // label matrices
	RG_Matrix node_labels;             // mapping of all node IDs to all labels possessed by each node
	RG_Matrix *relations;              // relation matrices
	RG_Matrix _zero_matrix;            // zero matrix
	pthread_rwlock_t _rwlock;          // read-write lock scoped to this specific graph
	bool _writelocked;                 // true if the read-write lock was acquired by a writer
	SyncMatrixFunc SynchronizeMatrix;  // function pointer to matrix synchronization routine
	GraphStatistics stats;             // graph related statistics
};

// graph synchronization functions
// the graph is initialized with a read-write lock allowing
// concurrent access from one writer or N readers
// acquire a lock that does not restrict access from additional reader threads
void Graph_AcquireReadLock
(
	Graph *g
);

// acquire a lock for exclusive access to this graph's data
void Graph_AcquireWriteLock
(
	Graph *g
);

// release the held lock
void Graph_ReleaseLock
(
	Graph *g
);

// synchronize and resize all matrices in graph
void Graph_ApplyAllPending
(
	Graph *g,           // graph to sync
	bool force_flush    // force sync of delta matrices
);

// Retrieve graph matrix synchronization policy
MATRIX_POLICY Graph_GetMatrixPolicy
(
	const Graph *g
);

// choose the current matrix synchronization policy
MATRIX_POLICY Graph_SetMatrixPolicy
(
	Graph *g,
	MATRIX_POLICY policy
);

// checks to see if graph has pending operations
bool Graph_Pending
(
	const Graph *g
);

// create a new graph
Graph *Graph_New
(
	size_t node_cap,    // Allocation size for node datablocks and matrix dimensions.
	size_t edge_cap     // Allocation size for edge datablocks.
);

// creates a new label matrix, returns id given to label
LabelID Graph_AddLabel
(
	Graph *g
);

// adds a label from the graph
void Graph_RemoveLabel
(
	Graph *g,
	LabelID label_id
);

// label node with each label in 'lbls'
void Graph_LabelNode
(
	Graph *g,       // graph to operate on
	NodeID id,      // node ID to update
	LabelID *lbls,  // set to labels to associate with node
	uint lbl_count  // number of labels
);

// dissociates each label in 'lbls' from given node
void Graph_RemoveNodeLabels
(
	Graph *g,       // graph to operate against
	NodeID id,      // node ID to update
	LabelID *lbls,  // set of labels to remove
	uint lbl_count  // number of labels to remove
);

// return true if node is labeled as 'l'
bool Graph_IsNodeLabeled
(
	Graph *g,   // graph to operate on
	NodeID id,  // node ID to inspect
	LabelID l   // label to check for
);

// creates a new relation matrix, returns id given to relation
RelationID Graph_AddRelationType
(
	Graph *g
);

// removes a relation from the graph
void Graph_RemoveRelation
(
	Graph *g,
	int relation_id
);

// make sure graph can hold an additional N nodes
void Graph_AllocateNodes
(
	Graph *g,               // graph for which nodes will be added
	size_t n                // number of nodes to create
);

// make sure graph can hold an additional N edges
void Graph_AllocateEdges
(
	Graph *g,               // graph for which nodes will be added
	size_t n                // number of edges to create
);

void Graph_ResetReservedNode
(
	Graph *g
);

Node Graph_ReserveNode
(
	Graph *g  // graph for which nodes will be added
);

// Create a single node and labels it accordingly.
// Return newly created node.
void Graph_CreateNode
(
	Graph *g,
	Node *n,
	LabelID *labels,
	uint label_count
);

// connects source node to destination node
// returns 1 if connection is formed, 0 otherwise
void Graph_CreateEdge
(
	Graph *g,           // graph on which to operate
	NodeID src,         // source node ID
	NodeID dest,        // destination node ID
	RelationID r,       // edge type
	Edge *e
);

// deletes nodes from the graph
void Graph_DeleteNodes
(
	Graph *g,       // graph to delete nodes from
	Node *nodes,    // nodes to delete
	uint64_t count  // number of nodes
);

// removes edges from Graph and updates graph relevent matrices
void Graph_DeleteEdges
(
	Graph *g,
	Edge *edges,
	uint64_t n
);

// returns true if the given entity has been deleted
bool Graph_EntityIsDeleted
(
	const GraphEntity *e
);

// all graph matrices are required to be squared NXN
// where N is Graph_RequiredMatrixDim
size_t Graph_RequiredMatrixDim
(
	const Graph *g
);

// retrieves a node iterator which can be used to access
// every node in the graph
DataBlockIterator *Graph_ScanNodes
(
	const Graph *g
);

// retrieves an edge iterator which can be used to access
// every edge in the graph
DataBlockIterator *Graph_ScanEdges
(
	const Graph *g
);

// returns number of nodes in the graph
size_t Graph_NodeCount
(
	const Graph *g
);

// returns number of deleted nodes in the graph
uint Graph_DeletedNodeCount
(
	const Graph *g
);

// returns number of existing and deleted nodes in the graph
size_t Graph_UncompactedNodeCount
(
	const Graph *g
);

// returns number of nodes with given label
uint64_t Graph_LabeledNodeCount
(
	const Graph *g,
	int label
);

// returns number of edges in the graph
size_t Graph_EdgeCount
(
	const Graph *g
);

// returns number of edges of a specific relation type
uint64_t Graph_RelationEdgeCount
(
	const Graph *g,
	int relation_idx
);

// returns number of deleted edges in the graph
uint Graph_DeletedEdgeCount
(
	const Graph *g
);

// returns number of different edge types
int Graph_RelationTypeCount
(
	const Graph *g
);

// returns number of different node types
int Graph_LabelTypeCount
(
	const Graph *g
);

// returns true if relationship matrix 'r' contains multi-edge entries
// false otherwise
bool Graph_RelationshipContainsMultiEdge
(
	const Graph *g,  // Graph containing matrix to inspect
	RelationID r,    // Relationship ID
	bool transpose   // false for R, true for transpose R
);

// retrieves node with given id from graph,
// returns NULL if node wasn't found
bool Graph_GetNode
(
	const Graph *g,
	NodeID id,
	Node *n
);

// retrieves edge with given id from graph,
// returns NULL if edge wasn't found
bool Graph_GetEdge
(
	const Graph *g,
	EdgeID id,
	Edge *e
);

// retrieves edge relation type
// returns GRAPH_NO_RELATION if edge has no relation type
RelationID Graph_GetEdgeRelation
(
	const Graph *g,
	Edge *e
);

// retrieves edges connecting source to destination,
// relation is optional, pass GRAPH_NO_RELATION if you do not care
// about edge type
void Graph_GetEdgesConnectingNodes
(
	const Graph *g,     // Graph to get edges from.
	NodeID srcID,       // Source node of edge
	NodeID destID,      // Destination node of edge
	RelationID r,       // Edge type.
	Edge **edges        // array_t of edges connecting src to dest of type r.
);

// get node edges
void Graph_GetNodeEdges
(
	const Graph *g,         // graph to get edges from
	const Node *n,          // node to extract edges from
	GRAPH_EDGE_DIR dir,     // edge direction
	RelationID edgeType,    // relation type
	Edge **edges            // array_t incoming/outgoing edges
);

// returns node incoming/outgoing degree
uint64_t Graph_GetNodeDegree
(
	const Graph *g,      // graph to inquery
	const Node *n,       // node to get degree of
	GRAPH_EDGE_DIR dir,  // incoming/outgoing/both
	RelationID edgeType  // relation type
);

// populate array of node's label IDs, return number of labels on node.
uint Graph_GetNodeLabels
(
	const Graph *g,         // graph the node belongs to
	const Node *n,          // node to extract labels from
	LabelID *labels,        // array to populate with labels
	uint label_count        // size of labels array
);

// retrieves the adjacency matrix
// matrix is resized if its size doesn't match graph's node count
RG_Matrix Graph_GetAdjacencyMatrix
(
	const Graph *g,
	bool transposed
);

// retrieves a label matrix
// matrix is resized if its size doesn't match graph's node count
RG_Matrix Graph_GetLabelMatrix
(
	const Graph *g,     // graph from which to get adjacency matrix
	int label           // label described by matrix
);

// retrieves a typed adjacency matrix
// matrix is resized if its size doesn't match graph's node count
RG_Matrix Graph_GetRelationMatrix
(
	const Graph *g,     // graph from which to get adjacency matrix
	int relation,       // relation described by matrix
	bool transposed
);

// retrieves the node-label mapping matrix,
// matrix is resized if its size doesn't match graph's node count.
RG_Matrix Graph_GetNodeLabelMatrix
(
	const Graph *g
);

// retrieves the zero matrix
// the function will resize it to match all other
// internal matrices, caller mustn't modify it in any way
RG_Matrix Graph_GetZeroMatrix
(
	const Graph *g
);

RG_Matrix Graph_GetLabelRGMatrix
(
	const Graph *g,
	int label_idx
);

// free partial graph
void Graph_PartialFree
(
	Graph *g
);

// free graph
void Graph_Free
(
	Graph *g
);
