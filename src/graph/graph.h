/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include <pthread.h>

#include "rax.h"
#include "GraphBLAS.h"
#include "entities/node.h"
#include "entities/edge.h"
#include "../redismodule.h"
#include "tensor/tensor.h"
#include "delta_matrix/delta_matrix.h"
#include "../util/datablock/datablock.h"
#include "../util/datablock/datablock_iterator.h"

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
	SYNC_POLICY_FLUSH_RESIZE,
	SYNC_POLICY_RESIZE,
	SYNC_POLICY_NOP,
} MATRIX_POLICY;

// forward declaration of Graph struct
typedef struct Graph Graph;
// typedef for synchronization function pointer
typedef void (*SyncMatrixFunc)(const Graph *, Delta_Matrix, GrB_Index, GrB_Index);


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

// retrieve graph matrix synchronization policy
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

// synchronize and resize all matrices in graph
void Graph_ApplyAllPending
(
	Graph *g,           // graph to sync
	bool force_flush    // force sync of delta matrices
);

// checks to see if graph has pending operations
bool Graph_Pending
(
	const Graph *g
);

// create a new graph
Graph *Graph_New
(
	size_t node_cap,  // allocation size for node datablocks and matrix dimensions
	size_t edge_cap   // allocation size for edge datablocks
);

// creates a new label matrix, returns id given to label
LabelID Graph_AddLabel
(
	Graph *g
);

// creates a new relation matrix, returns id given to relation
RelationID Graph_AddRelationType
(
	Graph *g
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

// reset graph's reserved node count
void Graph_ResetReservedNode
(
	Graph *g  // graph
);

// reserve a node
Node Graph_ReserveNode
(
	Graph *g  // graph for which nodes will be added
);

// create a single node and labels it accordingly.
// Return newly created node.
void Graph_CreateNode
(
	Graph *g,         // graph
	Node *n,          // node to create
	LabelID *labels,  // node's labels
	uint label_count  // number of labels
);

// label node with each label in 'lbls'
void Graph_LabelNode
(
	Graph *g,       // graph to operate on
	NodeID id,      // node ID to update
	LabelID *lbls,  // set to labels to associate with node
	uint lbl_count  // number of labels
);

// return true if node is labeled as 'l'
bool Graph_IsNodeLabeled
(
	Graph *g,   // graph to operate on
	NodeID id,  // node ID to inspect
	LabelID l   // label to check for
);

// dissociates each label in 'lbls' from given node
void Graph_RemoveNodeLabels
(
	Graph *g,       // graph to operate against
	NodeID id,      // node ID to update
	LabelID *lbls,  // set of labels to remove
	uint lbl_count  // number of labels to remove
);

// update ADJ and relation matrices with a new entry
// ADJ[src, dest] = true & R[src, dest] = edge_id
void Graph_FormConnection
(
	Graph *g,        // graph
	NodeID src,      // src node id
	NodeID dest,     // dest node id
	EdgeID edge_id,  // edge id
	RelationID r     // relation id
);

// connects source node to destination node
void Graph_CreateEdge
(
	Graph *g,      // graph on which to operate
	NodeID src,    // source node ID
	NodeID dest,   // destination node ID
	RelationID r,  // edge type
	Edge *e        // [output] created edge
);

// create multiple edges
void Graph_CreateEdges
(
	Graph *g,       // graph on which to operate
	RelationID r,   // relationship type
	Edge **edges,   // edges to create
	int edge_count  // number of edges to create 
);

// deletes nodes from the graph
void Graph_DeleteNodes
(
	Graph *g,       // graph to delete nodes from
	Node *nodes,    // nodes to delete
	uint64_t count  // number of nodes
);

// deletes edges from the graph
void Graph_DeleteEdges
(
	Graph *g,     // graph to delete edges from
	Edge *edges,  // edges to delete
	uint64_t n    // number of edges
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

// return number of nodes graph can contain
uint64_t Graph_NodeCap
(
	const Graph *g
);

// returns number of nodes in the graph
uint64_t Graph_NodeCount
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
uint64_t Graph_EdgeCount
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
	const Graph *g  // graph
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
	RelationID r     // Relationship ID
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

typedef struct EdgeIterator EdgeIterator;
struct EdgeIterator {
	char private[192];
};


// retrieves edges connecting source to destination,
// relation is optional, pass GRAPH_NO_RELATION if you do not care
// about edge type
void Graph_EdgeIteratorInit
(
	const Graph *g,    // Graph to iterate over
	EdgeIterator *it,  // Iterator to initialize
	NodeID srcID,      // Source node of edge
	NodeID destID,     // Destination node of edge
	RelationID r       // Edge type.
);

// returns the next edge in the iterator
bool EdgeIterator_Next
(
	EdgeIterator *it,  // Iterator to extract edge from
	Edge *e            // Edge to populate
);

typedef struct NodeEdgeIterator NodeEdgeIterator;
struct NodeEdgeIterator {
	char private[632];
};

void Graph_NodeEdgeIteratorInit
(
	const Graph *g,    // Graph to iterate over
	NodeEdgeIterator *it,  // Iterator to initialize
	NodeID nodeID,      // Source node of edge
	GRAPH_EDGE_DIR dir, // Direction of edge
	RelationID r        // Edge type.
);

// returns the next edge in the iterator
bool NodeEdgeIterator_Next
(
	NodeEdgeIterator *it,  // Iterator to extract edge from
	Edge *e                // Edge to populate
);

// returns node incoming/outgoing degree
uint64_t Graph_GetNodeDegree
(
	const Graph *g,      // graph to inquery
	const Node *n,       // node to get degree of
	GRAPH_EDGE_DIR dir,  // incoming/outgoing/both
	RelationID r         // relation type
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
Delta_Matrix Graph_GetAdjacencyMatrix
(
	const Graph *g
);

// retrieves a label matrix
// matrix is resized if its size doesn't match graph's node count
Delta_Matrix Graph_GetLabelMatrix
(
	const Graph *g,     // graph from which to get adjacency matrix
	int label           // label described by matrix
);

// retrieves a typed adjacency matrix
// matrix is resized if its size doesn't match graph's node count
Tensor Graph_GetRelationMatrix
(
	const Graph *g,           // graph from which to get adjacency matrix
	RelationID relation_idx   // relation described by matrix
);

// retrieves the node-label mapping matrix,
// matrix is resized if its size doesn't match graph's node count.
Delta_Matrix Graph_GetNodeLabelMatrix
(
	const Graph *g
);

// retrieves the zero matrix
// the function will resize it to match all other
// internal matrices, caller mustn't modify it in any way
Delta_Matrix Graph_GetZeroMatrix
(
	const Graph *g
);

// sets a node in the graph
void Graph_SetNode
(
	Graph *g,               // graph to add node to
	NodeID id,              // node ID
	LabelID *labels,        // node labels
	uint label_count,       // label count
	Node *n                 // pointer to node
);

// sets graph's node labels matrix
void Graph_SetNodeLabels
(
	Graph *g
);

// optimized version of Graph_FormConnection
void Graph_OptimizedFormConnections
(
	Graph *g,
	RelationID r,                  // relation id
	const NodeID *restrict srcs,   // src node id
	const NodeID *restrict dests,  // dest node id
	const EdgeID *restrict ids,    // edge id
	uint64_t n,                    // number of entries
	bool multi_edge                // multi edge batch
);

// allocate edge attribute-set
void Graph_AllocEdgeAttributes
(
	Graph *g,
	EdgeID edge_id,
	Edge *e
);

// marks a node ID as deleted
void Graph_MarkNodeDeleted
(
	Graph *g,               // graph from which to mark node as deleted
	NodeID ID               // node ID
);

// marks an edge ID as deleted
void Graph_MarkEdgeDeleted
(
	Graph *g,               // graph from which to mark edge as deleted
	EdgeID ID               // edge ID
);

// returns the graph deleted nodes list
uint64_t *Graph_GetDeletedNodesList
(
	Graph *g
);

// returns the graph deleted edges list
uint64_t *Graph_GetDeletedEdgesList
(
	Graph *g
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

