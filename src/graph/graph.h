/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include <pthread.h>

#include "rax.h"
#include "GraphBLAS.h"
#include "tensor/tensor.h"
#include "entities/node.h"
#include "entities/edge.h"
#include "../redismodule.h"
#include "graph_statistics.h"
#include "../commands/cmd_memory.h"
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
	SYNC_POLICY_UNKNOWN,
	SYNC_POLICY_FLUSH_RESIZE,
	SYNC_POLICY_RESIZE,
	SYNC_POLICY_NOP,
} MATRIX_POLICY;

// forward declaration of Graph struct
typedef struct Graph Graph;
// typedef for synchronization function pointer
typedef void (*SyncMatrixFunc)(const Graph *, Delta_Matrix, GrB_Index, GrB_Index);

struct Graph {
	int reserved_node_count;           // number of nodes not commited yet
	DataBlock *nodes;                  // graph nodes stored in blocks
	DataBlock *edges;                  // graph edges stored in blocks
	Delta_Matrix adjacency_matrix;     // adjacency matrix, holds all graph connections
	Delta_Matrix *labels;              // label matrices
	Delta_Matrix node_labels;          // mapping of all node IDs to all labels possessed by each node
	Tensor *relations;                 // relation matrices
	Delta_Matrix _zero_matrix;         // zero matrix
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

// acquire the graph write lock with a timeout
// attempts to acquire the write lock on the given graph
// if the lock is not acquired immediately the function will block until either
// the lock becomes available or the timeout elapses
//
// returns:
// - 0 on success (lock acquired)
// - ETIMEDOUT if the timeout expired before acquiring the lock
// - EBUSY if called with timeout_ms == 0 and the lock could not be acquired
// - other nonzero error codes may be returned for unexpected failures
int Graph_TimeAcquireWriteLock
(
	Graph *g,       // graph to lock
	int timeout_ms  // maximum time in milliseconds to wait for the lock:
                    // - timeout_ms < 0 : block until the lock is acquired
                    // - timeout_ms = 0 : non-blocking attempt (try-lock)
                    // - timeout_ms > 0 : wait up to timeout_ms milliseconds
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

// lock all matrices:
// 1. adjacency matrix
// 2. label matrices
// 3. node labels matrix
// 4. relation matrices
//
// currently only used just before forking for the purpose of
// taking a snapshot
void Graph_LockAllMatrices
(
	Graph *g  // graph to lock
);

// the counter-part of GraphLockAllMatrices
// unlocks all matrices:
//
// 1. adjacency matrix
// 2. label matrices
// 3. node labels matrix
// 4. relation matrices
//
// currently only used after a fork had been issued on both
// the parent and child processes
void Graph_UnlockAllMatrices
(
	Graph *g  // graph to unlock
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

// adds a label from the graph
void Graph_RemoveLabel
(
	Graph *g,
	LabelID label_id
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
	Graph *g,      // graph on which to operate
	RelationID r,  // relationship type
	Edge **edges   // edges to create
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

// populate 'nodes' with deleted node ids
void Graph_DeletedNodes
(
	const Graph *g,  // graph
	NodeID **nodes,  // [output] array of deleted node IDs
	uint64_t *n      // [output] number of deleted node IDs
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
	LabelID label
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
	RelationID relation_idx
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

// retrieves edges connecting source to destination,
// relation is optional, pass GRAPH_NO_RELATION if you do not care
// about edge type
void Graph_GetEdgesConnectingNodes
(
	const Graph *g,  // Graph to get edges from.
	NodeID srcID,    // Source node of edge
	NodeID destID,   // Destination node of edge
	RelationID r,    // Edge type.
	Edge **edges     // array_t of edges connecting src to dest of type r.
);

// returns true and sets edge's relation ID if edge is associated
// with one of the specified relations, otherwise returns false and does not
// change edge's relation ID
bool Graph_LookupEdgeRelationID
(
	const Graph *g,          // graph to get edges from
	Edge *edge,    	         // edge to check
	const RelationID *rels,  // relationships (can't contain unknown relations)
	int n_rels               // the number of relations
);

// get node edges
void Graph_GetNodeEdges
(
	const Graph *g,       // graph to get edges from
	const Node *n,        // node to extract edges from
	GRAPH_EDGE_DIR dir,   // edge direction
	RelationID edgeType,  // relation type
	Edge **edges          // array_t incoming/outgoing edges
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
	const Graph *g,
	bool transposed
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
	RelationID relation_idx,  // relation described by matrix
	bool transposed           // transposed
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

