/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "graph.h"
#include "../util/arr.h"
#include "../util/rwlock.h"
#include "../util/rmalloc.h"
#include "delta_matrix/delta_matrix_iter.h"
#include "../util/datablock/oo_datablock.h"

//------------------------------------------------------------------------------
// Synchronization functions
//------------------------------------------------------------------------------

static void _CreateRWLock
(
	Graph *g
) {
	// create a read write lock which favors writes
	//
	// consider the following locking sequence:
	// T0 read lock  (acquired)
	// T1 write lock (waiting)
	// T2 read lock  (acquired if lock favor reads, waiting if favor writes)
	//
	// we don't want to cause write starvation as this can impact overall
	// system performance

	// specify prefer write in lock creation attributes
	int res = 0 ;
	UNUSED(res) ;

	pthread_rwlockattr_t attr ;
	res = pthread_rwlockattr_init(&attr) ;
	ASSERT(res == 0) ;

#ifdef PTHREAD_RWLOCK_PREFER_WRITER_NONRECURSIVE_NP
	int pref = PTHREAD_RWLOCK_PREFER_WRITER_NONRECURSIVE_NP ;
	res = pthread_rwlockattr_setkind_np(&attr, pref) ;
	ASSERT(res == 0) ;
#endif

	res = pthread_rwlock_init(&g->_rwlock, &attr);
	ASSERT(res == 0) ;
}

// acquire a lock that does not restrict access from additional reader threads
void Graph_AcquireReadLock
(
	Graph *g
) {
	ASSERT(g != NULL);

	pthread_rwlock_rdlock(&g->_rwlock);
}

// acquire a lock for exclusive access to this graph's data
void Graph_AcquireWriteLock
(
	Graph *g
) {
	ASSERT(g != NULL);
	ASSERT(g->_writelocked == false);

	pthread_rwlock_wrlock(&g->_rwlock);
	g->_writelocked = true;
}

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
) {
	ASSERT (g != NULL) ;
	ASSERT (g->_writelocked == false) ;

	int res = rwlock_timedwrlock (&g->_rwlock, timeout_ms) ;
	g->_writelocked = (res == 0) ;

	return res ;
}

// Release the held lock
void Graph_ReleaseLock
(
	Graph *g
) {
	ASSERT(g != NULL);

	// set _writelocked to false BEFORE unlocking
	// if this is a reader thread no harm done,
	// if this is a writer thread the writer is about to unlock so once again
	// no harm done, if we set `_writelocked` to false after unlocking it is possible
	// for a reader thread to be considered as writer, performing illegal access to
	// underline matrices, consider a context switch after unlocking `_rwlock` but
	// before setting `_writelocked` to false
	g->_writelocked = false;
	pthread_rwlock_unlock(&g->_rwlock);
}

//------------------------------------------------------------------------------
// Graph utility functions
//------------------------------------------------------------------------------

// retrieves edges connecting source to destination
static void _Graph_GetEdgesConnectingNodes
(
	const Graph *g,     // Graph to get edges from.
	NodeID srcID,       // Source node of edge
	NodeID destID,      // Destination node of edge
	RelationID r,       // Edge type.
	Edge **edges        // array_t of edges connecting src to dest of type r
) {
	ASSERT(g);
	ASSERT(r      != GRAPH_NO_RELATION);
	ASSERT(r      < Graph_RelationTypeCount(g));
	ASSERT(srcID  < Graph_NodeCap(g));
	ASSERT(destID < Graph_NodeCap(g));

	Tensor R = Graph_GetRelationMatrix(g, r, false);
	Edge e = {.src_id = srcID, .dest_id = destID, .relationID = r};
	GrB_Index edge_id;
	TensorIterator it;
	TensorIterator_ScanEntry(&it, R, srcID, destID);

	while(TensorIterator_next(&it, NULL, NULL, &edge_id, NULL)) {
		e.id         = edge_id;
		e.attributes = DataBlock_GetItem(g->edges, edge_id);
		ASSERT(e.attributes);
		array_append(*edges, e);
	}
}

static inline AttributeSet *_Graph_GetEntity
(
	const DataBlock *entities,
	EntityID id
) {
	return DataBlock_GetItem(entities, id);
}

//------------------------------------------------------------------------------
// Matrix synchronization and resizing functions
//------------------------------------------------------------------------------

// resize given matrix, such that its number of row and columns
// matches the number of nodes in the graph. Also, synchronize
// matrix to execute any pending operations
void _MatrixSynchronize
(
	const Graph *g,   // graph the matrix is related to
	Delta_Matrix M,   // matrix to synchronize
	GrB_Index nrows,  // # of rows for the resize
	GrB_Index ncols   // # of columns for the resize
) {
	Delta_Matrix_synchronize(M, nrows, ncols);
}

// resize matrix to node capacity
void _MatrixResizeToCapacity
(
	const Graph *g,   // graph the matrix is related to
	Delta_Matrix M,   // matrix to synchronize
	GrB_Index nrows,  // # of rows for the resize
	GrB_Index ncols   // # of columns for the resize
) {
	GrB_Index n_rows;
	GrB_Index n_cols;
	Delta_Matrix_nrows(&n_rows, M);
	Delta_Matrix_ncols(&n_cols, M);

	// this policy should only be used in a thread-safe context,
	// so no locking is required
	if(n_rows < nrows || n_cols < ncols) {
		GrB_Info res = Delta_Matrix_resize(M, nrows, ncols);
		ASSERT(res == GrB_SUCCESS);
	}
}

// do not update matrices
void _MatrixNOP
(
	const Graph *g,   // graph the matrix is related to
	Delta_Matrix M,   // matrix to synchronize
	GrB_Index nrows,  // # of rows for the resize
	GrB_Index ncols   // # of columns for the resize
) {
	return;
}

// retrieve graph matrix synchronization policy
MATRIX_POLICY Graph_GetMatrixPolicy
(
	const Graph *g
) {
	ASSERT(g != NULL);
	MATRIX_POLICY policy = SYNC_POLICY_UNKNOWN;
	SyncMatrixFunc f = g->SynchronizeMatrix;

	if(f == _MatrixSynchronize) {
		policy = SYNC_POLICY_FLUSH_RESIZE;
	} else if(f == _MatrixResizeToCapacity) {
		policy = SYNC_POLICY_RESIZE;
	} else if(f == _MatrixNOP) {
		policy = SYNC_POLICY_NOP;
	} else {
		ASSERT(false);
	}

	return policy;
}

// define the current behavior for matrix creations and retrievals on this graph
MATRIX_POLICY Graph_SetMatrixPolicy
(
	Graph *g,
	MATRIX_POLICY policy
) {
	MATRIX_POLICY prev_policy = Graph_GetMatrixPolicy(g);

	switch(policy) {
		case SYNC_POLICY_FLUSH_RESIZE:
			// Default behavior; forces execution of pending GraphBLAS operations
			// when appropriate and sizes matrices to the current node count.
			g->SynchronizeMatrix = _MatrixSynchronize;
			break;
		case SYNC_POLICY_RESIZE:
			// Bulk insertion and creation behavior; does not force pending operations
			// and resizes matrices to the graph's current node capacity.
			g->SynchronizeMatrix = _MatrixResizeToCapacity;
			break;
		case SYNC_POLICY_NOP:
			// Used when deleting or freeing a graph; forces no matrix updates or resizes.
			g->SynchronizeMatrix = _MatrixNOP;
			break;
		default:
			ASSERT(false);
	}

	return prev_policy;
}

// synchronize and resize all matrices in graph
void Graph_ApplyAllPending
(
	Graph *g,
	bool force_flush
) {
	ASSERT(g != NULL);

	uint          n  =  0;
	Delta_Matrix  M  =  NULL;

	// set matrix sync policy, backup previous sync policy
	MATRIX_POLICY policy = Graph_SetMatrixPolicy(g, SYNC_POLICY_FLUSH_RESIZE);

	//--------------------------------------------------------------------------
	// sync every matrix
	//--------------------------------------------------------------------------

	// sync the adjacency matrix
	M = Graph_GetAdjacencyMatrix(g, false);
	Delta_Matrix_wait(M, force_flush);

	// sync node labels matrix
	M = Graph_GetNodeLabelMatrix(g);
	Delta_Matrix_wait(M, force_flush);

	// sync the zero matrix
	M = Graph_GetZeroMatrix(g);
	Delta_Matrix_wait(M, force_flush);

	// sync each label matrix
	n = array_len(g->labels);
	for(int i = 0; i < n; i ++) {
		M = Graph_GetLabelMatrix(g, i);
		Delta_Matrix_wait(M, force_flush);
	}

	// sync each relation matrix
	n = array_len(g->relations);
	for(int i = 0; i < n; i ++) {
		M = Graph_GetRelationMatrix(g, i, false);
		Delta_Matrix_wait(M, force_flush);
	}

	// restore previous matrix sync policy
	Graph_SetMatrixPolicy(g, policy);
}

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
) {
	ASSERT(g != NULL);

	uint n = 0;  // length of matrices array

	//--------------------------------------------------------------------------
	// lock matrices
	//--------------------------------------------------------------------------

	// lock the adjacency matrix
	Delta_Matrix_lock(g->adjacency_matrix);

	// lock node labels matrix
	Delta_Matrix_lock(g->node_labels);

	// lock each label matrix
	n = array_len(g->labels);
	for(int i = 0; i < n; i ++) {
		Delta_Matrix_lock(g->labels[i]);
	}

	// lock each relation matrix
	n = array_len(g->relations);
	for(int i = 0; i < n; i ++) {
		Delta_Matrix_lock(g->relations[i]);
	}
}

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
) {
	ASSERT(g != NULL);

	uint n = 0;  // length of matrices array

	//--------------------------------------------------------------------------
	// unlock matrices
	//--------------------------------------------------------------------------

	// unlock the adjacency matrix
	Delta_Matrix_unlock(g->adjacency_matrix);

	// unlock node labels matrix
	Delta_Matrix_unlock(g->node_labels);

	// unlock each label matrix
	n = array_len(g->labels);
	for(int i = 0; i < n; i ++) {
		Delta_Matrix_unlock(g->labels[i]);
	}

	// unlock each relation matrix
	n = array_len(g->relations);
	for(int i = 0; i < n; i ++) {
		Delta_Matrix_unlock(g->relations[i]);
	}
}

// checks to see if graph has pending operations
bool Graph_Pending
(
	const Graph *g
) {
	ASSERT(g != NULL);

	GrB_Info   info;
	UNUSED(info);

	uint          n        =  0;
	Delta_Matrix  M        =  NULL;
	bool          pending  =  false;

	//--------------------------------------------------------------------------
	// see if ADJ matrix contains pending changes
	//--------------------------------------------------------------------------

	M = g->adjacency_matrix;
	info = Delta_Matrix_pending(M, &pending);
	ASSERT(info == GrB_SUCCESS);
	if(pending) {
		return true;
	}

	//--------------------------------------------------------------------------
	// see if node_labels matrix contains pending changes
	//--------------------------------------------------------------------------

	M = g->node_labels;
	info = Delta_Matrix_pending(M, &pending);
	ASSERT(info == GrB_SUCCESS);
	if(pending) {
		return true;
	}

	//--------------------------------------------------------------------------
	// see if any label matrix contains pending changes
	//--------------------------------------------------------------------------

	n = array_len(g->labels);
	for(int i = 0; i < n; i ++) {
		M = g->labels[i];
		info = Delta_Matrix_pending(M, &pending);
		ASSERT(info == GrB_SUCCESS);
		if(pending) {
			return true;
		}
	}

	//--------------------------------------------------------------------------
	// see if any relationship matrix contains pending changes
	//--------------------------------------------------------------------------

	n = array_len(g->relations);
	for(int i = 0; i < n; i ++) {
		M = g->relations[i];
		info = Delta_Matrix_pending(M, &pending);
		ASSERT(info == GrB_SUCCESS);
		if(pending) {
			return true;
		}
	}

	return false;
}

//------------------------------------------------------------------------------
// Graph API
//------------------------------------------------------------------------------

// create a new graph
Graph *Graph_New
(
	size_t node_cap,  // allocation size for node datablocks and matrix dimensions
	size_t edge_cap   // allocation size for edge datablocks
) {
	fpDestructor cb = (fpDestructor)AttributeSet_Free;
	Graph *g = rm_calloc(1, sizeof(Graph));

	g->nodes     = DataBlock_New(node_cap, node_cap, sizeof(AttributeSet), cb);
	g->edges     = DataBlock_New(edge_cap, edge_cap, sizeof(AttributeSet), cb);
	g->labels    = array_new(Delta_Matrix, GRAPH_DEFAULT_LABEL_CAP);
	g->relations = array_new(Tensor, GRAPH_DEFAULT_RELATION_TYPE_CAP);

	GrB_Info info;
	UNUSED(info);

	GrB_Index n = Graph_RequiredMatrixDim(g);
	Delta_Matrix_new(&g->node_labels, GrB_BOOL, n, n, false);
	Delta_Matrix_new(&g->adjacency_matrix, GrB_BOOL, n, n, true);
	Delta_Matrix_new(&g->_zero_matrix, GrB_BOOL, n, n, false);

	// init graph statistics
	GraphStatistics_init(&g->stats);

	// initialize a read-write lock scoped to the individual graph
	_CreateRWLock(g);
	g->_writelocked = false;

	// force GraphBLAS updates and resize matrices to node count by default
	g->SynchronizeMatrix = _MatrixSynchronize;

	return g;
}

// get outgoing edges of node 'n'
static void _GetOutgoingNodeEdges
(
	const Graph *g,  // graph to collect edges from
	const Node *n,   // either source or destination node
	RelationID r,    // relationship type
	Edge **edges     // [output] array of edges
) {
	ASSERT (g) ;
	ASSERT (n) ;
	ASSERT (edges) ;
	ASSERT (r != GRAPH_NO_RELATION && r != GRAPH_UNKNOWN_RELATION) ;

	TensorIterator it ;
	NodeID src_id = ENTITY_GET_ID (n) ;
	Tensor R      = Graph_GetRelationMatrix (g, r, false) ;

	Edge e = {.src_id = src_id, .relationID = r};

	TensorIterator_ScanRange(&it, R, src_id, src_id, false);
	while(TensorIterator_next(&it, NULL, &e.dest_id, &e.id, NULL)) {
		e.attributes = DataBlock_GetItem(g->edges, e.id);
		ASSERT(e.attributes);
		array_append(*edges, e);
	}
}

// get incoming edges of node 'n'
static void _GetIncomingNodeEdges
(
	const Graph *g,        // graph to collect edges from
	const Node *n,         // either source or destination node
	RelationID r,          // relationship type
	bool skip_self_edges,  // skip self referencing edges
	Edge **edges           // [output] array of edges
) {
	ASSERT(g);
	ASSERT(n);
	ASSERT(edges);
	ASSERT(r != GRAPH_NO_RELATION && r != GRAPH_UNKNOWN_RELATION);

	TensorIterator it;
	Tensor T       = Graph_GetRelationMatrix(g, r, false);
	NodeID src_id  = INVALID_ENTITY_ID;
	NodeID dest_id = ENTITY_GET_ID(n);

	Edge e = {.dest_id = dest_id, .relationID = r};

	TensorIterator_ScanRange(&it, T, dest_id, dest_id, true);
	while(TensorIterator_next(&it, &e.src_id, NULL, &e.id, NULL)) {
		// skip self edges
		if(skip_self_edges && e.src_id == e.dest_id) {
			continue;
		}

		e.attributes = DataBlock_GetItem(g->edges, e.id);
		ASSERT(e.attributes);
		array_append(*edges, e);
	}
}

// free every relation matrix
static void _Graph_FreeRelationMatrices
(
	const Graph *g
) {
	uint n = Graph_RelationTypeCount(g);

	for(uint i = 0; i < n; i++) {
		// in case relation contains multi-edges free tensor
		// otherwise treat the relation matrix as a regular 2D matrix
		// which is a bit faster to free
		if(Graph_RelationshipContainsMultiEdge(g, i)) {
			Tensor_free(g->relations + i);
		} else {
			Delta_Matrix_free(g->relations + i);
		}
	}
}

// creates a new label matrix, returns id given to label
LabelID Graph_AddLabel
(
	Graph *g
) {
	ASSERT(g != NULL);

	Delta_Matrix m;
	GrB_Info info;
	size_t n = Graph_RequiredMatrixDim(g);
	Delta_Matrix_new(&m, GrB_BOOL, n, n, false);

	array_append(g->labels, m);
	// adding a new label, update the stats structures to support it
	GraphStatistics_IntroduceLabel(&g->stats);

	LabelID l = Graph_LabelTypeCount(g) - 1;
	return l;
}

// adds a label from the graph
void Graph_RemoveLabel
(
	Graph *g,
	LabelID label_id
) {
	ASSERT(g != NULL);
	ASSERT(label_id == Graph_LabelTypeCount(g) - 1);

	#ifdef RG_DEBUG
	GrB_Index nvals;
	GrB_Info info = Delta_Matrix_nvals(&nvals, g->labels[label_id]);
	ASSERT(info == GrB_SUCCESS);
	ASSERT(nvals == 0);
	#endif

	Delta_Matrix_free(&g->labels[label_id]);
	g->labels = array_del(g->labels, label_id);
}

// creates a new relation matrix, returns id given to relation
RelationID Graph_AddRelationType
(
	Graph *g
) {
	ASSERT(g);

	size_t n = Graph_RequiredMatrixDim(g);

	Tensor R = Tensor_new(n, n);
	array_append(g->relations, R);

	// adding a new relationship type, update the stats structures to support it
	GraphStatistics_IntroduceRelationship(&g->stats);

	RelationID relationID = Graph_RelationTypeCount(g) - 1;
	return relationID;
}

// removes a relation from the graph
void Graph_RemoveRelation
(
	Graph *g,
	RelationID relation_id
) {
	ASSERT(g != NULL);
	ASSERT(relation_id == Graph_RelationTypeCount(g) - 1);

	Tensor R = g->relations[relation_id];

	#ifdef RG_DEBUG
	GrB_Index nvals;
	GrB_Info info = Delta_Matrix_nvals(&nvals, R);
	ASSERT(info == GrB_SUCCESS);
	ASSERT(nvals == 0);
	#endif

	Tensor_free(&R);
	g->relations = array_del(g->relations, relation_id);
}

// make sure graph can hold an additional N nodes
void Graph_AllocateNodes
(
	Graph *g,               // graph for which nodes will be added
	size_t n                // number of nodes to create
) {
	ASSERT(g);
	DataBlock_Accommodate(g->nodes, n);
}

// make sure graph can hold an additional N edges
void Graph_AllocateEdges
(
	Graph *g,               // graph for which nodes will be added
	size_t n                // number of edges to create
) {
	ASSERT(g);
	DataBlock_Accommodate(g->edges, n);
}

// reset graph's reserved node count
void Graph_ResetReservedNode
(
	Graph *g  // graph
) {
	ASSERT(g != NULL);
	g->reserved_node_count = 0;
}

// reserve a node
Node Graph_ReserveNode
(
	Graph *g  // graph for which nodes will be added
) {
	ASSERT(g != NULL);

	// reserve node ID
	NodeID id = DataBlock_GetReservedIdx(g->nodes, g->reserved_node_count);

	// increment reserved node count
	g->reserved_node_count++;

	// create node
	Node n = (Node) { .attributes = NULL, .id = id };

	return n;
}

// create a single node and labels it accordingly.
// Return newly created node.
void Graph_CreateNode
(
	Graph *g,         // graph
	Node *n,          // node to create
	LabelID *labels,  // node's labels
	uint label_count  // number of labels
) {
	ASSERT(g != NULL);
	ASSERT(n != NULL);
	ASSERT(label_count == 0 || (label_count > 0 && labels != NULL));

	NodeID id      = n->id;  // save node ID
	n->attributes  = DataBlock_AllocateItem(g->nodes, &n->id);
	*n->attributes = NULL;   // initialize attributes to NULL

	// node ID was reserved, make reserved ID was assigned
	if(id != INVALID_ENTITY_ID) {
		ASSERT(id == n->id);
		g->reserved_node_count--;
		ASSERT(g->reserved_node_count >= 0);
	}

	if(label_count > 0) {
		Graph_LabelNode(g, ENTITY_GET_ID(n), labels, label_count);
	}
}

// create multiple nodes
// all nodes share the same set of labels
void Graph_CreateNodes
(
	Graph *g,            // graph
	Node **nodes,        // array of nodes to create
	AttributeSet *sets,  // nodes attributes
	uint node_count,     // number of nodes
	LabelID *labels,     // labels, same set of labels applied to all nodes
	uint label_count     // number of labels
) {
	ASSERT (g     != NULL) ;
	ASSERT (sets  != NULL) ;
	ASSERT (nodes != NULL) ;
	ASSERT (node_count > 0) ;
	ASSERT (label_count == 0 || labels != NULL) ;

	// collect label matrices
	Delta_Matrix lbl_matrices[label_count] ;
	Delta_Matrix node_label_matrix = Graph_GetNodeLabelMatrix (g) ;
	for (uint i = 0; i < label_count; i++) {
		lbl_matrices[i] = Graph_GetLabelMatrix (g, labels[i]) ;
	}

	// add nodes
	for (uint i = 0; i < node_count; i++) {
		Node *n = nodes[i] ;
		NodeID id = n->id ;  // save node ID

		// set attributes
		n->attributes  = DataBlock_AllocateItem (g->nodes, &n->id) ;
		*n->attributes = (sets == NULL) ? NULL : sets[i] ;

		// node ID was reserved, make sure reserved ID was assigned
		if (id != INVALID_ENTITY_ID) {
			// NodeID was preallocated via reservation
			// so now that it’s used we decrement the counter
			ASSERT (id == n->id) ;
			g->reserved_node_count-- ;
			ASSERT (g->reserved_node_count >= 0) ;
		}

		// label node
		for (uint j = 0; j < label_count; j++) {
			// set matrix at position [id, id]
			Delta_Matrix L = lbl_matrices[j] ;
			GrB_OK (Delta_Matrix_setElement_BOOL (L, n->id, n->id)) ;

			// map this label in this node's set of labels
			LabelID l = labels[j] ;
			GrB_OK (Delta_Matrix_setElement_BOOL (node_label_matrix, n->id, l)) ;
		}
	}

	// update statistics
	for (uint i = 0; i < label_count; i++) {
		LabelID l = labels[i] ;
		GraphStatistics_IncNodeCount (&g->stats, l, node_count) ;
	}
}

// label node with each label in 'lbls'
void Graph_LabelNode
(
	Graph *g,       // graph to operate on
	NodeID id,      // node ID to update
	LabelID *lbls,  // set to labels to associate with node
	uint lbl_count  // number of labels
) {
	// validations
	ASSERT(g != NULL);
	ASSERT(lbls != NULL);
	ASSERT(lbl_count > 0);
	ASSERT(id != INVALID_ENTITY_ID);

	GrB_Info info;
	UNUSED(info);

	Delta_Matrix nl = Graph_GetNodeLabelMatrix(g);
	for(uint i = 0; i < lbl_count; i++) {
		LabelID l = lbls[i];
		Delta_Matrix L = Graph_GetLabelMatrix(g, l);

		// set matrix at position [id, id]
		info = Delta_Matrix_setElement_BOOL(L, id, id);
		ASSERT(info == GrB_SUCCESS);

		// map this label in this node's set of labels
		info = Delta_Matrix_setElement_BOOL(nl, id, l);
		ASSERT(info == GrB_SUCCESS);

		// update labels statistics
		GraphStatistics_IncNodeCount(&g->stats, l, 1);
	}
}

// return true if node is labeled as 'l'
bool Graph_IsNodeLabeled
(
	Graph *g,   // graph to operate on
	NodeID id,  // node ID to inspect
	LabelID l   // label to check for
) {
	ASSERT(g  != NULL);
	ASSERT(id != INVALID_ENTITY_ID);

	// consult with labels matrix
	Delta_Matrix nl = Graph_GetNodeLabelMatrix(g);
	GrB_Info info = Delta_Matrix_isStoredElement(nl, id, l);
	ASSERT(info == GrB_SUCCESS || info == GrB_NO_VALUE);
	return info == GrB_SUCCESS;
}

// dissociates each label in 'lbls' from given node
void Graph_RemoveNodeLabels
(
	Graph *g,       // graph to operate against
	NodeID id,      // node ID to update
	LabelID  *lbls, // set of labels to remove
	uint lbl_count  // number of labels to remove
) {
	ASSERT(g != NULL);
	ASSERT(id != INVALID_ENTITY_ID);
	ASSERT(lbls != NULL);
	ASSERT(lbl_count > 0);

	GrB_Info info;
	UNUSED(info);

	Delta_Matrix nl = Graph_GetNodeLabelMatrix(g);
	for(uint i = 0; i < lbl_count; i++) {
		LabelID   l = lbls[i];
		Delta_Matrix M = Graph_GetLabelMatrix(g, l);

		// remove matrix at position [id, id]
		info = Delta_Matrix_removeElement(M, id, id);
		ASSERT(info == GrB_SUCCESS);

		// remove this label from node's set of labels
		info = Delta_Matrix_removeElement(nl, id, l);
		ASSERT(info == GrB_SUCCESS);

		// a label was removed from node, update statistics
		GraphStatistics_DecNodeCount(&g->stats, l, 1);
	}
}

// update ADJ and relation matrices with a new entry
// ADJ[src, dest] = true & R[src, dest] = edge_id
void Graph_FormConnection
(
	Graph *g,        // graph
	NodeID src,      // src node id
	NodeID dest,     // dest node id
	EdgeID edge_id,  // edge id
	RelationID r     // relation id
) {
	ASSERT(g != NULL);

	GrB_Info info;
	UNUSED(info);

	// sync matrices
	Tensor       R   = Graph_GetRelationMatrix(g, r, false);
	Delta_Matrix adj = Graph_GetAdjacencyMatrix(g, false);

	// rows represent source nodes, columns represent destination nodes
	info = Delta_Matrix_setElement_BOOL(adj, src, dest);
	ASSERT(info == GrB_SUCCESS);

	// add entry to relation tensor
	Tensor_SetElement(R, src, dest, edge_id);

	// an edge of type r has just been created, update statistics
	GraphStatistics_IncEdgeCount(&g->stats, r, 1);
}

// connects source node to destination node
void Graph_CreateEdge
(
	Graph *g,      // graph on which to operate
	NodeID src,    // source node ID
	NodeID dest,   // destination node ID
	RelationID r,  // edge type
	Edge *e        // [output] created edge
) {
	ASSERT(g != NULL);
	ASSERT(r < Graph_RelationTypeCount(g));

#ifdef RG_DEBUG
	// make sure both src and destination nodes exists
	Node node = GE_NEW_NODE();
	ASSERT(Graph_GetNode(g, src, &node)  == true);
	ASSERT(Graph_GetNode(g, dest, &node) == true);
#endif

	EdgeID id;
	AttributeSet *set = DataBlock_AllocateItem(g->edges, &id);
	*set = NULL;

	e->id         = id;
	e->src_id     = src;
	e->dest_id    = dest;
	e->attributes = set;
	e->relationID = r;

	Graph_FormConnection(g, src, dest, id, r);
}

// qsort edge compare function
// edge A is "greater" then edge B if
// A.src_id > B.src_id, in case A.src_id == B.src_id then
// A is "greater" then edge B if A.dest_id > B.dest_id
static int _edge_src_dest_cmp
(
	const void *a,
	const void *b
) {
	const Edge *ea = *(Edge **)a ;
	const Edge *eb = *(Edge **)b ;

	if (ea->src_id < eb->src_id) {
		return -1 ;
	}

	if (ea->src_id > eb->src_id) {
		return  1 ;
	}

	// src_id equal
	if (ea->dest_id < eb->dest_id) {
		return -1 ;
	}

	if (ea->dest_id > eb->dest_id) {
		return  1 ;
	}

	return 0 ;
}

// create multiple edges
void Graph_CreateEdges
(
	Graph *g,           // graph on which to operate
	RelationID r,       // relationship type
	Edge **edges,       // edges to create
	AttributeSet *sets  // [optional] attribute sets
) {
	ASSERT (g != NULL) ;
	ASSERT (r < Graph_RelationTypeCount (g)) ;
	ASSERT (r != GRAPH_NO_RELATION && r != GRAPH_UNKNOWN_RELATION) ;

	if (sets != NULL) {
		ASSERT (array_len (edges) == array_len (sets)) ;
	}

	uint edge_count = array_len(edges);
	Edge **edges_copy = rm_malloc (sizeof (Edge*) * edge_count) ;
	memcpy (edges_copy, edges, sizeof (Edge*) * edge_count) ;

	// sort edges by src & dest IDs
	//qsort(edges, edge_count, sizeof(Edge *), _edge_src_dest_cmp);

#ifdef RG_DEBUG
	// make sure both src and destination nodes exists
	for(uint i = 0; i < edge_count; i++) {
		Edge   *e   = edges[i] ;
		NodeID src  = e->src_id ;
		NodeID dest = e->dest_id ;
		Node   node = GE_NEW_NODE () ;
		ASSERT (Graph_GetNode (g, src, &node)  == true) ;
		ASSERT (Graph_GetNode (g, dest, &node) == true) ;
	}
#endif

	// make sure we have room for 'edge_count' edges
	DataBlock_Accommodate (g->edges, edge_count) ;

	// sync matrices
	Tensor       R   = Graph_GetRelationMatrix  (g, r, false) ;
	Delta_Matrix adj = Graph_GetAdjacencyMatrix (g, false) ;

	// in case R is empty switch to a more optimize construction approach
	// using GrB_Matrix_Build to build R

	// allocate edges and update ADJ matrix
	for (uint i = 0; i < edge_count; i++) {
		Edge *e = edges_copy[i] ;

		// TODO: switch to batch allocation of items
		AttributeSet *set = DataBlock_AllocateItem (g->edges, &e->id) ;
		*set = (sets != NULL) ? sets[i] : NULL ;

		e->relationID = r ;
		e->attributes = set ;

		NodeID src  = e->src_id ;
		NodeID dest = e->dest_id ;

		// TODO: introduce batch version of setElement, e.g. GrB_Matrix_build
		GrB_Info info = Delta_Matrix_setElement_BOOL (adj, src, dest) ;
		ASSERT (info == GrB_SUCCESS) ;
	}

	// sort edges by src & dest IDs
	qsort (edges_copy, edge_count, sizeof (Edge *), _edge_src_dest_cmp) ;

	// update R tensor
	Tensor_SetEdges (R, (const Edge **)edges_copy, edge_count) ;

	// update graph statistics
	GraphStatistics_IncEdgeCount (&g->stats, r, edge_count) ;

	rm_free (edges_copy) ;
}

// forward declaration
// defined in graph_delete_edges.c
// clears connections from the graph by updating relevent matrices
void Graph_ClearConnections
(
	Graph *g,     // graph to update
	Edge *edges,  // edges to clear
	uint64_t n    // number of edges
);

// deletes edges from the graph
void Graph_DeleteEdges
(
	Graph *g,     // graph to delete edges from
	Edge *edges,  // edges to delete
	uint64_t n    // number of edges
) {
	ASSERT(n     > 0);
	ASSERT(g     != NULL);
	ASSERT(edges != NULL);

	for(uint64_t i = 0; i < n; i++) {
		Edge *e = edges + i;

		// make sure edge isn't already deleted
		ASSERT(!DataBlock_ItemIsDeleted((void *)e->attributes));

		EdgeID id = ENTITY_GET_ID(e);
		DataBlock_DeleteItem(g->edges, id);
	}

	Graph_ClearConnections(g, edges, n);
}

// returns true if the given entity has been deleted
inline bool Graph_EntityIsDeleted
(
	const GraphEntity *e
) {
	if (e->attributes == NULL) {
		// most likely an entity which wasn't created just yet (reserved)
		return false;
	}

	return DataBlock_ItemIsDeleted (e->attributes) ;
}

// populate 'nodes' with deleted node ids
void Graph_DeletedNodes
(
	const Graph *g,  // graph
	NodeID **nodes,  // [output] array of deleted node IDs
	uint64_t *n      // [output] number of deleted node IDs
) {
	ASSERT(g     != NULL);
	ASSERT(n     != NULL);
	ASSERT(nodes != NULL);

	*n = DataBlock_DeletedItemsCount(g->nodes);
	const uint64_t *deleted_nodes = DataBlock_DeletedItems(g->nodes);

	*nodes = rm_malloc(sizeof(NodeID) * (*n));
	ASSERT(*nodes != NULL);

	*nodes = memcpy(*nodes, deleted_nodes, sizeof(uint64_t) * (*n));
}

// All graph matrices are required to be squared NXN
// where N = Graph_RequiredMatrixDim.
inline size_t Graph_RequiredMatrixDim
(
	const Graph *g
) {
	return Graph_NodeCap(g);
}

// retrieves a node iterator which can be used to access
// every node in the graph
DataBlockIterator *Graph_ScanNodes
(
	const Graph *g
) {
	ASSERT(g);
	return DataBlock_Scan(g->nodes);
}

// retrieves an edge iterator which can be used to access
// every edge in the graph
DataBlockIterator *Graph_ScanEdges
(
	const Graph *g
) {
	ASSERT(g);
	return DataBlock_Scan(g->edges);
}

// return number of nodes graph can contain
uint64_t Graph_NodeCap
(
	const Graph *g
) {
	return g->nodes->itemCap;
}

// returns number of nodes in the graph
uint64_t Graph_NodeCount
(
	const Graph *g
) {
	ASSERT(g);
	return g->nodes->itemCount;
}

// returns number of deleted nodes in the graph
uint Graph_DeletedNodeCount
(
	const Graph *g
) {
	ASSERT(g);
	return DataBlock_DeletedItemsCount(g->nodes);
}

// returns number of existing and deleted nodes in the graph
size_t Graph_UncompactedNodeCount
(
	const Graph *g
) {
	return Graph_NodeCount(g) + Graph_DeletedNodeCount(g);
}

// returns number of nodes with given label
uint64_t Graph_LabeledNodeCount
(
	const Graph *g,
	LabelID label
) {
	return GraphStatistics_NodeCount(&g->stats, label);
}

// returns number of edges in the graph
uint64_t Graph_EdgeCount
(
	const Graph *g
) {
	ASSERT(g);
	return g->edges->itemCount;
}

// returns number of edges of a specific relation type
uint64_t Graph_RelationEdgeCount
(
	const Graph *g,
	RelationID relation
) {
	return GraphStatistics_EdgeCount(&g->stats, relation);
}

// returns number of deleted edges in the graph
uint Graph_DeletedEdgeCount
(
	const Graph *g  // graph
) {
	ASSERT(g);
	return DataBlock_DeletedItemsCount(g->edges);
}

// returns number of different edge types
int Graph_RelationTypeCount
(
	const Graph *g
) {
	return array_len(g->relations);
}

// returns number of different node types
int Graph_LabelTypeCount
(
	const Graph *g
) {
	return array_len(g->labels);
}

// returns true if relationship matrix 'r' contains multi-edge entries,
// false otherwise
bool Graph_RelationshipContainsMultiEdge
(
	const Graph *g,
	RelationID r
) {
	ASSERT(Graph_RelationTypeCount(g) > r);

	GrB_Info info;
	GrB_Index nvals;

	// get the number of active entries in Tensor
	Tensor R = Graph_GetRelationMatrix(g, r, false);
	info = Delta_Matrix_nvals(&nvals, R);
	ASSERT(info == GrB_SUCCESS);

	// a tensor has Vector entries if
	// the number of active entries != number of edges of type 'r'
	return nvals != Graph_RelationEdgeCount(g, r);
}

// retrieves node with given id from graph,
// returns NULL if node wasn't found
bool Graph_GetNode
(
	const Graph *g,
	NodeID id,
	Node *n
) {
	ASSERT(g != NULL);
	ASSERT(n != NULL);

	n->id         = id;
	n->attributes = _Graph_GetEntity(g->nodes, id);

	return (n->attributes != NULL);
}

// retrieves edge with given id from graph,
// returns NULL if edge wasn't found
bool Graph_GetEdge
(
	const Graph *g,
	EdgeID id,
	Edge *e
) {
	ASSERT(g != NULL);
	ASSERT(e != NULL);
	ASSERT(id < g->edges->itemCap);

	e->id         = id;
	e->attributes = _Graph_GetEntity(g->edges, id);

	return (e->attributes != NULL);
}

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
) {
	ASSERT(g);
	ASSERT(edges);
	ASSERT(r < Graph_RelationTypeCount(g));

	// invalid relation type specified;
	// this can occur on multi-type traversals like:
	// MATCH ()-[:real_type|fake_type]->()
	if(r == GRAPH_UNKNOWN_RELATION) return;

#ifdef RG_DEBUG
	Node  srcNode   =  GE_NEW_NODE();
	Node  destNode  =  GE_NEW_NODE();
	ASSERT(Graph_GetNode(g, srcID, &srcNode)   == true);
	ASSERT(Graph_GetNode(g, destID, &destNode) == true);
#endif

	if(r != GRAPH_NO_RELATION) {
		_Graph_GetEdgesConnectingNodes(g, srcID, destID, r, edges);
	} else {
		// relation type missing, scan through each edge type
		int relationCount = Graph_RelationTypeCount(g);
		for(int i = 0; i < relationCount; i++) {
			_Graph_GetEdgesConnectingNodes(g, srcID, destID, i, edges);
		}
	}
}

// returns true and sets edge's relation ID if edge is associated
// with one of the specified relations, otherwise returns false and does not
// change edge's relation ID
bool Graph_LookupEdgeRelationID
(
	const Graph *g,          // graph to get edges from
	Edge *edge,    	         // edge to check
	const RelationID *rels,  // relationships (can't contain unknown relations)
	int n_rels               // the number of relations
) {
	ASSERT(g    != NULL);
	ASSERT(edge != NULL);
	ASSERT((rels && n_rels > 0) || (rels == NULL && n_rels == 0));
	
	GrB_Info info;

	uint64_t  x      = 0;
	Tensor    R      = NULL; 
	EntityID  id     = ENTITY_GET_ID(edge);
	bool      found  = false;
	GrB_Index srcID  = Edge_GetSrcNodeID(edge);
	GrB_Index destID = Edge_GetDestNodeID(edge);

	// if rels are not specified run through all Relationships in the graph
	n_rels = (rels == NULL) ? Graph_RelationTypeCount(g) : n_rels;

	for (int i = 0; i < n_rels; i++) {
		// use the next entry in rels if given, otherwise, the ith rel ID
		RelationID curr = rels ? rels[i] : i;
		ASSERT(curr != GRAPH_UNKNOWN_RELATION);
		ASSERT(curr < Graph_RelationTypeCount(g));

		R = Graph_GetRelationMatrix(g, curr, false);
		info = Delta_Matrix_extractElement_UINT64(&x, R, srcID, destID);
		ASSERT(info >= 0);

		if (info == GrB_NO_VALUE) {
			// edge isn't associated with current relation id
			// move on to the next one
			continue;
		}

		ASSERT(info == GrB_SUCCESS);

		// try to find the edge id within the matrix entry
		if (SCALAR_ENTRY(x)) {
			found = ((EntityID) x == id);
		} else {
			// multi-edge
			GrB_Vector x_vec = AS_VECTOR(x);
			info = GxB_Vector_isStoredElement(x_vec, id);
			found = (info == GrB_SUCCESS);
		}

		if (found) {
			Edge_SetRelationID(edge, curr);
			break;
		}
	}

	return found;
}

// retrieves all either incoming or outgoing edges
// to/from given node N, depending on given direction
void Graph_GetNodeEdges
(
	const Graph *g,      // graph to collect edges from
	const Node *n,       // either source or destination node
	GRAPH_EDGE_DIR dir,  // edge direction ->, <-, <->
	RelationID r,        // relationship type
	Edge **edges         // [output] array of edges
) {
	ASSERT (g) ;
	ASSERT (n) ;
	ASSERT (edges) ;

	if (r == GRAPH_UNKNOWN_RELATION) {
		return ;
	}

	bool outgoing = (dir == GRAPH_EDGE_DIR_OUTGOING ||
					 dir == GRAPH_EDGE_DIR_BOTH);

	bool incoming = (dir == GRAPH_EDGE_DIR_INCOMING ||
					 dir == GRAPH_EDGE_DIR_BOTH);

	// when direction is BOTH to avoid duplication we want to skip over
	// self referencing edges, as those will show up twice once for (a)->(a)
	// and (a)<-(a)
	bool skip_self_edges = (dir == GRAPH_EDGE_DIR_BOTH);

	if (outgoing) {
		if (r != GRAPH_NO_RELATION) {
			_GetOutgoingNodeEdges (g, n, r, edges) ;
		} else {
			// relation type missing, scan through each edge type
			int relationCount = Graph_RelationTypeCount (g) ;
			for (int i = 0; i < relationCount; i++) {
				_GetOutgoingNodeEdges (g, n, i, edges) ;
			}
		}
	}

	if(incoming) {
		if(r != GRAPH_NO_RELATION) {
			_GetIncomingNodeEdges(g, n, r, skip_self_edges, edges);
		} else {
			// relation type missing, scan through each edge type
			int relationCount = Graph_RelationTypeCount(g);
			for(int i = 0; i < relationCount; i++) {
				_GetIncomingNodeEdges(g, n, i, skip_self_edges, edges);
			}
		}
	}
}

// returns node incoming/outgoing degree
uint64_t Graph_GetNodeDegree
(
	const Graph *g,      // graph to inquery
	const Node *n,       // node to get degree of
	GRAPH_EDGE_DIR dir,  // incoming/outgoing/both
	RelationID r         // relation type
) {
	ASSERT(g != NULL);
	ASSERT(n != NULL);

	NodeID   srcID      = ENTITY_GET_ID(n);
	NodeID   destID     = INVALID_ENTITY_ID;
	EdgeID   edgeID     = INVALID_ENTITY_ID;
	uint64_t edge_count = 0;

	if(r == GRAPH_UNKNOWN_RELATION) {
		return 0;  // no edges
	}

	bool outgoing = (dir == GRAPH_EDGE_DIR_OUTGOING ||
					 dir == GRAPH_EDGE_DIR_BOTH);

	bool incoming = (dir == GRAPH_EDGE_DIR_INCOMING ||
					 dir == GRAPH_EDGE_DIR_BOTH);

	// relationships to consider
	RelationID start_rel;
	RelationID end_rel;

	if(r != GRAPH_NO_RELATION) {
		// consider only specified relationship
		start_rel = r;
		end_rel = start_rel + 1;
	} else {
		// consider all relationship types
		start_rel = 0;
		end_rel = Graph_RelationTypeCount(g);
	}

	// for each relationship type to consider
	for(RelationID edgeType = start_rel; edgeType < end_rel; edgeType++) {
		//----------------------------------------------------------------------
		// outgoing edges
		//----------------------------------------------------------------------

		Tensor R = Graph_GetRelationMatrix(g, edgeType, false);

		if(outgoing) {
			edge_count += Tensor_RowDegree(R, srcID);
		}

		//----------------------------------------------------------------------
		// incoming edges
		//----------------------------------------------------------------------

		if(incoming) {
			edge_count += Tensor_ColDegree(R, srcID);
		}
	}

	return edge_count;
}

// populate array of node's label IDs, return number of labels on node
uint Graph_GetNodeLabels
(
	const Graph *g,   // graph the node belongs to
	const Node *n,    // node to extract labels from
	LabelID *labels,  // array to populate with labels
	uint label_count  // size of labels array
) {
	// validate inputs
	ASSERT(g      != NULL);
	ASSERT(n      != NULL);
	ASSERT(labels != NULL);

	GrB_Info res;
	UNUSED(res);

	Delta_Matrix M = Graph_GetNodeLabelMatrix(g);

	EntityID id = ENTITY_GET_ID(n);
	Delta_MatrixTupleIter iter;
	res = Delta_MatrixTupleIter_AttachRange(&iter, M, id, id);
	ASSERT(res == GrB_SUCCESS);

	uint i = 0;

	for(; i < label_count; i++) {
		GrB_Index col;
		res = Delta_MatrixTupleIter_next_BOOL(&iter, NULL, &col, NULL);
		labels[i] = col;

		if(res == GxB_EXHAUSTED) break;
	}

	Delta_MatrixTupleIter_detach(&iter);

	return i;
}

// retrieves the adjacency matrix
// matrix is resized if its size doesn't match graph's node count
Delta_Matrix Graph_GetAdjacencyMatrix
(
	const Graph *g,
	bool transposed
) {
	return Graph_GetRelationMatrix(g, GRAPH_NO_RELATION, transposed);
}

// retrieves a label matrix
// matrix is resized if its size doesn't match graph's node count
Delta_Matrix Graph_GetLabelMatrix
(
	const Graph *g,
	LabelID label_idx
) {
	ASSERT(g != NULL);
	ASSERT(label_idx < Graph_LabelTypeCount(g));

	// return zero matrix if label_idx is out of range
	if(label_idx < 0) return Graph_GetZeroMatrix(g);

	Delta_Matrix m = g->labels[label_idx];
	size_t n = Graph_RequiredMatrixDim(g);
	g->SynchronizeMatrix(g, m, n, n);

	return m;
}

// retrieves a typed adjacency matrix
// matrix is resized if its size doesn't match graph's node count
Tensor Graph_GetRelationMatrix
(
	const Graph *g,           // graph from which to get adjacency matrix
	RelationID relation_idx,  // relation described by matrix
	bool transposed           // transposed
) {
	ASSERT(g);
	ASSERT(relation_idx == GRAPH_NO_RELATION ||
		   relation_idx < Graph_RelationTypeCount(g));

	Tensor m = GrB_NULL;

	if(relation_idx == GRAPH_NO_RELATION) {
		m = g->adjacency_matrix;
	} else {
		m = g->relations[relation_idx];
	}

	size_t n = Graph_RequiredMatrixDim(g);
	g->SynchronizeMatrix(g, m, n, n);

	if(transposed) m = Delta_Matrix_getTranspose(m);

	return m;
}

// retrieves the node-label mapping matrix,
// matrix is resized if its size doesn't match graph's node count.
Delta_Matrix Graph_GetNodeLabelMatrix
(
	const Graph *g
) {
	ASSERT(g != NULL);

	Delta_Matrix m = g->node_labels;

	size_t n = Graph_RequiredMatrixDim(g);
	g->SynchronizeMatrix(g, m, n, n);

	return m;
}

// retrieves the zero matrix
// the function will resize it to match all other
// internal matrices, caller mustn't modify it in any way
Delta_Matrix Graph_GetZeroMatrix
(
	const Graph *g
) {
	Delta_Matrix z = g->_zero_matrix;
	size_t n = Graph_RequiredMatrixDim(g);
	g->SynchronizeMatrix(g, z, n, n);

#if RG_DEBUG
	// make sure zero matrix is indeed empty
	GrB_Index nvals;
	Delta_Matrix_nvals(&nvals, z);
	ASSERT(nvals == 0);
#endif

	return z;
}

static void _Graph_Free
(
	Graph *g,
	bool is_full_graph
) {
	ASSERT(g);
	// free matrices
	AttributeSet *set;
	DataBlockIterator *it;

	Delta_Matrix_free(&g->_zero_matrix);
	Delta_Matrix_free(&g->adjacency_matrix);

	_Graph_FreeRelationMatrices(g);
	array_free(g->relations);

	uint32_t labelCount = array_len(g->labels);
	for(int i = 0; i < labelCount; i++) Delta_Matrix_free(&g->labels[i]);
	array_free(g->labels);
	Delta_Matrix_free(&g->node_labels);

	it = is_full_graph ? Graph_ScanNodes(g) : DataBlock_FullScan(g->nodes);
	while((set = (AttributeSet *)DataBlockIterator_Next(it, NULL)) != NULL) {
		if(*set != NULL) {
			AttributeSet_Free(set);
		}
	}
	DataBlockIterator_Free(it);

	it = is_full_graph ? Graph_ScanEdges(g) : DataBlock_FullScan(g->edges);
	while((set = DataBlockIterator_Next(it, NULL)) != NULL) {
		if(*set != NULL) {
			AttributeSet_Free(set);
		}
	}
	DataBlockIterator_Free(it);


	// free blocks
	DataBlock_Free(g->nodes);
	DataBlock_Free(g->edges);

	GraphStatistics_FreeInternals(&g->stats);

	int res;
	UNUSED(res);

	if(g->_writelocked) Graph_ReleaseLock(g);
	res = pthread_rwlock_destroy(&g->_rwlock);
	ASSERT(res == 0);

	rm_free(g);
}

void Graph_PartialFree
(
	Graph *g
) {
	_Graph_Free(g, false);
}

// free graph
void Graph_Free
(
	Graph *g
) {
	_Graph_Free(g, true);
}

