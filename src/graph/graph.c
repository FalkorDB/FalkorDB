/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "graph.h"
#include "../util/arr.h"
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

#if !defined(__APPLE__) && !defined(__FreeBSD__)
	int pref = PTHREAD_RWLOCK_PREFER_WRITER_NONRECURSIVE_NP ;
	res = pthread_rwlockattr_setkind_np(&attr, pref) ;
	ASSERT(res == 0) ;
#endif

	res = pthread_rwlock_init(&g->_rwlock, &attr);
	ASSERT(res == 0) ;
}

// acquire a lock that does not restrict access from additional reader threads
void Graph_AcquireReadLock(Graph *g) {
	ASSERT(g != NULL);

	pthread_rwlock_rdlock(&g->_rwlock);
}

// acquire a lock for exclusive access to this graph's data
void Graph_AcquireWriteLock(Graph *g) {
	ASSERT(g != NULL);
	ASSERT(g->_writelocked == false);

	pthread_rwlock_wrlock(&g->_rwlock);
	g->_writelocked = true;
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

// return number of nodes graph can contain
static inline size_t _Graph_NodeCap(const Graph *g) {
	return g->nodes->itemCap;
}

// Locates edges connecting src to destination.
void _Graph_GetEdgesConnectingNodes
(
	const Graph *g,
	NodeID src,
	NodeID dest,
	int r,
	Edge **edges
) {
	ASSERT(g);
	ASSERT(r    != GRAPH_NO_RELATION);
	ASSERT(r    < Graph_RelationTypeCount(g));
	ASSERT(src  < _Graph_NodeCap(g));
	ASSERT(dest < _Graph_NodeCap(g));

	// relation map, maps (src, dest, r) to edge IDs.
	Graph_GetRelationMatrix(g, r, false);
	Graph_GetMultiEdgeRelationMatrix(g, r);
	Edge e = {.src_id = src, .dest_id = dest, .relationID = r};
	GrB_Index edge_id;
	RelationIterator it = {0};
	RelationIterator_AttachSourceDest(&it, g->relations[r], src, dest);

	while(RelationIterator_next(&it, NULL, NULL, &edge_id)){
		e.id          =  edge_id;
		e.attributes  =  DataBlock_GetItem(g->edges, edge_id);
		ASSERT(e.attributes);
		array_append(*edges, e);
	}
}

static inline AttributeSet *_Graph_GetEntity(const DataBlock *entities, EntityID id) {
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
		M = Graph_GetMultiEdgeRelationMatrix(g, i);
		Delta_Matrix_wait(M, force_flush);
	}

	// restore previous matrix sync policy
	Graph_SetMatrixPolicy(g, policy);
}

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
		pending = RelationMatrix_pending(g->relations[i]);
		if(pending) {
			return true;
		}
	}

	return false;
}

//------------------------------------------------------------------------------
// Graph API
//------------------------------------------------------------------------------

Graph *Graph_New
(
	size_t node_cap,
	size_t edge_cap
) {

	fpDestructor cb = (fpDestructor)AttributeSet_Free;
	Graph *g = rm_calloc(1, sizeof(Graph));

	g->nodes     = DataBlock_New(node_cap, node_cap, sizeof(AttributeSet), cb);
	g->edges     = DataBlock_New(edge_cap, edge_cap, sizeof(AttributeSet), cb);
	g->labels    = array_new(Delta_Matrix, GRAPH_DEFAULT_LABEL_CAP);
	g->relations = array_new(RelationMatrix, GRAPH_DEFAULT_RELATION_TYPE_CAP);

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

// All graph matrices are required to be squared NXN
// where N = Graph_RequiredMatrixDim.
inline size_t Graph_RequiredMatrixDim(const Graph *g) {
	return _Graph_NodeCap(g);
}

size_t Graph_NodeCount(const Graph *g) {
	ASSERT(g);
	return g->nodes->itemCount;
}

uint Graph_DeletedNodeCount(const Graph *g) {
	ASSERT(g);
	return DataBlock_DeletedItemsCount(g->nodes);
}

size_t Graph_UncompactedNodeCount(const Graph *g) {
	return Graph_NodeCount(g) + Graph_DeletedNodeCount(g);
}

uint64_t Graph_LabeledNodeCount
(
	const Graph *g,
	LabelID label
) {
	return GraphStatistics_NodeCount(&g->stats, label);
}

size_t Graph_EdgeCount(const Graph *g) {
	ASSERT(g);
	return g->edges->itemCount;
}

uint64_t Graph_RelationEdgeCount
(
	const Graph *g,
	RelationID relation
) {
	return GraphStatistics_EdgeCount(&g->stats, relation);
}

uint Graph_DeletedEdgeCount(const Graph *g) {
	ASSERT(g);
	return DataBlock_DeletedItemsCount(g->edges);
}

int Graph_RelationTypeCount(const Graph *g) {
	return array_len(g->relations);
}

int Graph_LabelTypeCount(const Graph *g) {
	return array_len(g->labels);
}

void Graph_AllocateNodes(Graph *g, size_t n) {
	ASSERT(g);
	DataBlock_Accommodate(g->nodes, n);
}

void Graph_AllocateEdges(Graph *g, size_t n) {
	ASSERT(g);
	DataBlock_Accommodate(g->edges, n);
}

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

void Graph_GetEdgesConnectingNodes
(
	const Graph *g,
	NodeID srcID,
	NodeID destID,
	RelationID r,
	Edge **edges
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

void Graph_ResetReservedNode
(
	Graph *g
) {
	ASSERT(g != NULL);
	g->reserved_node_count = 0;
}

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

void Graph_CreateNode
(
	Graph *g,
	Node *n,
	LabelID *labels,
	uint label_count
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

	bool x;
	// consult with labels matrix
	Delta_Matrix nl = Graph_GetNodeLabelMatrix(g);
	GrB_Info info = Delta_Matrix_extractElement_BOOL(&x, nl, id, l);
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

void Graph_FormConnection
(
	Graph *g,
	NodeID src,
	NodeID dest,
	EdgeID edge_id,
	int r
) {
	ASSERT(g != NULL);

	GrB_Info info;
	UNUSED(info);
	// sync matrices
	Graph_GetRelationMatrix(g, r, false);
	Graph_GetMultiEdgeRelationMatrix(g, r);

	Delta_Matrix adj  = Graph_GetAdjacencyMatrix(g, false);

	// rows represent source nodes, columns represent destination nodes
	info = Delta_Matrix_setElement_BOOL(adj, src, dest);
	ASSERT(info == GrB_SUCCESS);

	RelationMatrix_FormConnection(g->relations[r], src, dest, edge_id);

	// an edge of type r has just been created, update statistics
	GraphStatistics_IncEdgeCount(&g->stats, r, 1);
}

void Graph_CreateEdge
(
	Graph *g,
	NodeID src,
	NodeID dest,
	RelationID r,
	Edge *e
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

void Graph_FormConnections
(
	Graph *g,
	Edge **edges,
	int r
) {
	ASSERT(g != NULL);

	GrB_Info info;
	UNUSED(info);
	// sync matrices
	Graph_GetRelationMatrix(g, r, false);
	Graph_GetMultiEdgeRelationMatrix(g, r);

	Delta_Matrix adj = Graph_GetAdjacencyMatrix(g, false);

	uint edge_count = array_len(edges);
	for(uint i = 0; i < edge_count; i++) {
		Edge  *e    = edges[i];
		NodeID src  = e->src_id;
		NodeID dest = e->dest_id;

		// rows represent source nodes, columns represent destination nodes
		info = Delta_Matrix_setElement_BOOL(adj, src, dest);
		ASSERT(info == GrB_SUCCESS);
	}

	// create connections in multi-edge matrix
	RelationMatrix_FormConnections(g->relations[r], (const Edge **)edges);

	// an edge of type r has just been created, update statistics
	GraphStatistics_IncEdgeCount(&g->stats, r, edge_count);
}

static int _cmp
(
	const void *a,
	const void *b
) {
	Edge *ea = *(Edge **)a;
	Edge *eb = *(Edge **)b;
	if(ea->src_id == eb->src_id) return ea->dest_id - eb->dest_id;
	return ea->src_id - eb->src_id;
}

void Graph_CreateEdges
(
	Graph *g,
	RelationID r,
	Edge **edges
) {
	ASSERT(g != NULL);
	ASSERT(r < Graph_RelationTypeCount(g));

	uint edge_count = array_len(edges);
	qsort(edges, edge_count, sizeof(Edge *), _cmp);
#ifdef RG_DEBUG
	// make sure both src and destination nodes exists
	for(uint i = 0; i < edge_count; i++) {
		Edge *e = edges[i];
		NodeID src  = e->src_id;
		NodeID dest = e->dest_id;
		Node node = GE_NEW_NODE();
		ASSERT(Graph_GetNode(g, src, &node)  == true);
		ASSERT(Graph_GetNode(g, dest, &node) == true);
	}
#endif

	EdgeID id;
	for(uint i = 0; i < edge_count; i++) {
		Edge *e = edges[i];
		AttributeSet *set = DataBlock_AllocateItem(g->edges, &id);
		*set = NULL;

		e->id         = id;
		e->attributes = set;
		e->relationID = r;
	}

	Graph_FormConnections(g, edges, r);
}

void _GetOutgoingNodeEdges
(
	const Graph *g,       // graph to collect edges from
	const Node *n,        // either source or destination node
	RelationID edgeType,  // relationship type
	Edge **edges          // [output] array of edges
) {
	ASSERT(g);
	ASSERT(n);
	ASSERT(edges);
	ASSERT(edgeType != GRAPH_NO_RELATION && edgeType != GRAPH_UNKNOWN_RELATION);

	GrB_Info info;
	RelationIterator        it       =  {0};
	NodeID                  src_id   =  ENTITY_GET_ID(n);
	NodeID                  dest_id  =  INVALID_ENTITY_ID;
	EdgeID                  edge_id  =  INVALID_ENTITY_ID;
	UNUSED(info);

	Graph_GetRelationMatrix(g, edgeType, false);
	Graph_GetMultiEdgeRelationMatrix(g, edgeType);

	Edge e = {.src_id = src_id, .relationID = edgeType};
	RelationIterator_AttachSourceRange(&it, g->relations[edgeType], src_id, src_id, false);
	while(RelationIterator_next(&it, NULL, &dest_id, &edge_id)) {
		e.dest_id     =  dest_id;
		e.id          =  edge_id;
		e.attributes  =  DataBlock_GetItem(g->edges, edge_id);
		ASSERT(e.attributes);
		array_append(*edges, e);
	}
}

void _GetIncomingNodeEdges
(
	const Graph *g,       // graph to collect edges from
	const Node *n,        // either source or destination node
	GRAPH_EDGE_DIR dir,   // edge direction ->, <-, <->
	RelationID edgeType,  // relationship type
	Edge **edges          // [output] array of edges
) {
	ASSERT(g);
	ASSERT(n);
	ASSERT(edges);
	ASSERT(edgeType != GRAPH_NO_RELATION && edgeType != GRAPH_UNKNOWN_RELATION);

	GrB_Info info;
	RelationIterator        it       =  {0};
	NodeID                  src_id   =  INVALID_ENTITY_ID;
	NodeID                  dest_id  =  ENTITY_GET_ID(n);
	EdgeID                  edge_id  =  INVALID_ENTITY_ID;
	UNUSED(info);

	Graph_GetRelationMatrix(g, edgeType, false);
	Graph_GetRelationMatrix(g, edgeType, true);
	Graph_GetMultiEdgeRelationMatrix(g, edgeType);

	Edge e = {.dest_id = dest_id, .relationID = edgeType};
	RelationIterator_AttachSourceRange(&it, g->relations[edgeType], dest_id, dest_id, true);
	while(RelationIterator_next(&it, &src_id, NULL, &edge_id)) {
		e.src_id      =  src_id;
		e.id          =  edge_id;
		e.attributes  =  DataBlock_GetItem(g->edges, edge_id);
		ASSERT(e.attributes);
		array_append(*edges, e);
	}
}

// retrieves all either incoming or outgoing edges
// to/from given node N, depending on given direction
void Graph_GetNodeEdges
(
	const Graph *g,       // graph to collect edges from
	const Node *n,        // either source or destination node
	GRAPH_EDGE_DIR dir,   // edge direction ->, <-, <->
	RelationID edgeType,  // relationship type
	Edge **edges          // [output] array of edges
) {
	ASSERT(g);
	ASSERT(n);
	ASSERT(edges);

	if(edgeType == GRAPH_UNKNOWN_RELATION) return;

	bool outgoing = (dir == GRAPH_EDGE_DIR_OUTGOING ||
					 dir == GRAPH_EDGE_DIR_BOTH);

	bool incoming = (dir == GRAPH_EDGE_DIR_INCOMING ||
					 dir == GRAPH_EDGE_DIR_BOTH);

	if(outgoing) {
		if(edgeType != GRAPH_NO_RELATION) {
			_GetOutgoingNodeEdges(g, n, edgeType, edges);
		} else {
			// relation type missing, scan through each edge type
			int relationCount = Graph_RelationTypeCount(g);
			for(int i = 0; i < relationCount; i++) {
				_GetOutgoingNodeEdges(g, n, i, edges);
			}
		}
	}

	if(incoming) {
		if(edgeType != GRAPH_NO_RELATION) {
			_GetIncomingNodeEdges(g, n, dir, edgeType, edges);
		} else {
			// relation type missing, scan through each edge type
			int relationCount = Graph_RelationTypeCount(g);
			for(int i = 0; i < relationCount; i++) {
				_GetIncomingNodeEdges(g, n, dir, i, edges);
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
	RelationID edgeType  // relation type
) {
	ASSERT(g != NULL);
	ASSERT(n != NULL);

	NodeID                 srcID      = ENTITY_GET_ID(n);
	NodeID                 destID     = INVALID_ENTITY_ID;
	EdgeID                 edgeID     = INVALID_ENTITY_ID;
	uint64_t               edge_count = 0;

	if(edgeType == GRAPH_UNKNOWN_RELATION) {
		return 0;  // no edges
	}

	bool outgoing = (dir == GRAPH_EDGE_DIR_OUTGOING ||
					 dir == GRAPH_EDGE_DIR_BOTH);

	bool incoming = (dir == GRAPH_EDGE_DIR_INCOMING ||
					 dir == GRAPH_EDGE_DIR_BOTH);

	// relationships to consider
	int start_rel;
	int end_rel;

	if(edgeType != GRAPH_NO_RELATION) {
		// consider only specified relationship
		start_rel = edgeType;
		end_rel = start_rel + 1;
	} else {
		// consider all relationship types
		start_rel = 0;
		end_rel = Graph_RelationTypeCount(g);
	}

	// for each relationship type to consider
	for(edgeType = start_rel; edgeType < end_rel; edgeType++) {
		//----------------------------------------------------------------------
		// outgoing edges
		//----------------------------------------------------------------------

		Graph_GetRelationMatrix(g, edgeType, false);
		Graph_GetMultiEdgeRelationMatrix(g, edgeType);
		
		if(outgoing) {
			RelationIterator it = {0};
			// construct an iterator to traverse over the source node row,
			// containing all outgoing edges
			RelationIterator_AttachSourceRange(&it, g->relations[edgeType], srcID, srcID, false);

			// scan row
			while(RelationIterator_next(&it, NULL, NULL, &edgeID)) {
				edge_count++;
			}
		}

		//----------------------------------------------------------------------
		// incoming edges
		//----------------------------------------------------------------------

		if(incoming) {
			Graph_GetRelationMatrix(g, edgeType, true);
			RelationIterator it = {0};
			// construct an iterator to traverse over the source node row,
			// containing all incoming edges
			RelationIterator_AttachSourceRange(&it, g->relations[edgeType], srcID, srcID, true);

			// scan row
			while(RelationIterator_next(&it, NULL, NULL, &edgeID)) {
				edge_count++;
			}
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
	Delta_MatrixTupleIter iter = {0};
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

inline bool Graph_EntityIsDeleted
(
	const GraphEntity *e
) {
	if(e->attributes == NULL) {
		// most likely an entity which wasn't created just yet (reserved)
		return false;
	}

	return DataBlock_ItemIsDeleted(e->attributes);
}

static void _Graph_FreeRelationMatrices
(
	const Graph *g
) {
	uint relationCount = Graph_RelationTypeCount(g);
	for(uint i = 0; i < relationCount; i++) {
		RelationMatrix_free(g->relations[i]);
	}
}

DataBlockIterator *Graph_ScanNodes(const Graph *g) {
	ASSERT(g);
	return DataBlock_Scan(g->nodes);
}

DataBlockIterator *Graph_ScanEdges(const Graph *g) {
	ASSERT(g);
	return DataBlock_Scan(g->edges);
}

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

RelationID Graph_AddRelationType
(
	Graph *g
) {
	ASSERT(g);

	RelationMatrix r;
	size_t n = Graph_RequiredMatrixDim(g);

	r = RelationMatrix_new(n, n);

	array_append(g->relations, r);
	// adding a new relationship type, update the stats structures to support it
	GraphStatistics_IntroduceRelationship(&g->stats);

	RelationID relationID = Graph_RelationTypeCount(g) - 1;
	return relationID;
}

void Graph_RemoveRelation
(
	Graph *g,
	int relation_id
) {
	ASSERT(g != NULL);
	ASSERT(relation_id == Graph_RelationTypeCount(g) - 1);
	#ifdef RG_DEBUG
	GrB_Index nvals;
	GrB_Info info = Delta_Matrix_nvals(&nvals, g->relations[relation_id]->R);
	ASSERT(info == GrB_SUCCESS);
	ASSERT(nvals == 0);
	#endif
	RelationMatrix_free(g->relations[relation_id]);
	g->relations = array_del(g->relations, relation_id);
}

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

Delta_Matrix Graph_GetRelationMatrix
(
	const Graph *g,
	RelationID relation_idx,
	bool transposed
) {
	ASSERT(g);
	ASSERT(relation_idx == GRAPH_NO_RELATION ||
		   relation_idx < Graph_RelationTypeCount(g));

	Delta_Matrix m = GrB_NULL;

	if(relation_idx == GRAPH_NO_RELATION) {
		m = g->adjacency_matrix;
	} else {
		m = g->relations[relation_idx]->R;
	}

	size_t n = Graph_RequiredMatrixDim(g);
	g->SynchronizeMatrix(g, m, n, n);

	if(transposed) m = Delta_Matrix_getTranspose(m);

	return m;
}

Delta_Matrix Graph_GetMultiEdgeRelationMatrix
(
	const Graph *g,
	RelationID relation_idx
) {
	ASSERT(g);
	ASSERT(relation_idx != GRAPH_NO_RELATION &&
		   relation_idx < Graph_RelationTypeCount(g));

	Delta_Matrix m = g->relations[relation_idx]->E;

	size_t edge_cap = g->edges->itemCap;
	g->SynchronizeMatrix(g, m, edge_cap, edge_cap);

	return m;
}

Delta_Matrix Graph_GetAdjacencyMatrix
(
	const Graph *g,
	bool transposed
) {
	return Graph_GetRelationMatrix(g, GRAPH_NO_RELATION, transposed);
}

// returns true if relationship matrix 'r' contains multi-edge entries,
// false otherwise
bool Graph_RelationshipContainsMultiEdge
(
	const Graph *g,
	RelationID r,
	bool transpose
) {
	ASSERT(Graph_RelationTypeCount(g) > r);

	GrB_Index nvals;
	Delta_Matrix R = Graph_GetMultiEdgeRelationMatrix(g, r);
	Delta_Matrix_nvals(&nvals, R);

	return nvals > 0;
}

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
	GraphStatistics_FreeInternals(&g->stats);

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

	int res;
	UNUSED(res);

	if(g->_writelocked) Graph_ReleaseLock(g);
	res = pthread_rwlock_destroy(&g->_rwlock);
	ASSERT(res == 0);

	rm_free(g);
}

void Graph_Free
(
	Graph *g
) {
	_Graph_Free(g, true);
}

void Graph_PartialFree
(
	Graph *g
) {
	_Graph_Free(g, false);
}
