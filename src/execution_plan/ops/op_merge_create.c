/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "op_merge_create.h"
#include "../../util/arr.h"
#include "../../query_ctx.h"
#include "../../errors/errors.h"

// forward declarations
static Record MergeCreateConsume(OpBase *opBase);
static OpBase *MergeCreateClone(const ExecutionPlan *plan, const OpBase *opBase);
static OpResult MergeCreateInit(OpBase* opBase);
static void MergeCreateFree(OpBase *opBase);

// convert a graph entity's components into an identifying hash code
static void _IncrementalHashEntity
(
	XXH64_state_t *state,
	const char **labels,
	uint label_count,
	AttributeSet *set
) {
	AttributeSet _set = *set ;

	// update hash with label if one is provided
	XXH_errorcode res ;
	UNUSED (res) ;

	for (uint i = 0; i < label_count; i++) {
		res = XXH64_update (state, labels[i], strlen (labels[i])) ;
		ASSERT (res != XXH_ERROR) ;
	}

	XXH64_hash_t set_hash = AttributeSet_HashCode (_set) ;
	res = XXH64_update (state, &set_hash, sizeof (set_hash)) ;
	ASSERT (res != XXH_ERROR) ;
}

// revert the most recent set of buffered creations and free any allocations
static void _RollbackPendingCreations
(
	OpMergeCreate *op
) {
	uint nodes_to_create_count = array_len(op->pending.nodes.nodes_to_create);
	for(uint i = 0; i < nodes_to_create_count; i++) {
		AttributeSet props = array_pop(op->pending.nodes.node_attributes);
		AttributeSet_Free(&props);
	}

	uint edges_to_create_count = array_len(op->pending.edges);
	for(uint i = 0; i < edges_to_create_count; i++) {
		PendingEdgeCreations *pending_edge = op->pending.edges + i;
		AttributeSet props = array_pop(pending_edge->edge_attributes);
		AttributeSet_Free(&props);
	}
}

OpBase *NewMergeCreateOp
(
	const ExecutionPlan *plan,
	NodeCreateCtx *nodes,
	EdgeCreateCtx *edges
) {
	OpMergeCreate *op = rm_calloc(1, sizeof(OpMergeCreate));

	op->records         = array_new(Record, 32);  // accumulated records
	op->hash_state      = XXH64_createState();    // create a hash state
	op->unique_entities = raxNew();               // create a map to unique pending creations

	// insert one NULL value to terminate execution of the op
	array_append(op->records, NULL);

	// set our Op operations
	OpBase_Init((OpBase *)op, OPType_MERGE_CREATE, "MergeCreate",
			MergeCreateInit, MergeCreateConsume, NULL, NULL, MergeCreateClone,
			MergeCreateFree, true, plan);

	uint node_blueprint_count = array_len(nodes);
	uint edge_blueprint_count = array_len(edges);

	// construct the array of IDs this operation modifies
	for(uint i = 0; i < node_blueprint_count; i ++) {
		NodeCreateCtx *n = nodes + i;
		n->node_idx = OpBase_Modifies((OpBase *)op, n->alias);
	}

	for(uint i = 0; i < edge_blueprint_count; i ++) {
		EdgeCreateCtx *e = edges + i;
		e->edge_idx = OpBase_Modifies((OpBase *)op, e->alias);
		bool aware;
		UNUSED(aware);

		aware = OpBase_AliasMapping((OpBase *)op, e->src, &e->src_idx);
		ASSERT(aware == true);

		aware = OpBase_AliasMapping((OpBase *)op, e->dest, &e->dest_idx);
		ASSERT(aware == true);
	}

	// prepare all creation variables
	NewPendingCreationsContainer(&op->pending, nodes, edges);

	return (OpBase *)op;
}

static OpResult MergeCreateInit
(
	OpBase* opBase
) {
	OpMergeCreate *op = (OpMergeCreate *)opBase;
	op->gc = QueryCtx_GetGraphCtx();
	return OP_OK;
}

// prepare all creations associated with the current Record
// returns false and do not buffer data if every entity to create for this Record
// has been created in a previous call
static bool _CreateEntities
(
	OpMergeCreate *op,
	Record r,
	GraphContext *gc
) {
	XXH_errorcode res = XXH64_reset(op->hash_state, 0); // reset hash state
	UNUSED(res);
	ASSERT(res != XXH_ERROR);

	//--------------------------------------------------------------------------
	// hash nodes
	//--------------------------------------------------------------------------

	uint nodes_to_create_count = array_len(op->pending.nodes.nodes_to_create);

	for(uint i = 0; i < nodes_to_create_count; i++) {
		// get specified node to create
		NodeCreateCtx *n = op->pending.nodes.nodes_to_create + i;

		// convert properties
		PropertyMap *map = n->properties;
		AttributeSet converted_attr = NULL;
		if(map != NULL) {
			ConvertPropertyMap(gc, &converted_attr, r, map, true);
		}

		// update the hash code with this entity
		uint label_count = array_len(n->labels);
		_IncrementalHashEntity(op->hash_state, n->labels, label_count,
				&converted_attr);

		// save attributes
		array_append(op->pending.nodes.node_attributes, converted_attr);

		// save labels
		array_append(op->pending.nodes.node_labels, n->labelsId);
	}

	//--------------------------------------------------------------------------
	// hash edges
	//--------------------------------------------------------------------------

	uint edges_to_create_count = array_len(op->pending.edges);

	for(uint i = 0; i < edges_to_create_count; i++) {
		PendingEdgeCreations *pending_edge = op->pending.edges + i;
		// get specified edge to create
		EdgeCreateCtx *e = &pending_edge->edges_to_create;

		// retrieve source and dest nodes
		Node *src_node = Record_GetNode(r, e->src_idx);
		Node *dest_node = Record_GetNode(r, e->dest_idx);

		// convert query-level properties
		PropertyMap *map = e->properties;
		AttributeSet converted_attr = NULL;
		if(map != NULL) {
			ConvertPropertyMap(gc, &converted_attr, r, map, true);
		}

		// update the hash code with this entity, an edge is represented by its
		// relation, properties, source and destination nodes.
		// note: unbounded nodes were already presented to the hash.
		// incase node has its internal attribute-set
		// this means the node has been retrieved from the graph
		// i.e. bounded node

		_IncrementalHashEntity(op->hash_state, &e->relation, 1, &converted_attr);

		// hash source node
		if(src_node != NULL) {
			EntityID id = ENTITY_GET_ID(src_node);
			ASSERT(id != INVALID_ENTITY_ID);
			void *data = &id;
			size_t len = sizeof(id);
			res = XXH64_update(op->hash_state, data, len);
			ASSERT(res != XXH_ERROR);
		}

		// hash dest node
		if(dest_node != NULL) {
			EntityID id = ENTITY_GET_ID(dest_node);
			ASSERT(id != INVALID_ENTITY_ID);
			void *data = &id;
			size_t len = sizeof(id);
			res = XXH64_update(op->hash_state, data, len);
			ASSERT(res != XXH_ERROR);
		}

		// save attributes
		array_append(pending_edge->edge_attributes, converted_attr);
	}

	// finalize the hash value for all processed creations
	XXH64_hash_t const hash = XXH64_digest(op->hash_state);

	// check if any creations are unique
	bool should_create_entities = raxTryInsert(op->unique_entities,
			(unsigned char *)&hash, sizeof(hash), NULL, NULL);

	// if no entity to be created is unique
	// roll back all the creations that have just been prepared
	if(should_create_entities) {
		// reserve node ids for edges creation
		for(uint i = 0; i < nodes_to_create_count; i++) {
			NodeCreateCtx *n = op->pending.nodes.nodes_to_create + i;

			Node newNode = Graph_ReserveNode(gc->g);

			// add new node to Record and save a reference to it
			Node *node_ref = Record_AddNode(r, n->node_idx, newNode);

			// save node for later insertion
			array_append(op->pending.nodes.created_nodes, node_ref);
		}

		// updated edges with reserved node ids
		for(uint i = 0; i < edges_to_create_count; i++) {
			PendingEdgeCreations *pending_edge = op->pending.edges + i;
			EdgeCreateCtx *ctx = &pending_edge->edges_to_create;

			// retrieve source and dest nodes
			Node *src_node  = Record_GetNode(r, ctx->src_idx);
			Node *dest_node = Record_GetNode(r, ctx->dest_idx);

			if(!src_node || !dest_node) {
				ErrorCtx_RaiseRuntimeException(
						"Failed to create relationship; endpoint was not found."
				);
			}

			// create edge
			Edge newEdge = {0};
			newEdge.relationship = ctx->relation;
			Edge *e = Record_AddEdge(r, ctx->edge_idx, newEdge);
			Edge_SetSrcNodeID(e, ENTITY_GET_ID(src_node));
			Edge_SetDestNodeID(e, ENTITY_GET_ID(dest_node));

			// save edge for later insertion
			array_append(pending_edge->created_edges, e);
		}
	} else {
		_RollbackPendingCreations(op);
	}

	return should_create_entities;
}

// emit a populated Record
static Record _handoff
(
	OpMergeCreate *op
) {
	return array_pop(op->records);
}

// operation consume method
// depending on the operation's mode 'handoff_mode'
// this function will either emit records (handoff_mode == true)
// or (handoff_mode == false) compute future changes
static Record MergeCreateConsume
(
	OpBase *opBase
) {
	OpMergeCreate *op = (OpMergeCreate *)opBase;
	Record r;

	// return mode, all data was consumed
	if(op->handoff_mode) return _handoff(op);

	// consume mode
	if(!opBase->childCount) {
		// no child operation to call
		r = OpBase_CreateRecord(opBase);

		// buffer all entity creations
		// if this operation has no children, it should always have unique creations
		bool entities_created = _CreateEntities(op, r, op->gc);
		ASSERT(entities_created == true);

		// save record for later use
		array_append(op->records, r);

		r = NULL; // record scheduled for creation nullify it
	} else {
		// pull record from child
		r = OpBase_Consume(opBase->children[0]);
		if(r) {
			// create entities
			if(_CreateEntities(op, r, op->gc)) {
				// save record for later use
				array_append(op->records, r);

				r = NULL; // record scheduled for creation nullify it
			}
		}
	}

	// return NULL if record is scheduled for creation
	// return the input record in case it represents a duplicate
	// which will later on be matched
	return r;
}

// commit accumulated changes and switch to Record handoff mode
void MergeCreate_Commit
(
	OpBase *opBase
) {
	ASSERT(opBase != NULL);

	OpMergeCreate *op = (OpMergeCreate *)opBase;

	// switch to hand-off mode
	op->handoff_mode = true;

	// done reading, we're not going to call consume any longer
	// there might be operations e.g. index scan that need to free
	// index R/W lock, as such free all execution plan operation up the chain
	if(opBase->childCount > 0) OpBase_PropagateReset(opBase->children[0]);

	// create entities
	CommitNewEntities(&op->pending);
}

// clone operation
static OpBase *MergeCreateClone
(
	const ExecutionPlan *plan,
	const OpBase *opBase
) {
	ASSERT(plan         != NULL);
	ASSERT(opBase       != NULL);
	ASSERT(opBase->type == OPType_MERGE_CREATE);

	OpMergeCreate *op = (OpMergeCreate *)opBase;

	NodeCreateCtx *nodes;
	array_clone_with_cb(nodes, op->pending.nodes.nodes_to_create,
			NodeCreateCtx_Clone);

	EdgeCreateCtx *edges = array_new(EdgeCreateCtx,
			array_len(op->pending.edges));

	for(uint i = 0; i < array_len(op->pending.edges); i++) {
		EdgeCreateCtx ctx =
			EdgeCreateCtx_Clone(op->pending.edges[i].edges_to_create);
		array_append(edges, ctx);
	}

	return NewMergeCreateOp(plan, nodes, edges);
}

// free any memory allocated by operation
static void MergeCreateFree
(
	OpBase *ctx
) {
	OpMergeCreate *op = (OpMergeCreate *)ctx;

	if(op->records) {
		uint rec_count = array_len(op->records);
		for(uint i = 1; i < rec_count; i++) OpBase_DeleteRecord(op->records+i);
		array_free(op->records);
		op->records = NULL;
	}

	if(op->unique_entities) {
		raxFree(op->unique_entities);
		op->unique_entities = NULL;
	}

	if(op->hash_state) {
		XXH64_freeState(op->hash_state);
		op->hash_state = NULL;
	}

	PendingCreationsFree(&op->pending);
}

