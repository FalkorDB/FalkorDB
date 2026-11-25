/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

// the v3 entity writers


#include "../../../RG.h"
#include "../../effects.h"
#include "../../effects_internal.h"
#include "write_v3.h"
#include "../../../query_ctx.h"
#include "../../../util/identifier_limits.h"
#include "../../../graph/graph.h"
#include "../../../graph/graphcontext.h"

#include "../../effects_v3_group.h"


// stage one attribute of an entity's update into the v3 accumulator
//
// All three per-attribute writers converge here: add, update and remove are one
// record family in v2 and one shape in v3. A v3 record's shape is the entity's
// WHOLE updated attribute set, so this stages rather than emits.
//
// REMOVE-ALL IS STATED EXPLICITLY, not as a sentinel. v2 writes ATTRIBUTE_ID_ALL
// as the attribute id and lets apply special-case it; v3 cannot, because the
// attribute ids ARE the record's shape. `SET n = {} SET n.x = 1` would then
// produce a shape mixing the two with nothing to say which applied first, and a
// dedicated record has the same problem between records. Stating every removed
// attribute explicitly needs no ordering, because it resolves to the end state
// rather than replaying operations.
static void _StageV3Update
(
	EffectsBuffer *buff,          // effect buffer
	GraphEntity *entity,          // entity being updated
	AttributeID attr_id,          // attribute, or ATTRIBUTE_ID_ALL
	SIValue value,                // value; null for a removal
	GraphEntityType entity_type,  // node or edge
	const LabelID *lbls,          // node labels, or NULL for an edge
	uint16_t n_labels             // how many
) {
	EffectType opcode = (entity_type == GETYPE_NODE)
		? EFFECT_UPDATE_NODE
		: EFFECT_UPDATE_EDGE;

	RelationID rel = (entity_type == GETYPE_NODE)
		? 0
		: Edge_GetRelationID((Edge *)entity);

	EntityID id = ENTITY_GET_ID(entity);

	if(attr_id == ATTRIBUTE_ID_ALL) {
		AttributeSet attrs = *entity->attributes;
		uint16_t n = AttributeSet_Count(attrs);

		for(uint16_t i = 0; i < n; i++) {
			AttributeID a_id;

			// 'v' is a required out parameter and is deliberately dropped:
			// this is the remove-all arm, so every attribute is staged with
			// a null value whatever it currently holds
			SIValue v;

			AttributeSet_GetIdx(attrs, i, &a_id, &v);
			EffectsV3Grouping_StageUpdate(EffectsBuffer_V3(buff), opcode, id, lbls,
					n_labels, rel, a_id, SI_NullVal());
		}
		return;
	}

	EffectsV3Grouping_StageUpdate(EffectsBuffer_V3(buff), opcode, id, lbls, n_labels, rel,
			attr_id, value);
}

// file a label vector into the v3 accumulator
//
// v3 states the LABEL SET and the node ids rather than a serialized GraphBLAS
// vector, which is the point of the record changing: v2's blob couples the wire
// to whatever GraphBLAS each engine was built against. The vector is named with
// its label and has already had redundancies stripped upstream
// (staged_updates.c), so every node in it genuinely gains or loses the label.
static void _StageV3Labels
(
	EffectsBuffer *buff,  // effect buffer
	GrB_Vector nodes,     // nodes the label applies to
	EffectType opcode     // SET_LABELS or REMOVE_LABELS
) {
	// GrB_get with GrB_NAME copies into the buffer, so pass a char array, not
	// &ptr
	char lbl_name[MAX_IDENTIFIER_LEN + 1] = {0};
	GrB_OK(GrB_get(nodes, lbl_name, GrB_NAME));

	GraphContext *gc = QueryCtx_GetGraphCtx();
	const Schema *sch = GraphContext_GetSchema(gc, lbl_name, SCHEMA_NODE);
	ASSERT(sch != NULL);

	LabelID lbl = Schema_GetID(sch);

	GxB_Iterator it;
	GrB_OK(GxB_Iterator_new(&it));
	GrB_OK(GxB_Vector_Iterator_attach(it, nodes, NULL));

	// NO FAST PATH FOR THE GROUP LOOKUP, though every node in this vector does
	// share one group: (opcode, this one label, no attributes). _group_for
	// already memoizes the group its last lookup found, so after the first node
	// the lookup itself costs a single comparison.
	//
	// What a bulk path WOULD save is the rest of AddNode's per-node work - a
	// memcpy and a sort of a one-element label array. Unmeasured, and small
	// against the vector iteration driving it
	GrB_Info info = GxB_Vector_Iterator_seek(it, 0);
	while(info != GxB_EXHAUSTED) {
		GrB_Index node_id = GxB_Vector_Iterator_getIndex(it);
		EffectsV3Grouping_AddNode(EffectsBuffer_V3(buff), opcode, &lbl, 1, node_id,
				NULL, NULL, 0);
		info = GxB_Vector_Iterator_next(it);
	}

	GrB_free(&it);
}


// an entity's attributes, flattened into the two parallel arrays the
// accumulator takes
//
// SIZED BY THE ENTITY. A fixed AttributeID[256] with the count clamped to match
// silently dropped every attribute past the 256th - measured on a live pair at
// 300 properties, replica 256, with no error, log or resync.
//
// Not a VLA either: n is a uint16_t, so 65,535 SIValues is a megabyte of stack.
// Small sets stay in the struct's own scratch, larger go to the heap - which is
// why the scratch lives HERE and not at the call site: whether 'ids' is the
// scratch is what _FlatAttrs_Free reads to decide whether to free it, so the
// two cannot be separated.
#define ATTRS_STACK 64

typedef struct {
	AttributeID id_buf[ATTRS_STACK];  // scratch, used while n <= ATTRS_STACK
	SIValue     val_buf[ATTRS_STACK];
	AttributeID *ids;                 // the ids: 'id_buf', or the heap
	SIValue     *vals;                // their values, parallel to 'ids'
	uint16_t    n;                    // how many
} FlatAttrs;

static void _FlatAttrs_Fill
(
	FlatAttrs *f,       // uninitialised; filled here
	AttributeSet attrs  // the entity's attributes
) {
	f->n = AttributeSet_Count(attrs);

	if(f->n > ATTRS_STACK) {
		f->ids  = rm_malloc(sizeof(AttributeID) * f->n);
		f->vals = rm_malloc(sizeof(SIValue) * f->n);
	} else {
		f->ids  = f->id_buf;
		f->vals = f->val_buf;
	}

	for(uint16_t i = 0; i < f->n; i++) {
		AttributeSet_GetIdx(attrs, i, f->ids + i, f->vals + i);
	}
}

static void _FlatAttrs_Free
(
	FlatAttrs *f  // from _FlatAttrs_Fill
) {
	if(f->ids != f->id_buf) {
		rm_free(f->ids);
		rm_free(f->vals);
	}
}

void EffectsWriteV3_CreateNode
(
	EffectsBuffer *buff,    // effect buffer
	const Node *n,          // node created
	const LabelID *labels,  // node labels
	ushort label_count      // number of labels
) {
	FlatAttrs f;
	_FlatAttrs_Fill(&f, *n->attributes);

	EffectsV3Grouping_AddNode(EffectsBuffer_V3(buff), EFFECT_CREATE_NODE, labels,
			label_count, ENTITY_GET_ID(n), f.ids, f.vals, f.n);
	EffectsBuffer_IncEffectCount(buff);

	_FlatAttrs_Free(&f);
}


void EffectsWriteV3_CreateEdge
(
	EffectsBuffer *buff,  // effect buffer
	const Edge *edge      // edge created
) {
	FlatAttrs f;
	_FlatAttrs_Fill(&f, *edge->attributes);

	EffectsV3Grouping_AddEdge(EffectsBuffer_V3(buff), EFFECT_CREATE_EDGE,
			Edge_GetRelationID(edge), ENTITY_GET_ID(edge),
			Edge_GetSrcNodeID(edge), Edge_GetDestNodeID(edge),
			f.ids, f.vals, f.n);
	EffectsBuffer_IncEffectCount(buff);

	_FlatAttrs_Free(&f);
}


void EffectsWriteV3_DeleteNode
(
	EffectsBuffer *buff,  // effect buffer
	const Node *node      // node deleted
) {
	// the labels the node ACTUALLY held, read while it is still alive -
	// GraphHub_DeleteNodes records the effect before Graph_DeleteNodes, so
	// the label matrices are still intact here. A replica needs them to
	// clear the right label-scoped index documents
	Graph *g = QueryCtx_GetGraph();
	uint lbl_count;
	NODE_GET_LABELS(g, node, lbl_count);

	EffectsV3Grouping_AddNode(EffectsBuffer_V3(buff), EFFECT_DELETE_NODE, labels,
			(uint16_t)lbl_count, ENTITY_GET_ID(node), NULL, NULL, 0);
	EffectsBuffer_IncEffectCount(buff);
}


void EffectsWriteV3_DeleteEdge
(
	EffectsBuffer *eb,  // effect buffer
	const Edge *edge    // edge deleted
) {
	// the type is captured HERE, while the edge still carries it. v3
	// groups deleted edges by relationship type and the edge is gone by
	// the time the payload is built
	EffectsV3Grouping_AddEdge(EffectsBuffer_V3(eb), EFFECT_DELETE_EDGE,
			Edge_GetRelationID(edge), ENTITY_GET_ID(edge),
			Edge_GetSrcNodeID(edge), Edge_GetDestNodeID(edge),
			NULL, NULL, 0);
	EffectsBuffer_IncEffectCount(eb);
}


void EffectsWriteV3_UpdateEntity
(
	EffectsBuffer *buff,         // effect buffer
	GraphEntity *entity,         // updated entity
	AttributeID attr_id,         // updated attribute, or ATTRIBUTE_ID_ALL
	SIValue value,               // value; a null is a removal
	GraphEntityType entity_type  // entity type
) {
	// TWO CALLS RATHER THAN ONE, because NODE_GET_LABELS declares a
	// variable-length array named 'labels' in the scope it expands into.
	// Hoisting a pointer to it out of the branch to reach a single call below
	// would leave that pointer dangling the moment the branch closes.
	if(entity_type == GETYPE_NODE) {
		Graph *g = QueryCtx_GetGraph();
		uint n;
		NODE_GET_LABELS(g, (Node *)entity, n);

		_StageV3Update(buff, entity, attr_id, value, entity_type, labels,
				(uint16_t)n);
	} else {
		_StageV3Update(buff, entity, attr_id, value, entity_type, NULL, 0);
	}

	EffectsBuffer_IncEffectCount(buff);
}


void EffectsWriteV3_Labels
(
	EffectsBuffer *buff,  // effect buffer
	GrB_Vector nodes,     // nodes the label applies to
	EffectType opcode     // SET_LABELS or REMOVE_LABELS
) {
	_StageV3Labels(buff, nodes, opcode);
	EffectsBuffer_IncEffectCount(buff);
}


void EffectsWriteV3_NewSchema
(
	EffectsBuffer *buff,      // effect buffer
	const char *schema_name,  // name of the schema
	SchemaType st             // type of the schema
) {
	// v3 carries the id as well as the name, so the replica can assert the
	// id it would assign matches - a numbering disagreement is otherwise
	// introduced by a record that cannot report it
	GraphContext *gc = QueryCtx_GetGraphCtx();
	const Schema *sch = GraphContext_GetSchema(gc, schema_name, st);
	ASSERT(sch != NULL);

	EffectsV3Grouping_AddSchema(EffectsBuffer_V3(buff), st, Schema_GetID(sch),
			schema_name);
	EffectsBuffer_IncEffectCount(buff);
}


void EffectsWriteV3_NewAttribute
(
	EffectsBuffer *buff,  // effect buffer
	const char *attr      // attribute name
) {
	GraphContext *gc = QueryCtx_GetGraphCtx();
	AttributeID id = GraphContext_GetAttributeID(gc, attr);
	ASSERT(id != ATTRIBUTE_ID_NONE);

	EffectsV3Grouping_AddAttribute(EffectsBuffer_V3(buff), id, attr);
	EffectsBuffer_IncEffectCount(buff);
}
