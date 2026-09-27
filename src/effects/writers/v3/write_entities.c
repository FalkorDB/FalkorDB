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

// stage an update, reading a node's labels in the scope the macro needs
//
// NODE_GET_LABELS declares a variable-length array named `labels` in the
// enclosing scope, so the staging has to happen where that array is still
// alive rather than through a pointer that outlives it
#define STAGE_V3_UPDATE(buff, entity, attr_id, value, entity_type)          \
	do {                                                                    \
		if((entity_type) == GETYPE_NODE) {                                  \
			Graph *_g = QueryCtx_GetGraph();                                \
			uint _n;                                                        \
			NODE_GET_LABELS(_g, (Node *)(entity), _n);                      \
			_StageV3Update((buff), (entity), (attr_id), (value),            \
					(entity_type), labels, (uint16_t)_n);                   \
		} else {                                                            \
			_StageV3Update((buff), (entity), (attr_id), (value),            \
					(entity_type), NULL, 0);                                \
		}                                                                   \
	} while(0)

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
	// a real buffer, not a pointer's address. GrB_get with GrB_NAME COPIES the
	// name into what you hand it, so passing &lbl_name wrote the label's
	// characters into the pointer variable itself and the next line
	// dereferenced them as an address - strcmp on a pointer built out of the
	// label's own letters. The (char *) cast I had here is what silenced the
	// char** / char* mismatch that would otherwise have caught it
	char lbl_name[MAX_IDENTIFIER_LEN + 1] = {0};
	GrB_OK(GrB_get(nodes, lbl_name, GrB_NAME));

	GraphContext *gc = QueryCtx_GetGraphCtx();
	const Schema *sch = GraphContext_GetSchema(gc, lbl_name, SCHEMA_NODE);
	ASSERT(sch != NULL);

	LabelID lbl = Schema_GetID(sch);

	GxB_Iterator it;
	GxB_Iterator_new(&it);
	GrB_OK(GxB_Vector_Iterator_attach(it, nodes, NULL));

	GrB_Info info = GxB_Vector_Iterator_seek(it, 0);
	while(info != GxB_EXHAUSTED) {
		GrB_Index node_id = GxB_Vector_Iterator_getIndex(it);
		EffectsV3Grouping_AddNode(EffectsBuffer_V3(buff), opcode, &lbl, 1, node_id,
				NULL, NULL, 0);
		info = GxB_Vector_Iterator_next(it);
	}

	GrB_free(&it);
}


// flatten an entity's attributes into parallel arrays for the accumulator
//
// SIZED BY THE ENTITY. A fixed AttributeID[256] with the count clamped to match
// silently dropped every attribute past the 256th - measured on a live pair at
// 300 properties, replica 256, with no error, log or resync.
//
// Not a VLA either: n_attrs is a uint16_t, so 65,535 SIValues is a megabyte of
// stack. Small sets stay on the stack, larger go to the heap.
#define ATTRS_STACK 64

static uint16_t _flatten_attrs
(
	AttributeSet attrs,    // the entity's attributes
	AttributeID **ids,     // out: ids, caller releases
	SIValue **vals,        // out: values, caller releases
	AttributeID *id_buf,   // caller's ATTRS_STACK-sized scratch
	SIValue *val_buf       // caller's ATTRS_STACK-sized scratch
) {
	uint16_t n = AttributeSet_Count(attrs);

	if(n > ATTRS_STACK) {
		*ids  = rm_malloc(sizeof(AttributeID) * n);
		*vals = rm_malloc(sizeof(SIValue) * n);
	} else {
		*ids  = id_buf;
		*vals = val_buf;
	}

	for(uint16_t i = 0; i < n; i++) {
		AttributeSet_GetIdx(attrs, i, (*ids) + i, (*vals) + i);
	}

	return n;
}

static void _release_attrs
(
	AttributeID *ids,      // ids from _flatten_attrs
	SIValue *vals,         // values from _flatten_attrs
	uint16_t n,            // how many
	AttributeID *id_buf    // the scratch that was offered
) {
	if(ids != id_buf) {
		rm_free(ids);
		rm_free(vals);
	}
}

void EffectsWriteV3_CreateNode
(
	EffectsBuffer *buff,    // effect buffer
	const Node *n,          // node created
	const LabelID *labels,  // node labels
	ushort label_count      // number of labels
) {
	AttributeID id_buf[ATTRS_STACK];
	SIValue     val_buf[ATTRS_STACK];
	AttributeID *ids;
	SIValue     *vals;

	uint16_t n_attrs = _flatten_attrs(*n->attributes, &ids, &vals, id_buf,
			val_buf);

	EffectsV3Grouping_AddNode(EffectsBuffer_V3(buff), EFFECT_CREATE_NODE, labels,
			label_count, ENTITY_GET_ID(n), ids, vals, n_attrs);
	EffectsBuffer_IncEffectCount(buff);

	_release_attrs(ids, vals, n_attrs, id_buf);
}


void EffectsWriteV3_CreateEdge
(
	EffectsBuffer *buff,  // effect buffer
	const Edge *edge      // edge created
) {
	AttributeID id_buf[ATTRS_STACK];
	SIValue     val_buf[ATTRS_STACK];
	AttributeID *ids;
	SIValue     *vals;

	uint16_t n_attrs = _flatten_attrs(*edge->attributes, &ids, &vals, id_buf,
			val_buf);

	EffectsV3Grouping_AddEdge(EffectsBuffer_V3(buff), EFFECT_CREATE_EDGE,
			Edge_GetRelationID(edge), ENTITY_GET_ID(edge),
			Edge_GetSrcNodeID(edge), Edge_GetDestNodeID(edge),
			ids, vals, n_attrs);
	EffectsBuffer_IncEffectCount(buff);

	_release_attrs(ids, vals, n_attrs, id_buf);
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
	STAGE_V3_UPDATE(buff, entity, attr_id, value, entity_type);
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
