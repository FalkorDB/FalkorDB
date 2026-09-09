/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "effects.h"
#include "effects_bytes.h"
#include "effects_internal.h"
#include "../query_ctx.h"
#include "../datatypes/map.h"
#include "../datatypes/vector.h"

// initial size of a buffer's block
#define EFFECTS_BUFFER_BLOCK_SIZE 62500

struct _EffectsBuffer {
	EffectsBytes *records;  // encoded records
	uint64_t n;             // number of effects in buffer
	uint8_t version;        // payload version this buffer emits
	bool owns_records;      // whether freeing the buffer frees the sink
};

// forward declarations

// write array to effects buffer
static void EffectsBuffer_WriteSIArray
(
	const SIValue *arr,  // array
	EffectsBuffer *buff  // effect buffer
);

// write map to effects buffer
static void EffectsBuffer_WriteSIMap
(
	const SIValue *map,  // map
	EffectsBuffer *buff  // effect buffer
);

// write vector to effects buffer
static void EffectsBuffer_WriteSIVector
(
	const SIValue *v,    // vector
	EffectsBuffer *buff  // effect buffer
);

// number of bytes the payload header occupies for a given version
//
// v2 is a bare version byte. v3 adds the flags byte, which is reserved whether
// or not compression is on - adding it later would cost another version bump
static size_t _EffectsBuffer_HeaderLen
(
	uint8_t version  // payload version
) {
	return (version >= 3) ? 2 : 1;
}

// write the payload header into dst, returning dst advanced past it
//
// the header is written here rather than when the buffer is created, which is
// where v2 wrote it. Two reasons, and the second is the one that forces it:
// v3's flags byte is not settled until the record stream is complete, and a v3
// payload's records cannot be emitted in arrival order at all - they are
// grouped, so nothing can precede them in the buffer
static unsigned char *_EffectsBuffer_WriteHeader
(
	const EffectsBuffer *eb,  // effects-buffer
	unsigned char *dst        // destination
) {
	*dst++ = eb->version;

	if(eb->version >= 3) {
		// flags; bit 0 = compressed. C does not compress yet, so this is 0,
		// but the byte is part of the format regardless
		*dst++ = 0;
	}

	return dst;
}

// write n bytes from ptr into effects-buffer
void EffectsBuffer_WriteBytes
(
	const void *ptr,   // data to write
	size_t n,          // number of bytes to write
	EffectsBuffer *eb  // effects-buffer
) {
	ASSERT (n   > 0) ;
	ASSERT (eb  != NULL) ;
	ASSERT (ptr != NULL) ;

	EffectsBytes_Write (eb->records, ptr, n) ;
}

// write a length-prefixed, NUL-terminated string
//
// THE SPEC NOW SAYS BYTE LENGTH, NOT strlen. The corrected rule is "the value's
// byte length + 1, then len bytes, the last a NUL", and a reader must use the
// length rather than call strlen, because a value may contain interior NULs.
// Rust produces them: openCypher's \uXXXX escape can encode one, so
// size('a<NUL>b') is 3 there where strlen would report 1.
//
// THIS STILL WRITES strlen + 1, deliberately, for three independent reasons.
// They are listed separately because they EXPIRE SEPARATELY - collapsing them
// into one "we cannot do this" would read as permanent, and none of them is:
//
//   * C cannot express such a value. It does not implement \uXXXX at all -
//     measured, C reports size 6 for the escape Rust reports 1 for, and 8 where
//     Rust reports 3 - so no input with an interior NUL can reach here.
//
//   * SIValue has no length for a string. `char *stringval` is the entire
//     representation (value.h), so the byte length does not exist to be
//     written. Conforming means changing a core type used everywhere, which is
//     not the effects path's to change.
//
//   * This writer is SHARED WITH v2, whose framing must not move. Changing it
//     alters shipped v2 bytes, so conforming would need a v3-only string
//     writer - a second SIValue codec, which is the one thing this file must
//     not grow.
//
// WHICH ONE EXPIRES WHEN:
//
//   the escape gap      ends the day C implements the unicode escape
//   the missing length  ends if anyone adds one to SIValue
//   the shared writer   ends only when v2 does
//
// So a reader arriving later should check which of the three still holds
// rather than assuming the conclusion survived.
//
// So C conforms by construction rather than by intent. The day C gains \uXXXX
// support this becomes a silent truncation on the wire, and the fix then is a
// length on SIValue rather than anything here.
void EffectsBuffer_WriteString
(
	const char *str,
	EffectsBuffer *eb
) {
	ASSERT(eb  != NULL);
	ASSERT(str != NULL);

	size_t l = strlen(str) + 1;
	EffectsBuffer_WriteBytes(&l, sizeof(size_t), eb);
	EffectsBuffer_WriteBytes(str, l, eb);
}

// writes a binary representation of v into Effect-Buffer
void EffectsBuffer_WriteSIValue
(
	const SIValue *v,
	EffectsBuffer *buff
) {
	ASSERT (v    != NULL) ;
	ASSERT (buff != NULL) ;

	// format:
	//    type
	//    value
	bool b;
	size_t len = 0;

	// set type to intern string incase the allocation type is intern
	// otherwise use v's original type
	SIType t = v->type;

	// write type
	EffectsBuffer_WriteBytes(&t, sizeof(SIType), buff);

	// write value
	switch(t) {
		case T_POINT:
			// write value to stream
			EffectsBuffer_WriteBytes (&v->point, sizeof (Point), buff) ;
			break ;

		case T_ARRAY:
			// write array to stream
			EffectsBuffer_WriteSIArray (v, buff) ;
			break ;

		case T_STRING:
		case T_INTERN_STRING:
			EffectsBuffer_WriteString (v->stringval, buff) ;
			break ;

		case T_BOOL:
			// write bool to stream
			b = SIValue_IsTrue (*v) ;
			EffectsBuffer_WriteBytes (&b, sizeof (bool), buff) ;
			break ;

		case T_INT64:
			// write int to stream
			EffectsBuffer_WriteBytes (&v->longval, sizeof (v->longval), buff) ;
			break ;

		case T_DOUBLE:
			// write double to stream
			EffectsBuffer_WriteBytes (&v->doubleval, sizeof (v->doubleval), buff) ;
			break ;

		case T_TIME:
		case T_DATE:
		case T_DATETIME:
		case T_DURATION:
			// write temporal time_t
			EffectsBuffer_WriteBytes (&v->datetimeval, sizeof (v->datetimeval), buff) ;
			break ;

		case T_NULL:
			// no additional data is required to represent NULL
			break ;

		case T_VECTOR_F32:
			EffectsBuffer_WriteSIVector (v, buff) ;
			break ;

		case T_MAP:
			EffectsBuffer_WriteSIMap (v, buff) ;
			break ;

		default:
			assert (false && "unknown SIValue type") ;
	}
}

// writes a binary representation of arr into Effect-Buffer
static void EffectsBuffer_WriteSIArray
(
	const SIValue *arr,  // array
	EffectsBuffer *buff  // effect buffer
) {
	// format:
	// number of elements
	// elements

	SIValue *elements = arr->array;
	uint32_t len = arr_len(elements);

	// write number of elements
	EffectsBuffer_WriteBytes(&len, sizeof(uint32_t), buff);

	// write each element
	for (uint32_t i = 0; i < len; i++) {
		EffectsBuffer_WriteSIValue(elements + i, buff);
	}
}

// writes a binary representation of map into Effect-Buffer
static void EffectsBuffer_WriteSIMap
(
	const SIValue *map,  // map
	EffectsBuffer *buff  // effect buffer
) {
	// format:
	// number of pairs
	// (key, value) pairs

	uint32_t len = Map_KeyCount (*map) ;

	// write number of pairs
	EffectsBuffer_WriteBytes (&len, sizeof (uint32_t), buff) ;

	// write each (key, value) pair
	for (uint32_t i = 0; i < len; i++) {
		SIValue key ;
		SIValue val ;
		Map_GetIdx (*map, i, &key, &val) ;

		EffectsBuffer_WriteSIValue (&key, buff) ;
		EffectsBuffer_WriteSIValue (&val, buff) ;
	}
}

// write vector to effects buffer
static void EffectsBuffer_WriteSIVector
(
	const SIValue *v,    // vector
	EffectsBuffer *buff  // effect buffer
) {
	// format:
	// number of elements
	// elements

	// write vector dimension
	uint32_t dim = SIVector_Dim(*v);
	EffectsBuffer_WriteBytes(&dim, sizeof(uint32_t), buff);

	// write vector elements
	void *elements   = SIVector_Elements(*v);
	size_t elem_size = sizeof(float);
	size_t n = dim * elem_size;

	if(n > 0) {
		EffectsBuffer_WriteBytes(elements, n, buff);
	}
}

// dump attributes to stream
static void EffectsBuffer_WriteAttributeSet
(
	const AttributeSet attrs,  // attribute set to write to stream
	EffectsBuffer *buff
) {
	//--------------------------------------------------------------------------
	// write attribute count
	//--------------------------------------------------------------------------

	ushort attr_count = AttributeSet_Count(attrs);
	EffectsBuffer_WriteBytes(&attr_count, sizeof(attr_count), buff);

	//--------------------------------------------------------------------------
	// write attributes
	//--------------------------------------------------------------------------

	for(ushort i = 0; i < attr_count; i++) {
		// get current attribute name and value
		SIValue attr ;
		AttributeID attr_id ;
		AttributeSet_GetIdx (attrs, i, &attr_id, &attr) ;

		// write attribute ID
		EffectsBuffer_WriteBytes (&attr_id, sizeof (AttributeID), buff) ;

		// write attribute value
		EffectsBuffer_WriteSIValue (&attr, buff) ;

		// free attribute
		SIValue_Free (attr) ;
	}
}

void EffectsBuffer_IncEffectCount
(
	EffectsBuffer *buff
) {
	ASSERT(buff != NULL);
	
	buff->n++;
}

// create a new effects-buffer
EffectsBuffer *EffectsBuffer_New
(
	void
) {
	EffectsBuffer *eb = rm_malloc(sizeof(EffectsBuffer));

	eb->n            = 0;
	eb->records      = EffectsBytes_New(EFFECTS_BUFFER_BLOCK_SIZE);
	eb->version      = EFFECTS_VERSION_EMIT;
	eb->owns_records = true;

	// note: no header is written here. v2 stamped its version byte at
	// construction; it is now written by EffectsBuffer_Buffer, so that the
	// records a buffer holds are only records
	return eb;
}

// reset effects-buffer
void EffectsBuffer_Reset
(
	EffectsBuffer *buff  // effects-buffer
) {
	ASSERT(buff != NULL);

	EffectsBytes_Clear(buff->records);

	buff->n       = 0;
	buff->version = EFFECTS_VERSION_EMIT;
}

// returns number of effects in buffer
//
// this counts EFFECTS, not records, and must keep doing so. It is the
// predicate deciding whether a query replicates at all, and it is the divisor
// in the average-modification-time comparison against EFFECTS_THRESHOLD
// (cmd_query.c), whose units are effects. Under v3 one record covers every
// entity of its shape, so a record count would both under-report a query that
// changed something and inflate the average until the threshold flipped
uint64_t EffectsBuffer_Length
(
	const EffectsBuffer *buff  // effects-buffer
) {
	ASSERT(buff != NULL);

	return buff->n;
}

// get a copy of effects-buffer internal buffer
unsigned char *EffectsBuffer_Buffer
(
	const EffectsBuffer *eb,  // effects-buffer
	size_t *n                 // size of returned buffer
) {
	ASSERT(eb != NULL);

	//--------------------------------------------------------------------------
	// determine required buffer size
	//--------------------------------------------------------------------------

	size_t hdr = _EffectsBuffer_HeaderLen(eb->version);
	size_t l   = hdr + EffectsBytes_Len(eb->records);

	//--------------------------------------------------------------------------
	// allocate buffer and populate
	//--------------------------------------------------------------------------

	unsigned char *buffer = rm_malloc(sizeof(unsigned char) * l);
	unsigned char *offset = _EffectsBuffer_WriteHeader(eb, buffer);

	EffectsBytes_CopyInto(eb->records, offset);

	*n = l;
	return buffer;
}

//------------------------------------------------------------------------------
// effects creation API
//------------------------------------------------------------------------------

// add a node creation effect to buffer
void EffectsBuffer_AddCreateNodeEffect
(
	EffectsBuffer *buff,    // effect buffer
	const Node *n,          // node created
	const LabelID *labels,  // node labels
	ushort label_count      // number of labels
) {
	//--------------------------------------------------------------------------
	// effect format:
	// effect type
	// label count
	// labels
	// attribute count
	// attributes (id,value) pair
	//--------------------------------------------------------------------------
	
	//--------------------------------------------------------------------------
	// update query stats
	//--------------------------------------------------------------------------

	ResultSetStatistics *stats = QueryCtx_GetResultSetStatistics () ;
	stats->nodes_created++ ;
	stats->labels_added   += label_count ;
	stats->properties_set += AttributeSet_Count (*n->attributes) ;

	EffectType t = EFFECT_CREATE_NODE;
	EffectsBuffer_WriteBytes(&t, sizeof(t), buff);

	//--------------------------------------------------------------------------
	// write label count
	//--------------------------------------------------------------------------

	EffectsBuffer_WriteBytes(&label_count, sizeof(label_count), buff);

	//--------------------------------------------------------------------------
	// write labels
	//--------------------------------------------------------------------------

	if(label_count > 0) {
		EffectsBuffer_WriteBytes(labels, sizeof(LabelID) * label_count, buff);
	}

	//--------------------------------------------------------------------------
	// write attribute set
	//--------------------------------------------------------------------------

	const AttributeSet attrs = GraphEntity_GetAttributes((const GraphEntity*)n);
	EffectsBuffer_WriteAttributeSet(attrs, buff);

	EffectsBuffer_IncEffectCount(buff);
}

// add a edge creation effect to buffer
void EffectsBuffer_AddCreateEdgeEffect
(
	EffectsBuffer *buff,  // effect buffer
	const Edge *edge      // edge created
) {
	//--------------------------------------------------------------------------
	// effect format:
	// effect type
	// relationship count
	// relationships
	// src node ID
	// dest node ID
	// attribute count
	// attributes (id,value) pair
	//--------------------------------------------------------------------------
	
	ResultSetStatistics *stats = QueryCtx_GetResultSetStatistics () ;
	stats->relationships_created++ ;
	stats->properties_set += AttributeSet_Count (*edge->attributes) ;

	// encoded edge struct
	#pragma pack(push, 1)
	struct {
		EffectType t ;
		uint16_t rel_count ;
		RelationID r ;
		NodeID src_id ;
		NodeID dest_id ;
	} _create_edge_desc;
	#pragma pack(pop)

	//--------------------------------------------------------------------------
	// populate & write edge
	//--------------------------------------------------------------------------

	_create_edge_desc.t         = EFFECT_CREATE_EDGE ;
	_create_edge_desc.rel_count = 1 ;
	_create_edge_desc.r         = Edge_GetRelationID (edge) ;
	_create_edge_desc.src_id    = Edge_GetSrcNodeID  (edge) ;
	_create_edge_desc.dest_id   = Edge_GetDestNodeID (edge) ;

	EffectsBuffer_WriteBytes (&_create_edge_desc, sizeof (_create_edge_desc),
			buff) ;

	//--------------------------------------------------------------------------
	// write attribute set 
	//--------------------------------------------------------------------------

	const AttributeSet attrs =
		GraphEntity_GetAttributes ((const GraphEntity*)edge) ;

	EffectsBuffer_WriteAttributeSet (attrs, buff) ;

	EffectsBuffer_IncEffectCount (buff) ;
}

// add a node deletion effect to buffer
void EffectsBuffer_AddDeleteNodeEffect
(
	EffectsBuffer *buff,  // effect buffer
	const Node *node      // node deleted
) {
	//--------------------------------------------------------------------------
	// effect format:
	//    effect type
	//    node ID
	//--------------------------------------------------------------------------

	// update query statistics
	ResultSetStatistics *stats = QueryCtx_GetResultSetStatistics () ;
	stats->nodes_deleted++ ;

	#pragma pack(push, 1)
	struct {
		EffectType t;
		EntityID id;
	} _delete_node_desc ;
	#pragma pack(pop)

	_delete_node_desc.t  = EFFECT_DELETE_NODE ;
	_delete_node_desc.id = ENTITY_GET_ID (node) ;

	EffectsBuffer_WriteBytes (&_delete_node_desc, sizeof(_delete_node_desc),
			buff) ;

	EffectsBuffer_IncEffectCount (buff) ;
}

// add a edge deletion effect to buffer
void EffectsBuffer_AddDeleteEdgeEffect
(
	EffectsBuffer *eb,  // effect buffer
	const Edge *edge    // edge deleted
) {
	//--------------------------------------------------------------------------
	// effect format:
	//    effect type
	//    edge ID
	//    relation ID
	//    src ID
	//    dest ID
	//--------------------------------------------------------------------------

	ResultSetStatistics *stats = QueryCtx_GetResultSetStatistics () ;
	stats->relationships_deleted++ ;

	// encoded edge struct
	#pragma pack(push, 1)
	struct {
		EffectType t ;
		EntityID id;
		RelationID r ;
		NodeID src_id ;
		NodeID dest_id ;
	} _delete_edge_desc;
	#pragma pack(pop)

	_delete_edge_desc.t       = EFFECT_DELETE_EDGE ;
	_delete_edge_desc.id      = ENTITY_GET_ID      (edge) ;
	_delete_edge_desc.r       = Edge_GetRelationID (edge) ;
	_delete_edge_desc.src_id  = Edge_GetSrcNodeID  (edge) ;
	_delete_edge_desc.dest_id = Edge_GetDestNodeID (edge) ;

	EffectsBuffer_WriteBytes (&_delete_edge_desc, sizeof(_delete_edge_desc),
			eb) ;

	EffectsBuffer_IncEffectCount(eb);
}

// add an entity update effect to buffer
static void EffectsBuffer_AddNodeUpdateEffect
(
	EffectsBuffer *buff,  // effect buffer
	Node *node,           // updated node
	AttributeID attr_id,  // updated attribute ID
 	SIValue value         // value
) {
	//--------------------------------------------------------------------------
	// effect format:
	//    effect type
	//    entity ID
	//    attribute id
	//    attribute value
	//--------------------------------------------------------------------------

	#pragma pack(push, 1)
	struct {
		EffectType t ;
		EntityID id;
		AttributeID attr_id ;
	} _update_node_desc;
	#pragma pack(pop)

	_update_node_desc.t       = EFFECT_UPDATE_NODE ;
	_update_node_desc.id      = ENTITY_GET_ID (node) ;
	_update_node_desc.attr_id = attr_id ;

	EffectsBuffer_WriteBytes (&_update_node_desc, sizeof(_update_node_desc),
			buff) ;

	//--------------------------------------------------------------------------
	// write attribute value
	//--------------------------------------------------------------------------

	EffectsBuffer_WriteSIValue (&value, buff) ;

	EffectsBuffer_IncEffectCount (buff) ;
}

// add an entity update effect to buffer
static void EffectsBuffer_AddEdgeUpdateEffect
(
	EffectsBuffer *buff,  // effect buffer
	Edge *edge,           // updated edge
	AttributeID attr_id,  // updated attribute ID
 	SIValue value         // value
) {
	//--------------------------------------------------------------------------
	// effect format:
	//    effect type
	//    edge ID
	//    relation ID
	//    src ID
	//    dest ID
	//    attribute count (=n)
	//    attributes (id,value) pair
	//--------------------------------------------------------------------------

	#pragma pack(push, 1)
	struct {
		EffectType t ;
		EntityID id;
		RelationID r;
		NodeID s;
		NodeID d;
		AttributeID attr_id ;
	} _update_edge_desc;
	#pragma pack(pop)

	_update_edge_desc.t       = EFFECT_UPDATE_EDGE ;
	_update_edge_desc.id      = ENTITY_GET_ID      (edge) ;
	_update_edge_desc.r       = Edge_GetRelationID (edge) ;
	_update_edge_desc.s       = Edge_GetSrcNodeID  (edge) ;
	_update_edge_desc.d       = Edge_GetDestNodeID (edge) ;
	_update_edge_desc.attr_id = attr_id ;

	EffectsBuffer_WriteBytes (&_update_edge_desc, sizeof (_update_edge_desc),
			buff) ;

	//--------------------------------------------------------------------------
	// write attribute value
	//--------------------------------------------------------------------------

	EffectsBuffer_WriteSIValue(&value, buff);

	EffectsBuffer_IncEffectCount(buff);
}

// add an entity attribute removal effect to buffer
void EffectsBuffer_AddEntityRemoveAttributeEffect
(
	EffectsBuffer *buff,         // effect buffer
	GraphEntity *entity,         // updated entity ID
	AttributeID attr_id,         // updated attribute ID
	GraphEntityType entity_type  // entity type
) {
	// attribute was deleted
	int n = (attr_id == ATTRIBUTE_ID_ALL)
		? AttributeSet_Count(*entity->attributes)
		: 1;

	ResultSetStatistics *stats = QueryCtx_GetResultSetStatistics () ;
	stats->properties_removed += n ;

	SIValue v = SI_NullVal();
	if(entity_type == GETYPE_NODE) {
		EffectsBuffer_AddNodeUpdateEffect(buff, (Node*)entity, attr_id, v);
	} else {
		EffectsBuffer_AddEdgeUpdateEffect(buff, (Edge*)entity, attr_id, v);
	}
}

// add an entity add new attribute effect to buffer
void EffectsBuffer_AddEntityAddAttributeEffect
(
	EffectsBuffer *buff,         // effect buffer
	GraphEntity *entity,         // updated entity ID
	AttributeID attr_id,         // updated attribute ID
	SIValue value,               // value
	GraphEntityType entity_type  // entity type
) {
	// attribute was added
	ResultSetStatistics *stats = QueryCtx_GetResultSetStatistics () ;
	stats->properties_set++ ;

	if(entity_type == GETYPE_NODE) {
		EffectsBuffer_AddNodeUpdateEffect(buff, (Node*)entity, attr_id, value);
	} else {
		EffectsBuffer_AddEdgeUpdateEffect(buff, (Edge*)entity, attr_id, value);
	}
}

// add an entity update attribute effect to buffer
void EffectsBuffer_AddEntityUpdateAttributeEffect
(
	EffectsBuffer *buff,         // effect buffer
	GraphEntity *entity,         // updated entity ID
	AttributeID attr_id,         // updated attribute ID
	SIValue value,               // value
	GraphEntityType entity_type  // entity type
) {
	ResultSetStatistics *stats = QueryCtx_GetResultSetStatistics () ;
	stats->properties_set++ ;     // attribute was set
	stats->properties_removed++ ; // old attribute was deleted

	if(entity_type == GETYPE_NODE) {
		EffectsBuffer_AddNodeUpdateEffect(buff, (Node*)entity, attr_id, value);
	} else {
		EffectsBuffer_AddEdgeUpdateEffect(buff, (Edge*)entity, attr_id, value);
	}
}

// records a SET_LABELS effect into the buffer:
// writes the effect type followed by the serialized node vector
//
// effect format:
//   [EffectType]         effect type tag
//   [GxB serialized]     GxB_Vector_serialize blob of the node vector
void EffectsBuffer_AddLabelsEffect
(
	EffectsBuffer *buff,  // effect buffer to write into
	GrB_Vector nodes      // nodes that received the label
) {
	//--------------------------------------------------------------------------
	// update query statistics
	//--------------------------------------------------------------------------

	GrB_Index nvals ;
	GrB_OK (GrB_Vector_nvals (&nvals, nodes)) ;

	ResultSetStatistics *stats = QueryCtx_GetResultSetStatistics () ;
	stats->labels_added += nvals ;

	EffectType t = EFFECT_SET_LABELS;
	EffectsBuffer_WriteBytes (&t, sizeof (t), buff) ;

	//--------------------------------------------------------------------------
	// encode vector
	//--------------------------------------------------------------------------

	void *blob ;
	GrB_Index blob_size ;
	GrB_OK (GxB_Vector_serialize (&blob, &blob_size, nodes, NULL)) ;

	EffectsBuffer_WriteBytes (&blob_size, sizeof (blob_size), buff) ;
	EffectsBuffer_WriteBytes (blob, blob_size, buff) ;

	rm_free (blob) ;

	EffectsBuffer_IncEffectCount (buff) ;
}

// records a REMOVE_LABELS effect into the buffer:
// writes the effect type followed by the serialized node vector
//
// effect format:
//   [EffectType]         effect type tag
//   [GxB serialized]     GxB_Vector_serialize blob of the node vector
void EffectsBuffer_AddRemoveLabelsEffect
(
	EffectsBuffer *buff,  // effect buffer to write into
	GrB_Vector     nodes  // nodes that lost the label
) {
	//--------------------------------------------------------------------------
	// update query statistics
	//--------------------------------------------------------------------------

	GrB_Index nvals ;
	GrB_OK (GrB_Vector_nvals (&nvals, nodes)) ;
	ResultSetStatistics *stats = QueryCtx_GetResultSetStatistics () ;
	stats->labels_removed += nvals ;

	EffectType t = EFFECT_REMOVE_LABELS ;
	EffectsBuffer_WriteBytes (&t, sizeof (t), buff) ;

	// encode vector
	void *blob ;
	GrB_Index blob_size ;
	GrB_OK (GxB_Vector_serialize (&blob, &blob_size, nodes, NULL)) ;

	EffectsBuffer_WriteBytes (&blob_size, sizeof (blob_size), buff) ;
	EffectsBuffer_WriteBytes (blob, blob_size, buff) ;

	rm_free (blob) ;

	EffectsBuffer_IncEffectCount (buff) ;
}

// add a schema addition effect to buffer
void EffectsBuffer_AddNewSchemaEffect
(
	EffectsBuffer *buff,      // effect buffer
	const char *schema_name,  // id of the schema
	SchemaType st             // type of the schema
) {
	//--------------------------------------------------------------------------
	// effect format:
	//    effect type
	//    schema type
	//    schema name
	//--------------------------------------------------------------------------

	EffectType t = EFFECT_ADD_SCHEMA;
	EffectsBuffer_WriteBytes(&t, sizeof(t), buff);

	//--------------------------------------------------------------------------
	// write schema type
	//--------------------------------------------------------------------------

	EffectsBuffer_WriteBytes(&st, sizeof(st), buff);

	//--------------------------------------------------------------------------
	// write schema name
	//--------------------------------------------------------------------------

	EffectsBuffer_WriteString(schema_name, buff);

	EffectsBuffer_IncEffectCount(buff);
}

// add an attribute addition effect to buffer
void EffectsBuffer_AddNewAttributeEffect
(
	EffectsBuffer *buff,  // effect buffer
	const char *attr      // attribute name
) {
	//--------------------------------------------------------------------------
	// effect format:
	// effect type
	// attribute name
	//--------------------------------------------------------------------------

	EffectType t = EFFECT_ADD_ATTRIBUTE;
	EffectsBuffer_WriteBytes(&t, sizeof(t), buff);

	//--------------------------------------------------------------------------
	// write attribute name
	//--------------------------------------------------------------------------

	EffectsBuffer_WriteString(attr, buff);

	EffectsBuffer_IncEffectCount(buff);
}

void EffectsBuffer_Free
(
	EffectsBuffer *eb
) {
	if(eb == NULL) return;

	if(eb->owns_records) {
		EffectsBytes_Free(eb->records);
	}

	rm_free(eb);
}

// wrap a byte sink the caller owns as an effects-buffer
//
// This exists so v3 can use the SHARED SIValue codec against its own sinks
// without a second copy of it. A v3 record is one record per (opcode, shape),
// so a group's values accumulate in that group's sink rather than in a
// buffer's record stream - but they must be encoded by exactly the codec v2
// uses, because writing a second one is how the Rust side acquired a bug where
// a replica's string pool stayed empty.
//
// The returned buffer borrows the sink: freeing it frees the wrapper only. It
// carries no header and does not count effects, because it is not a payload -
// it is a handle for the writers that take one.
EffectsBuffer *EffectsBuffer_Wrap
(
	EffectsBytes *sink  // sink to write into; not owned
) {
	ASSERT(sink != NULL);

	EffectsBuffer *eb = rm_malloc(sizeof(EffectsBuffer));

	eb->n            = 0;
	eb->records      = sink;
	eb->version      = EFFECTS_VERSION_EMIT;
	eb->owns_records = false;

	return eb;
}

