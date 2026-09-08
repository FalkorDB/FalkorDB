/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "effects.h"
#include "effects_bytes.h"
#include "effects_internal.h"
#include "effects_v3_group.h"
#include "../configuration/config.h"
#include "../util/identifier_limits.h"
#include "../query_ctx.h"
#include "../datatypes/map.h"
#include "../datatypes/vector.h"

// initial size of a buffer's block
#define EFFECTS_BUFFER_BLOCK_SIZE 62500

struct _EffectsBuffer {
	EffectsBytes *records;  // encoded records; v2 only
	uint64_t n;             // number of effects in buffer
	uint8_t version;        // payload version this buffer emits
	bool owns_records;      // whether freeing the buffer frees the sink

	// v3 accumulates into groups instead of writing records on arrival,
	// because a record states its count and shape ahead of its rows. NULL
	// when this buffer emits v2
	EffectsV3Grouping *v3;

	// set when an effect arrived that the v3 encoder cannot represent - today
	// the index and constraint DDL of records 11-14, which still write v2
	// bytes. Such a buffer must not be sent: it would describe a subset of the
	// query's effects and the replica would silently miss the rest
	bool v3_incomplete;
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

	// A v2 record written while v3 is active means some Add*Effect was never
	// routed into the accumulator. EffectsBuffer_Buffer serializes the groups
	// and ignores this stream in v3, so the record would be SILENTLY DROPPED -
	// a replica that never learns of the mutation, which is worse than a
	// refused payload because nothing reports it.
	//
	// Logged rather than asserted: ASSERT compiles to nothing without RG_DEBUG,
	// and this is exactly the case that must not pass quietly in a release
	// build.
	if(unlikely(eb->v3 != NULL)) {
		// An effect the v3 encoder cannot represent. Records 11-14 - the index
		// and constraint DDL - still write v2 bytes into a stream that
		// EffectsBuffer_Buffer ignores in v3 mode, so this record would simply
		// vanish: master indexed, replica not, no error and no resync.
		//
		// Marking the buffer incomplete is what stops that. The caller then
		// replicates the query verbatim instead, which is how the DDL reaches
		// the replica until the encoder implements those records. Dropping the
		// write here is harmless once the buffer will not be sent.
		//
		// WHAT VERBATIM ASSUMES: that the replica can execute the same query
		// text. Between two C engines that is exact, and a replica whose replay
		// fails escalates through DivergenceGuard_OnFailure (cmd_query.c), so a
		// failure is loud. Across engines it is only as good as their query
		// surfaces matching - measured: a Rust replica rejects
		// db.idx.fulltext.createNodeIndex on arity, logs, and continues without
		// escalating, leaving it quietly without the index.
		//
		// So this fallback is correct for C-to-C and is an interim for
		// C-to-Rust. Records 11-14 in the encoder are what remove the
		// assumption; until then a mixed-engine deployment should not run DDL
		// on a v3-emitting C master.
		if(!eb->v3_incomplete) {
			RedisModule_Log(NULL, "notice",
				"GRAPH.EFFECT v3 cannot encode this effect; replicating the "
				"query verbatim instead");
			((EffectsBuffer *)eb)->v3_incomplete = true;
		}
		return;
	}

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

	uint64_t emit = EFFECTS_VERSION_EMIT;
	Config_Option_get(Config_EFFECTS_VERSION, &emit);

	eb->n            = 0;
	eb->records      = EffectsBytes_New(EFFECTS_BUFFER_BLOCK_SIZE);
	eb->version      = (uint8_t)emit;
	eb->owns_records = true;
	eb->v3           = (emit >= 3) ? EffectsV3Grouping_New() : NULL;
	eb->v3_incomplete = false;

	// note: no header is written here. v2 stamped its version byte at
	// construction; it is now written by EffectsBuffer_Buffer, so that the
	// records a buffer holds are only records
	return eb;
}

EffectsBuffer *EffectsBuffer_NewV2
(
	void
) {
	EffectsBuffer *eb = rm_malloc(sizeof(EffectsBuffer));

	eb->n             = 0;
	eb->records       = EffectsBytes_New(EFFECTS_BUFFER_BLOCK_SIZE);
	eb->version       = 2;
	eb->owns_records  = true;
	eb->v3            = NULL;
	eb->v3_incomplete = false;

	return eb;
}

bool EffectsBuffer_ForceV2
(
	EffectsBuffer *eb  // effects-buffer
) {
	ASSERT(eb != NULL);

	if(eb->v3 == NULL) {
		return true;  // already emitting v2
	}

	// effects already staged as v3 groups would be lost by switching
	if(eb->n != 0) {
		return false;
	}

	EffectsV3Grouping_Free(eb->v3);

	eb->v3            = NULL;
	eb->v3_incomplete = false;
	eb->version       = 2;

	return true;
}

// reset effects-buffer
void EffectsBuffer_Reset
(
	EffectsBuffer *buff  // effects-buffer
) {
	ASSERT(buff != NULL);

	EffectsBytes_Clear(buff->records);

	EffectsV3Grouping_Free(buff->v3);

	uint64_t emit = EFFECTS_VERSION_EMIT;
	Config_Option_get(Config_EFFECTS_VERSION, &emit);

	buff->n       = 0;
	buff->version = (uint8_t)emit;
	buff->v3      = (emit >= 3) ? EffectsV3Grouping_New() : NULL;
	buff->v3_incomplete = false;
}

// whether every effect in this buffer can be encoded
//
// False once an effect arrives that the v3 encoder cannot represent. A buffer
// that is not complete MUST NOT be sent: it would describe some of the query's
// effects and silently omit the rest, which leaves a replica differing from its
// master with nothing reporting it. The caller replicates the query verbatim
// instead - the same fallback v2 uses for statements it cannot express.
bool EffectsBuffer_Complete
(
	const EffectsBuffer *buff  // effects-buffer
) {
	ASSERT(buff != NULL);

	return !buff->v3_incomplete;
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

	// v3 serializes its grouped records here, which is the only point at which
	// they can be: a record states its count and shape ahead of its rows, so
	// nothing could be written while effects were still arriving
	EffectsBytes *body = eb->records;
	EffectsBytes *v3_body = NULL;

	if(eb->v3 != NULL) {
		v3_body = EffectsBytes_New(EFFECTS_BUFFER_BLOCK_SIZE);
		EffectsV3Grouping_Encode(eb->v3, v3_body);
		body = v3_body;
	}

	size_t hdr = _EffectsBuffer_HeaderLen(eb->version);
	size_t l   = hdr + EffectsBytes_Len(body);

	//--------------------------------------------------------------------------
	// allocate buffer and populate
	//--------------------------------------------------------------------------

	unsigned char *buffer = rm_malloc(sizeof(unsigned char) * l);
	unsigned char *offset = _EffectsBuffer_WriteHeader(eb, buffer);

	EffectsBytes_CopyInto(body, offset);

	EffectsBytes_Free(v3_body);

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

	if(buff->v3 != NULL) {
		AttributeSet attrs = *n->attributes;
		uint16_t n_attrs = AttributeSet_Count(attrs);

		AttributeID ids[256];
		SIValue vals[256];
		uint16_t k = (n_attrs <= 256) ? n_attrs : 256;
		for(uint16_t i = 0; i < k; i++) {
			AttributeSet_GetIdx(attrs, i, ids + i, vals + i);
		}

		EffectsV3Grouping_AddNode(buff->v3, EFFECT_CREATE_NODE, labels,
				label_count, ENTITY_GET_ID(n), ids, vals, k);
		EffectsBuffer_IncEffectCount(buff);
		return;
	}

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

	if(buff->v3 != NULL) {
		AttributeSet attrs = *edge->attributes;
		uint16_t n_attrs = AttributeSet_Count(attrs);

		AttributeID ids[256];
		SIValue vals[256];
		uint16_t k = (n_attrs <= 256) ? n_attrs : 256;
		for(uint16_t i = 0; i < k; i++) {
			AttributeSet_GetIdx(attrs, i, ids + i, vals + i);
		}

		EffectsV3Grouping_AddEdge(buff->v3, EFFECT_CREATE_EDGE,
				Edge_GetRelationID(edge), ENTITY_GET_ID(edge),
				Edge_GetSrcNodeID(edge), Edge_GetDestNodeID(edge),
				ids, vals, k);
		EffectsBuffer_IncEffectCount(buff);
		return;
	}

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

	if(buff->v3 != NULL) {
		// the labels the node ACTUALLY held, read while it is still alive -
		// GraphHub_DeleteNodes records the effect before Graph_DeleteNodes, so
		// the label matrices are still intact here. A replica needs them to
		// clear the right label-scoped index documents
		Graph *g = QueryCtx_GetGraph();
		uint lbl_count;
		NODE_GET_LABELS(g, node, lbl_count);

		EffectsV3Grouping_AddNode(buff->v3, EFFECT_DELETE_NODE, labels,
				(uint16_t)lbl_count, ENTITY_GET_ID(node), NULL, NULL, 0);
		EffectsBuffer_IncEffectCount(buff);
		return;
	}

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

	if(eb->v3 != NULL) {
		// the type is captured HERE, while the edge still carries it. v3
		// groups deleted edges by relationship type and the edge is gone by
		// the time the payload is built
		EffectsV3Grouping_AddEdge(eb->v3, EFFECT_DELETE_EDGE,
				Edge_GetRelationID(edge), ENTITY_GET_ID(edge),
				Edge_GetSrcNodeID(edge), Edge_GetDestNodeID(edge),
				NULL, NULL, 0);
		EffectsBuffer_IncEffectCount(eb);
		return;
	}

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


// stage one attribute of an entity's update into the v3 accumulator
//
// All three per-attribute writers converge here: add, update and remove are one
// record family in v2 and one shape in v3. A v3 record's shape is the entity's
// WHOLE updated attribute set, so this stages rather than emits - the group is
// not selectable until the query stops producing attributes for this entity.
//
// REMOVE-ALL IS STATED EXPLICITLY, not as a sentinel. v2 writes
// ATTRIBUTE_ID_ALL as the attribute id and lets apply special-case it; v3
// cannot, because the attribute ids ARE the record's shape, so a sentinel there
// would name something that is not an attribute. `SET n = {} SET n.x = 1` would
// then produce a shape mixing the two with nothing to say which applied first,
// and a dedicated record has the same problem between records. Stating every
// removed attribute explicitly needs no ordering, because it resolves to the
// end state rather than replaying operations.
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
			EffectsV3Grouping_StageUpdate(buff->v3, opcode, id, lbls,
					n_labels, rel, a_id, SI_NullVal());
		}
		return;
	}

	EffectsV3Grouping_StageUpdate(buff->v3, opcode, id, lbls, n_labels, rel,
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

	if(buff->v3 != NULL) {
		STAGE_V3_UPDATE(buff, entity, attr_id, v, entity_type);
		EffectsBuffer_IncEffectCount(buff);
		return;
	}

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

	if(buff->v3 != NULL) {
		STAGE_V3_UPDATE(buff, entity, attr_id, value, entity_type);
		EffectsBuffer_IncEffectCount(buff);
		return;
	}

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

	if(buff->v3 != NULL) {
		STAGE_V3_UPDATE(buff, entity, attr_id, value, entity_type);
		EffectsBuffer_IncEffectCount(buff);
		return;
	}

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

// file a label vector into the v3 accumulator
//
// v3 states the LABEL SET and the node ids rather than a serialized GraphBLAS
// vector - which is the point of the record changing: v2's blob couples the
// wire to whatever GraphBLAS each engine was built against.
//
// The vector is named with its label, and it has already had redundancies
// stripped upstream (staged_updates.c), so every node in it genuinely gains or
// loses the label.
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
		EffectsV3Grouping_AddNode(buff->v3, opcode, &lbl, 1, node_id,
				NULL, NULL, 0);
		info = GxB_Vector_Iterator_next(it);
	}

	GrB_free(&it);
}

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

	if(buff->v3 != NULL) {
		_StageV3Labels(buff, nodes, EFFECT_SET_LABELS);
		EffectsBuffer_IncEffectCount(buff);
		return;
	}

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

	if(buff->v3 != NULL) {
		_StageV3Labels(buff, nodes, EFFECT_REMOVE_LABELS);
		EffectsBuffer_IncEffectCount(buff);
		return;
	}

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

	if(buff->v3 != NULL) {
		// v3 carries the id as well as the name, so the replica can assert the
		// id it would assign matches - a numbering disagreement is otherwise
		// introduced by a record that cannot report it
		GraphContext *gc = QueryCtx_GetGraphCtx();
		const Schema *sch = GraphContext_GetSchema(gc, schema_name, st);
		ASSERT(sch != NULL);

		EffectsV3Grouping_AddSchema(buff->v3, st, Schema_GetID(sch),
				schema_name);
		EffectsBuffer_IncEffectCount(buff);
		return;
	}

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

	if(buff->v3 != NULL) {
		GraphContext *gc = QueryCtx_GetGraphCtx();
		AttributeID id = GraphContext_GetAttributeID(gc, attr);
		ASSERT(id != ATTRIBUTE_ID_NONE);

		EffectsV3Grouping_AddAttribute(buff->v3, id, attr);
		EffectsBuffer_IncEffectCount(buff);
		return;
	}

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

	EffectsV3Grouping_Free(eb->v3);

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
	eb->v3           = NULL;
	eb->v3_incomplete = false;

	return eb;
}

