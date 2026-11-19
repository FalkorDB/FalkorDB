/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "effects.h"
#include "effects_bytes.h"
#include "effects_internal.h"
#include "writers/effects_writer.h"
#include "effects_v3_group.h"
#include "../configuration/config.h"
#include "../util/identifier_limits.h"
#include "../query_ctx.h"
#include "../datatypes/map.h"
#include "../datatypes/vector.h"

// initial size of a buffer's block
#define EFFECTS_BUFFER_BLOCK_SIZE 62500

// a byte sink and who owns it
typedef struct {
	EffectsBytes *bytes;  // the sink
	bool owns;            // whether freeing the buffer frees it
} RecordSink;

// which body a v3 buffer is holding
typedef enum {
	BODY_RECORDS,   // bytes, after EffectsBuffer_TakeBody handed the sink over
	BODY_GROUPING,  // groups, serialized once the query stops producing
} BodyKind;

// ONE ARM PER PAYLOAD VERSION, because each version's state is its own
//
// v2 is a sink and nothing else. v3 adds a flags byte - its header is two
// bytes where v2's is one (_EffectsBuffer_WriteHeader) - and can hold either
// a sink or an accumulator, because a v3 record states its count and shape
// ahead of its rows and so cannot be written as effects arrive.
//
// Keyed on 'version', which is the one thing that always answers it. The inner
// choice is NOT derivable from the version and needs its own tag:
// EffectsBuffer_TakeBody stamps the version of the payload being reproduced -
// 3, on the round trip - and then hands back a sink, so a v3 buffer is
// perfectly able to be holding bytes.
//
// A fourth version adds an arm here rather than changing one.
struct _EffectsBuffer {
	uint64_t n;       // number of effects in buffer
	uint8_t  version; // payload version this buffer emits, and the arm below

	union {
		RecordSink v2;

		struct {
			// zero for everything this build emits - nothing here compresses -
			// and settable only so EffectsV3_Encode can reproduce the header
			// of a payload it was handed rather than assert one
			uint8_t flags;

			BodyKind body;
			union {
				RecordSink        records;   // BODY_RECORDS
				EffectsV3Grouping *grouping; // BODY_GROUPING
			};
		} v3;
	};

	// the write table for the live arm, chosen when the arm is opened so no
	// writer re-tests it; see writers/effects_writer.h
	const EffectsWriter *w;
};

// the sink this buffer writes bytes into, or NULL while it is accumulating
static const RecordSink *_sink_const
(
	const EffectsBuffer *eb  // effects-buffer
) {
	if(eb->version < 3)             return &eb->v2;
	if(eb->v3.body == BODY_RECORDS) return &eb->v3.records;

	return NULL;
}

static RecordSink *_sink
(
	EffectsBuffer *eb  // effects-buffer
) {
	return (RecordSink *)_sink_const(eb);
}

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
// written here rather than when the buffer is created, which is where v2 wrote
// it: v3's flags byte is not settled until the record stream is complete, and a
// v3 payload's records are grouped rather than emitted in arrival order, so
// nothing can precede them in the buffer
static unsigned char *_EffectsBuffer_WriteHeader
(
	const EffectsBuffer *eb,  // effects-buffer
	unsigned char *dst        // destination
) {
	*dst++ = eb->version;

	if(eb->version >= 3) {
		// flags; bit 0 = compressed. Nothing here compresses, so every buffer
		// builds carries 0 here - but a re-encode has to reproduce the header
		// of the payload it decoded, so the value is read rather than assumed
		*dst++ = eb->v3.flags;
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
	// worse than a refused payload, because nothing reports it.
	//
	// A log AND an assert, because they cover different builds: ASSERT compiles
	// to nothing without RG_DEBUG, so it stops a debug and CI build at the
	// offending writer while the log is what a release build leaves behind.
	if(unlikely(_sink(eb) == NULL)) {
		// AN EFFECT WITH NO v3 PATH IS A BUG, NOT A CONDITION TO SURVIVE.
		//
		// All 14 records have a producing path (EFFECTS_V3_ENCODE_READY), so
		// nothing reaches here. A fifteenth effect added without routing it
		// shows up here, and it has to be loud, because both quiet answers are
		// wrong: dropping the write loses the effect, and replaying the query
		// text instead is exact between two C engines and silently wrong
		// against Rust, whose db.idx.fulltext.createNodeIndex takes a single
		// map where C's is variadic.
		//
		// So: assert, stopping a debug and CI build at the writer that forgot
		// its v3 path, and log at warning in release. The fix is always to
		// route the effect, never to re-add a fallback.
		RedisModule_Log(NULL, "warning",
			"GRAPH.EFFECT an effect (%zu bytes) reached a v3 buffer with no v3 "
			"encoding path and WILL NOT BE REPLICATED; its writer needs to "
			"call into EffectsV3Grouping", n);
		ASSERT(false && "effect has no v3 encoding path");
		return;
	}

	EffectsBytes_Write (_sink(eb)->bytes, ptr, n) ;
}

// write a length-prefixed, NUL-terminated string
//
// THE SPEC SAYS BYTE LENGTH, NOT strlen: "the value's byte length + 1, then len
// bytes, the last a NUL", and a reader must use the length rather than call
// strlen, because a value may contain interior NULs. Rust produces them -
// openCypher's \uXXXX escape can encode one, so size('a<NUL>b') is 3 there where
// strlen would report 1.
//
// THIS STILL WRITES strlen + 1, for three reasons that EXPIRE SEPARATELY:
//
//   * no value here can express one. The \uXXXX escape is not implemented -
//     measured, size 6 for the escape Rust reports 1 for, and 8 where Rust
//     reports 3 - so no input with an interior NUL can reach here. Ends the
//     day the unicode escape is implemented.
//
//   * SIValue has no length for a string. `char *stringval` is the entire
//     representation (value.h), so the byte length does not exist to be
//     written. Ends if anyone adds one to SIValue.
//
//   * This writer is SHARED WITH v2, whose framing must not move. Conforming
//     would need a v3-only string writer - a second SIValue codec, the one
//     thing this file must not grow. Ends only when v2 does.
//
// So C conforms by construction rather than by intent, and a reader arriving
// later should check which of the three still holds. The day C gains \uXXXX
// support this becomes a silent truncation on the wire, and the fix is then a
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
void EffectsBuffer_WriteAttributeSet
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

// the write table the live arm requires
//
// The arm is enough TODAY, and only because there are two of each. The rule is
// two-level: the arm decides which tables are legal at all - a grouping table's
// arms dereference the grouping, so a records-arm buffer cannot be given one
// whatever version it is stamped with - and among the legal ones the version
// picks. A second grouping version separates them, and this is the function
// that has to grow the version argument; _open_body already holds it.
//
// A records-arm buffer never consults the table in the first place: the two
// that exist, EffectsBuffer_Wrap's and EffectsBuffer_TakeBody's, are written
// through WriteSIValue and EffectsV3_EncodeRecord, never through an Add*.
static const EffectsWriter *_writer_for
(
	const EffectsBuffer *eb  // effects-buffer
) {
	if(eb->version >= 3 && eb->v3.body == BODY_GROUPING) {
		return &EFFECTS_WRITER_V3;
	}

	return &EFFECTS_WRITER_V2;
}

// release whichever arm is live
static void _release_body
(
	EffectsBuffer *eb  // effects-buffer
) {
	if(eb->version >= 3 && eb->v3.body == BODY_GROUPING) {
		EffectsV3Grouping_Free(eb->v3.grouping);
		return;
	}

	RecordSink *s = _sink(eb);
	if(s->owns) {
		EffectsBytes_Free(s->bytes);
	}
}

// open the sink arm, borrowing 'sink' when one is supplied
static void _open_sink
(
	EffectsBuffer *eb,   // effects-buffer
	EffectsBytes *sink   // sink to borrow, or NULL to allocate one
) {
	RecordSink rs = {
		.bytes = (sink != NULL) ? sink : EffectsBytes_New(EFFECTS_BUFFER_BLOCK_SIZE),
		.owns  = (sink == NULL)
	};

	if(eb->version >= 3) {
		eb->v3.body    = BODY_RECORDS;
		eb->v3.records = rs;
	} else {
		eb->v2 = rs;
	}
}

// open the arm this emit version calls for
static void _open_body
(
	EffectsBuffer *eb,  // effects-buffer
	uint64_t emit       // version to emit
) {
	eb->version = (uint8_t)emit;

	if(emit >= 3) {
		eb->v3.flags    = 0;
		eb->v3.body     = BODY_GROUPING;
		eb->v3.grouping = EffectsV3Grouping_New();
	} else {
		_open_sink(eb, NULL);
	}

	eb->w = _writer_for(eb);
}

// create a new effects-buffer
EffectsBuffer *EffectsBuffer_New
(
	void
) {
	EffectsBuffer *eb = rm_malloc(sizeof(EffectsBuffer));

	uint64_t emit = EFFECTS_VERSION_EMIT;
	Config_Option_get(Config_EFFECTS_VERSION, &emit);

	eb->n = 0;

	_open_body(eb, emit);

	// note: no header is written here. v2 stamped its version byte at
	// construction; it is now written by EffectsBuffer_Buffer, so that the
	// records a buffer holds are only records
	return eb;
}

EffectsV3Grouping *EffectsBuffer_V3
(
	const EffectsBuffer *eb  // effects-buffer
) {
	ASSERT(eb != NULL);

	return (eb->version >= 3 && eb->v3.body == BODY_GROUPING)
		? eb->v3.grouping
		: NULL;
}

const EffectsWriter *EffectsBuffer_Writer
(
	const EffectsBuffer *eb  // effects-buffer
) {
	ASSERT(eb != NULL);

	return eb->w;
}

// reset effects-buffer
void EffectsBuffer_Reset
(
	EffectsBuffer *buff  // effects-buffer
) {
	ASSERT(buff != NULL);

	uint64_t emit = EFFECTS_VERSION_EMIT;
	Config_Option_get(Config_EFFECTS_VERSION, &emit);

	// the arm can change between queries, because the version is re-read here
	// and an operator may have moved it
	RecordSink *s = _sink(buff);

	if(emit < 3 && buff->version < 3 && s != NULL) {
		// same arm as last time: retain the first block rather than churn it
		EffectsBytes_Clear(s->bytes);
	} else {
		_release_body(buff);
		_open_body(buff, emit);
	}

	buff->n = 0;
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

// take over a buffer's body so pre-built records can be written into it
//
// EffectsBuffer_Buffer has TWO possible bodies: the accumulator's output when
// one is attached, and eb->records otherwise. EffectsV3_Encode writes records it
// was handed, so the accumulator has to be out of the way first - otherwise
// everything written here is serialised over and silently discarded.
//
// Refuses a buffer that has already staged effects rather than throwing them
// away. 'version' and 'flags' come from the payload being reproduced: re-encoding
// what was decoded must reproduce its header too.
EffectsBytes *EffectsBuffer_TakeBody
(
	EffectsBuffer *eb,  // effects-buffer
	uint8_t version,    // version byte to emit
	uint8_t flags       // flags byte to emit
) {
	ASSERT(eb != NULL);

	if(eb->version >= 3 && eb->v3.body == BODY_GROUPING) {
		if(EffectsV3Grouping_RecordCount(eb->v3.grouping) > 0) {
			return NULL;
		}

		// the transition: the accumulator goes and a sink opens in its place,
		// inside the same v3 arm
		EffectsV3Grouping_Free(eb->v3.grouping);
		_open_sink(eb, NULL);
		eb->w = _writer_for(eb);
	}

	// MIGRATE THE SINK, do not just move the discriminant. 'version' names an
	// arm, so a buffer built at v2 and taken over at v3 has to carry its sink
	// across - assigning eb->version alone leaves the sink in the other arm's
	// storage, and on a union that is memory corruption rather than a wrong
	// answer: eb->v3.flags would land on byte 0 of eb->v2.bytes.
	RecordSink rs = *_sink(eb);

	eb->version = version;

	if(version >= 3) {
		eb->v3.flags   = flags;
		eb->v3.body    = BODY_RECORDS;
		eb->v3.records = rs;
	} else {
		eb->v2 = rs;
	}

	eb->w = _writer_for(eb);

	return _sink(eb)->bytes;
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
	EffectsBytes *body    = NULL;
	EffectsBytes *v3_body = NULL;

	if(eb->version >= 3 && eb->v3.body == BODY_GROUPING) {
		v3_body = EffectsBytes_New(EFFECTS_BUFFER_BLOCK_SIZE);
		EffectsV3Grouping_Encode(eb->v3.grouping, v3_body);
		body = v3_body;
	} else {
		body = _sink_const(eb)->bytes;
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
	// update query stats
	//--------------------------------------------------------------------------

	ResultSetStatistics *stats = QueryCtx_GetResultSetStatistics () ;
	stats->nodes_created++ ;
	stats->labels_added   += label_count ;
	stats->properties_set += AttributeSet_Count (*n->attributes) ;

	EffectsBuffer_Writer (buff)->CreateNode(buff, n, labels, label_count) ;
}

// add a edge creation effect to buffer
void EffectsBuffer_AddCreateEdgeEffect
(
	EffectsBuffer *buff,  // effect buffer
	const Edge *edge      // edge created
) {
	
	ResultSetStatistics *stats = QueryCtx_GetResultSetStatistics () ;
	stats->relationships_created++ ;
	stats->properties_set += AttributeSet_Count (*edge->attributes) ;

	EffectsBuffer_Writer (buff)->CreateEdge(buff, edge) ;
}

// add a node deletion effect to buffer
void EffectsBuffer_AddDeleteNodeEffect
(
	EffectsBuffer *buff,  // effect buffer
	const Node *node      // node deleted
) {

	// update query statistics
	ResultSetStatistics *stats = QueryCtx_GetResultSetStatistics () ;
	stats->nodes_deleted++ ;

	EffectsBuffer_Writer (buff)->DeleteNode(buff, node) ;
}

// add a edge deletion effect to buffer
void EffectsBuffer_AddDeleteEdgeEffect
(
	EffectsBuffer *eb,  // effect buffer
	const Edge *edge    // edge deleted
) {

	ResultSetStatistics *stats = QueryCtx_GetResultSetStatistics () ;
	stats->relationships_deleted++ ;

	EffectsBuffer_Writer (eb)->DeleteEdge(eb, edge) ;
}


// add an entity update effect to buffer



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

	EffectsBuffer_Writer (buff)->UpdateEntity(buff, entity, attr_id, v, entity_type) ;
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

	EffectsBuffer_Writer (buff)->UpdateEntity(buff, entity, attr_id, value, entity_type) ;
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

	EffectsBuffer_Writer (buff)->UpdateEntity(buff, entity, attr_id, value, entity_type) ;
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

	GrB_Index nvals ;
	GrB_OK (GrB_Vector_nvals (&nvals, nodes)) ;

	ResultSetStatistics *stats = QueryCtx_GetResultSetStatistics () ;
	stats->labels_added += nvals ;

	EffectsBuffer_Writer (buff)->Labels(buff, nodes, EFFECT_SET_LABELS) ;
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

	GrB_Index nvals ;
	GrB_OK (GrB_Vector_nvals (&nvals, nodes)) ;
	ResultSetStatistics *stats = QueryCtx_GetResultSetStatistics () ;
	stats->labels_removed += nvals ;

	EffectsBuffer_Writer (buff)->Labels(buff, nodes, EFFECT_REMOVE_LABELS) ;
}

// add a schema addition effect to buffer
void EffectsBuffer_AddNewSchemaEffect
(
	EffectsBuffer *buff,      // effect buffer
	const char *schema_name,  // id of the schema
	SchemaType st             // type of the schema
) {
	EffectsBuffer_Writer (buff)->NewSchema(buff, schema_name, st) ;
}

// add an attribute addition effect to buffer
void EffectsBuffer_AddNewAttributeEffect
(
	EffectsBuffer *buff,  // effect buffer
	const char *attr      // attribute name
) {
	EffectsBuffer_Writer (buff)->NewAttribute(buff, attr) ;
}

void EffectsBuffer_Free
(
	EffectsBuffer *eb
) {
	if(eb == NULL) return;

	_release_body(eb);

	rm_free(eb);
}

// wrap a byte sink the caller owns as an effects-buffer
//
// This exists so v3 can use the SHARED SIValue codec against its own sinks
// without a second copy of it: a group's values accumulate in that group's sink
// rather than in a buffer's record stream, but must be encoded by exactly the
// codec v2 uses - writing a second one is how the Rust side acquired a bug where
// a replica's string pool stayed empty.
//
// The returned buffer borrows the sink: freeing it frees the wrapper only. It
// carries no header and does not count effects, because it is not a payload.
EffectsBuffer *EffectsBuffer_Wrap
(
	EffectsBytes *sink  // sink to write into; not owned
) {
	ASSERT(sink != NULL);

	EffectsBuffer *eb = rm_malloc(sizeof(EffectsBuffer));

	eb->n       = 0;
	eb->version = EFFECTS_VERSION_EMIT;

	if(eb->version >= 3) {
		eb->v3.flags = 0;
	}

	_open_sink(eb, sink);
	eb->w = _writer_for(eb);

	return eb;
}

