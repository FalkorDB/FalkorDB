/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "effects_v3.h"
#include "effects_wire.h"
#include "../datatypes/map.h"
#include "../util/rmalloc.h"
#include "../util/roaring_include.h"

#include <string.h>

//------------------------------------------------------------------------------
// v3 decode: bytes -> records
//------------------------------------------------------------------------------
//
// Reads a v3 payload into EffectsV3Records without a GraphContext. Every count
// and length here comes off the wire, so each one is validated against the
// bytes actually remaining before it reaches an allocator - fstream_remaining
// exists for exactly that. No ceiling is imposed on any count: the payload's
// own length is the bound. An invented limit is a divergence that fires on
// legitimate data and cannot be repaired by a resync, because it fails
// identically every time.
//
// Ids are never expanded here. One valid segment describes four billion ids in
// seven bytes, and GRAPH.EFFECT is applied inline on the main thread with no
// timeout, so a decoder whose cost is unbounded by its input is a denial of
// service. Segments are returned as written; EffectsV3_Apply expands them.
//
//------------------------------------------------------------------------------

// the smallest a segment can be: a header byte plus a 1-byte value and a 1-byte
// count (Range/Repeat at width code 0). Ascending is larger - header plus a u32
// blob length - so this is the floor for any kind, and it is what lets a
// segment count be rejected before it sizes an allocation
#define SEGMENT_MIN_BYTES 3

// the smallest an SIValue can be: the u32 SIType on its own, which is exactly
// what T_NULL is
#define SIVALUE_MIN_BYTES 4

// flag bits this build understands (docs/effects-v3.md:104)
// bit 0 = compressed; everything else is reserved and rejected
#define FLAG_COMPRESSED 0x01
#define FLAGS_KNOWN     0x01

//------------------------------------------------------------------------------
// fixed-width reads
//------------------------------------------------------------------------------

static inline bool _ReadU8
(
	FILE *stream,
	uint8_t *v
) {
	return fread_checked (v, sizeof (*v), stream) ;
}

static inline bool _ReadU16
(
	FILE *stream,
	uint16_t *v
) {
	return fread_checked (v, sizeof (*v), stream) ;
}

static inline bool _ReadU32
(
	FILE *stream,
	uint32_t *v
) {
	return fread_checked (v, sizeof (*v), stream) ;
}

static inline bool _ReadI32
(
	FILE *stream,
	int32_t *v
) {
	return fread_checked (v, sizeof (*v), stream) ;
}

// read a little-endian unsigned integer 1, 2, 4 or 8 bytes wide, as selected by
// a 2-bit width code off a segment header
//
// assembled byte by byte rather than memcpy'd into a uint64_t so the result
// does not depend on the host's byte order - the widths are a wire encoding,
// and a narrow one is not a prefix of the value on a big-endian machine
static bool _ReadWidth
(
	FILE *stream,
	uint8_t width_code,
	uint64_t *v
) {
	unsigned char buf[8] ;
	const size_t n = EFFECTS_V3_WIDTH_BYTES (width_code) ;

	if (!fread_checked (buf, n, stream)) {
		return false ;
	}

	uint64_t out = 0 ;
	for (size_t i = 0 ; i < n ; i++) {
		out |= ((uint64_t)buf[i]) << (i * 8) ;
	}

	*v = out ;
	return true ;
}

//------------------------------------------------------------------------------
// the five blocks
//------------------------------------------------------------------------------

// read one IdList segment
//
// the observed width codes are retained on the segment rather than discarded:
// re-encoding reproduces the peer's bytes instead of our own narrowest-fit
// arithmetic, which is what makes a round trip a test of the wire rather than
// of ourselves
static EffectsV3Status _ReadSegment
(
	FILE *stream,
	EffectsV3Segment *seg
) {
	uint8_t header ;
	if (!_ReadU8 (stream, &header)) {
		return EFFECTS_V3_TRUNCATED ;
	}

	// bit 7 is reserved and MUST be zero
	//
	// rejected rather than masked off: a payload that sets it was written by
	// something that knows a format this build does not, so decoding the rest
	// would apply a prefix of a record whose shape we are guessing at
	if (header & EFFECTS_V3_SEG_RESERVED) {
		return EFFECTS_V3_MALFORMED ;
	}

	const uint8_t kind = header & EFFECTS_V3_SEG_KIND_MASK ;

	seg->value_width = (header >> EFFECTS_V3_SEG_VWIDTH_SHIFT) & 0x03 ;
	seg->count_width = (header >> EFFECTS_V3_SEG_CWIDTH_SHIFT) & 0x03 ;
	seg->descending  = (header & EFFECTS_V3_SEG_DESCENDING) != 0 ;

	switch (kind) {
		case EFFECTS_V3_SEG_RANGE:
			seg->kind = EFFECTS_V3_SEG_RANGE ;
			if (!_ReadWidth (stream, seg->value_width, &seg->range.base) ||
				!_ReadWidth (stream, seg->count_width, &seg->range.len)) {
				return EFFECTS_V3_TRUNCATED ;
			}

			// the range has to fit in the id space it is describing. Ascending
			// must not run past UINT64_MAX; descending counts DOWN from base,
			// which is its highest id, so it must not run below zero. Both are
			// derived from the segment's own two numbers, not from a ceiling.
			if (seg->range.len > 0) {
				const uint64_t span = seg->range.len - 1 ;
				if (seg->descending) {
					if (span > seg->range.base) {
						return EFFECTS_V3_MALFORMED ;
					}
				} else {
					if (span > UINT64_MAX - seg->range.base) {
						return EFFECTS_V3_MALFORMED ;
					}
				}
			}
			return EFFECTS_V3_OK ;

		case EFFECTS_V3_SEG_REPEAT:
			seg->kind = EFFECTS_V3_SEG_REPEAT ;

			// a Repeat holds one id 'count' times, so it reads the same in
			// either direction and the bit cannot mean anything. A peer that
			// set it meant something this build does not know, so refuse the
			// buffer rather than ignore the bit.
			if (seg->descending) {
				return EFFECTS_V3_MALFORMED ;
			}

			if (!_ReadWidth (stream, seg->value_width, &seg->repeat.id) ||
				!_ReadWidth (stream, seg->count_width, &seg->repeat.count)) {
				return EFFECTS_V3_TRUNCATED ;
			}
			return EFFECTS_V3_OK ;

		case EFFECTS_V3_SEG_ASCENDING: {
			seg->kind = EFFECTS_V3_SEG_ASCENDING ;

			uint32_t blob_len ;
			if (!_ReadU32 (stream, &blob_len)) {
				return EFFECTS_V3_TRUNCATED ;
			}

			// validate the length against what is actually left BEFORE it
			// reaches rm_malloc - a corrupt u32 is otherwise an allocation
			// request
			const long remaining = fstream_remaining (stream) ;
			if (remaining < 0 || blob_len > (uint64_t)remaining) {
				return EFFECTS_V3_MALFORMED ;
			}

			if (blob_len == 0) {
				// an Ascending segment describes at least one id, so it cannot
				// carry an empty bitmap
				return EFFECTS_V3_MALFORMED ;
			}

			unsigned char *blob = rm_malloc (blob_len) ;
			if (!fread_checked (blob, blob_len, stream)) {
				rm_free (blob) ;
				return EFFECTS_V3_TRUNCATED ;
			}

			// the cardinality is needed to check this segment against what the
			// record still owes, and roaring gives it without expanding: it is
			// a function of the container headers, not of the ids
			roaring64_bitmap_t *bitmap =
				roaring64_bitmap_portable_deserialize_safe ((const char*)blob,
						blob_len) ;
			if (bitmap == NULL) {
				rm_free (blob) ;
				return EFFECTS_V3_MALFORMED ;
			}

			seg->ascending.blob        = blob ;
			seg->ascending.n           = blob_len ;
			seg->ascending.cardinality =
				roaring64_bitmap_get_cardinality (bitmap) ;

			roaring64_bitmap_free (bitmap) ;

			if (seg->ascending.cardinality == 0) {
				rm_free (blob) ;
				seg->ascending.blob = NULL ;
				return EFFECTS_V3_MALFORMED ;
			}

			return EFFECTS_V3_OK ;
		}

		default:
			// kind 3 is not assigned
			return EFFECTS_V3_MALFORMED ;
	}
}

// how many ids a segment describes, without expanding it
static uint64_t _SegmentCardinality
(
	const EffectsV3Segment *seg
) {
	switch (seg->kind) {
		case EFFECTS_V3_SEG_RANGE:     return seg->range.len ;
		case EFFECTS_V3_SEG_REPEAT:    return seg->repeat.count ;
		case EFFECTS_V3_SEG_ASCENDING: return seg->ascending.cardinality ;
		default:                       return 0 ;
	}
}

static void _IdListFree
(
	EffectsV3IdList *list
) {
	if (list->segments == NULL) {
		return ;
	}

	for (uint32_t i = 0 ; i < list->n ; i++) {
		if (list->segments[i].kind == EFFECTS_V3_SEG_ASCENDING) {
			rm_free (list->segments[i].ascending.blob) ;
		}
	}

	rm_free (list->segments) ;
	list->segments = NULL ;
	list->n        = 0 ;
}

// read an IdList: u32 segment count, then the segments
//
// 'owed' is how many ids the record says this list holds. Both the segment
// count and every segment's length are on the wire even though either could be
// inferred from it, and that redundancy is the point: it makes a truncated list
// distinguishable from a complete one, and stops a wrong record count being
// absorbed by the final segment. So it is checked, in both directions.
static EffectsV3Status _ReadIdList
(
	FILE *stream,
	uint64_t owed,
	EffectsV3IdList *list
) {
	list->segments = NULL ;
	list->n        = 0 ;

	uint32_t n ;
	if (!_ReadU32 (stream, &n)) {
		return EFFECTS_V3_TRUNCATED ;
	}

	if (n == 0) {
		// no segments describes no ids; only consistent with an empty record
		return (owed == 0) ? EFFECTS_V3_OK : EFFECTS_V3_MALFORMED ;
	}

	// reject a segment count that outruns the payload before allocating for it
	const long remaining = fstream_remaining (stream) ;
	if (remaining < 0 ||
		(uint64_t)n * SEGMENT_MIN_BYTES > (uint64_t)remaining) {
		return EFFECTS_V3_MALFORMED ;
	}

	list->segments = rm_calloc (n, sizeof (EffectsV3Segment)) ;
	list->n        = n ;

	uint64_t total = 0 ;
	for (uint32_t i = 0 ; i < n ; i++) {
		EffectsV3Status status = _ReadSegment (stream, list->segments + i) ;
		if (status != EFFECTS_V3_OK) {
			_IdListFree (list) ;
			return status ;
		}

		const uint64_t card = _SegmentCardinality (list->segments + i) ;

		// a segment that overruns what the record owes binds rows to the wrong
		// entities rather than failing, so it fails here. This is also the
		// Ascending cardinality check the format calls for: a bitmap one id
		// short would land every later row on the wrong entity.
		if (card > owed - total) {
			_IdListFree (list) ;
			return EFFECTS_V3_MALFORMED ;
		}

		total += card ;
	}

	// the segment list as a whole must total the record's count
	if (total != owed) {
		_IdListFree (list) ;
		return EFFECTS_V3_MALFORMED ;
	}

	return EFFECTS_V3_OK ;
}

// read a LabelSet: u16 n, then n LabelIDs
//
// 'n' is the label count, NOT the record's entity count - the labels are the
// shape, hoisted once
static EffectsV3Status _ReadLabelSet
(
	FILE *stream,
	LabelID **labels,
	uint16_t *n_labels
) {
	*labels   = NULL ;
	*n_labels = 0 ;

	uint16_t n ;
	if (!_ReadU16 (stream, &n)) {
		return EFFECTS_V3_TRUNCATED ;
	}

	if (n == 0) {
		return EFFECTS_V3_OK ;
	}

	const long remaining = fstream_remaining (stream) ;
	if (remaining < 0 ||
		(uint64_t)n * sizeof (int32_t) > (uint64_t)remaining) {
		return EFFECTS_V3_MALFORMED ;
	}

	LabelID *l = rm_malloc (n * sizeof (LabelID)) ;
	for (uint16_t i = 0 ; i < n ; i++) {
		int32_t id ;
		if (!_ReadI32 (stream, &id)) {
			rm_free (l) ;
			return EFFECTS_V3_TRUNCATED ;
		}
		l[i] = (LabelID)id ;
	}

	*labels   = l ;
	*n_labels = n ;
	return EFFECTS_V3_OK ;
}

// read the AttrSet header: u16 n, then n AttributeIDs
//
// the ids only - the values follow the IdList, because a row is bound to an id
// by position and the ids have to be in hand first
static EffectsV3Status _ReadAttrIds
(
	FILE *stream,
	AttributeID **attr_ids,
	uint16_t *n_attrs
) {
	*attr_ids = NULL ;
	*n_attrs  = 0 ;

	uint16_t n ;
	if (!_ReadU16 (stream, &n)) {
		return EFFECTS_V3_TRUNCATED ;
	}

	if (n == 0) {
		return EFFECTS_V3_OK ;
	}

	const long remaining = fstream_remaining (stream) ;
	if (remaining < 0 ||
		(uint64_t)n * sizeof (AttributeID) > (uint64_t)remaining) {
		return EFFECTS_V3_MALFORMED ;
	}

	AttributeID *ids = rm_malloc (n * sizeof (AttributeID)) ;
	for (uint16_t i = 0 ; i < n ; i++) {
		if (!_ReadU16 (stream, ids + i)) {
			rm_free (ids) ;
			return EFFECTS_V3_TRUNCATED ;
		}
	}

	*attr_ids = ids ;
	*n_attrs  = n ;
	return EFFECTS_V3_OK ;
}

// read a RelType: a single i32
static EffectsV3Status _ReadRelType
(
	FILE *stream,
	RelationID *r
) {
	int32_t id ;
	if (!_ReadI32 (stream, &id)) {
		return EFFECTS_V3_TRUNCATED ;
	}

	*r = (RelationID)id ;
	return EFFECTS_V3_OK ;
}

//------------------------------------------------------------------------------
// records
//------------------------------------------------------------------------------

// which records carry what - see the table in effects_v3.h
static bool _IsNodeShaped
(
	EffectType t
) {
	return t == EFFECT_UPDATE_NODE   || t == EFFECT_CREATE_NODE ||
		   t == EFFECT_DELETE_NODE   || t == EFFECT_SET_LABELS  ||
		   t == EFFECT_REMOVE_LABELS ;
}

static bool _IsEdgeShaped
(
	EffectType t
) {
	return t == EFFECT_UPDATE_EDGE || t == EFFECT_CREATE_EDGE ||
		   t == EFFECT_DELETE_EDGE ;
}

static bool _HasValues
(
	EffectType t
) {
	return t == EFFECT_UPDATE_NODE || t == EFFECT_UPDATE_EDGE ||
		   t == EFFECT_CREATE_NODE || t == EFFECT_CREATE_EDGE ;
}

static bool _HasEndpoints
(
	EffectType t
) {
	return t == EFFECT_CREATE_EDGE || t == EFFECT_DELETE_EDGE ;
}

static void _RecordFree
(
	EffectsV3Record *rec
) {
	_IdListFree (&rec->ids) ;
	_IdListFree (&rec->src) ;
	_IdListFree (&rec->dst) ;

	if (rec->labels != NULL) {
		rm_free (rec->labels) ;
		rec->labels = NULL ;
	}

	if (rec->attr_ids != NULL) {
		rm_free (rec->attr_ids) ;
		rec->attr_ids = NULL ;
	}

	if (rec->values != NULL) {
		for (uint64_t i = 0 ; i < rec->n_values ; i++) {
			SIValue_Free (rec->values[i]) ;
		}
		rm_free (rec->values) ;
		rec->values   = NULL ;
		rec->n_values = 0 ;
	}

	if (rec->name != NULL) {
		rm_free (rec->name) ;
		rec->name = NULL ;
	}
}

// read ADD_SCHEMA: SchemaType, the assigned id, then the name
//
// this record and ADD_ATTRIBUTE are the two that ESTABLISH an id space, which
// is why v3 puts the id on the wire: the replica computes the id it would
// assign and refuses the buffer if it disagrees, so a numbering divergence
// surfaces where it is introduced rather than never. Checking that is
// EffectsV3_Apply's job - decode's job is to carry it.
static EffectsV3Status _ReadAddSchema
(
	FILE *stream,
	EffectsV3Record *rec
) {
	uint32_t t ;
	if (!_ReadU32 (stream, &t)) {
		return EFFECTS_V3_TRUNCATED ;
	}

	if (t != SCHEMA_NODE && t != SCHEMA_EDGE) {
		return EFFECTS_V3_MALFORMED ;
	}
	rec->schema_type = (SchemaType)t ;

	int32_t id ;
	if (!_ReadI32 (stream, &id)) {
		return EFFECTS_V3_TRUNCATED ;
	}
	rec->schema_id = id ;

	rec->name = ReadWireString (stream) ;
	if (rec->name == NULL) {
		return EFFECTS_V3_MALFORMED ;
	}

	return EFFECTS_V3_OK ;
}

// read ADD_ATTRIBUTE: the assigned AttributeID, then the name
//
// no node/relationship discriminator: #2459 unified the two attribute
// dictionaries, and C has always had a single one
static EffectsV3Status _ReadAddAttribute
(
	FILE *stream,
	EffectsV3Record *rec
) {
	if (!_ReadU16 (stream, &rec->attr_id)) {
		return EFFECTS_V3_TRUNCATED ;
	}

	rec->name = ReadWireString (stream) ;
	if (rec->name == NULL) {
		return EFFECTS_V3_MALFORMED ;
	}

	return EFFECTS_V3_OK ;
}

// read the value rows: count * n_attrs SIValues, row-major
//
// row k is values[k * n_attrs ..] and belongs to the k-th id in the record's
// IdList AS WRITTEN - id order is the only thing binding values to entities,
// since no per-row id is sent.
//
// T_NULL in a slot means REMOVE THIS ATTRIBUTE. FalkorDB never stores a null
// property, so SET n.x = NULL is a removal and this is what replicates it. A
// reader that filtered nulls out would turn every removal into a no-op, so
// they are carried through verbatim.
//
// The SIValue codec is REUSED, not reimplemented. Writing a second one is how
// the Rust side acquired a bug where a replica's string pool stayed empty, and
// interned strings are already correct in C - T_INTERN_STRING is written
// verbatim and read back through SI_InternStringVal. SIValue_FromBinary bounds
// every wire length against the bytes remaining and reports failure, so a
// truncated or corrupt value is caught per row here.
static EffectsV3Status _ReadValues
(
	FILE *stream,
	EffectsV3Record *rec
) {
	// count is u32 and n_attrs is u16, so the product cannot overflow u64
	const uint64_t n_values = (uint64_t)rec->count * (uint64_t)rec->n_attrs ;

	// AN EMPTY ATTRIBUTE SET IS A LEGITIMATE SHAPE, not a malformed record.
	// `CREATE (:Person)` and `CREATE (a)-[:R]->(b)` create entities with no
	// properties at all, so n_attrs is 0 and there are simply no value rows to
	// read - the record carries its ids and stops.
	//
	// An earlier version refused this, reasoning that "a shape is exact, so a
	// record with entities but no attribute ids states nothing about them".
	// That had it backwards: the empty shape states precisely that these
	// entities have no properties, which is exact. It refused the most ordinary
	// write there is, and no fixture in the corpus exercises a create with zero
	// attributes, which is why it survived a full corpus run.
	if (n_values == 0) {
		return EFFECTS_V3_OK ;
	}

	// the smallest a value can be is its 4-byte SIType, so this rejects a
	// count that outruns the payload before it sizes an allocation. It is a
	// floor derived from the payload's own length, not an invented ceiling.
	const long remaining = fstream_remaining (stream) ;
	if (remaining < 0 ||
		n_values * SIVALUE_MIN_BYTES > (uint64_t)remaining) {
		return EFFECTS_V3_MALFORMED ;
	}

	rec->values   = rm_calloc (n_values, sizeof (SIValue)) ;
	rec->n_values = n_values ;

	for (uint64_t i = 0 ; i < n_values ; i++) {
		if (!SIValue_FromBinary (stream, rec->values + i)) {
			// on failure 'out' is still set to something safe to free, so this
			// slot is included in what the record's free path releases
			rec->n_values = i + 1 ;

			// SIValue_FromBinary does not distinguish the two, so the stream
			// does: out of bytes is truncation, anything else is malformed
			return feof (stream) ? EFFECTS_V3_TRUNCATED
			                     : EFFECTS_V3_MALFORMED ;
		}
	}

	return EFFECTS_V3_OK ;
}

//------------------------------------------------------------------------------
// records 11-14 - index and constraint DDL
//------------------------------------------------------------------------------

// read a v3 OPTIONS map
//
// THE ONE PLACE v2 AND v3 DISAGREE ABOUT A VALUE'S FRAMING, so it cannot go
// through SIValue_FromBinary:
//
//     v3   u32 T_MAP · u32 n · (u64 len · key bytes · SIValue value) × n
//     v2   u32 T_MAP · u32 n · (SIValue key         · SIValue value) × n
//
// v3 writes the key as a bare length-prefixed string. C's Map_FromBinary reads
// it as a full SIValue, tag and all, so handing it v3 bytes makes it read the
// key's u64 length as a SIType - for a short key that is a small number, and 4
// is T_EDGE, which has no binary form. It refuses rather than misparsing, which
// is loud, and is how the difference was found.
//
// A key needs no tag because it is always a string: map.c narrows it straight
// back with `SI_TYPE(key) & T_STRING` at nine call sites. Tagging it would put
// C's type-enum discriminant on the wire as if it were format.
//
// v2's tagged reader is untouched, so traffic between released C engines is
// unaffected. This is the same shape effects_apply.c already uses for the label
// records, where version 1 and version 2 take different readers.
static EffectsV3Status _ReadOptionsMap
(
	FILE *stream,
	SIValue *out
) {
	*out = SI_NullVal () ;

	uint32_t n ;
	if (!_ReadU32 (stream, &n)) {
		return EFFECTS_V3_TRUNCATED ;
	}

	// a pair is at minimum a u64 key length and a bare SIType, so a count
	// larger than the bytes remaining is corruption - checked before it
	// reserves anything
	const long remaining = fstream_remaining (stream) ;
	const uint64_t min_pair = sizeof (uint64_t) + sizeof (uint32_t) ;
	if (remaining < 0 || (uint64_t)n * min_pair > (uint64_t)remaining) {
		return EFFECTS_V3_MALFORMED ;
	}

	SIValue map = Map_New (n) ;

	for (uint32_t i = 0 ; i < n ; i++) {
		// the key: a bare wire string, NOT an SIValue
		char *key = ReadWireString (stream) ;
		if (key == NULL) {
			Map_Free (map) ;
			return EFFECTS_V3_MALFORMED ;
		}

		SIValue val ;
		if (!SIValue_FromBinary (stream, &val)) {
			rm_free (key) ;
			Map_Free (map) ;
			return feof (stream) ? EFFECTS_V3_TRUNCATED
			                     : EFFECTS_V3_MALFORMED ;
		}

		// Map_AddNoClone takes ownership of both
		Map_AddNoClone (&map, SI_TransferStringVal (key), val) ;
	}

	*out = map ;
	return EFFECTS_V3_OK ;
}

// read CREATE_INDEX's options field
//
// always a map in practice, but the type tag is read first so a non-map is
// delegated rather than misread. T_MAP cannot appear anywhere else in a v3
// payload: SI_VALID_PROPERTY_VALUE (value.h) excludes it, so no node or edge
// property can be one, and this is the only field that carries it.
static EffectsV3Status _ReadOptions
(
	FILE *stream,
	SIValue *out,
	bool *has_options
) {
	*out         = SI_NullVal () ;
	*has_options = false ;

	uint32_t t ;
	if (!_ReadU32 (stream, &t)) {
		return EFFECTS_V3_TRUNCATED ;
	}

	if (t == T_MAP) {
		const EffectsV3Status status = _ReadOptionsMap (stream, out) ;
		*has_options = (status == EFFECTS_V3_OK) ;
		return status ;
	}

	// not a map: rewind over the tag and let the shared codec have it
	if (fseek (stream, -((long)sizeof (uint32_t)), SEEK_CUR) != 0) {
		return EFFECTS_V3_MALFORMED ;
	}

	if (!SIValue_FromBinary (stream, out)) {
		return feof (stream) ? EFFECTS_V3_TRUNCATED : EFFECTS_V3_MALFORMED ;
	}

	*has_options = true ;
	return EFFECTS_V3_OK ;
}

// read a counted list of (attribute id, attribute name) pairs
//
// shared by index fields and constraint properties, which are the same shape.
// THE COUNT WIDTH IS NOT: an index field count is a u16 and a constraint
// property count is a u8, so the caller reads it and passes it in.
static EffectsV3Status _ReadAttrRefs
(
	FILE *stream,
	uint16_t n,
	EffectsV3AttrRef **out,
	uint16_t *out_n
) {
	*out   = NULL ;
	*out_n = 0 ;

	if (n == 0) {
		return EFFECTS_V3_OK ;
	}

	// each pair is at least an AttributeID and a one-byte string plus its
	// length prefix, so bound the count before it sizes an allocation
	const long remaining = fstream_remaining (stream) ;
	const uint64_t min_ref = sizeof (AttributeID) + sizeof (uint64_t) + 1 ;
	if (remaining < 0 || (uint64_t)n * min_ref > (uint64_t)remaining) {
		return EFFECTS_V3_MALFORMED ;
	}

	EffectsV3AttrRef *refs = rm_calloc (n, sizeof (EffectsV3AttrRef)) ;

	for (uint16_t i = 0 ; i < n ; i++) {
		if (!_ReadU16 (stream, &refs[i].id)) {
			for (uint16_t j = 0 ; j < i ; j++) rm_free (refs[j].name) ;
			rm_free (refs) ;
			return EFFECTS_V3_TRUNCATED ;
		}

		refs[i].name = ReadWireString (stream) ;
		if (refs[i].name == NULL) {
			for (uint16_t j = 0 ; j < i ; j++) rm_free (refs[j].name) ;
			rm_free (refs) ;
			return EFFECTS_V3_MALFORMED ;
		}
	}

	*out   = refs ;
	*out_n = n ;
	return EFFECTS_V3_OK ;
}

// read CREATE_INDEX (11) and DROP_INDEX (12)
//
// ONE RECORD PER STATEMENT, which v2 was not - v2 sent one record per field.
// Two single-field records are NOT equivalent to one two-field statement: the
// second is refused with "Can not override index configuration", because
// index-level options belong to the index and cannot be set twice. So the field
// type is stated once at statement level, ahead of a counted field list.
//
// A drop carries n = 0 and no options.
static EffectsV3Status _ReadIndexRecord
(
	FILE *stream,
	EffectsV3Record *rec,
	bool create
) {
	uint32_t t ;
	if (!_ReadU32 (stream, &t)) {
		return EFFECTS_V3_TRUNCATED ;
	}
	if (t != SCHEMA_NODE && t != SCHEMA_EDGE) {
		return EFFECTS_V3_MALFORMED ;
	}
	rec->schema_type = (SchemaType)t ;

	int32_t label_id ;
	if (!_ReadI32 (stream, &label_id)) {
		return EFFECTS_V3_TRUNCATED ;
	}
	rec->schema_id = label_id ;

	rec->name = ReadWireString (stream) ;
	if (rec->name == NULL) {
		return EFFECTS_V3_MALFORMED ;
	}

	// IndexFieldType is a BIT FLAG SET, not a discriminant - a range index is
	// NUMERIC|GEO|STR == 0x0E. Not range-checked here for that reason: the
	// meaningful test is against the local index API, at apply.
	if (!_ReadU32 (stream, &rec->field_type)) {
		return EFFECTS_V3_TRUNCATED ;
	}

	// index field count is a u16
	uint16_t n_fields ;
	if (!_ReadU16 (stream, &n_fields)) {
		return EFFECTS_V3_TRUNCATED ;
	}

	EffectsV3Status status = _ReadAttrRefs (stream, n_fields,
			&rec->attrs_ref, &rec->n_attrs_ref) ;
	if (status != EFFECTS_V3_OK) {
		return status ;
	}

	if (create) {
		return _ReadOptions (stream, &rec->options, &rec->has_options) ;
	}

	return EFFECTS_V3_OK ;
}

// read CREATE_CONSTRAINT (13) and DROP_CONSTRAINT (14)
//
// CREATE carries a ConstraintStatus that C never sends, and DROP omits it -
// which is why rec_drop_constraint decoded under the old prose and the other
// three did not.
static EffectsV3Status _ReadConstraintRecord
(
	FILE *stream,
	EffectsV3Record *rec,
	bool create
) {
	if (!_ReadU32 (stream, &rec->constraint_type)) {
		return EFFECTS_V3_TRUNCATED ;
	}

	// GraphEntityType is 1-BASED: GETYPE_UNKNOWN takes 0, so a node is 1
	if (!_ReadU32 (stream, &rec->entity_type)) {
		return EFFECTS_V3_TRUNCATED ;
	}

	if (create) {
		if (!_ReadU32 (stream, &rec->status)) {
			return EFFECTS_V3_TRUNCATED ;
		}
		rec->has_status = true ;
	}

	int32_t label_id ;
	if (!_ReadI32 (stream, &label_id)) {
		return EFFECTS_V3_TRUNCATED ;
	}
	rec->schema_id = label_id ;

	rec->name = ReadWireString (stream) ;
	if (rec->name == NULL) {
		return EFFECTS_V3_MALFORMED ;
	}

	// the constraint property count is a u8, NOT the u16 used everywhere else.
	// Read from C source rather than inferred, and it is the trap in this
	// record: reading it as a u16 swallows the first attribute id's low byte.
	uint8_t n_props ;
	if (!_ReadU8 (stream, &n_props)) {
		return EFFECTS_V3_TRUNCATED ;
	}

	return _ReadAttrRefs (stream, n_props, &rec->attrs_ref,
			&rec->n_attrs_ref) ;
}

// read one record: opcode, then whatever that opcode carries
static EffectsV3Status _ReadRecord
(
	FILE *stream,
	EffectsV3Record *rec
) {
	memset (rec, 0, sizeof (*rec)) ;

	uint32_t opcode ;
	if (!_ReadU32 (stream, &opcode)) {
		return EFFECTS_V3_TRUNCATED ;
	}

	if (opcode <= EFFECT_UNKNOWN || opcode > EFFECT_DROP_CONSTRAINT) {
		return EFFECTS_V3_MALFORMED ;
	}
	rec->opcode = (EffectType)opcode ;

	// the two inherently singular records: no count, no ids
	if (rec->opcode == EFFECT_ADD_SCHEMA) {
		return _ReadAddSchema (stream, rec) ;
	}
	if (rec->opcode == EFFECT_ADD_ATTRIBUTE) {
		return _ReadAddAttribute (stream, rec) ;
	}

	// records 11-14 (index and constraint DDL) are a separate PR
	//
	// UNIMPLEMENTED rather than MALFORMED: these bytes are not corrupt, and
	// calling them corrupt sends an operator hunting a wire problem that does
	// not exist. It is still a refusal - the buffer is rejected and the caller
	// treats it as divergence - because silently skipping a record we cannot
	// apply is data loss. The status becomes unreachable once 11-14 land, which
	// is the point: it makes the gap visible instead of disguising it.
	//
	// Rollout consequence: while this is reachable, the version switch must not
	// flip a writer to v3, or a peer emitting DDL gets refused.
	switch (rec->opcode) {
		case EFFECT_CREATE_INDEX:
			return _ReadIndexRecord (stream, rec, true) ;
		case EFFECT_DROP_INDEX:
			return _ReadIndexRecord (stream, rec, false) ;
		case EFFECT_CREATE_CONSTRAINT:
			return _ReadConstraintRecord (stream, rec, true) ;
		case EFFECT_DROP_CONSTRAINT:
			return _ReadConstraintRecord (stream, rec, false) ;
		default:
			break ;
	}

	if (!_ReadU32 (stream, &rec->count)) {
		return EFFECTS_V3_TRUNCATED ;
	}

	// A RECORD DESCRIBING NO ENTITIES IS ILLEGAL, not merely useless.
	//
	// Checked here, at the header, before any block is parsed - 'count' governs
	// the length of every block that follows, so one check covers every record
	// shape. Put it inside the id-list read instead and it would need repeating
	// in three or four places, because that read is per-list and the count
	// belongs to the record.
	//
	// Refused rather than tolerated, and the reason is NOT "it means nothing" -
	// that invites a later reader to relax it as harmless. It is refused
	// because a record carrying no information is a SYMPTOM of something wrong
	// upstream, and applying nothing silently is the worst available response:
	// the divergence guard exists for exactly this class of fault, and a no-op
	// record would slip straight past it.
	//
	// Two supporting reasons. No emitter can produce one - a group only comes
	// into existence by pushing an id into it, so every group holds at least
	// one. And the format already refuses the same idea one level down, where a
	// zero-length segment is malformed; accepting a zero-entity record while
	// refusing a zero-length segment would be incoherent.
	//
	// Records 9 and 10 are inherently singular and carry no count at all, so
	// they are already dispatched above and unaffected.
	//
	// This SHOULD carry its own status rather than MALFORMED - "the segments do
	// not total the count" and "the count is not a legal count" are different
	// faults, and the log line is all an operator sees. The status enum lives in
	// the shared contract, so adding a variant is the organizer's to make; asked
	// for, and this reverts to it when it exists.
	if (rec->count == 0) {
		return EFFECTS_V3_MALFORMED ;
	}

	EffectsV3Status status ;

	// the shape, hoisted once per record, and stated ahead of the rows
	if (_IsNodeShaped (rec->opcode)) {
		status = _ReadLabelSet (stream, &rec->labels, &rec->n_labels) ;
		if (status != EFFECTS_V3_OK) {
			goto fail ;
		}
	} else if (_IsEdgeShaped (rec->opcode)) {
		status = _ReadRelType (stream, &rec->relation_id) ;
		if (status != EFFECTS_V3_OK) {
			goto fail ;
		}
	}

	if (_HasValues (rec->opcode)) {
		status = _ReadAttrIds (stream, &rec->attr_ids, &rec->n_attrs) ;
		if (status != EFFECTS_V3_OK) {
			goto fail ;
		}
	}

	// the ids, positionally bound to the rows below
	status = _ReadIdList (stream, rec->count, &rec->ids) ;
	if (status != EFFECTS_V3_OK) {
		goto fail ;
	}

	if (_HasEndpoints (rec->opcode)) {
		status = _ReadIdList (stream, rec->count, &rec->src) ;
		if (status != EFFECTS_V3_OK) {
			goto fail ;
		}

		status = _ReadIdList (stream, rec->count, &rec->dst) ;
		if (status != EFFECTS_V3_OK) {
			goto fail ;
		}
	}

	if (_HasValues (rec->opcode)) {
		status = _ReadValues (stream, rec) ;
		if (status != EFFECTS_V3_OK) {
			goto fail ;
		}
	}

	return EFFECTS_V3_OK ;

fail:
	_RecordFree (rec) ;
	return status ;
}

//------------------------------------------------------------------------------
// the entry point
//------------------------------------------------------------------------------

void EffectsV3_RecordsFree
(
	EffectsV3Records *records
) {
	if (records == NULL) {
		return ;
	}

	if (records->records != NULL) {
		for (uint32_t i = 0 ; i < records->n ; i++) {
			_RecordFree (records->records + i) ;
		}
		rm_free (records->records) ;
	}

	rm_free (records) ;
}

EffectsV3Status EffectsV3_Decode
(
	const char *buff,
	size_t n,
	EffectsV3Records **records
) {
	ASSERT (buff    != NULL) ;
	ASSERT (records != NULL) ;

	*records = NULL ;

	if (buff == NULL || records == NULL || n == 0) {
		return EFFECTS_V3_TRUNCATED ;
	}

	FILE *stream = fmemopen ((void*)buff, n, "r") ;
	if (stream == NULL) {
		return EFFECTS_V3_MALFORMED ;
	}

	EffectsV3Status status = EFFECTS_V3_OK ;

	// the header is never compressed, so a reader always knows what it holds
	// before committing to decode anything
	uint8_t version ;
	if (!_ReadU8 (stream, &version)) {
		status = EFFECTS_V3_TRUNCATED ;
		goto done ;
	}

	if (version != 3) {
		status = EFFECTS_V3_UNSUPPORTED_VERSION ;
		goto done ;
	}

	uint8_t flags ;
	if (!_ReadU8 (stream, &flags)) {
		status = EFFECTS_V3_TRUNCATED ;
		goto done ;
	}

	// a flag bit outside the mask we understand rejects the buffer rather than
	// being masked off: an old node meeting a future payload must fail loudly,
	// because decoding the records anyway applies a prefix of something whose
	// shape it does not know
	if (flags & ~FLAGS_KNOWN) {
		status = EFFECTS_V3_UNSUPPORTED_FLAGS ;
		goto done ;
	}

	// compression is understood as a flag but not yet implemented - zstd is a
	// separate PR. Refusing is correct until then: the alternative is reading
	// a zstd frame as records.
	//
	// Refused HERE, before any length field is read, which is why the
	// compressed header's layout does not reach this decoder. For whoever
	// implements it, that header is (e456ec802 on feat/effects-v3):
	//
	//   u8 version . u8 flags . u32 plain_len . u32 comp_len . u32 checksum
	//     . <zstd frame>
	//
	// twelve bytes of prefix, not eight. All little-endian. Three things the
	// reader has to get right, and each exists because of a specific failure:
	//
	//   * read exactly 'comp_len' bytes as the frame, not to the end of the
	//     buffer, and refuse a comp_len that outruns what remains BEFORE zstd
	//     sees it
	//   * bytes after the frame are an error, not padding - ignoring them
	//     silently accepts a truncated-then-appended buffer
	//   * 'plain_len' is the decompress allocation CEILING, not just a
	//     cross-check. A ~100 byte frame of zeros expands to gigabytes, so
	//     bound the output first, then verify the expanded length equals
	//     plain_len, then check the CRC-32 over the plaintext. That order.
	if (flags & FLAG_COMPRESSED) {
		status = EFFECTS_V3_UNSUPPORTED_FLAGS ;
		goto done ;
	}

	EffectsV3Records *out = rm_calloc (1, sizeof (EffectsV3Records)) ;
	out->version = version ;
	out->flags   = flags ;

	uint32_t cap = 0 ;

	// as long as there is data left in the stream
	while ((size_t)ftell (stream) < n) {
		if (out->n == cap) {
			cap = (cap == 0) ? 4 : cap * 2 ;
			out->records = rm_realloc (out->records,
					cap * sizeof (EffectsV3Record)) ;
		}

		status = _ReadRecord (stream, out->records + out->n) ;
		if (status != EFFECTS_V3_OK) {
			EffectsV3_RecordsFree (out) ;
			goto done ;
		}

		out->n++ ;
	}

	*records = out ;

done:
	fclose (stream) ;
	return status ;
}
