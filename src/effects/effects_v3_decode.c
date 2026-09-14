/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "effects_v3.h"
#include "effects_v3_stream.h"
#include "../util/wire_string.h"
#include "../index/index_field.h"
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
// THE WIRE'S kind FIELD, bits 0-1 of the segment header. Distinct from
// EffectsV3IdListSegmentKind, which folds direction in and so has five values
// to the wire's three - these are the numbers on the bytes, and they do not
// move.
#define WIRE_SEG_RANGE  0
#define WIRE_SEG_SET    1
#define WIRE_SEG_REPEAT 2

#define SEGMENT_MIN_BYTES 3

// the smallest an SIValue can be: the u32 SIType on its own, which is exactly
// what T_NULL is
#define SIVALUE_MIN_BYTES 4

// flag bits this build understands (docs/effects-v3.md:104)
// bit 0 = compressed; everything else is reserved and rejected
#define FLAG_COMPRESSED 0x01
#define FLAGS_KNOWN     0x01

// defined with the DDL readers below; declared here because the record's free
// path precedes them
static void _IndexOptionsFree (EffectsV3IndexOptions *o) ;

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
	EffectsV3IdListSegment *seg
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

	// THE WIRE KIND AND BIT 6 TOGETHER SELECT THE ARM. On the wire, direction
	// is a bit beside a three-value kind; in the decoded form it is folded into
	// the kind, so the pairing that has no meaning - a descending Repeat -
	// cannot be represented. The mapping is the only place the two models meet.
	const uint8_t kind       = header & EFFECTS_V3_SEG_KIND_MASK ;
	const uint8_t vw         = (header >> EFFECTS_V3_SEG_VWIDTH_SHIFT) & 0x03 ;
	const uint8_t cw         = (header >> EFFECTS_V3_SEG_CWIDTH_SHIFT) & 0x03 ;
	const bool    descending = (header & EFFECTS_V3_SEG_DESCENDING) != 0 ;

	switch (kind) {
		case WIRE_SEG_RANGE: {
			uint64_t base ;
			uint64_t len ;
			if (!_ReadWidth (stream, vw, &base) ||
				!_ReadWidth (stream, cw, &len)) {
				return EFFECTS_V3_TRUNCATED ;
			}

			// the range has to fit in the id space it is describing. Ascending
			// must not run past UINT64_MAX; descending counts DOWN from base,
			// which is its highest id, so it must not run below zero. Both are
			// derived from the segment's own two numbers, not from a ceiling.
			if (len > 0) {
				const uint64_t span = len - 1 ;
				if (descending) {
					if (span > base) {
						return EFFECTS_V3_MALFORMED ;
					}
				} else {
					if (span > UINT64_MAX - base) {
						return EFFECTS_V3_MALFORMED ;
					}
				}
			}

			// the width codes are retained rather than discarded: re-encoding
			// reproduces the peer's bytes instead of our own narrowest-fit
			// arithmetic, which is what makes a round trip a test of the wire
			// rather than of ourselves
			if (descending) {
				seg->kind = EFFECTS_V3_SEG_RANGE_DESCENDING ;
				seg->range_descending.value_width = vw ;
				seg->range_descending.count_width = cw ;
				seg->range_descending.base        = base ;
				seg->range_descending.len         = len ;
			} else {
				seg->kind = EFFECTS_V3_SEG_RANGE_ASCENDING ;
				seg->range_ascending.value_width = vw ;
				seg->range_ascending.count_width = cw ;
				seg->range_ascending.base        = base ;
				seg->range_ascending.len         = len ;
			}
			return EFFECTS_V3_OK ;
		}

		case WIRE_SEG_REPEAT: {
			// a Repeat holds one id 'count' times, so it reads the same in
			// either direction and the bit cannot mean anything. A peer that
			// set it meant something this build does not know, so refuse the
			// buffer rather than ignore the bit.
			if (descending) {
				return EFFECTS_V3_MALFORMED ;
			}

			seg->kind = EFFECTS_V3_SEG_REPEAT ;
			seg->repeat.value_width = vw ;
			seg->repeat.count_width = cw ;
			if (!_ReadWidth (stream, vw, &seg->repeat.id) ||
				!_ReadWidth (stream, cw, &seg->repeat.count)) {
				return EFFECTS_V3_TRUNCATED ;
			}
			return EFFECTS_V3_OK ;
		}

		case WIRE_SEG_SET: {
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
				// a Set segment describes at least one id, so it cannot carry
				// an empty bitmap
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

			const uint64_t cardinality =
				roaring64_bitmap_get_cardinality (bitmap) ;
			roaring64_bitmap_free (bitmap) ;

			if (cardinality == 0) {
				rm_free (blob) ;
				return EFFECTS_V3_MALFORMED ;
			}

			// the two directions over one id set differ in bit 6 and nowhere
			// else, so the blob is identical and only the arm changes
			if (descending) {
				seg->kind = EFFECTS_V3_SEG_SET_DESCENDING ;
				seg->set_descending.blob        = blob ;
				seg->set_descending.n           = blob_len ;
				seg->set_descending.cardinality = cardinality ;
			} else {
				seg->kind = EFFECTS_V3_SEG_SET_ASCENDING ;
				seg->set_ascending.blob        = blob ;
				seg->set_ascending.n           = blob_len ;
				seg->set_ascending.cardinality = cardinality ;
			}
			return EFFECTS_V3_OK ;
		}

		default:
			// wire kind 3 is not assigned
			return EFFECTS_V3_MALFORMED ;
	}
}

// how many ids a segment describes, without expanding it
static uint64_t _SegmentCardinality
(
	const EffectsV3IdListSegment *seg
) {
	switch (seg->kind) {
		case EFFECTS_V3_SEG_RANGE_ASCENDING:  return seg->range_ascending.len ;
		case EFFECTS_V3_SEG_RANGE_DESCENDING: return seg->range_descending.len ;
		case EFFECTS_V3_SEG_SET_ASCENDING:    return seg->set_ascending.cardinality ;
		case EFFECTS_V3_SEG_SET_DESCENDING:   return seg->set_descending.cardinality ;
		case EFFECTS_V3_SEG_REPEAT:           return seg->repeat.count ;
		default:                              return 0 ;
	}
}

static void _IdListFree
(
	EffectsV3IdList *list
) {
	if (list->segments == NULL) {
		return ;
	}

	// BOTH Set arms own a blob. They are one wire kind read two ways, so a
	// free that names only the ascending arm leaks every descending segment.
	for (uint32_t i = 0 ; i < list->n ; i++) {
		switch (list->segments[i].kind) {
			case EFFECTS_V3_SEG_SET_ASCENDING:
				rm_free (list->segments[i].set_ascending.blob) ;
				break ;
			case EFFECTS_V3_SEG_SET_DESCENDING:
				rm_free (list->segments[i].set_descending.blob) ;
				break ;
			default:
				break ;
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

	list->segments = rm_calloc (n, sizeof (EffectsV3IdListSegment)) ;
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

// POINTERS INTO WHICHEVER ARM THE OPCODE SELECTS
//
// Records 1-8 are read by one routine because they are one wire shape - opcode,
// count, shape, attr ids, IdList(s), values - and only the subset of fields
// each opcode carries differs. The typed model puts those fields in eight
// different structs, so the reader gathers pointers to them once and fills
// through the pointers.
//
// The alternative is eight readers differing by a few lines each. This file
// already fixed two bugs that existed because a rule was written out more than
// once (the per-entry emptiness check, the segment blob free), and the record
// parser is the last place worth duplicating.
//
// A NULL member means THIS OPCODE DOES NOT CARRY THAT FIELD, and the reader
// tests for NULL rather than consulting a second list of which opcode has what.
// That is what makes "DELETE_EDGE has endpoints but no values" a property of
// this table instead of a rule repeated at every use site.
typedef struct {
	uint32_t        *count;
	LabelID        **labels;
	uint16_t        *n_labels;
	RelationID      *relation_id;
	AttributeID    **attr_ids;
	uint16_t        *n_attrs;
	EffectsV3IdList *ids;
	EffectsV3IdList *src;
	EffectsV3IdList *dst;
	SIValue        **values;
	uint64_t        *n_values;
} BatchFields;

// fill 'f' for a batchable record, or return false for records 9-14
static bool _BatchFields
(
	EffectsV3Record *rec,
	BatchFields *f
) {
	memset (f, 0, sizeof (*f)) ;

	switch (rec->opcode) {
		case EFFECT_UPDATE_NODE:
			f->count    = &rec->update_node.count ;
			f->labels   = &rec->update_node.labels ;
			f->n_labels = &rec->update_node.n_labels ;
			f->attr_ids = &rec->update_node.attr_ids ;
			f->n_attrs  = &rec->update_node.n_attrs ;
			f->ids      = &rec->update_node.ids ;
			f->values   = &rec->update_node.values ;
			f->n_values = &rec->update_node.n_values ;
			return true ;

		case EFFECT_UPDATE_EDGE:
			f->count       = &rec->update_edge.count ;
			f->relation_id = &rec->update_edge.relation_id ;
			f->attr_ids    = &rec->update_edge.attr_ids ;
			f->n_attrs     = &rec->update_edge.n_attrs ;
			f->ids         = &rec->update_edge.ids ;
			f->values      = &rec->update_edge.values ;
			f->n_values    = &rec->update_edge.n_values ;
			return true ;

		case EFFECT_CREATE_NODE:
			f->count    = &rec->create_node.count ;
			f->labels   = &rec->create_node.labels ;
			f->n_labels = &rec->create_node.n_labels ;
			f->attr_ids = &rec->create_node.attr_ids ;
			f->n_attrs  = &rec->create_node.n_attrs ;
			f->ids      = &rec->create_node.ids ;
			f->values   = &rec->create_node.values ;
			f->n_values = &rec->create_node.n_values ;
			return true ;

		case EFFECT_CREATE_EDGE:
			f->count       = &rec->create_edge.count ;
			f->relation_id = &rec->create_edge.relation_id ;
			f->attr_ids    = &rec->create_edge.attr_ids ;
			f->n_attrs     = &rec->create_edge.n_attrs ;
			f->ids         = &rec->create_edge.ids ;
			f->src         = &rec->create_edge.src ;
			f->dst         = &rec->create_edge.dst ;
			f->values      = &rec->create_edge.values ;
			f->n_values    = &rec->create_edge.n_values ;
			return true ;

		case EFFECT_DELETE_NODE:
			f->count    = &rec->delete_node.count ;
			f->labels   = &rec->delete_node.labels ;
			f->n_labels = &rec->delete_node.n_labels ;
			f->ids      = &rec->delete_node.ids ;
			return true ;

		case EFFECT_DELETE_EDGE:
			f->count       = &rec->delete_edge.count ;
			f->relation_id = &rec->delete_edge.relation_id ;
			f->ids         = &rec->delete_edge.ids ;
			f->src         = &rec->delete_edge.src ;
			f->dst         = &rec->delete_edge.dst ;
			return true ;

		case EFFECT_SET_LABELS:
			f->count    = &rec->set_labels.count ;
			f->labels   = &rec->set_labels.labels ;
			f->n_labels = &rec->set_labels.n_labels ;
			f->ids      = &rec->set_labels.ids ;
			return true ;

		case EFFECT_REMOVE_LABELS:
			f->count    = &rec->remove_labels.count ;
			f->labels   = &rec->remove_labels.labels ;
			f->n_labels = &rec->remove_labels.n_labels ;
			f->ids      = &rec->remove_labels.ids ;
			return true ;

		default:
			return false ;
	}
}

// the (attrs, n_attrs, name) triple the four DDL records share, by arm
static void _DDLRefs
(
	EffectsV3Record *rec,
	EffectsV3AttrRef ***attrs,
	uint16_t **n_attrs,
	char ***name
) {
	switch (rec->opcode) {
		case EFFECT_CREATE_INDEX:
			*attrs = &rec->create_index.attrs ;
			*n_attrs = &rec->create_index.n_attrs ;
			*name = &rec->create_index.name ;
			return ;
		case EFFECT_DROP_INDEX:
			*attrs = &rec->drop_index.attrs ;
			*n_attrs = &rec->drop_index.n_attrs ;
			*name = &rec->drop_index.name ;
			return ;
		case EFFECT_CREATE_CONSTRAINT:
			*attrs = &rec->create_constraint.attrs ;
			*n_attrs = &rec->create_constraint.n_attrs ;
			*name = &rec->create_constraint.name ;
			return ;
		case EFFECT_DROP_CONSTRAINT:
			*attrs = &rec->drop_constraint.attrs ;
			*n_attrs = &rec->drop_constraint.n_attrs ;
			*name = &rec->drop_constraint.name ;
			return ;
		default:
			*attrs = NULL ; *n_attrs = NULL ; *name = NULL ;
			return ;
	}
}

// free whatever the record's arm owns
//
// FREED BY ARM, and a record that failed part-way through decode is freed by
// the same path - _ReadRecord memsets the record before reading, so an arm the
// reader never reached is all-zero and every branch below is a no-op on it.
// That is what lets the fail path call this unconditionally.
static void _RecordFree
(
	EffectsV3Record *rec
) {
	BatchFields f ;
	if (_BatchFields (rec, &f)) {
		if (f.ids != NULL) _IdListFree (f.ids) ;
		if (f.src != NULL) _IdListFree (f.src) ;
		if (f.dst != NULL) _IdListFree (f.dst) ;

		if (f.labels != NULL && *f.labels != NULL) {
			rm_free (*f.labels) ;
			*f.labels = NULL ;
		}

		if (f.attr_ids != NULL && *f.attr_ids != NULL) {
			rm_free (*f.attr_ids) ;
			*f.attr_ids = NULL ;
		}

		if (f.values != NULL && *f.values != NULL) {
			for (uint64_t i = 0 ; i < *f.n_values ; i++) {
				SIValue_Free ((*f.values)[i]) ;
			}
			rm_free (*f.values) ;
			*f.values   = NULL ;
			*f.n_values = 0 ;
		}
		return ;
	}

	if (rec->opcode == EFFECT_ADD_SCHEMA) {
		if (rec->add_schema.name != NULL) {
			rm_free (rec->add_schema.name) ;
			rec->add_schema.name = NULL ;
		}
		return ;
	}

	if (rec->opcode == EFFECT_ADD_ATTRIBUTE) {
		if (rec->add_attribute.name != NULL) {
			rm_free (rec->add_attribute.name) ;
			rec->add_attribute.name = NULL ;
		}
		return ;
	}

	// records 11-14
	EffectsV3AttrRef **attrs ;
	uint16_t *n_attrs ;
	char **name ;
	_DDLRefs (rec, &attrs, &n_attrs, &name) ;
	if (attrs == NULL) {
		return ;
	}

	if (*name != NULL) {
		rm_free (*name) ;
		*name = NULL ;
	}

	if (*attrs != NULL) {
		for (uint16_t i = 0 ; i < *n_attrs ; i++) {
			rm_free ((*attrs)[i].name) ;
		}
		rm_free (*attrs) ;
		*attrs   = NULL ;
		*n_attrs = 0 ;
	}

	// only CREATE_INDEX has options at all - DROP_INDEX carries none, so there
	// is no has_options to test on it
	if (rec->opcode == EFFECT_CREATE_INDEX && rec->create_index.has_options) {
		_IndexOptionsFree (&rec->create_index.options) ;
		rec->create_index.has_options = false ;
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
	rec->add_schema.schema_type = (SchemaType)t ;

	int32_t id ;
	if (!_ReadI32 (stream, &id)) {
		return EFFECTS_V3_TRUNCATED ;
	}
	rec->add_schema.schema_id = id ;

	rec->add_schema.name = ReadWireString (stream) ;
	if (rec->add_schema.name == NULL) {
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
	if (!_ReadU16 (stream, &rec->add_attribute.attr_id)) {
		return EFFECTS_V3_TRUNCATED ;
	}

	rec->add_attribute.name = ReadWireString (stream) ;
	if (rec->add_attribute.name == NULL) {
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
	const BatchFields *f
) {
	// count is u32 and n_attrs is u16, so the product cannot overflow u64
	const uint64_t n_values = (uint64_t)(*f->count) * (uint64_t)(*f->n_attrs) ;

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

	*f->values   = rm_calloc (n_values, sizeof (SIValue)) ;
	*f->n_values = n_values ;

	for (uint64_t i = 0 ; i < n_values ; i++) {
		if (!SIValue_FromBinary (stream, *f->values + i)) {
			// on failure 'out' is still set to something safe to free, so this
			// slot is included in what the record's free path releases
			*f->n_values = i + 1 ;

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

// read a presence byte: 0 absent, 1 present, anything else malformed
static EffectsV3Status _ReadPresence
(
	FILE *stream,
	bool *present
) {
	uint8_t p ;
	if (!_ReadU8 (stream, &p)) {
		return EFFECTS_V3_TRUNCATED ;
	}
	if (p > 1) {
		return EFFECTS_V3_MALFORMED ;
	}
	*present = (p == 1) ;
	return EFFECTS_V3_OK ;
}

// read an optional u64: presence byte then, if present, the value
static EffectsV3Status _ReadOptionalU64
(
	FILE *stream,
	bool *has,
	uint64_t *v
) {
	const EffectsV3Status status = _ReadPresence (stream, has) ;
	if (status != EFFECTS_V3_OK || !*has) {
		return status ;
	}
	return fread_checked (v, sizeof (*v), stream) ? EFFECTS_V3_OK
	                                              : EFFECTS_V3_TRUNCATED ;
}

// read an optional wire string
static EffectsV3Status _ReadOptionalString
(
	FILE *stream,
	bool *has,
	char **out
) {
	*out = NULL ;

	const EffectsV3Status status = _ReadPresence (stream, has) ;
	if (status != EFFECTS_V3_OK || !*has) {
		return status ;
	}

	*out = ReadWireString (stream) ;
	return (*out != NULL) ? EFFECTS_V3_OK : EFFECTS_V3_MALFORMED ;
}

static void _IndexOptionsFree
(
	EffectsV3IndexOptions *o
) {
	if (o->language != NULL) { rm_free (o->language) ; o->language = NULL ; }
	if (o->phonetic != NULL) { rm_free (o->phonetic) ; o->phonetic = NULL ; }

	if (o->stopwords != NULL) {
		for (uint64_t i = 0 ; i < o->n_stopwords ; i++) {
			rm_free (o->stopwords[i]) ;
		}
		rm_free (o->stopwords) ;
		o->stopwords   = NULL ;
		o->n_stopwords = 0 ;
	}
}

// read CREATE_INDEX's options: a TYPED BLOCK, not a map
//
// See EffectsV3IndexOptions for the layout and for why the presence bytes
// matter. In short: absence means the statement did not say, NOT the default -
// an effect mutates an index that may already exist, so substituting a default
// is what produced "Can not override index configuration" on a live replica.
//
// The text half is unconditional whatever the field type; only the vector half
// is gated, on INDEX_FLD_VECTOR.
static EffectsV3Status _ReadIndexOptions
(
	FILE *stream,
	uint32_t field_type,
	EffectsV3IndexOptions *o
) {
	memset (o, 0, sizeof (*o)) ;

	EffectsV3Status status ;

	status = _ReadOptionalString (stream, &o->has_language, &o->language) ;
	if (status != EFFECTS_V3_OK) goto fail ;

	//--------------------------------------------------------------------------
	// stopwords: presence, then a count, then that many strings
	//--------------------------------------------------------------------------

	status = _ReadPresence (stream, &o->has_stopwords) ;
	if (status != EFFECTS_V3_OK) goto fail ;

	if (o->has_stopwords) {
		uint64_t n ;
		if (!fread_checked (&n, sizeof (n), stream)) {
			status = EFFECTS_V3_TRUNCATED ;
			goto fail ;
		}

		// the smallest stopword on the wire is a u64 length plus its NUL, so
		// reject a count that outruns the payload before it sizes anything
		const long remaining = fstream_remaining (stream) ;
		if (remaining < 0 ||
			n > (uint64_t)remaining / (sizeof (uint64_t) + 1)) {
			status = EFFECTS_V3_MALFORMED ;
			goto fail ;
		}

		if (n > 0) {
			o->stopwords = rm_calloc (n, sizeof (char*)) ;
			for (uint64_t i = 0 ; i < n ; i++) {
				o->stopwords[i] = ReadWireString (stream) ;
				if (o->stopwords[i] == NULL) {
					o->n_stopwords = i ;
					status = EFFECTS_V3_MALFORMED ;
					goto fail ;
				}
			}
		}
		o->n_stopwords = n ;
	}

	//--------------------------------------------------------------------------
	// weight, nostem, phonetic
	//--------------------------------------------------------------------------

	status = _ReadPresence (stream, &o->has_weight) ;
	if (status != EFFECTS_V3_OK) goto fail ;
	if (o->has_weight &&
		!fread_checked (&o->weight, sizeof (o->weight), stream)) {
		status = EFFECTS_V3_TRUNCATED ;
		goto fail ;
	}

	status = _ReadPresence (stream, &o->has_nostem) ;
	if (status != EFFECTS_V3_OK) goto fail ;
	if (o->has_nostem) {
		uint8_t v ;
		if (!_ReadU8 (stream, &v)) {
			status = EFFECTS_V3_TRUNCATED ;
			goto fail ;
		}
		// stated as 1 or 0; anything else is a value this build cannot
		// interpret rather than a truthy byte to coerce
		if (v > 1) {
			status = EFFECTS_V3_MALFORMED ;
			goto fail ;
		}
		o->nostem = (v == 1) ;
	}

	status = _ReadOptionalString (stream, &o->has_phonetic, &o->phonetic) ;
	if (status != EFFECTS_V3_OK) goto fail ;

	//--------------------------------------------------------------------------
	// the vector half, gated on the field type
	//--------------------------------------------------------------------------

	if (field_type & INDEX_FLD_VECTOR) {
		o->is_vector = true ;

		// dimension has NO presence byte: a vector field must have one
		if (!fread_checked (&o->dimension, sizeof (o->dimension), stream)) {
			status = EFFECTS_V3_TRUNCATED ;
			goto fail ;
		}

		status = _ReadOptionalU64 (stream, &o->has_m, &o->m) ;
		if (status != EFFECTS_V3_OK) goto fail ;

		status = _ReadOptionalU64 (stream, &o->has_ef_construction,
				&o->ef_construction) ;
		if (status != EFFECTS_V3_OK) goto fail ;

		status = _ReadOptionalU64 (stream, &o->has_ef_runtime,
				&o->ef_runtime) ;
		if (status != EFFECTS_V3_OK) goto fail ;

		status = _ReadOptionalU64 (stream, &o->has_sim_func, &o->sim_func) ;
		if (status != EFFECTS_V3_OK) goto fail ;
	}

	return EFFECTS_V3_OK ;

fail:
	_IndexOptionsFree (o) ;
	return status ;
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
	// CREATE_INDEX and DROP_INDEX read an identical prefix into two different
	// arms. The fields are bound once here so the read below is written once -
	// the arms diverge only at 'options', which a drop does not have AT ALL
	// rather than having-and-leaving-unset.
	SchemaType  *schema_type ;
	int         *schema_id ;
	char       **name ;
	uint32_t    *field_type ;
	EffectsV3AttrRef **attrs ;
	uint16_t    *n_attrs ;

	if (create) {
		schema_type = &rec->create_index.schema_type ;
		schema_id   = &rec->create_index.schema_id ;
		name        = &rec->create_index.name ;
		field_type  = &rec->create_index.field_type ;
		attrs       = &rec->create_index.attrs ;
		n_attrs     = &rec->create_index.n_attrs ;
	} else {
		schema_type = &rec->drop_index.schema_type ;
		schema_id   = &rec->drop_index.schema_id ;
		name        = &rec->drop_index.name ;
		field_type  = &rec->drop_index.field_type ;
		attrs       = &rec->drop_index.attrs ;
		n_attrs     = &rec->drop_index.n_attrs ;
	}

	uint32_t t ;
	if (!_ReadU32 (stream, &t)) {
		return EFFECTS_V3_TRUNCATED ;
	}
	if (t != SCHEMA_NODE && t != SCHEMA_EDGE) {
		return EFFECTS_V3_MALFORMED ;
	}
	*schema_type = (SchemaType)t ;

	int32_t label_id ;
	if (!_ReadI32 (stream, &label_id)) {
		return EFFECTS_V3_TRUNCATED ;
	}
	*schema_id = label_id ;

	*name = ReadWireString (stream) ;
	if (*name == NULL) {
		return EFFECTS_V3_MALFORMED ;
	}

	// IndexFieldType is a BIT FLAG SET, not a discriminant - a range index is
	// NUMERIC|GEO|STR == 0x0E. Not range-checked here for that reason: the
	// meaningful test is against the local index API, at apply.
	if (!_ReadU32 (stream, field_type)) {
		return EFFECTS_V3_TRUNCATED ;
	}

	// index field count is a u16
	uint16_t n_fields ;
	if (!_ReadU16 (stream, &n_fields)) {
		return EFFECTS_V3_TRUNCATED ;
	}

	EffectsV3Status status = _ReadAttrRefs (stream, n_fields, attrs, n_attrs) ;
	if (status != EFFECTS_V3_OK) {
		return status ;
	}

	if (create) {
		status = _ReadIndexOptions (stream, rec->create_index.field_type,
				&rec->create_index.options) ;
		rec->create_index.has_options = (status == EFFECTS_V3_OK) ;
		return status ;
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
	// bound once, for the same reason as the index pair: the two records read
	// an identical prefix and differ only in that a create carries a status.
	// DROP_CONSTRAINT has no 'status' field at all, so there is nothing to
	// leave unset and nothing for a later reader to wonder about.
	uint32_t *constraint_type ;
	uint32_t *entity_type ;
	int      *schema_id ;
	char    **name ;
	EffectsV3AttrRef **attrs ;
	uint16_t *n_attrs ;

	if (create) {
		constraint_type = &rec->create_constraint.constraint_type ;
		entity_type     = &rec->create_constraint.entity_type ;
		schema_id       = &rec->create_constraint.schema_id ;
		name            = &rec->create_constraint.name ;
		attrs           = &rec->create_constraint.attrs ;
		n_attrs         = &rec->create_constraint.n_attrs ;
	} else {
		constraint_type = &rec->drop_constraint.constraint_type ;
		entity_type     = &rec->drop_constraint.entity_type ;
		schema_id       = &rec->drop_constraint.schema_id ;
		name            = &rec->drop_constraint.name ;
		attrs           = &rec->drop_constraint.attrs ;
		n_attrs         = &rec->drop_constraint.n_attrs ;
	}

	if (!_ReadU32 (stream, constraint_type)) {
		return EFFECTS_V3_TRUNCATED ;
	}

	// GraphEntityType is 1-BASED: GETYPE_UNKNOWN takes 0, so a node is 1
	if (!_ReadU32 (stream, entity_type)) {
		return EFFECTS_V3_TRUNCATED ;
	}

	if (create) {
		if (!_ReadU32 (stream, &rec->create_constraint.status)) {
			return EFFECTS_V3_TRUNCATED ;
		}
		rec->create_constraint.has_status = true ;
	}

	int32_t label_id ;
	if (!_ReadI32 (stream, &label_id)) {
		return EFFECTS_V3_TRUNCATED ;
	}
	*schema_id = label_id ;

	*name = ReadWireString (stream) ;
	if (*name == NULL) {
		return EFFECTS_V3_MALFORMED ;
	}

	// the constraint property count is a u8, NOT the u16 used everywhere else.
	// Read from C source rather than inferred, and it is the trap in this
	// record: reading it as a u16 swallows the first attribute id's low byte.
	uint8_t n_props ;
	if (!_ReadU8 (stream, &n_props)) {
		return EFFECTS_V3_TRUNCATED ;
	}

	return _ReadAttrRefs (stream, n_props, attrs, n_attrs) ;
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

	// WHICH FIELDS THIS OPCODE CARRIES COMES FROM ONE TABLE.
	//
	// This used to ask four predicates - _IsNodeShaped, _IsEdgeShaped,
	// _HasValues, _HasEndpoints - each listing opcodes again. That was a second
	// statement of what the arms already say, and the two could disagree
	// silently: a predicate naming an opcode whose arm has no such field
	// compiles and writes nowhere useful. Now a NULL member means the opcode
	// does not carry the field, and the table is the only place that is said.
	BatchFields f ;
	if (!_BatchFields (rec, &f)) {
		// every non-batchable opcode is dispatched above, so reaching here
		// means the opcode passed the range check and matched no arm
		return EFFECTS_V3_MALFORMED ;
	}

	if (!_ReadU32 (stream, f.count)) {
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
	if (*f.count == 0) {
		return EFFECTS_V3_MALFORMED ;
	}

	EffectsV3Status status ;

	// the shape, hoisted once per record, and stated ahead of the rows. A
	// record is node-shaped or edge-shaped, never both - the arms enforce it,
	// because no arm has 'labels' and 'relation_id' together.
	if (f.labels != NULL) {
		status = _ReadLabelSet (stream, f.labels, f.n_labels) ;
		if (status != EFFECTS_V3_OK) {
			goto fail ;
		}
	} else if (f.relation_id != NULL) {
		status = _ReadRelType (stream, f.relation_id) ;
		if (status != EFFECTS_V3_OK) {
			goto fail ;
		}
	}

	if (f.attr_ids != NULL) {
		status = _ReadAttrIds (stream, f.attr_ids, f.n_attrs) ;
		if (status != EFFECTS_V3_OK) {
			goto fail ;
		}
	}

	// the ids, positionally bound to the rows below
	status = _ReadIdList (stream, *f.count, f.ids) ;
	if (status != EFFECTS_V3_OK) {
		goto fail ;
	}

	if (f.src != NULL) {
		status = _ReadIdList (stream, *f.count, f.src) ;
		if (status != EFFECTS_V3_OK) {
			goto fail ;
		}

		status = _ReadIdList (stream, *f.count, f.dst) ;
		if (status != EFFECTS_V3_OK) {
			goto fail ;
		}
	}

	if (f.values != NULL) {
		status = _ReadValues (stream, &f) ;
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

// decode a payload, handing each record to 'fn' as it is read
//
// the header is parsed here and the record loop runs here; the two public
// entry points differ only in what their callback does with each record
EffectsV3Status EffectsV3_DecodeEach
(
	const char *buff,
	size_t n,
	EffectsV3RecordFn fn,
	void *ctx
) {
	ASSERT (buff != NULL) ;
	ASSERT (fn   != NULL) ;

	if (buff == NULL || fn == NULL || n == 0) {
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

	// ONE RECORD AT A TIME. The record lives on this stack frame and is freed
	// before the next is read, so peak memory is one record rather than the
	// whole payload.
	while ((size_t)ftell (stream) < n) {
		EffectsV3Record rec ;

		status = _ReadRecord (stream, &rec) ;
		if (status != EFFECTS_V3_OK) {
			// _ReadRecord frees what it partially built
			goto done ;
		}

		const bool keep_going = fn (&rec, ctx) ;

		// freed whether or not the callback refused. A callback that kept the
		// record zeroed it, and freeing a zeroed record is a no-op.
		_RecordFree (&rec) ;

		if (!keep_going) {
			// THE CALLBACK REFUSED, and that is not a decode status. The bytes
			// were well formed; what the caller did with them is the caller's
			// to report, and it carries that verdict in 'ctx'. Returning a
			// failure status here would report a refused payload as corrupt
			// and send an operator hunting a wire problem that does not exist.
			break ;
		}
	}

done:
	fclose (stream) ;
	return status ;
}

//------------------------------------------------------------------------------
// the collecting form
//------------------------------------------------------------------------------

// accumulator for EffectsV3_Decode
typedef struct {
	EffectsV3Records *out;
	uint32_t          cap;
	bool              oom;
} _Collector;

// take ownership of each record by copying the struct and zeroing the original
static bool _Collect
(
	EffectsV3Record *rec,
	void *ctx
) {
	_Collector *c = (_Collector*)ctx ;

	if (c->out->n == c->cap) {
		c->cap = (c->cap == 0) ? 4 : c->cap * 2 ;
		c->out->records = rm_realloc (c->out->records,
				c->cap * sizeof (EffectsV3Record)) ;
	}

	c->out->records[c->out->n] = *rec ;
	c->out->n++ ;

	// OWNERSHIP MOVED. The driver frees the record as soon as this returns, so
	// the original is zeroed - otherwise every owned pointer in it would be
	// freed out from under the copy.
	memset (rec, 0, sizeof (*rec)) ;

	return true ;
}

// decode a payload whole
//
// KEPT FOR THE ROUND TRIP. EffectsV3_Encode takes a materialised record set and
// the conformance harness is its only caller, so this wrapper exists to serve
// that and nothing else - the production path streams. A test's needs do not
// get to set the production path's memory profile, and breaking the round trip
// to avoid carrying this would be the wrong trade in the other direction.
EffectsV3Status EffectsV3_Decode
(
	const char *buff,
	size_t n,
	EffectsV3Records **records
) {
	ASSERT (buff    != NULL) ;
	ASSERT (records != NULL) ;

	if (records == NULL) {
		return EFFECTS_V3_TRUNCATED ;
	}

	*records = NULL ;

	if (buff == NULL || n == 0) {
		return EFFECTS_V3_TRUNCATED ;
	}

	_Collector c = { 0 } ;
	c.out = rm_calloc (1, sizeof (EffectsV3Records)) ;

	const EffectsV3Status status = EffectsV3_DecodeEach (buff, n, _Collect, &c) ;

	if (status != EFFECTS_V3_OK) {
		EffectsV3_RecordsFree (c.out) ;
		return status ;
	}

	// the header bytes the caller still expects to see on the record set
	//
	// READ FROM THE PAYLOAD, and read AFTER the driver has returned OK, which
	// is what makes them trustworthy rather than assumed. An earlier version
	// wrote a literal 3 here before decoding; that was correct, because
	// EffectsV3_DecodeEach refuses any other version, but it read as an
	// assumption and it would have become a real defect the moment this
	// function returned a record set on a non-3 payload - the round trip
	// re-encodes from records->version, so it would have produced bytes
	// claiming a version the payload never had.
	//
	// Taking them off the payload after a clean decode makes the value a
	// consequence of the refusal rather than a restatement of it. n >= 2 here:
	// a payload too short for both header bytes cannot decode OK.
	c.out->version = (uint8_t)buff[0] ;
	c.out->flags   = (uint8_t)buff[1] ;

	*records = c.out ;
	return EFFECTS_V3_OK ;
}
