/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "effects.h"

#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>

//------------------------------------------------------------------------------
// effects v3 - the decode/apply seam
//------------------------------------------------------------------------------
//
//     encode:  records --EffectsV3_Encode--> bytes
//     decode:  bytes   --EffectsV3_Decode--> records
//     apply:   records --EffectsV3_Apply-->  Graph
//
// `EffectsV3Records` is a plain value. It holds no GraphContext, no Graph and
// no live entity, so it can be decoded, compared, re-encoded and freed with no
// graph in scope.
//
// Two invariants the type exists to preserve:
//
//   * IDS ARE NOT EXPANDED AT DECODE TIME. An IdList is held as its segments.
//     One valid segment describes four billion ids in seven bytes, and
//     GRAPH.EFFECT is applied inline on the main thread with no timeout, so a
//     decoder whose cost is unbounded by its input size is a denial of service.
//     Expansion happens in EffectsV3_Apply, where the memory is inherent to the
//     write being performed.
//
//   * DECODE THEN ENCODE REPRODUCES THE INPUT BYTES. Everything a peer chose
//     that the format leaves open - segmentation, width codes, direction - is
//     retained rather than recomputed.

//------------------------------------------------------------------------------
// IdList
//------------------------------------------------------------------------------

// width code -> width in bytes: 0 -> 1, 1 -> 2, 2 -> 4, 3 -> 8
#define EFFECTS_V3_WIDTH_BYTES(code) (1 << (code))

// segment header
//
//   bits 0-1  kind: 0 Range, 1 Set (roaring64), 2 Repeat; 3 unused
//   bits 2-3  value width code - Range base, Repeat id
//   bits 4-5  count width code - Range len,  Repeat count
//   bit  6    descending
//   bit  7    reserved - a segment setting it is refused
#define EFFECTS_V3_SEG_KIND_MASK    0x03
#define EFFECTS_V3_SEG_VWIDTH_SHIFT 2
#define EFFECTS_V3_SEG_CWIDTH_SHIFT 4
#define EFFECTS_V3_SEG_DESCENDING   0x40
#define EFFECTS_V3_SEG_RESERVED     0x80

// a segment's kind and direction, as one closed set
//
// Direction is a separate bit on the wire and is folded in here so the
// combination that has no meaning - a descending Repeat - cannot be
// constructed. A Repeat holds one id 'count' times; read either way it is the
// same sequence, so a peer setting bit 6 on one means something this build does
// not know, and the decoder refuses it.
//
// The spec names wire kind 1 `Ascending`. It is a set, and bit 6 says which way
// it is read, so it is named SET here and the direction is in the kind.
typedef enum {
	EFFECTS_V3_SEG_RANGE_ASCENDING,
	EFFECTS_V3_SEG_RANGE_DESCENDING,
	EFFECTS_V3_SEG_SET_ASCENDING,
	EFFECTS_V3_SEG_SET_DESCENDING,
	EFFECTS_V3_SEG_REPEAT,
} EffectsV3IdListSegmentKind;

// one segment of an IdList
//
// One arm per kind. The width codes live in the arms that have them: a Set
// segment has no widths on the wire at all, so it does not carry them here.
//
// Widths are retained rather than recomputed on re-encode. A segment built
// fresh rather than decoded sets them to the narrowest width holding its value.
typedef struct {
	EffectsV3IdListSegmentKind kind;

	union {
		// base is the LOWEST id: base, base+1, ... base+(len-1)
		struct {
			uint8_t  value_width;  // header bits 2-3, holds base
			uint8_t  count_width;  // header bits 4-5, holds len
			uint64_t base;
			uint64_t len;
		} range_ascending;

		// base is the HIGHEST id: base, base-1, ... base-(len-1)
		struct {
			uint8_t  value_width;
			uint8_t  count_width;
			uint64_t base;
			uint64_t len;
		} range_descending;

		// a roaring64 set, read low to high
		struct {
			unsigned char *blob;         // serialized roaring64, owned
			uint32_t       n;            // blob length, as the u32 on the wire
			uint64_t       cardinality;  // ids described; checked at decode
		} set_ascending;

		// the same blob, read high to low. The two directions over one id set
		// differ in bit 6 and nowhere else.
		struct {
			unsigned char *blob;         // owned
			uint32_t       n;
			uint64_t       cardinality;
		} set_descending;

		// one id, 'count' times. No direction.
		struct {
			uint8_t  value_width;  // holds id
			uint8_t  count_width;  // holds count
			uint64_t id;
			uint64_t count;
		} repeat;
	};
} EffectsV3IdListSegment;

// an ordered, duplicate-preserving list of entity ids, held as segments
typedef struct {
	EffectsV3IdListSegment *segments;  // owned, 'n' entries
	uint32_t                n;         // segment count, as the u32 on the wire
} EffectsV3IdList;

//------------------------------------------------------------------------------
// record payloads
//------------------------------------------------------------------------------
//
// One struct per opcode. A record's opcode selects the arm, and an arm carries
// exactly the fields that opcode has on the wire - no more, and none that are
// meaningful only sometimes.
//
// Every batchable record is:
//
//     u32 opcode
//     u32 count
//     <shape>      LabelSet (node-shaped) | RelType (edge-shaped)
//     <attr ids>   only on records that carry values
//     IdList       the entity ids, positionally bound to the rows
//     IdList x2    src then dst, CREATE_EDGE and DELETE_EDGE only
//     <values>     count * n_attrs SIValues, row-major
//
// DELETE_NODE carries a LabelSet: a replica deleting a node needs its labels to
// maintain the label matrices and the label-scoped indexes.

// 1 - UPDATE_NODE: set or remove properties on nodes that already exist
//
// 'values' is row-major: row k is values[k * n_attrs ..] and belongs to the
// k-th id in 'ids' as written, column j to attr_ids[j].
//
// T_NULL in a slot means REMOVE THIS ATTRIBUTE - it is not padding, and
// filtering it out turns every property removal into a no-op.
typedef struct {
	uint32_t     count;
	LabelID     *labels;    // owned
	uint16_t     n_labels;  // as the u16 on the wire - n, NOT count
	AttributeID *attr_ids;  // owned, ascending
	uint16_t     n_attrs;
	EffectsV3IdList ids;
	SIValue     *values;    // owned; each freed with SIValue_Free
	uint64_t     n_values;  // count * n_attrs
} EffectsV3UpdateNode;

// 2 - UPDATE_EDGE: set or remove properties on edges that already exist
typedef struct {
	uint32_t     count;
	RelationID   relation_id;
	AttributeID *attr_ids;  // owned, ascending
	uint16_t     n_attrs;
	EffectsV3IdList ids;
	SIValue     *values;    // owned
	uint64_t     n_values;  // count * n_attrs
} EffectsV3UpdateEdge;

// 3 - CREATE_NODE: 'labels' are the labels the new nodes are created with
typedef struct {
	uint32_t     count;
	LabelID     *labels;    // owned, ascending
	uint16_t     n_labels;
	AttributeID *attr_ids;  // owned, ascending
	uint16_t     n_attrs;
	EffectsV3IdList ids;
	SIValue     *values;    // owned
	uint64_t     n_values;  // count * n_attrs
} EffectsV3CreateNode;

// 4 - CREATE_EDGE: 'src' and 'dst' are positionally bound to 'ids'
typedef struct {
	uint32_t     count;
	RelationID   relation_id;
	AttributeID *attr_ids;  // owned, ascending
	uint16_t     n_attrs;
	EffectsV3IdList ids;
	EffectsV3IdList src;
	EffectsV3IdList dst;
	SIValue     *values;    // owned
	uint64_t     n_values;  // count * n_attrs
} EffectsV3CreateEdge;

// 5 - DELETE_NODE
//
// 'labels' are the labels the deleted nodes CARRIED. A replica needs them to
// maintain the label matrices and the label-scoped indexes, which is why a
// delete states them at all.
typedef struct {
	uint32_t count;
	LabelID *labels;    // owned, ascending
	uint16_t n_labels;
	EffectsV3IdList ids;
} EffectsV3DeleteNode;

// 6 - DELETE_EDGE: endpoints are stated because the edge is gone by apply time
typedef struct {
	uint32_t   count;
	RelationID relation_id;
	EffectsV3IdList ids;
	EffectsV3IdList src;
	EffectsV3IdList dst;
} EffectsV3DeleteEdge;

// 7 - SET_LABELS: 'labels' are the labels being ADDED to each id
typedef struct {
	uint32_t count;
	LabelID *labels;    // owned, ascending
	uint16_t n_labels;
	EffectsV3IdList ids;
} EffectsV3SetLabels;

// 8 - REMOVE_LABELS: 'labels' are the labels being REMOVED from each id
typedef struct {
	uint32_t count;
	LabelID *labels;    // owned, ascending
	uint16_t n_labels;
	EffectsV3IdList ids;
} EffectsV3RemoveLabels;

// 9 - ADD_SCHEMA: inherently singular - no count, no ids
typedef struct {
	SchemaType schema_type;
	int        schema_id;  // LabelID or RelationID
	char      *name;       // owned, NUL terminated
} EffectsV3AddSchema;

// 10 - ADD_ATTRIBUTE: inherently singular
typedef struct {
	AttributeID attr_id;
	char       *name;  // owned, NUL terminated
} EffectsV3AddAttribute;

// an attribute named by id and by name together, for DDL records
typedef struct {
	AttributeID  id;
	char        *name;  // owned, NUL terminated
} EffectsV3AttrRef;

// index options, as a typed block
//
// Each option carries a presence byte, so an option that was not stated is
// distinguishable from one stated at its default. 'dimension' has no presence
// byte of its own: a vector field always has one.
typedef struct {
	bool      has_language;
	char     *language;             // owned
	bool      has_stopwords;
	char    **stopwords;            // owned, and each entry owned
	uint64_t  n_stopwords;
	bool      has_weight;
	double    weight;
	bool      has_nostem;
	bool      nostem;
	bool      has_phonetic;
	char     *phonetic;             // owned
	bool      is_vector;
	uint64_t  dimension;
	bool      has_m;                uint64_t m;
	bool      has_ef_construction;  uint64_t ef_construction;
	bool      has_ef_runtime;       uint64_t ef_runtime;
	bool      has_sim_func;         uint64_t sim_func;
} EffectsV3IndexOptions;

// 11 - CREATE_INDEX
//
// 'field_type' is an IndexFieldType and is a BIT FLAG SET rather than a
// discriminant: one field can be several index kinds at once.
typedef struct {
	SchemaType schema_type;
	int        schema_id;
	char      *name;              // owned - the schema's name, stated as well
	                              // as its id so a diverged id is caught
	uint32_t   field_type;
	EffectsV3AttrRef *attrs;      // owned
	uint16_t          n_attrs;
	bool                  has_options;
	EffectsV3IndexOptions options;
} EffectsV3CreateIndex;

// 12 - DROP_INDEX: names the field to drop; carries no options
typedef struct {
	SchemaType schema_type;
	int        schema_id;
	char      *name;              // owned
	uint32_t   field_type;
	EffectsV3AttrRef *attrs;      // owned
	uint16_t          n_attrs;
} EffectsV3DropIndex;

// 13 - CREATE_CONSTRAINT
//
// 'entity_type' is a GraphEntityType and is 1-BASED on this wire.
//
// 'status' is a ConstraintStatus. A replica cannot tell an enforcing constraint
// from one still being built, so the master states which.
typedef struct {
	uint32_t constraint_type;
	uint32_t entity_type;
	int      schema_id;
	char    *name;                // owned
	EffectsV3AttrRef *attrs;      // owned
	uint16_t          n_attrs;
	bool     has_status;
	uint32_t status;
} EffectsV3CreateConstraint;

// 14 - DROP_CONSTRAINT: no status - a constraint being dropped has no state
typedef struct {
	uint32_t constraint_type;
	uint32_t entity_type;
	int      schema_id;
	char    *name;                // owned
	EffectsV3AttrRef *attrs;      // owned
	uint16_t          n_attrs;
} EffectsV3DropConstraint;

// a single decoded record
//
// 'opcode' selects the arm. Reading any other arm is undefined.
typedef struct {
	EffectType opcode;
	union {
		EffectsV3UpdateNode       update_node;        //  1
		EffectsV3UpdateEdge       update_edge;        //  2
		EffectsV3CreateNode       create_node;        //  3
		EffectsV3CreateEdge       create_edge;        //  4
		EffectsV3DeleteNode       delete_node;        //  5
		EffectsV3DeleteEdge       delete_edge;        //  6
		EffectsV3SetLabels        set_labels;         //  7
		EffectsV3RemoveLabels     remove_labels;      //  8
		EffectsV3AddSchema        add_schema;         //  9
		EffectsV3AddAttribute     add_attribute;      // 10
		EffectsV3CreateIndex      create_index;       // 11
		EffectsV3DropIndex        drop_index;         // 12
		EffectsV3CreateConstraint create_constraint;  // 13
		EffectsV3DropConstraint   drop_constraint;    // 14
	};
} EffectsV3Record;

// a decoded payload: the header, then the records in apply order
struct EffectsV3Records {
	uint8_t version;  // the version byte the payload declared
	uint8_t flags;    // the flags byte; bit 0 = compressed

	EffectsV3Record *records;  // owned, 'n' entries, in apply order
	uint32_t         n;
};

typedef struct EffectsV3Records EffectsV3Records;

//------------------------------------------------------------------------------
// refusal
//------------------------------------------------------------------------------

// why a buffer was refused
//
// UNIMPLEMENTED is distinct from MALFORMED. A well-formed record this build
// does not implement is not a corrupt one, and reporting it as corrupt sends an
// operator hunting a wire problem that does not exist. It is still a refusal:
// the buffer is rejected and the caller treats it as divergence, because
// silently skipping a record it cannot apply is data loss.
typedef enum {
	EFFECTS_V3_OK = 0,              // decoded cleanly
	EFFECTS_V3_TRUNCATED,           // ran out of bytes mid-field
	EFFECTS_V3_MALFORMED,           // well-sized but invalid: reserved bits
	                                // set, unknown opcode, bad width code, a
	                                // count that outruns the payload, a
	                                // cardinality that disagrees with its record
	EFFECTS_V3_UNSUPPORTED_VERSION, // version byte above what this build reads
	EFFECTS_V3_UNSUPPORTED_FLAGS,   // a flag bit outside the mask we understand
	EFFECTS_V3_UNIMPLEMENTED,       // a well-formed record not implemented here
} EffectsV3Status;

//------------------------------------------------------------------------------
// the seam
//------------------------------------------------------------------------------

// decode a v3 payload into records
//
// takes no GraphContext: a pure byte-to-value transformation, which is what
// makes it testable against a fixture corpus and cheap to fuzz
//
// on EFFECTS_V3_OK the caller owns '*records' and frees it with
// EffectsV3_RecordsFree; on anything else '*records' is NULL and nothing is
// left allocated
EffectsV3Status EffectsV3_Decode
(
	const char *buff,           // encoded payload, including the version byte
	size_t n,                   // size of buff
	EffectsV3Records **records  // [output] decoded records
);

// encode records back into a payload
//
// re-encoding what EffectsV3_Decode produced reproduces the input bytes exactly
//
// returns false only on an internal failure; records that decoded cleanly
// always re-encode
bool EffectsV3_Encode
(
	const EffectsV3Records *records,  // records to encode
	EffectsBuffer *eb                 // buffer to write into
);

// apply decoded records to a graph
//
// where ids are expanded and where divergence is detected, so where a
// GraphContext is required
//
// returns false if the records reference graph state that does not exist
// locally; the caller treats that as divergence
bool EffectsV3_Apply
(
	GraphContext *gc,                 // graph to operate on
	const EffectsV3Records *records   // records to apply
);

// free records returned by EffectsV3_Decode
void EffectsV3_RecordsFree
(
	EffectsV3Records *records
);

//------------------------------------------------------------------------------
// readiness - what is actually linkable
//------------------------------------------------------------------------------
//
// The entry points above are declared before they are defined, so a test that
// links one that does not exist yet breaks the build for everyone. These flags
// say what is safe to call, and each is defined by the commit that makes it
// true - never here.
//
// Two flags rather than one: a single EFFECTS_V3_CODEC_READY could only be
// defined honestly by whichever direction landed second, and defining it on
// decode alone would link a round-trip test against an undefined
// EffectsV3_Encode, which is the failure the flag exists to prevent.
//
// A truncation corpus needs decode alone, so it can run a PR earlier than a
// round trip.

// EffectsV3_Decode and EffectsV3_RecordsFree are defined
#define EFFECTS_V3_DECODE_READY 1

// EffectsV3_ENCODE_READY is defined by the writer when EffectsV3_Encode lands

#if defined(EFFECTS_V3_DECODE_READY) && defined(EFFECTS_V3_ENCODE_READY)
// both directions are linkable, so a round trip can be built
#define EFFECTS_V3_CODEC_READY 1
#endif
