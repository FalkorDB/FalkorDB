/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

// The record encoder, checked against the conformance fixtures.
//
// The segment tests compared a slice; these compare WHOLE PAYLOADS, version byte
// and flags byte included, because a record fixture is a complete GRAPH.EFFECT
// payload. That closes the one gap the slice comparison left: a field written in
// the wrong place relative to the record framing.
//
// Records are built by hand rather than through the grouping layer, to isolate
// the ENCODING from the GROUPING - so a failure here is a wire-format
// disagreement and not a decision about which entities share a record.
//
// The fixtures are not ground truth - they are generated from the Rust encoder.
// Ground truth is C's own source for the reused primitives and
// docs/effects-v3.md for the v3 blocks. A disagreement is settled against those
// and the fixture regenerated, never by editing an expectation here.

#include "src/effects/effects.h"
#include "src/effects/effects_internal.h"
#include "src/effects/effects_v3_encode.h"
#include "src/effects/effects_v3_id_list.h"
#include "src/effects/effects_bytes.h"
#include "src/util/rmalloc.h"
#include "effects_v3_hex.h"

void setup() {
	Alloc_Reset();
}

#define TEST_INIT setup();
#include "acutest.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// build an IdList from a literal id sequence
static EffectsV3IdList _ids(const uint64_t *ids, size_t n,
		EffectsV3IdListBuilder **keep) {
	EffectsV3IdListBuilder *b = EffectsV3IdListBuilder_New();
	for(size_t i = 0; i < n; i++) EffectsV3IdListBuilder_Push(b, ids[i]);
	*keep = b;
	return EffectsV3IdListBuilder_ToIdList(b);
}

// encode a record as a whole payload and compare against a fixture
//
// the version and flags bytes are written here rather than by the encoder,
// because a record does not own the payload header - the buffer does, and at
// emit version 2 it would write the wrong one until the version flag lands
static unsigned char *_encode(const EffectsV3Record *r, size_t *n) {
	EffectsBytes *out = EffectsBytes_New(1024);

	unsigned char header[2] = { 3, 0 };
	EffectsBytes_Write(out, header, 2);
	EffectsV3_EncodeRecord(r, out);

	*n = EffectsBytes_Len(out);
	unsigned char *got = malloc(*n);
	EffectsBytes_CopyInto(out, got);
	EffectsBytes_Free(out);

	return got;
}

static void _check(const EffectsV3Record *r, const char *hex,
		const char *name) {
	size_t got_n = 0;
	unsigned char *got = _encode(r, &got_n);

	size_t want_n = 0;
	unsigned char *want = EffectsV3Test_Unhex(hex, &want_n);

	char *got_s = EffectsV3Test_Hex(got, got_n), *want_s = EffectsV3Test_Hex(want, want_n);
	TEST_ASSERT_(got_n == want_n && memcmp(got, want, want_n) == 0,
			"%s: the corpus and this encoder disagree\n"
			"      got %s\n  expected %s", name, got_s, want_s);

	free(got_s); free(want_s); free(want); free(got);
}

// the six values every value-carrying fixture uses, row-major:
// row0 (int 1, "a"), row1 (null, true), row2 (0.25, -9)
static SIValue *_rows(void) {
	SIValue *v = malloc(sizeof(SIValue) * 6);
	v[0] = SI_LongVal(1);
	v[1] = SI_ConstStringVal("a");
	v[2] = SI_NullVal();
	v[3] = SI_BoolVal(true);
	v[4] = SI_DoubleVal(0.25);
	v[5] = SI_LongVal(-9);
	return v;
}

//------------------------------------------------------------------------------
// the entity records, as one table
//------------------------------------------------------------------------------
//
// SEVEN OF THE EIGHT ENTITY RECORDS, one row each. These differ only in which
// columns they
// carry and what is in them, and each used to spell out the same IdList
// lifecycle: build a column, keep its builder, free the list and the builder
// separately. That was five copies of the lifecycle - and two of them freed
// through the WRONG arm of the record union, r.create_node.ids on an
// UPDATE_NODE record and r.set_labels.ids on a REMOVE_LABELS one, which worked
// only because those arms happen to place the list at the same offset.
//
// The DDL and singular records below are deliberately NOT in this table: they
// carry no id columns, so they have no lifecycle to share, and a row for them
// would be mostly empty.

// the id columns the fixtures draw on
static const uint64_t IDS_1_2_3[]  = { 1, 2, 3 };
static const uint64_t SRC_10_12[]  = { 10, 11, 12 };
static const uint64_t DST_77_77[]  = { 77, 77, 77 };

// the label and attribute sets they draw on
static const LabelID     LBL_0_1_2[] = { 0, 1, 2 };
static const LabelID     LBL_4_6[]   = { 4, 6 };
static const LabelID     LBL_4[]     = { 4 };
static const LabelID     LBL_0_3[]   = { 0, 3 };
static const LabelID     LBL_1[]     = { 1 };
static const AttributeID ATTR_7_9[]  = { 7, 9 };

// one entity-record fixture
typedef struct {
	const char *name;  // the corpus fixture this row is
	const char *hex;   // its bytes

	EffectType opcode;
	uint32_t   count;

	const LabelID *labels;       // node-shaped records
	uint16_t       n_labels;
	RelationID     relation_id;  // edge-shaped records

	const AttributeID *attr_ids;
	uint16_t           n_attrs;

	const uint64_t *ids;  // the entity column; every row has one
	const uint64_t *src;  // endpoint columns, NULL where the record carries
	const uint64_t *dst;  // none - which is every row but the delete
	size_t          n;    // the length of each column above

	bool values;  // carries the six shared rows from _rows()
} EntityCase;

static const EntityCase ENTITY_CASES[] = {
	// DELETE_NODE carries a LabelSet, which the spec's record table once denied.
	// The labels are the ones the node actually held, captured as it was
	// deleted: a replica needs them to clear the right label-scoped index
	// documents, and the query's pattern cannot supply them - MATCH (n:A)
	// DELETE n over an (:A:B) node must clear :B's indexes too
	{
		.name = "rec_delete_node",
		.hex  = "03000500000003000000030000000000010000000200000001000000000103",
		.opcode = EFFECT_DELETE_NODE, .count = 3,
		.labels = LBL_0_1_2, .n_labels = 3,
		.ids = IDS_1_2_3, .n = 3,
	},

	// SET_LABELS and REMOVE_LABELS: the labels, then all their nodes - grouped
	// by label rather than interleaved as (node, label) pairs, which is what
	// lets the node ids be one contiguous Range
	{
		.name = "rec_set_labels",
		.hex  = "030007000000030000000200040000000600000001000000000103",
		.opcode = EFFECT_SET_LABELS, .count = 3,
		.labels = LBL_4_6, .n_labels = 2,
		.ids = IDS_1_2_3, .n = 3,
	},
	{
		.name = "rec_remove_labels",
		.hex  = "0300080000000300000001000400000001000000000103",
		.opcode = EFFECT_REMOVE_LABELS, .count = 3,
		.labels = LBL_4, .n_labels = 1,
		.ids = IDS_1_2_3, .n = 3,
	},

	// DELETE_EDGE: the type, then ids, then endpoints as their own lists. The
	// dst column is one Repeat - three edges into the same node - which is the
	// shape endpoint columns take and the reason Repeat exists
	{
		.name = "rec_delete_edge",
		.hex  = "0300060000000300000005000000010000000001030100000000"
		"0a0301000000024d03",
		.opcode = EFFECT_DELETE_EDGE, .count = 3, .relation_id = 5,
		.ids = IDS_1_2_3, .src = SRC_10_12, .dst = DST_77_77, .n = 3,
	},

	// CREATE_NODE and UPDATE_NODE share a layout: LabelSet, AttrIds, IdList,
	// values. The AttrSet is SPLIT across the IdList - ids with the shape,
	// values after the rows - which is the part the spec's record table got
	// wrong and the fixtures settled
	{
		.name = "rec_create_node",
		.hex  = "0300030000000300000002000000000003000000020007000900010000"
		"000001030020000001000000000000000008000002000000000000006100"
		"0080000000100000010040000000000000 0000d03f00200000f7ffffffff"
		"ffffff",
		.opcode = EFFECT_CREATE_NODE, .count = 3,
		.labels = LBL_0_3, .n_labels = 2,
		.attr_ids = ATTR_7_9, .n_attrs = 2,
		.ids = IDS_1_2_3, .n = 3, .values = true,
	},
	{
		.name = "rec_update_node",
		.hex  = "0300010000000300000001000100000002000700090001000000000103"
		"00200000010000000000000000080000020000000000000061000080000"
		"00010000001004000000000000000 00d03f00200000f7ffffffffffffff",
		.opcode = EFFECT_UPDATE_NODE, .count = 3,
		.labels = LBL_1, .n_labels = 1,
		.attr_ids = ATTR_7_9, .n_attrs = 2,
		.ids = IDS_1_2_3, .n = 3, .values = true,
	},

	// UPDATE_EDGE carries its type and NOT its endpoints: the replica indexes
	// under what the record states rather than re-deriving it, and endpoints are
	// recoverable for an update, which is why only create and delete carry them
	{
		.name = "rec_update_edge",
		.hex  = "030002000000030000000200000002000700090001000000000103002000"
		"000100000000000000000800000200000000000000610000800000001000"
		"0001004000000000000 00000d03f00200000f7ffffffffffffff",
		.opcode = EFFECT_UPDATE_EDGE, .count = 3, .relation_id = 2,
		.attr_ids = ATTR_7_9, .n_attrs = 2,
		.ids = IDS_1_2_3, .n = 3, .values = true,
	},
};

#define N_ENTITY_CASES ((int)(sizeof(ENTITY_CASES) / sizeof(ENTITY_CASES[0])))

// fill the arm the opcode names
//
// A SWITCH AND NOT ONE SHARED ASSIGNMENT. delete_node, set_labels and
// remove_labels have identical layouts, so a single assignment would encode all
// three correctly today and stop the moment a field moves in one of them. The
// casts drop const from the table's arrays: a record's fields are the
// non-const ones it would own if it had been decoded rather than built here.
static EffectsV3Record _entity_record
(
	const EntityCase *c,    // the row
	EffectsV3IdList ids,    // its entity column
	EffectsV3IdList src,    // its endpoint columns, empty where it has none
	EffectsV3IdList dst,
	SIValue *rows           // the shared values, or NULL
) {
	LabelID     *labels  = (LabelID *)c->labels;
	AttributeID *attrs   = (AttributeID *)c->attr_ids;
	const uint64_t n_val = c->values ? 6 : 0;

	EffectsV3Record r = { .opcode = c->opcode };

	switch(c->opcode) {
		case EFFECT_DELETE_NODE:
			r.delete_node.count    = c->count;
			r.delete_node.labels   = labels;
			r.delete_node.n_labels = c->n_labels;
			r.delete_node.ids      = ids;
			break;

		case EFFECT_SET_LABELS:
			r.set_labels.count    = c->count;
			r.set_labels.labels   = labels;
			r.set_labels.n_labels = c->n_labels;
			r.set_labels.ids      = ids;
			break;

		case EFFECT_REMOVE_LABELS:
			r.remove_labels.count    = c->count;
			r.remove_labels.labels   = labels;
			r.remove_labels.n_labels = c->n_labels;
			r.remove_labels.ids      = ids;
			break;

		case EFFECT_CREATE_NODE:
			r.create_node.count    = c->count;
			r.create_node.labels   = labels;
			r.create_node.n_labels = c->n_labels;
			r.create_node.attr_ids = attrs;
			r.create_node.n_attrs  = c->n_attrs;
			r.create_node.ids      = ids;
			r.create_node.values   = rows;
			r.create_node.n_values = n_val;
			break;

		case EFFECT_UPDATE_NODE:
			r.update_node.count    = c->count;
			r.update_node.labels   = labels;
			r.update_node.n_labels = c->n_labels;
			r.update_node.attr_ids = attrs;
			r.update_node.n_attrs  = c->n_attrs;
			r.update_node.ids      = ids;
			r.update_node.values   = rows;
			r.update_node.n_values = n_val;
			break;

		case EFFECT_DELETE_EDGE:
			r.delete_edge.count       = c->count;
			r.delete_edge.relation_id = c->relation_id;
			r.delete_edge.ids         = ids;
			r.delete_edge.src         = src;
			r.delete_edge.dst         = dst;
			break;

		case EFFECT_UPDATE_EDGE:
			r.update_edge.count       = c->count;
			r.update_edge.relation_id = c->relation_id;
			r.update_edge.attr_ids    = attrs;
			r.update_edge.n_attrs     = c->n_attrs;
			r.update_edge.ids         = ids;
			r.update_edge.values      = rows;
			r.update_edge.n_values    = n_val;
			break;

		default:
			TEST_ASSERT_(false, "%s: opcode %d is not an entity record",
					c->name, (int)c->opcode);
			break;
	}

	return r;
}

// the seven entity records that have a row here, against their fixtures
//
// CREATE_EDGE is the eighth and has none, here or in the five tests this
// replaced. It is the only entity record carrying both endpoint lists AND
// values - the arm with the most to get wrong - and the corpus has a fixture
// for it (rec_create_edge.json), which lives on the conformance-corpus branch
// rather than in this tree. Adding it is one row
void test_effectsV3Record_entityRecords(void) {
	for(int i = 0; i < N_ENTITY_CASES; i++) {
		const EntityCase *c = ENTITY_CASES + i;
		TEST_CASE(c->name);

		EffectsV3IdListBuilder *ki = NULL, *ks = NULL, *kd = NULL;
		EffectsV3IdList ids = _ids(c->ids, c->n, &ki);
		EffectsV3IdList src = { 0 };
		EffectsV3IdList dst = { 0 };

		if(c->src != NULL) {
			src = _ids(c->src, c->n, &ks);
			dst = _ids(c->dst, c->n, &kd);
		}

		SIValue *rows = c->values ? _rows() : NULL;

		EffectsV3Record r = _entity_record(c, ids, src, dst, rows);
		_check(&r, c->hex, c->name);

		EffectsV3IdListBuilder_FreeIdList(&ids);
		EffectsV3IdListBuilder_Free(ki);
		if(c->src != NULL) {
			EffectsV3IdListBuilder_FreeIdList(&src);
			EffectsV3IdListBuilder_FreeIdList(&dst);
			EffectsV3IdListBuilder_Free(ks);
			EffectsV3IdListBuilder_Free(kd);
		}
		free(rows);
	}
}

//------------------------------------------------------------------------------
// the two singular records
//------------------------------------------------------------------------------

// ADD_SCHEMA and ADD_ATTRIBUTE carry no count: they are inherently one
//
// and their id widths DIFFER - a schema id is four bytes, an attribute id two.
// The record listing does not spell that out inline, so an encoder written
// from the record beside it would use the wrong width for one of them
void test_effectsV3Record_singularRecords(void) {
	{
		EffectsV3Record r = { .opcode = EFFECT_ADD_SCHEMA,
			.add_schema.schema_type = SCHEMA_NODE, .add_schema.schema_id = 3, .add_schema.name = "Person" };
		_check(&r, "03000900000000000000030000000700000000000000506572736f6e00",
				"rec_add_schema_node");
	}
	{
		EffectsV3Record r = { .opcode = EFFECT_ADD_SCHEMA,
			.add_schema.schema_type = SCHEMA_EDGE, .add_schema.schema_id = 1, .add_schema.name = "KNOWS" };
		_check(&r, "030009000000010000000100000006000000000000004b4e4f575300",
				"rec_add_schema_edge");
	}
	{
		EffectsV3Record r = { .opcode = EFFECT_ADD_ATTRIBUTE,
			.add_attribute.attr_id = 12, .add_attribute.name = "name" };
		_check(&r, "03000a0000000c0005000000000000006e616d6500",
				"rec_add_attribute");
	}
}

//------------------------------------------------------------------------------
// records 11-14: index and constraint DDL
//------------------------------------------------------------------------------

// CREATE_INDEX and DROP_INDEX, and the difference between them that the spec
// sentence does not settle
//
// "Mirrors 11 without the options" reads two ways. The corpus decides it:
// rec_drop_index ends after its field list, so a drop carries NO OPTIONS BLOCK
// AT ALL - zero bytes, not an empty one. A drop that wrote five clear presence
// bytes would be five bytes longer than the fixture.
void test_effectsV3Record_indexDDL(void) {
	{
		// two fields, options {} - the text half is still written, five clear
		// presence bytes, because it is written whatever the field type
		EffectsV3AttrRef refs[] = { { 12, "name" }, { 13, "age" } };
		EffectsV3SchemaRef schemas[] = { { 3, "Person" } };
		EffectsV3Record r = {
			.opcode      = EFFECT_CREATE_INDEX,
			.create_index.schema_type = SCHEMA_NODE,
			.create_index.schemas   = schemas,
			.create_index.n_schemas = 1,
			.create_index.field_type = 0x000a,
			.create_index.attrs = refs,
			.create_index.n_attrs = 2,
		};
		_check(&r, "03000b000000000000000100030000000700000000000000506572736f6e000a"
		           "00000002000c0005000000000000006e616d65000d0004000000000000006167"
		           "65000000000000", "rec_create_index");
	}
	{
		// a drop: same shape, and the options block absent entirely
		EffectsV3AttrRef refs[] = { { 14, "vec" } };
		EffectsV3SchemaRef schemas[] = { { 1, "KNOWS" } };
		EffectsV3Record r = {
			.opcode      = EFFECT_DROP_INDEX,
			.drop_index.schema_type = SCHEMA_EDGE,
			.drop_index.schemas   = schemas,
			.drop_index.n_schemas = 1,
			.drop_index.field_type = 0x0010,
			.drop_index.attrs = refs,
			.drop_index.n_attrs = 1,
		};
		_check(&r, "03000c0000000100000001000100000006000000000000004b4e4f5753001000"
		           "000001000e00040000000000000076656300", "rec_drop_index");
	}
}

// THE ENCODER SORTS THE SCHEMA LIST, and only a hand-built test can say so
//
// ON ()-[r:ROAD|:R]-() and ON ()-[r:R|:ROAD]-() are one statement, so the bytes
// have to be the same. A generated fixture cannot arbitrate that: it is already
// in its generator's order, so a conformance run is green whichever rule holds.
// The two orders are therefore encoded here and compared against each other
// rather than against a corpus file.
void test_effectsV3Record_indexSchemasAreSorted(void) {
	EffectsV3AttrRef refs[] = { { 7, "w" } };

	EffectsV3SchemaRef ascending[]  = { { 2, "R" }, { 9, "ROAD" } };
	EffectsV3SchemaRef descending[] = { { 9, "ROAD" }, { 2, "R" } };

	EffectsV3Record r = {
		.opcode                   = EFFECT_CREATE_INDEX,
		.create_index.schema_type = SCHEMA_EDGE,
		.create_index.schemas     = ascending,
		.create_index.n_schemas   = 2,
		.create_index.field_type  = INDEX_FLD_CCH,
		.create_index.attrs       = refs,
		.create_index.n_attrs     = 1,
	};

	size_t a_n = 0;
	unsigned char *a = _encode(&r, &a_n);

	r.create_index.schemas = descending;

	size_t d_n = 0;
	unsigned char *d = _encode(&r, &d_n);

	TEST_ASSERT_(a_n == d_n && memcmp(a, d, a_n) == 0,
			"the same two schemas in two orders encoded differently");

	// AND THE ORDER IS ASCENDING, not merely stable. Comparing the two
	// encodings against each other would pass an encoder that reversed both.
	//
	// The list opens after 0300 . opcode u32 . schema_type u32 = 10 bytes,
	// with its u16 count.
	TEST_ASSERT(a[10] == 2 && a[11] == 0);
	TEST_ASSERT(a[12] == 2);

	free(a);
	free(d);
}

// THE VECTOR OPTIONS BLOCK, the only fixture exercising options non-empty
//
// Five values behind FOUR presence bytes, because `dimension` has none - a
// vector field must have one, so there is nothing for a presence byte to say.
// That asymmetry is what made one fixture insufficient: four markers for five
// values admits several readings that all total 49 bytes.
//
// The text half is written first regardless of field type, so this block opens
// with five clear bytes before the vector half begins.
void test_effectsV3Record_indexVectorOptions(void) {
	EffectsV3AttrRef refs[] = { { 0, "embedding" } };
	EffectsV3SchemaRef schemas[] = { { 0, "P" } };
	EffectsV3Record r = {
		.opcode      = EFFECT_CREATE_INDEX,
		.create_index.schema_type = SCHEMA_NODE,
		.create_index.schemas   = schemas,
		.create_index.n_schemas = 1,
		.create_index.field_type = 0x0010,
		.create_index.attrs = refs,
		.create_index.n_attrs = 1,
		// the vector half is gated on field_type above, not on a flag here
		.create_index.options = {
			.dimension = 128,
			.has_m = true, .m               = 32,
			.has_ef_construction = true, .ef_construction = 400,
			.has_ef_runtime = true, .ef_runtime      = 20,
			.has_sim_func = true, .sim_func        = 2,
		},
	};

	_check(&r, "03000b0000000000000001000000000002000000000000005000100000000100"
	           "00000a00000000000000656d62656464696e6700000000000080000000000000"
	           "0001200000000000000001900100000000000001140000000000000001020000"
	           "0000000000", "rec_create_index_vector");
}

// CREATE_CONSTRAINT and DROP_CONSTRAINT
//
// Two details a constraint encoder gets wrong by reading the index record beside
// it: the property count is a u8 where an index field count is a u16, and
// GraphEntityType is 1-BASED because GETYPE_UNKNOWN takes 0 - so a node is 1.
//
// And CREATE carries a ConstraintStatus that DROP does not - the one place v3
// states more than C: a replica never validates, so the announcement is the only
// thing that can distinguish an enforcing constraint from one still building.
void test_effectsV3Record_constraintDDL(void) {
	{
		EffectsV3AttrRef props[] = { { 12, "name" } };
		EffectsV3Record r = {
			.opcode          = EFFECT_CREATE_CONSTRAINT,
			.create_constraint.constraint_type = 0,   // unique
			.create_constraint.entity_type     = 1,   // node, 1-based
			.create_constraint.status          = 0,   // operational
			.create_constraint.schema_id       = 3,
			.create_constraint.name            = "Person",
			.create_constraint.attrs       = props,
			.create_constraint.n_attrs     = 1,
		};
		_check(&r, "03000d0000000000000001000000000000000300000007000000000000"
		           "00506572736f6e00010c0005000000000000006e616d6500",
				"rec_create_constraint");
	}
	{
		EffectsV3AttrRef props[] = { { 12, "name" }, { 13, "age" } };
		EffectsV3Record r = {
			.opcode          = EFFECT_DROP_CONSTRAINT,
			.drop_constraint.constraint_type = 1,   // mandatory
			.drop_constraint.entity_type     = 2,   // edge
			.drop_constraint.schema_id       = 1,
			.drop_constraint.name            = "KNOWS",
			.drop_constraint.attrs       = props,
			.drop_constraint.n_attrs     = 2,
		};
		_check(&r, "03000e00000001000000020000000100000006000000000000004b4e4f"
		           "575300020c0005000000000000006e616d65000d000400000000000000"
		           "61676500", "rec_drop_constraint");
	}
}

// a buffer taken over at a version other than the one it was built on must
// carry its sink across the arms
//
// EffectsBuffer_New picks the arm from Config_EFFECTS_VERSION, so a unit test -
// where that config is a zeroed static - gets the v2 arm, and EffectsV3_Encode
// then asks for the body at version 3. Moving the discriminant without moving
// the sink put eb->v3.flags on byte 0 of eb->v2.bytes and the next read of
// eb->v3.body on the remains of that pointer. A union arm mismatch is valid C,
// so the symptom was a segfault rather than a refusal.
//
// Not test-only: the same happens for any buffer on the v2 arm, which is what
// the DEFAULT EFFECTS_VERSION produces.
static void test_effectsV3Record_takeBodyAcrossArms(void) {
	EffectsBuffer *eb = EffectsBuffer_New();

	EffectsBytes *body = EffectsBuffer_TakeBody(eb, 3, 0);
	TEST_ASSERT(body != NULL);

	const unsigned char payload[] = { 0xAB, 0xCD };
	EffectsBytes_Write(body, payload, sizeof(payload));

	size_t n = 0;
	unsigned char *out = EffectsBuffer_Buffer(eb, &n);

	// the v3 header is two bytes - version then flags - and then the body it
	// was handed, which is what proves the sink survived the move
	TEST_ASSERT_(n == 4, "expected 2 header + 2 body bytes, got %zu", n);
	TEST_ASSERT_(out[0] == 3, "version byte %u, want 3", out[0]);
	TEST_ASSERT_(out[1] == 0, "flags byte %u, want 0", out[1]);
	TEST_ASSERT(out[2] == 0xAB && out[3] == 0xCD);

	rm_free(out);
	EffectsBuffer_Free(eb);
}

TEST_LIST = {
	{ "EffectsV3Record:entityRecords",   test_effectsV3Record_entityRecords },
	{ "EffectsV3Record:indexDDL",
		test_effectsV3Record_indexDDL },
	{ "EffectsV3Record:indexSchemasAreSorted",
		test_effectsV3Record_indexSchemasAreSorted },
	{ "EffectsV3Record:indexVectorOptions",
		test_effectsV3Record_indexVectorOptions },
	{ "EffectsV3Record:constraintDDL",
		test_effectsV3Record_constraintDDL },
	{ "EffectsV3Record:singularRecords",  test_effectsV3Record_singularRecords },
	{ "EffectsV3Record:takeBodyAcrossArms",
		test_effectsV3Record_takeBodyAcrossArms },
	{ NULL, NULL }
};
