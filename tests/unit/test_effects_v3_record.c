/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

// The record encoder, checked against the conformance fixtures.
//
// The segment tests compared a slice; these compare WHOLE PAYLOADS, version
// byte and flags byte included, because a record fixture is a complete
// GRAPH.EFFECT payload. That closes the one gap the slice comparison left: it
// could not catch a field written in the wrong place relative to the record
// framing, only one written wrongly inside the list.
//
// Records are built by hand rather than through a grouping layer, which does
// not exist yet. That is deliberate for 2c: it isolates the ENCODING from the
// GROUPING, so a failure here is a wire-format disagreement and not a decision
// about which entities belong in the same record.
//
// The fixtures are not ground truth - they are generated from the Rust encoder.
// Ground truth is C's own source for the reused primitives and
// docs/effects-v3.md for the v3 blocks. A disagreement is settled against those
// and the fixture regenerated, never by editing an expectation here.

#include "src/effects/effects_v3_encode.h"
#include "src/effects/effects_v3_id_list.h"
#include "src/effects/effects_bytes.h"
#include "src/util/rmalloc.h"

void setup() {
	Alloc_Reset();
}

#define TEST_INIT setup();
#include "acutest.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// skips whitespace, so an expectation can be wrapped across source lines the
// way the .hex files themselves are
static int _nibble(char c) {
	if(c >= '0' && c <= '9') return c - '0';
	if(c >= 'a' && c <= 'f') return c - 'a' + 10;
	if(c >= 'A' && c <= 'F') return c - 'A' + 10;
	return -1;
}

static unsigned char *_unhex(const char *hex, size_t *n) {
	unsigned char *out = malloc(strlen(hex) / 2 + 1);
	size_t k = 0;
	int hi = -1;

	for(const char *p = hex; *p; p++) {
		int v = _nibble(*p);
		if(v < 0) continue;
		if(hi < 0) { hi = v; continue; }
		out[k++] = (unsigned char)((hi << 4) | v);
		hi = -1;
	}

	*n = k;
	return out;
}

static char *_hex(const unsigned char *b, size_t n) {
	char *out = malloc(2 * n + 1);
	for(size_t i = 0; i < n; i++) sprintf(out + 2 * i, "%02x", b[i]);
	out[2 * n] = '\0';
	return out;
}

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
static void _check(const EffectsV3Record *r, const char *hex,
		const char *name) {
	EffectsBytes *out = EffectsBytes_New(1024);

	unsigned char header[2] = { 3, 0 };
	EffectsBytes_Write(out, header, 2);
	EffectsV3_EncodeRecord(r, out);

	size_t got_n = EffectsBytes_Len(out);
	unsigned char *got = malloc(got_n);
	EffectsBytes_CopyInto(out, got);

	size_t want_n = 0;
	unsigned char *want = _unhex(hex, &want_n);

	char *got_s = _hex(got, got_n), *want_s = _hex(want, want_n);
	TEST_ASSERT_(got_n == want_n && memcmp(got, want, want_n) == 0,
			"%s: the corpus and this encoder disagree\n"
			"      got %s\n  expected %s", name, got_s, want_s);

	free(got_s); free(want_s); free(want); free(got);
	EffectsBytes_Free(out);
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
// records that carry no values
//------------------------------------------------------------------------------

// DELETE_NODE carries a LabelSet, which the spec's record table once denied
//
// The labels are the ones the node actually held, captured as it was deleted.
// A replica needs them to clear the right label-scoped index documents, and
// the query's pattern cannot supply them: MATCH (n:A) DELETE n over an (:A:B)
// node must clear :B's indexes too.
void test_effectsV3Record_deleteNode(void) {
	EffectsV3IdListBuilder *keep;
	uint64_t ids[] = { 1, 2, 3 };
	LabelID labels[] = { 0, 1, 2 };

	EffectsV3Record r = {
		.opcode   = EFFECT_DELETE_NODE,
		.count    = 3,
		.labels   = labels,
		.n_labels = 3,
		.ids      = _ids(ids, 3, &keep),
	};

	_check(&r, "03000500000003000000030000000000010000000200000001000000000103",
			"rec_delete_node");

	EffectsV3IdListBuilder_FreeIdList(&r.ids);
	EffectsV3IdListBuilder_Free(keep);
}

// SET_LABELS and REMOVE_LABELS: the labels, then all their nodes
//
// grouped by label rather than interleaved as (node, label) pairs, which is
// what lets the node ids be one contiguous Range
void test_effectsV3Record_labels(void) {
	{
		EffectsV3IdListBuilder *keep;
		uint64_t ids[] = { 1, 2, 3 };
		LabelID labels[] = { 4, 6 };
		EffectsV3Record r = {
			.opcode = EFFECT_SET_LABELS, .count = 3,
			.labels = labels, .n_labels = 2,
			.ids = _ids(ids, 3, &keep),
		};
		_check(&r, "030007000000030000000200040000000600000001000000000103",
				"rec_set_labels");
		EffectsV3IdListBuilder_FreeIdList(&r.ids);
		EffectsV3IdListBuilder_Free(keep);
	}
	{
		EffectsV3IdListBuilder *keep;
		uint64_t ids[] = { 1, 2, 3 };
		LabelID labels[] = { 4 };
		EffectsV3Record r = {
			.opcode = EFFECT_REMOVE_LABELS, .count = 3,
			.labels = labels, .n_labels = 1,
			.ids = _ids(ids, 3, &keep),
		};
		_check(&r, "0300080000000300000001000400000001000000000103",
				"rec_remove_labels");
		EffectsV3IdListBuilder_FreeIdList(&r.ids);
		EffectsV3IdListBuilder_Free(keep);
	}
}

// DELETE_EDGE: the type, then ids, then endpoints as their own lists
//
// the dst column here is one Repeat - three edges into the same node - which
// is the shape endpoint columns take and the reason Repeat exists
void test_effectsV3Record_deleteEdge(void) {
	EffectsV3IdListBuilder *ki, *ks, *kd;
	uint64_t ids[] = { 1, 2, 3 }, src[] = { 10, 11, 12 }, dst[] = { 77, 77, 77 };

	EffectsV3Record r = {
		.opcode      = EFFECT_DELETE_EDGE,
		.count       = 3,
		.relation_id = 5,
		.ids         = _ids(ids, 3, &ki),
		.src         = _ids(src, 3, &ks),
		.dst         = _ids(dst, 3, &kd),
	};

	_check(&r, "0300060000000300000005000000010000000001030100000000"
	           "0a0301000000024d03", "rec_delete_edge");

	EffectsV3IdListBuilder_FreeIdList(&r.ids);
	EffectsV3IdListBuilder_FreeIdList(&r.src);
	EffectsV3IdListBuilder_FreeIdList(&r.dst);
	EffectsV3IdListBuilder_Free(ki);
	EffectsV3IdListBuilder_Free(ks);
	EffectsV3IdListBuilder_Free(kd);
}

//------------------------------------------------------------------------------
// records that carry values
//------------------------------------------------------------------------------

// CREATE_NODE and UPDATE_NODE share a layout: LabelSet, AttrIds, IdList, values
//
// the AttrSet is SPLIT across the IdList - ids with the shape, values after the
// rows - which is the part the spec's record table got wrong and the fixtures
// settled
void test_effectsV3Record_nodeWithValues(void) {
	{
		EffectsV3IdListBuilder *keep;
		uint64_t ids[] = { 1, 2, 3 };
		LabelID labels[] = { 0, 3 };
		AttributeID attrs[] = { 7, 9 };
		SIValue *rows = _rows();

		EffectsV3Record r = {
			.opcode = EFFECT_CREATE_NODE, .count = 3,
			.labels = labels, .n_labels = 2,
			.attr_ids = attrs, .n_attrs = 2,
			.ids = _ids(ids, 3, &keep),
			.values = rows, .n_values = 6,
		};

		_check(&r, "0300030000000300000002000000000003000000020007000900010000"
		           "000001030020000001000000000000000008000002000000000000006100"
		           "0080000000100000010040000000000000 0000d03f00200000f7ffffffff"
		           "ffffff", "rec_create_node");

		EffectsV3IdListBuilder_FreeIdList(&r.ids);
		EffectsV3IdListBuilder_Free(keep);
		free(rows);
	}
	{
		EffectsV3IdListBuilder *keep;
		uint64_t ids[] = { 1, 2, 3 };
		LabelID labels[] = { 1 };
		AttributeID attrs[] = { 7, 9 };
		SIValue *rows = _rows();

		EffectsV3Record r = {
			.opcode = EFFECT_UPDATE_NODE, .count = 3,
			.labels = labels, .n_labels = 1,
			.attr_ids = attrs, .n_attrs = 2,
			.ids = _ids(ids, 3, &keep),
			.values = rows, .n_values = 6,
		};

		_check(&r, "0300010000000300000001000100000002000700090001000000000103"
		           "00200000010000000000000000080000020000000000000061000080000"
		           "00010000001004000000000000000 00d03f00200000f7ffffffffffffff",
				"rec_update_node");

		EffectsV3IdListBuilder_FreeIdList(&r.ids);
		EffectsV3IdListBuilder_Free(keep);
		free(rows);
	}
}

// UPDATE_EDGE carries its type and NOT its endpoints
//
// the replica indexes under what the record states rather than re-deriving it;
// endpoints are recoverable for an update, which is why only the create and
// delete forms carry them
void test_effectsV3Record_updateEdge(void) {
	EffectsV3IdListBuilder *keep;
	uint64_t ids[] = { 1, 2, 3 };
	AttributeID attrs[] = { 7, 9 };
	SIValue *rows = _rows();

	EffectsV3Record r = {
		.opcode = EFFECT_UPDATE_EDGE, .count = 3, .relation_id = 2,
		.attr_ids = attrs, .n_attrs = 2,
		.ids = _ids(ids, 3, &keep),
		.values = rows, .n_values = 6,
	};

	_check(&r, "030002000000030000000200000002000700090001000000000103002000"
	           "000100000000000000000800000200000000000000610000800000001000"
	           "0001004000000000000 00000d03f00200000f7ffffffffffffff",
			"rec_update_edge");

	EffectsV3IdListBuilder_FreeIdList(&r.ids);
	EffectsV3IdListBuilder_Free(keep);
	free(rows);
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
			.schema_type = SCHEMA_NODE, .schema_id = 3, .name = "Person" };
		_check(&r, "03000900000000000000030000000700000000000000506572736f6e00",
				"rec_add_schema_node");
	}
	{
		EffectsV3Record r = { .opcode = EFFECT_ADD_SCHEMA,
			.schema_type = SCHEMA_EDGE, .schema_id = 1, .name = "KNOWS" };
		_check(&r, "030009000000010000000100000006000000000000004b4e4f575300",
				"rec_add_schema_edge");
	}
	{
		EffectsV3Record r = { .opcode = EFFECT_ADD_ATTRIBUTE,
			.attr_id = 12, .name = "name" };
		_check(&r, "03000a0000000c0005000000000000006e616d6500",
				"rec_add_attribute");
	}
}

TEST_LIST = {
	{ "EffectsV3Record:deleteNode",       test_effectsV3Record_deleteNode },
	{ "EffectsV3Record:labels",           test_effectsV3Record_labels },
	{ "EffectsV3Record:deleteEdge",       test_effectsV3Record_deleteEdge },
	{ "EffectsV3Record:nodeWithValues",   test_effectsV3Record_nodeWithValues },
	{ "EffectsV3Record:updateEdge",       test_effectsV3Record_updateEdge },
	{ "EffectsV3Record:singularRecords",  test_effectsV3Record_singularRecords },
	{ NULL, NULL }
};
