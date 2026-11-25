/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

// Grouping: which entities end up in the same record.
//
// THE CONFORMANCE CORPUS CANNOT ARBITRATE ANY OF THIS. A fixture pins what one
// record looks like; grouping decides which entities are inside it, and that
// leaves no trace in the bytes of a single record. Two engines can agree on
// every fixture and still emit different payloads for the same write. So these
// are property tests over the accumulator rather than byte comparisons.
//
// The four rules under test are all invisible in the wire format:
//
//   1. one record per (run, opcode, shape); shape is the attribute IDS, never
//      the values, plus labels for a node and relationship type for an edge.
//      A RUN is a stretch of the query's writes that creates or deletes but
//      never both, so one payload may carry the same (opcode, shape) twice
//   2. runs in the order they opened, and inside one the stages and then the
//      key - a hash order differs between executions, let alone between
//      engines
//   3. label sets normalised ascending, so [7,8] and [8,7] are one shape
//   4. a new attribute announced ONCE per payload - the only rule that spans
//      groups, so the one a per-group implementation gets locally right and
//      globally wrong

#include <inttypes.h>

#include "src/effects/effects_v3_group.h"
#include "src/effects/effects_v3_encode.h"
#include "src/effects/effects_bytes.h"
#include "src/util/rmalloc.h"
#include "src/datatypes/map.h"
#include "src/datatypes/array.h"
#include "src/index/index_field.h"

void setup() {
	Alloc_Reset();
}

#define TEST_INIT setup();
#include "acutest.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// the opcode of the i-th record in an encoded payload, by walking it
//
// only the fields needed to skip a record are parsed; this is a test helper,
// not a decoder
static uint32_t _u32(const unsigned char *p) {
	return (uint32_t)p[0] | ((uint32_t)p[1] << 8)
		| ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

// encode and return the payload bytes
static unsigned char *_encode(EffectsV3Grouping *g, size_t *n) {
	EffectsBytes *out = EffectsBytes_New(4096);
	EffectsV3Grouping_Encode(g, out);
	*n = EffectsBytes_Len(out);
	unsigned char *b = malloc(*n > 0 ? *n : 1);
	EffectsBytes_CopyInto(out, b);
	EffectsBytes_Free(out);
	return b;
}

// encode, then read the payload back as a record list
//
// THE DECODER WALKS IT, not this file. A record's length depends on its
// opcode, its widths and its values, so a hand-rolled skip is a second encoder
// - and one written from the same assumption as the first agrees with it even
// when both are wrong. Everything above that needs only record 0 reads the
// bytes directly; a test about SEQUENCE cannot.
//
// The two bytes prepended are the payload header - version, then flags - which
// the accumulator does not write: it emits the body alone and
// EffectsBuffer_Buffer stamps the header in front of it.
static EffectsV3Records *_decode(EffectsV3Grouping *g) {
	size_t n = 0;
	unsigned char *body = _encode(g, &n);

	unsigned char *buf = malloc(n + 2);
	buf[0] = 3;     // version
	buf[1] = 0;     // flags: nothing this build emits is compressed
	memcpy(buf + 2, body, n);

	EffectsV3Records *records = NULL;
	EffectsV3Status status =
		EffectsV3_Decode((const char *)buf, n + 2, &records);

	TEST_ASSERT_(status == EFFECTS_V3_OK && records != NULL,
			"the accumulator emitted a payload its own decoder refuses: "
			"status %d", (int)status);

	free(buf);
	free(body);

	return records;
}

// the opcodes of a payload's records, as one line, for a failure message
static void _opcode_line(const EffectsV3Records *r, char *out, size_t n) {
	size_t w = 0;
	out[0] = '\0';
	for(uint32_t i = 0; i < r->n && w + 8 < n; i++) {
		w += (size_t)snprintf(out + w, n - w, "%s%d",
				(i == 0) ? "" : ", ", (int)r->records[i].opcode);
	}
}

//------------------------------------------------------------------------------
// rule 1: one record per (opcode, shape)
//------------------------------------------------------------------------------

// two entities of the SAME shape share one record
void test_effectsV3Group_sameShapeIsOneRecord(void) {
	EffectsV3Grouping *g = EffectsV3Grouping_New();

	LabelID labels[] = { 1 };
	AttributeID attrs[] = { 7 };
	SIValue v = SI_LongVal(1);

	EffectsV3Grouping_AddNode(g, EFFECT_CREATE_NODE, labels, 1, 10, attrs, &v, 1);
	EffectsV3Grouping_AddNode(g, EFFECT_CREATE_NODE, labels, 1, 11, attrs, &v, 1);
	EffectsV3Grouping_AddNode(g, EFFECT_CREATE_NODE, labels, 1, 12, attrs, &v, 1);

	TEST_ASSERT_(EffectsV3Grouping_RecordCount(g) == 1,
			"three nodes of one shape should be ONE record, got %u",
			EffectsV3Grouping_RecordCount(g));

	EffectsV3Grouping_Free(g);
}

// each shape component splits a record on its own
//
// labels, the attribute id set and the relationship type are tested
// separately, because a grouping that keys on two of the three passes any test
// that varies all three at once
void test_effectsV3Group_eachShapeComponentSplits(void) {
	{
		// differing LABELS
		EffectsV3Grouping *g = EffectsV3Grouping_New();
		LabelID a[] = { 1 }, b[] = { 2 };
		AttributeID attrs[] = { 7 };
		SIValue v = SI_LongVal(1);

		EffectsV3Grouping_AddNode(g, EFFECT_CREATE_NODE, a, 1, 10, attrs, &v, 1);
		EffectsV3Grouping_AddNode(g, EFFECT_CREATE_NODE, b, 1, 11, attrs, &v, 1);

		TEST_ASSERT_(EffectsV3Grouping_RecordCount(g) == 2,
				"differing labels must split: got %u records",
				EffectsV3Grouping_RecordCount(g));
		EffectsV3Grouping_Free(g);
	}
	{
		// differing ATTRIBUTE IDS
		EffectsV3Grouping *g = EffectsV3Grouping_New();
		LabelID labels[] = { 1 };
		AttributeID a[] = { 7 }, b[] = { 8 };
		SIValue v = SI_LongVal(1);

		EffectsV3Grouping_AddNode(g, EFFECT_CREATE_NODE, labels, 1, 10, a, &v, 1);
		EffectsV3Grouping_AddNode(g, EFFECT_CREATE_NODE, labels, 1, 11, b, &v, 1);

		TEST_ASSERT_(EffectsV3Grouping_RecordCount(g) == 2,
				"differing attribute ids must split: got %u records",
				EffectsV3Grouping_RecordCount(g));
		EffectsV3Grouping_Free(g);
	}
	{
		// differing RELATIONSHIP TYPE
		EffectsV3Grouping *g = EffectsV3Grouping_New();
		AttributeID attrs[] = { 7 };
		SIValue v = SI_LongVal(1);

		EffectsV3Grouping_AddEdge(g, EFFECT_CREATE_EDGE, 1, 10, 1, 2, attrs, &v, 1);
		EffectsV3Grouping_AddEdge(g, EFFECT_CREATE_EDGE, 2, 11, 1, 2, attrs, &v, 1);

		TEST_ASSERT_(EffectsV3Grouping_RecordCount(g) == 2,
				"differing relationship types must split: got %u records",
				EffectsV3Grouping_RecordCount(g));
		EffectsV3Grouping_Free(g);
	}
	{
		// differing OPCODE - a query that creates and updates groups each
		// family independently
		EffectsV3Grouping *g = EffectsV3Grouping_New();
		LabelID labels[] = { 1 };
		AttributeID attrs[] = { 7 };
		SIValue v = SI_LongVal(1);

		EffectsV3Grouping_AddNode(g, EFFECT_CREATE_NODE, labels, 1, 10, attrs, &v, 1);
		EffectsV3Grouping_AddNode(g, EFFECT_UPDATE_NODE, labels, 1, 11, attrs, &v, 1);

		TEST_ASSERT_(EffectsV3Grouping_RecordCount(g) == 2,
				"create and update must not share a record: got %u",
				EffectsV3Grouping_RecordCount(g));
		EffectsV3Grouping_Free(g);
	}
}

// SHAPE IS THE ATTRIBUTE IDS AND NEVER THE VALUES
//
// two entities with the same attribute but different values are one shape.
// A grouping that keyed on values would emit one record per entity and undo
// the batching entirely, while still passing every fixture
void test_effectsV3Group_valuesAreNotPartOfTheShape(void) {
	EffectsV3Grouping *g = EffectsV3Grouping_New();
	LabelID labels[] = { 1 };
	AttributeID attrs[] = { 7 };

	SIValue a = SI_LongVal(1);
	SIValue b = SI_DoubleVal(2.5);
	SIValue c = SI_NullVal();

	EffectsV3Grouping_AddNode(g, EFFECT_CREATE_NODE, labels, 1, 10, attrs, &a, 1);
	EffectsV3Grouping_AddNode(g, EFFECT_CREATE_NODE, labels, 1, 11, attrs, &b, 1);
	EffectsV3Grouping_AddNode(g, EFFECT_CREATE_NODE, labels, 1, 12, attrs, &c, 1);

	TEST_ASSERT_(EffectsV3Grouping_RecordCount(g) == 1,
			"different values of the same attribute are ONE shape, got %u "
			"records - values must never be part of the key",
			EffectsV3Grouping_RecordCount(g));

	EffectsV3Grouping_Free(g);
}

//------------------------------------------------------------------------------
// rule 3: label sets normalised ascending
//------------------------------------------------------------------------------

// [7,8] and [8,7] are one shape, and the emitted set is ascending
void test_effectsV3Group_labelOrderDoesNotSplit(void) {
	EffectsV3Grouping *g = EffectsV3Grouping_New();
	LabelID asc[]  = { 7, 8 };
	LabelID desc[] = { 8, 7 };
	AttributeID attrs[] = { 7 };
	SIValue v = SI_LongVal(1);

	EffectsV3Grouping_AddNode(g, EFFECT_CREATE_NODE, desc, 2, 10, attrs, &v, 1);
	EffectsV3Grouping_AddNode(g, EFFECT_CREATE_NODE, asc,  2, 11, attrs, &v, 1);

	TEST_ASSERT_(EffectsV3Grouping_RecordCount(g) == 1,
			"[8,7] and [7,8] are the same set and must be ONE record, got %u",
			EffectsV3Grouping_RecordCount(g));

	// and the labels reach the wire ascending, whichever order arrived first
	size_t n = 0;
	unsigned char *p = _encode(g, &n);

	// CREATE_NODE: u32 opcode, u32 count, u16 n_labels, then the labels
	TEST_ASSERT_(_u32(p) == EFFECT_CREATE_NODE, "expected a CREATE_NODE record");
	TEST_ASSERT_(_u32(p + 10) == 7 && _u32(p + 14) == 8,
			"labels must be emitted ascending: got %u then %u",
			_u32(p + 10), _u32(p + 14));

	free(p);
	EffectsV3Grouping_Free(g);
}

// rule 3 is over the label SET, so a repeated label is not a second label
//
// THE TWO FILING PATHS MUST NORMALISE ALIKE. AddNode sorts and deduplicates;
// StageUpdate used to only sort, so [7,7] opened a group of its own and wrote
// the label twice into the record's LabelSet - while the same update arriving
// through AddNode landed somewhere else entirely.
//
// Staged against a one-label update of the same shape rather than checked
// alone: a count asserted on [7,7] by itself passes whenever the two paths
// agree AND whenever neither deduplicates, which is the bug.
void test_effectsV3Group_stagedUpdateLabelsAreASet(void) {
	EffectsV3Grouping *g = EffectsV3Grouping_New();
	LabelID twice[] = { 7, 7 };
	LabelID once[]  = { 7 };
	SIValue v = SI_LongVal(1);

	EffectsV3Grouping_StageUpdate(g, EFFECT_UPDATE_NODE, 10, twice, 2, 0, 7, v);
	EffectsV3Grouping_StageUpdate(g, EFFECT_UPDATE_NODE, 11, once,  1, 0, 7, v);

	TEST_ASSERT_(EffectsV3Grouping_RecordCount(g) == 1,
			"[7,7] and [7] are the same label set and must be ONE record, "
			"got %u", EffectsV3Grouping_RecordCount(g));

	// and the repeat must not reach the wire: UPDATE_NODE is u32 opcode,
	// u32 count, u16 n_labels, then the labels
	size_t n = 0;
	unsigned char *p = _encode(g, &n);

	TEST_ASSERT_(_u32(p) == EFFECT_UPDATE_NODE, "expected an UPDATE_NODE record");

	uint16_t n_labels = (uint16_t)(p[8] | (p[9] << 8));
	TEST_ASSERT_(n_labels == 1,
			"a repeated label is one label on the wire, got %u", n_labels);
	TEST_ASSERT_(_u32(p + 10) == 7, "the label must be 7, got %u", _u32(p + 10));

	free(p);
	EffectsV3Grouping_Free(g);
}

//------------------------------------------------------------------------------
// rule 2: groups sorted by key
//------------------------------------------------------------------------------

// records come out in key order, not arrival order
//
// the shapes are added in DESCENDING key order, so arrival order and key order
// disagree - without that the sort is untested, since any order passes
void test_effectsV3Group_recordsAreSortedByKey(void) {
	EffectsV3Grouping *g = EffectsV3Grouping_New();
	AttributeID attrs[] = { 7 };
	SIValue v = SI_LongVal(1);

	// added 3, 2, 1; must emit 1, 2, 3
	for(int rel = 3; rel >= 1; rel--) {
		EffectsV3Grouping_AddEdge(g, EFFECT_CREATE_EDGE, rel, 10 + rel, 1, 2,
				attrs, &v, 1);
	}

	TEST_ASSERT_(EffectsV3Grouping_RecordCount(g) == 3,
			"three relationship types, three records, got %u",
			EffectsV3Grouping_RecordCount(g));

	size_t n = 0;
	unsigned char *p = _encode(g, &n);

	// CREATE_EDGE: u32 opcode, u32 count, i32 relation_id, u16 n_attrs,
	// u16 attr_id, then ids/src/dst as three IdLists of one Range each, then
	// one value.
	//
	// ALL THREE ARE CHECKED, not just the first. Asserting only that record 0
	// carries type 1 passes an emission order of 1, 3, 2 - which is not sorted
	// and is exactly what a partial sort or an unstable comparison produces.
	//
	// The three records are the same shape with single-byte ids, so they are
	// the same length and record k begins at k * (n / 3). That assumption is
	// asserted rather than assumed: if a future change makes the records
	// differ in length this fails here instead of reading a relation id out of
	// the middle of a value.
	TEST_ASSERT_(n % 3 == 0,
			"expected three equal-length records, got %zu bytes", n);

	const size_t rec = n / 3;
	TEST_ASSERT_(_u32(p) == _u32(p + rec) && _u32(p) == _u32(p + 2 * rec),
			"the three records should start with the same opcode; if they do "
			"not, the equal-length assumption below is wrong");

	for(uint32_t k = 0; k < 3; k++) {
		TEST_ASSERT_(_u32(p + k * rec + 8) == k + 1,
				"record %u should carry relationship type %u, got %u - "
				"records were not emitted in key order", k, k + 1,
				_u32(p + k * rec + 8));
	}

	free(p);
	EffectsV3Grouping_Free(g);

	// ARRIVAL STARTING AT THE SMALLEST KEY, which is the case that separates
	// "sorted" from "record 0 happens to be right". Added 1, 3, 2: an emitter
	// that does not sort at all leaves them in that order, and record 0 still
	// carries type 1.
	g = EffectsV3Grouping_New();
	const int arrival[] = { 1, 3, 2 };
	for(int i = 0; i < 3; i++) {
		EffectsV3Grouping_AddEdge(g, EFFECT_CREATE_EDGE, arrival[i],
				10 + arrival[i], 1, 2, attrs, &v, 1);
	}

	n = 0;
	p = _encode(g, &n);
	TEST_ASSERT_(n % 3 == 0, "expected three equal-length records");

	const size_t rec2 = n / 3;
	TEST_ASSERT_(_u32(p) == _u32(p + rec2) && _u32(p) == _u32(p + 2 * rec2),
			"the three records should start with the same opcode; if they do "
			"not, the equal-length assumption below is wrong");
	for(uint32_t k = 0; k < 3; k++) {
		TEST_ASSERT_(_u32(p + k * rec2 + 8) == k + 1,
				"arriving 1, 3, 2 must still emit 1, 2, 3: record %u carries "
				"type %u", k, _u32(p + k * rec2 + 8));
	}

	free(p);
	EffectsV3Grouping_Free(g);
}

// the same shapes added in two different orders produce IDENTICAL payloads
//
// this is the property the sort exists for, and it is stronger than checking a
// particular order: it fails for any ordering that depends on arrival
//
// WITHIN ONE RUN, which is as far as the property goes. A payload that both
// creates and deletes DOES depend on arrival, and has to - the records either
// side of the switch cannot be swapped without naming a live id in a create or
// a dead one in an update. These three shapes are all creates, so they are one
// run and the key alone decides.
void test_effectsV3Group_insertionOrderDoesNotChangeBytes(void) {
	AttributeID attrs[] = { 7 };
	SIValue v = SI_LongVal(1);
	LabelID l1[] = { 1 }, l2[] = { 2 }, l3[] = { 3 };
	LabelID *labels[] = { l1, l2, l3 };

	int forward[]  = { 0, 1, 2 };
	int backward[] = { 2, 1, 0 };
	unsigned char *out[2];
	size_t lens[2];

	for(int pass = 0; pass < 2; pass++) {
		const int *order = pass ? backward : forward;
		EffectsV3Grouping *g = EffectsV3Grouping_New();

		for(int i = 0; i < 3; i++) {
			int k = order[i];
			EffectsV3Grouping_AddNode(g, EFFECT_CREATE_NODE, labels[k], 1,
					100 + k, attrs, &v, 1);
		}

		out[pass] = _encode(g, lens + pass);
		EffectsV3Grouping_Free(g);
	}

	TEST_ASSERT_(lens[0] == lens[1] && memcmp(out[0], out[1], lens[0]) == 0,
			"two accumulators filled in opposite orders must produce identical "
			"buffers - %zu vs %zu bytes", lens[0], lens[1]);

	free(out[0]);
	free(out[1]);
}

//------------------------------------------------------------------------------
// rule 4: a new attribute is announced once per payload
//------------------------------------------------------------------------------

// two groups using the same attribute announce it ONCE
//
// This is the only rule that spans groups, so a per-group implementation gets
// it locally right and globally wrong: each group would announce the attribute
// it uses, and the replica would meet the same id twice under the same name.
void test_effectsV3Group_attributeAnnouncedOncePerPayload(void) {
	EffectsV3Grouping *g = EffectsV3Grouping_New();
	LabelID l1[] = { 1 }, l2[] = { 2 };
	AttributeID attrs[] = { 7 };
	SIValue v = SI_LongVal(1);

	// two DIFFERENT shapes, both using attribute 7
	EffectsV3Grouping_AddAttribute(g, 7, "name");
	EffectsV3Grouping_AddNode(g, EFFECT_CREATE_NODE, l1, 1, 10, attrs, &v, 1);

	EffectsV3Grouping_AddAttribute(g, 7, "name");
	EffectsV3Grouping_AddNode(g, EFFECT_CREATE_NODE, l2, 1, 11, attrs, &v, 1);

	// an edge kind too, since v3's ADD_ATTRIBUTE carries no node/relationship
	// discriminator and iterating both dictionaries is the way this goes wrong
	EffectsV3Grouping_AddAttribute(g, 7, "name");
	EffectsV3Grouping_AddEdge(g, EFFECT_CREATE_EDGE, 1, 12, 1, 2, attrs, &v, 1);

	// 1 announcement + 3 groups
	TEST_ASSERT_(EffectsV3Grouping_RecordCount(g) == 4,
			"attribute 7 must be announced once across the payload: expected "
			"4 records (1 announcement + 3 groups), got %u",
			EffectsV3Grouping_RecordCount(g));

	size_t n = 0;
	unsigned char *p = _encode(g, &n);
	TEST_ASSERT_(_u32(p) == EFFECT_ADD_ATTRIBUTE,
			"the announcement must precede the records that use the id");

	free(p);
	EffectsV3Grouping_Free(g);
}

// a schema is announced once per (type, id), and node 1 is not edge 1
void test_effectsV3Group_schemaAnnouncedOncePerTypeAndId(void) {
	EffectsV3Grouping *g = EffectsV3Grouping_New();

	EffectsV3Grouping_AddSchema(g, SCHEMA_NODE, 1, "Person");
	EffectsV3Grouping_AddSchema(g, SCHEMA_NODE, 1, "Person");
	TEST_ASSERT_(EffectsV3Grouping_RecordCount(g) == 1,
			"the same schema twice is one announcement, got %u",
			EffectsV3Grouping_RecordCount(g));

	// the same id under the other schema type is a different schema
	EffectsV3Grouping_AddSchema(g, SCHEMA_EDGE, 1, "KNOWS");
	TEST_ASSERT_(EffectsV3Grouping_RecordCount(g) == 2,
			"node schema 1 and edge schema 1 are different schemas, got %u",
			EffectsV3Grouping_RecordCount(g));

	EffectsV3Grouping_Free(g);
}

//------------------------------------------------------------------------------
// empty blocks
//------------------------------------------------------------------------------

// a create with NO PROPERTIES, and a node with NO LABELS
//
// The conformance corpus has a present case for 26 blocks whose cardinality can
// be zero and an empty counterpart for one of them, so it cannot arbitrate these
// - and this suite had the same hole, its shapes being drawn from the fixtures
// that exist. The decoder half of exactly this gap was a live bug: C's v3 decoder
// refused `CREATE (:Person)` while a fully green corpus and 26 green flow tests
// did not notice.
void test_effectsV3Group_emptyBlocks(void) {
	{
		// CREATE (:Person) - a label, no properties
		EffectsV3Grouping *g = EffectsV3Grouping_New();
		LabelID labels[] = { 1 };

		EffectsV3Grouping_AddNode(g, EFFECT_CREATE_NODE, labels, 1, 10,
				NULL, NULL, 0);
		EffectsV3Grouping_AddNode(g, EFFECT_CREATE_NODE, labels, 1, 11,
				NULL, NULL, 0);

		TEST_ASSERT_(EffectsV3Grouping_RecordCount(g) == 1,
				"two propertyless nodes of one label are ONE record, got %u",
				EffectsV3Grouping_RecordCount(g));

		size_t n = 0;
		unsigned char *p = _encode(g, &n);

		// u32 opcode, u32 count, u16 n_labels, i32 label, u16 n_attrs
		TEST_ASSERT_(_u32(p) == EFFECT_CREATE_NODE, "expected CREATE_NODE");
		TEST_ASSERT_(_u32(p + 4) == 2, "expected count 2, got %u", _u32(p + 4));
		// u32 opcode | u32 count | u16 n_labels | i32 label | u16 n_attrs
		TEST_ASSERT_(p[8] == 1 && p[9] == 0,
				"expected one label, got n_labels %u", (unsigned)p[8]);
		TEST_ASSERT_(p[14] == 0 && p[15] == 0,
				"a propertyless create must state n_attrs = 0, got %u",
				(unsigned)p[14]);

		free(p);
		EffectsV3Grouping_Free(g);
	}
	{
		// CREATE ({v:1}) - a property, no labels
		EffectsV3Grouping *g = EffectsV3Grouping_New();
		AttributeID attrs[] = { 7 };
		SIValue v = SI_LongVal(1);

		EffectsV3Grouping_AddNode(g, EFFECT_CREATE_NODE, NULL, 0, 10,
				attrs, &v, 1);

		TEST_ASSERT_(EffectsV3Grouping_RecordCount(g) == 1,
				"an unlabelled create is still a record, got %u",
				EffectsV3Grouping_RecordCount(g));

		size_t n = 0;
		unsigned char *p = _encode(g, &n);
		TEST_ASSERT_(p[8] == 0 && p[9] == 0,
				"an unlabelled node must state n_labels = 0, got %u",
				(unsigned)p[8]);

		free(p);
		EffectsV3Grouping_Free(g);
	}
	{
		// a labelless node and a labelled one are DIFFERENT shapes
		EffectsV3Grouping *g = EffectsV3Grouping_New();
		LabelID labels[] = { 1 };
		AttributeID attrs[] = { 7 };
		SIValue v = SI_LongVal(1);

		EffectsV3Grouping_AddNode(g, EFFECT_CREATE_NODE, NULL, 0, 10, attrs, &v, 1);
		EffectsV3Grouping_AddNode(g, EFFECT_CREATE_NODE, labels, 1, 11, attrs, &v, 1);

		TEST_ASSERT_(EffectsV3Grouping_RecordCount(g) == 2,
				"no labels and one label are different shapes, got %u records",
				EffectsV3Grouping_RecordCount(g));

		EffectsV3Grouping_Free(g);
	}
}

// A RECORD WITH NO EFFECT IS NOT EMITTED.
//
// The rule, which covers three categories without a per-record table: a record
// has no effect if removing an empty block leaves it saying nothing about any
// entity it names. An empty block that DESCRIBES the entities the record names
// is information and is legal; a schema announcement is a binding rather than an
// instruction and is legal regardless of what references it.
//
// So the distinction is which block is the record's SUBJECT, not whether a block
// is empty - which is why the empty-block cases above stay legal while these do
// not. `MATCH (n) SET n:Foo REMOVE n:Foo` leaves a label set that empties out,
// and a label record's entire payload IS its label set.
//
// The ruling that settled it: Rust's own emitter already suppresses the
// exactly-analogous no-effect update - set_node_attributes refuses to stage an
// empty attribute map - so the label path was an inconsistency in their emitter
// rather than a property of the format, and fixing the outlier beat legalising
// it.
//
// Readers TOLERATE these rather than refusing them: rejecting a zero-label
// record removes no parse surface, since DELETE_NODE and CREATE_NODE require the
// zero-length LabelSet path anyway.
void test_effectsV3Group_recordsWithNoEffectAreNotEmitted(void) {
	{
		// a label record with no labels: its whole payload is the label set
		EffectsV3Grouping *g = EffectsV3Grouping_New();
		EffectsV3Grouping_AddNode(g, EFFECT_SET_LABELS, NULL, 0, 10,
				NULL, NULL, 0);

		TEST_ASSERT_(EffectsV3Grouping_RecordCount(g) == 0,
				"a zero-label SET_LABELS has no effect and must not be emitted, "
				"got %u records", EffectsV3Grouping_RecordCount(g));

		size_t n = 0;
		unsigned char *p = _encode(g, &n);
		TEST_ASSERT_(n == 0,
				"a payload of only no-effect records must be empty, got %zu bytes",
				n);

		free(p);
		EffectsV3Grouping_Free(g);
	}
	{
		// the mirror, so the rule is not written for one opcode
		EffectsV3Grouping *g = EffectsV3Grouping_New();
		EffectsV3Grouping_AddNode(g, EFFECT_REMOVE_LABELS, NULL, 0, 10,
				NULL, NULL, 0);
		TEST_ASSERT_(EffectsV3Grouping_RecordCount(g) == 0,
				"a zero-label REMOVE_LABELS has no effect either, got %u",
				EffectsV3Grouping_RecordCount(g));
		EffectsV3Grouping_Free(g);
	}
	{
		// AND THE LINE HOLDS THE OTHER WAY. A DELETE_NODE with no labels says
		// these nodes carried no labels, which is information about the
		// entities it names - so it stays legal and must still be emitted
		EffectsV3Grouping *g = EffectsV3Grouping_New();
		EffectsV3Grouping_AddNode(g, EFFECT_DELETE_NODE, NULL, 0, 10,
				NULL, NULL, 0);

		TEST_ASSERT_(EffectsV3Grouping_RecordCount(g) == 1,
				"a labelless DELETE_NODE describes its entities and is NOT "
				"no effect, got %u records", EffectsV3Grouping_RecordCount(g));

		EffectsV3Grouping_Free(g);
	}
	{
		// a schema announcement is a binding, legal whether or not anything
		// references it
		EffectsV3Grouping *g = EffectsV3Grouping_New();
		EffectsV3Grouping_AddSchema(g, SCHEMA_NODE, 1, "Person");
		EffectsV3Grouping_AddAttribute(g, 7, "name");

		TEST_ASSERT_(EffectsV3Grouping_RecordCount(g) == 2,
				"announcements are bindings and stand alone, got %u",
				EffectsV3Grouping_RecordCount(g));

		EffectsV3Grouping_Free(g);
	}
}

//------------------------------------------------------------------------------
// staged updates
//------------------------------------------------------------------------------

// v2 hands over one (entity, attribute, value) at a time; a v3 record's shape
// is the entity's WHOLE updated attribute set
//
// So an update cannot be filed into a group on arrival - the shape that selects
// the group is not known until the query stops producing attributes for that
// entity. These pin the folding.
void test_effectsV3Group_stagedUpdates(void) {
	{
		// one entity, two attributes, arriving separately -> ONE record
		EffectsV3Grouping *g = EffectsV3Grouping_New();
		LabelID labels[] = { 1 };
		SIValue a = SI_LongVal(1), b = SI_LongVal(2);

		EffectsV3Grouping_StageUpdate(g, EFFECT_UPDATE_NODE, 10, labels, 1, 0,
				7, a);
		EffectsV3Grouping_StageUpdate(g, EFFECT_UPDATE_NODE, 10, labels, 1, 0,
				9, b);

		TEST_ASSERT_(EffectsV3Grouping_RecordCount(g) == 1,
				"two attributes of one entity are one record, got %u",
				EffectsV3Grouping_RecordCount(g));
		EffectsV3Grouping_Free(g);
	}
	{
		// ATTRIBUTE ARRIVAL ORDER MUST NOT SPLIT A SHAPE. Two entities with the
		// same attribute set, staged in opposite orders, are one record - which
		// is what the sort at flush exists for
		EffectsV3Grouping *g = EffectsV3Grouping_New();
		LabelID labels[] = { 1 };
		SIValue v = SI_LongVal(1);

		EffectsV3Grouping_StageUpdate(g, EFFECT_UPDATE_NODE, 10, labels, 1, 0, 7, v);
		EffectsV3Grouping_StageUpdate(g, EFFECT_UPDATE_NODE, 10, labels, 1, 0, 9, v);

		EffectsV3Grouping_StageUpdate(g, EFFECT_UPDATE_NODE, 11, labels, 1, 0, 9, v);
		EffectsV3Grouping_StageUpdate(g, EFFECT_UPDATE_NODE, 11, labels, 1, 0, 7, v);

		TEST_ASSERT_(EffectsV3Grouping_RecordCount(g) == 1,
				"{7,9} and {9,7} are one shape however they arrive, got %u "
				"records", EffectsV3Grouping_RecordCount(g));
		EffectsV3Grouping_Free(g);
	}
	{
		// entities with DIFFERENT attribute sets still split
		EffectsV3Grouping *g = EffectsV3Grouping_New();
		LabelID labels[] = { 1 };
		SIValue v = SI_LongVal(1);

		EffectsV3Grouping_StageUpdate(g, EFFECT_UPDATE_NODE, 10, labels, 1, 0, 7, v);
		EffectsV3Grouping_StageUpdate(g, EFFECT_UPDATE_NODE, 11, labels, 1, 0, 7, v);
		EffectsV3Grouping_StageUpdate(g, EFFECT_UPDATE_NODE, 11, labels, 1, 0, 9, v);

		TEST_ASSERT_(EffectsV3Grouping_RecordCount(g) == 2,
				"{7} and {7,9} are different shapes, got %u records",
				EffectsV3Grouping_RecordCount(g));
		EffectsV3Grouping_Free(g);
	}
	{
		// setting the same attribute twice keeps ONE slot: the wire carries one
		// value per attribute, and the query's own order decides which
		EffectsV3Grouping *g = EffectsV3Grouping_New();
		LabelID labels[] = { 1 };
		SIValue first = SI_LongVal(1), second = SI_LongVal(2);

		EffectsV3Grouping_StageUpdate(g, EFFECT_UPDATE_NODE, 10, labels, 1, 0,
				7, first);
		EffectsV3Grouping_StageUpdate(g, EFFECT_UPDATE_NODE, 10, labels, 1, 0,
				7, second);

		size_t n = 0;
		unsigned char *p = _encode(g, &n);

		// u32 opcode | u32 count | u16 n_labels | i32 label | u16 n_attrs
		TEST_ASSERT_(p[14] == 1 && p[15] == 0,
				"setting one attribute twice is still ONE attribute in the "
				"shape, got n_attrs %u", (unsigned)p[14]);

		free(p);
		EffectsV3Grouping_Free(g);
	}
	{
		// a T_NULL is a REMOVAL and is part of the shape like any other value -
		// filtering it would turn a property removal into a no-op
		EffectsV3Grouping *g = EffectsV3Grouping_New();
		LabelID labels[] = { 1 };
		SIValue null = SI_NullVal();

		EffectsV3Grouping_StageUpdate(g, EFFECT_UPDATE_NODE, 10, labels, 1, 0,
				7, null);

		TEST_ASSERT_(EffectsV3Grouping_RecordCount(g) == 1,
				"a null-valued update is a removal and must be emitted, got %u",
				EffectsV3Grouping_RecordCount(g));

		size_t n = 0;
		unsigned char *p = _encode(g, &n);
		TEST_ASSERT_(p[14] == 1 && p[15] == 0,
				"the removed attribute is still in the shape, got n_attrs %u",
				(unsigned)p[14]);

		free(p);
		EffectsV3Grouping_Free(g);
	}
}

// SETTING THE SAME ATTRIBUTE TWICE KEEPS THE LAST VALUE
//
// The overwrite arm of the staging path. It matters more since values moved into
// an arena: the superseded encoding is left behind as dead bytes rather than
// freed, so "the last value wins" depends on the staged attribute's offset being
// repointed. Getting that wrong emits the FIRST value and still produces a
// well-formed payload of exactly the right length.
void test_effectsV3Group_lastValueWins(void) {
	EffectsV3Grouping *g = EffectsV3Grouping_New();
	LabelID labels[] = { 1 };

	// same entity, same attribute, three times - and a distinctive final value
	EffectsV3Grouping_StageUpdate(g, EFFECT_UPDATE_NODE, 10, labels, 1, 0,
			7, SI_LongVal(111));
	EffectsV3Grouping_StageUpdate(g, EFFECT_UPDATE_NODE, 10, labels, 1, 0,
			7, SI_LongVal(222));
	EffectsV3Grouping_StageUpdate(g, EFFECT_UPDATE_NODE, 10, labels, 1, 0,
			7, SI_LongVal(333));

	TEST_ASSERT_(EffectsV3Grouping_RecordCount(g) == 1,
			"one entity setting one attribute three times is one record, got %u",
			EffectsV3Grouping_RecordCount(g));

	size_t n;
	unsigned char *b = _encode(g, &n);

	// the wire carries ONE value for the attribute, and it is the last one.
	// Searched rather than offset-walked: this test is about which value
	// survived, not about the record layout, and the layout has its own tests.
	int seen111 = 0, seen222 = 0, seen333 = 0;
	for(size_t i = 0; i + 8 <= n; i++) {
		uint64_t v;
		memcpy(&v, b + i, 8);
		if(v == 111) seen111 = 1;
		if(v == 222) seen222 = 1;
		if(v == 333) seen333 = 1;
	}

	TEST_ASSERT_(seen333, "the last value must be the one on the wire");
	TEST_ASSERT_(!seen111,
			"the first value must not reach the wire - a superseded encoding "
			"is dead arena, not a row");
	TEST_ASSERT_(!seen222, "the middle value must not reach the wire either");

	free(b);
	EffectsV3Grouping_Free(g);
}

// A FLUSHED GROUP MUST NOT DEPEND ON THE ARENA
//
// This is what makes reusing the arena safe. _flush_updates copies each staged
// value into the group's own byte sequence and only then resets arena_len, so
// the next statement writes over bytes nobody references. Had the group kept an
// offset instead of a copy, the second statement would silently rewrite the
// first statement's values - same record shape, same length, wrong contents.
//
// Both records are in the second payload by design: an accumulator is
// cumulative, and groups are never cleared by an encode.
void test_effectsV3Group_flushedGroupOutlivesTheArena(void) {
	EffectsV3Grouping *g = EffectsV3Grouping_New();
	LabelID labels[] = { 1 };

	EffectsV3Grouping_StageUpdate(g, EFFECT_UPDATE_NODE, 10, labels, 1, 0,
			7, SI_LongVal(4242));

	// forces the flush: the value is copied into its group and arena_len goes
	// back to zero
	TEST_ASSERT_(EffectsV3Grouping_RecordCount(g) == 1,
			"one staged entity is one record");

	// a second statement now writes over the very bytes the first one used
	EffectsV3Grouping_StageUpdate(g, EFFECT_UPDATE_NODE, 11, labels, 1, 0,
			7, SI_LongVal(9999));

	size_t n;
	unsigned char *b = _encode(g, &n);

	int saw4242 = 0, saw9999 = 0;
	for(size_t i = 0; i + 8 <= n; i++) {
		uint64_t v;
		memcpy(&v, b + i, 8);
		if(v == 4242) saw4242 = 1;
		if(v == 9999) saw9999 = 1;
	}

	TEST_ASSERT_(saw4242,
			"the FLUSHED value must survive the arena being overwritten - if "
			"the group kept an offset rather than a copy, this is the second "
			"statement's bytes instead");
	TEST_ASSERT_(saw9999, "the second statement's value must be there too");

	free(b);
	EffectsV3Grouping_Free(g);
}


// PROPERTY ORDER IN THE QUERY MUST NOT CHANGE THE BYTES
//
// attr_ids is half the partition key of a batched record, so without a
// canonical order `{a,b}` and `{b,a}` are different shapes - the same logical
// write lands in two groups, produces a different record count, and emits
// different bytes on two engines that agree about everything else. That is
// also a same-engine defect: _group_for compares the id array as given.
void test_effectsV3Group_attributeOrderDoesNotSplit(void) {
	LabelID labels[] = { 1 };
	AttributeID fwd[] = { 7, 9 };
	AttributeID rev[] = { 9, 7 };
	SIValue vf[] = { SI_LongVal(70), SI_LongVal(90) };
	SIValue vr[] = { SI_LongVal(90), SI_LongVal(70) };

	// one accumulator, two entities, the same set written both ways round
	EffectsV3Grouping *g = EffectsV3Grouping_New();
	EffectsV3Grouping_AddNode(g, EFFECT_CREATE_NODE, labels, 1, 10, fwd, vf, 2);
	EffectsV3Grouping_AddNode(g, EFFECT_CREATE_NODE, labels, 1, 11, rev, vr, 2);
	TEST_ASSERT_(EffectsV3Grouping_RecordCount(g) == 1,
			"the same attribute SET written in two orders is one shape, got %u "
			"records", EffectsV3Grouping_RecordCount(g));
	EffectsV3Grouping_Free(g);

	// and byte-for-byte: two payloads, each written one way round
	size_t na, nb;
	EffectsV3Grouping *a = EffectsV3Grouping_New();
	EffectsV3Grouping_AddNode(a, EFFECT_CREATE_NODE, labels, 1, 10, fwd, vf, 2);
	unsigned char *ba = _encode(a, &na);

	EffectsV3Grouping *b = EffectsV3Grouping_New();
	EffectsV3Grouping_AddNode(b, EFFECT_CREATE_NODE, labels, 1, 10, rev, vr, 2);
	unsigned char *bb = _encode(b, &nb);

	TEST_ASSERT_(na == nb && memcmp(ba, bb, na) == 0,
			"the same logical write must emit the same bytes whichever order "
			"the query listed its properties in");

	free(ba); free(bb);
	EffectsV3Grouping_Free(a);
	EffectsV3Grouping_Free(b);
}

// AND THE VALUES MOVE WITH THE IDS
//
// The failure this guards is silent: sorting ids while leaving values in query
// order emits a payload that is well-formed, passes every length check and
// passes a receiver's ascending check - with every value on the wrong
// attribute. Only reading the values back catches it.
void test_effectsV3Group_valuesFollowTheirAttributes(void) {
	LabelID labels[] = { 1 };
	// descending ids, so sorting must actually move something
	AttributeID rev[] = { 9, 7 };
	SIValue vr[] = { SI_LongVal(0xBBBB), SI_LongVal(0xAAAA) };

	EffectsV3Grouping *g = EffectsV3Grouping_New();
	EffectsV3Grouping_AddNode(g, EFFECT_CREATE_NODE, labels, 1, 10, rev, vr, 2);

	size_t n;
	unsigned char *b = _encode(g, &n);

	// attribute 7 is written first, so its value must be the one paired with
	// 7 in the call - 0xAAAA - and not the first value as written
	long long first = -1;
	for(size_t i = 0; i + 8 <= n; i++) {
		uint64_t v;
		memcpy(&v, b + i, 8);
		if(v == 0xAAAA || v == 0xBBBB) { first = (long long)v; break; }
	}

	TEST_ASSERT_(first == 0xAAAA,
			"attribute 7 sorts first, so 0xAAAA must precede 0xBBBB on the "
			"wire; got 0x%llX first - the ids were sorted and the values were "
			"not", (unsigned long long)first);

	free(b);
	EffectsV3Grouping_Free(g);
}

// the accumulator must hold its OWN copy of a staged value
//
// SIValue_Persist clones an M_VOLATILE value and leaves every other kind
// alone, while _update_release frees exactly M_SELF - so an owned value used
// to be stored as the caller's pointer and then freed out from under it. The
// accumulator's own bookkeeping stayed consistent either way, which is why
// nothing here saw it.
//
// ASSERTED WITHOUT A SANITIZER, by mutating the caller's buffer after staging
// and before the flush. A copy encodes what was staged; the caller's pointer
// encodes what the buffer says now. The difference is visible in the bytes,
// so this fails on a build that stores the pointer rather than needing a
// double free to be detected.
void test_effectsV3Group_stagedValueIsCopiedFromTheCaller(void) {
	EffectsV3Grouping *g = EffectsV3Grouping_New();
	LabelID labels[] = { 1 };

	// M_SELF: owns its own heap string, which is the case Persist left alone.
	// A literal cannot stand in here - SI_ConstStringVal is M_CONST and
	// SIValue_Free ignores it, so the bug is unreachable with one.
	SIValue owned = SI_DuplicateStringVal("ORIGINAL");
	TEST_ASSERT_(owned.allocation == M_SELF,
			"this test needs an owned value; got allocation %d",
			(int)owned.allocation);

	EffectsV3Grouping_StageUpdate(g, EFFECT_UPDATE_NODE, 10, labels, 1, 0,
			7, owned);

	// same length, so nothing reallocates: only the content changes
	memcpy(owned.stringval, "MUTATED!", 8);

	size_t n = 0;
	unsigned char *buf = _encode(g, &n);

	bool saw_original = false;
	bool saw_mutated  = false;
	for(size_t i = 0; i + 8 <= n; i++) {
		if(memcmp(buf + i, "ORIGINAL", 8) == 0) saw_original = true;
		if(memcmp(buf + i, "MUTATED!", 8) == 0) saw_mutated  = true;
	}

	TEST_ASSERT_(saw_original && !saw_mutated,
			"the staged value must be the accumulator's own copy: "
			"original in payload=%d, caller's later edit in payload=%d",
			(int)saw_original, (int)saw_mutated);

	free(buf);
	EffectsV3Grouping_Free(g);

	// and the caller still owns what it passed - freeing it here is the only
	// free of this allocation
	SIValue_Free(owned);

	// THE REPLACEMENT BRANCH IS A SECOND PLACE TO GET THIS WRONG, and it is
	// not reached above: staging a second value for an attribute already
	// staged frees the old one and stores the new, and that store had the same
	// defect. The existing last-value-wins test walks this branch with
	// SI_LongVal, which is M_NONE and cannot show it.
	g = EffectsV3Grouping_New();

	SIValue first  = SI_DuplicateStringVal("FIRSTVAL");
	SIValue second = SI_DuplicateStringVal("SECONDVL");
	TEST_ASSERT(first.allocation == M_SELF && second.allocation == M_SELF);

	EffectsV3Grouping_StageUpdate(g, EFFECT_UPDATE_NODE, 10, labels, 1, 0,
			7, first);
	EffectsV3Grouping_StageUpdate(g, EFFECT_UPDATE_NODE, 10, labels, 1, 0,
			7, second);   // same attribute: replaces

	memcpy(second.stringval, "EDITED!!", 8);

	n = 0;
	buf = _encode(g, &n);

	bool saw_second = false, saw_edit = false;
	for(size_t i = 0; i + 8 <= n; i++) {
		if(memcmp(buf + i, "SECONDVL", 8) == 0) saw_second = true;
		if(memcmp(buf + i, "EDITED!!", 8) == 0) saw_edit   = true;
	}

	TEST_ASSERT_(saw_second && !saw_edit,
			"the replacement must be copied too: replacement in payload=%d, "
			"caller's later edit in payload=%d",
			(int)saw_second, (int)saw_edit);

	free(buf);
	EffectsV3Grouping_Free(g);
	SIValue_Free(first);
	SIValue_Free(second);
}
// AND A REPEATED ID NEVER REACHES THE WIRE
//
// The contract is STRICTLY ascending, so a repeat is as much a violation as a
// descending pair - and the more dangerous of the two, because the two readers
// disagree about it. Rust refuses `w[0] >= w[1]` and resyncs; a C replica
// ACCEPTS it, takes both pairs, drops the second write to the id and leaves the
// attribute that lost its column unset. No error, no log, no resync, and a
// state-only check calls that green.
//
// Nothing upstream produces one today, so this pins a guarantee the encoder
// makes rather than a bug it had: the invariant is local now instead of
// borrowed.
void test_effectsV3Group_repeatedAttributeIsCollapsed(void) {
	LabelID labels[] = { 1 };
	// attribute 7 written twice, and out of order, so the collapse has to
	// survive the sort rather than depend on arrival order
	AttributeID dup[] = { 9, 7, 7 };
	SIValue vd[] = { SI_LongVal(0x9999), SI_LongVal(0xAAAA), SI_LongVal(0xBBBB) };

	EffectsV3Grouping *g = EffectsV3Grouping_New();
	EffectsV3Grouping_AddNode(g, EFFECT_CREATE_NODE, labels, 1, 10, dup, vd, 3);

	size_t n;
	unsigned char *b = _encode(g, &n);

	// the superseded value must not appear anywhere on the wire, and the one
	// that won must
	bool saw_first = false, saw_last = false, saw_other = false;
	for(size_t i = 0; i + 8 <= n; i++) {
		uint64_t v;
		memcpy(&v, b + i, 8);
		if(v == 0xAAAA) saw_first = true;
		if(v == 0xBBBB) saw_last  = true;
		if(v == 0x9999) saw_other = true;
	}

	TEST_ASSERT_(!saw_first,
			"the superseded value for attribute 7 reached the wire - the "
			"repeat was not collapsed, and a C replica will silently drop a "
			"column applying this");
	TEST_ASSERT_(saw_last,
			"last write must win for a repeated attribute id, but 0xBBBB is "
			"not on the wire");
	TEST_ASSERT_(saw_other,
			"collapsing the repeat must not disturb the other attribute");

	free(b);
	EffectsV3Grouping_Free(g);

	// and the shape itself shrank: {9,7,7} is the two-attribute set {7,9},
	// so it must be the SAME record as a plain {7,9} rather than a third shape
	AttributeID plain[] = { 7, 9 };
	SIValue vp[] = { SI_LongVal(0xBBBB), SI_LongVal(0x9999) };

	EffectsV3Grouping *m = EffectsV3Grouping_New();
	EffectsV3Grouping_AddNode(m, EFFECT_CREATE_NODE, labels, 1, 10, dup, vd, 3);
	EffectsV3Grouping_AddNode(m, EFFECT_CREATE_NODE, labels, 1, 11, plain, vp, 2);
	TEST_ASSERT_(EffectsV3Grouping_RecordCount(m) == 1,
			"a collapsed repeat is the same shape as the set it collapses to, "
			"got %u records", EffectsV3Grouping_RecordCount(m));
	EffectsV3Grouping_Free(m);
}

// AN UPDATED EDGE CARRIES NO ENDPOINTS
//
// An update-edge record states a relationship type and the edges it touches and
// nothing else - src and dst are on the wire for create and delete only. So
// _group_for opens the two endpoint builders for those opcodes alone, and an
// update-edge group reaches the encoder with them NULL.
//
// Nothing covered this: every edge case in this file was a CREATE_EDGE, so the
// null test guarding the endpoint conversion could be deleted and the suite
// stayed green.
//
// WHAT FAILS HERE IS THE CONVERSION, NOT THE BYTES. EffectsV3UpdateEdge has no
// src or dst fields at all (effects_v3.h) and the encoder's arm writes none, so
// stray endpoints have nowhere to appear in the payload. Drop the NULL test
// from _group_has_endpoints and _group_to_record hands a NULL builder to
// EffectsV3IdListBuilder_ToIdList, which dereferences it - this test dies on
// that, before it compares a byte. The byte comparison below is here to pin the
// record's shape, not to catch the endpoints.
void test_effectsV3Group_updatedEdgeHasNoEndpoints(void) {
	EffectsV3Grouping *g = EffectsV3Grouping_New();

	SIValue v = SI_LongVal(7);
	EffectsV3Grouping_StageUpdate(g, EFFECT_UPDATE_EDGE, 42, NULL, 0, 3, 11, v);

	TEST_ASSERT_(EffectsV3Grouping_RecordCount(g) == 1,
			"one staged edge update is one record, got %u",
			EffectsV3Grouping_RecordCount(g));

	size_t got_n = 0;
	unsigned char *got = _encode(g, &got_n);

	TEST_ASSERT_(got_n >= 4, "payload is %zu bytes, too short to hold a record",
			got_n);
	TEST_ASSERT_(_u32(got) == (uint32_t)EFFECT_UPDATE_EDGE,
			"first record's opcode is %u, expected UPDATE_EDGE (%u)",
			_u32(got), (uint32_t)EFFECT_UPDATE_EDGE);

	// the same record, built by hand with no endpoint lists
	EffectsV3IdListBuilder *ib = EffectsV3IdListBuilder_New();
	EffectsV3IdListBuilder_Push(ib, 42);
	EffectsV3IdList ids = EffectsV3IdListBuilder_ToIdList(ib);

	AttributeID attr_ids[] = { 11 };
	SIValue values[] = { SI_LongVal(7) };
	EffectsV3Record r = { .opcode = EFFECT_UPDATE_EDGE };
	r.update_edge.count       = 1;
	r.update_edge.relation_id = 3;
	r.update_edge.attr_ids    = attr_ids;
	r.update_edge.n_attrs     = 1;
	r.update_edge.ids         = ids;
	r.update_edge.values      = values;
	r.update_edge.n_values    = 1;

	EffectsBytes *wb = EffectsBytes_New(256);
	EffectsV3_EncodeRecord(&r, wb);
	size_t want_n = EffectsBytes_Len(wb);
	unsigned char *want = malloc(want_n > 0 ? want_n : 1);
	EffectsBytes_CopyInto(wb, want);

	TEST_ASSERT_(got_n == want_n,
			"encoded %zu bytes, the endpoint-free record is %zu - a group that "
			"converted null endpoint builders would be longer", got_n, want_n);
	TEST_ASSERT_(memcmp(got, want, want_n < got_n ? want_n : got_n) == 0,
			"bytes differ from the hand-built endpoint-free record");

	EffectsBytes_Free(wb);
	free(want);
	EffectsV3IdListBuilder_FreeIdList(&ids);
	EffectsV3IdListBuilder_Free(ib);
	free(got);
	EffectsV3Grouping_Free(g);
}

// step over an ADD_SCHEMA or ADD_ATTRIBUTE record
//
// The two announcement shapes are fixed - no count, no list - so walking them
// needs no decoder. A name is a u64 length that INCLUDES the NUL followed by
// that many bytes. Sizes come from the real types, as elsewhere in this file.
static size_t _skip_announcement
(
	const unsigned char *b,  // payload
	size_t o,                // offset of the record
	uint32_t opcode          // its opcode, already read at 'o'
) {
	o += sizeof(EffectType);

	if(opcode == EFFECT_ADD_SCHEMA) {
		o += sizeof(SchemaType) + sizeof(int);
	} else {
		o += sizeof(AttributeID);
	}

	uint64_t len;
	memcpy(&len, b + o, 8);
	return o + 8 + len;
}

// A FOLDED FIELD MUST NOT CARRY ITS RECORD AHEAD OF ITS OWN ANNOUNCEMENT
//
// GraphHub_AddIndex announces the attribute and THEN emits the index effect,
// once per field (graph_hub.c:671 and :710, called once per field from
// index_operations.c). The fold appends the second field into the record the
// first one created, which sits EARLIER in the list than the second field's
// own ADD_ATTRIBUTE - so written in list order the CREATE_INDEX names an
// attribute the replica has not been told about yet.
//
// That order is not merely untidy, it is refused. EffectsV3_Apply applies
// records in the order they arrive and says so in as many words - "records 9
// and 10 are normatively ahead of anything referencing the ids they introduce,
// so nothing here reorders" - and _VerifyDDLRefs turns a CREATE_INDEX naming
// an unresolvable attribute id into a false return (effects_v3_apply.c, and
// VerifyAttribute at effects_apply.c:554). The master loses the whole payload
// and the replica is marked diverged.
//
// WHAT FAILS HERE IS THE ORDER, NOT THE CONTENT. Collapse the two passes in
// EffectsV3Grouping_Encode back into one and the record count is still 4, the
// fold still works and every field is still on the record - and only this
// test notices. RecordCount cannot see it: the records are all present.
void test_effectsV3Group_announcementsPrecedeTheIndexRecord(void) {
	EffectsV3Grouping *g = EffectsV3Grouping_New();
	SIValue opts = SI_NullVal();

	// `CREATE INDEX FOR (n:P) ON (n.a, n.b)` on a graph that has seen neither
	// attribute before, which is what the hub does call by call
	EffectsV3Grouping_AddSchema(g, SCHEMA_NODE, 3, "P");

	EffectsV3Grouping_AddAttribute(g, 12, "a");
	EffectsV3Grouping_AddIndexField(g, EFFECT_CREATE_INDEX, SCHEMA_NODE, 3,
			"P", INDEX_FLD_RANGE, 12, "a", opts);

	// the second field's attribute is new too, and is announced AFTER the
	// record its field folds into
	EffectsV3Grouping_AddAttribute(g, 13, "b");
	EffectsV3Grouping_AddIndexField(g, EFFECT_CREATE_INDEX, SCHEMA_NODE, 3,
			"P", INDEX_FLD_RANGE, 13, "b", opts);

	TEST_ASSERT_(EffectsV3Grouping_RecordCount(g) == 4,
			"one schema, two attributes and one folded index record is 4, got %u",
			EffectsV3Grouping_RecordCount(g));

	size_t n = 0;
	unsigned char *p = _encode(g, &n);

	const uint32_t want[] = {
		EFFECT_ADD_SCHEMA,
		EFFECT_ADD_ATTRIBUTE,
		EFFECT_ADD_ATTRIBUTE,
		EFFECT_CREATE_INDEX
	};

	size_t o = 0;
	for(int i = 0; i < 4; i++) {
		TEST_ASSERT_(o + sizeof(EffectType) <= n,
				"payload ends after %zu of %zu bytes, before record %d", o, n, i);

		const uint32_t op = _u32(p + o);
		TEST_ASSERT_(op == want[i],
				"record %d is opcode %u, expected %u - every attribute "
				"announcement must be written before the index record that "
				"names it", i, op, want[i]);

		if(op == EFFECT_CREATE_INDEX) {
			break;
		}

		o = _skip_announcement(p, o, op);
	}

	free(p);
	EffectsV3Grouping_Free(g);
}

// walk a payload holding ONE CREATE_INDEX record to the start of its options
//
// The record is `opcode . schema_type . SCHEMAS . field_type . fields .
// OPTIONS`, where SCHEMAS is `u16 n . (u32 id, name) x n` and a string is a
// u64 length that INCLUDES the NUL followed by that many bytes. Sizes come
// from the real types rather than literals, so a widened enum moves the test
// with the encoder.
static size_t _options_offset(const unsigned char *b, size_t n) {
	size_t o = sizeof(EffectType) + sizeof(SchemaType);

	uint64_t len;

	uint16_t n_schemas;
	memcpy(&n_schemas, b + o, 2);
	o += 2;

	for(uint16_t i = 0; i < n_schemas; i++) {
		o += 4;                        // schema id
		memcpy(&len, b + o, 8);        // its name
		o += 8 + len;
	}

	o += 4;                            // field_type

	uint16_t n_fields;
	memcpy(&n_fields, b + o, 2);
	o += 2;

	for(uint16_t i = 0; i < n_fields; i++) {
		o += sizeof(AttributeID);
		memcpy(&len, b + o, 8);        // field name
		o += 8 + len;
	}

	TEST_ASSERT_(o <= n, "walked past the payload: %zu of %zu", o, n);
	return o;
}

// two fields of one statement become ONE record
void test_effectsV3Group_indexFieldsFoldIntoOneRecord(void) {
	EffectsV3Grouping *g = EffectsV3Grouping_New();
	SIValue opts = SI_NullVal();

	// what GraphHub_AddIndex does for `CREATE INDEX FOR (n:P) ON (n.a, n.b)`:
	// two calls, and an attribute announcement interleaved between them, which
	// is why the fold cannot just look at the most recent record
	EffectsV3Grouping_AddIndexField(g, EFFECT_CREATE_INDEX, SCHEMA_NODE, 3,
			"P", INDEX_FLD_RANGE, 12, "a", opts);
	EffectsV3Grouping_AddAttribute(g, 13, "b");
	EffectsV3Grouping_AddIndexField(g, EFFECT_CREATE_INDEX, SCHEMA_NODE, 3,
			"P", INDEX_FLD_RANGE, 13, "b", opts);

	// the attribute announcement plus ONE index record
	TEST_ASSERT_(EffectsV3Grouping_RecordCount(g) == 2,
			"two fields of one statement should be one record beside the "
			"attribute announcement, got %u", EffectsV3Grouping_RecordCount(g));

	EffectsV3Grouping_Free(g);
}

// A CCH STATEMENT IS ONE RECORD NAMING EVERY RELATIONSHIP TYPE IT SPANS
//
// The path index is defined over the types TOGETHER, so a record naming a
// subset of them would describe a different index. This is the case the schema
// list was added for - every other index statement names exactly one.
void test_effectsV3Group_cchIsOneRecordOverEveryType(void) {
	EffectsV3Grouping *g = EffectsV3Grouping_New();

	// named high-id first, so the sort below is doing something
	const uint32_t ids[]   = { 9, 2 };
	const char    *names[] = { "ROAD", "R" };

	EffectsV3Grouping_AddCCHIndex(g, EFFECT_CREATE_INDEX, ids, names, 2, 7,
			"w");

	TEST_ASSERT_(EffectsV3Grouping_RecordCount(g) == 1,
			"a CCH statement is one record, got %u",
			EffectsV3Grouping_RecordCount(g));

	size_t n;
	unsigned char *b = _encode(g, &n);

	size_t o = sizeof(EffectType) + sizeof(SchemaType);

	uint16_t n_schemas;
	memcpy(&n_schemas, b + o, 2);
	TEST_ASSERT_(n_schemas == 2,
			"both relationship types belong to the one record, got %u",
			n_schemas);

	uint32_t first;
	memcpy(&first, b + o + 2, 4);
	TEST_ASSERT_(first == 2,
			"the schema list reaches the wire ascending, got %u first", first);

	free(b);
	EffectsV3Grouping_Free(g);
}

// AN ORDINARY INDEX FIELD NEVER FOLDS INTO A CCH RECORD
//
// The fold key is the schema LIST. A range index on one of the same
// relationship types is a different statement, and folding it in would both
// change the CCH index and lose the range one.
void test_effectsV3Group_cchDoesNotAbsorbAnIndexField(void) {
	EffectsV3Grouping *g = EffectsV3Grouping_New();
	SIValue opts = SI_NullVal();

	const uint32_t ids[]   = { 2, 9 };
	const char    *names[] = { "R", "ROAD" };

	EffectsV3Grouping_AddCCHIndex(g, EFFECT_CREATE_INDEX, ids, names, 2, 7,
			"w");

	EffectsV3Grouping_AddIndexField(g, EFFECT_CREATE_INDEX, SCHEMA_EDGE, 2,
			"R", INDEX_FLD_RANGE, 12, "a", opts);

	TEST_ASSERT_(EffectsV3Grouping_RecordCount(g) == 2,
			"a CCH index and a range index are two statements, got %u",
			EffectsV3Grouping_RecordCount(g));

	EffectsV3Grouping_Free(g);
}

// does the payload contain these bytes anywhere?
//
// A blunt search on purpose: the assertion below is about a string SURVIVING,
// and walking to it would mean teaching the walker to step over a whole record
// it is not interested in.
static bool _contains(const unsigned char *b, size_t n, const char *needle) {
	size_t m = strlen(needle);
	if(m > n) return false;

	for(size_t i = 0; i + m <= n; i++) {
		if(memcmp(b + i, needle, m) == 0) return true;
	}
	return false;
}

// A SINGLE-TYPE CCH RECORD IS NOT "AN EARLIER RECORD NAMING THE SAME INDEX"
//
// ON ()-[r:ROAD]-() is an ordinary CCH statement and names one relationship
// type, so it matches an ordinary index statement on opcode, schema type and
// schema id. It is still a different index - a path index is its own object -
// and counting it as the same one strips language and stopwords off the first
// real record that follows, because those are stated once per index per
// payload.
//
// Two types would not show this: _names_one_schema is false for them.
void test_effectsV3Group_singleTypeCCHDoesNotStripOptions(void) {
	EffectsV3Grouping *g = EffectsV3Grouping_New();

	const uint32_t ids[]   = { 2 };
	const char    *names[] = { "R" };

	EffectsV3Grouping_AddCCHIndex(g, EFFECT_CREATE_INDEX, ids, names, 1, 7,
			"w");

	SIValue sw = SIArray_New(1);
	SIArray_Append(&sw, SI_ConstStringVal("the"));

	SIValue opts = SI_Map(2);
	Map_Add(&opts, SI_ConstStringVal("stopwords"), sw);
	Map_Add(&opts, SI_ConstStringVal("language"), SI_ConstStringVal("english"));
	SIArray_Free(sw);  // Map_Add clones it (map.c:157)

	// a fulltext index on the SAME relationship type, in the same payload
	EffectsV3Grouping_AddIndexField(g, EFFECT_CREATE_INDEX, SCHEMA_EDGE, 2,
			"R", INDEX_FLD_FULLTEXT, 12, "a", opts);

	TEST_ASSERT_(EffectsV3Grouping_RecordCount(g) == 2,
			"a path index and a fulltext index are two records, got %u",
			EffectsV3Grouping_RecordCount(g));

	size_t n;
	unsigned char *b = _encode(g, &n);

	TEST_ASSERT_(_contains(b, n, "english"),
			"the fulltext record lost its language to the CCH record");
	TEST_ASSERT_(_contains(b, n, "the"),
			"the fulltext record lost its stopwords to the CCH record");

	free(b);
	SIValue_Free(opts);
	EffectsV3Grouping_Free(g);
}

// the same field twice does not inflate the count the record states
void test_effectsV3Group_indexFieldNotAddedTwice(void) {
	EffectsV3Grouping *g = EffectsV3Grouping_New();
	SIValue opts = SI_NullVal();

	EffectsV3Grouping_AddIndexField(g, EFFECT_CREATE_INDEX, SCHEMA_NODE, 3,
			"P", INDEX_FLD_RANGE, 12, "a", opts);
	EffectsV3Grouping_AddIndexField(g, EFFECT_CREATE_INDEX, SCHEMA_NODE, 3,
			"P", INDEX_FLD_RANGE, 12, "a", opts);

	size_t n;
	unsigned char *b = _encode(g, &n);

	// opcode . schema_type . u16 n . (u32 id, name) . field_type
	size_t o = sizeof(EffectType) + sizeof(SchemaType) + 2 + 4;
	uint64_t len;
	memcpy(&len, b + o, 8);
	o += 8 + len + 4;

	uint16_t n_fields;
	memcpy(&n_fields, b + o, 2);
	TEST_ASSERT_(n_fields == 1, "the same field twice should be one, got %u",
			n_fields);

	free(b);
	EffectsV3Grouping_Free(g);
}

// AN UNSTATED PHONETIC IS A CLEAR FLAG, NOT C'S STORED DEFAULT
//
// C seeds every field's phonetic with the literal string "no"
// (index_field.h:15). Rust reads any non-empty phonetic as ENABLED, so a
// default emitted in place of an absence turns the option on when it crosses
// engines. Only the byte can see this: the record decodes either way, and a
// replica ends up with a working index either way.
void test_effectsV3Group_unstatedPhoneticIsAClearFlag(void) {
	EffectsV3Grouping *g = EffectsV3Grouping_New();

	// a statement that says weight and nothing else
	SIValue opts = SI_Map(1);
	Map_Add(&opts, SI_ConstStringVal("weight"), SI_DoubleVal(2.0));

	EffectsV3Grouping_AddIndexField(g, EFFECT_CREATE_INDEX, SCHEMA_NODE, 3,
			"P", INDEX_FLD_FULLTEXT, 12, "a", opts);

	size_t n;
	unsigned char *b = _encode(g, &n);
	size_t o = _options_offset(b, n);

	// language, stopwords: clear
	TEST_ASSERT_(b[o] == 0, "language should be clear, got %u", b[o]);
	o += 1;
	TEST_ASSERT_(b[o] == 0, "stopwords should be clear, got %u", b[o]);
	o += 1;

	// weight: stated, so a flag and eight bytes of f64
	TEST_ASSERT_(b[o] == 1, "weight was stated and should be set, got %u", b[o]);
	o += 1;
	double w;
	memcpy(&w, b + o, 8);
	TEST_ASSERT_(w == 2.0, "weight should be 2.0, got %f", w);
	o += 8;

	// nostem: clear
	TEST_ASSERT_(b[o] == 0, "nostem should be clear, got %u", b[o]);
	o += 1;

	// phonetic: THE ONE THIS TEST EXISTS FOR
	TEST_ASSERT_(b[o] == 0,
			"an unstated phonetic must emit a CLEAR FLAG, got %u - C's stored "
			"default is the string \"no\", which Rust reads as ENABLED", b[o]);
	o += 1;

	TEST_ASSERT_(o == n, "a non-vector options block should end the record: "
			"%zu of %zu", o, n);

	// and the string itself is nowhere in the payload
	for(size_t i = 0; i + 2 < n; i++) {
		TEST_ASSERT_(!(b[i] == 'n' && b[i + 1] == 'o' && b[i + 2] == '\0'),
				"the payload carries the literal \"no\" at offset %zu", i);
	}

	free(b);
	Map_Free(opts);
	EffectsV3Grouping_Free(g);
}

// AN OPTION STATED AT ITS DEFAULT VALUE IS STILL STATED
//
// `OPTIONS {weight: 1.0}` names the value the field would have taken anyway.
// A conversion that compared against the default rather than reading the key
// would drop it, and every behavioural test would still pass - the field ends
// up with weight 1.0 either way. The distinction only matters on the wire,
// where absence means "the statement did not say".
void test_effectsV3Group_optionStatedAtItsDefaultIsPresent(void) {
	EffectsV3Grouping *g = EffectsV3Grouping_New();

	SIValue opts = SI_Map(1);
	Map_Add(&opts, SI_ConstStringVal("weight"),
			SI_DoubleVal(INDEX_FIELD_DEFAULT_WEIGHT));

	EffectsV3Grouping_AddIndexField(g, EFFECT_CREATE_INDEX, SCHEMA_NODE, 3,
			"P", INDEX_FLD_FULLTEXT, 12, "a", opts);

	size_t n;
	unsigned char *b = _encode(g, &n);
	size_t o = _options_offset(b, n) + 2;  // past language and stopwords

	TEST_ASSERT_(b[o] == 1,
			"a weight stated at its default is STATED and must set the "
			"presence byte, got %u", b[o]);

	free(b);
	Map_Free(opts);
	EffectsV3Grouping_Free(g);
}

// fields that disagree about their options CANNOT share a record
//
// A record carries one options block and apply hands it to every field in it,
// so folding two different weights into one record would give both fields
// whichever weight was written. C reaches this through
// db.idx.fulltext.createNodeIndex, which rebuilds weight, phonetic and nostem
// per field.
void test_effectsV3Group_differingFieldOptionsSplit(void) {
	EffectsV3Grouping *g = EffectsV3Grouping_New();

	SIValue a = SI_Map(1);
	Map_Add(&a, SI_ConstStringVal("weight"), SI_DoubleVal(2.0));
	SIValue b = SI_Map(1);
	Map_Add(&b, SI_ConstStringVal("weight"), SI_DoubleVal(5.0));

	EffectsV3Grouping_AddIndexField(g, EFFECT_CREATE_INDEX, SCHEMA_NODE, 3,
			"P", INDEX_FLD_FULLTEXT, 12, "a", a);
	EffectsV3Grouping_AddIndexField(g, EFFECT_CREATE_INDEX, SCHEMA_NODE, 3,
			"P", INDEX_FLD_FULLTEXT, 13, "b", b);

	TEST_ASSERT_(EffectsV3Grouping_RecordCount(g) == 2,
			"two fields with different weights need two records, got %u",
			EffectsV3Grouping_RecordCount(g));

	Map_Free(a);
	Map_Free(b);
	EffectsV3Grouping_Free(g);
}

// ...but identical options still fold, whatever else differs about arrival
void test_effectsV3Group_identicalFieldOptionsFold(void) {
	EffectsV3Grouping *g = EffectsV3Grouping_New();

	SIValue a = SI_Map(1);
	Map_Add(&a, SI_ConstStringVal("weight"), SI_DoubleVal(2.0));
	SIValue b = SI_Map(1);
	Map_Add(&b, SI_ConstStringVal("weight"), SI_DoubleVal(2.0));

	EffectsV3Grouping_AddIndexField(g, EFFECT_CREATE_INDEX, SCHEMA_NODE, 3,
			"P", INDEX_FLD_FULLTEXT, 12, "a", a);
	EffectsV3Grouping_AddIndexField(g, EFFECT_CREATE_INDEX, SCHEMA_NODE, 3,
			"P", INDEX_FLD_FULLTEXT, 13, "b", b);

	TEST_ASSERT_(EffectsV3Grouping_RecordCount(g) == 1,
			"two separately built but equal option maps should fold, got %u",
			EffectsV3Grouping_RecordCount(g));

	Map_Free(a);
	Map_Free(b);
	EffectsV3Grouping_Free(g);
}

// INDEX-LEVEL OPTIONS ARE STATED ONCE PER INDEX PER PAYLOAD
//
// Index_SetStopwords objects if stopwords are set AT ALL, so a statement that
// splits into two records must not repeat them - the second application fails
// and the replica refuses a payload that is otherwise correct. The split here is
// forced by differing weights, the only way C produces one. A language-only
// version of this test is green either way, because Index_SetLanguage objects
// only when the language DIFFERS.
void test_effectsV3Group_indexLevelOptionsStatedOnce(void) {
	EffectsV3Grouping *g = EffectsV3Grouping_New();

	SIValue sw = SIArray_New(1);
	SIArray_Append(&sw, SI_ConstStringVal("the"));

	SIValue a = SI_Map(3);
	Map_Add(&a, SI_ConstStringVal("stopwords"), sw);
	Map_Add(&a, SI_ConstStringVal("language"), SI_ConstStringVal("english"));
	Map_Add(&a, SI_ConstStringVal("weight"), SI_DoubleVal(2.0));

	SIValue b = SI_Map(3);
	Map_Add(&b, SI_ConstStringVal("stopwords"), sw);
	Map_Add(&b, SI_ConstStringVal("language"), SI_ConstStringVal("english"));
	Map_Add(&b, SI_ConstStringVal("weight"), SI_DoubleVal(5.0));

	EffectsV3Grouping_AddIndexField(g, EFFECT_CREATE_INDEX, SCHEMA_NODE, 3,
			"P", INDEX_FLD_FULLTEXT, 12, "a", a);
	EffectsV3Grouping_AddIndexField(g, EFFECT_CREATE_INDEX, SCHEMA_NODE, 3,
			"P", INDEX_FLD_FULLTEXT, 13, "b", b);

	TEST_ASSERT_(EffectsV3Grouping_RecordCount(g) == 2,
			"differing weights should still split, got %u",
			EffectsV3Grouping_RecordCount(g));

	size_t n;
	unsigned char *buf = _encode(g, &n);

	// the FIRST record states both index-level options
	size_t o = _options_offset(buf, n);
	TEST_ASSERT_(buf[o] == 1, "record 1 should state the language");
	uint64_t len;
	memcpy(&len, buf + o + 1, 8);
	size_t after_lang = o + 1 + 8 + len;
	TEST_ASSERT_(buf[after_lang] == 1, "record 1 should state the stopwords");

	// walk to the end of record 1 to reach record 2
	size_t p = after_lang + 1;
	uint64_t n_sw;
	memcpy(&n_sw, buf + p, 8);
	p += 8;
	for(uint64_t i = 0; i < n_sw; i++) {
		memcpy(&len, buf + p, 8);
		p += 8 + len;
	}
	p += 1 + 8;   // weight: set, and its f64
	p += 1;       // nostem: clear
	p += 1;       // phonetic: clear

	TEST_ASSERT_(p < n, "record 2 should follow record 1 at %zu of %zu", p, n);

	size_t o2 = _options_offset(buf + p, n - p) + p;
	TEST_ASSERT_(buf[o2] == 0,
			"record 2 must NOT repeat the language, got %u", buf[o2]);
	TEST_ASSERT_(buf[o2 + 1] == 0,
			"record 2 must NOT repeat the stopwords, got %u - Index_SetStopwords "
			"refuses when they are already set, so the replica would refuse a "
			"payload that is otherwise correct", buf[o2 + 1]);

	free(buf);
	SIArray_Free(sw);
	Map_Free(a);
	Map_Free(b);
	EffectsV3Grouping_Free(g);
}

// A STRIPPED RECORD IS STILL SOMETHING TO FOLD INTO
//
// Index-level options are stated once per index per payload, so every record
// for an index after the first has its language and stopwords removed. The
// block ARRIVING with a field never is - the caller repeats the statement's
// index-level options on every field's map (proc_fulltext_create_index.c:340
// onwards). So a fold test that reads language or stopwords compares a
// stripped candidate against an unstripped arrival and can only say no, for
// every record after the first, forever: field 3 would not fold into field
// 2's record even stating exactly what field 2 stated, and would open a third
// record identical to it but for the name - then a fourth, one per remaining
// field.
//
// Three fields, two distinct weights. Two records is the only answer: 'c'
// cannot fold into 'a' - 2.0 and 5.0 are different weights, which is what
// differingFieldOptionsSplit pins - so two records means 'c' went where it
// belongs, onto 'b'. Compare language or stopwords again and this reads 3.
void test_effectsV3Group_foldsIntoAStrippedRecord(void) {
	EffectsV3Grouping *g = EffectsV3Grouping_New();

	SIValue sw = SIArray_New(1);
	SIArray_Append(&sw, SI_ConstStringVal("the"));

	// one fulltext statement's three field maps, built the way the proc builds
	// them: the index-level options on every one of them
	const double   weight[3] = { 2.0, 5.0, 5.0 };
	const char    *name[3]   = { "a", "b", "c" };
	SIValue        m[3];

	for(int i = 0; i < 3; i++) {
		m[i] = SI_Map(3);
		Map_Add(&m[i], SI_ConstStringVal("stopwords"), sw);
		Map_Add(&m[i], SI_ConstStringVal("language"),
				SI_ConstStringVal("english"));
		Map_Add(&m[i], SI_ConstStringVal("weight"), SI_DoubleVal(weight[i]));
	}

	for(int i = 0; i < 3; i++) {
		EffectsV3Grouping_AddIndexField(g, EFFECT_CREATE_INDEX, SCHEMA_NODE, 3,
				"P", INDEX_FLD_FULLTEXT, 12 + i, name[i], m[i]);
	}

	TEST_ASSERT_(EffectsV3Grouping_RecordCount(g) == 2,
			"'b' and 'c' state the same weight and belong in one record, got "
			"%u records - 'c' did not fold into the record 'b' opened, because "
			"that record had its index-level options stripped and 'c' still "
			"carries them", EffectsV3Grouping_RecordCount(g));

	for(int i = 0; i < 3; i++) {
		Map_Free(m[i]);
	}
	SIArray_Free(sw);  // Map_Add clones it (map.c:157)
	EffectsV3Grouping_Free(g);
}

// a drop folds the same way and carries no options at all
void test_effectsV3Group_dropIndexFoldsWithNoOptions(void) {
	EffectsV3Grouping *g = EffectsV3Grouping_New();

	EffectsV3Grouping_AddIndexField(g, EFFECT_DROP_INDEX, SCHEMA_NODE, 3,
			"P", INDEX_FLD_RANGE, 12, "a", SI_NullVal());
	EffectsV3Grouping_AddIndexField(g, EFFECT_DROP_INDEX, SCHEMA_NODE, 3,
			"P", INDEX_FLD_RANGE, 13, "b", SI_NullVal());

	TEST_ASSERT_(EffectsV3Grouping_RecordCount(g) == 1,
			"two dropped fields of one statement should be one record, got %u",
			EffectsV3Grouping_RecordCount(g));

	size_t n;
	unsigned char *b = _encode(g, &n);

	// the record ends where the options would have begun: zero bytes, not an
	// empty block
	TEST_ASSERT_(_options_offset(b, n) == n,
			"a drop must carry NO options block at all, %zu bytes follow",
			n - _options_offset(b, n));

	free(b);
	EffectsV3Grouping_Free(g);
}

// two indexes on different labels never fold, however alike their options
void test_effectsV3Group_differentIndexesNeverFold(void) {
	EffectsV3Grouping *g = EffectsV3Grouping_New();
	SIValue opts = SI_NullVal();

	EffectsV3Grouping_AddIndexField(g, EFFECT_CREATE_INDEX, SCHEMA_NODE, 3,
			"P", INDEX_FLD_RANGE, 12, "a", opts);
	EffectsV3Grouping_AddIndexField(g, EFFECT_CREATE_INDEX, SCHEMA_NODE, 4,
			"Q", INDEX_FLD_RANGE, 12, "a", opts);
	// same label, different ENTITY - a node index and an edge index are two
	EffectsV3Grouping_AddIndexField(g, EFFECT_CREATE_INDEX, SCHEMA_EDGE, 3,
			"P", INDEX_FLD_RANGE, 12, "a", opts);
	// same everything but the field type
	EffectsV3Grouping_AddIndexField(g, EFFECT_CREATE_INDEX, SCHEMA_NODE, 3,
			"P", INDEX_FLD_FULLTEXT, 12, "a", opts);

	TEST_ASSERT_(EffectsV3Grouping_RecordCount(g) == 4,
			"label, entity and field type each split a record, got %u",
			EffectsV3Grouping_RecordCount(g));

	EffectsV3Grouping_Free(g);
}

// ANOTHER INDEX OVER THE SAME LABEL IS NOT THIS INDEX
//
// Index-level options are stated once per index per payload, and the record
// that may not repeat them is one for the SAME index. A range index and a
// fulltext index over one label are two objects - differentIndexesNeverFold
// pins exactly that - so the range index's record states nothing about the
// fulltext index's options and must not make them look already sent.
//
// Read the other way round, this is a replica that builds a fulltext index
// with no language and no stopwords while the primary has both, from a query
// that indexed a property twice. Raise the flag before the field-type test and
// the second record below comes out empty.
void test_effectsV3Group_anotherIndexDoesNotStripOptions(void) {
	EffectsV3Grouping *g = EffectsV3Grouping_New();

	SIValue sw = SIArray_New(1);
	SIArray_Append(&sw, SI_ConstStringVal("the"));

	SIValue none = SI_Map(1);               // a range field states nothing
	SIValue ft   = SI_Map(2);
	Map_Add(&ft, SI_ConstStringVal("language"), SI_ConstStringVal("english"));
	Map_Add(&ft, SI_ConstStringVal("stopwords"), sw);

	// a range index over P, then a FULLTEXT index over the same label
	EffectsV3Grouping_AddIndexField(g, EFFECT_CREATE_INDEX, SCHEMA_NODE, 3,
			"P", INDEX_FLD_RANGE, 12, "a", none);
	EffectsV3Grouping_AddIndexField(g, EFFECT_CREATE_INDEX, SCHEMA_NODE, 3,
			"P", INDEX_FLD_FULLTEXT, 13, "b", ft);

	TEST_ASSERT_(EffectsV3Grouping_RecordCount(g) == 2,
			"a range index and a fulltext index are two records, got %u",
			EffectsV3Grouping_RecordCount(g));

	size_t n;
	unsigned char *buf = _encode(g, &n);

	// record 1, the range index: the text half is five clear bytes
	size_t o = _options_offset(buf, n);
	for(int i = 0; i < 5; i++) {
		TEST_ASSERT_(buf[o + i] == 0,
				"the range index should state no text option, byte %d is %u",
				i, buf[o + i]);
	}

	// record 2, the fulltext index, still states both
	size_t q  = o + 5;
	size_t o2 = _options_offset(buf + q, n - q) + q;

	TEST_ASSERT_(buf[o2] == 1,
			"the fulltext index must keep its language - the range index "
			"before it is a different index and stated none");

	uint64_t len;
	memcpy(&len, buf + o2 + 1, 8);
	TEST_ASSERT_(buf[o2 + 1 + 8 + len] == 1,
			"the fulltext index must keep its stopwords");

	free(buf);
	SIArray_Free(sw);
	Map_Free(none);
	Map_Free(ft);
	EffectsV3Grouping_Free(g);
}

// AN OPTION NO EARLIER RECORD STATED IS NOT A REPEAT
//
// Stripping exists so that a second record for one index does not re-send what
// the first already sent - Index_SetStopwords refuses stopwords that are
// already set. What makes that a repeat is the EARLIER RECORD CARRYING IT, not
// the earlier record existing.
//
// Two statements against one index in one payload: the first field asks for no
// language, the second asks for english. Differing weights keep them apart, so
// the second is the only record that could carry the language anywhere. Strip
// on "a record exists" and it is cleared here and stated nowhere - the replica
// builds the index with no language while the primary has english.
void test_effectsV3Group_unstatedOptionIsNotARepeat(void) {
	EffectsV3Grouping *g = EffectsV3Grouping_New();

	SIValue a = SI_Map(1);                  // no language
	Map_Add(&a, SI_ConstStringVal("weight"), SI_DoubleVal(2.0));

	SIValue b = SI_Map(2);                  // the language, first stated here
	Map_Add(&b, SI_ConstStringVal("language"), SI_ConstStringVal("english"));
	Map_Add(&b, SI_ConstStringVal("weight"), SI_DoubleVal(5.0));

	EffectsV3Grouping_AddIndexField(g, EFFECT_CREATE_INDEX, SCHEMA_NODE, 3,
			"P", INDEX_FLD_FULLTEXT, 12, "a", a);
	EffectsV3Grouping_AddIndexField(g, EFFECT_CREATE_INDEX, SCHEMA_NODE, 3,
			"P", INDEX_FLD_FULLTEXT, 13, "b", b);

	TEST_ASSERT_(EffectsV3Grouping_RecordCount(g) == 2,
			"differing weights should split, got %u",
			EffectsV3Grouping_RecordCount(g));

	size_t n;
	unsigned char *buf = _encode(g, &n);

	// record 1 states no language, no stopwords, a weight, no nostem,
	// no phonetic - thirteen bytes in all
	size_t o = _options_offset(buf, n);
	TEST_ASSERT_(buf[o] == 0, "record 1 states no language");
	size_t q = o + 1 + 1 + (1 + 8) + 1 + 1;

	size_t o2 = _options_offset(buf + q, n - q) + q;
	TEST_ASSERT_(buf[o2] == 1,
			"record 2 must state the language: no earlier record carried it, "
			"so clearing it here states it nowhere at all");

	free(buf);
	Map_Free(a);
	Map_Free(b);
	EffectsV3Grouping_Free(g);
}

// A FOLD MUST NOT SWALLOW AN OPTION NOBODY ELSE STATES
//
// _field_options_eq reads the field-level half only, and has to: a record that
// has had its language stripped must still be something to fold into, which is
// what foldsIntoAStrippedRecord pins. The consequence is that a block STATING
// an index-level option folds into a record WITHOUT one whenever the weights
// and the rest agree.
//
// Both fields below ask for weight 2.0, so they are one record. Only the
// second mentions a language. Free it with the rest of the folded block and
// the payload states it nowhere: the replica builds the index with no language
// while the primary has english - the same divergence as a lost strip, reached
// by folding rather than by splitting.
void test_effectsV3Group_foldKeepsAnOptionNobodyElseStates(void) {
	EffectsV3Grouping *g = EffectsV3Grouping_New();

	SIValue a = SI_Map(1);                  // no language
	Map_Add(&a, SI_ConstStringVal("weight"), SI_DoubleVal(2.0));

	SIValue b = SI_Map(2);                  // the language, and the SAME weight
	Map_Add(&b, SI_ConstStringVal("language"), SI_ConstStringVal("english"));
	Map_Add(&b, SI_ConstStringVal("weight"), SI_DoubleVal(2.0));

	EffectsV3Grouping_AddIndexField(g, EFFECT_CREATE_INDEX, SCHEMA_NODE, 3,
			"P", INDEX_FLD_FULLTEXT, 12, "a", a);
	EffectsV3Grouping_AddIndexField(g, EFFECT_CREATE_INDEX, SCHEMA_NODE, 3,
			"P", INDEX_FLD_FULLTEXT, 13, "b", b);

	TEST_ASSERT_(EffectsV3Grouping_RecordCount(g) == 1,
			"equal weights fold whatever the index-level half says, got %u",
			EffectsV3Grouping_RecordCount(g));

	size_t n;
	unsigned char *buf = _encode(g, &n);

	size_t o = _options_offset(buf, n);
	TEST_ASSERT_(buf[o] == 1,
			"the folded-into record must carry the language the second field "
			"stated - no other record states it, so dropping it states it "
			"nowhere");

	// and it is the language that was stated, not an empty one
	uint64_t len;
	memcpy(&len, buf + o + 1, 8);
	TEST_ASSERT_(len == strlen("english") + 1,
			"expected 'english' and its NUL, got a length of %llu",
			(unsigned long long)len);
	TEST_ASSERT_(memcmp(buf + o + 1 + 8, "english", 7) == 0,
			"the adopted language should read 'english'");

	free(buf);
	Map_Free(a);
	Map_Free(b);
	EffectsV3Grouping_Free(g);
}

// A NODE IS CREATED BEFORE IT IS UPDATED
//
// CREATE (n:L) SET n.x = 1 is ONE entity in TWO groups: the create carries the
// labels, the SET stages an update, and nothing merges them. So the payload
// has to state them in an order the replica can apply.
//
// The opcode numbering is v2's and carries no order - UPDATE_NODE is 1 while
// CREATE_NODE is 3 - so a sort on the opcode itself puts the update first. The
// replica then updates a node it does not have; and were it to tolerate that,
// the CREATE_NODE arriving afterwards carries no attributes, which v3 states
// as "this node has no properties", so the property would be dropped instead.
//
// Rust's for_each_record (emit.rs) emits created nodes, created edges,
// updates, label changes, deleted edges then deleted nodes, and says that is
// the only ordering the format requires - so this is byte parity too, not just
// an apply-order question.
void test_effectsV3Group_createPrecedesUpdate(void) {
	EffectsV3Grouping *g = EffectsV3Grouping_New();
	const LabelID L = 7;

	EffectsV3Grouping_AddNode(g, EFFECT_CREATE_NODE, &L, 1, 100, NULL, NULL, 0);
	EffectsV3Grouping_StageUpdate(g, EFFECT_UPDATE_NODE, 100, &L, 1, 0, 42,
			SI_LongVal(1));

	TEST_ASSERT_(EffectsV3Grouping_RecordCount(g) == 2,
			"a create and an update are two records, got %u",
			EffectsV3Grouping_RecordCount(g));

	size_t n;
	unsigned char *b = _encode(g, &n);

	TEST_ASSERT_(_u32(b) == EFFECT_CREATE_NODE,
			"the node must be created before it is updated: first record is "
			"opcode %u, wanted CREATE_NODE (%d)",
			_u32(b), EFFECT_CREATE_NODE);

	free(b);
	EffectsV3Grouping_Free(g);
}

// AN EDGE IS UNHOOKED BEFORE THE NODE IT HANGS OFF IS REMOVED
//
// The other half of the same rule, and the one the opcode numbering happens to
// get backwards too: DELETE_NODE is 5 and DELETE_EDGE is 6.
void test_effectsV3Group_edgeDeletedBeforeNode(void) {
	EffectsV3Grouping *g = EffectsV3Grouping_New();
	const LabelID L = 7;

	EffectsV3Grouping_AddNode(g, EFFECT_DELETE_NODE, &L, 1, 100, NULL, NULL, 0);
	EffectsV3Grouping_AddEdge(g, EFFECT_DELETE_EDGE, 3, 500, 100, 101, NULL,
			NULL, 0);

	size_t n;
	unsigned char *b = _encode(g, &n);

	TEST_ASSERT_(_u32(b) == EFFECT_DELETE_EDGE,
			"edges are deleted before nodes: first record is opcode %u, "
			"wanted DELETE_EDGE (%d)", _u32(b), EFFECT_DELETE_EDGE);

	free(b);
	EffectsV3Grouping_Free(g);
}

//------------------------------------------------------------------------------
// runs: a payload that both creates and deletes
//------------------------------------------------------------------------------
//
// WHY A STAGE ORDER OVER THE WHOLE PAYLOAD CANNOT BE RIGHT. C's DataBlock
// hands a freed id back to the query that freed it, so one payload can name
// one id for two different entities - and nothing in either record says which
// of the two it means. Sorting creates ahead of deletes ships CREATE id N for
// an id the replica still has live; sorting deletes ahead of creates breaks
// the same pair written the other way round. The records have to go out in the
// order the query made them, and a RUN is how much of that order is kept.
//
// These assert the OPCODE SEQUENCE of the payload rather than a byte layout,
// because that is the whole of the claim: which record a replica meets first.

// the opcode of every record, in order
static void _assert_opcodes
(
	const EffectsV3Records *r,  // decoded payload
	const int *want,            // expected opcodes
	uint32_t n,                 // how many
	const char *what            // what this payload is
) {
	char line[256];
	_opcode_line(r, line, sizeof(line));

	TEST_ASSERT_(r->n == n,
			"%s: expected %u records, got %u - [%s]", what, n, r->n, line);

	for(uint32_t i = 0; i < n; i++) {
		TEST_ASSERT_(r->records[i].opcode == (EffectType)want[i],
				"%s: record %u is opcode %d, wanted %d - [%s]",
				what, i, (int)r->records[i].opcode, want[i], line);
	}
}

// A DELETE IS NOT OVERTAKEN BY THE CREATE THAT REUSES ITS ID
//
// `MATCH (n:A) DELETE n CREATE (:B)` - the measured shape. The delete frees an
// id and the create is handed it straight back, so both records name it. With
// the creates staged first the replica meets CREATE_NODE for a node it still
// has, refuses the whole payload and resyncs; the delete behind it is never
// reached, because apply stops at the first failure.
void test_effectsV3Group_aDeleteIsNotOvertakenByACreate(void) {
	EffectsV3Grouping *g = EffectsV3Grouping_New();
	const LabelID A = 1, B = 2;

	// the SAME id in both records, which is what the allocator does here
	EffectsV3Grouping_AddNode(g, EFFECT_DELETE_NODE, &A, 1, 1, NULL, NULL, 0);
	EffectsV3Grouping_AddNode(g, EFFECT_CREATE_NODE, &B, 1, 1, NULL, NULL, 0);

	EffectsV3Records *r = _decode(g);

	const int want[] = { EFFECT_DELETE_NODE, EFFECT_CREATE_NODE };
	_assert_opcodes(r, want, 2, "delete then create of one id");

	EffectsV3_RecordsFree(r);
	EffectsV3Grouping_Free(g);
}

// AND THE MIRROR, WHICH IS WHY NO FIXED STAGE ORDER WORKS
//
// `CREATE (n:B) DELETE n` emits a create and a delete of the same id in the
// other order. Deletes-first would carry this one and break the test above by
// exactly the same mechanism, so the pair of them is what rules out ordering
// by opcode altogether.
void test_effectsV3Group_aCreateIsNotOvertakenByADelete(void) {
	EffectsV3Grouping *g = EffectsV3Grouping_New();
	const LabelID B = 2;

	EffectsV3Grouping_AddNode(g, EFFECT_CREATE_NODE, &B, 1, 1, NULL, NULL, 0);
	EffectsV3Grouping_AddNode(g, EFFECT_DELETE_NODE, &B, 1, 1, NULL, NULL, 0);

	EffectsV3Records *r = _decode(g);

	const int want[] = { EFFECT_CREATE_NODE, EFFECT_DELETE_NODE };
	_assert_opcodes(r, want, 2, "create then delete of one id");

	EffectsV3_RecordsFree(r);
	EffectsV3Grouping_Free(g);
}

// AND AN EDGE ID IS REUSED THE SAME WAY A NODE ID IS
//
// Not a restatement of the node case: the edge id space is its own DataBlock
// and the refusal is its own code path - _ApplyCreateEdge claims the stated
// ids and reports "names an edge id that is already live on this replica"
// where _ApplyCreateNode reports the node equivalent. A fix that filed only
// node records into runs would pass every test above and still resync on
// `MATCH ()-[e:R]->() DELETE e CREATE ()-[:R]->()`.
void test_effectsV3Group_aDeletedEdgeIdIsNotOvertakenByACreate(void) {
	EffectsV3Grouping *g = EffectsV3Grouping_New();

	// the edge at id 1 goes, and the edge created after it is handed that id
	EffectsV3Grouping_AddEdge(g, EFFECT_DELETE_EDGE, 3, 1, 10, 11, NULL, NULL,
			0);
	EffectsV3Grouping_AddEdge(g, EFFECT_CREATE_EDGE, 3, 1, 12, 13, NULL, NULL,
			0);

	EffectsV3Records *r = _decode(g);

	const int want[] = { EFFECT_DELETE_EDGE, EFFECT_CREATE_EDGE };
	_assert_opcodes(r, want, 2, "delete then create of one edge id");

	EffectsV3_RecordsFree(r);
	EffectsV3Grouping_Free(g);
}

// AN ENTITY FILED LATE DOES NOT JOIN A GROUP EMITTED EARLY
//
// This is the case group-level appearance order alone does not cover, and it
// is measured rather than imagined: `MATCH (n:A) CREATE (:B {n:'b1'}) DELETE n
// CREATE (:B {n:'b2'})` puts both CREATE clauses in ONE group whose first
// member predates the delete, and the second clause is handed the ids the
// delete just freed. Appending it to that group emits it ahead of the delete
// again - the bug, with the groups in arrival order.
//
// So the two creates are two records here, not one, and the delete sits
// between them. The payload is one record larger and is the only sequence a
// replica can apply.
void test_effectsV3Group_aLaterCreateDoesNotJoinAnEarlierRun(void) {
	EffectsV3Grouping *g = EffectsV3Grouping_New();
	const LabelID A = 1, B = 2;

	// b1, at an id of its own
	EffectsV3Grouping_AddNode(g, EFFECT_CREATE_NODE, &B, 1, 3, NULL, NULL, 0);

	// the delete frees id 1
	EffectsV3Grouping_AddNode(g, EFFECT_DELETE_NODE, &A, 1, 1, NULL, NULL, 0);

	// b2, handed it back - SAME shape as b1, so one group would swallow it
	EffectsV3Grouping_AddNode(g, EFFECT_CREATE_NODE, &B, 1, 1, NULL, NULL, 0);

	EffectsV3Records *r = _decode(g);

	const int want[] = {
		EFFECT_CREATE_NODE, EFFECT_DELETE_NODE, EFFECT_CREATE_NODE
	};
	_assert_opcodes(r, want, 3, "a create either side of a delete");

	// ONE ENTITY EACH, which is what says the second create opened its own
	// record rather than folding into the first: two records of the right
	// opcodes with counts 2 and 0 would pass the sequence check above
	TEST_ASSERT_(r->records[0].create_node.count == 1 &&
			r->records[2].create_node.count == 1,
			"the two creates should hold one node each, got %u and %u",
			r->records[0].create_node.count, r->records[2].create_node.count);

	EffectsV3_RecordsFree(r);
	EffectsV3Grouping_Free(g);
}

// EVERY CREATED NODE STILL PRECEDES EVERY CREATED EDGE, ACROSS CLAUSES
//
// The run keeps the payload's halves apart; INSIDE a run the stages still
// decide, and this is why. A group accepts entities for as long as its run is
// open, so the edge record opened by the first clause is reused by the second
// and emitted where the FIRST clause put it - ahead of the record creating the
// second clause's endpoints. _ApplyCreateEdge checks both endpoints exist
// before it claims anything, so that payload is refused.
void test_effectsV3Group_createdNodesPrecedeCreatedEdgesAcrossClauses(void) {
	EffectsV3Grouping *g = EffectsV3Grouping_New();
	const LabelID A = 1, B = 2;

	// CREATE (:A)-[:R]->(:A)
	EffectsV3Grouping_AddNode(g, EFFECT_CREATE_NODE, &A, 1, 0, NULL, NULL, 0);
	EffectsV3Grouping_AddNode(g, EFFECT_CREATE_NODE, &A, 1, 1, NULL, NULL, 0);
	EffectsV3Grouping_AddEdge(g, EFFECT_CREATE_EDGE, 1, 0, 0, 1, NULL, NULL, 0);

	// CREATE (:B)-[:R]->(:B), the same relationship type and a new label
	EffectsV3Grouping_AddNode(g, EFFECT_CREATE_NODE, &B, 1, 2, NULL, NULL, 0);
	EffectsV3Grouping_AddNode(g, EFFECT_CREATE_NODE, &B, 1, 3, NULL, NULL, 0);
	EffectsV3Grouping_AddEdge(g, EFFECT_CREATE_EDGE, 1, 1, 2, 3, NULL, NULL, 0);

	EffectsV3Records *r = _decode(g);

	const int want[] = {
		EFFECT_CREATE_NODE, EFFECT_CREATE_NODE, EFFECT_CREATE_EDGE
	};
	_assert_opcodes(r, want, 3, "two create clauses sharing a relationship");

	// the two edges are still ONE record: a run that both creates nodes and
	// creates edges does not split, it sorts
	TEST_ASSERT_(r->records[2].create_edge.count == 2,
			"both edges share a shape and should be one record of two, got %u",
			r->records[2].create_edge.count);

	EffectsV3_RecordsFree(r);
	EffectsV3Grouping_Free(g);
}

// AND THE MIRROR ON THE WAY OUT: every deleted edge precedes every deleted node
//
// op_delete unhooks before it removes, and it does so once per batch - so two
// delete clauses interleave edge and node records the way the two create
// clauses above interleave node and edge ones.
void test_effectsV3Group_deletedEdgesPrecedeDeletedNodesAcrossClauses(void) {
	EffectsV3Grouping *g = EffectsV3Grouping_New();
	const LabelID A = 1, B = 2;

	EffectsV3Grouping_AddEdge(g, EFFECT_DELETE_EDGE, 1, 0, 0, 1, NULL, NULL, 0);
	EffectsV3Grouping_AddNode(g, EFFECT_DELETE_NODE, &A, 1, 0, NULL, NULL, 0);

	EffectsV3Grouping_AddEdge(g, EFFECT_DELETE_EDGE, 1, 1, 2, 3, NULL, NULL, 0);
	EffectsV3Grouping_AddNode(g, EFFECT_DELETE_NODE, &B, 1, 2, NULL, NULL, 0);

	EffectsV3Records *r = _decode(g);

	const int want[] = {
		EFFECT_DELETE_EDGE, EFFECT_DELETE_NODE, EFFECT_DELETE_NODE
	};
	_assert_opcodes(r, want, 3, "two delete clauses sharing a relationship");

	TEST_ASSERT_(r->records[0].delete_edge.count == 2,
			"both edges share a shape and should be one record of two, got %u",
			r->records[0].delete_edge.count);

	EffectsV3_RecordsFree(r);
	EffectsV3Grouping_Free(g);
}

// A STAGED UPDATE IS EMITTED WHERE THE QUERY STAGED IT, NOT AT THE END
//
// An update is the one record type that cannot be filed on arrival - its shape
// is the entity's whole attribute set, which is not known until the query
// stops producing attributes for it - so it is folded into a group at the
// flush, after every create and delete has already been filed. Appending it
// there would put `MATCH (n:A) SET n.v = 1 DELETE n` on the wire as a delete
// followed by an update of a node that is gone.
//
// It carries the run it was staged in instead, and the stage order inside that
// run does the rest.
void test_effectsV3Group_anUpdateIsEmittedInTheRunItWasStagedIn(void) {
	{
		// MATCH (n:A) SET n.v = 1 DELETE n - one run, stages decide
		EffectsV3Grouping *g = EffectsV3Grouping_New();
		const LabelID A = 1;

		EffectsV3Grouping_StageUpdate(g, EFFECT_UPDATE_NODE, 1, &A, 1, 0, 7,
				SI_LongVal(1));
		EffectsV3Grouping_AddNode(g, EFFECT_DELETE_NODE, &A, 1, 1, NULL, NULL,
				0);

		EffectsV3Records *r = _decode(g);

		const int want[] = { EFFECT_UPDATE_NODE, EFFECT_DELETE_NODE };
		_assert_opcodes(r, want, 2, "an update of a node the query deletes");

		EffectsV3_RecordsFree(r);
		EffectsV3Grouping_Free(g);
	}
	{
		// MATCH (n:A) DELETE n CREATE (m:B) SET m.v = 1 - the update belongs to
		// the SECOND run, behind the create that made the node it names
		EffectsV3Grouping *g = EffectsV3Grouping_New();
		const LabelID A = 1, B = 2;

		EffectsV3Grouping_AddNode(g, EFFECT_DELETE_NODE, &A, 1, 1, NULL, NULL,
				0);
		EffectsV3Grouping_AddNode(g, EFFECT_CREATE_NODE, &B, 1, 1, NULL, NULL,
				0);
		EffectsV3Grouping_StageUpdate(g, EFFECT_UPDATE_NODE, 1, &B, 1, 0, 7,
				SI_LongVal(1));

		EffectsV3Records *r = _decode(g);

		const int want[] = {
			EFFECT_DELETE_NODE, EFFECT_CREATE_NODE, EFFECT_UPDATE_NODE
		};
		_assert_opcodes(r, want, 3, "an update of a node the query created "
				"over a freed id");

		EffectsV3_RecordsFree(r);
		EffectsV3Grouping_Free(g);
	}
}

// AN UPDATE DOES NOT FOLLOW AN ID ACROSS THE ENTITY THAT REUSED IT
//
// `MATCH (n:A) SET n.v = 1 DELETE n CREATE (m:B) SET m.w = 2` stages TWO
// updates under ONE id, for two different entities, because the create is
// handed the id the delete just freed. Staged per entity id alone they merge,
// and the merged record carries the FIRST entity's labels and the FIRST
// entity's run - so m's property is written to n, ahead of the delete that
// removes it, and never reaches m at all.
//
// WHAT MAKES THIS THE WORST OF THE FAMILY: nothing refuses it. The payload is
// well formed, every record applies, no divergence is logged and no resync
// follows - the replica is simply a property short. Measured on a live pair
// before this was keyed on the run: master {w: 9}, replica {}.
void test_effectsV3Group_anUpdateDoesNotCrossIntoAReusedId(void) {
	EffectsV3Grouping *g = EffectsV3Grouping_New();
	const LabelID A = 1, B = 2;

	// SET n.v = 1   (attribute 7 on the :A node at id 1)
	EffectsV3Grouping_StageUpdate(g, EFFECT_UPDATE_NODE, 1, &A, 1, 0, 7,
			SI_LongVal(1));

	// DELETE n, then CREATE (m:B) on the id it just freed
	EffectsV3Grouping_AddNode(g, EFFECT_DELETE_NODE, &A, 1, 1, NULL, NULL, 0);
	EffectsV3Grouping_AddNode(g, EFFECT_CREATE_NODE, &B, 1, 1, NULL, NULL, 0);

	// SET m.w = 2   (attribute 8, SAME id, different entity)
	EffectsV3Grouping_StageUpdate(g, EFFECT_UPDATE_NODE, 1, &B, 1, 0, 8,
			SI_LongVal(2));

	EffectsV3Records *r = _decode(g);

	const int want[] = {
		EFFECT_UPDATE_NODE, EFFECT_DELETE_NODE,
		EFFECT_CREATE_NODE, EFFECT_UPDATE_NODE
	};
	_assert_opcodes(r, want, 4, "an update either side of a reused id");

	// THE ATTRIBUTES ARE WHAT DISCRIMINATES, not the record count alone: a
	// merged entry emits one UPDATE_NODE holding BOTH attributes, under the
	// label set of the entity that is about to be deleted
	TEST_ASSERT_(r->records[0].update_node.n_attrs == 1 &&
			r->records[0].update_node.attr_ids[0] == 7,
			"the first update should carry n's attribute alone, got %u "
			"attribute(s) starting with %u",
			r->records[0].update_node.n_attrs,
			r->records[0].update_node.n_attrs
				? r->records[0].update_node.attr_ids[0] : 0);

	TEST_ASSERT_(r->records[3].update_node.n_attrs == 1 &&
			r->records[3].update_node.attr_ids[0] == 8,
			"the second update should carry m's attribute alone, got %u "
			"attribute(s) starting with %u",
			r->records[3].update_node.n_attrs,
			r->records[3].update_node.n_attrs
				? r->records[3].update_node.attr_ids[0] : 0);

	// and each names the labels of the entity it is actually about
	TEST_ASSERT_(r->records[0].update_node.n_labels == 1 &&
			r->records[0].update_node.labels[0] == A,
			"the first update should name :A");
	TEST_ASSERT_(r->records[3].update_node.n_labels == 1 &&
			r->records[3].update_node.labels[0] == B,
			"the second update should name :B");

	EffectsV3_RecordsFree(r);
	EffectsV3Grouping_Free(g);
}

// AND THE SAME ENTITY WRITTEN TWICE INSIDE ONE RUN IS STILL ONE RECORD
//
// The guard on the line above. Keying the staging index on the run splits an
// entity's updates whenever the payload switches; it must not split them when
// it does not, or every `SET n.a = 1, n.b = 2` becomes two records.
void test_effectsV3Group_twoWritesToOneEntityInOneRunAreOneRecord(void) {
	EffectsV3Grouping *g = EffectsV3Grouping_New();
	const LabelID A = 1;

	EffectsV3Grouping_StageUpdate(g, EFFECT_UPDATE_NODE, 1, &A, 1, 0, 7,
			SI_LongVal(1));
	// a create in between, which does NOT switch the run: nothing has deleted
	EffectsV3Grouping_AddNode(g, EFFECT_CREATE_NODE, &A, 1, 2, NULL, NULL, 0);
	EffectsV3Grouping_StageUpdate(g, EFFECT_UPDATE_NODE, 1, &A, 1, 0, 8,
			SI_LongVal(2));

	EffectsV3Records *r = _decode(g);

	const int want[] = { EFFECT_CREATE_NODE, EFFECT_UPDATE_NODE };
	_assert_opcodes(r, want, 2, "two writes to one entity in one run");

	TEST_ASSERT_(r->records[1].update_node.n_attrs == 2,
			"both attributes belong to one record, got %u",
			r->records[1].update_node.n_attrs);

	EffectsV3_RecordsFree(r);
	EffectsV3Grouping_Free(g);
}

// A PAYLOAD THAT ONLY CREATES IS STILL ONE RUN
//
// The guard against the opposite mistake. Runs cost a record every time they
// split, so a rule that opened one per clause, per batch or per shape would
// turn a bulk create into a record per entity and pass every test above while
// undoing what the grouping is for. Nothing here switches between creating and
// deleting, so nothing may split.
void test_effectsV3Group_aCreateOnlyPayloadIsOneRun(void) {
	EffectsV3Grouping *g = EffectsV3Grouping_New();
	const LabelID A = 1, B = 2;
	AttributeID attrs[] = { 7 };

	// the two shapes ALTERNATE, which is what `CREATE (:A), (:B)` over many
	// rows produces: a rule keyed on "the shape changed" splits here
	for(int i = 0; i < 50; i++) {
		SIValue v = SI_LongVal(i);
		EffectsV3Grouping_AddNode(g, EFFECT_CREATE_NODE, &A, 1, 2 * i, attrs,
				&v, 1);
		EffectsV3Grouping_AddNode(g, EFFECT_CREATE_NODE, &B, 1, 2 * i + 1,
				attrs, &v, 1);
	}

	TEST_ASSERT_(EffectsV3Grouping_RecordCount(g) == 2,
			"a hundred nodes over two shapes is two records, got %u",
			EffectsV3Grouping_RecordCount(g));

	EffectsV3Records *r = _decode(g);

	const int want[] = { EFFECT_CREATE_NODE, EFFECT_CREATE_NODE };
	_assert_opcodes(r, want, 2, "an alternating create-only payload");

	TEST_ASSERT_(r->records[0].create_node.count == 50 &&
			r->records[1].create_node.count == 50,
			"each shape should hold fifty nodes, got %u and %u",
			r->records[0].create_node.count, r->records[1].create_node.count);

	EffectsV3_RecordsFree(r);
	EffectsV3Grouping_Free(g);
}

// a vector statement's options survive the conversion, codes and all
void test_effectsV3Group_vectorOptions(void) {
	EffectsV3Grouping *g = EffectsV3Grouping_New();

	SIValue opts = SI_Map(3);
	Map_Add(&opts, SI_ConstStringVal("dimension"), SI_LongVal(8));
	Map_Add(&opts, SI_ConstStringVal("similarityFunction"),
			SI_ConstStringVal("cosine"));
	Map_Add(&opts, SI_ConstStringVal("M"), SI_LongVal(24));

	EffectsV3Grouping_AddIndexField(g, EFFECT_CREATE_INDEX, SCHEMA_NODE, 3,
			"P", INDEX_FLD_VECTOR, 12, "v", opts);

	size_t n;
	unsigned char *b = _encode(g, &n);
	size_t o = _options_offset(b, n);

	// the text half is written whatever the field type - five clear bytes
	for(int i = 0; i < 5; i++) {
		TEST_ASSERT_(b[o + i] == 0,
				"a vector statement states no text option, byte %d is %u",
				i, b[o + i]);
	}
	o += 5;

	// dimension has no presence byte of its own
	uint64_t dim;
	memcpy(&dim, b + o, 8);
	TEST_ASSERT_(dim == 8, "dimension should be 8, got %" PRIu64, dim);
	o += 8;

	TEST_ASSERT_(b[o] == 1, "M was stated");
	uint64_t m;
	memcpy(&m, b + o + 1, 8);
	TEST_ASSERT_(m == 24, "M should be 24, got %" PRIu64, m);
	o += 9;

	TEST_ASSERT_(b[o] == 0, "efConstruction was not stated");
	o += 1;
	TEST_ASSERT_(b[o] == 0, "efRuntime was not stated");
	o += 1;

	// the wire carries the CODE, not the name C's statement used
	TEST_ASSERT_(b[o] == 1, "similarityFunction was stated");
	uint64_t sim;
	memcpy(&sim, b + o + 1, 8);
	TEST_ASSERT_(sim == VecSimMetric_Cosine,
			"cosine is code %d on the wire, got %" PRIu64,
			VecSimMetric_Cosine, sim);
	o += 9;

	TEST_ASSERT_(o == n, "the vector block should end the record: %zu of %zu",
			o, n);

	free(b);
	Map_Free(opts);
	EffectsV3Grouping_Free(g);
}


TEST_LIST = {
	{ "EffectsV3Group:indexFieldsFoldIntoOneRecord",
		test_effectsV3Group_indexFieldsFoldIntoOneRecord },
	{ "EffectsV3Group:announcementsPrecedeTheIndexRecord",
		test_effectsV3Group_announcementsPrecedeTheIndexRecord },
	{ "EffectsV3Group:indexFieldNotAddedTwice",
		test_effectsV3Group_indexFieldNotAddedTwice },
	{ "EffectsV3Group:cchIsOneRecordOverEveryType",
		test_effectsV3Group_cchIsOneRecordOverEveryType },
	{ "EffectsV3Group:cchDoesNotAbsorbAnIndexField",
		test_effectsV3Group_cchDoesNotAbsorbAnIndexField },
	{ "EffectsV3Group:singleTypeCCHDoesNotStripOptions",
		test_effectsV3Group_singleTypeCCHDoesNotStripOptions },
	{ "EffectsV3Group:unstatedPhoneticIsAClearFlag",
		test_effectsV3Group_unstatedPhoneticIsAClearFlag },
	{ "EffectsV3Group:optionStatedAtItsDefaultIsPresent",
		test_effectsV3Group_optionStatedAtItsDefaultIsPresent },
	{ "EffectsV3Group:differingFieldOptionsSplit",
		test_effectsV3Group_differingFieldOptionsSplit },
	{ "EffectsV3Group:identicalFieldOptionsFold",
		test_effectsV3Group_identicalFieldOptionsFold },
	{ "EffectsV3Group:indexLevelOptionsStatedOnce",
		test_effectsV3Group_indexLevelOptionsStatedOnce },
	{ "EffectsV3Group:foldsIntoAStrippedRecord",
		test_effectsV3Group_foldsIntoAStrippedRecord },
	{ "EffectsV3Group:dropIndexFoldsWithNoOptions",
		test_effectsV3Group_dropIndexFoldsWithNoOptions },
	{ "EffectsV3Group:createPrecedesUpdate",
		test_effectsV3Group_createPrecedesUpdate },
	{ "EffectsV3Group:aDeleteIsNotOvertakenByACreate",
			test_effectsV3Group_aDeleteIsNotOvertakenByACreate },
	{ "EffectsV3Group:aCreateIsNotOvertakenByADelete",
			test_effectsV3Group_aCreateIsNotOvertakenByADelete },
	{ "EffectsV3Group:aDeletedEdgeIdIsNotOvertakenByACreate",
			test_effectsV3Group_aDeletedEdgeIdIsNotOvertakenByACreate },
	{ "EffectsV3Group:aLaterCreateDoesNotJoinAnEarlierRun",
			test_effectsV3Group_aLaterCreateDoesNotJoinAnEarlierRun },
	{ "EffectsV3Group:createdNodesPrecedeCreatedEdgesAcrossClauses",
			test_effectsV3Group_createdNodesPrecedeCreatedEdgesAcrossClauses },
	{ "EffectsV3Group:deletedEdgesPrecedeDeletedNodesAcrossClauses",
			test_effectsV3Group_deletedEdgesPrecedeDeletedNodesAcrossClauses },
	{ "EffectsV3Group:anUpdateIsEmittedInTheRunItWasStagedIn",
			test_effectsV3Group_anUpdateIsEmittedInTheRunItWasStagedIn },
	{ "EffectsV3Group:anUpdateDoesNotCrossIntoAReusedId",
			test_effectsV3Group_anUpdateDoesNotCrossIntoAReusedId },
	{ "EffectsV3Group:twoWritesToOneEntityInOneRunAreOneRecord",
			test_effectsV3Group_twoWritesToOneEntityInOneRunAreOneRecord },
	{ "EffectsV3Group:aCreateOnlyPayloadIsOneRun",
			test_effectsV3Group_aCreateOnlyPayloadIsOneRun },
	{ "EffectsV3Group:edgeDeletedBeforeNode",
		test_effectsV3Group_edgeDeletedBeforeNode },
	{ "EffectsV3Group:differentIndexesNeverFold",
		test_effectsV3Group_differentIndexesNeverFold },
	{ "EffectsV3Group:anotherIndexDoesNotStripOptions",
		test_effectsV3Group_anotherIndexDoesNotStripOptions },
	{ "EffectsV3Group:unstatedOptionIsNotARepeat",
		test_effectsV3Group_unstatedOptionIsNotARepeat },
	{ "EffectsV3Group:foldKeepsAnOptionNobodyElseStates",
		test_effectsV3Group_foldKeepsAnOptionNobodyElseStates },
	{ "EffectsV3Group:vectorOptions",
		test_effectsV3Group_vectorOptions },
	{ "EffectsV3Group:attributeOrderDoesNotSplit",
		test_effectsV3Group_attributeOrderDoesNotSplit },
	{ "EffectsV3Group:valuesFollowTheirAttributes",
		test_effectsV3Group_valuesFollowTheirAttributes },
	{ "EffectsV3Group:repeatedAttributeIsCollapsed",
		test_effectsV3Group_repeatedAttributeIsCollapsed },
	{ "EffectsV3Group:lastValueWins",
		test_effectsV3Group_lastValueWins },
	{ "EffectsV3Group:flushedGroupOutlivesTheArena",
		test_effectsV3Group_flushedGroupOutlivesTheArena },
	{ "EffectsV3Group:sameShapeIsOneRecord",
		test_effectsV3Group_sameShapeIsOneRecord },
	{ "EffectsV3Group:eachShapeComponentSplits",
		test_effectsV3Group_eachShapeComponentSplits },
	{ "EffectsV3Group:valuesAreNotPartOfTheShape",
		test_effectsV3Group_valuesAreNotPartOfTheShape },
	{ "EffectsV3Group:labelOrderDoesNotSplit",
		test_effectsV3Group_labelOrderDoesNotSplit },
	{ "EffectsV3Group:stagedUpdateLabelsAreASet",
		test_effectsV3Group_stagedUpdateLabelsAreASet },
	{ "EffectsV3Group:recordsAreSortedByKey",
		test_effectsV3Group_recordsAreSortedByKey },
	{ "EffectsV3Group:stagedValueIsCopiedFromTheCaller",
		test_effectsV3Group_stagedValueIsCopiedFromTheCaller },
	{ "EffectsV3Group:insertionOrderDoesNotChangeBytes",
		test_effectsV3Group_insertionOrderDoesNotChangeBytes },
	{ "EffectsV3Group:attributeAnnouncedOncePerPayload",
		test_effectsV3Group_attributeAnnouncedOncePerPayload },
	{ "EffectsV3Group:stagedUpdates",
		test_effectsV3Group_stagedUpdates },
	{ "EffectsV3Group:updatedEdgeHasNoEndpoints",
		test_effectsV3Group_updatedEdgeHasNoEndpoints },
	{ "EffectsV3Group:recordsWithNoEffectAreNotEmitted",
		test_effectsV3Group_recordsWithNoEffectAreNotEmitted },
	{ "EffectsV3Group:emptyBlocks",
		test_effectsV3Group_emptyBlocks },
	{ "EffectsV3Group:schemaAnnouncedOncePerTypeAndId",
		test_effectsV3Group_schemaAnnouncedOncePerTypeAndId },
	{ NULL, NULL }
};
