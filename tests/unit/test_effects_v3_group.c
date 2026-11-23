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
//   1. one record per (opcode, shape); shape is the attribute IDS, never the
//      values, plus labels for a node and relationship type for an edge
//   2. groups sorted by key before emission - a hash order differs between
//      runs, let alone between engines
//   3. label sets normalised ascending, so [7,8] and [8,7] are one shape
//   4. a new attribute announced ONCE per payload - the only rule that spans
//      groups, so the one a per-group implementation gets locally right and
//      globally wrong

#include "src/effects/effects_v3_group.h"
#include "src/effects/effects_v3_encode.h"
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

TEST_LIST = {
	{ "EffectsV3Group:attributeOrderDoesNotSplit",
		test_effectsV3Group_attributeOrderDoesNotSplit },
	{ "EffectsV3Group:valuesFollowTheirAttributes",
		test_effectsV3Group_valuesFollowTheirAttributes },
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
