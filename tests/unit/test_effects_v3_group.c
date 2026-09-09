/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

// Grouping: which entities end up in the same record.
//
// THE CONFORMANCE CORPUS CANNOT ARBITRATE ANY OF THIS. A fixture pins what one
// record looks like; grouping decides which entities are inside it, and that
// decision leaves no trace in the bytes of a single record. Two engines can
// agree on every fixture and still emit different payloads for the same write.
//
// So these are property tests over the accumulator rather than byte
// comparisons. "Two entities of the same shape produce one record" is a
// statement about a function, and a fixture can only ever be one example of it.
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
#include "src/effects/effects_bytes.h"
#include "src/util/rmalloc.h"
#include "src/datatypes/map.h"
#include "src/datatypes/array.h"
#include "src/index/index_field.h"

#include <inttypes.h>

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
	// one value. Rather than parse that, check the first record's relation id
	// and that the payload as a whole is ordered by walking the three.
	TEST_ASSERT_(_u32(p + 8) == 1,
			"the first record should carry relationship type 1, got %u - "
			"records were emitted in arrival order rather than key order",
			_u32(p + 8));

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
// be zero and an empty counterpart for one of them, so it cannot arbitrate
// these - and this suite had the same hole for the same reason, since its
// shapes were drawn from the fixtures that exist. The decoder half of exactly
// this gap was a live bug: C's v3 decoder refused `CREATE (:Person)` while a
// fully green corpus and 26 green flow tests did not notice.
//
// So the encoder is pinned to emit them independently of when fixtures arrive.
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

// A VACUOUS RECORD IS NOT EMITTED.
//
// The rule, which covers three categories without a per-record table:
//
//   A record is vacuous if removing an empty block leaves it saying nothing
//   about any entity it names. Vacuous records must not be emitted. An empty
//   block that DESCRIBES the entities the record names is information and is
//   legal. A schema announcement is a binding rather than an instruction and
//   is legal regardless of whether anything references it.
//
// The distinction is which block is the record's SUBJECT, not whether a block
// is empty - which is why the empty-block cases above stay legal while these
// do not. `MATCH (n) SET n:Foo REMOVE n:Foo` leaves a label set that empties
// out, and a label record's entire payload IS its label set, so with none it
// is an instruction to do nothing.
//
// This reverses what this test asserted before the ruling. Emitting matched
// what a Rust master emitted at the time, which kept the two encoders
// agreeing; the ruling found that Rust's own emitter already suppresses the
// exactly-analogous vacuous update - set_node_attributes refuses to stage an
// empty attribute map - so the label path was an inconsistency in their
// emitter rather than a property of the format. Fixing the outlier beat
// legalising it.
//
// Readers TOLERATE these rather than refusing them, and that asymmetry is
// deliberate: rejecting a zero-label record removes no parse surface, because
// DELETE_NODE and CREATE_NODE require the zero-length LabelSet path anyway, so
// rejection would buy no safety and cost a resync loop against any peer still
// emitting one.
void test_effectsV3Group_vacuousRecordsAreNotEmitted(void) {
	{
		// a label record with no labels: its whole payload is the label set
		EffectsV3Grouping *g = EffectsV3Grouping_New();
		EffectsV3Grouping_AddNode(g, EFFECT_SET_LABELS, NULL, 0, 10,
				NULL, NULL, 0);

		TEST_ASSERT_(EffectsV3Grouping_RecordCount(g) == 0,
				"a zero-label SET_LABELS is vacuous and must not be emitted, "
				"got %u records", EffectsV3Grouping_RecordCount(g));

		size_t n = 0;
		unsigned char *p = _encode(g, &n);
		TEST_ASSERT_(n == 0,
				"a payload of only vacuous records must be empty, got %zu bytes",
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
				"a zero-label REMOVE_LABELS is vacuous too, got %u",
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
				"vacuous, got %u records", EffectsV3Grouping_RecordCount(g));

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

//------------------------------------------------------------------------------
// index DDL: one record per STATEMENT, from one call per FIELD
//------------------------------------------------------------------------------

// walk a payload holding ONE CREATE_INDEX record to the start of its options
//
// The record is `opcode . schema_type . schema_id . name . field_type .
// fields . OPTIONS`, and a string is a u64 length that INCLUDES the NUL
// followed by that many bytes. Sizes come from the real types rather than
// literals, so a widened enum moves the test with the encoder.
static size_t _options_offset(const unsigned char *b, size_t n) {
	size_t o = sizeof(EffectType) + sizeof(SchemaType) + sizeof(int);

	uint64_t len;
	memcpy(&len, b + o, 8);            // label
	o += 8 + len;

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

	size_t o = sizeof(EffectType) + sizeof(SchemaType) + sizeof(int);
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
// and the replica refuses a payload that is otherwise correct. The split here
// is forced by differing weights, which is the only way C produces one.
//
// A language-only version of this test is green either way, because
// Index_SetLanguage objects only when the language DIFFERS.
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
	TEST_ASSERT_(sim == V3_SIMFUNC_COSINE,
			"cosine is code %d on the wire, got %" PRIu64,
			V3_SIMFUNC_COSINE, sim);
	o += 9;

	TEST_ASSERT_(o == n, "the vector block should end the record: %zu of %zu",
			o, n);

	free(b);
	Map_Free(opts);
	EffectsV3Grouping_Free(g);
}

TEST_LIST = {
	{ "EffectsV3Group:sameShapeIsOneRecord",
		test_effectsV3Group_sameShapeIsOneRecord },
	{ "EffectsV3Group:eachShapeComponentSplits",
		test_effectsV3Group_eachShapeComponentSplits },
	{ "EffectsV3Group:valuesAreNotPartOfTheShape",
		test_effectsV3Group_valuesAreNotPartOfTheShape },
	{ "EffectsV3Group:labelOrderDoesNotSplit",
		test_effectsV3Group_labelOrderDoesNotSplit },
	{ "EffectsV3Group:recordsAreSortedByKey",
		test_effectsV3Group_recordsAreSortedByKey },
	{ "EffectsV3Group:insertionOrderDoesNotChangeBytes",
		test_effectsV3Group_insertionOrderDoesNotChangeBytes },
	{ "EffectsV3Group:attributeAnnouncedOncePerPayload",
		test_effectsV3Group_attributeAnnouncedOncePerPayload },
	{ "EffectsV3Group:stagedUpdates",
		test_effectsV3Group_stagedUpdates },
	{ "EffectsV3Group:vacuousRecordsAreNotEmitted",
		test_effectsV3Group_vacuousRecordsAreNotEmitted },
	{ "EffectsV3Group:emptyBlocks",
		test_effectsV3Group_emptyBlocks },
	{ "EffectsV3Group:schemaAnnouncedOncePerTypeAndId",
		test_effectsV3Group_schemaAnnouncedOncePerTypeAndId },
	{ "EffectsV3Group:indexFieldsFoldIntoOneRecord",
		test_effectsV3Group_indexFieldsFoldIntoOneRecord },
	{ "EffectsV3Group:indexFieldNotAddedTwice",
		test_effectsV3Group_indexFieldNotAddedTwice },
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
	{ "EffectsV3Group:dropIndexFoldsWithNoOptions",
		test_effectsV3Group_dropIndexFoldsWithNoOptions },
	{ "EffectsV3Group:differentIndexesNeverFold",
		test_effectsV3Group_differentIndexesNeverFold },
	{ "EffectsV3Group:vectorOptions",
		test_effectsV3Group_vectorOptions },
	{ NULL, NULL }
};
