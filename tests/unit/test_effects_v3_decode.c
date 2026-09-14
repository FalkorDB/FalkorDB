/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

// Covers the WHOLE-PAYLOAD decode form, EffectsV3_Decode.
//
// It exists because that form has no production caller any more. GRAPH.EFFECT
// streams - EffectsV3_DecodeEach, one record decoded, applied and freed at a
// time - and EffectsV3_Decode is kept only for EffectsV3_Encode's round trip,
// whose harness lives on a side branch. So nothing in this repository exercised
// it, and a function with no caller and no test is one nobody notices breaking
// until it breaks somewhere else.
//
// The payload here is INT-VALUED on purpose. Decode is free of any graph but not
// of module-global state: an interned string on the wire reaches
// _globals.string_pool through SI_InternStringVal, which only Globals_Init
// creates, and StringPool_rent guards it with an ASSERT that compiles to nothing
// in a release build - so a string payload here would segfault inside the
// decoder rather than fail. Integers take no such path, which is what lets this
// run with nothing but the allocator.

#include "src/util/rmalloc.h"
#include "src/effects/effects_v3.h"
#include "src/effects/effects_v3_stream.h"

#include <string.h>

void setup() {
	Alloc_Reset();
}

#define TEST_INIT setup();
#include "acutest.h"

// two CREATE_NODE records, one node each:
//
//   version 3, flags 0
//   CREATE_NODE count 1, labels [0], attrs [0], ids Range(1,1), values [11]
//   CREATE_NODE count 1, labels [0], attrs [0], ids Range(2,1), values [12]
static const unsigned char TWO_RECORDS[] = {
	0x03, 0x00,
	0x03, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
	0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x01,
	0x01, 0x00, 0x20, 0x00, 0x00, 0x0b, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
	0x00,
	0x03, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
	0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x02,
	0x01, 0x00, 0x20, 0x00, 0x00, 0x0c, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
	0x00
};

void test_decode_collects_every_record() {
	EffectsV3Records *records = NULL;

	EffectsV3Status status = EffectsV3_Decode((const char*)TWO_RECORDS,
			sizeof(TWO_RECORDS), &records);

	TEST_ASSERT(status  == EFFECTS_V3_OK);
	TEST_ASSERT(records != NULL);

	TEST_ASSERT(records->version == 3);
	TEST_ASSERT(records->flags   == 0);
	TEST_ASSERT(records->n       == 2);

	// THE CONTENTS, not just the count. The collecting wrapper takes ownership
	// of each record by copying the struct and zeroing the original, so a bug
	// there shows up as the right NUMBER of records holding freed or empty
	// fields - which a count-only assertion would sail straight past.
	for (uint32_t i = 0; i < records->n; i++) {
		const EffectsV3Record *rec = records->records + i;

		TEST_ASSERT(rec->opcode              == EFFECT_CREATE_NODE);
		TEST_ASSERT(rec->create_node.count   == 1);
		TEST_ASSERT(rec->create_node.n_labels == 1);
		TEST_ASSERT(rec->create_node.labels  != NULL);
		TEST_ASSERT(rec->create_node.labels[0] == 0);
		TEST_ASSERT(rec->create_node.n_attrs == 1);
		TEST_ASSERT(rec->create_node.attr_ids != NULL);
		TEST_ASSERT(rec->create_node.n_values == 1);
		TEST_ASSERT(rec->create_node.values  != NULL);
		TEST_ASSERT(rec->create_node.ids.n   == 1);
	}

	// the ids and values are per record, and in wire order
	TEST_ASSERT(records->records[0].create_node.ids.segments[0].kind
			== EFFECTS_V3_SEG_RANGE_ASCENDING);
	TEST_ASSERT(records->records[0].create_node.ids.segments[0]
			.range_ascending.base == 1);
	TEST_ASSERT(records->records[1].create_node.ids.segments[0]
			.range_ascending.base == 2);

	TEST_ASSERT(records->records[0].create_node.values[0].longval == 11);
	TEST_ASSERT(records->records[1].create_node.values[0].longval == 12);

	EffectsV3_RecordsFree(records);
}

void test_decode_refuses_a_truncated_payload() {
	EffectsV3Records *records = NULL;

	// cut into the last record's value
	EffectsV3Status status = EffectsV3_Decode((const char*)TWO_RECORDS,
			sizeof(TWO_RECORDS) - 3, &records);

	TEST_ASSERT(status  == EFFECTS_V3_TRUNCATED);

	// AND NOTHING IS HANDED BACK. The wrapper collected one whole record before
	// the second failed; returning it would hand the caller a payload prefix
	// that looks like a complete one.
	TEST_ASSERT(records == NULL);
}

void test_decode_refuses_a_future_version() {
	EffectsV3Records *records = NULL;
	unsigned char buf[sizeof(TWO_RECORDS)];

	memcpy(buf, TWO_RECORDS, sizeof(TWO_RECORDS));
	buf[0] = 4;  // a version above this build's read ceiling

	EffectsV3Status status = EffectsV3_Decode((const char*)buf, sizeof(buf),
			&records);

	TEST_ASSERT(status  == EFFECTS_V3_UNSUPPORTED_VERSION);
	TEST_ASSERT(records == NULL);
}

TEST_LIST = {
	{"decode_collects_every_record",      test_decode_collects_every_record},
	{"decode_refuses_a_truncated_payload", test_decode_refuses_a_truncated_payload},
	{"decode_refuses_a_future_version",   test_decode_refuses_a_future_version},
	{NULL, NULL}
};
