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


//------------------------------------------------------------------------------
// streaming: the behaviour that did not exist before
//------------------------------------------------------------------------------
//
// THIS IS THE HALF NOTHING ELSE PINS. Streaming decode introduced exactly one
// new behaviour - a refusal raised AFTER k records have already been handed
// over, with k >= 1. The consequence of that refusal (it still reaches
// DivergenceGuard_OnFailure, sync_full still rises, the replica still
// converges) belongs to a flow test and already has machinery behind it. The
// cause does not: no server is involved, and if the driver ever started
// refusing before delivering anything, every flow test would still pass.
//
// A unit test cannot reach the guard - there is no RedisModuleCtx here - so it
// deliberately does not try. It pins that the callback ran, how many times, and
// that decoding stopped when it said to.

// three CREATE_NODE records, one node each, ids 1 2 3 and values 11 12 13
static const unsigned char THREE_RECORDS[] = {
	0x03, 0x00, 0x03, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00,
	0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
	0x00, 0x01, 0x01, 0x00, 0x20, 0x00, 0x00, 0x0b, 0x00, 0x00, 0x00, 0x00,
	0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
	0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00,
	0x00, 0x00, 0x02, 0x01, 0x00, 0x20, 0x00, 0x00, 0x0c, 0x00, 0x00, 0x00,
	0x00, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
	0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00,
	0x00, 0x00, 0x00, 0x03, 0x01, 0x00, 0x20, 0x00, 0x00, 0x0d, 0x00, 0x00,
	0x00, 0x00, 0x00, 0x00, 0x00
};

typedef struct {
	uint32_t calls;       // how many records were handed over
	uint64_t ids[8];      // the first id of each, in delivery order
	uint32_t refuse_at;   // return false on this call number, 0 to never
} _CountCtx;

static bool _CountingCb
(
	EffectsV3Record *rec,
	void *ctx
) {
	_CountCtx *c = (_CountCtx*)ctx;

	if (c->calls < 8) {
		c->ids[c->calls] = rec->create_node.ids.segments[0].range_ascending.base;
	}
	c->calls++;

	return !(c->refuse_at != 0 && c->calls == c->refuse_at);
}

void test_stream_hands_over_every_record_in_order() {
	_CountCtx c = { 0 };

	EffectsV3Status status = EffectsV3_DecodeEach((const char*)THREE_RECORDS,
			sizeof(THREE_RECORDS), _CountingCb, &c);

	TEST_ASSERT(status   == EFFECTS_V3_OK);
	TEST_ASSERT(c.calls  == 3);

	// in WIRE ORDER - records 9 and 10 are normatively ahead of anything
	// referencing the ids they introduce, so a driver that reordered would
	// break apply in a way no count could see
	TEST_ASSERT(c.ids[0] == 1);
	TEST_ASSERT(c.ids[1] == 2);
	TEST_ASSERT(c.ids[2] == 3);
}

void test_stream_delivers_k_records_before_a_truncation() {
	_CountCtx c = { 0 };

	// cut into the third record's value: two records decode whole, the third
	// runs out of bytes
	EffectsV3Status status = EffectsV3_DecodeEach((const char*)THREE_RECORDS,
			sizeof(THREE_RECORDS) - 3, _CountingCb, &c);

	TEST_ASSERT(status == EFFECTS_V3_TRUNCATED);

	// K >= 1 IS THE WHOLE POINT. Before streaming, a truncated payload was
	// refused with nothing handed over; the count is what distinguishes the new
	// behaviour from the old, and it is why the refusal log line carries it.
	TEST_ASSERT(c.calls == 2);
	TEST_ASSERT(c.ids[0] == 1);
	TEST_ASSERT(c.ids[1] == 2);
}

void test_stream_stops_when_the_callback_refuses() {
	_CountCtx c = { 0 };
	c.refuse_at = 2;   // refuse the second record

	EffectsV3Status status = EffectsV3_DecodeEach((const char*)THREE_RECORDS,
			sizeof(THREE_RECORDS), _CountingCb, &c);

	// THE THIRD RECORD IS NEVER READ. A driver that kept going would apply a
	// record after the caller had already refused the payload.
	TEST_ASSERT(c.calls == 2);

	// AND THE STATUS IS STILL OK, which is deliberate rather than an oversight.
	// Decode statuses describe the BYTES; these bytes were well formed. What
	// the caller made of them is the caller's to carry, and reporting a
	// refused-but-well-formed payload as corrupt sends an operator hunting a
	// wire problem that does not exist.
	TEST_ASSERT(status == EFFECTS_V3_OK);
}

TEST_LIST = {
	{"decode_collects_every_record",      test_decode_collects_every_record},
	{"decode_refuses_a_truncated_payload", test_decode_refuses_a_truncated_payload},
	{"decode_refuses_a_future_version",   test_decode_refuses_a_future_version},
	{"stream_hands_over_every_record_in_order",
			test_stream_hands_over_every_record_in_order},
	{"stream_delivers_k_records_before_a_truncation",
			test_stream_delivers_k_records_before_a_truncation},
	{"stream_stops_when_the_callback_refuses",
			test_stream_stops_when_the_callback_refuses},
	{NULL, NULL}
};
