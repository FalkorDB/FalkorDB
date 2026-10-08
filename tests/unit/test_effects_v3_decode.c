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

// walk a payload with the reader, recording what came back
typedef struct {
	uint32_t calls;       // how many records were handed over
	uint64_t ids[8];      // the first id of each, in delivery order
	EffectsV3Status status;
} _Walk;

// stop_at: refuse after this many records, 0 to read to the end
static void _walk
(
	const unsigned char *buf,
	size_t len,
	uint32_t stop_at,
	_Walk *w
) {
	memset(w, 0, sizeof(*w));

	EffectsV3Reader r;
	w->status = EffectsV3_ReaderOpen((const char*)buf, len, &r);

	if (w->status == EFFECTS_V3_OK) {
		EffectsV3Record rec;
		while (EffectsV3_ReaderNext(&r, &rec)) {
			if (w->calls < 8) {
				w->ids[w->calls] =
					rec.create_node.ids.segments[0].range_ascending.base;
			}
			w->calls++;

			// the caller owns each record the reader hands over
			EffectsV3_RecordFree(&rec);

			if (stop_at != 0 && w->calls == stop_at) {
				break;   // a consumer refusing mid-payload
			}
		}
		w->status = EffectsV3_ReaderStatus(&r);
	}

	EffectsV3_ReaderClose(&r);
}

void test_reader_hands_over_every_record_in_order() {
	_Walk w;
	_walk(THREE_RECORDS, sizeof(THREE_RECORDS), 0, &w);

	TEST_ASSERT(w.status == EFFECTS_V3_OK);
	TEST_ASSERT(w.calls  == 3);

	// in WIRE ORDER - records 9 and 10 are normatively ahead of anything
	// referencing the ids they introduce, so a reader that reordered would
	// break apply in a way no count could see
	TEST_ASSERT(w.ids[0] == 1);
	TEST_ASSERT(w.ids[1] == 2);
	TEST_ASSERT(w.ids[2] == 3);
}

void test_reader_delivers_k_records_before_a_truncation() {
	_Walk w;
	// cut into the third record's value: two decode whole, the third runs out
	_walk(THREE_RECORDS, sizeof(THREE_RECORDS) - 3, 0, &w);

	TEST_ASSERT(w.status == EFFECTS_V3_TRUNCATED);

	// K >= 1 IS THE WHOLE POINT. Before streaming, a truncated payload was
	// refused with nothing handed over; the count is what distinguishes the new
	// behaviour from the old, and it is why the refusal log line carries it.
	TEST_ASSERT(w.calls  == 2);
	TEST_ASSERT(w.ids[0] == 1);
	TEST_ASSERT(w.ids[1] == 2);
}

void test_reader_stops_where_the_consumer_stops() {
	_Walk w;
	_walk(THREE_RECORDS, sizeof(THREE_RECORDS), 2, &w);

	// THE THIRD RECORD IS NEVER READ. A consumer that refuses mid-payload stops
	// pumping the cursor, and nothing reads past that point.
	TEST_ASSERT(w.calls == 2);

	// AND THE STATUS IS STILL OK, which is deliberate. Decode statuses describe
	// the BYTES; these bytes were well formed. What the consumer made of them is
	// the consumer's to carry, and reporting a refused-but-well-formed payload
	// as corrupt sends an operator hunting a wire problem that does not exist.
	TEST_ASSERT(w.status == EFFECTS_V3_OK);
}

// a CREATE_INDEX: two fulltext fields with index-level language and stopwords.
//
// A DDL RECORD IS IN THE SWEEP DELIBERATELY. Records 1-8 are read by arms that
// exit through `goto fail` into _RecordFree; records 11-14 go through
// _ReadIndexRecord and _ReadConstraintRecord. Those used to return past that
// cleanup and leaked 2,369 bytes over a sweep. A sweep over CREATE_NODE
// payloads alone cannot call either function, so it cannot see what they leak.
static const unsigned char CREATE_INDEX_REC[] = {
	0x03, 0x00, 0x0b, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
	0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x50, 0x00,
	0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x06, 0x00, 0x00, 0x00,
	0x00, 0x00, 0x00, 0x00, 0x74, 0x69, 0x74, 0x6c, 0x65, 0x00, 0x01, 0x00,
	0x05, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x62, 0x6f, 0x64, 0x79,
	0x00, 0x01, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x65, 0x6e,
	0x67, 0x6c, 0x69, 0x73, 0x68, 0x00, 0x01, 0x02, 0x00, 0x00, 0x00, 0x00,
	0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x74,
	0x68, 0x65, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x61,
	0x00, 0x00, 0x00, 0x00
};

static void _sweep
(
	const unsigned char *buf,
	size_t len
) {
	// every prefix, including the empty one and the whole payload
	for (size_t k = 0; k <= len; k++) {
		EffectsV3Records *records = NULL;
		EffectsV3Status status = EffectsV3_Decode((const char*)buf, k, &records);

		if (status == EFFECTS_V3_OK) {
			// A PREFIX ENDING ON A RECORD BOUNDARY DECODES CLEANLY, and that is
			// a property of the format rather than a hole in the decoder: the
			// header carries no record count, so a payload cut at a boundary is
			// byte-for-byte a shorter valid payload. Sweeping 76 bytes of two
			// records accepts exactly k = 2, 39 and 76.
			//
			// So the assertion here is NOT "only the whole payload decodes".
			// That was this test's first version and it failed immediately -
			// correctly, against an expectation that was wrong.
			TEST_ASSERT(records != NULL);
			EffectsV3_RecordsFree(records);
		} else {
			// AND NOTHING IS HANDED BACK on a refusal - a caller that got a
			// record set here would be holding a payload prefix that looks
			// complete
			TEST_ASSERT(records == NULL);
		}
	}
}

void test_every_prefix_decodes_or_refuses_cleanly() {
	_sweep(TWO_RECORDS,   sizeof(TWO_RECORDS));
	_sweep(THREE_RECORDS, sizeof(THREE_RECORDS));
	_sweep(CREATE_INDEX_REC, sizeof(CREATE_INDEX_REC));
}

TEST_LIST = {
	{"decode_collects_every_record",      test_decode_collects_every_record},
	{"decode_refuses_a_truncated_payload", test_decode_refuses_a_truncated_payload},
	{"decode_refuses_a_future_version",   test_decode_refuses_a_future_version},
	{"reader_hands_over_every_record_in_order",
			test_reader_hands_over_every_record_in_order},
	{"reader_delivers_k_records_before_a_truncation",
			test_reader_delivers_k_records_before_a_truncation},
	{"reader_stops_where_the_consumer_stops",
			test_reader_stops_where_the_consumer_stops},
	{"every_prefix_decodes_or_refuses_cleanly",
			test_every_prefix_decodes_or_refuses_cleanly},
	{NULL, NULL}
};
