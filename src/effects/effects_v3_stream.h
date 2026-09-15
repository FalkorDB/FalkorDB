/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "effects_v3.h"

//------------------------------------------------------------------------------
// streaming decode: one record at a time
//------------------------------------------------------------------------------
//
// EffectsV3_Decode materialises the whole record set; this reads one record at
// a time, so peak memory is one record rather than the payload. A record is
// still applied as ONE bulk operation - only the set of records stops being
// materialised.
//
// The cost is that a refusal can now land after records 1..k are already
// applied. That is intended: a refusal makes Effects_Apply return false, which
// DivergenceGuard_OnFailure turns into a forced resync, and the resync
// overwrites the graph wholesale.
//
// THE INVARIANT TO PRESERVE: a mid-stream refusal must surface as Effects_Apply
// returning false. That is its only route to the guard - Effects_Apply has one
// caller and cmd_effect.c consumes its bool directly, with no status mapping in
// between. Never return success from a payload that stopped early.
//
// A CURSOR RATHER THAN A CALLBACK. There are two consumers - apply, which
// discards each record, and EffectsV3_Decode, which keeps them - and they differ
// only in what they do with a record once it exists. A callback inverts control
// for that, and forces an ownership contract on every caller; two copies of the
// loop invite the two from drifting apart on refusal boundaries. A cursor has
// neither problem: _ReadRecord is called in exactly one place, and each consumer
// writes the three lines it actually needs.

// reads records out of a v3 payload, one at a time
//
// Treat the fields as opaque; open it, pump it, close it.
typedef struct {
	FILE            *stream;  // over the caller's buffer, owned
	size_t           n;       // payload length
	EffectsV3Status  status;  // why the walk stopped
} EffectsV3Reader;

// open a payload for reading, validating the header
//
// On anything but EFFECTS_V3_OK the reader is not usable, but calling
// EffectsV3_ReaderClose on it is still safe and still required.
EffectsV3Status EffectsV3_ReaderOpen
(
	const char *buff,       // encoded payload
	size_t n,               // payload length
	EffectsV3Reader *r      // [output] reader
);

// read the next record
//
// Returns false at the end of the payload AND on a malformed one - call
// EffectsV3_ReaderStatus to tell those apart. Stopping on both is deliberate:
// a caller that ignores the status still stops reading rather than looping.
//
// OWNERSHIP: 'rec' belongs to the caller, which must EffectsV3_RecordFree it.
bool EffectsV3_ReaderNext
(
	EffectsV3Reader *r,
	EffectsV3Record *rec    // [output] the record just read
);

// why the walk stopped: OK at a clean end of payload, otherwise the refusal
EffectsV3Status EffectsV3_ReaderStatus
(
	const EffectsV3Reader *r
);

// release the reader; safe on one that failed to open
void EffectsV3_ReaderClose
(
	EffectsV3Reader *r
);

// free one decoded record
void EffectsV3_RecordFree
(
	EffectsV3Record *rec
);

// apply a single decoded record
//
// the per-record half of EffectsV3_Apply, which is now a loop over this
bool EffectsV3_ApplyRecord
(
	GraphContext *gc,
	const EffectsV3Record *rec
);
