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
// EffectsV3_Decode materialises the whole record set and EffectsV3_Apply walks
// it; this decodes, applies and frees one record at a time, so peak memory is
// one record rather than the payload. A record is still applied as ONE bulk
// operation - only the set of records stops being materialised.
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

// called once per decoded record, in wire order
//
// OWNERSHIP: the record is freed as soon as this returns. A callback that keeps
// it takes ownership by copying the struct and zeroing the original - freeing a
// zeroed record is a no-op.
//
// return false to REFUSE the payload; decoding stops
typedef bool (*EffectsV3RecordFn)
(
	EffectsV3Record *rec,  // the record just decoded
	void *ctx              // caller's context
);

// decode a payload, handing each record to 'fn' as it is read
//
// Returns EFFECTS_V3_OK when the BYTES decoded cleanly, whether or not 'fn'
// refused - a callback's verdict belongs in 'ctx', not in a decode status.
// Reporting a refused-but-well-formed payload as corrupt sends an operator
// hunting a wire problem that does not exist.
EffectsV3Status EffectsV3_DecodeEach
(
	const char *buff,       // encoded payload
	size_t n,               // payload length
	EffectsV3RecordFn fn,   // called per record
	void *ctx               // passed through to 'fn'
);

// apply a single decoded record
//
// the per-record half of EffectsV3_Apply, which is now a loop over this
bool EffectsV3_ApplyRecord
(
	GraphContext *gc,
	const EffectsV3Record *rec
);
