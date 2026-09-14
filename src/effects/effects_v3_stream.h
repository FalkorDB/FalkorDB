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
// EffectsV3_Decode materialises the whole record set and EffectsV3_Apply then
// walks it. That costs peak memory proportional to the PAYLOAD; this costs one
// record.
//
// The all-or-nothing property the whole-payload form appears to buy is not
// worth what it costs. It looks like "a malformed record at position 9 of 10 is
// caught before the graph is touched" - true, and worth less than it sounds,
// because both paths end at the same place. A refusal returns false, which
// DivergenceGuard_OnFailure turns into a forced full resync, and a resync
// overwrites the graph wholesale. Having touched it first costs nothing the
// resync does not already undo.
//
// A record is still applied as ONE bulk operation. Only the SET of records
// stops being materialised, so nothing regresses toward the per-entity graph
// calls that cost 91x on edges.
//
// THE RETURN PATH FOR A MID-STREAM REFUSAL, traced rather than assumed, because
// it is the one way this change could turn a loud failure into a silent one:
//
//     Effects_Apply              effects_apply.c        returns bool
//     its ONE caller             cmd_effect.c           bool ok = Effects_Apply
//     the !ok branch there       cmd_effect.c           DivergenceGuard_OnFailure
//
// Symbols rather than line numbers on purpose: three sessions have now quoted
// three different lines for the same function because each read a different
// checkout.
//
// Effects_Apply has exactly ONE caller and cmd_effect.c consumes its bool
// directly - there is no status mapping and no second return route between the
// two. So a refusal raised from inside the decode loop reaches the guard by the
// same path a whole-payload refusal does, PROVIDED it surfaces as
// Effects_Apply returning false. That is the invariant to preserve here: never
// return success from a payload that stopped early.
//
// The guard then either schedules a forced resync, or exits when the failure
// happened while loading from disk, where a resync cannot repair what is
// already loaded (DivergenceGuard_OnFailure, divergence_guard.c).

// called once per decoded record, in wire order
//
// OWNERSHIP: the record belongs to the decoder and is freed as soon as this
// returns. A callback that needs to keep it takes ownership by copying the
// struct and zeroing the original - freeing a zeroed record is a no-op, which
// is what makes the collecting wrapper below a few lines rather than a second
// decoder.
//
// return false to REFUSE the payload; decoding stops and the remaining bytes
// are not read
typedef bool (*EffectsV3RecordFn)
(
	EffectsV3Record *rec,  // the record just decoded
	void *ctx              // caller's context
);

// decode a payload, handing each record to 'fn' as it is read
//
// Returns EFFECTS_V3_OK when the payload decoded cleanly, whether or not 'fn'
// refused - a callback's verdict is the CALLER'S to carry in 'ctx', not a decode
// status. Decode statuses describe the bytes; they do not describe what the
// graph made of them, and conflating the two is how a refused-but-well-formed
// payload gets reported as corrupt.
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
