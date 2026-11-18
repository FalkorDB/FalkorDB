/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include <stdint.h>
#include <stdbool.h>

// a re-walkable sequence of ids, supplied by the caller
//
// DataBlock_ClaimIds needs the ids TWICE - once to decide whether the whole
// batch is claimable, once to claim it - and must not hold an array
// proportional to how many there are. A wire record states its count in four
// bytes and describes the ids themselves in about seven, so an array indexed
// by that count is an allocation the payload does not pay for. So the caller
// supplies the walk instead of the ids, and claiming restarts it rather than
// remembering what it saw.
//
// NON-DECREASING is what a cursor promises. STRICTLY ascending is what
// DataBlock_ClaimIds demands, and the difference between the two is where
// duplicates are caught:
//
//   * a cursor cannot promise strictness. EffectsV3IdCursor walks a decoded
//     wire record, where a Repeat segment names one id several times and two
//     segments may overlap - shapes a peer can put on the wire, malformed or
//     not, and which this interface has no way to forbid
//   * so the claim's adjacent-id comparison is not a second opinion on a
//     promise already kept above. IT IS THE DUPLICATE CHECK. Deleting it as
//     redundant does not tighten anything; it removes the only place a
//     repeated id is refused
//
// ORDER is what both sides rest on, and that much a cursor does promise:
//
//   * claiming has to push every slot it skips past the high-water mark onto
//     the free list EXACTLY ONCE, which is only well defined in order
//   * in order, a repeated id is the id beside it - which is what makes the
//     duplicate check exact. Comparing id RANGES instead would be exact for a
//     run but not for a bitmap, whose bounds can overlap while its members do
//     not, and would refuse a payload that is perfectly good
//
// So a cursor over an ordered-but-not-sorted source sorts it. What it must not
// do is expand it: the whole point is that nothing here is proportional to
// 'count'.
typedef struct IdCursor IdCursor;

struct IdCursor {
	// restart the walk at its first id
	//
	// called once per pass, including before the first. BOTH PASSES MUST YIELD
	// THE SAME IDS: claiming decides on the first and acts on the second, so a
	// cursor that disagrees with itself claims a different set from the one it
	// was cleared for. That is a bug in the cursor, not something a caller can
	// recover from, so a cursor resolves whatever might vary - deserialising,
	// allocating - at construction time, where it can still fail.
	void (*reset) (IdCursor *self);

	// next id, never below the one before it; false when the walk is spent
	//
	// NOT guaranteed to be above it. A caller that needs strictness checks for
	// it and decides what a repeat means to it - for claiming, divergence.
	bool (*next) (IdCursor *self, uint64_t *id);

	// how many ids the walk states it holds, known before it starts
	//
	// claiming checks the walk against this rather than trusting either: a
	// source that yields fewer ids than it claims would otherwise leave the
	// batch half-claimed and reported as whole.
	uint64_t count;
};
