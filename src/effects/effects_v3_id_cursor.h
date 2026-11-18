/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "effects_v3.h"
#include "../util/roaring.h"
#include "../util/datablock/id_cursor.h"

#include <stdbool.h>
#include <stdint.h>

// an ASCENDING walk over an IdList, for claiming the ids it names
//
// NOT THE SAME WALK AS IdIter, and the difference is the point. IdIter yields
// WIRE order, which is what applying a record needs: row k's attributes belong
// to the k-th id as written. Claiming has no rows and no wire order - it needs
// the ids sorted, because the free list it maintains is only well defined in
// order and because a duplicate is then the id beside it. So the two walks
// coexist over one list.
//
// Direction disappears here. A descending range's base is its highest id and a
// descending set reads its bitmap backwards; ascending, both are the same ids
// the other way round, so this takes them from the other end rather than
// reversing anything.
//
// NOTHING IS EXPANDED. A segment keeps its own position - arithmetic for a
// Range or a Repeat, roaring's own iterator for a Set - and the walk takes the
// lowest head each step through a heap over the segments. The segment count is
// bounded by the payload, so the heap is too; the id count is not, and nothing
// here is sized by it.
//
// EVERYTHING THAT CAN FAIL FAILS AT INIT. The two walks must yield the same
// ids, so the bitmaps are deserialised once, here, and checked against the
// cardinality decode recorded - a set that is one id short is refused before
// any id has been claimed rather than discovered half way through the second
// walk, when there is no longer anything to refuse.
typedef struct {
	IdCursor base;  // MUST BE FIRST: the callbacks cast 'self' back to this

	const EffectsV3IdList *list;

	// where the walk stands in each segment
	struct {
		uint64_t              head;    // its current id, valid while left > 0
		uint64_t              left;    // ids still to yield, 'head' included
		roaring64_bitmap_t   *bitmap;  // Set kinds only, owned
		roaring64_iterator_t *iter;    // Set kinds only, owned
	} *seg;

	uint32_t *heap;    // segment indices, lowest head at the root
	uint32_t  n_heap;  // segments still holding ids

	// the wire-order walk, which is a different traversal of the same
	// segments and the same bitmaps - see EffectsV3IdCursor_RewindWire
	uint32_t wire_seg;   // segment being walked, in wire order
	uint64_t wire_left;  // ids left in it
	uint64_t wire_head;  // its current id
} EffectsV3IdCursor;

// prepare a walk over 'list', which must hold exactly 'count' ids
//
// false when a Set blob will not open, when one holds a different number of
// ids than decode recorded, or when the segments do not total 'count'. The
// cursor is safe to free either way.
bool EffectsV3IdCursor_Init
(
	EffectsV3IdCursor     *cur,
	const EffectsV3IdList *list,
	uint64_t               count
);

void EffectsV3IdCursor_Free
(
	EffectsV3IdCursor *cur
);

//------------------------------------------------------------------------------
// the same list, in wire order
//------------------------------------------------------------------------------
//
// ONE CURSOR HOLDS BOTH WALKS so that nothing can fail between them. Applying
// a create is claim-then-write: the claim needs ascending order, the write
// needs wire order because row k's attributes belong to the k-th id AS
// WRITTEN, and by the time the write runs the slots are already live. A
// SEPARATE walk there would have to open the Set blobs a second time, and an
// open that failed at that point would leave the batch claimed, half written
// and reported as whole - there is nothing left to refuse. Sharing the bitmaps
// this cursor already opened removes the question.
//
// Wire order is segment by segment, each in its own direction, which is what
// the format means by an ORDERED list. No heap: the segments are not merged.
//
// THE TWO WALKS RUN ONE AFTER THE OTHER, not interleaved. They share each Set
// segment's iterator, and entering a segment re-points it - ascending for the
// claim, from whichever end the segment states for the wire - so a walk
// started while the other is mid-segment would resume somewhere else. Applying
// a create claims the whole record and then writes the whole record, which is
// the order this is built for.

// restart the wire-order walk at the first segment's first id
void EffectsV3IdCursor_RewindWire
(
	EffectsV3IdCursor *cur
);

// next id in wire order; false when the list is spent
bool EffectsV3IdCursor_NextWire
(
	EffectsV3IdCursor *cur,
	uint64_t          *id
);

//------------------------------------------------------------------------------
// distinctness
//------------------------------------------------------------------------------

// does this list name 'count' DIFFERENT ids?
//
// Here rather than beside the appliers because this is a question about an
// EffectsV3IdList, and this file is where one is interpreted.
//
// NOT BY LOOKING FOR A REPEAT. A record's segments are not required to be
// sorted or disjoint, so a repeat has three places to hide - inside one Repeat
// segment, across two segments that overlap, and between a bitmap and a range
// covering the same id.
//
// TWO STAGES, because the cheap question settles every list a conforming
// encoder produces. Each segment is checked on its own - a Repeat of more than
// one is its own duplicate, a bitmap must hold as many ids as it claims, a
// range must fit in the id space - the positions are totalled against 'count',
// and the segments' spans are sorted and scanned for any overlap. Sorted by
// lowest id, an overlap anywhere shows up between neighbours, so one pass
// answers it. No overlap plus those per-segment facts IS distinctness, and
// nothing is built to prove it.
//
// An overlap is handed to a union of every id instead of settled in place.
// Settling it needs an EXACT answer, since a bitmap's span can enclose a
// neighbour it shares no id with, and once a pair can come back clean the scan
// loses the right to skip non-adjacent pairs - {0,50}, [10..20], {50,60} hides
// a repeated 50 that way. Comparing every pair still in reach is quadratic in
// a segment count that comes off the wire; the union reads each segment once
// and compares none of them.
//
// The create path does not need this: it walks the ids ascending to claim
// storage, and DataBlock_ClaimIds refuses a walk that does not strictly
// increase. The paths that do not claim have no such walk, and what a repeat
// costs them differs by record, which is worth stating because only one of
// them is silent:
//
//   DELETE_NODE   the slot is NOT deleted twice - DataBlock_DeleteItem returns
//                 early on an already-deleted item, so the free list and the
//                 item count are safe. What is counted twice is the LABEL
//                 STATISTIC: graph_delete_nodes.c walks each array entry's
//                 label row before any label matrix is cleared, so a repeated
//                 node increments encountered_labels twice and
//                 GraphStatistics_DecNodeCount over-subtracts. Permanent,
//                 silent, and apply reports success
//   UPDATE_NODE   two rows land on one entity, last write wins, success
//   UPDATE_EDGE   the (id -> row) table runs short of the count, so row 0 is
//                 applied and the record is THEN refused - a partial apply
//                 reported as a refusal
//   SET/REMOVE_LABELS
//                 nothing: the ids accumulate into a GrB_Vector before
//                 anything is applied, and setting an element twice is
//                 idempotent. The check is defensive on that path
//
// NOT FOR AN ENDPOINT COLUMN: ten edges off one node is that node's id ten
// times, so a repeat is the ordinary case on src and dst.
//
// IT ANSWERS BOTH HALVES ITSELF: that the segments describe exactly 'count'
// positions, and that those positions hold 'count' different ids. Decode
// establishes the first already, but this is reachable from anywhere a list
// can be built - the unit tests build them directly - so it does not inherit
// the precondition it depends on.
//
// COST. The first stage builds nothing for a Range or a Repeat, whatever it
// spans - a bulk delete of a million nodes is one Range and allocates only the
// per-segment array. A Set segment is deserialized there, because its span and
// its true size cannot be known without opening it; that is one blob at a
// time, freed before the next, and the blob is already on the wire.
//
// The union, when it runs, builds one roaring container per 65,536 ids spanned
// (roaring64 keys on the high 48 bits). Bounded by what the caller has already
// bounded: every call site refuses a count larger than the local graph, and
// the positions must total that count, so the span a record can ask for is
// bounded by entities that actually exist.
//
// False also when a Set segment's blob will not open, or when a range's span
// would leave the id space, which the caller cannot tell apart from a repeat:
// all three refuse the record, and none of them is a thing to carry on past.
bool EffectsV3IdList_AreDistinct
(
	const EffectsV3IdList *ids,  // list to measure
	uint64_t               count // ids it states it holds
);
