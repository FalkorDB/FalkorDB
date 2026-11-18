/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "effects_v3_id_cursor.h"

#include "../RG.h"
#include "../util/rmalloc.h"

//------------------------------------------------------------------------------
// the heap over segments
//------------------------------------------------------------------------------
//
// A HEAP RATHER THAN A SCAN of the segment heads. Both are bounded by the
// payload, but the scan is paid per id: a record describing 67 million ids
// across a thousand segments would compare a thousand heads 67 million times.
// The heap makes it one comparison per level.

static inline uint64_t _Head
(
	const EffectsV3IdCursor *cur,
	uint32_t slot
) {
	return cur->seg[cur->heap[slot]].head ;
}

static void _SiftDown
(
	EffectsV3IdCursor *cur,
	uint32_t parent
) {
	while (true) {
		// 64-bit so a child index cannot wrap on the way to the bounds check
		// that would have rejected it
		const uint64_t left  = (uint64_t)parent * 2 + 1 ;
		const uint64_t right = left + 1 ;

		uint32_t lowest = parent ;

		if (left < cur->n_heap &&
			_Head (cur, (uint32_t)left) < _Head (cur, lowest)) {
			lowest = (uint32_t)left ;
		}
		if (right < cur->n_heap &&
			_Head (cur, (uint32_t)right) < _Head (cur, lowest)) {
			lowest = (uint32_t)right ;
		}

		if (lowest == parent) {
			return ;
		}

		const uint32_t swap = cur->heap[parent] ;
		cur->heap[parent]   = cur->heap[lowest] ;
		cur->heap[lowest]   = swap ;

		parent = lowest ;
	}
}

//------------------------------------------------------------------------------
// per-segment position
//------------------------------------------------------------------------------

// how many ids a segment holds
static uint64_t _SegmentCardinality
(
	const EffectsV3IdListSegment *s
) {
	switch (s->kind) {
		case EFFECTS_V3_SEG_RANGE_ASCENDING:  return s->range_ascending.len ;
		case EFFECTS_V3_SEG_RANGE_DESCENDING: return s->range_descending.len ;
		case EFFECTS_V3_SEG_REPEAT:           return s->repeat.count ;
		case EFFECTS_V3_SEG_SET_ASCENDING:    return s->set_ascending.cardinality ;
		case EFFECTS_V3_SEG_SET_DESCENDING:   return s->set_descending.cardinality ;
		default:                              return 0 ;
	}
}

// put a segment back at its LOWEST id
static void _RewindSegment
(
	EffectsV3IdCursor *cur,
	uint32_t i
) {
	const EffectsV3IdListSegment *s = cur->list->segments + i ;

	cur->seg[i].left = _SegmentCardinality (s) ;
	cur->seg[i].head = 0 ;

	if (cur->seg[i].left == 0) {
		// a zero-length segment is representable - decode only bounds a range
		// when its length is non-zero - and it simply yields nothing
		return ;
	}

	switch (s->kind) {
		case EFFECTS_V3_SEG_RANGE_ASCENDING:
			cur->seg[i].head = s->range_ascending.base ;
			break ;

		case EFFECTS_V3_SEG_RANGE_DESCENDING:
			// its base is the HIGHEST id, so ascending starts at the far end.
			// Decode has already checked the span does not run below zero.
			cur->seg[i].head =
				s->range_descending.base - (s->range_descending.len - 1) ;
			break ;

		case EFFECTS_V3_SEG_REPEAT:
			cur->seg[i].head = s->repeat.id ;
			break ;

		default:
			// both Set directions read the same bitmap low to high
			roaring64_iterator_reinit (cur->seg[i].bitmap, cur->seg[i].iter) ;
			cur->seg[i].head = roaring64_iterator_value (cur->seg[i].iter) ;
			break ;
	}
}

//------------------------------------------------------------------------------
// the IdCursor interface
//------------------------------------------------------------------------------

static void _Reset
(
	IdCursor *self
) {
	EffectsV3IdCursor *cur = (EffectsV3IdCursor *)self ;

	cur->n_heap = 0 ;

	for (uint32_t i = 0 ; i < cur->list->n ; i++) {
		_RewindSegment (cur, i) ;
		if (cur->seg[i].left > 0) {
			cur->heap[cur->n_heap++] = i ;
		}
	}

	for (uint32_t i = cur->n_heap / 2 ; i > 0 ; i--) {
		_SiftDown (cur, i - 1) ;
	}
}

static bool _Next
(
	IdCursor *self,
	uint64_t *id
) {
	EffectsV3IdCursor *cur = (EffectsV3IdCursor *)self ;

	if (cur->n_heap == 0) {
		return false ;
	}

	const uint32_t i = cur->heap[0] ;
	*id = cur->seg[i].head ;

	cur->seg[i].left-- ;

	if (cur->seg[i].left == 0) {
		// that segment is spent: the last entry takes the root
		cur->heap[0] = cur->heap[--cur->n_heap] ;
		if (cur->n_heap > 0) {
			_SiftDown (cur, 0) ;
		}
		return true ;
	}

	switch (cur->list->segments[i].kind) {
		case EFFECTS_V3_SEG_RANGE_ASCENDING:
		case EFFECTS_V3_SEG_RANGE_DESCENDING:
			cur->seg[i].head++ ;
			break ;

		case EFFECTS_V3_SEG_REPEAT:
			// the same id again. A create refuses the repeat as a duplicate,
			// which is the claim's job and not this walk's
			break ;

		default:
			// 'left' came from the bitmap's own cardinality at init, so a
			// value is still there to read
			roaring64_iterator_advance (cur->seg[i].iter) ;
			cur->seg[i].head = roaring64_iterator_value (cur->seg[i].iter) ;
			break ;
	}

	_SiftDown (cur, 0) ;
	return true ;
}

//------------------------------------------------------------------------------
// construction
//------------------------------------------------------------------------------

bool EffectsV3IdCursor_Init
(
	EffectsV3IdCursor     *cur,
	const EffectsV3IdList *list,
	uint64_t               count
) {
	ASSERT (cur  != NULL) ;
	ASSERT (list != NULL) ;

	cur->base.reset = _Reset ;
	cur->base.next  = _Next ;
	cur->base.count = count ;
	cur->list       = list ;
	cur->seg        = NULL ;
	cur->heap       = NULL ;
	cur->n_heap     = 0 ;
	cur->wire_seg   = 0 ;
	cur->wire_left  = 0 ;
	cur->wire_head  = 0 ;

	const uint32_t n = list->n ;

	if (n == 0) {
		// no segments describes no ids
		return count == 0 ;
	}

	// calloc so a partially built cursor is still safe to free
	cur->seg  = rm_calloc (n, sizeof (*cur->seg)) ;
	cur->heap = rm_malloc (sizeof (uint32_t) * n) ;

	uint64_t total = 0 ;

	for (uint32_t i = 0 ; i < n ; i++) {
		const EffectsV3IdListSegment *s = list->segments + i ;
		const uint64_t card = _SegmentCardinality (s) ;

		// the greatest id this segment names, or 0 for an empty one - which
		// names nothing and so cannot name the sentinel
		uint64_t highest = 0 ;

		switch (s->kind) {
			case EFFECTS_V3_SEG_RANGE_ASCENDING:
				// the mirror of the descending check below: the walk ends at
				// base + (len - 1), which would run past the top of the id
				// space. Decode checks this too, and it is re-checked for the
				// same reason - a list can be built by something other than
				// the decoder, and the tests already do exactly that.
				if (card > 0) {
					if (s->range_ascending.base > UINT64_MAX - (card - 1)) {
						EffectsV3IdCursor_Free (cur) ;
						return false ;
					}
					highest = s->range_ascending.base + (card - 1) ;
				}
				break ;

			case EFFECTS_V3_SEG_REPEAT:
				if (card > 0) {
					highest = s->repeat.id ;
				}
				break ;

			case EFFECTS_V3_SEG_RANGE_DESCENDING:
				// the ascending walk starts at base - (len - 1), which would
				// underflow for a range running below zero. Decode checks this
				// too; it is re-checked for the reason the Set cardinality is
				// - a list can be built by something other than the decoder,
				// and the tests already do exactly that.
				if (card > 0 && card - 1 > s->range_descending.base) {
					EffectsV3IdCursor_Free (cur) ;
					return false ;
				}
				// a descending range's base IS its highest id
				if (card > 0) {
					highest = s->range_descending.base ;
				}
				break ;

			case EFFECTS_V3_SEG_SET_ASCENDING:
			case EFFECTS_V3_SEG_SET_DESCENDING: {
				const bool desc = (s->kind == EFFECTS_V3_SEG_SET_DESCENDING) ;
				const unsigned char *blob =
					desc ? s->set_descending.blob : s->set_ascending.blob ;
				const uint32_t blob_n =
					desc ? s->set_descending.n : s->set_ascending.n ;

				// DESERIALISED ONCE, AND HERE. Both walks must yield the same
				// ids, which only holds if both read the same bitmap object -
				// and this is the last moment a failure can still be a refusal
				// rather than a half-claimed batch.
				cur->seg[i].bitmap =
					roaring64_bitmap_portable_deserialize_safe (
							(const char *)blob, blob_n) ;
				if (cur->seg[i].bitmap == NULL) {
					EffectsV3IdCursor_Free (cur) ;
					return false ;
				}

				// the blob must hold what decode counted it as holding.
				// Checked against the bitmap rather than trusted: the walk
				// bounds itself on this number, and a set one id short would
				// otherwise read an exhausted iterator.
				if (roaring64_bitmap_get_cardinality (cur->seg[i].bitmap)
						!= card) {
					EffectsV3IdCursor_Free (cur) ;
					return false ;
				}

				cur->seg[i].iter =
					roaring64_iterator_create (cur->seg[i].bitmap) ;
				if (cur->seg[i].iter == NULL) {
					EffectsV3IdCursor_Free (cur) ;
					return false ;
				}

				if (card > 0) {
					highest = roaring64_bitmap_maximum (cur->seg[i].bitmap) ;
				}
				break ;
			}

			default:
				// a kind this build does not know
				EffectsV3IdCursor_Free (cur) ;
				return false ;
		}

		// NO ENTITY CARRIES THE SENTINEL, so a segment that names it is
		// refused here rather than downstream. This is the cursor's half of
		// that rule: the claim refuses it for the entity-id column
		// (DataBlock_ClaimIds), but an edge record's endpoint columns are
		// never claimed - they are walked in wire order and handed to
		// Graph_HasNode, whose own contract says the sentinel is invalid and
		// whose debug assertion is the only thing that says so. Refusing it at
		// the one place that sees every segment of every column covers both.
		if (highest == UINT64_MAX) {   // == INVALID_ENTITY_ID
			EffectsV3IdCursor_Free (cur) ;
			return false ;
		}

		if (card > UINT64_MAX - total) {
			EffectsV3IdCursor_Free (cur) ;
			return false ;
		}
		total += card ;
	}

	// the segments must total what the record says the list holds
	if (total != count) {
		EffectsV3IdCursor_Free (cur) ;
		return false ;
	}

	_Reset (&cur->base) ;

	return true ;
}

void EffectsV3IdCursor_Free
(
	EffectsV3IdCursor *cur
) {
	if (cur == NULL) {
		return ;
	}

	if (cur->seg != NULL) {
		for (uint32_t i = 0 ; i < cur->list->n ; i++) {
			if (cur->seg[i].iter != NULL) {
				roaring64_iterator_free (cur->seg[i].iter) ;
				cur->seg[i].iter = NULL ;
			}
			if (cur->seg[i].bitmap != NULL) {
				roaring64_bitmap_free (cur->seg[i].bitmap) ;
				cur->seg[i].bitmap = NULL ;
			}
		}
		rm_free (cur->seg) ;
		cur->seg = NULL ;
	}

	if (cur->heap != NULL) {
		rm_free (cur->heap) ;
		cur->heap = NULL ;
	}

	cur->n_heap = 0 ;
}

//------------------------------------------------------------------------------
// the same list, in wire order
//------------------------------------------------------------------------------
//
// Segment by segment, each read in its own direction. The ascending walk above
// merges the segments; this one does not, because wire order IS the order they
// were written in.

// take up position at the start of the current segment
static void _WireEnterSegment
(
	EffectsV3IdCursor *cur
) {
	if (cur->wire_seg >= cur->list->n) {
		cur->wire_left = 0 ;
		return ;
	}

	const EffectsV3IdListSegment *s = cur->list->segments + cur->wire_seg ;

	cur->wire_left = _SegmentCardinality (s) ;
	cur->wire_head = 0 ;

	if (cur->wire_left == 0) {
		return ;
	}

	switch (s->kind) {
		case EFFECTS_V3_SEG_RANGE_ASCENDING:
			cur->wire_head = s->range_ascending.base ;
			break ;

		case EFFECTS_V3_SEG_RANGE_DESCENDING:
			// its base is its FIRST id, and it counts down from there
			cur->wire_head = s->range_descending.base ;
			break ;

		case EFFECTS_V3_SEG_REPEAT:
			cur->wire_head = s->repeat.id ;
			break ;

		case EFFECTS_V3_SEG_SET_DESCENDING:
			roaring64_iterator_reinit_last (cur->seg[cur->wire_seg].bitmap,
					cur->seg[cur->wire_seg].iter) ;
			cur->wire_head =
				roaring64_iterator_value (cur->seg[cur->wire_seg].iter) ;
			break ;

		default:
			roaring64_iterator_reinit (cur->seg[cur->wire_seg].bitmap,
					cur->seg[cur->wire_seg].iter) ;
			cur->wire_head =
				roaring64_iterator_value (cur->seg[cur->wire_seg].iter) ;
			break ;
	}
}

// step to the segment's next id; only called while it still holds one
static void _WireAdvance
(
	EffectsV3IdCursor *cur
) {
	switch (cur->list->segments[cur->wire_seg].kind) {
		case EFFECTS_V3_SEG_RANGE_ASCENDING:
			cur->wire_head++ ;
			break ;

		case EFFECTS_V3_SEG_RANGE_DESCENDING:
			cur->wire_head-- ;
			break ;

		case EFFECTS_V3_SEG_REPEAT:
			// the same id again
			break ;

		case EFFECTS_V3_SEG_SET_DESCENDING:
			roaring64_iterator_previous (cur->seg[cur->wire_seg].iter) ;
			cur->wire_head =
				roaring64_iterator_value (cur->seg[cur->wire_seg].iter) ;
			break ;

		default:
			roaring64_iterator_advance (cur->seg[cur->wire_seg].iter) ;
			cur->wire_head =
				roaring64_iterator_value (cur->seg[cur->wire_seg].iter) ;
			break ;
	}
}

void EffectsV3IdCursor_RewindWire
(
	EffectsV3IdCursor *cur
) {
	ASSERT (cur != NULL) ;

	cur->wire_seg = 0 ;
	_WireEnterSegment (cur) ;
}

bool EffectsV3IdCursor_NextWire
(
	EffectsV3IdCursor *cur,
	uint64_t          *id
) {
	ASSERT (cur != NULL) ;
	ASSERT (id  != NULL) ;

	// a zero-length segment is representable, so more than one may be stepped
	// over before an id turns up
	while (cur->wire_left == 0 && cur->wire_seg < cur->list->n) {
		cur->wire_seg++ ;
		_WireEnterSegment (cur) ;
	}

	if (cur->wire_left == 0) {
		return false ;
	}

	*id = cur->wire_head ;

	cur->wire_left-- ;
	if (cur->wire_left > 0) {
		_WireAdvance (cur) ;
	}

	return true ;
}

//------------------------------------------------------------------------------
// distinctness
//------------------------------------------------------------------------------

bool EffectsV3IdList_AreDistinct
(
	const EffectsV3IdList *ids,  // list to measure
	uint64_t               count // ids it states it holds
) {
	ASSERT (ids != NULL) ;

	roaring64_bitmap_t *seen = roaring64_bitmap_create () ;
	if (seen == NULL) {
		return false ;
	}

	bool ok = true ;
	uint64_t total = 0 ;   // POSITIONS the segments describe, not members

	for (uint32_t i = 0 ; ok && i < ids->n ; i++) {
		const EffectsV3IdListSegment *s = ids->segments + i ;
		const uint64_t card = _SegmentCardinality (s) ;

		switch (s->kind) {
			// A ZERO-LENGTH SEGMENT contributes nothing and is stepped over
			// rather than turned into a span. The spans are validated here
			// rather than taken on trust: decode makes the same checks
			// (effects_v3_decode.c:181-192), but a list can be built by
			// something other than the decoder and the tests do exactly that.
			// Without them a wrapped span reaches add_range_closed with
			// min > max, which returns silently and makes the segment vanish -
			// an accidental 'false' instead of a stated one.
			case EFFECTS_V3_SEG_RANGE_ASCENDING:
				if (card > 0) {
					if (s->range_ascending.base > UINT64_MAX - (card - 1)) {
						ok = false ;
						break ;
					}
					roaring64_bitmap_add_range_closed (seen,
							s->range_ascending.base,
							s->range_ascending.base + (card - 1)) ;
				}
				break ;

			case EFFECTS_V3_SEG_RANGE_DESCENDING:
				// a descending range's base is its HIGHEST id, so its span
				// runs down from there
				if (card > 0) {
					if (card - 1 > s->range_descending.base) {
						ok = false ;
						break ;
					}
					roaring64_bitmap_add_range_closed (seen,
							s->range_descending.base - (card - 1),
							s->range_descending.base) ;
				}
				break ;

			case EFFECTS_V3_SEG_REPEAT:
				// THE SEGMENT THIS EXISTS FOR. It names one id 'count' times
				// and contributes exactly one member, so the cardinality falls
				// short by every repetition past the first
				if (card > 0) {
					roaring64_bitmap_add (seen, s->repeat.id) ;
				}
				break ;

			case EFFECTS_V3_SEG_SET_ASCENDING:
			case EFFECTS_V3_SEG_SET_DESCENDING: {
				// one wire kind read two ways: a set has no direction, so both
				// arms deserialize the same blob and the union is the same
				const bool descending =
					(s->kind == EFFECTS_V3_SEG_SET_DESCENDING) ;
				const unsigned char *blob = descending
					? s->set_descending.blob : s->set_ascending.blob ;
				const uint32_t blob_n = descending
					? s->set_descending.n : s->set_ascending.n ;

				roaring64_bitmap_t *b =
					roaring64_bitmap_portable_deserialize_safe (
							(const char*)blob, blob_n) ;
				if (b == NULL) {
					ok = false ;
					break ;
				}

				roaring64_bitmap_or_inplace (seen, b) ;
				roaring64_bitmap_free (b) ;
				break ;
			}

			default:
				// a kind this build does not know is refused, not counted
				ok = false ;
				break ;
		}

		if (ok) {
			if (card > UINT64_MAX - total) {
				ok = false ;
			} else {
				total += card ;
			}
		}
	}

	// BOTH, and neither alone is the question asked.
	//
	// 'total' is how many POSITIONS the segments describe; the cardinality is
	// how many DIFFERENT ids they hold. Comparing only the cardinality calls a
	// list distinct whenever a repeat is offset by a segment list that is also
	// short of the count - Range{10,2} + Repeat{12,2} against count 3 holds
	// three different ids and names four, and 12 twice. Comparing only the
	// total misses the repeat entirely. Decode happens to establish the first
	// of these, but this is reachable from anywhere a list can be built, so it
	// establishes its own preconditions rather than inheriting them.
	if (ok) {
		ok = (total == count) &&
			 (roaring64_bitmap_get_cardinality (seen) == count) ;
	}

	roaring64_bitmap_free (seen) ;
	return ok ;
}
