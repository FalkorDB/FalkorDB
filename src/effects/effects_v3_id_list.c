/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "effects_v3_id_list.h"
#include "effects_v3_run_cost.h"
#include "../util/rmalloc.h"

// the run's direction, once a second segment has settled it
//
// UNDECIDED is not a third direction: a lone id belongs to neither, and the id
// after it is what decides. Until then a push either way continues this run
// rather than starting another
typedef enum {
	RUN_UNDECIDED = 0,
	RUN_ASCENDING,
	RUN_DESCENDING,
} RunDirection;

struct EffectsV3IdListBuilder {
	// a plain array rather than util/arr.h: that header puts a 12-byte
	// arr_hdr_t immediately before the elements, so every element lands at
	// offset 4 mod 8 and any type needing 8-byte alignment is misaligned.
	// EffectsV3Seg holds uint64_t and a pointer, so it does. Loads work on
	// arm64 and x86-64 but it is undefined behaviour, and UBSan reports it -
	// which would bury the reports this code is actually being checked for
	EffectsV3Seg *segments;  // segments, in wire order
	uint32_t n_segments;     // segments in use
	uint32_t cap_segments;   // segments allocated
	uint64_t len;            // ids pushed

	EffectsV3Run run;        // the collapse tally for the open run
	uint32_t run_start;      // index in 'segments' where the run begins
	RunDirection run_dir;    // which way the run goes
};

//------------------------------------------------------------------------------
// segment geometry
//------------------------------------------------------------------------------

uint8_t EffectsV3_WidthFor
(
	uint64_t v  // value to size
) {
	if(v <= UINT8_MAX)  return 1;
	if(v <= UINT16_MAX) return 2;
	if(v <= UINT32_MAX) return 4;
	return 8;
}

uint8_t EffectsV3_WidthCode
(
	uint8_t width  // width in bytes
) {
	switch(width) {
		case 1:  return 0;
		case 2:  return 1;
		case 4:  return 2;
		default: return 3;
	}
}

uint32_t EffectsV3Seg_Len
(
	const EffectsV3Seg *s  // segment
) {
	switch(s->kind) {
		case EFFECTS_V3_SEG_RANGE:  return s->range.len;
		case EFFECTS_V3_SEG_REPEAT: return s->repeat.count;
		default:                    return s->bitmap.len;
	}
}

uint64_t EffectsV3Seg_Min
(
	const EffectsV3Seg *s  // segment
) {
	switch(s->kind) {
		case EFFECTS_V3_SEG_RANGE:
			// a descending range's base is its highest id
			return s->descending
				? s->range.base - (uint64_t)(s->range.len - 1)
				: s->range.base;
		case EFFECTS_V3_SEG_REPEAT:
			return s->repeat.id;
		default:
			return s->bitmap.min;
	}
}

uint64_t EffectsV3Seg_Max
(
	const EffectsV3Seg *s  // segment
) {
	switch(s->kind) {
		case EFFECTS_V3_SEG_RANGE:
			return s->descending
				? s->range.base
				: s->range.base + (uint64_t)(s->range.len - 1);
		case EFFECTS_V3_SEG_REPEAT:
			return s->repeat.id;
		default:
			return s->bitmap.max;
	}
}

size_t EffectsV3Seg_EncodedLen
(
	const EffectsV3Seg *s  // segment
) {
	switch(s->kind) {
		case EFFECTS_V3_SEG_RANGE:
			return 1
				+ EffectsV3_WidthFor(s->range.base)
				+ EffectsV3_WidthFor(s->range.len);
		case EFFECTS_V3_SEG_REPEAT:
			return 1
				+ EffectsV3_WidthFor(s->repeat.id)
				+ EffectsV3_WidthFor(s->repeat.count);
		default:
			// header byte, u32 length prefix, then the blob itself
			return 1 + 4
				+ roaring64_bitmap_portable_size_in_bytes(s->bitmap.bitmap);
	}
}

//------------------------------------------------------------------------------
// the builder
//------------------------------------------------------------------------------

EffectsV3IdListBuilder *EffectsV3IdListBuilder_New
(
	void
) {
	EffectsV3IdListBuilder *b = rm_malloc(sizeof(EffectsV3IdListBuilder));

	b->cap_segments = 4;
	b->segments     = rm_malloc(sizeof(EffectsV3Seg) * b->cap_segments);
	b->n_segments   = 0;
	b->len          = 0;
	b->run_start = 0;
	b->run_dir   = RUN_UNDECIDED;
	EffectsV3Run_Restart(&b->run);

	return b;
}

uint64_t EffectsV3IdListBuilder_Len
(
	const EffectsV3IdListBuilder *b  // builder
) {
	return b->len;
}

uint32_t EffectsV3IdListBuilder_SegmentCount
(
	const EffectsV3IdListBuilder *b  // builder
) {
	return b->n_segments;
}

const EffectsV3Seg *EffectsV3IdListBuilder_Segment
(
	const EffectsV3IdListBuilder *b,  // builder
	uint32_t i                        // segment index
) {
	ASSERT(i < b->n_segments);

	return b->segments + i;
}

// begin a new run at 'start', discarding what the old one tallied
static void _restart_run
(
	EffectsV3IdListBuilder *b,  // builder
	uint32_t start              // index the new run begins at
) {
	EffectsV3Run_Restart(&b->run);
	b->run_start = start;
	b->run_dir   = RUN_UNDECIDED;
}

// append an ascending single-id range
static void _append(EffectsV3IdListBuilder *b, EffectsV3Seg s) {
	if(b->n_segments == b->cap_segments) {
		b->cap_segments *= 2;
		b->segments = rm_realloc(b->segments,
				sizeof(EffectsV3Seg) * b->cap_segments);
	}

	b->segments[b->n_segments++] = s;
}

static void _push_singleton
(
	EffectsV3IdListBuilder *b,  // builder
	uint64_t id                 // the id
) {
	EffectsV3Seg s = {
		.kind       = EFFECTS_V3_SEG_RANGE,
		.descending = false,
		.range      = { .base = id, .len = 1 },
	};

	_append(b, s);
}

// replace the open run with its bitmap, if the arithmetic has already gone
// that way
static void _maybe_collapse_run
(
	EffectsV3IdListBuilder *b  // builder
) {
	if(!EffectsV3Run_PrefersBitmap(&b->run)) {
		return;
	}

	uint32_t n = b->n_segments;

	roaring64_bitmap_t *bitmap = roaring64_bitmap_create();
	uint32_t len = 0;
	uint64_t min = UINT64_MAX;
	uint64_t max = 0;

	for(uint32_t i = b->run_start; i < n; i++) {
		const EffectsV3Seg *s = b->segments + i;

		// a run under consideration holds only ranges: a Repeat starts its own
		// run, and a collapsed bitmap is left behind by _restart_run
		ASSERT(s->kind == EFFECTS_V3_SEG_RANGE);

		uint64_t lo = EffectsV3Seg_Min(s);
		uint64_t hi = EffectsV3Seg_Max(s);

		// ONE range insertion per contributing range, over the id SET rather
		// than the direction - the blob is identical either way, which is why
		// the header carries the order instead. Never id by id: that produces
		// a different serialization of the same set
		roaring64_bitmap_add_range_closed(bitmap, lo, hi);

		len += EffectsV3Seg_Len(s);
		if(lo < min) min = lo;
		if(hi > max) max = hi;
	}

	// normative, and not a size win on this construction path - see the header
	roaring64_bitmap_run_optimize(bitmap);

	EffectsV3Seg collapsed = {
		.kind       = EFFECTS_V3_SEG_BITMAP,
		.descending = (b->run_dir == RUN_DESCENDING),
		.bitmap     = {
			.bitmap = bitmap,
			.len    = len,
			.min    = min,
			.max    = max,
		},
	};

	// drop the ranges the bitmap replaces and put it in their place
	b->n_segments = b->run_start;
	_append(b, collapsed);

	// the run begins AFTER the collapsed segment. A bitmap is never a candidate
	// for re-collapsing: inserting its span into a second bitmap would add every
	// id in that span the set does not hold. Ids continuing in the bitmap's own
	// direction are taken by the hot path below and never reach a run at all
	_restart_run(b, b->run_start + 1);
}

void EffectsV3IdListBuilder_Push
(
	EffectsV3IdListBuilder *b,  // builder
	uint64_t id                 // id to append
) {
	b->len++;

	uint32_t n = b->n_segments;
	EffectsV3Seg *last = (n > 0) ? (b->segments + (n - 1)) : NULL;

	//--------------------------------------------------------------------------
	// the hot paths, in the order they are taken
	//
	// all of them extend the segment already there, and none touches the run
	// tally, because a segment's cost is only folded in once it stops growing
	//--------------------------------------------------------------------------

	if(last != NULL) {
		if(last->kind == EFFECTS_V3_SEG_RANGE) {
			// "is this the id one past the end?" - a question with no answer at
			// the top of the id space, where base + len leaves it. The wrap is
			// defined in C rather than a trap, which makes it worse here than
			// undefined: base = UINT64_MAX, len = 1 computes 0, so pushing id 0
			// after the highest id extends the range instead of starting a new
			// segment, and the result claims an id that does not exist and
			// reports max below min. So the sum is only asked for when it exists
			if(!last->descending &&
			   last->range.base <= UINT64_MAX - (uint64_t)last->range.len &&
			   id == last->range.base + (uint64_t)last->range.len) {
				// one more consecutive id: every bulk create, every
				// delete-by-label, from first push to last
				last->range.len++;
				return;
			}

			if(last->descending &&
			   last->range.base >= (uint64_t)last->range.len &&
			   id == last->range.base - (uint64_t)last->range.len) {
				// the mirror, one more step down
				last->range.len++;
				return;
			}
		} else if(last->kind == EFFECTS_V3_SEG_REPEAT) {
			if(id == last->repeat.id) {
				// one more of the same id: a supernode's endpoint list is this
				// on every push after the first
				last->repeat.count++;
				return;
			}
		} else {
			// the run already collapsed and this id continues it: straight into
			// the bitmap, no new segment and nothing left to weigh
			if(!last->descending && id > last->bitmap.max) {
				roaring64_bitmap_add(last->bitmap.bitmap, id);
				last->bitmap.len++;
				last->bitmap.max = id;
				return;
			}

			if(last->descending && id < last->bitmap.min) {
				roaring64_bitmap_add(last->bitmap.bitmap, id);
				last->bitmap.len++;
				last->bitmap.min = id;
				return;
			}
		}
	}

	//--------------------------------------------------------------------------
	// a lone id has no direction, so the one after it decides
	//
	// both of these rewrite that segment rather than opening another
	//--------------------------------------------------------------------------

	if(last != NULL && last->kind == EFFECTS_V3_SEG_RANGE &&
	   !last->descending && last->range.len == 1) {
		uint64_t base = last->range.base;

		// the mirror question - "is this the id one BELOW base?" - and it has no
		// answer when base is 0. Asked as id + 1 == base it wraps instead:
		// pushing UINT64_MAX after id 0 computes 0 and rewrites the segment
		// descending from base 0, which then steps below the id space
		if(base > 0 && id == base - 1) {
			// it steps down: the same payload, read the other way
			last->descending = true;
			last->range.len  = 2;
			if(b->run_dir == RUN_UNDECIDED) {
				b->run_dir = RUN_DESCENDING;
			}
			return;
		}

		if(base == id) {
			// a repeat of the immediately preceding id folds into a Repeat -
			// the supernode case, where a whole endpoint list is one value.
			// It ends any run: a bitmap holds a value once
			last->kind         = EFFECTS_V3_SEG_REPEAT;
			last->descending   = false;
			last->repeat.id    = base;
			last->repeat.count = 2;

			// the run begins AFTER the Repeat, not at it. A Repeat can never
			// become a bitmap - a bitmap holds a value once, a Repeat holds it
			// count times - so leaving it inside the run lets a later collapse
			// fold it in, which keeps one copy of the id while still counting
			// all of them. That mismatch fails the decoder's cardinality check
			// and refuses the buffer, so it resyncs and fails again identically
			_restart_run(b, n);
			return;
		}
	}

	//--------------------------------------------------------------------------
	// otherwise this id either continues the run or ends it
	//--------------------------------------------------------------------------

	bool continues_run = false;
	uint64_t last_min = 0;

	if(last != NULL) {
		last_min = EffectsV3Seg_Min(last);
		uint64_t last_max = EffectsV3Seg_Max(last);

		switch(b->run_dir) {
			case RUN_ASCENDING:  continues_run = (id > last_max); break;
			case RUN_DESCENDING: continues_run = (id < last_min); break;
			default:
				// undecided: this id settles it, whichever side it falls
				continues_run = (id > last_max || id < last_min);
				break;
		}
	}

	if(continues_run) {
		if(b->run_dir == RUN_UNDECIDED) {
			b->run_dir = (id < last_min) ? RUN_DESCENDING : RUN_ASCENDING;
		}

		// the segment being superseded has its final length now, so this is the
		// moment its contribution is known - and the only moment it may be
		// charged, since charging an open segment would make the collapse
		// decision depend on when the encoder looked
		if(last->kind == EFFECTS_V3_SEG_RANGE) {
			EffectsV3Run_AddRangeBytes(&b->run, EffectsV3Seg_EncodedLen(last));
			EffectsV3Run_AddRange(&b->run, EffectsV3Seg_Min(last),
					EffectsV3Seg_Len(last));
		}

		_push_singleton(b, id);
		_maybe_collapse_run(b);
	} else {
		// a step that reverses the run's own direction ends it: a bitmap holds
		// one order, so everything before this is settled
		_push_singleton(b, id);
		_restart_run(b, b->n_segments - 1);
	}
}

void EffectsV3IdListBuilder_Free
(
	EffectsV3IdListBuilder *b  // builder
) {
	if(b == NULL) {
		return;
	}

	for(uint32_t i = 0; i < b->n_segments; i++) {
		if(b->segments[i].kind == EFFECTS_V3_SEG_BITMAP) {
			roaring64_bitmap_free(b->segments[i].bitmap.bitmap);
		}
	}

	rm_free(b->segments);
	rm_free(b);
}
