/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "effects_v3_encode.h"
#include "../util/rmalloc.h"

void EffectsV3_WriteUint
(
	EffectsBytes *out,  // sink
	uint64_t v,         // value
	uint8_t width       // 1, 2, 4 or 8 bytes
) {
	ASSERT(width == 1 || width == 2 || width == 4 || width == 8);

	unsigned char buf[8];
	for(uint8_t i = 0; i < width; i++) {
		buf[i] = (unsigned char)(v >> (8 * i));
	}

	EffectsBytes_Write(out, buf, width);
}

void EffectsV3_EncodeSegment
(
	const EffectsV3Seg *s,  // segment to write
	EffectsBytes *out       // sink
) {
	uint8_t header = 0;

	if(s->descending) {
		// never on a Repeat: one id however many times reads the same both
		// ways, so the bit would name a distinction that does not exist, and a
		// decoder refuses it there rather than ignoring it
		ASSERT(s->kind != EFFECTS_V3_SEG_REPEAT);
		header |= EFFECTS_V3_SEG_DESCENDING;
	}

	switch(s->kind) {
		case EFFECTS_V3_SEG_RANGE: {
			uint8_t vw = EffectsV3_WidthFor(s->range.base);
			uint8_t cw = EffectsV3_WidthFor(s->range.len);

			header |= EFFECTS_V3_SEG_RANGE;
			header |= (uint8_t)(EffectsV3_WidthCode(vw)
					<< EFFECTS_V3_SEG_VWIDTH_SHIFT);
			header |= (uint8_t)(EffectsV3_WidthCode(cw)
					<< EFFECTS_V3_SEG_CWIDTH_SHIFT);

			EffectsBytes_Write(out, &header, 1);

			// base is the FIRST id: the lowest ascending, the highest
			// descending. Written as it stands rather than as the set's
			// minimum, so a descending range can need a wider value field than
			// the ascending form over the same ids
			EffectsV3_WriteUint(out, s->range.base, vw);
			EffectsV3_WriteUint(out, s->range.len, cw);
			break;
		}

		case EFFECTS_V3_SEG_REPEAT: {
			uint8_t vw = EffectsV3_WidthFor(s->repeat.id);
			uint8_t cw = EffectsV3_WidthFor(s->repeat.count);

			header |= EFFECTS_V3_SEG_REPEAT;
			header |= (uint8_t)(EffectsV3_WidthCode(vw)
					<< EFFECTS_V3_SEG_VWIDTH_SHIFT);
			header |= (uint8_t)(EffectsV3_WidthCode(cw)
					<< EFFECTS_V3_SEG_CWIDTH_SHIFT);

			EffectsBytes_Write(out, &header, 1);
			EffectsV3_WriteUint(out, s->repeat.id, vw);
			EffectsV3_WriteUint(out, s->repeat.count, cw);
			break;
		}

		default: {
			// a bitmap carries its own length and needs no width codes, so both
			// fields stay zero
			header |= EFFECTS_V3_SEG_BITMAP;
			EffectsBytes_Write(out, &header, 1);

			size_t n = roaring64_bitmap_portable_size_in_bytes(s->bitmap.bitmap);

			// the count is a u32 on the wire, and a record's id count is a u32
			// too, so no single blob can outrun it
			ASSERT(n <= UINT32_MAX);
			EffectsV3_WriteUint(out, (uint64_t)n, 4);

			char *blob = rm_malloc(n);
			size_t written =
				roaring64_bitmap_portable_serialize(s->bitmap.bitmap, blob);
			ASSERT(written == n);

			EffectsBytes_Write(out, blob, written);
			rm_free(blob);
			break;
		}
	}
}

void EffectsV3_EncodeIdList
(
	const EffectsV3IdListBuilder *b,  // list to write
	EffectsBytes *out                 // sink
) {
	uint32_t n = EffectsV3IdListBuilder_SegmentCount(b);

	EffectsV3_WriteUint(out, n, 4);

	for(uint32_t i = 0; i < n; i++) {
		EffectsV3_EncodeSegment(EffectsV3IdListBuilder_Segment(b, i), out);
	}
}
