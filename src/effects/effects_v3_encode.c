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
	const EffectsV3Segment *s,  // segment to write
	EffectsBytes *out           // sink
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
			uint8_t vw = s->value_width;
			uint8_t cw = s->count_width;

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
			uint8_t vw = s->value_width;
			uint8_t cw = s->count_width;

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
			header |= EFFECTS_V3_SEG_ASCENDING;
			EffectsBytes_Write(out, &header, 1);

			// the blob was serialized when the list was converted, or read
			// off the wire; either way it is written back verbatim
			EffectsV3_WriteUint(out, s->ascending.n, 4);
			EffectsBytes_Write(out, s->ascending.blob, s->ascending.n);
			break;
		}
	}
}

void EffectsV3_EncodeIdList
(
	const EffectsV3IdList *l,  // list to write
	EffectsBytes *out          // sink
) {
	EffectsV3_WriteUint(out, l->n, 4);

	for(uint32_t i = 0; i < l->n; i++) {
		EffectsV3_EncodeSegment(l->segments + i, out);
	}
}
