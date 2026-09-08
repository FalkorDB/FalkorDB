/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "effects_v3_encode.h"
#include "effects_internal.h"
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

// LabelSet: u16 n, then n label ids
//
// 'n' is the number of LABELS, not the record's entity count - the set is
// hoisted once for every entity in the record
static void _write_label_set
(
	const EffectsV3Record *r,  // record whose labels to write
	EffectsBytes *out          // sink
) {
	EffectsV3_WriteUint(out, r->n_labels, sizeof(uint16_t));

	for(uint16_t i = 0; i < r->n_labels; i++) {
		EffectsV3_WriteUint(out, (uint64_t)(uint32_t)r->labels[i],
				sizeof(LabelID));
	}
}

// AttrIds: u16 n, then n attribute ids
//
// AttributeID is two bytes where LabelID is four; the record listing does not
// spell that out inline, so an encoder written from the record beside this one
// would use the wrong width
static void _write_attr_ids
(
	const EffectsV3Record *r,  // record whose attribute ids to write
	EffectsBytes *out          // sink
) {
	EffectsV3_WriteUint(out, r->n_attrs, sizeof(uint16_t));

	for(uint16_t i = 0; i < r->n_attrs; i++) {
		EffectsV3_WriteUint(out, r->attr_ids[i], sizeof(AttributeID));
	}
}

// AttrValues: count * n_attrs values, row-major
//
// Written through the SHARED SIValue codec against a borrowed sink, so v2 and
// v3 encode a value identically. T_NULL in a slot means REMOVE THIS ATTRIBUTE
// and is written like any other value - it is not padding, and filtering it
// out would turn every property removal into a no-op.
static void _write_attr_values
(
	const EffectsV3Record *r,  // record whose values to write
	EffectsBytes *out          // sink
) {
	if(r->n_values == 0) {
		return;
	}

	EffectsBuffer *wrapper = EffectsBuffer_Wrap(out);

	for(uint64_t i = 0; i < r->n_values; i++) {
		EffectsBuffer_WriteSIValue(r->values + i, wrapper);
	}

	// frees the wrapper, not the sink
	EffectsBuffer_Free(wrapper);
}

// write a NUL-terminated name through the shared string writer
static void _write_name
(
	const char *name,  // name to write
	EffectsBytes *out  // sink
) {
	EffectsBuffer *wrapper = EffectsBuffer_Wrap(out);
	EffectsBuffer_WriteString(name, wrapper);
	EffectsBuffer_Free(wrapper);
}

void EffectsV3_EncodeRecord
(
	const EffectsV3Record *r,  // record to write
	EffectsBytes *out          // sink
) {
	ASSERT(r   != NULL);
	ASSERT(out != NULL);

	EffectsV3_WriteUint(out, (uint64_t)(uint32_t)r->opcode, sizeof(EffectType));

	// records 9 and 10 are inherently singular: one schema, one attribute. They
	// carry no count, which is the one exception to every batchable record
	// being `opcode . count . blocks`
	if(r->opcode == EFFECT_ADD_SCHEMA) {
		EffectsV3_WriteUint(out, (uint64_t)(uint32_t)r->schema_type,
				sizeof(SchemaType));
		EffectsV3_WriteUint(out, (uint64_t)(uint32_t)r->schema_id, sizeof(int));
		_write_name(r->name, out);
		return;
	}

	if(r->opcode == EFFECT_ADD_ATTRIBUTE) {
		// two bytes, where a schema id beside it is four
		EffectsV3_WriteUint(out, r->attr_id, sizeof(AttributeID));
		_write_name(r->name, out);
		return;
	}

	EffectsV3_WriteUint(out, r->count, sizeof(uint32_t));

	//--------------------------------------------------------------------------
	// the shape, once, BEFORE the ids
	//
	// every batchable record without exception: a record is self-describing
	// before its rows. AttrValues is the one part that follows the IdList,
	// because it is per row rather than per record
	//--------------------------------------------------------------------------

	switch(r->opcode) {
		case EFFECT_UPDATE_NODE:
		case EFFECT_CREATE_NODE:
		case EFFECT_DELETE_NODE:
		case EFFECT_SET_LABELS:
		case EFFECT_REMOVE_LABELS:
			_write_label_set(r, out);
			break;

		case EFFECT_UPDATE_EDGE:
		case EFFECT_CREATE_EDGE:
		case EFFECT_DELETE_EDGE:
			EffectsV3_WriteUint(out, (uint64_t)(uint32_t)r->relation_id,
					sizeof(RelationID));
			break;

		default:
			ASSERT(false && "unknown v3 record opcode");
			return;
	}

	// only the four value-carrying records state attribute ids, and they state
	// them here, with the shape - not beside the values
	switch(r->opcode) {
		case EFFECT_UPDATE_NODE:
		case EFFECT_UPDATE_EDGE:
		case EFFECT_CREATE_NODE:
		case EFFECT_CREATE_EDGE:
			_write_attr_ids(r, out);
			break;
		default:
			break;
	}

	//--------------------------------------------------------------------------
	// the rows
	//--------------------------------------------------------------------------

	EffectsV3_EncodeIdList(&r->ids, out);

	// endpoints are per edge rather than per record, so they are their own
	// lists. Only create and delete carry them: an update's endpoints are
	// recoverable from the graph, which is why UPDATE_EDGE has none
	if(r->opcode == EFFECT_CREATE_EDGE || r->opcode == EFFECT_DELETE_EDGE) {
		EffectsV3_EncodeIdList(&r->src, out);
		EffectsV3_EncodeIdList(&r->dst, out);
	}

	_write_attr_values(r, out);
}
