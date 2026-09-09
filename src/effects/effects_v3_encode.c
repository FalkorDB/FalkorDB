/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "effects_v3_encode.h"
#include "effects_internal.h"
#include "../util/rmalloc.h"

// defined below; records 11-14
static void _encode_ddl_record(const EffectsV3Record *r, EffectsBytes *out);

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

static void _encode_record
(
	const EffectsV3Record *r,    // record to write
	const EffectsBytes *raw,     // pre-encoded AttrValues, or NULL
	EffectsBytes *out            // sink
) {
	ASSERT(r   != NULL);
	ASSERT(out != NULL);

	EffectsV3_WriteUint(out, (uint64_t)(uint32_t)r->opcode, sizeof(EffectType));

	// records 11-14 are singular too, and carry no count for the same reason:
	// one index statement, one constraint
	if(r->opcode >= EFFECT_CREATE_INDEX && r->opcode <= EFFECT_DROP_CONSTRAINT) {
		_encode_ddl_record(r, out);
		return;
	}

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

	if(raw != NULL) {
		size_t n = EffectsBytes_Len(raw);
		if(n > 0) {
			unsigned char *buf = rm_malloc(n);
			EffectsBytes_CopyInto(raw, buf);
			EffectsBytes_Write(out, buf, n);
			rm_free(buf);
		}
	} else {
		_write_attr_values(r, out);
	}
}

void EffectsV3_EncodeRecord
(
	const EffectsV3Record *r,  // record to write
	EffectsBytes *out          // sink
) {
	_encode_record(r, NULL, out);
}

void EffectsV3_EncodeRecordWithRawValues
(
	const EffectsV3Record *r,    // record to write
	const EffectsBytes *values,  // the AttrValues block, already encoded
	EffectsBytes *out            // sink
) {
	_encode_record(r, values, out);
}

//------------------------------------------------------------------------------
// records 11-14: index and constraint DDL
//------------------------------------------------------------------------------

// the counted (attribute id, name) list both DDL families carry
//
// THE COUNT WIDTHS DIFFER: an index field count is a u16, a constraint property
// count is a u8. Same payload shape, two widths, and the listing does not spell
// it out inline - so an encoder written from the neighbouring record gets one
// of them wrong.
static void _write_attr_refs
(
	const EffectsV3Record *r,  // record whose refs to write
	size_t count_width,        // 2 for an index, 1 for a constraint
	EffectsBytes *out          // sink
) {
	EffectsV3_WriteUint(out, r->n_attrs_ref, count_width);

	for(uint16_t i = 0; i < r->n_attrs_ref; i++) {
		EffectsV3_WriteUint(out, r->attrs_ref[i].id, sizeof(AttributeID));
		_write_name(r->attrs_ref[i].name, out);
	}
}

// one option: its presence byte, then its value only when present
static void _write_opt_u64
(
	bool present,      // whether the statement said this
	uint64_t v,        // its value
	EffectsBytes *out  // sink
) {
	EffectsV3_WriteUint(out, present ? 1 : 0, 1);
	if(present) {
		EffectsV3_WriteUint(out, v, 8);
	}
}

// the index OPTIONS block
//
// Options travel in the RDB's field order, each behind a presence byte, and the
// presence byte means "THE STATEMENT SAID THIS" rather than "this is the
// default". An effect MUTATES an index that may already exist, where the RDB
// writes a whole one - so writing a default in place of an absent option is not
// a harmless substitution, it is an instruction to change something the
// statement never mentioned. That diverged a live replica with "Can not
// override index configuration: Language is already set".
//
// The text half is written WHATEVER the field type - five clear bytes when
// nothing is stated - so there is one gate, the vector half, not two.
//
// `dimension` alone has no presence byte, because a vector field must have one.
// That is why a vector block carries five values behind four markers.
static void _write_index_options
(
	const EffectsV3Record *r,  // record whose options to write
	EffectsBytes *out          // sink
) {
	const EffectsV3IndexOptions *o = &r->options;

	// language
	EffectsV3_WriteUint(out, o->has_language ? 1 : 0, 1);
	if(o->has_language) {
		_write_name(o->language, out);
	}

	// stopwords: a count then that many strings
	EffectsV3_WriteUint(out, o->has_stopwords ? 1 : 0, 1);
	if(o->has_stopwords) {
		EffectsV3_WriteUint(out, o->n_stopwords, 8);
		for(uint64_t i = 0; i < o->n_stopwords; i++) {
			_write_name(o->stopwords[i], out);
		}
	}

	// weight, an f64 written as its bits rather than through the SIValue codec
	EffectsV3_WriteUint(out, o->has_weight ? 1 : 0, 1);
	if(o->has_weight) {
		uint64_t bits;
		memcpy(&bits, &o->weight, sizeof(bits));
		EffectsV3_WriteUint(out, bits, 8);
	}

	// nostem, one byte and only ever 1 or 0 - a decoder refuses anything else
	EffectsV3_WriteUint(out, o->has_nostem ? 1 : 0, 1);
	if(o->has_nostem) {
		EffectsV3_WriteUint(out, o->nostem ? 1 : 0, 1);
	}

	// phonetic algorithm code
	EffectsV3_WriteUint(out, o->has_phonetic ? 1 : 0, 1);
	if(o->has_phonetic) {
		_write_name(o->phonetic, out);
	}

	if(!o->is_vector) {
		return;
	}

	// a vector field must have a dimension, so it carries no presence byte
	EffectsV3_WriteUint(out, o->dimension, 8);

	_write_opt_u64(o->has_m,              o->m,              out);
	_write_opt_u64(o->has_ef_construction, o->ef_construction, out);
	_write_opt_u64(o->has_ef_runtime,      o->ef_runtime,      out);
	_write_opt_u64(o->has_sim_func,        o->sim_func,        out);
}

// write one DDL record: 11 CREATE_INDEX, 12 DROP_INDEX, 13 CREATE_CONSTRAINT,
// 14 DROP_CONSTRAINT
//
//   11  schema_type · label_id · label · field_type · fields · OPTIONS
//   12  the same, and NO OPTIONS AT ALL - zero bytes, not an empty block.
//       "Mirrors 11 without the options" reads both ways; the corpus settles it
//   13  constraint_type · entity_type · STATUS · label_id · label · props
//   14  the same without the status
//
// GraphEntityType is 1-BASED - GETYPE_UNKNOWN takes 0, so a node is 1 - and
// IndexFieldType is a BIT FLAG SET rather than a discriminant, so a range index
// is NUMERIC|GEO|STR == 0x0E and must be tested with & rather than compared.
static void _encode_ddl_record
(
	const EffectsV3Record *r,  // record to write
	EffectsBytes *out          // sink
) {
	if(r->opcode == EFFECT_CREATE_INDEX || r->opcode == EFFECT_DROP_INDEX) {
		EffectsV3_WriteUint(out, (uint64_t)(uint32_t)r->schema_type,
				sizeof(SchemaType));
		EffectsV3_WriteUint(out, (uint64_t)(uint32_t)r->schema_id, sizeof(int));
		_write_name(r->name, out);
		EffectsV3_WriteUint(out, r->field_type, 4);
		_write_attr_refs(r, 2, out);

		// a drop carries none, and that is not the same as carrying empty ones
		if(r->opcode == EFFECT_CREATE_INDEX) {
			_write_index_options(r, out);
		}
		return;
	}

	EffectsV3_WriteUint(out, r->constraint_type, 4);
	EffectsV3_WriteUint(out, r->entity_type, 4);

	// the one place v3 carries more than C: a replica never validates, so the
	// announcement is the only thing that can tell it an enforcing constraint
	// from one still building. A drop has no such need and omits it
	if(r->opcode == EFFECT_CREATE_CONSTRAINT) {
		EffectsV3_WriteUint(out, r->status, 4);
	}

	EffectsV3_WriteUint(out, (uint64_t)(uint32_t)r->schema_id, sizeof(int));
	_write_name(r->name, out);
	_write_attr_refs(r, 1, out);
}
