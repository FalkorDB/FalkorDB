/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "effects_v3_encode.h"
#include "effects_v3.h"
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
	const EffectsV3IdListSegment *s,  // segment to write
	EffectsBytes *out                 // sink
) {
	// ONE ARM PER KIND AND NO `default:`. Direction lives in the kind, so an arm
	// that swallowed both Set variants would emit an ASCENDING header over a
	// descending payload - well formed, every length check passing, ids replayed
	// the wrong way round by the reader. A `default:` over a closed enum turns
	// "I handled every case" into "I handled the ones I thought of"; without one
	// a sixth kind is a compile error here.
	uint8_t header = 0;

	switch(s->kind) {
		case EFFECTS_V3_SEG_RANGE_ASCENDING:
		case EFFECTS_V3_SEG_RANGE_DESCENDING: {
			const bool desc = (s->kind == EFFECTS_V3_SEG_RANGE_DESCENDING);
			const uint8_t vwc = desc ? s->range_descending.value_width
			                         : s->range_ascending.value_width;
			const uint8_t cwc = desc ? s->range_descending.count_width
			                         : s->range_ascending.count_width;
			const uint64_t base = desc ? s->range_descending.base
			                           : s->range_ascending.base;
			const uint64_t len  = desc ? s->range_descending.len
			                           : s->range_ascending.len;

			header |= EFFECTS_V3_WIRE_SEG_RANGE;
			if(desc) header |= EFFECTS_V3_SEG_DESCENDING;
			header |= (uint8_t)(vwc << EFFECTS_V3_SEG_VWIDTH_SHIFT);
			header |= (uint8_t)(cwc << EFFECTS_V3_SEG_CWIDTH_SHIFT);

			EffectsBytes_Write(out, &header, 1);

			// base is the FIRST id: the lowest ascending, the highest
			// descending. Written as it stands rather than as the set's
			// minimum, so a descending range can need a wider value field
			// than the ascending form over the same ids
			EffectsV3_WriteUint(out, base, EFFECTS_V3_WIDTH_BYTES(vwc));
			EffectsV3_WriteUint(out, len,  EFFECTS_V3_WIDTH_BYTES(cwc));
			break;
		}

		case EFFECTS_V3_SEG_REPEAT: {
			// no direction: one id held 'count' times reads the same either
			// way, so the kind cannot carry one and bit 6 stays clear
			header |= EFFECTS_V3_WIRE_SEG_REPEAT;
			header |= (uint8_t)(s->repeat.value_width
					<< EFFECTS_V3_SEG_VWIDTH_SHIFT);
			header |= (uint8_t)(s->repeat.count_width
					<< EFFECTS_V3_SEG_CWIDTH_SHIFT);

			EffectsBytes_Write(out, &header, 1);
			EffectsV3_WriteUint(out, s->repeat.id,
					EFFECTS_V3_WIDTH_BYTES(s->repeat.value_width));
			EffectsV3_WriteUint(out, s->repeat.count,
					EFFECTS_V3_WIDTH_BYTES(s->repeat.count_width));
			break;
		}

		case EFFECTS_V3_SEG_SET_ASCENDING:
		case EFFECTS_V3_SEG_SET_DESCENDING: {
			const bool desc = (s->kind == EFFECTS_V3_SEG_SET_DESCENDING);
			const unsigned char *blob = desc ? s->set_descending.blob
			                                 : s->set_ascending.blob;
			const uint32_t n = desc ? s->set_descending.n : s->set_ascending.n;

			// a Set carries its own length and needs no width codes, so those
			// header bits stay clear
			header |= EFFECTS_V3_WIRE_SEG_SET;
			if(desc) header |= EFFECTS_V3_SEG_DESCENDING;
			EffectsBytes_Write(out, &header, 1);

			// serialized when the list was converted, or read off the wire;
			// either way written back verbatim
			EffectsV3_WriteUint(out, n, 4);
			EffectsBytes_Write(out, blob, n);
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
	const LabelID *labels,  // the set, ascending
	uint16_t n,             // how many
	EffectsBytes *out       // sink
) {
	EffectsV3_WriteUint(out, n, sizeof(uint16_t));

	for(uint16_t i = 0; i < n; i++) {
		EffectsV3_WriteUint(out, (uint64_t)(uint32_t)labels[i],
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
	const AttributeID *ids,  // the set, ascending
	uint16_t n,              // how many
	EffectsBytes *out        // sink
) {
	EffectsV3_WriteUint(out, n, sizeof(uint16_t));

	for(uint16_t i = 0; i < n; i++) {
		EffectsV3_WriteUint(out, ids[i], sizeof(AttributeID));
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
	const SIValue *values,  // count * n_attrs, row-major
	uint64_t n,             // how many
	EffectsBytes *out       // sink
) {
	if(n == 0) {
		return;
	}

	EffectsBuffer *wrapper = EffectsBuffer_Wrap(out);

	for(uint64_t i = 0; i < n; i++) {
		EffectsBuffer_WriteSIValue(values + i, wrapper);
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

// the AttrValues block: the bytes the accumulator already encoded, or the
// record's own SIValues
//
// The grouping path encodes values as they arrive and hands the finished block
// over, so it is copied rather than re-encoded from SIValues the group no
// longer holds. A record built by hand carries the values themselves.
static void _write_values
(
	const EffectsBytes *raw,  // pre-encoded block, or NULL
	const SIValue *values,    // count * n_attrs, row-major
	uint64_t n_values,        // how many
	EffectsBytes *out         // sink
) {
	if(raw == NULL) {
		_write_attr_values(values, n_values, out);
		return;
	}

	size_t n = EffectsBytes_Len(raw);
	if(n == 0) {
		return;
	}

	unsigned char *buf = rm_malloc(n);
	EffectsBytes_CopyInto(raw, buf);
	EffectsBytes_Write(out, buf, n);
	rm_free(buf);
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

	// ONE ARM PER OPCODE AND NO `default:`.
	//
	// Records 1-8 share a wire shape - count, the shape half, AttrIds, the
	// IdList, then AttrValues - written out per opcode rather than once through
	// a projection. A deliberate trade: eight arms restate the shared layout
	// eight times, and in exchange every record's format is readable in one
	// place, with no indirection between the opcode and the bytes.
	//
	// Records 9 and 10 are singular - one schema, one attribute, no count - the
	// one exception to `opcode . count . blocks`. Records 11-14 are singular too
	// and go to their own writer. Without a `default:` a fifteenth opcode is a
	// compile error rather than a record that silently encodes as nothing.
	switch(r->opcode) {
		case EFFECT_UPDATE_NODE:
			EffectsV3_WriteUint(out, r->update_node.count, sizeof(uint32_t));
			_write_label_set(r->update_node.labels, r->update_node.n_labels, out);
			_write_attr_ids(r->update_node.attr_ids, r->update_node.n_attrs, out);
			EffectsV3_EncodeIdList(&r->update_node.ids, out);
			_write_values(raw, r->update_node.values, r->update_node.n_values,
					out);
			break;

		case EFFECT_UPDATE_EDGE:
			EffectsV3_WriteUint(out, r->update_edge.count, sizeof(uint32_t));
			EffectsV3_WriteUint(out,
					(uint64_t)(uint32_t)r->update_edge.relation_id,
					sizeof(RelationID));
			_write_attr_ids(r->update_edge.attr_ids, r->update_edge.n_attrs, out);
			// no endpoints: an update's are recoverable from the graph
			EffectsV3_EncodeIdList(&r->update_edge.ids, out);
			_write_values(raw, r->update_edge.values, r->update_edge.n_values,
					out);
			break;

		case EFFECT_CREATE_NODE:
			EffectsV3_WriteUint(out, r->create_node.count, sizeof(uint32_t));
			_write_label_set(r->create_node.labels, r->create_node.n_labels, out);
			_write_attr_ids(r->create_node.attr_ids, r->create_node.n_attrs, out);
			EffectsV3_EncodeIdList(&r->create_node.ids, out);
			_write_values(raw, r->create_node.values, r->create_node.n_values,
					out);
			break;

		case EFFECT_CREATE_EDGE:
			EffectsV3_WriteUint(out, r->create_edge.count, sizeof(uint32_t));
			EffectsV3_WriteUint(out,
					(uint64_t)(uint32_t)r->create_edge.relation_id,
					sizeof(RelationID));
			_write_attr_ids(r->create_edge.attr_ids, r->create_edge.n_attrs, out);
			EffectsV3_EncodeIdList(&r->create_edge.ids, out);
			// endpoints are per edge rather than per record, so they are their
			// own lists and they follow the ids
			EffectsV3_EncodeIdList(&r->create_edge.src, out);
			EffectsV3_EncodeIdList(&r->create_edge.dst, out);
			_write_values(raw, r->create_edge.values, r->create_edge.n_values,
					out);
			break;

		case EFFECT_DELETE_NODE:
			EffectsV3_WriteUint(out, r->delete_node.count, sizeof(uint32_t));
			_write_label_set(r->delete_node.labels, r->delete_node.n_labels, out);
			// no AttrIds block at all, which is not the same as an empty one
			EffectsV3_EncodeIdList(&r->delete_node.ids, out);
			_write_values(raw, NULL, 0, out);
			break;

		case EFFECT_DELETE_EDGE:
			EffectsV3_WriteUint(out, r->delete_edge.count, sizeof(uint32_t));
			EffectsV3_WriteUint(out,
					(uint64_t)(uint32_t)r->delete_edge.relation_id,
					sizeof(RelationID));
			EffectsV3_EncodeIdList(&r->delete_edge.ids, out);
			EffectsV3_EncodeIdList(&r->delete_edge.src, out);
			EffectsV3_EncodeIdList(&r->delete_edge.dst, out);
			_write_values(raw, NULL, 0, out);
			break;

		case EFFECT_SET_LABELS:
			EffectsV3_WriteUint(out, r->set_labels.count, sizeof(uint32_t));
			_write_label_set(r->set_labels.labels, r->set_labels.n_labels, out);
			EffectsV3_EncodeIdList(&r->set_labels.ids, out);
			_write_values(raw, NULL, 0, out);
			break;

		case EFFECT_REMOVE_LABELS:
			EffectsV3_WriteUint(out, r->remove_labels.count, sizeof(uint32_t));
			_write_label_set(r->remove_labels.labels, r->remove_labels.n_labels,
					out);
			EffectsV3_EncodeIdList(&r->remove_labels.ids, out);
			_write_values(raw, NULL, 0, out);
			break;

		case EFFECT_ADD_SCHEMA:
			EffectsV3_WriteUint(out,
					(uint64_t)(uint32_t)r->add_schema.schema_type,
					sizeof(SchemaType));
			EffectsV3_WriteUint(out, (uint64_t)(uint32_t)r->add_schema.schema_id,
					sizeof(int));
			_write_name(r->add_schema.name, out);
			break;

		case EFFECT_ADD_ATTRIBUTE:
			// two bytes, where a schema id beside it is four
			EffectsV3_WriteUint(out, r->add_attribute.attr_id,
					sizeof(AttributeID));
			_write_name(r->add_attribute.name, out);
			break;

		case EFFECT_CREATE_INDEX:
		case EFFECT_DROP_INDEX:
		case EFFECT_CREATE_CONSTRAINT:
		case EFFECT_DROP_CONSTRAINT:
			_encode_ddl_record(r, out);
			break;

		case EFFECT_UNKNOWN:
			ASSERT(false && "unknown v3 record opcode");
			break;
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
	const EffectsV3AttrRef *attrs,  // the (id, name) pairs
	uint16_t n,                     // how many
	size_t count_width,             // 2 for an index, 1 for a constraint
	EffectsBytes *out               // sink
) {
	EffectsV3_WriteUint(out, n, count_width);

	for(uint16_t i = 0; i < n; i++) {
		EffectsV3_WriteUint(out, attrs[i].id, sizeof(AttributeID));
		_write_name(attrs[i].name, out);
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
// writes a whole one - so writing a default in place of an absent option is an
// instruction to change something the statement never mentioned. That diverged a
// live replica with "Can not override index configuration: Language is already
// set".
//
// The text half is written WHATEVER the field type - five clear bytes when
// nothing is stated - so there is one gate, the vector half, not two.
// `dimension` alone has no presence byte, because a vector field must have one:
// a vector block carries five values behind four markers.
static void _write_index_options
(
	const EffectsV3IndexOptions *o,  // options to write
	EffectsBytes *out                // sink
) {
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
	// ONE ARM PER OPCODE rather than per wire layout: each opcode has its own
	// fields, so reading r->create_index.name when the opcode is DROP_INDEX is
	// valid C and wrong.
	switch(r->opcode) {
		case EFFECT_CREATE_INDEX:
			EffectsV3_WriteUint(out, (uint64_t)(uint32_t)r->create_index.schema_type,
					sizeof(SchemaType));
			EffectsV3_WriteUint(out, (uint64_t)(uint32_t)r->create_index.schema_id,
					sizeof(int));
			_write_name(r->create_index.name, out);
			EffectsV3_WriteUint(out, r->create_index.field_type, 4);
			_write_attr_refs(r->create_index.attrs, r->create_index.n_attrs, 2, out);
			_write_index_options(&r->create_index.options, out);
			break;

		case EFFECT_DROP_INDEX:
			// no options block at all - not an empty one. The type has no
			// field for it, which is what the opcode check here used to assert
			EffectsV3_WriteUint(out, (uint64_t)(uint32_t)r->drop_index.schema_type,
					sizeof(SchemaType));
			EffectsV3_WriteUint(out, (uint64_t)(uint32_t)r->drop_index.schema_id,
					sizeof(int));
			_write_name(r->drop_index.name, out);
			EffectsV3_WriteUint(out, r->drop_index.field_type, 4);
			_write_attr_refs(r->drop_index.attrs, r->drop_index.n_attrs, 2, out);
			break;

		case EFFECT_CREATE_CONSTRAINT:
			EffectsV3_WriteUint(out, r->create_constraint.constraint_type, 4);
			EffectsV3_WriteUint(out, r->create_constraint.entity_type, 4);
			// the one place v3 carries more than C: a replica never validates,
			// so the announcement is the only thing that can tell it an
			// enforcing constraint from one still building
			EffectsV3_WriteUint(out, r->create_constraint.status, 4);
			EffectsV3_WriteUint(out, (uint64_t)(uint32_t)r->create_constraint.schema_id,
					sizeof(int));
			_write_name(r->create_constraint.name, out);
			_write_attr_refs(r->create_constraint.attrs,
					r->create_constraint.n_attrs, 1, out);
			break;

		case EFFECT_DROP_CONSTRAINT:
			EffectsV3_WriteUint(out, r->drop_constraint.constraint_type, 4);
			EffectsV3_WriteUint(out, r->drop_constraint.entity_type, 4);
			EffectsV3_WriteUint(out, (uint64_t)(uint32_t)r->drop_constraint.schema_id,
					sizeof(int));
			_write_name(r->drop_constraint.name, out);
			_write_attr_refs(r->drop_constraint.attrs,
					r->drop_constraint.n_attrs, 1, out);
			break;

		default:
			ASSERT(false && "not a DDL record");
			break;
	}
}

// encode records back into a payload
//
// The seam the conformance round trip is built on: decode a fixture the OTHER
// engine produced, re-encode it here, compare bytes. The only test that puts C's
// encoder and Rust's bytes against each other deterministically - a live pair
// shows the two engines agreeing about a graph, not about a payload, and a
// fixture comparison alone never runs C's encoder.
//
// Deliberately NOT the accumulator, which answers "what should this query emit";
// this answers "write exactly these records, in exactly this order". A round
// trip has to reproduce what it was handed, including a record order the
// accumulator would have chosen differently.
//
// Returns false only on an internal failure - records that decoded cleanly
// always re-encode.
bool EffectsV3_Encode
(
	const EffectsV3Records *records,  // records to encode
	EffectsBuffer *eb                 // buffer to write into
) {
	ASSERT(records != NULL);
	ASSERT(eb      != NULL);

	if(records == NULL || eb == NULL) {
		return false;
	}

	// the header comes from the PAYLOAD, not from this build's configuration,
	// which is the whole point of a round trip
	EffectsBytes *body = EffectsBuffer_TakeBody(eb, records->version,
			records->flags);
	if(body == NULL) {
		// the buffer was already accumulating effects of its own
		return false;
	}

	for(uint32_t i = 0; i < records->n; i++) {
		EffectsV3_EncodeRecord(records->records + i, body);
	}

	return true;
}
