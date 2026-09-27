/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "../effects.h"

// the per-version write table, chosen once and never consulted again
//
// The split the RDB uses on its READ side: Decode_Previous switches on the
// encoding version once, and a version's decoder never asks again. Here the
// switch is EffectsBuffer_New, which already knows the version it will emit.
//
// The RDB's write side needs no table, because a build encodes exactly one
// version. Effects cannot: EFFECTS_VERSION is a runtime config, since a
// v3-capable build still has to emit v2 for an older replica. So the version
// is a value rather than a compile-time fact, and the table is what stops it
// being re-tested at every call site.
//
// PARTIAL while the conversion proceeds. A slot exists per converted writer;
// the rest still branch on EffectsBuffer_V3 inside effects.c.
typedef struct {
	// index DDL
	//
	// one effect per FIELD on both wires. 'options' is the v2 wire and
	// 'stated' the subset the statement named; each arm reads one of them and
	// says which, rather than the shared entry point carrying both notes
	void (*CreateIndex)
	(
		EffectsBuffer *buff,
		SchemaType st,
		int label_id,
		const char *label,
		AttributeID attr_id,
		const char *attr,
		IndexFieldType t,
		SIValue options,
		SIValue stated
	);

	void (*DropIndex)
	(
		EffectsBuffer *buff,
		SchemaType st,
		int label_id,
		const char *label,
		AttributeID attr_id,
		const char *attr,
		IndexFieldType t
	);

	// constraint DDL
	//
	// 'status' is a ConstraintStatus that only v3 has a field for
	void (*CreateConstraint)
	(
		EffectsBuffer *buff,
		ConstraintType ct,
		GraphEntityType et,
		uint32_t status,
		int label_id,
		const char *label,
		const AttributeID *attr_ids,
		const char **attrs,
		uint8_t n
	);

	void (*DropConstraint)
	(
		EffectsBuffer *buff,
		ConstraintType ct,
		GraphEntityType et,
		int label_id,
		const char *label,
		const AttributeID *attr_ids,
		const char **attrs,
		uint8_t n
	);
} EffectsWriter;

extern const EffectsWriter EFFECTS_WRITER_V2;
extern const EffectsWriter EFFECTS_WRITER_V3;
