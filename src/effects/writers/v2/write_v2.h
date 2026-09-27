/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "../effects_writer.h"

// the v2 write arms, declared here so writer_v2.c can assemble the table from
// arms defined across several files - the shape serializers/encoder/v19 uses

void EffectsWriteV2_CreateIndex
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

void EffectsWriteV2_DropIndex
(
	EffectsBuffer *buff,
	SchemaType st,
	int label_id,
	const char *label,
	AttributeID attr_id,
	const char *attr,
	IndexFieldType t
);

void EffectsWriteV2_CreateConstraint
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

void EffectsWriteV2_DropConstraint
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
