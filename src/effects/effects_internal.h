/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "effects.h"
#include "effects_bytes.h"
#include "effects_v3_group.h"

#include <stdio.h>

//------------------------------------------------------------------------------
// shared low-level write primitives (defined in effects.c)
//
// exposed so each effect type's encoder can live in its own file (see
// create_index_effect.c, drop_index_effect.c, create_constraint_effect.c,
// drop_constraint_effect.c) while still writing into the same opaque
// EffectsBuffer block-list representation
//------------------------------------------------------------------------------

// write n bytes from ptr into effects-buffer
void EffectsBuffer_WriteBytes
(
	const void *ptr,   // data to write
	size_t n,          // number of bytes to write
	EffectsBuffer *eb  // effects-buffer
);

// write a length-prefixed, null-terminated string into effects-buffer
void EffectsBuffer_WriteString
(
	const char *str,
	EffectsBuffer *eb
);

// writes a binary representation of v into effects-buffer
void EffectsBuffer_WriteSIValue
(
	const SIValue *v,
	EffectsBuffer *buff
);

// increment number of effects recorded in buffer
void EffectsBuffer_IncEffectCount
(
	EffectsBuffer *buff
);

// the v3 accumulator this buffer is filling, or NULL when it emits v2
//
// lets the per-effect writers route into v3 without EffectsBuffer ceasing to be
// opaque to them
EffectsV3Grouping *EffectsBuffer_V3
(
	const EffectsBuffer *eb  // effects-buffer
);

// take over a buffer's body so pre-built v3 records can be written into it
//
// Returns the sink to write into, or NULL if the buffer has already staged
// effects of its own - those would be serialised over whatever is written here.
// 'version' and 'flags' become the header the buffer emits, so a re-encode
// reproduces the payload it decoded rather than the header this build prefers.
EffectsBytes *EffectsBuffer_TakeBody
(
	EffectsBuffer *eb,  // effects-buffer
	uint8_t version,    // version byte to emit
	uint8_t flags       // flags byte to emit
);

// wrap a byte sink the caller owns as an effects-buffer
//
// lets v3 reach the shared SIValue codec against its own sinks: a grouped
// record's values accumulate in that group's sink rather than in a buffer's
// record stream, and a second SIValue codec is exactly the thing not to write.
//
// the returned buffer BORROWS the sink - freeing it frees the wrapper only
EffectsBuffer *EffectsBuffer_Wrap
(
	EffectsBytes *sink  // sink to write into; not owned
);

//------------------------------------------------------------------------------
// shared apply-side helpers (defined in effects_apply.c)
//------------------------------------------------------------------------------

// resolve & verify a schema referenced by an effect via its id+name pair
// (the id is authoritative - it's only valid because every schema mutation
// is itself an effect, applied in the same order on every replica - the
// name is a cheap cross-check that surfaces divergence instead of silently
// trusting a stale/incorrect id)
//
// returns NULL if the replica has diverged from the master (the id doesn't
// resolve locally, or resolves to a schema with a different name)
Schema *VerifySchema
(
	GraphContext *gc,   // graph to operate on
	SchemaType st,      // schema type (node/edge)
	int label_id,       // expected label/relationship-type id
	const char *label   // expected label/relationship-type name
);

// resolve & verify an attribute referenced by an effect via its id+name pair
// returns false if the replica has diverged from the master
bool VerifyAttribute
(
	GraphContext *gc,     // graph to operate on
	AttributeID attr_id,  // expected attribute id
	const char *attr      // expected attribute name
);

// read (attribute id, attribute name) pairs off of stream, as encoded by
// EffectsBuffer_AddCreateConstraintEffect / EffectsBuffer_AddDropConstraintEffect
//
// returns props attribute-name allocations via 'props' (caller must free each
// entry with rm_free)
void ReadConstraintAttributes
(
	FILE *stream,           // effects stream
	uint8_t n,              // number of attributes
	AttributeID *attr_ids,  // [output] attribute ids
	char **props            // [output] mutable attribute-name view
) ;

//------------------------------------------------------------------------------
// per-effect-type apply functions, each defined in its own file, called
// from the Effects_Apply dispatch switch in effects_apply.c
//------------------------------------------------------------------------------

// process CreateIndex effect (create_index_effect.c)
bool ApplyCreateIndex
(
	FILE *stream,     // effects stream
	GraphContext *gc  // graph to operate on
);

// process DropIndex effect (drop_index_effect.c)
bool ApplyDropIndex
(
	FILE *stream,     // effects stream
	GraphContext *gc  // graph to operate on
);

// process CreateConstraint effect (create_constraint_effect.c)
bool ApplyCreateConstraint
(
	FILE *stream,     // effects stream
	GraphContext *gc  // graph to operate on
);

// process DropConstraint effect (drop_constraint_effect.c)
bool ApplyDropConstraint
(
	FILE *stream,     // effects stream
	GraphContext *gc  // graph to operate on
);

// process UpdateNode effect
// returns false if the effect references a node that doesn't exist locally
// (replica has diverged from the master)
bool ApplyUpdateNode
(
	FILE *stream,     // effects stream
	GraphContext *gc  // graph to operate on
);

// process Update_Edge effect
// returns false if the effect references an edge, or a relationship-type,
// that doesn't exist locally (replica has diverged from the master)
bool ApplyUpdateEdge
(
	FILE *stream,     // effects stream
	GraphContext *gc  // graph to operate on
);

