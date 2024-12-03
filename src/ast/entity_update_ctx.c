/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "../util/arr.h"
#include "../util/rmalloc.h"
#include "entity_update_ctx.h"

// clone property set context
static void _PropertySetCtx_Clone
(
	const PropertySetCtx *ctx,  // context to clone
	PropertySetCtx *clone       // cloned context
) {
	ASSERT(ctx   != NULL);
	ASSERT(clone != NULL);

	clone->exp       = AR_EXP_Clone(ctx->exp);
	clone->mode      = ctx->mode;
	clone->attr_id   = ctx->attr_id;
	clone->sub_path  = NULL;
	clone->attribute = ctx->attribute;

	if(ctx->sub_path != NULL) array_clone(clone->sub_path, ctx->sub_path);
}

static void _PropertySetCtx_Free
(
	PropertySetCtx *ctx  // context to free
) {
	ASSERT(ctx != NULL);

	AR_EXP_Free(ctx->exp);
	if(ctx->sub_path != NULL) array_free(ctx->sub_path);
}

// create a new update context
EntityUpdateEvalCtx *UpdateCtx_New
(
	const char *alias  // entity alias
) {
	ASSERT(alias != NULL);

	EntityUpdateEvalCtx *ctx = rm_malloc(sizeof(EntityUpdateEvalCtx));

	ctx->alias         = alias;
	ctx->record_idx    = INVALID_INDEX;
	ctx->properties    = array_new(PropertySetCtx, 1);
	ctx->add_labels    = NULL;
	ctx->remove_labels = NULL;

	return ctx;
}

// clone update context
EntityUpdateEvalCtx *UpdateCtx_Clone
(
	const EntityUpdateEvalCtx *orig  // context to clone
) {
	ASSERT(orig != NULL);

	// number of updated properties
	uint prop_count = array_len(orig->properties);
	EntityUpdateEvalCtx *clone = rm_malloc(sizeof(EntityUpdateEvalCtx));

	// clone fields
	clone->alias         = orig->alias;
	clone->record_idx    = orig->record_idx;
	clone->properties    = array_new(PropertySetCtx, prop_count);
	clone->add_labels    = NULL;
	clone->remove_labels = NULL;

	if(orig->add_labels != NULL) {
		array_clone(clone->add_labels, orig->add_labels);
	}

	if(orig->remove_labels != NULL) {
		array_clone(clone->remove_labels, orig->remove_labels);
	}

	// clone properties context
	clone->properties = array_ensure_len(clone->properties, prop_count);
	for(uint i = 0; i < prop_count; i++) {
		const PropertySetCtx *prop_ctx = orig->properties + i;
		_PropertySetCtx_Clone(prop_ctx, clone->properties + i);
	}

	return clone;
}

// clear update context
void UpdateCtx_Clear
(
	EntityUpdateEvalCtx *ctx  // context to clear
) {
	uint prop_count = array_len(ctx->properties);

	for(uint i = 0; i < prop_count; i++) {
		_PropertySetCtx_Free(ctx->properties + i);
	}

	array_clear(ctx->properties);
}

// free update contex
void UpdateCtx_Free
(
	EntityUpdateEvalCtx *ctx  // context to free
) {
	ASSERT(ctx != NULL);

	// free properties context
	uint prop_count = array_len(ctx->properties);
	for(uint i = 0; i < prop_count; i++) {
		_PropertySetCtx_Free(ctx->properties + i);
	}
	array_free(ctx->properties);

	if(ctx->add_labels    != NULL) array_free(ctx->add_labels);
	if(ctx->remove_labels != NULL) array_free(ctx->remove_labels);

	rm_free(ctx);
}

