/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
*/

#pragma once
#include "cypher-parser.h"
#include <sys/types.h>

typedef cypher_ast_annotation_context_t AnnotationCtx;

/* This struct holds the AST annotations context. Each annotation context is used in
 * specific AST nodes annotation use case. For each context there is a dedicated inlined
 * getter and setter. */
typedef struct {
	AnnotationCtx *named_paths_ctx; // Annotation context for named paths projections.
	AnnotationCtx *to_string_ctx;   // Annotation context for AST_ToString of astnode.
	uint32_t anon_count;            // Counter of anonymous entities already created.
} AST_AnnotationCtxCollection;

AST_AnnotationCtxCollection *AST_AnnotationCtxCollection_New();

AnnotationCtx *AST_AnnotationCtxCollection_GetNamedPathsCtx(const AST_AnnotationCtxCollection *anot_ctx_collection);

AnnotationCtx *AST_AnnotationCtxCollection_GetToStringCtx(const AST_AnnotationCtxCollection *anot_ctx_collection);

void AST_AnnotationCtxCollection_SetNamedPathsCtx(AST_AnnotationCtxCollection *anot_ctx_collection, AnnotationCtx *named_paths_ctx);

void AST_AnnotationCtxCollection_Free(AST_AnnotationCtxCollection *anot_ctx_collection);
