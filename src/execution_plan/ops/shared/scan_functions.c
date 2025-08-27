/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "scan_functions.h"
#include "../../../util/rmalloc.h"
#include "../../../graph/entities/qg_node.h"

// allocates and returns a new context
NodeScanCtx *NodeScanCtx_New
(
    const char *alias,  // alias
    const char *label,  // label
    LabelID label_id,   // label id
    const QGNode *n     // node
) {
	ASSERT (n     != NULL) ;
	ASSERT (alias != NULL) ;
	ASSERT (label != NULL) ;

    NodeScanCtx *ctx = rm_malloc(sizeof(NodeScanCtx));

	ctx->alias    = alias;
	ctx->label    = label;
	ctx->label_id = label_id;
	ctx->n        = QGNode_Clone(n);

    return ctx;
}

// clones a context
NodeScanCtx *NodeScanCtx_Clone
(
    const NodeScanCtx *ctx  // context
) {
    ASSERT(ctx    != NULL);
    ASSERT(ctx->n != NULL);

    NodeScanCtx *clone = rm_malloc(sizeof(NodeScanCtx));
    memcpy(clone, ctx, sizeof(NodeScanCtx));
    clone->n = QGNode_Clone(ctx->n);

    return clone;
}

// frees a context
void NodeScanCtx_Free
(
    NodeScanCtx *ctx  // context
) {
    ASSERT(ctx != NULL);
    ASSERT(ctx->n != NULL);
    
    QGNode_Free(ctx->n);

    rm_free(ctx);
}

