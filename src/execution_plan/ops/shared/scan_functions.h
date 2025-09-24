/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "../../../graph/entities/node.h"
#include "../../../graph/entities/qg_node.h"

// Storage struct for label data in node and index scans.
typedef struct {
	QGNode *n;          // node to scan (might hold multiple labels)
	LabelID label_id;   // label ID of the node being traversed
	const char *alias;  // alias of the node being traversed
	const char *label;  // label of the node being traversed
} NodeScanCtx;

// allocates and returns a new context
NodeScanCtx *NodeScanCtx_New
(
    const char *alias,  // alias
    const char *label,  // label
    LabelID label_id,   // label id
    const QGNode *n     // node
);

// clones a context
NodeScanCtx *NodeScanCtx_Clone
(
    const NodeScanCtx *ctx  // context
);

// frees a context
void NodeScanCtx_Free
(
    NodeScanCtx *ctx  // context
);

