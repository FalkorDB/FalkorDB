/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
*/

#pragma once

#include "../../value.h"
#include "graph_entity.h"

typedef int LabelID;

// helper macro that instantiates 'labels' as a stack array
// and updates 'label_count'
#define NODE_GET_LABELS(g, n, label_count)                              \
	LabelID labels[(label_count) = Graph_LabelTypeCount((g))];          \
	label_count = Graph_GetNodeLabels((g), (n), labels, (label_count))

// instantiate a new unpopulated node
#define GE_NEW_NODE()           \
(Node) {                        \
	.attributes = NULL,         \
	.id = INVALID_ENTITY_ID,    \
}

// struct representing a node in the graph
typedef struct {
	AttributeSet *attributes;       // MUST be the first member of Node
	EntityID id;                    // unique id, MUST be the second member
} Node;

// prints a string representation of the node to buffer
// return the string length
void Node_ToString
(
	const Node *n,
	char **buffer,
	size_t *bufferLen,
	size_t *bytesWritten,
	GraphEntityStringFormat format
);

