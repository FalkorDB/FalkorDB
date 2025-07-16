/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
*/

#include "node.h"

void Node_ToString
(
	const Node *n,
	char **buffer,
	size_t *bufferLen,
	size_t *bytesWritten,
	GraphEntityStringFormat format
) {
	GraphEntity_ToString((const GraphEntity *)n, buffer, bufferLen,
			bytesWritten, format, GETYPE_NODE);
}

