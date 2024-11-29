/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

// applys effects encoded in stream
void Effects_Apply_V1
(
	GraphContext *gc,  // graph to operate on
	FILE *stream,      // effects stream
	size_t l           // stream length
);

// applys effects encoded in stream
void Effects_Apply_V2
(
	GraphContext *gc,  // graph to operate on
	FILE *stream,      // effects stream
	size_t l           // stream length
);

