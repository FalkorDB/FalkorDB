/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "effects.h"

#include <stdio.h>

// returns false in case of effect encode/decode version mismatch
static bool ValidateVersion
(
	FILE *stream,  // effects stream
	uint8_t *v     // specified version
) {
	ASSERT(v      != NULL);
	ASSERT(stream != NULL);

	// read version
	fread_assert(v, sizeof(uint8_t), stream);

	if(*v > EFFECTS_VERSION) {
		// unexpected effects version
		RedisModule_Log(NULL, "warning",
				"GRAPH.EFFECT version mismatch expected: 1..%d got: %d",
				EFFECTS_VERSION, *v);
		return false;
	}

	return true;
}

// applys effects encoded in buffer
void Effects_Apply
(
	GraphContext *gc,          // graph to operate on
	const char *effects_buff,  // encoded effects
	size_t l                   // size of buffer
) {
	// validations
	ASSERT(l > 0);  // buffer can't be empty
	ASSERT(effects_buff != NULL);  // buffer can't be NULL

	// read buffer in a stream fashion
	FILE *stream = fmemopen((void*)effects_buff, l, "r");

	// validate effects version
	uint8_t v;
	if(ValidateVersion(stream, &v) == false) {
		// replica/primary out of sync
		exit(1);
	}

	// lock graph for writing
	Graph *g = GraphContext_GetGraph(gc);
	Graph_AcquireWriteLock(g);

	// update graph sync policy
	MATRIX_POLICY policy = Graph_SetMatrixPolicy(g, SYNC_POLICY_RESIZE);

	// determine effects decoder based on specified version
	switch(v) {
		case 1:
			Effects_Apply_V1(gc, stream, l);
			break;
		case 2:
			Effects_Apply_V2(gc, stream, l);
			break;
		default:
			ASSERT(false && "unsupported effects version");
			break;
	}

	// restore graph sync policy
	Graph_SetMatrixPolicy(g, policy);

	// release write lock
	Graph_ReleaseLock(g);

	// close stream
	fclose(stream);
}

