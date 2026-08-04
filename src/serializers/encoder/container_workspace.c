/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "container_workspace.h"
#include "../../globals.h"

// Child encoding can enter nested container usage:
// _Encode_GrB_Matrix -> _encode_multiedge_array -> _Encode_multiedge.
// Keep two pre-allocated slots so nested calls don't contend.
static GxB_Container _workspace[ENCODER_CONTAINER_COUNT] = {NULL};

void EncoderContainerWorkspace_Init(void) {
	for (int i = 0; i < ENCODER_CONTAINER_COUNT; i++) {
		if (_workspace[i] != NULL) {
			continue;
		}

		GrB_OK (GxB_Container_new(&_workspace[i]));
	}
}

void EncoderContainerWorkspace_Acquire
(
	EncoderContainerSlot slot,
	GxB_Container *container,
	bool *borrowed
) {
	ASSERT(container != NULL);
	ASSERT(borrowed  != NULL);
	ASSERT(slot >= 0 && slot < ENCODER_CONTAINER_COUNT);

	if (Globals_Get_ProcessIsChild()) {
		ASSERT(_workspace[slot] != NULL);
		*container = _workspace[slot];
		*borrowed  = true;
		return;
	}

	GrB_OK (GxB_Container_new (container));
	*borrowed = false;
}

void EncoderContainerWorkspace_Release
(
	GxB_Container *container,
	bool borrowed
) {
	ASSERT(container != NULL);

	if (borrowed) {
		return;
	}

	GrB_OK (GxB_Container_free(container));
}

