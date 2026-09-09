/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "../serializers_include.h"

typedef enum {
	ENCODER_CONTAINER_MATRIX = 0,  // matrix container slot
	ENCODER_CONTAINER_VECTOR = 1,  // vector container slot
	ENCODER_CONTAINER_COUNT  = 2   // number of slots
} EncoderContainerSlot;

// initialize reusable encoder containers.
// should be called in parent process before fork.
void EncoderContainerWorkspace_Init(void);

// acquire a container.
// in child process this borrows a pre-initialized workspace container.
// otherwise this allocates a temporary container and sets borrowed=false.
void EncoderContainerWorkspace_Acquire
(
	EncoderContainerSlot slot,  // requested workspace slot
	GxB_Container *container,   // [output] acquired container
	bool *borrowed              // [output] true when borrowing workspace
);

// release a container acquired via EncoderContainerWorkspace_Acquire.
// borrowed containers are kept for reuse and are not freed.
void EncoderContainerWorkspace_Release
(
	GxB_Container *container,  // container to release
	bool borrowed              // was this borrowed from workspace
);

