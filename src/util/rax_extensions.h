/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include <stddef.h>
#include <stdbool.h>
#include "../../deps/rax/rax.h"

// returns true if 'b' is a subset of 'a'
bool raxIsSubset
(
	rax *a,
	rax *b
);

// duplicates a rax, performing a shallow copy of the original's values
rax *raxClone
(
	rax *orig
);

// duplicates a rax, performing a deep copy of the original's values
rax *raxCloneWithCallback
(
	rax *orig,
	void *(*clone_callback)(void *)
);

// collect all values in a rax into an array
// this function assumes that each value is a pointer (or at least an 8-byte payload)
void **raxValues
(
	rax *rax
);

// collect all keys in a rax into an array
unsigned char **raxKeys
(
	rax *rax
);

