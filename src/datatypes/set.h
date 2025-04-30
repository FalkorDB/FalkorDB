/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

/* TODO: not sure if this is the right place for this file
 * another possibility would be ./src/util */
#pragma once

#include <stddef.h>
#include "../value.h"
#include "../util/hashmap.h"

typedef struct hashmap set;

// create a new set
set *Set_New(void);

// check to see if v is in set
bool Set_Contains
(
	set *s,
	SIValue v
);

// adds v to set
bool Set_Add
(
	set *s,
	SIValue v
);

// removes v from set
void Set_Remove
(
	set *s,
	SIValue v
);

// return number of elements in set
uint64_t Set_Size
(
	set *s
);

// free set
void Set_Free
(
	set *s
);
