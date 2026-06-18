/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include <stddef.h>

// the maximum byte size of an identifier. efforced on all schema and function
// names.
#define FDB_MAX_NAME_LEN 512

// "range:" + name + ":numeric:arr"
#define FDB_MAX_RANGE_NAME_LEN (FDB_MAX_NAME_LEN + 18)
// "vector:" + name
#define FDB_MAX_VECTOR_NAME_LEN (FDB_MAX_NAME_LEN + 7)
// max(range_field_name, vector_field_name)
#define FDB_MAX_TYPED_NAME_LEN FDB_MAX_RANGE_NAME_LEN

