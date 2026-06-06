/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include <stddef.h>

#define FALKORDB_MAX_IDENTIFIER_LEN 512

// "range:" + name + ":numeric:arr"
#define FALKORDB_MAX_RANGE_FIELD_NAME_LEN (FALKORDB_MAX_IDENTIFIER_LEN + 18)
// "vector:" + name
#define FALKORDB_MAX_VECTOR_FIELD_NAME_LEN (FALKORDB_MAX_IDENTIFIER_LEN + 7)
// max(range_field_name, vector_field_name)
#define FALKORDB_MAX_TYPE_AWARE_FIELD_NAME_LEN FALKORDB_MAX_RANGE_FIELD_NAME_LEN

