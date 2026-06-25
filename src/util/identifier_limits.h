/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include <stddef.h>

// maximum identifier characters, enforced on schema/function/property names.
#define FDB_MAX_IDENTIFIER_LEN 512

// Index-field identifier limits used when composing type-aware field names.
// "range:" + identifier + ":numeric:arr"
#define FDB_MAX_INDEX_RANGE_FIELD_NAME_LEN (FDB_MAX_IDENTIFIER_LEN + 18)
// "vector:" + identifier
#define FDB_MAX_INDEX_VECTOR_FIELD_NAME_LEN (FDB_MAX_IDENTIFIER_LEN + 7)
// max(range_field_name, vector_field_name)
#define FDB_MAX_INDEX_TYPED_FIELD_NAME_LEN FDB_MAX_INDEX_RANGE_FIELD_NAME_LEN

