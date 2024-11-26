/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

// a single header file including all important headers for serialization

#include "GraphBLAS.h"
#include "../util/arr.h"
#include "../query_ctx.h"
#include "serializer_io.h"
#include "../redismodule.h"
#include "../util/rmalloc.h"
#include "graph_extensions.h"
#include "../datatypes/array.h"
#include "../datatypes/vector.h"
#include "../graph/graphcontext.h"
#include "../configuration/config.h"

// this struct is used to describe the payload content of a key
// it contains the type and the number of entities that were encoded
typedef struct {
	uint64_t offset;          // offset within state
	EncodeState state;        // payload type
	uint64_t entities_count;  // number of entities in the payload
} PayloadInfo;

