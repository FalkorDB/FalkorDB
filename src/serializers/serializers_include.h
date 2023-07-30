/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

// A single header file including all important headers for serialization.

// include Redis Modules API
#include "../redismodule.h"
// include Graph Context
#include "../graph/graphcontext.h"
// include Query contxt
#include "../query_ctx.h"
// include GraphBLAS
#include "../../deps/GraphBLAS/Include/GraphBLAS.h"
// utils
#include "../util/arr.h"
#include "../util/rmalloc.h"
// non primitive data types
#include "../datatypes/array.h"
#include "../datatypes/vector.h"
// graph extentions
#include "graph_extensions.h"
// module configuration
#include "../configuration/config.h"

// this struct is used to describe the payload content of a key
// it contains the type and the number of entities that were encoded
typedef struct {
	EncodeState state;        // payload type
	uint64_t entities_count;  // number of entities in the payload
} PayloadInfo;

