/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */
#include "prev/v10/decode_v10.h"
#include "prev/v11/decode_v11.h"
#include "prev/v12/decode_v12.h"
#include "prev/v13/decode_v13.h"
#include "prev/v14/decode_v14.h"
#include "prev/v15/decode_v15.h"
#include "prev/v16/decode_v16.h"
#include "prev/v17/decode_v17.h"
#include "current/v18/decode_v18.h"

typedef GraphContext *(*DecoderFP) (SerializerIO io, const RedisModuleString *rm_key_name);

// get decoder for 
DecoderFP Decoder_GetDecoder
(
	uint64_t v  // version
);

