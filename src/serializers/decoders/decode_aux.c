/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "decode_graph.h"
#include "current/v18/decode_v18.h"

void AUXLoad
(
	RedisModuleIO *io
) {
	AUXLoadUDF_latest (io) ;
}

