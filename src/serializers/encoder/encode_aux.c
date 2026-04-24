/*
* Copyright FalkorDB Ltd. 2023 - present
* Licensed under the Server Side Public License v1 (SSPLv1).
*/

#include "encode_aux.h"
#include "v19/encode_v19.h"

void AUXSave
(
	RedisModuleIO *io
) {
	AUXSaveUDF_latest (io) ;
}

