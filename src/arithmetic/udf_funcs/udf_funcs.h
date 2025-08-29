/*
* Copyright FalkorDB Ltd. 2023 - present
* Licensed under the Server Side Public License v1 (SSPLv1).
*/

#pragma once

#include "../../value.h"

void Register_UDFFuncs(void);

SIValue AR_UDF
(
	SIValue *argv,
	int argc,
	void *private_data
);

