/*
* Copyright FalkorDB Ltd. 2023 - present
* Licensed under the Server Side Public License v1 (SSPLv1).
*/

#pragma once

#include "../../value.h"

// execute a JavaScript UDF function
// the function is execute is specified as a string at argv[0]
SIValue AR_UDF
(
	SIValue *argv,
	int argc,
	void *private_data
);

