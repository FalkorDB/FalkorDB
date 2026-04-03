/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "../util/dict.h"

dict *ParamParser_Parse
(
	char **input  // input to parse
);

