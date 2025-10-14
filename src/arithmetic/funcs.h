/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "map_funcs/map_funcs.h"
#include "udf_funcs/udf_funcs.h"
#include "list_funcs/list_funcs.h"
#include "time_funcs/time_funcs.h"
#include "point_funcs/point_funcs.h"
#include "entity_funcs/entity_funcs.h"
#include "general_funcs/general_funcs.h"
#include "string_funcs/string_funcs.h"
#include "aggregate_funcs/agg_funcs.h"
#include "vector_funcs/vector_funcs.h"
#include "boolean_funcs/boolean_funcs.h"
#include "numeric_funcs/numeric_funcs.h"
#include "conditional_funcs/conditional_funcs.h"
#include "comprehension_funcs/comprehension_funcs.h"
#include "path_funcs/path_funcs.h"
#include "placeholder_funcs/placeholder_funcs.h"

// registers all arithmetic functions
void AR_RegisterFuncs();

