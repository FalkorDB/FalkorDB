/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

// return the previous value of the first argument
// this function is used to store the previous value of an expression
// ex: UNWIND range(1, 5) AS x RETURN x, prev(x)
// the result will be:
// 1, NULL
// 2, 1
// 3, 2
// 4, 3
// 5, 4
void Register_GeneralFuncs();

