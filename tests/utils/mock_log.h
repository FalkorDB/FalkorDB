/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

// reset RedisModule_Log to use a mock version of it capable of running
// outside of Redis
void Logging_Reset (void) ;
