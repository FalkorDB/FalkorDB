/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "../../value.h"
/* This function register place holders as functions, for later execution plan modifications,
 * once a dedicated method encounters this place holder. */
void Register_PlaceholderFuncs();
