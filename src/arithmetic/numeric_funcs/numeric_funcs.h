/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "../../value.h"

void Register_NumericFuncs();

// tries to convert input to float
SIValue AR_TOFLOAT
(
    SIValue *argv,      // arguments
    int argc,           // number of arguments
    void *private_data  // private context
);

// tries to convert input to integer
SIValue AR_TOINTEGER
(
    SIValue *argv,      // arguments
    int argc,           // number of arguments
    void *private_data  // private context
);
