/*
* Copyright FalkorDB Ltd. 2023 - present
* Licensed under the Server Side Public License v1 (SSPLv1).
*/

#pragma once

#include "../../value.h"

void Register_UDFFuncs(void);

// invokes a user defined function
SIValue AR_INVOKE_UDF
(
    SIValue *argv,      // arguments
    int argc,           // number of arguments
    void *private_data  // private context
);

