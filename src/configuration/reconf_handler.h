/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "config.h"

// handler function invoked when configuration changes
// might be called at any time after config_init
void reconf_handler(Config_Option_Field type);

