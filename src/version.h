/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#define FALKOR_VERSION_MAJOR 4
#define FALKOR_VERSION_MINOR 16
#define FALKOR_VERSION_PATCH 2

#define FALKOR_SEMANTIC_VERSION(major, minor, patch) \
  (major * 10000 + minor * 100 + patch)

#define FALKOR_MODULE_VERSION FALKOR_SEMANTIC_VERSION(FALKOR_VERSION_MAJOR, FALKOR_VERSION_MINOR, FALKOR_VERSION_PATCH)

