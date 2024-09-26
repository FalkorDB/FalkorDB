/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#ifndef FALKORDB_MODULE_VERSION

#define FALKORDB_VERSION_MAJOR 4
#define FALKORDB_VERSION_MINOR 2
#define FALKORDB_VERSION_PATCH 2

#define FALKORDB_SEMANTIC_VERSION(major, minor, patch) \
  (major * 10000 + minor * 100 + patch)

#define FALKORDB_MODULE_VERSION FALKORDB_SEMANTIC_VERSION(FALKORDB_VERSION_MAJOR, FALKORDB_VERSION_MINOR, FALKORDB_VERSION_PATCH)

#endif
