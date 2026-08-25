/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

// roaring.h is an auto-generated CRoaring amalgamation (do not edit) that
// unconditionally #defines ALIGNED. Consumers that already pulled in a
// RediSearch header defining the identical macro (e.g. query_error.h) get a
// -Wmacro-redefined warning when roaring.h is included afterward - save and
// restore the prior definition around roaring.h's own so nothing later in
// the translation unit is silently left with the wrong one.
//
// include this header instead of util/roaring.h directly.
#pragma push_macro("ALIGNED")
#undef ALIGNED
#include "roaring.h"
#pragma pop_macro("ALIGNED")
