/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "effects_v3.h"

const char *EffectsV3Status_ToString
(
	EffectsV3Status status
) {
	switch (status) {
		case EFFECTS_V3_OK:                   return "ok" ;
		case EFFECTS_V3_TRUNCATED:            return "truncated" ;
		case EFFECTS_V3_MALFORMED:            return "malformed" ;
		case EFFECTS_V3_UNSUPPORTED_VERSION:  return "unsupported version" ;
		case EFFECTS_V3_UNSUPPORTED_FLAGS:    return "unsupported flags" ;
		case EFFECTS_V3_UNIMPLEMENTED:        return "unimplemented record" ;
		default:                              return "unknown" ;
	}
}
