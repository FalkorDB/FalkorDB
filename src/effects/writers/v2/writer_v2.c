/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

// the v2 write table
//
// v2 is shipped and read by engines this build will never see, so the
// table is frozen alongside the bytes

#include "write_v2.h"

const EffectsWriter EFFECTS_WRITER_V2 = {
	.CreateIndex      = EffectsWriteV2_CreateIndex,
	.DropIndex        = EffectsWriteV2_DropIndex,
	.CreateConstraint = EffectsWriteV2_CreateConstraint,
	.DropConstraint   = EffectsWriteV2_DropConstraint,
} ;
