/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

// the v3 write table
//
// none of these emit - v3 is one record per STATEMENT where C calls per
// FIELD, so a field is staged and the record built at the flush

#include "write_v3.h"

const EffectsWriter EFFECTS_WRITER_V3 = {
	.CreateIndex      = EffectsWriteV3_CreateIndex,
	.DropIndex        = EffectsWriteV3_DropIndex,
	.CreateConstraint = EffectsWriteV3_CreateConstraint,
	.DropConstraint   = EffectsWriteV3_DropConstraint,
} ;
