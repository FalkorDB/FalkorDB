/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

// the v2 write table, frozen alongside the bytes

#include "write_v2.h"

const EffectsWriter EFFECTS_WRITER_V2 = {
	.CreateNode       = EffectsWriteV2_CreateNode,
	.CreateEdge       = EffectsWriteV2_CreateEdge,
	.DeleteNode       = EffectsWriteV2_DeleteNode,
	.DeleteEdge       = EffectsWriteV2_DeleteEdge,
	.UpdateEntity     = EffectsWriteV2_UpdateEntity,
	.Labels           = EffectsWriteV2_Labels,
	.NewSchema        = EffectsWriteV2_NewSchema,
	.NewAttribute     = EffectsWriteV2_NewAttribute,

	.CreateIndex      = EffectsWriteV2_CreateIndex,
	.DropIndex        = EffectsWriteV2_DropIndex,
	.CreateConstraint = EffectsWriteV2_CreateConstraint,
	.DropConstraint   = EffectsWriteV2_DropConstraint,
} ;
