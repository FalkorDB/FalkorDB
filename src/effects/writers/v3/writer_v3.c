/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

// the v3 write table, whose arms stage rather than emit

#include "write_v3.h"

const EffectsWriter EFFECTS_WRITER_V3 = {
	.CreateNode       = EffectsWriteV3_CreateNode,
	.CreateEdge       = EffectsWriteV3_CreateEdge,
	.DeleteNode       = EffectsWriteV3_DeleteNode,
	.DeleteEdge       = EffectsWriteV3_DeleteEdge,
	.UpdateEntity     = EffectsWriteV3_UpdateEntity,
	.Labels           = EffectsWriteV3_Labels,
	.NewSchema        = EffectsWriteV3_NewSchema,
	.NewAttribute     = EffectsWriteV3_NewAttribute,

	.CreateIndex      = EffectsWriteV3_CreateIndex,
	.DropIndex        = EffectsWriteV3_DropIndex,
	.CreateConstraint = EffectsWriteV3_CreateConstraint,
	.DropConstraint   = EffectsWriteV3_DropConstraint,
} ;
