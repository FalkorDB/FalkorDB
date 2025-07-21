/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
*/

#include "funcs.h"
#include "../RG.h"
#include "../../deps/rax/rax.h"

extern rax *__aeRegisteredFuncs;

void AR_RegisterFuncs() {
	ASSERT(__aeRegisteredFuncs == NULL);
	__aeRegisteredFuncs = raxNew();

	Register_AggFuncs();
	Register_MapFuncs();
	Register_PathFuncs();
	Register_ListFuncs();
	Register_TimeFuncs();
	Register_PointFuncs();
	Register_EntityFuncs();
	Register_StringFuncs();
	Register_NumericFuncs();
	Register_BooleanFuncs();
	Register_ConditionalFuncs();
	Register_ComprehensionFuncs();
	Register_PlaceholderFuncs();
}

