/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "../func_desc.h"
#include "../../util/arr.h"

SIValue AR_NOP
(
	SIValue *argv,
	int argc,
	void *private_data
) {
	ASSERT (false) ;
	return SI_NullVal () ;
}

