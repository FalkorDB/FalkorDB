/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "effects_wire.h"
#include "../util/rmalloc.h"

char *ReadWireString
(
	FILE *stream  // effects stream
) {
	ASSERT (stream != NULL) ;

	size_t l ;
	if (!fread_checked (&l, sizeof (l), stream)) {
		return NULL ;
	}

	// a string is written as strlen + 1 bytes, so 0 is malformed
	if (l == 0) {
		return NULL ;
	}

	// reject a length that outruns the payload BEFORE allocating for it
	long remaining = fstream_remaining (stream) ;
	if (remaining < 0 || l > (size_t)remaining) {
		return NULL ;
	}

	char *str = rm_malloc (l) ;
	if (!fread_checked (str, l, stream)) {
		rm_free (str) ;
		return NULL ;
	}

	// the writer terminates; a payload that doesn't is malformed, and trusting
	// it would hand an unterminated buffer to every strlen downstream
	if (str[l - 1] != '\0') {
		rm_free (str) ;
		return NULL ;
	}

	return str ;
}
