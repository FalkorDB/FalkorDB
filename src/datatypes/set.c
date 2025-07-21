/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
*/

#include "set.h"

set *Set_New(void) {
	return raxNew();
}

bool Set_Contains(set *s, SIValue v) {
	unsigned long long const hash = SIValue_HashCode(v);
	return (raxFind(s, (unsigned char *)&hash, sizeof(hash)) != raxNotFound);
}

/* Adds v to set. */
bool Set_Add(set *s, SIValue v) {
	unsigned long long const hash = SIValue_HashCode(v);
	return raxTryInsert(s, (unsigned char *)&hash, sizeof(hash), NULL, NULL);
}

/* Removes v from set. */
void Set_Remove(set *s, SIValue v) {
	unsigned long long const hash = SIValue_HashCode(v);
	raxRemove(s, (unsigned char *)&hash, sizeof(hash), NULL);
}

/* Return number of elements in set. */
uint64_t Set_Size(set *s) {
	return raxSize(s);
}

/* Free set. */
void Set_Free(set *s) {
	raxFree(s);
}

