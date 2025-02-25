/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "rax_extensions.h"
#include "arr.h"

bool raxIsSubset(rax *a, rax *b) {
	raxIterator it;
	raxStart(&it, b);
	raxSeek(&it, "^", NULL, 0);

	while(raxNext(&it)) { // For each key in b
		// Break if key is not present in a
		if(raxFind(a, it.key, it.key_len) == raxNotFound) break;
	}
	bool is_subset = raxEOF(&it); // True if iterator was depleted.

	raxStop(&it);
	return is_subset;
}

rax *raxClone(rax *orig) {
	rax *clone = raxNew();

	raxIterator it;
	raxStart(&it, orig);
	raxSeek(&it, "^", NULL, 0);

	// For each key in the original, duplicate the key and its value in the clone.
	while(raxNext(&it)) {
		raxInsert(clone, it.key, it.key_len, it.data, NULL);
	}

	raxStop(&it);
	return clone;
}

rax *raxCloneWithCallback(rax *orig, void *(*clone_callback)(void *)) {
	rax *clone = raxNew();

	raxIterator it;
	raxStart(&it, orig);
	raxSeek(&it, "^", NULL, 0);

	// For each key in the original, duplicate the key and its value in the clone.
	while(raxNext(&it)) {
		void *data = clone_callback(it.data);
		raxInsert(clone, it.key, it.key_len, data, NULL);
	}

	raxStop(&it);
	return clone;
}

void **raxValues
(
	rax *rax
) {
	// instantiate an array to hold all of the values in the rax
	void **values = array_new(void *, raxSize(rax));
	raxIterator it;
	raxStart(&it, rax);
	// iterate over all keys in the rax
	raxSeek(&it, "^", NULL, 0);
	while(raxNext(&it)) {
		// copy the value associated with the key into the array
		array_append(values, it.data);
	}
	raxStop(&it);

	return values;
}

unsigned char **raxKeys
(
	rax *rax
) {
	// instantiate an array to hold all of the keys in the rax
	unsigned char **keys = array_new(unsigned char *, raxSize(rax));

	raxIterator it;
	raxStart(&it, rax);

	// iterate over all keys in the rax
	raxSeek(&it, "^", NULL, 0);
	while(raxNext(&it)) {
		// copy the key into the array
		array_append(keys, (unsigned char *)rm_strndup((const char *)it.key,
					(int)it.key_len));
	}

	raxStop(&it);

	return keys;
}

