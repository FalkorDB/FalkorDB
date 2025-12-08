/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "../../value.h"
#include "../../redismodule.h"

// indicates a none existing attribute ID
#define ATTRIBUTE_ID_NONE USHRT_MAX

// indicates all attributes for SET clauses that replace a property map
#define ATTRIBUTE_ID_ALL USHRT_MAX - 1

typedef uint16_t AttributeID;

// forward-declare the struct, without defining it
typedef struct _AttributeSet _AttributeSet;
typedef _AttributeSet* AttributeSet;

// type of change performed on the attribute-set
typedef enum {
	CT_NONE,    // no change
	CT_ADD,     // attribute been added
	CT_UPDATE,  // attribute been updated
	CT_DEL      // attribute been deleted
} AttributeSetChangeType;

// returns number of attributes within the set
uint16_t AttributeSet_Count
(
	const AttributeSet set  // attribute-set
);

// checks if attribute-set contains attribute
bool AttributeSet_Contains
(
	const AttributeSet set,  // attribute-set
	AttributeID id,          // attribute id to lookup
	uint16_t *idx            // [optional][output] attribute index
);

// returns the ith attribute ID
AttributeID AttributeSet_GetKey
(
	const AttributeSet set,  // attribute-set
	int16_t i                // i
);

// retrieves a value from set
// if attr_id isn't in the set returns false
bool AttributeSet_Get
(
	const AttributeSet set,  // set to retieve attribute from
	AttributeID attr_id,     // attribute id
	SIValue *v               // [output] value
);

// retrieves the ith attribute from attribute-set
void AttributeSet_GetIdx
(
	const AttributeSet set,  // attribute-set to retieve attribute from
	uint16_t i,              // index of attribute
	AttributeID *attr_id,    // [output] attribute ID
	SIValue *v               // [output] value
);

// adds new attributes to the attribute-set
// all attributes MUST NOT be in the set
void AttributeSet_Add
(
	AttributeSet *set,  // attribute-set to update
	AttributeID *ids,   // attribute ids
	SIValue *values,    // attribute values
	uint16_t n,         // number of attributes
	bool clone          // clone values
);

// add, remove or update an attribute
// returns the type of change performed
void AttributeSet_Update
(
	AttributeSetChangeType *change,  // [output] changes
	AttributeSet *set,               // set to update
	AttributeID *ids,                // attribute identifier
	SIValue *vals,                   // new value
	uint16_t n,                      // number of attributes
	bool clone                       // clone value
);

// removes an attribute from set
// returns true if attribute was removed false otherwise
bool AttributeSet_Remove
(
	AttributeSet *set,   // attribute-set
	AttributeID attr_id  // attribute ID to remove
);

// clones attribute-set
AttributeSet AttributeSet_Clone
(
	const AttributeSet set  // attribute-set to clone
);

// shallow clones attribute-set
AttributeSet AttributeSet_ShallowClone
(
	const AttributeSet set  // set to clone
);

// transfer attribute ownership from `src` set to `dst`
// if x is in src but not in dst, x ownership remains in src
// if x is in dst but not in src, x ownership remains in dst
// if x is in both, x ownership is transfered to dst
void AttributeSet_TransferOwnership
(
	AttributeSet src,  // set losing ownership
	AttributeSet dst   // set receiving ownership
);

// compute hash for attribute-set
XXH64_hash_t AttributeSet_HashCode
(
	const AttributeSet set  // attribute-set to compute hash of
);

// get attributeset's memory usage
size_t AttributeSet_memoryUsage
(
	const AttributeSet set  // set to compute memory consumption of
);

// defrag attribute-set
// present each heap allocated attribute to the allocator in the hope
// it will be able reduce fragmentation by relocating the memory
void AttributeSet_Defrag
(
	AttributeSet set,          // attribute-set
	RedisModuleDefragCtx *ctx  // defrag context
);

// free attribute-set
void AttributeSet_Free
(
	AttributeSet *set  // set to be freed
);

