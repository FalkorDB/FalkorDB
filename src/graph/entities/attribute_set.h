/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "RG.h"
#include "../../value.h"

// indicates a none existing attribute ID
#define ATTRIBUTE_ID_NONE USHRT_MAX

// indicates all attributes for SET clauses that replace a property map
#define ATTRIBUTE_ID_ALL USHRT_MAX - 1

// mark attribute-set as read-only
#define ATTRIBUTE_SET_MARK_READONLY(set) SET_MSB(set)

// check if attribute-set is read-only
#define ATTRIBUTE_SET_IS_READONLY(set) MSB_ON(set)

typedef uint16_t AttributeID;

// type of change performed on the attribute-set
typedef enum {
	CT_NONE,    // no change
	CT_ADD,     // attribute been added
	CT_UPDATE,  // attribute been updated
	CT_DEL      // attribute been deleted
} AttributeSetChangeType;

typedef struct _AttributeSet _AttributeSet; // forward declaration
typedef _AttributeSet* AttributeSet;        // define opaque type

// returns number of attributes within the set
uint16_t AttributeSet_Count
(
	const AttributeSet set  // set to query
);

// retrieves a value from set
// NOTE: if the key does not exist
// v is set to a special constant value ATTRIBUTE_NOTFOUND
// and false is returned
bool AttributeSet_Get
(
	const AttributeSet set,  // set to retieve attribute from
	AttributeID id,          // attribute identifier
	SIValue *v               // [output] attribute
);

// retrieves a value from set by index
bool AttributeSet_GetIdx
(
	const AttributeSet set,  // set to retieve attribute from
	uint16_t i,              // index of the property
	AttributeID *attr_id,    // [output] attribute identifier
	SIValue *v               // [output] attribute
);

// returns true if attribute set contains attribute
bool AttributeSet_Contains
(
	const AttributeSet set,  // set to search
	AttributeID attr_id      // attribute id to locate
);

// adds an attribute to the set without cloning the SIValue
void AttributeSet_AddNoClone
(
	AttributeSet *set,  // set to update
	AttributeID *ids,   // identifiers
	SIValue *values,    // values
	ushort n,           // number of values to add
	bool allowNull		// accept NULLs
);

// adds an attribute to the set (clones the value)
void AttributeSet_Add
(
	AttributeSet *set,     // set to update
	AttributeID attr_id,   // attribute identifier
	SIValue value          // attribute value
);

// add, remove or update an attribute
// this function allows NULL value to be added to the set
// returns the type of change performed
AttributeSetChangeType AttributeSet_Set_Allow_Null
(
	AttributeSet *set,     // set to update
	AttributeID attr_id,   // attribute identifier
	SIValue value          // attribute value
);

// updates an existing attribute in the set
// - if the new value is NULL, the attribute is removed
// - if the new value is the same as the current value, no update occurs
// - otherwise, the attribute is updated
// returns true if the attribute was updated, false otherwise
bool AttributeSet_Update
(
	AttributeSet *set,    // set to update
	AttributeID attr_id,  // attribute identifier
	SIValue value,        // new value
	bool clone            // clone value
);

// clones attribute set without si values
AttributeSet AttributeSet_ShallowClone
(
	const AttributeSet set  // set to clone
);

// persists all attributes within given set
void AttributeSet_PersistValues
(
	const AttributeSet set  // set to persist
);

// get direct access to the set attributes
char *AttributeSet_Attributes
(
	const AttributeSet set  // set to retrieve attributes from
);

// free attribute set
void AttributeSet_Free
(
	AttributeSet *set  // set to be freed
);

