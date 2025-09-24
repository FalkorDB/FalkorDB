/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include <limits.h>

#include "RG.h"
#include "attribute_set.h"
#include "../../util/rmalloc.h"
#include "../../errors/errors.h"

// compute size of attribute set in bytes
#define ATTRIBUTESET_BYTE_SIZE(set) ((set) == NULL ? \
		sizeof(_AttributeSet) :                      \
		sizeof(_AttributeSet) + sizeof(Attribute) * (set)->attr_count)

// mark attribute-set as mutable
#define ATTRIBUTE_SET_CLEAR_MSB(set) (CLEAR_MSB((intptr_t)set))

// returned value for a missing attribute
SIValue *ATTRIBUTE_NOTFOUND = &(SIValue) {
	.longval = 0, .type = T_NULL
};

// removes an attribute from set
// returns true if attribute was removed false otherwise
static bool _AttributeSet_Remove
(
	AttributeSet *set,
	AttributeID attr_id
) {
	AttributeSet _set = *set;
	const uint16_t attr_count = _set->attr_count;

	// attribute-set can't be read-only
	ASSERT(ATTRIBUTE_SET_IS_READONLY(_set) == false);

	// locate attribute position
	for (uint16_t i = 0; i < attr_count; ++i) {
		if(attr_id != _set->attributes[i].id) {
			continue;
		}

		// if this is the last attribute free the attribute-set
		if(_set->attr_count == 1) {
			AttributeSet_Free(set);
			return true;
		}

		// attribute located
		// free attribute value
		SIValue_Free(_set->attributes[i].value);

		// overwrite deleted attribute with the last
		// attribute and shrink set
		_set->attributes[i] = _set->attributes[attr_count - 1];

		// update attribute count
		_set->attr_count--;

		// compute new set size
		size_t n = ATTRIBUTESET_BYTE_SIZE(_set);
		*set = rm_realloc(_set, n);

		// attribute removed
		return true;
	}

	// unable to locate attribute
	return false;
}

// returns number of attributes within the set
uint16_t AttributeSet_Count
(
	const AttributeSet set  // set to query
) {
	// in case attribute-set is marked as read-only, clear marker
	AttributeSet _set = (AttributeSet)ATTRIBUTE_SET_CLEAR_MSB(set);

	return (_set == NULL) ? 0 : _set->attr_count;
}

// retrieves a value from set
// NOTE: if the key does not exist
// we return the special constant value ATTRIBUTE_NOTFOUND
SIValue *AttributeSet_Get
(
	const AttributeSet set,  // set to retieve attribute from
	AttributeID attr_id      // attribute identifier
) {
	// in case attribute-set is marked as read-only, clear marker
	AttributeSet _set = (AttributeSet)ATTRIBUTE_SET_CLEAR_MSB(set);

	if(_set == NULL) {
		return ATTRIBUTE_NOTFOUND;
	}

	if(attr_id == ATTRIBUTE_ID_NONE) {
		return ATTRIBUTE_NOTFOUND;
	}

	// TODO: benchmark, consider alternatives:
	// sorted set
	// array divided in two:
	// [attr_id_0, attr_id_1, attr_id_2, value_0, value_1, value_2]
	for (uint16_t i = 0; i < _set->attr_count; ++i) {
		Attribute *attr = _set->attributes + i;
		if(attr_id == attr->id) {
			// note, unsafe as attribute-set can get reallocated
			// TODO: why do we return a pointer to value instead of a copy ?
			// especially when AttributeSet_GetIdx returns SIValue
			// note AttributeSet_Update operate on this pointer
			return &attr->value;
		}
	}

	return ATTRIBUTE_NOTFOUND;
}

// retrieves a value from set by index
SIValue AttributeSet_GetIdx
(
	const AttributeSet set,  // set to retieve attribute from
	uint16_t i,              // index of the property
	AttributeID *attr_id     // attribute identifier
) {
	ASSERT(attr_id != NULL);

	// in case attribute-set is marked as read-only, clear marker
	AttributeSet _set = (AttributeSet)ATTRIBUTE_SET_CLEAR_MSB(set);

	ASSERT(_set != NULL);

	ASSERT(i < _set->attr_count);

	Attribute *attr = _set->attributes + i;
	*attr_id = attr->id;

	return attr->value;
}

static AttributeSet AttributeSet_AddPrepare
(
	AttributeSet *set,  // set to update
	ushort n            // number of attributes to add
) {
	ASSERT(set != NULL);

	AttributeSet _set = *set;

	// attribute-set can't be read-only
	ASSERT(ATTRIBUTE_SET_IS_READONLY(_set) == false);

	// allocate room for new attribute
	if(_set == NULL) {
		_set = rm_malloc(sizeof(_AttributeSet) + n * sizeof(Attribute));
		_set->attr_count = n;
	} else {
		_set->attr_count += n;
		_set = rm_realloc(_set, ATTRIBUTESET_BYTE_SIZE(_set));
	}

	return _set;
}

// adds an attribute to the set without cloning the SIvalue
void AttributeSet_AddNoClone
(
	AttributeSet *set,  // set to update
	AttributeID *ids,   // identifiers
	SIValue *values,    // values
	ushort n,           // number of values to add
	bool allowNull		// accept NULLs
) {
	ASSERT(set != NULL);

	// return if set is read-only
	if(unlikely(ATTRIBUTE_SET_IS_READONLY(*set))) {
		return;
	}

	// validate value type
	// value must be a valid property type
#ifdef RG_DEBUG
	SIType t = SI_VALID_PROPERTY_VALUE;
	if(allowNull == true) {
		t |= T_NULL;
	}

	for(ushort i = 0; i < n; i++) {
		ASSERT(SI_TYPE(values[i]) & t);
		// make sure attribute isn't already in set
		ASSERT(AttributeSet_Get(*set, ids[i]) == ATTRIBUTE_NOTFOUND);
		// make sure value isn't volotile
		ASSERT(SI_ALLOCATION(values + i) != M_VOLATILE);
	}
#endif

	ushort prev_count = AttributeSet_Count(*set);
	AttributeSet _set = AttributeSet_AddPrepare(set, n);
	Attribute *attrs  = _set->attributes + prev_count;

	// add attributes to set
	for(ushort i = 0; i < n; i++) {
		Attribute *attr = attrs + i;
		attr->id    = ids[i];
		attr->value = values[i];
	}

	// update pointer
	*set = _set;
}

// adds an attribute to the set
void AttributeSet_Add
(
	AttributeSet *set,    // set to update
	AttributeID attr_id,  // attribute identifier
	SIValue value         // attribute value
) {
	ASSERT(set != NULL);

	// return if set is read-only
	if(unlikely(ATTRIBUTE_SET_IS_READONLY(*set))) {
		return;
	}

#ifdef RG_DEBUG
	// value must be a valid property type
	ASSERT(SI_TYPE(value) & SI_VALID_PROPERTY_VALUE);
	// make sure attribute isn't already in set
	ASSERT(AttributeSet_Get(*set, attr_id) == ATTRIBUTE_NOTFOUND);
#endif

	AttributeSet _set = AttributeSet_AddPrepare(set, 1);

	// set attribute
	Attribute *attr = _set->attributes + _set->attr_count - 1;
	attr->id = attr_id;
	attr->value = SI_CloneValue(value);

	// update pointer
	*set = _set;
}

// add, remove or update an attribute
// this function allows NULL value to be added to the set
// returns the type of change performed
AttributeSetChangeType AttributeSet_Set_Allow_Null
(
	AttributeSet *set,    // set to update
	AttributeID attr_id,  // attribute identifier
	SIValue value         // attribute value
) {
	ASSERT(set != NULL);
	ASSERT(attr_id != ATTRIBUTE_ID_NONE);

	AttributeSet _set = *set;

	// return if set is read-only
	if(unlikely(ATTRIBUTE_SET_IS_READONLY(_set))) {
		return CT_NONE;
	}

	// validate value type
	ASSERT(SI_TYPE(value) & (SI_VALID_PROPERTY_VALUE | T_NULL));

	// update the attribute if it is already presented in the set
	if(AttributeSet_Get(_set, attr_id) != ATTRIBUTE_NOTFOUND) {
		if(AttributeSet_Update(&_set, attr_id, value)) {
			// update pointer
			*set = _set;
			// if value is NULL, indicate attribute removal
			// otherwise indicate attribute update
			return SIValue_IsNull(value) ? CT_DEL : CT_UPDATE;
		}

		// value did not change, indicate no modification
		return CT_NONE;
	}

	// can't remove a none existing attribute, indicate no modification
	if(SIValue_IsNull(value)) return CT_NONE;

	// allocate room for new attribute
	_set = AttributeSet_AddPrepare(set, 1);

	// set attribute
	Attribute *attr = _set->attributes + _set->attr_count - 1;
	attr->id = attr_id;
	attr->value = SI_CloneValue(value);

	// update pointer
	*set = _set;

	// new attribute added, indicate attribute addition
	return CT_ADD;
}

// updates existing attribute, return true if attribute been updated
bool AttributeSet_UpdateNoClone
(
	AttributeSet *set,     // set to update
	AttributeID attr_id,   // attribute identifier
	SIValue value          // new value
) {
	ASSERT(set != NULL);
	ASSERT(attr_id != ATTRIBUTE_ID_NONE);

	// return if set is read-only
	if(unlikely(ATTRIBUTE_SET_IS_READONLY(*set))) {
		return false;
	}

	// setting an attribute value to NULL removes that attribute
	if(unlikely(SIValue_IsNull(value))) {
		return _AttributeSet_Remove(set, attr_id);
	}

	SIValue *current = AttributeSet_Get(*set, attr_id);
	ASSERT(current != ATTRIBUTE_NOTFOUND);
	ASSERT(SIValue_Compare(*current, value, NULL) != 0);

	// value != current, update entity
	SIValue_Free(*current);  // free previous value
	*current = value;

	return true;
}

// updates existing attribute, return true if attribute been updated
bool AttributeSet_Update
(
	AttributeSet *set,     // set to update
	AttributeID attr_id,   // attribute identifier
	SIValue value          // new value
) {
	ASSERT(set != NULL);
	ASSERT(attr_id != ATTRIBUTE_ID_NONE);

	AttributeSet _set = *set;

	// return if set is read-only
	if(unlikely(ATTRIBUTE_SET_IS_READONLY(_set))) {
		return false;
	}

	ASSERT(_set != NULL);

	// setting an attribute value to NULL removes that attribute
	if(unlikely(SIValue_IsNull(value))) {
		return _AttributeSet_Remove(set, attr_id);
	}

	SIValue *current = AttributeSet_Get(_set, attr_id);
	ASSERT(current != ATTRIBUTE_NOTFOUND);

	// compare current value to new value, only update if current != new
	if(unlikely(SIValue_Compare(*current, value, NULL) == 0)) {
		return false;
	}

	// value != current, update entity
	SIValue_Free(*current);  // free previous value
	*current = SI_CloneValue(value);

	return true;
}

// clones attribute-set without SI values
AttributeSet AttributeSet_ShallowClone
(
	const AttributeSet set  // set to clone
) {
	// in case attribute-set is marked as read-only, clear marker
	AttributeSet _set = (AttributeSet)ATTRIBUTE_SET_CLEAR_MSB(set);

	if(_set == NULL) return NULL;

	size_t n = ATTRIBUTESET_BYTE_SIZE(set);
	AttributeSet clone = rm_malloc(n);
	clone->attr_count  = _set->attr_count;

	for(uint16_t i = 0; i < _set->attr_count; ++i) {
		Attribute *attr       = _set->attributes  + i;
		Attribute *clone_attr = clone->attributes + i;

		clone_attr->id = attr->id;
		clone_attr->value = SI_ShareValue(attr->value);
	}

    return clone;
}

// persists all attributes within given set
void AttributeSet_PersistValues
(
	const AttributeSet set  // set to persist
) {
	// return if set is read-only
	ASSERT(ATTRIBUTE_SET_IS_READONLY(set) == false);

	if(set == NULL) return;

	for (uint16_t i = 0; i < set->attr_count; ++i) {
		Attribute *attr = set->attributes + i;

		SIValue_Persist(&attr->value);
	}
}

// get attributeset's memory usage
size_t AttributeSet_memoryUsage
(
	const AttributeSet set  // set to compute memory consumption of
) {
	if(set == NULL) {
		return 0;
	}

	size_t   n = 0;  // memory consumption
	uint16_t l = AttributeSet_Count(set);

	// count memory consumption of each attribute
	for(int i = 0; i < l; i++) {
		Attribute *attr = set->attributes + i;
		SIValue v = attr->value;
		n += SIValue_memoryUsage(v);
	}

	// account for AttributeIDs
	n += (l * sizeof(AttributeID)) + sizeof(set->attr_count);

	return n;
}

// free attribute set
void AttributeSet_Free
(
	AttributeSet *set  // set to be freed
) {
	ASSERT(set != NULL);

	AttributeSet _set = *set;

	// return if set is read-only
	if(unlikely(ATTRIBUTE_SET_IS_READONLY(_set))) {
		return;
	}

	// return if set is NULL
	if(_set == NULL) {
		return;
	}

	// free all allocated properties
	for(uint16_t i = 0; i < _set->attr_count; ++i) {
		SIValue_Free(_set->attributes[i].value);
	}

	rm_free(_set);
	*set = NULL;
}

