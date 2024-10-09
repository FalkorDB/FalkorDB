/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include <limits.h>

#include "RG.h"
#include "attribute_set.h"
#include "../../util/arr.h"
#include "../../util/rmalloc.h"
#include "../../datatypes/map.h"
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
	AttributeSet *set,    // set to modify
	AttributeID attr_id,  // attribute id
	const char **path,    // [optional] sub path
	uint8_t n
) {
	AttributeSet _set = *set;

	// trying to remove from a none existing attribute set
	if(unlikely(_set == NULL)) return false;

	const uint16_t attr_count = _set->attr_count;

	// attribute-set can't be read-only
	ASSERT(ATTRIBUTE_SET_IS_READONLY(_set) == false);

	// locate attribute position
	for (uint16_t i = 0; i < attr_count; ++i) {
		if(attr_id != _set->attributes[i].id) {
			continue;
		}

		//----------------------------------------------------------------------
		// attribute located
		//----------------------------------------------------------------------

		// remove sub path
		// e.g. n.a.b.c = NULL
		if(unlikely(path != NULL)) {
			SIValue v = _set->attributes[i].value;
			// fail of value is not a map
			if(SI_TYPE(v) != T_MAP) return false;

			// try to remove sub path
			return Map_RemovePath(v, path, n);
		}

		// if this is the last attribute free the attribute-set
		if(_set->attr_count == 1) {
			AttributeSet_Free(set);
			return true;
		}

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
	for(uint16_t i = 0; i < _set->attr_count; i++) {
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
	const char **path,    // [optional] sub path
	uint8_t n,            // sub path length
	SIValue value         // attribute value
) {
	ASSERT(set     != NULL);
	ASSERT(attr_id != ATTRIBUTE_ID_NONE);

	// validate value type
	ASSERT(SI_TYPE(value) & (SI_VALID_PROPERTY_VALUE | T_NULL));

	AttributeSet _set = *set;

	// return if set is read-only
	if(unlikely(ATTRIBUTE_SET_IS_READONLY(_set))) {
		return CT_NONE;
	}

	AttributeSetChangeType ret      = CT_NONE;
	bool                   update   = false;
	bool                   remove   = SIValue_IsNull(value);
	bool                   sub_path = (path !=  NULL);
	SIValue                *curr    = NULL;

	if(!remove) {
		curr = AttributeSet_Get(_set, attr_id);
		update = (curr != ATTRIBUTE_NOTFOUND);
	}

	if(sub_path) {

		//----------------------------------------------------------------------
		// remove sub-path
		//----------------------------------------------------------------------

		if(remove) {
			return _AttributeSet_Remove(set, attr_id, path, n) ?
				CT_UPDATE :
				CT_NONE;
		}

		//----------------------------------------------------------------------
		// update sub-path
		//----------------------------------------------------------------------

		else if(update) {
			if(SI_TYPE(*curr) != T_MAP) {
				// trying to access a path of a none map object
				// fail
				return CT_NONE;
			}

			ret = CT_UPDATE;
		}

		//----------------------------------------------------------------------
		// new sub-path
		//----------------------------------------------------------------------

		else {
			_set = AttributeSet_AddPrepare(set, 1);
			*set = _set;

			// set attribute
			Attribute *attr = _set->attributes + _set->attr_count - 1;

			// set attribute
			attr->id    = attr_id;
			attr->value = SI_Map(1);
			curr        = &attr->value;

			ret = CT_ADD;
		}

		// add / update path
		return Map_AddPath(curr, path, n, value) ?
			ret:
			CT_NONE;
	}

	//--------------------------------------------------------------------------
	// root attribute
	//--------------------------------------------------------------------------

	if(remove) {
		ret = _AttributeSet_Remove(set, attr_id, NULL) ?
			CT_DEL :
			CT_NONE;
	}
	else if(update) {
		ret = AttributeSet_Update(set, attr_id, value) ?
			CT_UPDATE :
			CT_NONE;
	} else {
		// allocate room for new attribute
		_set = AttributeSet_AddPrepare(set, 1);
		*set = _set;

		// set attribute
		Attribute *attr = _set->attributes + _set->attr_count - 1;
		attr->id        = attr_id;
		attr->value     = SI_CloneValue(value);

		ret = CT_ADD;
	}

	return ret;
}

// updates existing attribute, return true if attribute been updated
bool AttributeSet_UpdateNoClone
(
	AttributeSet *set,    // set to update
	AttributeID attr_id,  // attribute identifier
	SIValue value         // new value
) {
	ASSERT(set     != NULL);
	ASSERT(attr_id != ATTRIBUTE_ID_NONE);

	// return if set is read-only
	if(unlikely(ATTRIBUTE_SET_IS_READONLY(*set))) {
		return false;
	}

	// setting an attribute value to NULL removes that attribute
	if(unlikely(SIValue_IsNull(value))) {
		return _AttributeSet_Remove(set, attr_id, NULL);
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
	ASSERT(set     != NULL);
	ASSERT(attr_id != ATTRIBUTE_ID_NONE);

	AttributeSet _set = *set;

	// return if set is read-only
	if(unlikely(ATTRIBUTE_SET_IS_READONLY(_set))) {
		return false;
	}

	ASSERT(_set != NULL);

	// setting an attribute value to NULL removes that attribute
	if(unlikely(SIValue_IsNull(value))) {
		return _AttributeSet_Remove(set, attr_id, NULL);
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
	if(set == NULL) return;

	// return if set is read-only
	ASSERT(ATTRIBUTE_SET_IS_READONLY(set) == false);

	for (uint16_t i = 0; i < set->attr_count; ++i) {
		Attribute *attr = set->attributes + i;

		SIValue_Persist(&attr->value);
	}
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

