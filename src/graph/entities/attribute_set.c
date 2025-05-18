/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "attribute_set.h"
#include "../../util/rmalloc.h"
#include "../../errors/errors.h"

#include <math.h>
#include <stdio.h>
#include <float.h>
#include <limits.h>
#include <stdbool.h>

// type of attributes
typedef enum : uint8_t {
	ATTR_TYPE_INT8       = 0,   // 1 byte  int
	ATTR_TYPE_INT16      = 1,   // 2 bytes int
	ATTR_TYPE_INT32      = 2,   // 4 bytes int
	ATTR_TYPE_INT64      = 3,   // 8 bytes int
	ATTR_TYPE_BOOL_TRUE  = 4,   // 0 bytes true value
	ATTR_TYPE_BOOL_FALSE = 5,   // 0 bytes false value
	ATTR_TYPE_FLOAT      = 6,   // 4 bytes floating point number
	ATTR_TYPE_DOUBLE     = 7,   // 8 bytes floating point number
	ATTR_TYPE_STRING     = 8,   // 8 bytes string
	ATTR_TYPE_NULL       = 9,   // 0 bytes NULL value
	ATTR_TYPE_POINT      = 10,  // 8 bytes point
	ATTR_TYPE_VECTOR_F32 = 11,  // 8 bytes pointer to vector
	ATTR_TYPE_ARRAY      = 12,  // 8 bytes pointer to array
	ATTR_TYPE_MAP        = 13,  // 8 bytes pointer to map
	ATTR_TYPE_INVALID    = 14   // 0 bytes invalid attribute type
} AttrType;

// size mapping table
static const uint8_t attr_type_to_size[14] = {
    [ATTR_TYPE_INT8]       = 1,  // 1 byte  int
    [ATTR_TYPE_INT16]      = 2,  // 2 bytes int
    [ATTR_TYPE_INT32]      = 4,  // 4 bytes int
    [ATTR_TYPE_INT64]      = 8,  // 8 bytes int
    [ATTR_TYPE_BOOL_TRUE]  = 0,  // 0 bytes true value
    [ATTR_TYPE_BOOL_FALSE] = 0,  // 0 bytes false value
    [ATTR_TYPE_FLOAT]      = 4,  // 4 bytes floating point
    [ATTR_TYPE_DOUBLE]     = 8,  // 8 bytes floating point
    [ATTR_TYPE_STRING]     = 8,  // 8 bytes string pointer
    [ATTR_TYPE_NULL]       = 0,  // 0 bytes NULL value
    [ATTR_TYPE_POINT]      = 8,  // 8 bytes point
    [ATTR_TYPE_VECTOR_F32] = 8,  // 8 bytes pointer to vector
    [ATTR_TYPE_ARRAY]      = 8,  // 8 bytes pointer to array
    [ATTR_TYPE_MAP]        = 8   // 8 bytes pointer to map
};

// map between AttrType to SIType
static const SIType attr_type_to_sivalue_type[14] = {
    [ATTR_TYPE_INT8]       = T_INT64,       // INT8   -> T_INT64
	[ATTR_TYPE_INT16]      = T_INT64,       // INT16  -> T_INT64
	[ATTR_TYPE_INT32]      = T_INT64,       // INT32  -> T_INT64
	[ATTR_TYPE_INT64]      = T_INT64,       // INT64  -> T_INT64
	[ATTR_TYPE_BOOL_TRUE]  = T_BOOL,        // TRUE   -> T_BOOL
	[ATTR_TYPE_BOOL_FALSE] = T_BOOL,        // FALSE  -> T_BOOL
	[ATTR_TYPE_FLOAT]      = T_DOUBLE,      // FLOAT  -> T_DOUBLE
	[ATTR_TYPE_DOUBLE]     = T_DOUBLE,      // DOUBLE -> T_DOUBLE
	[ATTR_TYPE_STRING]     = T_STRING,      // STRING -> T_STRING
	[ATTR_TYPE_NULL]       = T_NULL,        // NULL   -> T_NULL
	[ATTR_TYPE_POINT]      = T_POINT,       // POINT  -> T_POINT
	[ATTR_TYPE_VECTOR_F32] = T_VECTOR_F32,  // VECF32 -> T_VECTOR_F32
	[ATTR_TYPE_ARRAY]      = T_ARRAY,       // ARRAY  -> T_ARRAY
	[ATTR_TYPE_MAP]        = T_MAP          // MAP    -> T_MAP
};


// map between AttrType allocation type true for heap
// false for stack
static const bool attr_type_heap_allocated[14] = {
    [ATTR_TYPE_INT8]       = false,
	[ATTR_TYPE_INT16]      = false,
	[ATTR_TYPE_INT32]      = false,
	[ATTR_TYPE_INT64]      = false,
	[ATTR_TYPE_BOOL_TRUE]  = false,
	[ATTR_TYPE_BOOL_FALSE] = false,
	[ATTR_TYPE_FLOAT]      = false,
	[ATTR_TYPE_DOUBLE]     = false,
	[ATTR_TYPE_STRING]     = true,
	[ATTR_TYPE_NULL]       = false,
	[ATTR_TYPE_POINT]      = false,
	[ATTR_TYPE_VECTOR_F32] = true,
	[ATTR_TYPE_ARRAY]      = true,
	[ATTR_TYPE_MAP]        = true
};

// map between SIValue type to Attribute type
static const AttrType sivalue_type_to_attr_type[19] = {
	ATTR_TYPE_MAP,        // T_MAP           -> MAP
	ATTR_TYPE_INVALID,    // T_NODE          -> INVALID
	ATTR_TYPE_INVALID,    // T_EDGE          -> INVALID
	ATTR_TYPE_ARRAY,      // T_ARRAY         -> ARRAY
	ATTR_TYPE_INVALID,    // T_PATH          -> INVALID
	ATTR_TYPE_INVALID,    // T_DATETIME      -> INVALID
	ATTR_TYPE_INVALID,    // T_LOCALDATETIME -> INVALID
	ATTR_TYPE_INVALID,    // T_DATE          -> INVALID
	ATTR_TYPE_INVALID,    // T_TIME          -> INVALID
	ATTR_TYPE_INVALID,    // T_LOCALTIME     -> INVALID
	ATTR_TYPE_INVALID,    // T_DURATION      -> INVALID
	ATTR_TYPE_STRING,     // T_STRING        -> STRING
	ATTR_TYPE_BOOL_TRUE,  // T_BOOL          -> TRUE / FALSE
	ATTR_TYPE_INT64,      // T_INT64         -> INT8 / INT16 / INT32 / INT64
	ATTR_TYPE_DOUBLE,     // T_DOUBLE        -> DOBULE / FLOAT
	ATTR_TYPE_NULL,       // T_NULL          -> NULL
	ATTR_TYPE_INVALID,    // T_PTR           -> INVALID
	ATTR_TYPE_POINT,      // T_POINT         -> INVALID
	ATTR_TYPE_VECTOR_F32  // T_VECTOR_F32    -> VECF32
};

// skips attribute id
#define SKIP_ATTR_ID(buff) (buff) += sizeof(AttributeID)

// skips attribute type
#define SKIP_ATTR_TYPE(buff) (buff) += sizeof(AttrType)

// skips attribute value
#define SKIP_ATTR_VALUE(buff) (buff) += attr_type_to_size[*(buff-1)]

// skips entire attribute
#define SKIP_ATTR(buff)                                         \
	SKIP_ATTR_ID(buff);                                         \
	(buff) += attr_type_to_size[(buff)[0]] + sizeof(AttrType);

// get attribute id
#define GET_ATTR_ID(buff)                                       \
	*((AttributeID*) (buff))

// get attribute type
#define GET_ATTR_TYPE(buff)                                     \
	*((AttrType*)((buff) + sizeof(AttributeID)))

// get attribute value
#define GET_ATTR_VALUE(buff)                                    \
	(buff) + sizeof(AttributeID) + sizeof(AttrType)

// get attribute's value as type t
#define GET_ATTR_VALUE_AS(t, attr)                              \
	(*((t*)(attr)))

#define GET_ATTR_TOTAL_SIZE(attr)                               \
	sizeof(AttributeID) +                                       \
	sizeof(AttrType)    +                                       \
	attr_type_to_size[(attr + sizeof(AttributeID))[0]]

// set attribute id
#define SET_ATTR_ID(buff, id)                                   \
	*(AttributeID*)(buff) = id;                                 \
	SKIP_ATTR_ID(buff);

// set attribute type
#define SET_ATTR_TYPE(buff, t)                                  \
	offset[0] = t;                                              \
	SKIP_ATTR_TYPE(buff);

// converts between SIType to AttrType
static inline AttrType SIValue_To_AttrType
(
	SIValue *v  // value to determine attribute type of
) {
	SIType t = v->type;
	AttrType at = sivalue_type_to_attr_type[__builtin_ctz(t)];

	if(at == ATTR_TYPE_INT64) {
		// see if int can be represented by int8_t, int16_t or int32_t
		int64_t n = v->longval;

		// check if the number fits in 1 byte (signed char: -128 to 127)
		if(n >= INT8_MIN && n <= INT8_MAX) {
			at = ATTR_TYPE_INT8;
		}

		// check if the number fits in 2 bytes (signed short: -32,768 to 32,767)
		else if(n >= INT16_MIN && n <= INT16_MAX) {
			at = ATTR_TYPE_INT16;
		}

		// check if the number fits in 4 bytes (signed int: -2,147,483,648 to 2,147,483,647)
		else if(n >= INT32_MIN && n <= INT32_MAX) {
			at = ATTR_TYPE_INT32;
		}

	} else if(at == ATTR_TYPE_DOUBLE) {
		// see if double can be represented by a float without loosing
		// precision
		bool   can_convert = false;
		double value       = v->doubleval;
		float  float_value = (float)value;
		double round_trip  = (double)float_value;

		// check if the round-trip conversion preserves the value
		// we need to handle NaN specially because NaN != NaN
		if(unlikely(isnan(value))) {
			can_convert = isnan(round_trip);
		} else {
			const double epsilon = DBL_EPSILON;
			can_convert = fabs(value - round_trip) <= epsilon * fabs(value);
		}

		// able to convert without major precision loss
		if(can_convert) {
			at = ATTR_TYPE_FLOAT;
		}
	} else if(at == ATTR_TYPE_BOOL_TRUE) {
		if(v->longval == 0) {
			at = ATTR_TYPE_BOOL_FALSE;
		}
	}

	return at;
}

// populate an SIValue from an attribute
static void _attribute_to_sivalue
(
	AttrType t,        // type of attribute
	const void *attr,  // attribute
	SIValue *v         // [output] sivalue to populate
) {
	ASSERT(v    != NULL);
	ASSERT(attr != NULL);

	switch(t) {
		case ATTR_TYPE_INT8:
			*v = SI_LongVal(GET_ATTR_VALUE_AS(int8_t, attr));
			break;

		case ATTR_TYPE_INT16:
			*v = SI_LongVal(GET_ATTR_VALUE_AS(int16_t,attr));
			break;

		case ATTR_TYPE_INT32:
			*v = SI_LongVal(GET_ATTR_VALUE_AS(int32_t, attr));
			break;

		case ATTR_TYPE_INT64:
			*v = SI_LongVal(GET_ATTR_VALUE_AS(int64_t, attr));
			break;

		case ATTR_TYPE_BOOL_TRUE:
			*v = SI_BoolVal(true);
			break;

		case ATTR_TYPE_BOOL_FALSE:
			*v = SI_BoolVal(false);
			break;

		case ATTR_TYPE_FLOAT:
			*v = SI_DoubleVal(GET_ATTR_VALUE_AS(float, attr));
			break;

		case ATTR_TYPE_DOUBLE:
			*v = SI_DoubleVal(GET_ATTR_VALUE_AS(double, attr));
			break;

		case ATTR_TYPE_POINT:
			*v = SI_Point(GET_ATTR_VALUE_AS(Point, attr).latitude,
					GET_ATTR_VALUE_AS(Point, attr).longitude);
			break;

		case ATTR_TYPE_STRING:
			*v = SI_ConstStringVal(GET_ATTR_VALUE_AS(char*, CLEAR_MSB(attr)));
			break;

		case ATTR_TYPE_NULL:
			*v = SI_NullVal();
			break;

		// pointer based SIValues
		case ATTR_TYPE_MAP:
		case ATTR_TYPE_ARRAY:
		case ATTR_TYPE_VECTOR_F32:
			*v = SI_PtrVal(GET_ATTR_VALUE_AS(void*, CLEAR_MSB(attr)));
			break;

		case ATTR_TYPE_INVALID:
			assert(false);
			break;
	}

	// set SIValue type
	v->type = attr_type_to_sivalue_type[t];

	// TODO: not sure about the allocation type
	v->allocation = M_SELF;
}

struct _AttributeSet {
	uint8_t extra_usable_size;  // amount of extra usable memory we can use
	uint16_t attr_count;        // number of attributes
	char attributes[];          // key value pair of attributes
};

// compute size of attribute set in bytes
#define ATTRIBUTESET_BYTE_SIZE(set) ((set) == NULL ? \
		sizeof(_AttributeSet) :                      \
		sizeof(_AttributeSet) + sizeof(Attribute) * (set)->attr_count)

// mark attribute-set as mutable
#define ATTRIBUTE_SET_CLEAR_MSB(set) (CLEAR_MSB((intptr_t)set))

// returned value for a missing attribute
SIValue ATTRIBUTE_NOTFOUND = (SIValue) {
	.longval = 0, .type = T_NULL
};

// returns a pointer to the start of the `i`-th attribute in the attribute set
// if `i` is out of bounds or `set` is NULL, returns NULL
static char *_LocateAttrByIdx
(
	const AttributeSet set,  // set to search attribute in
	uint16_t i               // the ith attribute
) {
	// empty set
	if(unlikely(set == NULL)) {
		return NULL;
	}

	// index out of bounds
	if(unlikely(i >= set->attr_count)) {
		return NULL;
	}

	// place offset at the begining of the attributes buffer
	char *offset = set->attributes;

	// as long as we didn't get to the ith attribute
	for(; i > 0; i--) {
		SKIP_ATTR(offset);
	}

	return offset;
}

// returns a pointer to the start of the attribute with the given `id`
// if the attribute is not found or `set` is NULL, returns NULL
static char *_LocateAttrById
(
	const AttributeSet set,  // set to search attribute in
	AttributeID id           // attribute id to locate
) {
	// empty set
	if(unlikely(set == NULL)) {
		return NULL;
	}

	// place offset at the begining of the attributes buffer
	char *offset = set->attributes;

	// as long as we didn't get to the desiered attribute
	for(uint16_t i = 0; i < set->attr_count; i++) {
		// get the current attribute type
		AttributeID _id = *(AttributeID*)offset;

		// found the attribute we're looking for
		if(_id == id) {
			return offset;
		}

		SKIP_ATTR(offset);
	}

	return NULL;
}

// free individual attribute
static void _freeAttribute
(
	char *attr  // pointer to the begining of the attribute
) {
	// get the attribute type
	AttrType t = GET_ATTR_TYPE(attr);

	// free attribute if it's heap allocated
	if(attr_type_heap_allocated[t]) {
		const void *val = (const void*)GET_ATTR_VALUE(attr);

		// check if attr's MSB is on, if so don't free attribute
		if(!MSB_ON(val)) {
			// convert from Attribute to SIValue
			SIValue v;

			_attribute_to_sivalue(t, val, &v);

			// free SIValue
			SIValue_Free(v);
		}
	}
}

// removes an attribute from the given attribute set
// returns `true` if the attribute was successfully removed, otherwise `false`.
// if the attribute set contains only one attribute, the entire set is freed.
static bool _AttributeSet_Remove
(
	AttributeSet *set,   // set to remove attribute from
	AttributeID attr_id  // attribute ID to remove
) {
	ASSERT(set     != NULL);
	ASSERT(*set    != NULL);
	ASSERT(attr_id != ATTRIBUTE_ID_NONE);

	AttributeSet _set = *set;
	const uint16_t attr_count = _set->attr_count;

	// attribute-set can't be read-only
	ASSERT(ATTRIBUTE_SET_IS_READONLY(_set) == false);

	// locate attribute position
	char *offset = _LocateAttrById(_set, attr_id);

	// unable to locate attribute
	if(offset == NULL) {
		return false;
	}

	// if this is the last attribute, free the attribute-set
	if(_set->attr_count == 1) {
		AttributeSet_Free(set);
		return true;
	}

	_freeAttribute(offset);

	// attribute-set allocation size
	size_t n = RedisModule_MallocSize(_set);

	// shift left by the size of the removed attribute
	size_t shift_amount = GET_ATTR_TOTAL_SIZE(offset);

	// shift remaining attributes to fill the gap
    memmove(offset, offset + shift_amount,
			((char*)_set + n) - (offset + shift_amount));

	// shrink attribute set memory
	*set = rm_realloc(_set, n - shift_amount);

	// attribute removed
	return true;
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
// v is set to a special constant value ATTRIBUTE_NOTFOUND
// and false is returned
bool AttributeSet_Get
(
	const AttributeSet set,  // set to retieve attribute from
	AttributeID id,          // attribute identifier
	SIValue *v               // [output] attribute
) {
	// in case attribute-set is marked as read-only, clear marker
	AttributeSet _set = (AttributeSet)ATTRIBUTE_SET_CLEAR_MSB(set);

	if(_set == NULL || id == ATTRIBUTE_ID_NONE) {
		*v = ATTRIBUTE_NOTFOUND;
		return false;
	}

	char *attr = _LocateAttrById(set, id);

	// attribute wasn't found
	if(attr == NULL) {
		*v = ATTRIBUTE_NOTFOUND;
		return false;
	}

	AttrType t = GET_ATTR_TYPE(attr);
	_attribute_to_sivalue(t, GET_ATTR_VALUE(attr), v);

	return true;
}

// retrieves a value from set by index
bool AttributeSet_GetIdx
(
	const AttributeSet set,  // set to retieve attribute from
	uint16_t i,              // index of the property
	AttributeID *attr_id,    // [output] attribute identifier
	SIValue *v               // [output] attribute
) {
	// empty result set
	if(unlikely(set == NULL)) {
		if(v != NULL) {
			*v = ATTRIBUTE_NOTFOUND;
		}
		return false;
	}

	// in case attribute-set is marked as read-only, clear marker
	AttributeSet _set = (AttributeSet)ATTRIBUTE_SET_CLEAR_MSB(set);

	ASSERT(_set != NULL);

	if(unlikely(i >= _set->attr_count)) {
		if(v != NULL) {
			*v = ATTRIBUTE_NOTFOUND;
		}

		return false;
	}

	char *attr = _LocateAttrByIdx(_set, i);
	ASSERT(attr != NULL);

	// get the attribute id
	if(attr_id != NULL) {
		*attr_id = GET_ATTR_ID(attr);
	}

	// convert from attribute to SIValue
	if(v != NULL) {
		_attribute_to_sivalue(GET_ATTR_TYPE(attr), GET_ATTR_VALUE(attr), v);
	}

	return true;
}

// returns true if attribute set contains attribute
bool AttributeSet_Contains
(
	const AttributeSet set,  // set to search
	AttributeID attr_id      // attribute id to locate
) {
	ASSERT(attr_id != ATTRIBUTE_ID_ALL)
	ASSERT(attr_id != ATTRIBUTE_ID_NONE);

	// empty attribute-set
	if(unlikely(set == NULL)) {
		return false;
	}

	// in case attribute-set is marked as read-only, clear marker
	AttributeSet _set = (AttributeSet)ATTRIBUTE_SET_CLEAR_MSB(set);

	return _LocateAttrById(_set, attr_id) != NULL;
}

// extends the attribute set by `n` bytes
// if the attribute set is NULL, a new attribute set is allocated
// returns the updated attribute set
static AttributeSet AttributeSet_Grow
(
	AttributeSet *set,  // set to update
	ushort n            // number of bytes to add
) {
	ASSERT(set != NULL);

	AttributeSet _set = *set;

	// attribute-set can't be read-only
	ASSERT(ATTRIBUTE_SET_IS_READONLY(_set) == false);

	// allocate room for new attribute
	if(_set == NULL) {
		n += sizeof(_AttributeSet);
		_set = rm_malloc(n);
		_set->attr_count = 0;
		_set->extra_usable_size = RedisModule_MallocUsableSize(_set) - n;
		//printf("Asked for: %d, allocated: %zu, usable: %zu, waste: %zu\n",
		//		n,
		//		RedisModule_MallocSize(_set),
		//		RedisModule_MallocUsableSize(_set),
		//		RedisModule_MallocSize(_set) - n);
	} else if(_set->extra_usable_size >= n) {
		// set can accommodate additional n bytes
		_set->extra_usable_size -= n;
	} else {
		n -= _set->extra_usable_size;
		n += RedisModule_MallocUsableSize(_set);
		_set = rm_realloc(_set, n);
		_set->extra_usable_size = RedisModule_MallocUsableSize(_set) - n;
	}

	return _set;
}

// set attribute to value of the given type 't'
#define SET_ATTR_VALUE_AS(t, attr, v) *((t*)attr) = v

// adds one or more attributes to the set without cloning the SIValue
// ensures attributes are of valid types and do not already exist in the set
// returns immediately if the set is read-only
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

	// validate values type
	// values must be of a valid property type
#ifdef RG_DEBUG
	SIType t = SI_VALID_PROPERTY_VALUE;
	if(allowNull == true) {
		t |= T_NULL;
	}

	for(ushort i = 0; i < n; i++) {
		ASSERT(SI_TYPE(values[i]) & t);
		// make sure attribute isn't already in set
		SIValue v;
		ASSERT(AttributeSet_Get(*set, ids[i], &v) == false);
		// make sure value isn't volotile
		ASSERT(SI_ALLOCATION(values + i) != M_VOLATILE);
	}
#endif

	//--------------------------------------------------------------------------
	// compute number of required bytes
	//--------------------------------------------------------------------------

	AttrType ats[n];
	size_t nbytes = 0;
	for(ushort i = 0; i < n; i++) {
		AttrType at = SIValue_To_AttrType(values + i);
		nbytes += attr_type_to_size[at] +
				  sizeof(AttributeID)   +
				  sizeof(AttrType);
		ats[i] = at;
	}

	ushort prev_count = AttributeSet_Count(*set);
	AttributeSet _set = AttributeSet_Grow(set, nbytes);

	_set->attr_count += n;

	char *offset = _set->attributes;

	//--------------------------------------------------------------------------
	// place offset right after the last attribute
	//--------------------------------------------------------------------------

	if(prev_count > 0) {
		offset = _LocateAttrByIdx(_set, prev_count - 1);

		// skip pass the last attribute
		SKIP_ATTR(offset);
	}

	//--------------------------------------------------------------------------
	// add attributes to set
	//--------------------------------------------------------------------------

	for(ushort i = 0; i < n; i++) {
		// store attribute id
		SET_ATTR_ID(offset, ids[i])

		// store attribute type
		SET_ATTR_TYPE(offset, ats[i])

		// store value
		switch(ats[i]) {
			case ATTR_TYPE_INT8:
				SET_ATTR_VALUE_AS(int8_t, offset, values[i].longval);
				break;

			case ATTR_TYPE_INT16:
				SET_ATTR_VALUE_AS(int16_t, offset, values[i].longval);
				break;

			case ATTR_TYPE_INT32:
				SET_ATTR_VALUE_AS(int32_t, offset, values[i].longval);
				break;

			case ATTR_TYPE_INT64:
				SET_ATTR_VALUE_AS(int64_t, offset, values[i].longval);
				break;

			case ATTR_TYPE_NULL:
			case ATTR_TYPE_BOOL_TRUE:
			case ATTR_TYPE_BOOL_FALSE:
				break;

			case ATTR_TYPE_FLOAT:
				SET_ATTR_VALUE_AS(float, offset, values[i].doubleval);
				break;

			case ATTR_TYPE_DOUBLE:
				SET_ATTR_VALUE_AS(double, offset, values[i].doubleval);
				break;

			case ATTR_TYPE_POINT:
				SET_ATTR_VALUE_AS(Point, offset, values[i].point);
				break;

			case ATTR_TYPE_MAP:
			case ATTR_TYPE_ARRAY:
			case ATTR_TYPE_STRING:
			case ATTR_TYPE_VECTOR_F32:
				SET_ATTR_VALUE_AS(void*, offset, values[i].ptrval);
				break;

			case ATTR_TYPE_INVALID:
				assert(false);
				break;
		}

		SKIP_ATTR_VALUE(offset);
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
	ASSERT(set     != NULL);
	ASSERT(attr_id != ATTRIBUTE_ID_NONE);

	SIValue v = SI_CloneValue(value);
	AttributeSet_AddNoClone(set, &attr_id, &v, 1, false);
}

// modifies an attribute in the set:
// - adds a new attribute if it does not exist
// - updates an existing attribute if found
// - removes the attribute if the new value is NULL
// - returns the type of change performed (CT_ADD, CT_UPDATE, CT_DEL, CT_NONE)
AttributeSetChangeType AttributeSet_Set_Allow_Null
(
	AttributeSet *set,    // set to update
	AttributeID attr_id,  // attribute identifier
	SIValue value         // attribute value
) {
	ASSERT(set     != NULL);
	ASSERT(attr_id != ATTRIBUTE_ID_NONE);

	AttributeSet _set = *set;

	// return if set is read-only
	if(unlikely(ATTRIBUTE_SET_IS_READONLY(_set))) {
		return CT_NONE;
	}

	// validate value type
	ASSERT(SI_TYPE(value) & (SI_VALID_PROPERTY_VALUE | T_NULL));

	// update the attribute if it is already presented in the set
	SIValue v;
	bool exists  = AttributeSet_Get(_set, attr_id, &v);
	bool is_null = SIValue_IsNull(value);

	if(exists) {

		//----------------------------------------------------------------------
		// update
		//----------------------------------------------------------------------

		// setting an attribute value to NULL removes that attribute
		if(unlikely(is_null)) {
			_AttributeSet_Remove(set, attr_id);
			return CT_DEL;
		}

		return AttributeSet_Update(set, attr_id, value, true) ?
			CT_UPDATE :
			CT_NONE;
	}

	// can't remove a none existing attribute, indicate no modification
	if(is_null) {
		return CT_NONE;
	}

	//--------------------------------------------------------------------------
	// add
	//--------------------------------------------------------------------------

	AttributeSet_Add(set, attr_id, value);

	// new attribute added, indicate attribute addition
	return CT_ADD;
}

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
) {
	ASSERT(set     != NULL);
	ASSERT(attr_id != ATTRIBUTE_ID_NONE);

	AttributeSet _set = *set;
	ASSERT(_set != NULL);

	// return if set is read-only
	if(unlikely(ATTRIBUTE_SET_IS_READONLY(_set))) {
		return false;
	}

	// setting an attribute value to NULL removes that attribute
	if(unlikely(SIValue_IsNull(value))) {
		return _AttributeSet_Remove(set, attr_id);
	}

	SIValue current;
	bool res = AttributeSet_Get(_set, attr_id, &current);
	if(!res) {
		return false;
	}

	// compare current value to new value, only update if current != new
	if(unlikely(SIValue_Compare(current, value, NULL) == 0)) {
		return false;
	}

	// TODO: no need to look up the attribute again
	// remove old attribute
	res = _AttributeSet_Remove(set, attr_id);
	ASSERT(res);

	if(clone) {
		value = SI_CloneValue(value);
	}

	// TODO: inplace update when possible
	AttributeSet_AddNoClone(set, &attr_id, &value, 1, false);

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

	size_t n = RedisModule_MallocSize(_set);
	AttributeSet clone = rm_malloc(n);
	clone = (AttributeSet)memcpy(clone, _set, n);

	// for each heap allocated value, set MSB ON indicate value shouldn't
	// be free
	char *buff = clone->attributes;
	for(uint16_t i = 0; i < clone->attr_count; ++i) {
		// persist only heap allocated attributes
		if(attr_type_heap_allocated[GET_ATTR_TYPE(buff)]) {
			char *val = GET_ATTR_VALUE(buff);
			val[0] = SET_MSB(val[0]);
		}
		SKIP_ATTR(buff);
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

	char *buff = set->attributes;

	// scan through each attribute
	for(uint16_t i = 0; i < set->attr_count; ++i) {
		// persist only heap allocated attributes
		if(attr_type_heap_allocated[GET_ATTR_TYPE(buff)]) {
			// get SIValue representation of attribute and free it
			SIValue v;
			char *attr = GET_ATTR_VALUE(buff);
			_attribute_to_sivalue(GET_ATTR_TYPE(buff), attr, &v);

			SIValue_Persist(&v);

			// TODO: store value back into buffer
			ASSERT(false && "todo");
		}

		// move on to the next attribute
		SKIP_ATTR(buff);
	}
}

// get direct access to the set attributes
char *AttributeSet_Attributes
(
	const AttributeSet set  // set to retrieve attributes from
) {
	if(set == NULL) {
		return NULL;
	}

	return set->attributes;
}

// get attributeset's memory usage
size_t AttributeSet_memoryUsage
(
	const AttributeSet set,  // set to compute memory consumption of
	uint64_t node_id
) {
	if(set == NULL) {
		return 0;
	}

	size_t   n = 0;  // memory consumption
	uint16_t l = AttributeSet_Count(set);

	// count memory consumption of each attribute
	for(int i = 0; i < l; i++) {
		SIValue v;
		AttributeID id;
		AttributeSet_GetIdx(set, i, &id, &v);
		if(v.type == T_STRING && v.stringval == NULL) {
			v = SIValue_FromDisk(node_id, id);
		}
		n += SIValue_memoryUsage(v);
		if(v.type == T_STRING && v.stringval == NULL) {
			free(v.stringval);
			v.stringval = NULL;
		}
	}

	// account for AttributeIDs
	n += (l * sizeof(AttributeID)) + sizeof(set->attr_count);

	return n;
}

// frees an attribute set and all its allocated attributes
// - if the set is NULL or read-only, it is not modified
// - heap-allocated attributes are deallocated before freeing the set
// - the pointer is set to NULL to prevent dangling references
void AttributeSet_Free
(
	AttributeSet *set  // set to be freed
) {
	ASSERT(set  != NULL);
	ASSERT(*set != NULL);

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
	char *buff = _set->attributes;

	// scan through attributes
	// free heap allocated attributes
	for(uint16_t i = 0; i < _set->attr_count; i++) {
		// free current attribute
		_freeAttribute(buff);

		SKIP_ATTR(buff);
	}

	rm_free(_set);
	*set = NULL;
}

