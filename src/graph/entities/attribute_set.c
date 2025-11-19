/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include <limits.h>

#include "RG.h"
#include "attribute_set.h"
#include "../../util/lz4/lz4.h"
#include "../../util/rmalloc.h"
#include "../../errors/errors.h"

#define COMPRESS_THRESHOLD 50

// check if attribute-set is read-only
#define ATTRIBUTE_SET_IS_READONLY(set) ((intptr_t)(set) & MSB_MASK)

// mark attribute-set as mutable
#define ATTRIBUTE_SET_CLEAR_MSB(set) (AttributeSet)(CLEAR_MSB((intptr_t)set))

// returns true if attribute_id is valid
#define VALID_ATTRIBUTE_ID(attr_id)                   \
	((attr_id) != ATTRIBUTE_ID_NONE && (attr_id) != ATTRIBUTE_ID_ALL)

// compute size in bytes of attribute set in bytes
// attribute-set size =
//    sizeof(AttributeSet) +
//    number of attributes * (sizeof(AttributeID) + sizeof(AttrValue_t))
#define ATTRIBUTESET_BYTE_SIZE(set) ((set) == NULL ?  \
		sizeof (_AttributeSet) :                      \
		sizeof (_AttributeSet) +                      \
		(sizeof (AttributeID)  + sizeof (AttrValue_t)) * (set)->attr_count)

// pointer to the begining of the attribute-set attribute ids
#define ATTRIBUTE_SET_IDS(set)                        \
			(AttributeID*)((set)->attributes)

// pointer to the begining of the attribute-set values
#define ATTRIBUTE_SET_VALS(set)                       \
			(AttrValue_t*)((set)->attributes +        \
					(sizeof(AttributeID) * (set)->attr_count))

//------------------------------------------------------------------------------
// attribute type
//------------------------------------------------------------------------------

#define ATTRTYPE_FLAG  0x80  // 1000 0000
#define ATTRTYPE_MASK  0x7F  // 0111 1111 (mask to extract base type)

// returns true if the value is marked as shared
#define AttrValue_Shared(v)      (((v)->t & ATTRTYPE_FLAG) != 0)

// mark the value as shared
#define AttrValue_SetShared(v)   ((v)->t |= ATTRTYPE_FLAG)

// clear the shared mark
#define AttrValue_ClearShared(v) ((v)->t &= ATTRTYPE_MASK)

// return the base type (without shared flag)
#define AttrValue_Type(v)        ((v)->t & ATTRTYPE_MASK)

typedef uint8_t AttrType_t;  // 1 byte attribute type
typedef enum {
	ATTR_TYPE_MAP               = 0,
	ATTR_TYPE_ARRAY             = 1,
	ATTR_TYPE_DATETIME          = 2,
	ATTR_TYPE_LOCALDATETIME     = 3,
	ATTR_TYPE_DATE              = 4,
	ATTR_TYPE_TIME              = 5,
	ATTR_TYPE_LOCALTIME         = 6,
	ATTR_TYPE_DURATION          = 7,
	ATTR_TYPE_STRING            = 8,
	ATTR_TYPE_BOOL              = 9,
	ATTR_TYPE_INT64             = 10,
	ATTR_TYPE_DOUBLE            = 11,
	ATTR_TYPE_NULL              = 12,
	ATTR_TYPE_POINT             = 13,
	ATTR_TYPE_VECTOR_F32        = 14,
	ATTR_TYPE_INTERN_STRING     = 15,
	ATTR_TYPE_COMPRESSED_STRING = 16,
	ATTR_TYPE_INVALID           = UINT8_MAX,
} _AttrType_t;

//------------------------------------------------------------------------------
// attribute value
//------------------------------------------------------------------------------

#pragma pack(push, 1)
typedef struct {
	union {
		Point point;         // point value
		void *ptrval;        // pointer value (string, map, array)
		int64_t longval;     // integer value
		time_t datetimeval;  // datetime value
		double doubleval;    // floating point value
	} ;
	AttrType_t t;            // attribute type
} AttrValue_t ;
#pragma pack(pop)

//------------------------------------------------------------------------------
// attribute-set
//------------------------------------------------------------------------------

struct _AttributeSet {
	uint16_t attr_count;         // number of attributes
	unsigned char attributes[];  // AttributeID * n, AttrValue_t * n
} ;

// map between SIValue type to Attribute type
static const AttrType_t SIType_to_AttrType[20] = {
	ATTR_TYPE_MAP,            // T_MAP           -> MAP
	ATTR_TYPE_INVALID,        // T_NODE          -> INVALID
	ATTR_TYPE_INVALID,        // T_EDGE          -> INVALID
	ATTR_TYPE_ARRAY,          // T_ARRAY         -> ARRAY
	ATTR_TYPE_INVALID,        // T_PATH          -> INVALID
	ATTR_TYPE_DATETIME,       // T_DATETIME      -> DATETIME
	ATTR_TYPE_LOCALDATETIME,  // T_LOCALDATETIME -> LOCALDATETIME
	ATTR_TYPE_DATE,           // T_DATE          -> DATE
	ATTR_TYPE_TIME,           // T_TIME          -> TIME
	ATTR_TYPE_LOCALTIME,      // T_LOCALTIME     -> LOCALTIME
	ATTR_TYPE_DURATION,       // T_DURATION      -> DURATION
	ATTR_TYPE_STRING,         // T_STRING        -> STRING
	ATTR_TYPE_BOOL,           // T_BOOL          -> BOOL
	ATTR_TYPE_INT64,          // T_INT64         -> INT64
	ATTR_TYPE_DOUBLE,         // T_DOUBLE        -> DOBULE
	ATTR_TYPE_NULL,           // T_NULL          -> NULL
	ATTR_TYPE_INVALID,        // T_PTR           -> INVALID
	ATTR_TYPE_POINT,          // T_POINT         -> POINT
	ATTR_TYPE_VECTOR_F32,     // T_VECTOR_F32    -> VECF32
	ATTR_TYPE_INTERN_STRING   // T_INTERN_STRING -> INTERN_STRING
};

// map between AttrType_t to SIType
static const SIType AttrType_to_SIType[17] = {
	[ATTR_TYPE_MAP]               = T_MAP,
	[ATTR_TYPE_ARRAY]             = T_ARRAY,
	[ATTR_TYPE_DATETIME]          = T_DATETIME,
	[ATTR_TYPE_LOCALDATETIME]     = T_LOCALDATETIME,
	[ATTR_TYPE_DATE]              = T_DATE,
	[ATTR_TYPE_TIME]              = T_TIME,
	[ATTR_TYPE_LOCALTIME]         = T_LOCALTIME,
	[ATTR_TYPE_DURATION]          = T_DURATION,
	[ATTR_TYPE_STRING]            = T_STRING,
	[ATTR_TYPE_BOOL]              = T_BOOL,
	[ATTR_TYPE_INT64]             = T_INT64,
	[ATTR_TYPE_DOUBLE]            = T_DOUBLE,
	[ATTR_TYPE_NULL]              = T_NULL,
	[ATTR_TYPE_POINT]             = T_POINT,
	[ATTR_TYPE_VECTOR_F32]        = T_VECTOR_F32,
	[ATTR_TYPE_INTERN_STRING]     = T_INTERN_STRING,
	[ATTR_TYPE_COMPRESSED_STRING] = T_STRING,
};

// map between AttrType_t to SIValue allocation type
static const SIAllocation AttrType_to_Allocation[17] = {
	[ATTR_TYPE_MAP]               = M_VOLATILE,
	[ATTR_TYPE_ARRAY]             = M_VOLATILE,
	[ATTR_TYPE_DATETIME]          = M_NONE,
	[ATTR_TYPE_LOCALDATETIME]     = M_NONE,
	[ATTR_TYPE_DATE]              = M_NONE,
	[ATTR_TYPE_TIME]              = M_NONE,
	[ATTR_TYPE_LOCALTIME]         = M_NONE,
	[ATTR_TYPE_DURATION]          = M_NONE,
	[ATTR_TYPE_STRING]            = M_VOLATILE,
	[ATTR_TYPE_BOOL]              = M_NONE,
	[ATTR_TYPE_INT64]             = M_NONE,
	[ATTR_TYPE_DOUBLE]            = M_NONE,
	[ATTR_TYPE_NULL]              = M_NONE,
	[ATTR_TYPE_POINT]             = M_NONE,
	[ATTR_TYPE_VECTOR_F32]        = M_VOLATILE,
	[ATTR_TYPE_INTERN_STRING]     = M_VOLATILE,
	[ATTR_TYPE_COMPRESSED_STRING] = M_VOLATILE
};

// returns the AttributeID of the ith attribute
static inline AttributeID _GetAttrID
(
	const AttributeSet set,  // attribute-set
	uint16_t i               // attribute index
) {
	ASSERT (i < AttributeSet_Count (set)) ;

	return (ATTRIBUTE_SET_IDS (set))[i] ;
}

// returns the value of the ith attribute
static inline AttrValue_t *_GetAttrVal
(
	const AttributeSet set,  // attribute-set
	uint16_t i               // attribute index
) {
	ASSERT (i < AttributeSet_Count (set)) ;

	return ATTRIBUTE_SET_VALS (set) + i ;
}

// convert SIValue to AttrValue_t
static inline void _AttrValueFromSIValue
(
	AttrValue_t* restrict attr,  // [output] attribute to set
	SIValue* restrict v          // value to convert
) {
	ASSERT (v    != NULL) ;
	ASSERT (attr != NULL) ;

	// left most active bit position
    int bit_idx = __builtin_clz (v->type) ;
	bit_idx = (sizeof (SIType) * 8) - 1 - bit_idx ;

	attr->t = SIType_to_AttrType[bit_idx] ;  // attribute type
	attr->ptrval = v->ptrval ;               // attribute value
}

static bool _AttrValueFromLongString
(
	AttrValue_t* restrict attr,
	const char* restrict str
) {
	ASSERT (str  != NULL) ;
	ASSERT (attr != NULL) ;

	const size_t orig_len = strlen (str) ;
	ASSERT (orig_len >= COMPRESS_THRESHOLD) ;

	const int max_comp = LZ4_compressBound (orig_len) ;

	// allocate worst-case buffer + 8-byte header
	size_t alloc_size = 8 + max_comp ;
	uint8_t *buf = rm_malloc (alloc_size) ;

	// compress into buf + 8
	const uint32_t comp_len = LZ4_compress_default ( str, (char *)(buf + 8),
			orig_len, max_comp) ;

	ASSERT (comp_len > 0) ;

	// actual required size = header + compressed payload
	const size_t actual_size = 8 + comp_len ;

	// check if compression ratio isn't that great e.g. less then 20%
	// store original string
	if (actual_size >= orig_len * 0.8) {
		rm_free (buf) ;
		return false ;
	}

	// shrink buffer to actual size
	if (alloc_size > actual_size) {
		buf = rm_realloc (buf, actual_size) ;
		ASSERT(buf != NULL);
	}

	uint32_t *header = (uint32_t*)buf ;
	header[0] = comp_len ;  // write compressed length
	header[1] = orig_len ;  // write original length

	attr->t      = ATTR_TYPE_COMPRESSED_STRING ;
	attr->ptrval = buf ;

	return true ;
}

static char *DecompressLongString
(
	const AttrValue_t *attr
) {
    ASSERT (attr         != NULL) ;
    ASSERT (attr->t      == ATTR_TYPE_COMPRESSED_STRING) ;
    ASSERT (attr->ptrval != NULL) ;

    const uint8_t  *buf    = (const uint8_t *) attr->ptrval ;
	const uint32_t *header = (const uint32_t*) buf ;

    uint32_t comp_len = header[0] ;  // read compressed size
    uint32_t orig_len = header[1] ;  // read original size

    const char *comp_data = (const char *) (buf + 8) ;

    // allocate output string (+1 for NULL terminator)
    char *out = rm_malloc (orig_len + 1) ;
    int decompressed = LZ4_decompress_safe (comp_data, out, comp_len, orig_len) ;

	ASSERT (decompressed == orig_len) ;
	ASSERT (decompressed > COMPRESS_THRESHOLD) ;

    out[orig_len] = '\0';  // safe—it’s a real C string now
    return out;
}

// converts AttrValue_t to SIValue
static inline void _AttrValueToSIValue
(
	SIValue* restrict v,              // [output] SIValue
	const AttrValue_t* restrict attr  // attribute to convert
) {
	ASSERT (v    != NULL) ;
	ASSERT (attr != NULL) ;

	// get attribute type
	AttrType_t t = AttrValue_Type (attr) ;

	if (unlikely (t == ATTR_TYPE_COMPRESSED_STRING)) {
		// decompress string
		v->ptrval     = DecompressLongString (attr) ;  // set value
		v->type       = T_STRING ;                     // set type
		v->allocation = M_SELF ;                       // set allocation type

		return ;
	}

	// construct SIValue
	v->ptrval     = attr->ptrval ;                // set value
	v->type       = AttrType_to_SIType     [t] ;  // set type
	v->allocation = AttrType_to_Allocation [t] ;  // set allocation type
}

// free AttrValue_t
static void _AttrValue_Free
(
	AttrValue_t **attr  // attribute to free
) {
	ASSERT (attr != NULL && *attr != NULL) ;

	AttrValue_t *_attr = *attr ;

	// do not free shared attribute
	if (unlikely (AttrValue_Shared (_attr))) {
		*attr = NULL ;
		return ;
	}

	switch (AttrValue_Type (_attr)) {
		// direct free
		case ATTR_TYPE_STRING:
		case ATTR_TYPE_VECTOR_F32:
			rm_free (_attr->ptrval) ;
			break ;

		// indirect free via SIValue_Free
		case ATTR_TYPE_MAP:
		case ATTR_TYPE_ARRAY:
		case ATTR_TYPE_INTERN_STRING:
			{
				SIValue v ;
				_AttrValueToSIValue (&v, _attr) ;
				v.allocation = M_SELF ;
				SIValue_Free (v) ;
				break ;
			}

		// nothing to free
		default:
			break ;
	}

	*attr = NULL ;
}

// expand attribute-set by n attributes
static AttributeSet AttributeSet_Accommodate
(
	AttributeSet *set,  // attribute-set to expand
	uint16_t n          // number of attributes to add
) {
	ASSERT (set != NULL) ;

	AttributeSet _set = *set ;

	// assert overflow
	ASSERT (AttributeSet_Count (_set) <= UINT16_MAX - n) ;

	// allocate room for new attribute
	if (_set == NULL) {
		_set = rm_calloc (1, sizeof(_AttributeSet) +
				n * (sizeof (AttributeID) + sizeof (AttrValue_t))) ;
		_set->attr_count = n ;
	} else {
		uint16_t prev_n = _set->attr_count ;
		_set->attr_count += n ;
		_set = rm_realloc (_set, ATTRIBUTESET_BYTE_SIZE (_set)) ;

		// shift values to the right
		void *dst = ATTRIBUTE_SET_VALS (_set) ;
		void *src = _set->attributes + (prev_n * sizeof (AttributeID)) ;
		memmove (dst, src, sizeof (AttrValue_t) * prev_n) ;
	}

	*set = _set ;
	return _set ;
}

// removes the ith attribute from the attribute-set
static void AttributeSet_RemoveIdx
(
	AttributeSet *set,  // attribute-set
	uint16_t i          // attribute index
) {
	ASSERT (set != NULL) ;
	ASSERT (i < AttributeSet_Count(*set)) ;

	AttributeSet _set = *set ;

	// if this is the last attribute, free the attribute-set
	const uint16_t n = _set->attr_count ;
	if (n == 1) {
		AttributeSet_Free (set) ;
		return ;
	}

	// free attribute
	AttrValue_t *attrs = ATTRIBUTE_SET_VALS (_set) ;
	AttrValue_t *attr = attrs + i ;
	_AttrValue_Free (&attr) ;

	//--------------------------------------------------------------------------
	// shrink attribute-set
	//--------------------------------------------------------------------------

	AttributeID *ids = ATTRIBUTE_SET_IDS(_set) ;

	// overwrite deleted attribute with last attribute
	ids[i]   = ids[n-1]   ;
	attrs[i] = attrs[n-1] ;

	_set->attr_count-- ;  // update attribute count

	// shrink
	memmove (&ids[n-1], attrs, sizeof (AttrValue_t) * (_set->attr_count)) ;
	*set = rm_realloc (_set, ATTRIBUTESET_BYTE_SIZE (_set)) ;
}

// replace the ith attribute with value
static bool AttributeSet_Replace
(
	AttributeSet set,  // attribute-set
	uint16_t i,        // attribute index
	SIValue *value,    // new value
	bool clone         // clone value
) {
	ASSERT (set   != NULL) ;
	ASSERT (value != NULL) ;
	ASSERT (i     < AttributeSet_Count (set)) ;

	SIValue current ;
	AttributeID attr_id ;
	AttributeSet_GetIdx (set, i, &attr_id, &current) ;

	// compare current value to new value, only update if current != new
	if (unlikely (SIValue_Compare (current, *value, NULL) == 0)) {
		return false ;
	}

	// value != current, update entity
	AttrValue_t *attr = ATTRIBUTE_SET_VALS (set) + i ;
	_AttrValue_Free (&attr) ;

	attr = ATTRIBUTE_SET_VALS (set) + i ;

	if (clone) {
		SIValue v = SI_CloneValue (*value) ;
		_AttrValueFromSIValue (attr, &v) ;
	} else {
		ASSERT (SI_ALLOCATION (value) != M_VOLATILE) ;
		_AttrValueFromSIValue (attr, value) ;
	}

	return true ;
}

// returns number of attributes within the set
inline uint16_t AttributeSet_Count
(
	const AttributeSet set  // attribute-set
) {
	const AttributeSet _set = ATTRIBUTE_SET_CLEAR_MSB (set) ;
	return (_set == NULL) ? 0 : _set->attr_count ;
}

// checks if attribute-set contains attribute
bool AttributeSet_Contains
(
	const AttributeSet set,  // attribute-set
	AttributeID id,          // attribute id to lookup
	uint16_t *idx            // [optional][output] attribute index
) {
	ASSERT (VALID_ATTRIBUTE_ID (id)) ;
	const AttributeSet _set = ATTRIBUTE_SET_CLEAR_MSB (set) ;

	if (_set == NULL) {
		return false ;
	}

	uint16_t n = _set->attr_count ;
	AttributeID *attrs = ATTRIBUTE_SET_IDS (_set) ;

	// TODO: use SIMD or support sort
	for (uint16_t i = 0; i < n; i++) {
		if (attrs[i] == id) {
			if (idx != NULL) {
				*idx = i ;
			}
			return true ;
		}
	}

	return false ;
}

// removes an attribute from set
// returns true if attribute was removed false otherwise
bool AttributeSet_Remove
(
	AttributeSet *set,   // attribute-set
	AttributeID attr_id  // attribute ID to remove
) {
	ASSERT (set != NULL) ;
	ASSERT (!ATTRIBUTE_SET_IS_READONLY (*set)) ;
	ASSERT (VALID_ATTRIBUTE_ID (attr_id)) ;

	AttributeSet _set = *set ;

	// locate attribute position
	uint16_t i;
	if (!AttributeSet_Contains (*set, attr_id, &i)) {
		// attribute is missing from the set, nothing to remove
		return false ;
	}

	AttributeSet_RemoveIdx (set, i) ;
	return true ;
}

// returns the ith attribute ID
AttributeID AttributeSet_GetKey
(
	const AttributeSet set,  // attribute-set
	int16_t i                // i
) {
	ASSERT (set != NULL) ;
	ASSERT (i < AttributeSet_Count (set)) ;

	// in case attribute-set is marked as read-only, clear marker
	const AttributeSet _set = ATTRIBUTE_SET_CLEAR_MSB (set) ;

	return *(ATTRIBUTE_SET_IDS(_set) + i) ;
}

// retrieves a value from set
// if attr_id isn't in the set returns false
bool AttributeSet_Get
(
	const AttributeSet set,  // set to retieve attribute from
	AttributeID attr_id,     // attribute id
	SIValue *v               // [output] value
) {
	ASSERT (v != NULL) ;

	// in case attribute-set is marked as read-only, clear marker
	const AttributeSet _set = ATTRIBUTE_SET_CLEAR_MSB (set) ;

	// empty set / unknown attribute id
	if (unlikely (_set == NULL || !VALID_ATTRIBUTE_ID (attr_id))) {
		*v = SI_NullVal () ;
		return false ;
	}

	//--------------------------------------------------------------------------
	// search for attribute
	//--------------------------------------------------------------------------

	uint16_t i;  // attribute index

	if (AttributeSet_Contains (_set, attr_id, &i)) {
		AttrValue_t *attr = _GetAttrVal (_set, i) ;
		_AttrValueToSIValue (v, attr) ;
		return true ;
	}

	*v = SI_NullVal () ;
	return false ;
}

// retrieves the ith attribute from attribute-set
void AttributeSet_GetIdx
(
	const AttributeSet set,  // attribute-set to retieve attribute from
	uint16_t i,              // index of attribute
	AttributeID *attr_id,    // [output] attribute ID
	SIValue *v               // [output] value
) {
	ASSERT (v       != NULL) ;
	ASSERT (set     != NULL) ;
	ASSERT (attr_id != NULL) ;

	// in case attribute-set is marked as read-only, clear marker
	const AttributeSet _set = ATTRIBUTE_SET_CLEAR_MSB (set) ;
	ASSERT (i < _set->attr_count) ;

	// set attribute id
	*attr_id = _GetAttrID (_set, i) ;

	// extract attribute value
	_AttrValueToSIValue (v, _GetAttrVal (_set, i)) ;
}

// adds new attributes to the attribute-set
// all attributes MUST NOT be in the set
void AttributeSet_Add
(
	AttributeSet *set,  // attribute-set to update
	AttributeID *ids,   // attribute ids
	SIValue *values,    // attribute values
	uint16_t n,         // number of attributes
	bool allowNull,     // accept NULLs
	bool clone          // clone values
) {
	ASSERT (!ATTRIBUTE_SET_IS_READONLY (*set)) ;
	ASSERT (set != NULL) ;

	if (unlikely (n == 0)) {
		return ;
	}

	// validate values type, values must be a valid property type
#ifdef RG_DEBUG
	SIType t = SI_VALID_PROPERTY_VALUE ;
	if (allowNull == true) {
		t |= T_NULL ;
	}

	for (uint16_t i = 0; i < n; i++) {
		ASSERT (SI_TYPE (values[i]) & t) ;

		// make sure attribute isn't already in set
		ASSERT (AttributeSet_Contains (*set, ids[i], NULL) == false) ;

		// if clone == false make sure value isn't volatile
		ASSERT (clone || SI_ALLOCATION (values + i) != M_VOLATILE) ;

		// ensure no duplicate attribute IDs within this batch
		for (ushort j = i + 1; j < n; j++) {
			ASSERT (ids[i] != ids[j]) ;
		}
	}
#endif

	// make sure attribute-set has enough space to accommodate all attributes
	ushort prev_count = AttributeSet_Count (*set) ;
	AttributeSet _set = AttributeSet_Accommodate (set, n) ;

	AttributeID *set_ids  = ATTRIBUTE_SET_IDS (_set)  + prev_count ;
	AttrValue_t *set_vals = ATTRIBUTE_SET_VALS (_set) + prev_count ;

	// add attributes
	memcpy (set_ids, ids, n * sizeof (AttributeID)) ;

	for (uint16_t i = 0; i < n; i++) {
		AttrValue_t *attr = set_vals + i ;
		SIValue *v = values + i ;

		// compress string if:
		// 1. string is not interned
		// 2. string is long enough
		// 3. compress ratio is good enough
		if (SI_TYPE (*v)  & T_STRING   &&
			!(SI_TYPE (*v) & T_INTERN) &&
			strnlen (v->stringval, COMPRESS_THRESHOLD) == COMPRESS_THRESHOLD) {
			if (_AttrValueFromLongString (attr, v->stringval)) {
				// free string if we're suppose to take ownership
				if (!clone) {
					SIValue_Free (*v) ;
				}
				continue ;
			}
		}

		if (clone) {
			SIValue v_clone = SI_CloneValue (*v) ;
			_AttrValueFromSIValue (attr, &v_clone) ;
		} else {
			_AttrValueFromSIValue (attr, v) ;
		}
	}
}

// add, remove or update an attribute
// returns the type of change performed
AttributeSetChangeType AttributeSet_Update
(
	AttributeSet *set,    // set to update
	AttributeID attr_id,  // attribute identifier
	SIValue value,        // new value
	bool allowNull,       // accept NULLs
	bool clone            // clone value
) {
	ASSERT (!ATTRIBUTE_SET_IS_READONLY (set)) ;
	ASSERT (set     != NULL) ;
	ASSERT (VALID_ATTRIBUTE_ID (attr_id)) ;

	AttributeSet _set = *set ;

	// validate value type
	SIType t = SI_VALID_PROPERTY_VALUE ;
	if (allowNull) {
		t |= T_NULL ;
	}
	ASSERT (SI_TYPE (value) & t) ;

	uint16_t i;
	bool remove   = SIValue_IsNull (value) ;
	bool contains = AttributeSet_Contains (_set, attr_id, &i) ;

	// trying to remove a nonexisting attribuet
	if (unlikely (!contains && remove)) {
		return CT_NONE ;
	}

	if (contains) {

		//----------------------------------------------------------------------
		// remove attribute
		//----------------------------------------------------------------------

		if (remove) {
			AttributeSet_RemoveIdx (set, i) ;
			return CT_DEL ;
		}

		//----------------------------------------------------------------------
		// update attribute
		//----------------------------------------------------------------------

		if (AttributeSet_Replace (_set, i, &value, clone)) {
			return CT_UPDATE ;
		}

		// value did not change, indicate no modification
		return CT_NONE ;
	}

	//--------------------------------------------------------------------------
	// add attribute
	//--------------------------------------------------------------------------

	ASSERT (!contains && !remove) ;

	AttributeSet_Add (set, &attr_id, &value, 1, allowNull, clone) ;

	// new attribute added, indicate attribute addition
	return CT_ADD ;
}

// shallow clones attribute-set
AttributeSet AttributeSet_ShallowClone
(
	const AttributeSet set  // set to clone
) {
	ASSERT (!ATTRIBUTE_SET_IS_READONLY (set)) ;

	if (set == NULL) {
		return NULL ;
	}

	size_t n = ATTRIBUTESET_BYTE_SIZE (set) ;
	AttributeSet clone = rm_malloc (n) ;

	// copy attribute set
	memcpy (clone, set, n) ;

	// mark each attribute as shared
	// the clone won't be able to free values
	uint16_t l = AttributeSet_Count (set) ;
	AttrValue_t *vals = ATTRIBUTE_SET_VALS (clone) ;

	for (uint16_t i = 0; i < l; i++) {
		AttrValue_SetShared (vals + i) ;
	}

	return clone ;
}

//
void AttributeSet_PersistValues
(
	AttributeSet set  // attribute-set to persist
) {
	ASSERT (!ATTRIBUTE_SET_IS_READONLY (set)) ;
	if (set == NULL) {
		return ;
	}

	uint16_t n = AttributeSet_Count (set) ;
	AttrValue_t *attrs = ATTRIBUTE_SET_VALS (set) ;

	for (uint16_t i = 0; i < n; i++) {
		AttrValue_t *attr = attrs + i;

		if (AttrValue_Shared (attr)) {
			AttrValue_ClearShared (attr) ;

			// get attribute type
			AttrType_t t = AttrValue_Type (attr) ;
			if (AttrType_to_Allocation[t] == M_VOLATILE) {
				SIValue v ;
				_AttrValueToSIValue (&v, attr) ;
				v = SI_CloneValue (v) ;
				_AttrValueFromSIValue (attr, &v) ;
			}
		}
	}
}

// compute hash for attribute-set
XXH64_hash_t AttributeSet_HashCode
(
	const AttributeSet set  // attribute-set to compute hash of
) {
	const AttributeSet _set = ATTRIBUTE_SET_CLEAR_MSB (set) ;

	XXH64_state_t *hash = XXH64_createState ();  // create a hash state
	XXH_errorcode res = XXH64_reset (hash, 0) ;
	ASSERT (res != XXH_ERROR) ;

	if (_set != NULL) {
		uint16_t n = AttributeSet_Count (_set) ;
		res = XXH64_update (hash, &n, sizeof (n)) ;
		ASSERT(res != XXH_ERROR);

		AttributeID *attr_ids = ATTRIBUTE_SET_IDS (_set) ;
		res = XXH64_update (hash, attr_ids, sizeof (AttributeID) * n) ;
		ASSERT(res != XXH_ERROR);

		for (uint16_t i = 0; i < n; i++) {
			SIValue v ;
			_AttrValueToSIValue (&v, _GetAttrVal (_set, i)) ;

			// update hash with the hashval of the associated SIValue
			XXH64_hash_t value_hash = SIValue_HashCode (v) ;
			res = XXH64_update (hash, &value_hash, sizeof (value_hash)) ;
			ASSERT (res != XXH_ERROR) ;
		}
	}

	XXH64_hash_t const digest = XXH64_digest (hash) ;
	XXH64_freeState (hash) ;

	return digest ;
}

// get attributeset's memory usage
size_t AttributeSet_memoryUsage
(
	const AttributeSet set  // set to compute memory consumption of
) {
	ASSERT (!ATTRIBUTE_SET_IS_READONLY (set)) ;

	if (set == NULL) {
		return 0 ;
	}

	size_t total = 0 ;  // memory consumption
	uint16_t n = AttributeSet_Count (set) ;

	// count memory consumption of each attribute
	for (uint16_t i = 0; i < n; i++) {
		AttrValue_t *attr = _GetAttrVal (set, i) ;
		AttrType_t t = attr->t ;

		if (AttrType_to_Allocation[t] == M_NONE) {
			// attribute is a "simple" primitive,  e.g. int64_t
			// no extra memory is allocated elsewhere
			total += sizeof (AttrValue_t) ;
			continue ;
		}

		switch (attr->t) {
			case ATTR_TYPE_STRING:
			case ATTR_TYPE_INTERN_STRING:
				total += strlen (attr->ptrval) ;  // misleading for intern
				break ;

			case ATTR_TYPE_COMPRESSED_STRING:
				// header + compressed length
				total += 8 + ((size_t*)attr->ptrval)[0] ;
				break ;

			case ATTR_TYPE_MAP:
			case ATTR_TYPE_ARRAY:
			case ATTR_TYPE_VECTOR_F32:
			{
				SIValue v ;
				AttributeID attr_id ;
				AttributeSet_GetIdx (set, i, &attr_id, &v) ;
				total += SIValue_memoryUsage (v) ;
				break ;
			}

			default:
				assert (false && "unhandeled attribute type") ;
		}
	}

	// account for AttributeIDs
	total += (n * sizeof (AttributeID)) + sizeof (set->attr_count) ;

	return total ;
}

// defrag attribute-set
// present each heap allocated attribute to the allocator in the hope
// it will be able reduce fragmentation by relocating the memory
void AttributeSet_Defrag
(
	AttributeSet set,          // attribute-set
	RedisModuleDefragCtx *ctx  // defrag context
) {
	ASSERT (set != NULL) ;
	ASSERT (ctx != NULL) ;
	ASSERT (!ATTRIBUTE_SET_IS_READONLY (set)) ;

	uint16_t n = AttributeSet_Count (set) ;
	AttrValue_t *attrs = ATTRIBUTE_SET_VALS (set) ;

	for (uint16_t i = 0; i < n ; i++) {
		AttrValue_t *attr = attrs + i ;
		switch (AttrValue_Type(attr)) {
			case ATTR_TYPE_MAP:
			case ATTR_TYPE_ARRAY:
			case ATTR_TYPE_STRING:
			case ATTR_TYPE_VECTOR_F32:
			{
				void *moved = RedisModule_DefragAlloc (ctx, attr->ptrval) ;
				if (moved != NULL) {
					attr->ptrval = moved ;
				}
				break ;
			}

			default:
				break ;
		}
	}
}

// free attribute set
void AttributeSet_Free
(
	AttributeSet *set  // set to be freed
) {
	ASSERT(set != NULL);

	AttributeSet _set = *set;

	// do not free if attribute-set is read-only
	if (unlikely (ATTRIBUTE_SET_IS_READONLY (_set))) {
		return ;
	}

	// return if set is NULL
	if (_set == NULL) {
		return ;
	}

	// free all allocated properties
	for(uint16_t i = 0; i < _set->attr_count; i++) {
		AttrValue_t *attr = ATTRIBUTE_SET_VALS (_set) + i;
		_AttrValue_Free (&attr) ;
	}

	rm_free (_set) ;
	*set = NULL ;
}

