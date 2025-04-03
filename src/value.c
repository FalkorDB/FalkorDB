/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "value.h"
#include "RG.h"
#include "graph/entities/graph_entity.h"
#include "graph/entities/node.h"
#include "graph/entities/edge.h"
#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <ctype.h>
#include <sys/param.h>
#include "util/rmalloc.h"
#include "datatypes/datatypes.h"

static inline void _SIString_ToString(SIValue str, char **buf, size_t *bufferLen,
									  size_t *bytesWritten) {
	size_t strLen = strlen(str.stringval);
	if(*bufferLen - *bytesWritten < strLen) {
		*bufferLen += strLen;
		*buf = rm_realloc(*buf, *bufferLen);
	}
	*bytesWritten += snprintf(*buf + *bytesWritten, *bufferLen, "%s", str.stringval);
}

SIValue SI_LongVal(int64_t i) {
	return (SIValue) {
		.longval = i, .type = T_INT64, .allocation = M_NONE
	};
}

SIValue SI_DoubleVal(double d) {
	return (SIValue) {
		.doubleval = d, .type = T_DOUBLE, .allocation = M_NONE
	};
}

SIValue SI_NullVal(void) {
	return (SIValue) {
		.longval = 0, .type = T_NULL, .allocation = M_NONE
	};
}

SIValue SI_BoolVal(int b) {
	return (SIValue) {
		.longval = b, .type = T_BOOL, .allocation = M_NONE
	};
}

SIValue SI_PtrVal(void *v) {
	return (SIValue) {
		.ptrval = v, .type = T_PTR, .allocation = M_NONE
	};
}

SIValue SI_Node(void *n) {
	return (SIValue) {
		.ptrval = n, .type = T_NODE, .allocation = M_VOLATILE
	};
}

SIValue SI_Edge(void *e) {
	return (SIValue) {
		.ptrval = e, .type = T_EDGE, .allocation = M_VOLATILE
	};
}

SIValue SI_Path(void *p) {
	Path *path = (Path *)p;
	return SIPath_New(path);
}

SIValue SI_EmptyArray() {
	return SIArray_New(0);
}

SIValue SI_Array(u_int64_t initialCapacity) {
	return SIArray_New(initialCapacity);
}

SIValue SI_EmptyMap() {
	return Map_New(0);
}

SIValue SI_Map(u_int64_t initialCapacity) {
	return Map_New(initialCapacity);
}

SIValue SI_Vectorf32
(
	uint32_t dim  // vector's dimension
) {
	return SIVectorf32_New(dim);
}

SIValue SI_DuplicateStringVal(const char *s) {
	return (SIValue) {
		.stringval = rm_strdup(s), .type = T_STRING, .allocation = M_SELF
	};
}

SIValue SI_ConstStringVal(const char *s) {
	return (SIValue) {
		.stringval = (char*)s, .type = T_STRING, .allocation = M_CONST
	};
}

SIValue SI_TransferStringVal(char *s) {
	return (SIValue) {
		.stringval = s, .type = T_STRING, .allocation = M_SELF
	};
}

SIValue SI_Point(float latitude, float longitude) {
	return (SIValue) {
		.type = T_POINT, .allocation = M_NONE,
			.point = {.latitude = latitude, .longitude = longitude}
	};
}

/* Make an SIValue that reuses the original's allocations, if any.
 * The returned value is not responsible for freeing any allocations,
 * and is not guaranteed that these allocations will remain in scope. */
SIValue SI_ShareValue(const SIValue v) {
	SIValue dup = v;
	// If the original value owns an allocation, mark that the duplicate shares it.
	if(v.allocation == M_SELF) dup.allocation = M_VOLATILE;
	return dup;
}

// make an SIValue that creates its own copies of the original's allocations
// if any this is not a deep clone: if the inner value holds its own references
// such as the Entity pointer to the properties of a Node or Edge
// those are unmodified
SIValue SI_CloneValue(const SIValue v) {
	if(v.allocation == M_NONE) {
		return v; // stack value; no allocation necessary
	}

	switch(v.type) {
		case T_STRING:
			// allocate a new copy of the input's string value
			return SI_DuplicateStringVal(v.stringval);

		case T_ARRAY:
			return SIArray_Clone(v);

		case T_PATH:
			return SIPath_Clone(v);

		case T_MAP:
			return Map_Clone(v);

		case T_VECTOR_F32:
			return SIVector_Clone(v);

		default:
			// handeled outside of the switch to avoid compiler warning
			break;
	}

	// copy the memory region for Node and Edge values
	// this does not modify the inner entity pointer to the value's properties
	SIValue clone;
	clone.type = v.type;
	clone.allocation = M_SELF;

	size_t size = 0;
	if(v.type == T_NODE) {
		size = sizeof(Node);
	} else if(v.type == T_EDGE) {
		size = sizeof(Edge);
	} else {
		ASSERT(false && "Encountered heap-allocated SIValue of unhandled type");
	}

	clone.ptrval = rm_malloc(size);
	memcpy(clone.ptrval, v.ptrval, size);
	return clone;
}

SIValue SI_ShallowCloneValue(const SIValue v) {
	if(v.allocation == M_CONST || v.allocation == M_NONE) return v;
	return SI_CloneValue(v);
}

// make an SIValue that shares the original's allocations but can safely expect
// those allocations to remain in scope
// this is most frequently the case for GraphEntity properties
SIValue SI_ConstValue
(
	const SIValue *v
) {
	SIValue dup = *v;
	if(v->allocation != M_NONE) dup.allocation = M_CONST;
	return dup;
}

// Clone 'v' and set v's allocation to volatile if 'v' owned the memory
SIValue SI_TransferOwnership(SIValue *v) {
	SIValue dup = *v;
	if(v->allocation == M_SELF) v->allocation = M_VOLATILE;
	return dup;
}

/* Update an SIValue marked as owning its internal allocations so that it instead is sharing them,
 * with no responsibility for freeing or guarantee regarding scope.
 * This is used in cases like performing shallow copies of scalars in Record entries. */
void SIValue_MakeVolatile(SIValue *v) {
	if(v->allocation == M_SELF) v->allocation = M_VOLATILE;
}

/* Ensure that any allocation held by the given SIValue is guaranteed to not go out
 * of scope during the lifetime of this query by copying references to volatile memory.
 * Heap allocations that are not scoped to the input SIValue, such as strings from the AST
 * or a GraphEntity property, are not modified. */
void SIValue_Persist(SIValue *v) {
	// do nothing for non-volatile values
	// for volatile values, persisting uses the same logic as cloning
	if(v->allocation == M_VOLATILE) *v = SI_CloneValue(*v);
}

/* Update an SIValue's allocation type to the provided value. */
inline void SIValue_SetAllocationType(SIValue *v, SIAllocation allocation) {
	v->allocation = allocation;
}

inline bool SIValue_IsNull(SIValue v) {
	return v.type == T_NULL;
}

inline bool SIValue_IsFalse(SIValue v) {
	ASSERT(SI_TYPE(v) ==  T_BOOL && "SIValue_IsFalse: Expected boolean");
	return !v.longval;
}

inline bool SIValue_IsTrue(SIValue v) {
	ASSERT(SI_TYPE(v) ==  T_BOOL && "SIValue_IsTrue: Expected boolean");
	return v.longval;
}

inline bool SIValue_IsNullPtr(SIValue *v) {
	return v == NULL || v->type == T_NULL;
}

const char *SIType_ToString(SIType t) {
	switch(t) {
		case T_MAP:
			return "Map";
		case T_STRING:
			return "String";
		case T_INT64:
			return "Integer";
		case T_BOOL:
			return "Boolean";
		case T_DOUBLE:
			return "Float";
		case T_PTR:
			return "Pointer";
		case T_NODE:
			return "Node";
		case T_EDGE:
			return "Edge";
		case T_ARRAY:
			return "List";
		case T_PATH:
			return "Path";
		case T_DATETIME:
			return "Datetime";
		case T_LOCALDATETIME:
			return "Local Datetime";
		case T_DATE:
			return "Date";
		case T_TIME:
			return "Time";
		case T_LOCALTIME:
			return "Local Time";
		case T_DURATION:
			return "Duration";
		case T_POINT:
			return "Point";
		case T_VECTOR_F32:
			return "Vectorf32";
		case T_NULL:
			return "Null";
		default:
			return "Unknown";
	}
}

void SIType_ToMultipleTypeString(SIType t, char *buf, size_t bufferLen) {
	// Worst case: Len(SIType names) + 19*Len(", ") + Len("Or") = 177 + 38 + 2 = 217
	ASSERT(bufferLen >= MULTIPLE_TYPE_STRING_BUFFER_SIZE);

	uint   count		= __builtin_popcount(t);
	char  *comma        = count > 2 ? ", or " : " or ";
	SIType currentType  = 1;
	size_t bytesWritten = 0;

	// Find first type
	while((t & currentType) == 0) {
		currentType = currentType << 1;
	}
	bytesWritten += snprintf(buf + bytesWritten, bufferLen, "%s", SIType_ToString(currentType));
	if(count == 1) return;

	count--;
	// Iterate over the possible SITypes except last one
	while(count > 1) {
		currentType = currentType << 1;
		if(t & currentType) {
			bytesWritten += snprintf(buf + bytesWritten, bufferLen, ", %s", SIType_ToString(currentType));
			count--;
		}
	}

	// Find last type
	do {
		currentType = currentType << 1;
	} while((t & currentType) == 0);

	// Concatenate "or" before the last SIType name
	// If there are more than two, the last comma should be present
	bytesWritten += snprintf(buf + bytesWritten, bufferLen, "%s%s", comma, SIType_ToString(currentType));
}

void SIValue_ToString
(
	SIValue v,
	char **buf,
	size_t *bufferLen,
	size_t *bytesWritten
) {
	// uint64 max and int64 min string representation requires 21 bytes
	// checkt for enough space
	if(*bufferLen - *bytesWritten < 64) {
		*bufferLen += 64;
		*buf = rm_realloc(*buf, sizeof(char) * *bufferLen);
	}

	switch(v.type) {
		case T_STRING:
			_SIString_ToString(v, buf, bufferLen, bytesWritten);
			break;
		case T_INT64:
			*bytesWritten += snprintf(*buf + *bytesWritten, *bufferLen, "%lld", (long long)v.longval);
			break;
		case T_BOOL:
			*bytesWritten += snprintf(*buf + *bytesWritten, *bufferLen, "%s", v.longval ? "true" : "false");
			break;
		case T_DOUBLE:
			{
				size_t n = snprintf(*buf + *bytesWritten, *bufferLen - *bytesWritten, "%f", v.doubleval);
				// check if there was enough space in the buffer
				if(*bytesWritten + n > *bufferLen) {
					// realloc the buffer
					*bufferLen = *bytesWritten + n + 1;
					*buf = rm_realloc(*buf, sizeof(char) * *bufferLen);

					// write it again
					snprintf(*buf + *bytesWritten, *bufferLen - *bytesWritten, "%f", v.doubleval);
				}
				*bytesWritten += n;
				break;
			}
		case T_NODE:
			Node_ToString(v.ptrval, buf, bufferLen, bytesWritten, ENTITY_ID);
			break;
		case T_EDGE:
			Edge_ToString(v.ptrval, buf, bufferLen, bytesWritten, ENTITY_ID);
			break;
		case T_ARRAY:
			SIArray_ToString(v, buf, bufferLen, bytesWritten);
			break;
		case T_MAP:
			Map_ToString(v, buf, bufferLen, bytesWritten);
			break;
		case T_PATH:
			SIPath_ToString(v, buf, bufferLen, bytesWritten);
			break;
		case T_NULL:
			*bytesWritten += snprintf(*buf + *bytesWritten, *bufferLen, "NULL");
			break;
		case T_PTR:
			*bytesWritten += snprintf(*buf + *bytesWritten, *bufferLen, "POINTER");
			break;
		case T_POINT:
			// max string length is 32 chars of string + 10 * 2 chars for the floats
			// = 52 bytes that already checked in the header of the function
			*bytesWritten += snprintf(*buf + *bytesWritten, *bufferLen, "point({latitude: %f, longitude: %f})", Point_lat(v), Point_lon(v));
			break;
		case T_VECTOR_F32:
			SIVector_ToString(v, buf, bufferLen, bytesWritten);
			break;
		default:
			// unrecognized type
			printf("unrecognized type: %d\n", v.type);
			ASSERT(false);
			break;
	}
}

int SIValue_ToDouble(const SIValue *v, double *d) {
	switch(v->type) {
	case T_DOUBLE:
		*d = v->doubleval;
		return 1;
	case T_INT64:
	case T_BOOL:
		*d = (double)v->longval;
		return 1;

	default:
		// cannot convert!
		return 0;
	}
}

SIValue SIValue_FromString(const char *s) {
	char *sEnd = NULL;

	errno = 0;
	double parsedval = strtod(s, &sEnd);
	// the input was not a complete number or represented a number that
	// cannot be represented as a double
	// create a string SIValue
	if(sEnd[0] != '\0' || errno == ERANGE) {
		return SI_DuplicateStringVal(s);
	}

	// The input was fully converted; create a double SIValue.
	return SI_DoubleVal(parsedval);
}

size_t SIValue_StringJoinLen(SIValue *strings, unsigned int string_count, const char *delimiter) {
	size_t length = 0;
	size_t elem_len;
	size_t delimiter_len = strlen(delimiter);
	/* Compute length. */
	for(int i = 0; i < string_count; i ++) {
		/* String elements representing bytes size strings,
		 * for all other SIValue types 64 bytes should be enough. */
		elem_len = (strings[i].type == T_STRING) ? strlen(strings[i].stringval) + delimiter_len : 64;
		length += elem_len;
	}

	/* Account for NULL terminating byte. */
	length++;
	return length;
}

void SIValue_StringJoin(SIValue *strings, unsigned int string_count, const char *delimiter,
						char **buf, size_t *buf_len, size_t *bytesWritten) {

	for(int i = 0; i < string_count; i ++) {
		SIValue_ToString(strings[i], buf, buf_len, bytesWritten);
		if(i < string_count - 1) *bytesWritten += snprintf(*buf + *bytesWritten, *buf_len, "%s", delimiter);
	}
}

// assumption: either a or b is a string
static SIValue SIValue_ConcatString(const SIValue a, const SIValue b) {
	size_t bufferLen = 512;
	size_t argument_len = 0;
	char *buffer = rm_calloc(bufferLen, sizeof(char));
	SIValue args[2] = {a, b};
	SIValue_StringJoin(args, 2, "", &buffer, &bufferLen, &argument_len);
	SIValue result = SI_DuplicateStringVal(buffer);
	rm_free(buffer);
	return result;
}

// assumption: either a or b is a list - static function, the caller validate types
static SIValue SIValue_ConcatList(const SIValue a, const SIValue b) {
	uint a_len = (a.type == T_ARRAY) ? SIArray_Length(a) : 1;
	uint b_len = (b.type == T_ARRAY) ? SIArray_Length(b) : 1;
	SIValue resultArray = SI_Array(a_len + b_len);

	// Append a to resultArray
	if(a.type == T_ARRAY) {
		// in thae case of a is an array
		for(uint i = 0; i < a_len; i++) SIArray_Append(&resultArray, SIArray_Get(a, i));
	} else {
		// in thae case of a is not an array
		SIArray_Append(&resultArray, a);
	}

	if(b.type == T_ARRAY) {
		// b is an array
		uint bArrayLen = SIArray_Length(b);
		for(uint i = 0; i < bArrayLen; i++) {
			SIArray_Append(&resultArray, SIArray_Get(b, i));
		}
	} else {
		// b is not an array
		SIArray_Append(&resultArray, b);
	}
	return resultArray;
}

SIValue SIValue_Add(const SIValue a, const SIValue b) {
	if(a.type == T_NULL || b.type == T_NULL) return SI_NullVal();
	if(a.type == T_ARRAY || b.type == T_ARRAY) return SIValue_ConcatList(a, b);
	if(a.type == T_STRING || b.type == T_STRING) return SIValue_ConcatString(a, b);	
	if(a.type == T_MAP || b.type == T_MAP) return Map_Merge(a, b);

	/* Only construct an integer return if both operands are integers. */
	if(a.type & b.type & T_INT64) {
		return SI_LongVal(a.longval + b.longval);
	}
	/* Return a double representation. */
	return SI_DoubleVal(SI_GET_NUMERIC(a) + SI_GET_NUMERIC(b));
}

SIValue SIValue_Subtract(const SIValue a, const SIValue b) {
	/* Only construct an integer return if both operands are integers. */
	if(a.type & b.type & T_INT64) {
		return SI_LongVal(a.longval - b.longval);
	}
	/* Return a double representation. */
	return SI_DoubleVal(SI_GET_NUMERIC(a) - SI_GET_NUMERIC(b));
}

SIValue SIValue_Multiply(const SIValue a, const SIValue b) {
	/* Only construct an integer return if both operands are integers. */
	if(SI_TYPE(a) & SI_TYPE(b) & T_INT64) {
		return SI_LongVal(a.longval * b.longval);
	}
	/* Return a double representation. */
	return SI_DoubleVal(SI_GET_NUMERIC(a) * SI_GET_NUMERIC(b));
}

SIValue SIValue_Divide(const SIValue a, const SIValue b) {
	if(SI_TYPE(a) & SI_TYPE(b) & T_INT64) {
		return SI_LongVal(SI_GET_NUMERIC(a) / SI_GET_NUMERIC(b));
	}
	return SI_DoubleVal(SI_GET_NUMERIC(a) / (double)SI_GET_NUMERIC(b));
}

// Calculate a mod n for integer and floating-point inputs.
SIValue SIValue_Modulo(const SIValue a, const SIValue n) {
	bool inputs_are_integers = SI_TYPE(a) & SI_TYPE(n) & T_INT64;
	if(inputs_are_integers) {
		// The modulo machine instruction may be used if a and n are both integers.

		int64_t res = 0;
		// workaround for https://gcc.gnu.org/bugzilla/show_bug.cgi?id=30484
		if (n.longval != -1){ // % -1 is always return 0
			res = (int64_t)a.longval % (int64_t)n.longval;
		}

		return SI_LongVal(res);
	} else {
		// Otherwise, use the library function fmod to calculate the modulo and return a double.
		return SI_DoubleVal(fmod(SI_GET_NUMERIC(a), SI_GET_NUMERIC(n)));
	}
}

int SIArray_Compare(SIValue arrayA, SIValue arrayB, int *disjointOrNull) {
	uint arrayALen = SIArray_Length(arrayA);
	uint arrayBLen = SIArray_Length(arrayB);
	// Check empty list.
	if(arrayALen == 0 && arrayBLen == 0) return 0;
	int lenDiff = arrayALen - arrayBLen;
	// Check for the common range of indices.
	uint minLength = arrayALen <= arrayBLen ? arrayALen : arrayBLen;
	// notEqual holds the first false (result != 0) comparison result between two values from the same type, which are not equal.
	int notEqual = 0;
	uint nullCounter = 0;       // Counter for the amount of null comparison.
	uint notEqualCounter = 0;   // Counter for the amount of false (compare(a,b) !=0) comparisons.

	// Go over the common range for both arrays.
	for(uint i = 0; i < minLength; i++) {
		SIValue aValue = SIArray_Get(arrayA, i);
		SIValue bValue = SIArray_Get(arrayB, i);
		// Current comparison special cases indication variable.
		int currentDisjointOrNull = 0;
		int compareResult = SIValue_Compare(aValue, bValue, &currentDisjointOrNull);
		// In case of special case such null or disjoint comparison.
		if(currentDisjointOrNull) {
			if(currentDisjointOrNull == COMPARED_NULL) nullCounter++;   // Update null comparison counter.
			// Null or disjoint comparison is also a false comparison, so increase the number of false comparisons in one.
			notEqualCounter++;
			// Set the first difference value, if not set before.
			if(notEqual == 0) notEqual = compareResult;
		} else if(compareResult != 0) {
			// In the normal false comparison case, update false comparison counter.
			notEqualCounter++;
			// Set the first difference value, if not set before.
			if(notEqual == 0) notEqual = compareResult;
		}
		// Note: In the case of compareResult = 0, there is nothing to be done.
	}
	// If all the elements in the shared range yielded false comparisons.
	if(notEqualCounter == minLength && notEqualCounter > nullCounter) return notEqual;
	// If there was a null comperison on non disjoint arrays.
	if(nullCounter && arrayALen == arrayBLen) {
		if(disjointOrNull) *disjointOrNull = COMPARED_NULL;
		return notEqual;
	}
	// If there was a difference in some member, without any null compare.
	if(notEqual) return notEqual;
	// In this state, the common range is equal. We return lenDiff, which is 0 in case the lists are equal, and not 0 otherwise.
	return lenDiff;
}

int SIValue_Compare
(
	const SIValue a,
	const SIValue b,
	int *disjointOrNull
) {
	// no special case (null or disjoint comparison) happened yet
	// if indication for such cases is required
	// first set the indication value to zero (not happen)
	if(disjointOrNull) *disjointOrNull = 0;

	// in order to be comparable, both SIValues must be from the same type
	if(a.type == b.type) {
		switch(a.type) {
		case T_INT64:
		case T_BOOL:
			return SAFE_COMPARISON_RESULT(a.longval - b.longval);
		case T_DOUBLE:
			if(isnan(a.doubleval) || isnan(b.doubleval)) {
				if(disjointOrNull) *disjointOrNull = COMPARED_NAN;
			}

			return SAFE_COMPARISON_RESULT(a.doubleval - b.doubleval);
		case T_STRING:
			if(a.stringval == b.stringval) {
				return 0;
			}
			return strcmp(a.stringval, b.stringval);
		case T_NODE:
		case T_EDGE:
			return ENTITY_GET_ID((GraphEntity *)a.ptrval) - ENTITY_GET_ID((GraphEntity *)b.ptrval);
		case T_ARRAY:
			return SIArray_Compare(a, b, disjointOrNull);
		case T_PATH:
			return SIPath_Compare(a, b);
		case T_MAP:
			return Map_Compare(a, b, disjointOrNull);
		case T_NULL:
			break;
		case T_POINT:
		{
			int lon_diff = SAFE_COMPARISON_RESULT(Point_lon(a) - Point_lon(b));
			if(lon_diff == 0)
				return SAFE_COMPARISON_RESULT(Point_lat(a) - Point_lat(b));
			return lon_diff;
		}
		case T_VECTOR_F32:
		return SIVector_Compare(a, b);
		default:
			// both inputs were of an incomparable type, like a pointer
			// or not implemented comparison yet
			ASSERT(false);
			break;
		}
	}

	// the inputs have different SITypes - compare them if they
	// are both numerics of differing types
	if(SI_TYPE(a) & SI_NUMERIC && SI_TYPE(b) & SI_NUMERIC) {
		if(isnan(SI_GET_NUMERIC(a)) || isnan(SI_GET_NUMERIC(b))) {
			if(disjointOrNull) *disjointOrNull = COMPARED_NAN;
		}

		double diff = SI_GET_NUMERIC(a) - SI_GET_NUMERIC(b);
		return SAFE_COMPARISON_RESULT(diff);
	}

	// check if either type is null
	if(a.type == T_NULL || b.type == T_NULL) {
		// check if indication is required and inform about null comparison
		if(disjointOrNull) *disjointOrNull = COMPARED_NULL;
	} else {
		// check if indication is required, and inform about disjoint comparison
		if(disjointOrNull) *disjointOrNull = DISJOINT;
	}

	// in case of disjoint or null comparison, return value type difference
	return a.type - b.type;
}

// hashes the id and properties of the node
XXH64_hash_t SINode_HashCode
(
	const SIValue v
) {
	XXH_errorcode res;
	XXH64_state_t state;
	res = XXH64_reset(&state, 0);
	UNUSED(res);
	ASSERT(res != XXH_ERROR);

	Node *n = (Node *)v.ptrval;
	int id = ENTITY_GET_ID(n);
	SIType t = SI_TYPE(v);
	XXH64_update(&state, &(t), sizeof(t));
	XXH64_update(&state, &id, sizeof(id));

	XXH64_hash_t hashCode = XXH64_digest(&state);
	return hashCode;
}

// Hashes the id and properties of the edge
XXH64_hash_t SIEdge_HashCode
(
	const SIValue v
) {
	XXH_errorcode res;
	XXH64_state_t state;
	res = XXH64_reset(&state, 0);
	UNUSED(res);
	ASSERT(res != XXH_ERROR);

	Edge *e = (Edge *)v.ptrval;
	int id = ENTITY_GET_ID(e);
	SIType t = SI_TYPE(v);
	XXH64_update(&state, &(t), sizeof(t));
	XXH64_update(&state, &id, sizeof(id));

	XXH64_hash_t hashCode = XXH64_digest(&state);
	return hashCode;
}

void SIValue_HashUpdate(SIValue v, XXH64_state_t *state) {
	// handles null value and defaults
	int64_t null = 0;
	XXH64_hash_t inner_hash;
	// in case of identical binary representation of the value,
	// we should hash the type as well
	SIType t = SI_TYPE(v);

	switch(t) {
		case T_NULL:
			XXH64_update(state, &t, sizeof(t));
			XXH64_update(state, &null, sizeof(null));
			return;
		case T_STRING:
			XXH64_update(state, &t, sizeof(t));
			XXH64_update(state, v.stringval, strlen(v.stringval));
			return;
		case T_INT64:
			// change type to numeric
			t = SI_NUMERIC;
			XXH64_update(state, &t, sizeof(t));
			XXH64_update(state, &v.longval, sizeof(v.longval));
			return;
		case T_BOOL:
			XXH64_update(state, &t, sizeof(t));
			XXH64_update(state, &v.longval, sizeof(v.longval));
			return;
		case T_DOUBLE:
			t = SI_NUMERIC;
			XXH64_update(state, &t, sizeof(t));
			// check if the double value is actually an integer
			// if so, hash it as Long
			int64_t casted = (int64_t) v.doubleval;
			double diff = v.doubleval - casted;
			if(diff != 0) XXH64_update(state, &v.doubleval, sizeof(v.doubleval));
			else XXH64_update(state, &casted, sizeof(casted));
			return;
		case T_EDGE:
			inner_hash = SIEdge_HashCode(v);
			XXH64_update(state, &inner_hash, sizeof(inner_hash));
			return;
		case T_NODE:
			inner_hash = SINode_HashCode(v);
			XXH64_update(state, &inner_hash, sizeof(inner_hash));
			return;
		case T_ARRAY:
			inner_hash = SIArray_HashCode(v);
			XXH64_update(state, &inner_hash, sizeof(inner_hash));
			return;
		case T_MAP:
			inner_hash = Map_HashCode(v);
			XXH64_update(state, &inner_hash, sizeof(inner_hash));
			return;
		case T_PATH:
			inner_hash = SIPath_HashCode(v);
			XXH64_update(state, &inner_hash, sizeof(inner_hash));
			return;
		case T_VECTOR_F32:
			inner_hash = SIVector_HashCode(v);
			XXH64_update(state, &inner_hash, sizeof(inner_hash));
			return;
			// TODO: Implement for temporal types once we support them.
		default:
			ASSERT(false);
			break;
	}
}

// hash SIValue
XXH64_hash_t SIValue_HashCode(SIValue v) {
	// initialize the hash state
	XXH64_state_t state;
	XXH_errorcode res = XXH64_reset(&state, 0);
	UNUSED(res);
	ASSERT(res != XXH_ERROR);

	// update the state with the SIValue
	SIValue_HashUpdate(v, &state);

	// generate and return the hash
	return XXH64_digest(&state);
}

// reads SIValue off of binary stream
SIValue SIValue_FromBinary
(
    FILE *stream  // stream to read value from
) {
	ASSERT(stream != NULL);

	// read value type
	SIType t;
	SIValue v;
	size_t len;  // string length

	bool     b;
	int64_t  i;
	double   d;
	Point    p;
	char    *s;
	struct SIValue *array;

	fread_assert(&t, sizeof(SIType), stream);
	switch(t) {
		case T_POINT:
			// read point from stream
			fread_assert(&p, sizeof(v.point), stream);
			v = SI_Point(p.latitude, p.longitude);
			break;
		case T_ARRAY:
			// read array from stream
			v = SIArray_FromBinary(stream);
			break;
		case T_STRING:
			// read string length from stream
			fread_assert(&len, sizeof(len), stream);
			s = rm_malloc(sizeof(char) * len);
			// read string from stream
			fread_assert(s, sizeof(char) * len, stream);
			v = SI_TransferStringVal(s);
			break;
		case T_BOOL:
			// read bool from stream
			fread_assert(&b, sizeof(b), stream);
			v = SI_BoolVal(b);
			break;
		case T_INT64:
			// read int from stream
			fread_assert(&i, sizeof(i), stream);
			v = SI_LongVal(i);
			break;
		case T_DOUBLE:
			// read double from stream
			fread_assert(&d, sizeof(d), stream);
			v = SI_DoubleVal(d);
			break;
		case T_VECTOR_F32:
			v = SIVector_FromBinary(stream, t);
			break;
		case T_NULL:
			v = SI_NullVal();
			break;
		default:
			assert(false && "unknown SIValue type");
	}

	return v;
}
			
void SIValue_Free(SIValue v) {
	// The free routine only performs work if it owns a heap allocation.
	if(v.allocation != M_SELF) return;

	switch(v.type) {
	case T_STRING:
		rm_free(v.stringval);
		v.stringval = NULL;
		return;
	case T_NODE:
	case T_EDGE:
		rm_free(v.ptrval);
		return;
	case T_ARRAY:
		SIArray_Free(v);
		return;
	case T_PATH:
		SIPath_Free(v);
		return;
	case T_MAP:
		Map_Free(v);
		break;
	case T_VECTOR_F32:
		SIVector_Free(v);
		return;
	default:
		return;
	}
}

