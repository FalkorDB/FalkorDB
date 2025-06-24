/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdbool.h>
#include <sys/types.h>
#include "xxhash.h"

/* Type defines the supported types by the system. The types are powers
 * of 2 so they can be used in bitmasks of matching types.
 *
 * The order of these values is significant, as the delta between values of
 * differing types is used to maintain the Cypher-defined global sort order
 * in the SIValue_Order routine. */

#define T_INTERN (1<< 19)

typedef enum {
	T_MAP           = (1 << 0),
	T_NODE          = (1 << 1),
	T_EDGE          = (1 << 2),
	T_ARRAY         = (1 << 3),
	T_PATH          = (1 << 4),
	T_DATETIME      = (1 << 5),
	T_LOCALDATETIME = (1 << 6),
	T_DATE          = (1 << 7),
	T_TIME          = (1 << 8),
	T_LOCALTIME     = (1 << 9),
	T_DURATION      = (1 << 10),
	T_STRING        = (1 << 11),
	T_BOOL          = (1 << 12),  // shares 'longval' representation in SIValue  union
	T_INT64         = (1 << 13),
	T_DOUBLE        = (1 << 14),
	T_NULL          = (1 << 15),
	T_PTR           = (1 << 16),
	T_POINT         = (1 << 17),
	T_VECTOR_F32    = (1 << 18),
	T_INTERN_STRING = T_INTERN | T_STRING,
} SIType;

typedef enum {
	M_NONE     = 0,         // SIValue is not heap-allocated
	M_SELF     = (1 << 0),  // SIValue is responsible for freeing its reference
	M_VOLATILE = (1 << 1),  // SIValue does not own its reference and may go out of scope
	M_CONST    = (1 << 2)   // SIValue does not own its allocation, but its access is safe
} SIAllocation;

#define T_VECTOR (T_VECTOR_F32)
#define SI_TYPE(value) (value).type
#define SI_ALLOCATION(value) (value)->allocation

// type aliases
#define SI_NUMERIC              (T_INT64 | T_DOUBLE)  // numbrical types
#define SI_GRAPHENTITY          (T_NODE | T_EDGE)     // graph entity types
#define SI_TEMPORAL             (T_DATETIME | T_LOCALDATETIME | T_DATE | T_TIME | T_LOCALTIME | T_DURATION)  // temporal types
#define SI_INDEXABLE            (SI_NUMERIC | T_BOOL | T_STRING | T_POINT | T_VECTOR | SI_TEMPORAL)  // indexable types
#define SI_VALID_PROPERTY_VALUE (T_POINT | T_ARRAY | T_STRING | T_INTERN_STRING | T_BOOL | SI_NUMERIC | T_VECTOR | SI_TEMPORAL)  // all valid attribute types
#define SI_ALL                  (T_MAP | T_NODE | T_EDGE | T_ARRAY | T_PATH | T_STRING | T_BOOL | T_INT64 | T_DOUBLE | T_NULL | T_PTR | T_POINT | T_VECTOR | SI_TEMPORAL)  // all supported types

// any values (except durations) are comparable with other values of the same type
// integer and floating-point values are also comparable with each other
#define SI_VALUES_ARE_COMPARABLE(a, b) (((a).type == (b).type) || ((a).type & SI_NUMERIC && (b).type & SI_NUMERIC))

/* Retrieve the numeric associated with an SIValue without explicitly
 * assigning it a type. */
#define SI_GET_NUMERIC(v) ((v).type == T_DOUBLE ? (v).doubleval : (v).longval)

/* Build an integer return value for a comparison routine in the style of strcmp.
 * This is necessary to construct safe returns when the delta between
 * two values is < 1.0 (and would thus be rounded to 0)
 * or the delta is too large to fit in a 32-bit integer. */
#define SAFE_COMPARISON_RESULT(a) SIGN(a)

/* Returns 1 if argument is positive, -1 if argument is negative,
 * and 0 if argument is zero.*/
#define SIGN(a) ((a) > 0) - ((a) < 0)

#define DISJOINT INT_MAX
#define COMPARED_NULL INT_MIN
#define COMPARED_NAN INT_MIN+1

// Minimum buffer size for string generated in SIType_ToMultipleTypeString
#define MULTIPLE_TYPE_STRING_BUFFER_SIZE 256

struct Pair;

typedef struct Point {
	float latitude;   // 32 bit
	float longitude;  // 32 bit
} Point;

typedef struct SIValue {
	union {
		int64_t longval;        // integer value
		time_t datetimeval;     // datetime value
		double doubleval;       // floating point value
		char *stringval;        // string value
		void *ptrval;           // pointer value
		struct Pair *map;       // map value
		struct SIValue *array;  // array value
		Point point;            // point value
	};
	SIType type;                // type of value
	SIAllocation allocation;    // allocation type
} SIValue;

// functions to construct an SIValue from a specific input type
SIValue SI_EmptyMap();
SIValue SI_EmptyArray();
SIValue SI_Node(void *n);
SIValue SI_Edge(void *e);
SIValue SI_Path(void *p);
SIValue SI_NullVal(void);
SIValue SI_BoolVal(int b);
SIValue SI_PtrVal(void *v);
SIValue SI_LongVal(int64_t i);
SIValue SI_DoubleVal(double d);
SIValue SI_Vectorf32(uint32_t dim);
SIValue SI_DateTime(time_t datetime);
SIValue SI_Map(u_int64_t initialCapacity);
SIValue SI_Array(u_int64_t initialCapacity);
SIValue SI_Point(float latitude, float longitude);

// create a new duration object
SIValue SI_Duration
(
	double years,    // years count
	double months,   // months count
	double weeks,    // weeks count
	double days,     // days count
	double hours,    // hours count
	double minutes,  // minutes count
	double seconds   // seconds count
);

// Duplicate and ultimately free the input string.
SIValue SI_DuplicateStringVal(const char *s);

// Neither duplicate nor assume ownership of input string.
SIValue SI_ConstStringVal(const char *s);

// Don't duplicate input string, but assume ownership.
SIValue SI_TransferStringVal(char *s);

// create an interned string
SIValue SI_InternStringVal
(
	const char *s // string to intern
);

/* Functions for copying and guaranteeing memory safety for SIValues. */
// SI_ShareValue creates an SIValue that shares all of the original's allocations.
SIValue SI_ShareValue(const SIValue v);

// SI_CloneValue creates an SIValue that duplicates all of the original's allocations.
SIValue SI_CloneValue(const SIValue v);

// SI_CloneValue creates an SIValue that duplicates all of the original's self-owned or volatile allocations.
SIValue SI_ShallowCloneValue(const SIValue v);

// SI_ConstValue creates an SIValue that shares the original's allocations, but does not need to persist them.
SIValue SI_ConstValue(const SIValue *v);

// SI_TransferOwnership duplicates 'v'.
// If 'v' owned its underlying value allocation,
// owership is transfered to the duplicate and 'v' allocation is set to M_VOLATILE.
SIValue SI_TransferOwnership(SIValue *v);

// SIValue_MakeVolatile updates an SIValue to mark that its allocations are shared rather than self-owned.
void SIValue_MakeVolatile(SIValue *v);

// SIValue_Persist updates an SIValue to duplicate any allocations that may go out of scope in the lifetime of this query.
void SIValue_Persist(SIValue *v);

// SIValue_SetAllocationType changes the SIValue's allocation to the explicitly provided value.
void SIValue_SetAllocationType(SIValue *v, SIAllocation allocation);

bool SIValue_IsNull(SIValue v);
bool SIValue_IsNullPtr(SIValue *v);
bool SIValue_IsFalse(SIValue v);
bool SIValue_IsTrue(SIValue v);

const char *SIType_ToString(SIType t);

// Prints all individual types represented by 't', multiple types are separated by comma,
// to a given buffer with length (bufferLen)
void SIType_ToMultipleTypeString(SIType t, char *buf, size_t bufferLen);

// Prints an SIValue to a given buffer, with length (bufferLen), sets bytesWritten to the actual length
// of string representation
// if there is not enough space for the value to be printed, the buffer will be re allocated with
// more space, and bufferLen will change accordingly
void SIValue_ToString(SIValue v, char **buf, size_t *bufferLen, size_t *bytesWritten);

/* Try to read a value as a double.
 * TODO Only used by agg_funcs, consider refactoring. */
int SIValue_ToDouble(const SIValue *v, double *d);

/* Try to parse a value by string. */
SIValue SIValue_FromString(const char *s);

/* Determines number of bytes required to join strings, with delimiter */
size_t SIValue_StringJoinLen(SIValue *strings, unsigned int string_count, const char *delimiter);

/* Concats strings as a delimiter. */
void SIValue_StringJoin(SIValue *strings, unsigned int string_count, const char *delimiter,
						char **buf, size_t *buf_len, size_t *bytesWritten);

/* Arithmetic operators for numeric SIValues.
 * The caller is responsible for ensuring that the arguments
 * are numeric types.
 * If both arguments are integer types, the result is as well,
 * otherwise a double is returned. */
SIValue SIValue_Add(const SIValue a, const SIValue b);
SIValue SIValue_Subtract(const SIValue a, const SIValue b);
SIValue SIValue_Multiply(const SIValue a, const SIValue b);
/* SIValue_Divide always returns a double value. */
SIValue SIValue_Divide(const SIValue a, const SIValue b);
/* SIValue_Modulo always gets integer values as input and return integer value. */
SIValue SIValue_Modulo(const SIValue a, const SIValue b);

// compares two SIValues and returns a value similar to strcmp
// if one of the values is null, the macro COMPARED_NULL is returned in disjointOrNull value.
// if the the values are not of the same type, the macro DISJOINT is returned in disjointOrNull value. */
int SIValue_Compare(const SIValue a, const SIValue b, int *disjointOrNull);

/* Update the provided hash state with the given SIValue. */
void SIValue_HashUpdate(SIValue v, XXH64_state_t *state);

/* Returns a hash code for a given SIValue. */
XXH64_hash_t SIValue_HashCode(SIValue v);

// reads SIValue off of binary stream
SIValue SIValue_FromBinary
(
	FILE *stream  // stream to read value from
);

// compute SIValue memory usage
size_t SIValue_memoryUsage
(
	SIValue v  // value
);

/* Free an SIValue's internal property if that property is a heap allocation owned
 * by this object. */
void SIValue_Free(SIValue v);

