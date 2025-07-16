/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
*/

#pragma once

#include "../value.h"

/**
  * @brief  Initialize a new SIValue array type with given capacity
  * @param  initialCapacity:
  * @retval Initialized array
  */
SIValue SIArray_New(u_int32_t initialCapacity);

/**
  * @brief  Appends a new SIValue to a given array
  * @param  siarray: pointer to array
  * @param  value: new value
  */
void SIArray_Append(SIValue *siarray, SIValue value);

/**
  * @brief  Returns a volatile copy of the SIValue from an array in a given index
  * @note   If index is out of bound, SI_NullVal is returned
  *         Caller is expected either to not free the returned value or take ownership on
  *         its own copy
  * @param  siarray: array
  * @param  index: index
  * @retval The value in the requested index
  */
SIValue SIArray_Get(SIValue siarray, u_int32_t index);

/**
  * @brief  Returns the array length
  * @param  siarray:
  * @retval array length
  */
u_int32_t SIArray_Length(SIValue siarray);

/**
  * @brief  Returns true if any of the types in 't' are contained in the array
            or its nested array children, if any
  * @param  siarray: array
  * @param  t: bitmap of types to search for
  * @retval a boolean indicating whether any types were matched
  */
bool SIArray_ContainsType(SIValue siarray, SIType t);

/**
  * @brief  Returns true if the array contains an element equals to 'value'
  * @param  siarray: array
  * @param  value: value to search for
  * @param  comparedNull: indicate if there was a null comparison during the array scan
  * @retval a boolean indicating whether value was found in siarray
  */
bool SIArray_ContainsValue(SIValue siarray, SIValue value, bool *comparedNull);

/**
  * @brief  Returns true if all of the elements in the array are of type 't'
  * @param  siarray: array
  * @param  t: type to compare
  * @retval a boolean indicating whether all elements are of type 't'
  */
bool SIArray_AllOfType(SIValue siarray, SIType t);

/**
  * @brief  Sorts the array in place
  * @param  siarray: array to sort
  * @param  ascending: sort order
  */
void SIArray_Sort(SIValue siarray, bool ascending);

/**
  * @brief  Returns a copy of the array
  * @note   The caller needs to free the array
  * @param  siarray:
  * @retval A clone of the given array
  */
SIValue SIArray_Clone(SIValue siarray);

/**
  * @brief  Prints an array into a given buffer
  * @param  list: array to print
  * @param  buf: print buffer (pointer to pointer to allow re allocation)
  * @param  len: print buffer length
  * @param  bytesWritten: the actual number of bytes written to the buffer
  */
void SIArray_ToString(SIValue list, char **buf, size_t *bufferLen, size_t *bytesWritten);

/**
 * @brief  Returns the array hash code.
 * @param  siarray: SIArray.
 * @retval The array hashCode.
 */
XXH64_hash_t SIArray_HashCode(SIValue siarray);

// creates an array from its binary representation
// this is the reverse of SIArray_ToBinary
// x = SIArray_FromBinary(SIArray_ToBinary(y));
// x == y
SIValue SIArray_FromBinary
(
	FILE *stream  // stream containing binary representation of an array
);

/**
  * @brief  delete an array
  * @param  siarray:
  * @retval None
  */
void SIArray_Free(SIValue siarray);

