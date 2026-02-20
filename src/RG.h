/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <stdbool.h>
#include "redismodule.h"

//------------------------------------------------------------------------------
// common bitwise operations
//------------------------------------------------------------------------------

// Mask with most significant bit on 10000...
#define MSB_MASK (1UL << (sizeof(uint64_t) * 8 - 1))

// Mask complement 01111...
#define MSB_MASK_CMP ~MSB_MASK

// Set X's most significant bit on.
#define SET_MSB(x) ((x) | MSB_MASK)

// Clear X's most significant bit.
#define CLEAR_MSB(x) ((x) & MSB_MASK_CMP)

//------------------------------------------------------------------------------
// alignment
//------------------------------------------------------------------------------

// helper macro to align a value to the next 8-byte boundary
#define ALIGN8(n) (((n) + 7) & ~7)

//------------------------------------------------------------------------------
// code development settings
//------------------------------------------------------------------------------

// to turn on Debug for a single file add:
// #define RG_DEBUG
// just before the statement:
// #include "RG.h"

// to turn on Debug globaly, uncomment this line:
// #define RG_DEBUG

//------------------------------------------------------------------------------
// debugging definitions
//------------------------------------------------------------------------------

#undef ASSERT

#ifdef RG_DEBUG

	// assert X is true
	#define ASSERT(X)                                               \
	_Pragma("GCC diagnostic push")                                  \
	_Pragma("GCC diagnostic ignored \"-Wnull-dereference\"")        \
	{                                                               \
		if (!(X))                                                   \
		{                                                           \
			if(RedisModule__Assert != NULL) {                       \
				RedisModule_Assert(X);                              \
			} else {                                                \
				printf ("assert(%s) failed: %s line %d\n",          \
				#X, __FILE__, __LINE__);                            \
				/* force crash */                                   \
				char x = *((char*)NULL); /* produce stack trace */  \
				assert(x); /* solves C++ unused var warning */      \
			}                                                       \
		}                                                           \
	}                                                               \
	_Pragma("GCC diagnostic pop")

#else

	// debugging disabled
	#define ASSERT(X)

#endif


// The unused macro should be applied to avoid compiler warnings
// on set but unused variables

#undef UNUSED
#define UNUSED(V) ((void)V)

// GraphBLAS return code validation
// both GrB_SUCCESS and GrB_NO_VALUE are valid "OK"
// return codes
#if RG_DEBUG
	#define GrB_OK(GrB_method)                                       \
	{                                                                \
		GrB_Info _info = (GrB_method) ;                              \
		ASSERT(_info == GrB_SUCCESS || _info == GrB_NO_VALUE);       \
	}
#else
	#define GrB_OK(GrB_method) (GrB_method)
#endif

// use likely and unlikely to provide the compiler with branch prediction information
// for example:
// if (likely(x > 0))
//         foo ();
#if defined(__GNUC__)
#define likely(x)       __builtin_expect((x),1)
#define unlikely(x)     __builtin_expect((x),0)
#else
#define likely(x)       (x)
#define unlikely(x)     (x)
#endif

#define RedisModule_ReplyWithErrorFormat(ctx, fmt, ...) do { \
	char *str;                                               \
	int res;                                                 \
	UNUSED(res);                                             \
	res = asprintf(&str, fmt, ##__VA_ARGS__);                \
	RedisModule_ReplyWithError(ctx, str);                    \
	free(str);                                               \
} while(0)


// write string to stream
#define fwrite_string(str, stream)                       \
{                                                        \
	size_t l = strlen(str) + 1;                          \
	fwrite_assert(&l, sizeof(size_t), stream);           \
	fwrite_assert(str, l, stream);                       \
}

#define fwrite_assert(input, size, stream)               \
{                                                        \
	int write = fwrite(input, size, 1, stream);          \
	ASSERT(write == 1);                                  \
}

#define fread_assert(output, size, stream)               \
{                                                        \
	int read = fread((void*)(output), size, 1, stream);  \
	/* short read! */                                    \
	ASSERT("short read" && read == 1);                   \
}

