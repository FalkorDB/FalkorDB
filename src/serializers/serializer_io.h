/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "../redismodule.h"

#include <setjmp.h>
#include <stdbool.h>

// SerializerIO
// acts as an abstraction layer for both graph encoding and decoding
// there are two types of serializers:
// 1. RedisIO serializer
// 2. Stream serializer
//
// The graph encoding / decoding logic uses this abstraction without knowing
// what is the underline serializer

typedef struct SerializerIO_Opaque *SerializerIO;

// create a serializer which uses a stream
SerializerIO SerializerIO_FromStream
(
	FILE *stream,  // stream
	bool encoder   // true for encoder, false decoder
);

// create a serializer which uses RedisIO
SerializerIO SerializerIO_FromRedisModuleIO
(
	RedisModuleIO *io,  // redis module io
	bool encoder        // true for encoder, false decoder
);

// create a buffered serializer which uses RedisIO
SerializerIO SerializerIO_FromBufferedRedisModuleIO
(
	RedisModuleIO *io,  // redis module io
	bool encoder        // true for encoder, false decoder
);

// create a buffered serializer which uses RedisIO
SerializerIO SerializerIOv2_FromBufferedRedisModuleIO
(
	RedisModuleIO *io,  // redis module io
	bool encoder        // true for encoder, false decoder
);


//------------------------------------------------------------------------------
// Serializer Write API
//------------------------------------------------------------------------------

// write unsingned to stream
void SerializerIO_WriteUnsigned
(
	SerializerIO io,  // stream to write to
	uint64_t value    // value
);

// write signed to stream
void SerializerIO_WriteSigned
(
	SerializerIO io,  // stream to write to
	int64_t value     // value
);

// write string to stream
void SerializerIO_WriteString
(
	SerializerIO io,      // stream to write to
	RedisModuleString *s  // string
);

// write buffer to stream
void SerializerIO_WriteBuffer
(
	SerializerIO io,   // stream to write to
	const void *buff,  // buffer
	size_t len         // number of bytes to write
);

// write double to stream
void SerializerIO_WriteDouble
(
	SerializerIO io,  // stream
	double value      // value
);

// write float to stream
void SerializerIO_WriteFloat
(
	SerializerIO io,  // stream
	float value       // value
);

// write long double to stream
void SerializerIO_WriteLongDouble
(
	SerializerIO io,   // stream
	long double value  // value
);

// type-generic write: dispatches to the matching SerializerIO_Write* above
#define SerializerIO_Write(io, value,...)                          \
	_Generic                                                       \
	(                                                              \
		(value),                                                   \
			uint64_t           : SerializerIO_WriteUnsigned     ,  \
			int64_t            : SerializerIO_WriteSigned       ,  \
			RedisModuleString* : SerializerIO_WriteString       ,  \
			double             : SerializerIO_WriteDouble       ,  \
			float              : SerializerIO_WriteFloat        ,  \
			long double        : SerializerIO_WriteLongDouble      \
	)                                                              \
	(io, value)

//------------------------------------------------------------------------------
// Serializer Try-Read API
//------------------------------------------------------------------------------
//
// the sole read API: the RDB decoder dispatcher arms a landing pad
// (SerializerIO_Try) and every decoder reads through these variants - each
// performs the read and, on a short read, longjmps to the landing pad instead
// of returning. an unarmed reader (e.g. graph offload) just latches the error
// flag and returns

// try to read unsigned from stream, aborting the decode on a short read
uint64_t SerializerIO_TryReadUnsigned
(
	SerializerIO io  // stream
);

// try to read signed from stream, aborting the decode on a short read
int64_t SerializerIO_TryReadSigned
(
	SerializerIO io  // stream
);

// try to read string from stream, aborting the decode on a short read
RedisModuleString *SerializerIO_TryReadString
(
	SerializerIO io  // stream
);

// try to read buffer from stream, aborting the decode on a short read
void *SerializerIO_TryReadBuffer
(
	SerializerIO io,  // stream
	size_t *lenptr    // number of bytes to read
);

// try to read double from stream, aborting the decode on a short read
double SerializerIO_TryReadDouble
(
	SerializerIO io  // stream
);

// try to read float from stream, aborting the decode on a short read
float SerializerIO_TryReadFloat
(
	SerializerIO io  // stream
);

// try to read long double from stream, aborting the decode on a short read
long double SerializerIO_TryReadLongDouble
(
	SerializerIO io  // stream
);

// try to read a NUL-terminated string; a short read aborts the decode, a buffer
// that doesn't end in NUL fails it. free with rm_free
char *SerializerIO_TryReadCString
(
	SerializerIO io  // stream
);

//------------------------------------------------------------------------------
// Error API
//------------------------------------------------------------------------------

// returns true if a short read / IO error or a malformed-data failure was
// latched during decoding
bool SerializerIO_Error
(
	SerializerIO io  // serializer
);

// fail the decode: the payload violated an encoding invariant (e.g. a count
// mismatch or a malformed matrix). latches the error flag and longjmps to the
// armed landing pad. the first reason is kept (printf-style)
// see SerializerIO_ErrorReason
void SerializerIO_SetError
(
	SerializerIO io,    // serializer
	const char *fmt,    // what was wrong with the payload
	...
) ;

// why the decode failed: the reason given for bad data, or NULL when there is
// no failure or the failure is a short read (Redis reports those itself)
const char *SerializerIO_ErrorReason
(
	SerializerIO io  // serializer
);

//------------------------------------------------------------------------------
// non-local decode abort
//------------------------------------------------------------------------------
//
// rather than check for a short read after every read
// installs a single landing pad with SerializerIO_Try and reads through the
// TryRead API, which longjmps straight to that pad whenever a read can't be
// satisfied (short read) or a decoder detects malformed data
// (SerializerIO_SetError). the handler stays armed for the whole
// decode

// arm the abort handler; returns a jmp_buf to pass to setjmp - callers use the
// SerializerIO_Try wrapper rather than calling this directly
jmp_buf *SerializerIO_Arm
(
	SerializerIO io  // serializer
);

// install an abort handler for 'io' in the current stack frame; evaluates to 0
// on the initial call and non-zero when a read later aborts the decode. must be
// used in a frame that stays live for the whole decode, e.g.
//     if(SerializerIO_Try(io) != 0) { /* landing pad: tear down, return */ }
#define SerializerIO_Try(io) setjmp(*SerializerIO_Arm(io))

// tear down the abort handler; a subsequent TryRead / SetError no longer jumps
void SerializerIO_Disarm
(
	SerializerIO io  // serializer
);

// stash / fetch the in-flight object (the GraphContext being decoded) so the
// landing pad can tear it down after a jump that unwound the decoder's own
// locals
void SerializerIO_SetContext
(
	SerializerIO io,  // serializer
	void *context     // object to recover at the landing pad
);

void *SerializerIO_GetContext
(
	SerializerIO io  // serializer
);

void SerializerIO_Free
(
	SerializerIO *io  // serializer to free
);

