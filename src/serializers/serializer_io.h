/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "../redismodule.h"

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

//------------------------------------------------------------------------------
// Serializer Read API
//------------------------------------------------------------------------------

// read unsigned from stream
uint64_t SerializerIO_ReadUnsigned
(
	SerializerIO io  // stream
);

// read signed from stream
int64_t SerializerIO_ReadSigned
(
	SerializerIO io  // stream
);

// read string from stream
RedisModuleString *SerializerIO_ReadString
(
	SerializerIO io  // stream
);

// read buffer from stream
void *SerializerIO_ReadBuffer
(
	SerializerIO io,  // stream
	size_t *lenptr     // number of bytes to read
);

// read double from stream
double SerializerIO_ReadDouble
(
	SerializerIO io  // stream
);

// read float from stream
float SerializerIO_ReadFloat
(
	SerializerIO io  // stream
);

// read long double from stream
long double SerializerIO_ReadLongDouble
(
	SerializerIO io  // stream
);

// read a NUL-terminated string
// a buffer that doesn't end in NUL fails the decode, an empty string is
// returned in its place; free with rm_free
char *SerializerIO_ReadCString
(
	SerializerIO io  // stream
);

// returns true if a short read / IO error was encountered during decoding
// once set the flag is sticky: further reads short-circuit and return zeroed
// values, so decoders can keep going until the next checkpoint aborts the load
bool SerializerIO_Error
(
	SerializerIO io  // serializer
);

// fail the decode: the payload violated an encoding invariant (e.g. a count
// mismatch or a malformed matrix). latches the same sticky flag as a short
// read, so the load aborts at the next checkpoint and the decoder returns NULL
// the first reason is kept (printf-style), see SerializerIO_ErrorReason
void SerializerIO_SetError
(
	SerializerIO io,    // serializer
	const char *fmt,    // what was wrong with the payload
	...
) __attribute__ ((format (printf, 2, 3)));

// why the decode failed: the reason given for bad data, or NULL when there is
// no failure or the failure is a short read (Redis reports those itself)
const char *SerializerIO_ErrorReason
(
	SerializerIO io  // serializer
);

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

void SerializerIO_Free
(
	SerializerIO *io  // serializer to free
);

