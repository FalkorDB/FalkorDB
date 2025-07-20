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

// write buffer to stream
void SerializerIO_WriteBuffer
(
	SerializerIO io,   // stream to write to
	const char *buff,  // buffer 
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
bool SerializerIO_ReadUnsigned
(
	SerializerIO io,  // stream
	uint64_t *x       // [output]
);

// read signed from stream
bool SerializerIO_ReadSigned
(
	SerializerIO io,  // stream
	int64_t *x        // [output]
);

// read buffer from stream
bool SerializerIO_ReadBuffer
(
	SerializerIO io,  // stream
	char **x,         // [output]
	size_t *lenptr    // [output] number of bytes to read
);

// read double from stream
bool SerializerIO_ReadDouble
(
	SerializerIO io,  // stream
	double *x         // [output]
);

// read float from stream
bool SerializerIO_ReadFloat
(
	SerializerIO io,  // stream
	float *x          // [output]
);

// read long double from stream
bool SerializerIO_ReadLongDouble
(
	SerializerIO io,  // stream
	long double *x    // [output]
);

#define SerializerIO_Read(io, ptr)                                 \
    _Generic(*(ptr),                                               \
        uint64_t           : SerializerIO_ReadUnsigned,            \
        int64_t            : SerializerIO_ReadSigned,              \
        double             : SerializerIO_ReadDouble,              \
        float              : SerializerIO_ReadFloat,               \
        long double        : SerializerIO_ReadLongDouble           \
    )                                                              \
	(io, ptr)

#define SerializerIO_Write(io, value,...)                          \
	_Generic                                                       \
	(                                                              \
		(value),                                                   \
			uint64_t           : SerializerIO_WriteUnsigned     ,  \
			int64_t            : SerializerIO_WriteSigned       ,  \
			double             : SerializerIO_WriteDouble       ,  \
			float              : SerializerIO_WriteFloat        ,  \
			long double        : SerializerIO_WriteLongDouble      \
	)                                                              \
	(io, value)

void SerializerIO_Free
(
	SerializerIO *io  // serializer to free
);

