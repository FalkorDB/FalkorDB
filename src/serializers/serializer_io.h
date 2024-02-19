/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

// write unsingned to stream
void SerializerIO_SaveUnsigned
(
	SerializerIO *io,  // stream to write to
	uint64_t value     // value
);

// read unsigned from stream
uint64_t SerializerIO_LoadUnsigned
(
	SerializerIO *io  // stream
);

// write signed to stream
void SerializerIO_SaveSigned
(
	SerializerIO *io,  // stream to write to
	int64_t value      // value
);

// read signed from stream
int64_t SerializerIO_LoadSigned
(
	SerializerIO *io  // stream
);

// write string to stream
void SerializerIO_SaveString
(
	SerializerIO *io,     // stream to write to
	RedisModuleString *s  // string
);

// write buffer to stream
void SerializerIO_SaveBuffer
(
	SerializerIO *io,  // stream to write to
	const char *buff,  // buffer 
	size_t len         // number of bytes to write
);

// read string from stream
RedisModuleString *SerializerIO_LoadString
(
	SerializerIO *io  // stream
);

// read buffer from stream
char *SerializerIO_LoadStringBuffer
(
	SerializerIO *io,  // stream
	size_t *lenptr     // number of bytes to read
);

// write double to stream
void SerializerIO_SaveDouble
(
	SerializerIO *io,  // stream
	double value       // value
);

// read double from stream
double SerializerIO_LoadDouble
(
	SerializerIO *io  // stream
);

// write float from stream
void SerializerIO_SaveFloat
(
	SerializerIO *io,  // stream
	float value        // value
);

// read float from stream
float SerializerIO_LoadFloat
(
	SerializerIO *io  // stream
);

// write long double to stream
void SerializerIO_SaveLongDouble
(
	SerializerIO *io,  // stream
	long double value  // value
);

// read long double from stream
long double SerializerIO_LoadLongDouble
(
	SerializerIO *io  // stream
);

#define SerializerIO_Save(io, value,...)                          \
	_Generic                                                      \
	(                                                             \
		(value),                                                  \
			uint64_t          : SerializerIO_SaveUnsigned     ,   \
			int64_t           : SerializerIO_SaveSigned       ,   \
			RedisModuleString : SerializerIO_SaveString       ,   \
			const char*       : SerializerIO_SaveStringBuffer ,   \
			double            : SerializerIO_SaveDouble       ,   \
			float             : SerializerIO_SaveFloat        ,   \
			long double       : SerializerIO_SaveLongDouble       \
	)                                                             \
	(io, value, __VA_ARGS__)

