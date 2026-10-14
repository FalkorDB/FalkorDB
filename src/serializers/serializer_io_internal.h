/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

// internal layout of the serializer, shared between the current serializer
// (serializer_io.c) and the previous-version buffered serializer
// (serializer_io_prev.c)
//
// a serializer created by one translation unit is read and freed by functions
// in the other, so both must agree on the exact struct layout - keeping the
// definitions here guarantees they never drift apart

#include "serializer_io.h"
#include "../redismodule.h"

#include <setjmp.h>
#include <stdbool.h>
#include <stdint.h>

// buffered io descriptor - the backing store of a buffered serializer
typedef struct {
	unsigned char *buffer;  // io buffer
	size_t cap;             // io buffer capacity
	size_t count;           // number of bytes written to io buffer
	RedisModuleIO *stream;  // redis module io
	SerializerIO io;        // owning serializer; malformed framing is reported
	                        // through it via the single SerializerIO_SetError API
} BufferedIO;

// generic serializer
// contains a number of function pointers for data serialization
struct SerializerIO_Opaque {
	void (*WriteFloat)(void*, float);                 // write float
	void (*WriteDouble)(void*, double);               // write dobule
	void (*WriteSigned)(void*, int64_t);              // write signed int
	void (*WriteUnsigned)(void*, uint64_t);           // write unsigned int
	void (*WriteLongDouble)(void*, long double);      // write long double
	void (*WriteString)(void*, RedisModuleString*);   // write RedisModuleString
	void (*WriteBuffer)(void*, const void*, size_t);  // write bytes

	float (*ReadFloat)(void*);                        // read float
	double (*ReadDouble)(void*);                      // read dobule
	int64_t (*ReadSigned)(void*);                     // read signed int
	uint64_t (*ReadUnsigned)(void*);                  // read unsigned int
	void* (*ReadBuffer)(void*, size_t*);              // read bytes
	long double	(*ReadLongDouble)(void*);             // read long double
	RedisModuleString* (*ReadString)(void*);          // read RedisModuleString

	bool encoder;           // true is serializer is used for encoding, false decoding
	void *stream;           // RedisModuleIO* or a Stream descriptor
	bool free_buff;         // true if serializer has a buffer to free
	bool error;             // decode failure flag (short read or bad data)
	char error_reason[256];  // decoder-detected bad data, empty otherwise
	bool (*IsError)(void*); // backend error probe, NULL for encoders

	// non-local abort support (serializer_io.h). the decoder dispatcher arms one
	// handler for the whole decode; while 'armed', a TryRead that short-reads or
	// a SerializerIO_SetError on malformed data longjmps to 'handler' instead of
	// returning. an unarmed reader (e.g. graph offload) latches 'error' and
	// returns instead
	bool armed;             // true while a handler is installed
	jmp_buf handler;        // where to jump on a decode failure
	void *context;          // in-flight object the landing pad tears down (gc)
};

// probe whether a buffered serializer hit a backend IO error (e.g. a short read
// during diskless replication or a truncated RESTORE payload). malformed buffer
// framing is reported separately, through SerializerIO_SetError on buffer->io
static inline bool Buffered_IsError
(
	void *stream  // BufferedIO*
) {
	BufferedIO *buffer = (BufferedIO*)stream;
	return RedisModule_IsIOError(buffer->stream);
}
