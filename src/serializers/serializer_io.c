/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "serializer_io.h"
#include "../util/rmalloc.h"

#include <stdio.h>
#include <stdint.h>
#include <unistd.h>

// generic serializer
// contains a number of function pointers for data serialization
struct SerializerIO_Opaque {
	void (*WriteFloat)(void*, float);                 // write float
	void (*WriteDouble)(void*, double);               // write dobule
	void (*WriteSigned)(void*, int64_t);              // write signed int
	void (*WriteUnsigned)(void*, uint64_t);           // write unsigned int
	void (*WriteLongDouble)(void*, long double);      // write long double
	void (*WriteString)(void*, RedisModuleString*);   // write RedisModuleString
	void (*WriteBuffer)(void*, const char*, size_t);  // write bytes

	float (*ReadFloat)(void*);                        // read float
	double (*ReadDouble)(void*);                      // read dobule
	int64_t (*ReadSigned)(void*);                     // read signed int
	uint64_t (*ReadUnsigned)(void*);                  // read unsigned int
	char* (*ReadBuffer)(void*, size_t*);              // read bytes
	long double	(*ReadLongDouble)(void*);             // read long double
	RedisModuleString* (*ReadString)(void*);          // read RedisModuleString

	void *stream;     // RedisModuleIO* or a Stream descriptor
	FILE *memstream;  // memory stream accumulating all read / write data
};

//------------------------------------------------------------------------------
// Serializer Write API
//------------------------------------------------------------------------------

// macro for creating the stream serializer write functions
#define SERIALIZERIO_WRITE(suffix, t)                          \
void SerializerIO_Write##suffix(SerializerIO io, t value) {    \
	ASSERT(io != NULL);                                        \
	io->Write##suffix(io->stream, value);                      \
	/* save read buffer to memory stream */                    \
	if(io->memstream != NULL) {                                \
		fwrite(&value, 1, sizeof(t), io->memstream);           \
	}                                                          \
}

SERIALIZERIO_WRITE(Unsigned, uint64_t)
SERIALIZERIO_WRITE(Signed, int64_t)
SERIALIZERIO_WRITE(String, RedisModuleString*)
SERIALIZERIO_WRITE(Double, double)
SERIALIZERIO_WRITE(Float, float)
SERIALIZERIO_WRITE(LongDouble, long double)

// write buffer to stream
void SerializerIO_WriteBuffer
(
	SerializerIO io,   // stream to write to
	const char *buff,  // buffer 
	size_t len         // number of bytes to write
) {
	ASSERT(io != NULL);
	io->WriteBuffer(io->stream, buff, len);

	// save read buffer to memory stream
	if(io->memstream != NULL) {
		fwrite(buff, sizeof(char), len, io->memstream);
	}
}

// macro for creating stream serializer write functions
#define STREAM_WRITE(suffix, t)                         \
static void Stream_Write##suffix(void *stream, t v) {   \
	FILE *f = (FILE*)stream;                            \
	size_t n = fwrite(&v, sizeof(t), 1 , f);            \
	ASSERT(n == 1);                                     \
}

// create stream write functions
STREAM_WRITE(Unsigned, uint64_t)          // Stream_WriteUnsigned
STREAM_WRITE(Signed, int64_t)             // Stream_WriteSigned
STREAM_WRITE(Double, double)              // Stream_WriteDouble
STREAM_WRITE(Float, float)                // Stream_WriteFloat
STREAM_WRITE(LongDouble, long double)     // Stream_WriteLongDouble
STREAM_WRITE(String, RedisModuleString*)  // Stream_WriteString

// write buffer to stream
static void Stream_WriteBuffer
(
	void *stream,      // stream to write to
	const char *buff,  // buffer
	size_t n           // number of bytes to write
) {
	FILE *f = (FILE*)stream;

	// write size
	size_t written = fwrite(&n, sizeof(size_t), 1, f);
	ASSERT(written == 1);

	// write data
	written = fwrite(buff, n, 1, f);
	ASSERT(written == 1);
}

// macro for creating stream serializer read functions
#define STREAM_READ(suffix, t)                      \
	static t Stream_Read##suffix(void *stream) {    \
		ASSERT(stream != NULL);                     \
		FILE *f = (FILE*)stream;                    \
		t v;                                        \
		size_t n = fread(&v, sizeof(t), 1, f);      \
		ASSERT(n == 1);                             \
		return v;                                   \
	}

// create stream read functions
STREAM_READ(Unsigned, uint64_t)          // Stream_ReadUnsigned
STREAM_READ(Signed, int64_t)             // Stream_ReadSigned
STREAM_READ(Double, double)              // Stream_ReadDouble
STREAM_READ(Float, float)                // Stream_ReadFloat
STREAM_READ(LongDouble, long double)     // Stream_ReadLongDouble
STREAM_READ(String, RedisModuleString*)  // Stream_ReadString

// read buffer from stream
static char *Stream_ReadBuffer
(
	void *stream,  // stream to read from
	size_t *n      // number of bytes read
) {
	ASSERT(stream != NULL);

	FILE *f = (FILE*)stream;

	// read buffer's size
	size_t len;
	size_t read = fread(&len, sizeof(size_t), 1, f);
	ASSERT(read == 1);

	char *data = rm_malloc(sizeof(char) * len);

	// read data
	read = fread(data, len, 1, f);
	ASSERT(read == 1);

	*n = len;

	return data;
}

//------------------------------------------------------------------------------
// Serializer Read API
//------------------------------------------------------------------------------

// macro for creating the serializer read functions
#define SERIALIZERIO_READ(suffix, t)              \
t SerializerIO_Read##suffix(SerializerIO io) {    \
	ASSERT(io != NULL);                           \
	t v = io->Read##suffix(io->stream);           \
	/* save read buffer to memory stream */       \
	if(io->memstream != NULL) {                   \
		fwrite(&v, 1, sizeof(t), io->memstream);  \
	}                                             \
	return v;                                     \
}

SERIALIZERIO_READ(Unsigned, uint64_t)
SERIALIZERIO_READ(Signed, int64_t)
SERIALIZERIO_READ(String, RedisModuleString*)
SERIALIZERIO_READ(Double, double)
SERIALIZERIO_READ(Float, float)
SERIALIZERIO_READ(LongDouble, long double)

// read buffer from stream
char *SerializerIO_ReadBuffer
(
	SerializerIO io,  // stream
	size_t *lenptr    // number of bytes to read
) {
	size_t l;
	char *v = io->ReadBuffer(io->stream, &l);

	// save read buffer to memory stream
	if(io->memstream != NULL) {
		fwrite(v, sizeof(char), l, io->memstream);
	}

	if(lenptr != NULL) {
		*lenptr = l;
	}

	return v;
}

//------------------------------------------------------------------------------
// Serializer Create API
//------------------------------------------------------------------------------

// create a serializer which uses stream
SerializerIO SerializerIO_FromStream
(
	FILE *f  // stream
) {
	ASSERT(f != NULL);

	SerializerIO serializer = rm_calloc(1, sizeof(struct SerializerIO_Opaque));

	serializer->stream = (void*)f;

	// set serializer function pointers
	serializer->WriteUnsigned   = Stream_WriteUnsigned;
	serializer->WriteSigned     = Stream_WriteSigned;
	serializer->WriteString     = Stream_WriteString;
	serializer->WriteBuffer     = Stream_WriteBuffer;
	serializer->WriteDouble     = Stream_WriteDouble;
	serializer->WriteFloat      = Stream_WriteFloat;
	serializer->WriteLongDouble = Stream_WriteLongDouble;

	serializer->ReadFloat       = Stream_ReadFloat;
	serializer->ReadDouble      = Stream_ReadDouble;
	serializer->ReadSigned      = Stream_ReadSigned;
	serializer->ReadBuffer      = Stream_ReadBuffer;
	serializer->ReadString      = Stream_ReadString;
	serializer->ReadUnsigned    = Stream_ReadUnsigned;
	serializer->ReadLongDouble  = Stream_ReadLongDouble;
	
	return serializer;
}

// create a serializer which uses RedisIO
SerializerIO SerializerIO_FromRedisModuleIO
(
	RedisModuleIO *io
) {
	ASSERT(io != NULL);

	SerializerIO serializer = rm_calloc(1, sizeof(struct SerializerIO_Opaque));

	serializer->stream = io;

	// set serializer function pointers
	serializer->WriteUnsigned   = (void (*)(void*, uint64_t))RedisModule_SaveUnsigned;
	serializer->WriteSigned     = (void (*)(void*, int64_t))RedisModule_SaveSigned;
	serializer->WriteString     = (void (*)(void*, RedisModuleString*))RedisModule_SaveString;
	serializer->WriteBuffer     = (void (*)(void*, const char*, size_t))RedisModule_SaveStringBuffer;
	serializer->WriteDouble     = (void (*)(void*, double))RedisModule_SaveDouble;
	serializer->WriteFloat      = (void (*)(void*, float))RedisModule_SaveFloat;
	serializer->WriteLongDouble = (void (*)(void*, long double))RedisModule_SaveLongDouble;

	serializer->ReadFloat       = (float (*)(void*))RedisModule_LoadFloat;
	serializer->ReadDouble      = (double (*)(void*))RedisModule_LoadDouble;
	serializer->ReadSigned      = (int64_t (*)(void*))RedisModule_LoadSigned;
	serializer->ReadBuffer      = (char* (*)(void*, size_t*))RedisModule_LoadStringBuffer;
	serializer->ReadString      = (RedisModuleString* (*)(void*))RedisModule_LoadString;
	serializer->ReadUnsigned    = (uint64_t (*)(void*))RedisModule_LoadUnsigned;
	serializer->ReadLongDouble  = (long double (*)(void*))RedisModule_LoadLongDouble;	

	return serializer;
}

// saves all read / write data into buffer
void SerializerIO_SaveDataToBuffer
(
	SerializerIO io,  // serializer
	char **buffer,    // pointer to buffer, should be NULL
	size_t *size      // holds the buffer size once SerializerIO_Free is called
) {
	ASSERT(io            != NULL);
	ASSERT(buffer        != NULL);
	ASSERT(*buffer       == NULL);
	ASSERT(io->memstream == NULL);

	// create memory stream from buffer
	io->memstream = open_memstream(buffer, size);
}

// free serializer
void SerializerIO_Free
(
	SerializerIO *io  // serializer to free
) {
	ASSERT(io  != NULL);
	ASSERT(*io != NULL);

	SerializerIO _io = *io;

	// close memory stream
	if(_io->memstream != NULL) {
		fflush(_io->memstream);
		fclose(_io->memstream);
	}

	rm_free(*io);
	*io = NULL;
}

