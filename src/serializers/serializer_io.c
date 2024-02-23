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

	void *stream;  // RedisModuleIO* or a Pipe
};

//------------------------------------------------------------------------------
// Serializer Write API
//------------------------------------------------------------------------------

// macro for creating the pipe serializer write functions
#define SERIALIZERIO_WRITE(suffix, t)                          \
void SerializerIO_Write##suffix(SerializerIO io, t value) {    \
	ASSERT(io != NULL);                                        \
	io->Write##suffix(io->stream, value);                      \
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
}

// macro for creating pipe serializer write functions
#define PIPE_WRITE(suffix, t)                       \
static void Pipe_Write##suffix(void *pipe, t v) {   \
	int pipefd = (intptr_t)pipe;                    \
	ssize_t n = write(pipefd, &v, sizeof(t));       \
	ASSERT(n == sizeof(t));                         \
}

// create pipe write functions
PIPE_WRITE(Unsigned, uint64_t)          // Pipe_WriteUnsigned
PIPE_WRITE(Signed, int64_t)             // Pipe_WriteSigned
PIPE_WRITE(Double, double)              // Pipe_WriteDouble
PIPE_WRITE(Float, float)                // Pipe_WriteFloat
PIPE_WRITE(LongDouble, long double)     // Pipe_WriteLongDouble
PIPE_WRITE(String, RedisModuleString*)  // Pipe_WriteString

static void Pipe_WriteBuffer
(
	void *pipe,
	const char *buff,
	size_t n
) {
	int pipefd = (intptr_t)pipe;

	// write size
	write(pipefd, &n , sizeof(size_t));

	// write data
	write(pipefd, buff, n);
}

// macro for creating pipe serializer read functions
#define PIPE_READ(suffix, t)                        \
	static t Pipe_Read##suffix(void *pipe) {        \
		ASSERT(pipe != NULL);                       \
		int pipefd = (intptr_t)pipe;                \
		t v;                                        \
		ssize_t n = read(pipefd, &v, sizeof(t));    \
		ASSERT(n == sizeof(t));                     \
		return v;                                   \
	}

// create pipe read functions
PIPE_READ(Unsigned, uint64_t)          // Pipe_ReadUnsigned
PIPE_READ(Signed, int64_t)             // Pipe_ReadSigned
PIPE_READ(Double, double)              // Pipe_ReadDouble
PIPE_READ(Float, float)                // Pipe_ReadFloat
PIPE_READ(LongDouble, long double)     // Pipe_ReadLongDouble
PIPE_READ(String, RedisModuleString*)  // Pipe_ReadString

static char *Pipe_ReadBuffer
(
	void *pipe,
	size_t *n
) {
	ASSERT(pipe != NULL);

	int pipefd = (intptr_t)pipe;

	// read buffer's size
	size_t len;
	read(pipefd, &len, sizeof(size_t));

	char *data = rm_malloc(sizeof(char) * len);

	// read data
	read(pipefd, data, len);

	if(n != NULL) *n = len;

	return data;
}

//------------------------------------------------------------------------------
// Serializer Read API
//------------------------------------------------------------------------------

// macro for creating the pipe serializer write functions
#define SERIALIZERIO_READ(suffix, t)              \
t SerializerIO_Read##suffix(SerializerIO io) {    \
	ASSERT(io != NULL);                           \
	return io->Read##suffix(io->stream);          \
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
	return io->ReadBuffer(io->stream, lenptr);
}

//------------------------------------------------------------------------------
// Serializer Create API
//------------------------------------------------------------------------------

// create a serializer which uses pipe
SerializerIO SerializerIO_FromPipe
(
	int pipefd  // either the read or write end of a pipe
) {
	SerializerIO serializer = rm_calloc(1, sizeof(struct SerializerIO_Opaque));

	serializer->stream = (void*)(intptr_t)pipefd;

	// set serializer function pointers
	serializer->WriteUnsigned   = Pipe_WriteUnsigned;
	serializer->WriteSigned     = Pipe_WriteSigned;
	serializer->WriteString     = Pipe_WriteString;
	serializer->WriteBuffer     = Pipe_WriteBuffer;
	serializer->WriteDouble     = Pipe_WriteDouble;
	serializer->WriteFloat      = Pipe_WriteFloat;
	serializer->WriteLongDouble = Pipe_WriteLongDouble;

	serializer->ReadFloat       = Pipe_ReadFloat;
	serializer->ReadDouble      = Pipe_ReadDouble;
	serializer->ReadSigned      = Pipe_ReadSigned;
	serializer->ReadBuffer      = Pipe_ReadBuffer;
	serializer->ReadString      = Pipe_ReadString;
	serializer->ReadUnsigned    = Pipe_ReadUnsigned;
	serializer->ReadLongDouble  = Pipe_ReadLongDouble;
	
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

// free serializer
void SerializerIO_Free
(
	SerializerIO *io  // serializer to free
) {
	ASSERT(io  != NULL);
	ASSERT(*io != NULL);

	// TODO: close pipe
	rm_free(*io);
	*io = NULL;
}

