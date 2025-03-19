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

#define BUFFER_SIZE 256000  // buffered searializer buffer size 256KB

typedef struct {
	unsigned char *buffer;  // io buffer
	size_t cap;             // io buffer capacity
	size_t count;           // number of bytes written to io buffer
	RedisModuleIO *stream;  // redis module io
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
	void (*WriteBuffer)(void*, const char*, size_t);  // write bytes

	float (*ReadFloat)(void*);                        // read float
	double (*ReadDouble)(void*);                      // read dobule
	int64_t (*ReadSigned)(void*);                     // read signed int
	uint64_t (*ReadUnsigned)(void*);                  // read unsigned int
	char* (*ReadBuffer)(void*, size_t*);              // read bytes
	long double	(*ReadLongDouble)(void*);             // read long double
	RedisModuleString* (*ReadString)(void*);          // read RedisModuleString

	bool encoder;  // true is serializer is used for encoding, false decoding
	void *stream;  // RedisModuleIO* or a Stream descriptor
};

//------------------------------------------------------------------------------
// Serializer Write API
//------------------------------------------------------------------------------

// macro for creating the stream serializer write functions
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
	size_t *n      // [optional] number of bytes read
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

	if(n != NULL) *n = len;

	return data;
}

//------------------------------------------------------------------------------
// Serializer Read API
//------------------------------------------------------------------------------

// macro for creating the serializer read functions
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
// Buffered Serializer Write API
//------------------------------------------------------------------------------

// flush buffer to underline stream
static void _flush_buffer
(
	BufferedIO *buffer  // buffer
) {
	// empty buffer
	if(unlikely(buffer->count == 0)) {
		return;
	}

	//--------------------------------------------------------------------------
	// flush buffer to stream
	//--------------------------------------------------------------------------

	// write buffer
	RedisModule_SaveStringBuffer(buffer->stream, (const char*)buffer->buffer,
			buffer->count);

	// reset buffer
	buffer->count = 0;
}

// TODO: think about a better name, making it clear that this function might
// flush
// another options could be adding a boolean 'flush' that if true and needed
// this function will flush the buffer

// ensure buffer can accommodate n additional bytes
// if there's no room the buffer is flushed
static inline void _ensure_cap
(
	BufferedIO *buffer,  // buffer
	size_t n             // number of bytes to write
) {
	ASSERT(n > 0);

	// make sure there's enough room in buffer
	if( (buffer->cap - buffer->count) < n) {
		_flush_buffer(buffer);
	}
}

// TODO: in debug mode add value type e.g. INT64_T
//     similarly in debug mode when reading validate that the right type is read

// macro for creating buffered RDB serializer write functions
#define BUFFERED_SERIALIZER_WRITE(suffix, t)                    \
static void BufferSerializerIO_Write##suffix(void *io, t v) {   \
	BufferedIO *buffer = (BufferedIO*)io;                       \
																\
	/* make sure buffer has enough room */                      \
	_ensure_cap(buffer, sizeof(t));                             \
																\
	/* write value to buffer */                                 \
	*((t*)(buffer->buffer + buffer->count)) = v;                \
																\
	/* update buffer offset */                                  \
	buffer->count += sizeof(t);                                 \
}

// create buffer serializer write functions
BUFFERED_SERIALIZER_WRITE(Float,      float)        // BufferSerializerIO_WriteFloat
BUFFERED_SERIALIZER_WRITE(Double,     double)       // BufferSerializerIO_WriteDouble
BUFFERED_SERIALIZER_WRITE(Signed,     int64_t)      // BufferSerializerIO_WriteSigned
BUFFERED_SERIALIZER_WRITE(Unsigned,   uint64_t)     // BufferSerializerIO_WriteUnsigned
BUFFERED_SERIALIZER_WRITE(LongDouble, long double)  // BufferSerializerIO_WriteLongDouble

// write unsigned to buffer
//void BufferSerializerIO_WriteUnsigned
//(
//	void *io,       // serializer
//	uint64_t value  // value
//) {
//	BufferedIO *buffer = (BufferedIO*)io;
//
//	// make sure buffer has enough room
//	_ensure_cap(buffer, sizeof(uint64_t));
//
//	// write value to buffer
//	*((uint64_t*)(buffer->buffer + buffer->count)) = value;
//
//	// update buffer offset
//	buffer->count += sizeof(uint64_t);
//}

// write buffer to stream
void BufferSerializerIO_WriteBuffer
(
	void *io,           // serializer
	const char *value,  // value
	size_t len          // value size
) {
	ASSERT(io    != NULL);
	ASSERT(value != NULL);
	ASSERT(len   > 0);

	BufferedIO *buffer = (BufferedIO*)io;

	// make sure value has enough room
	_ensure_cap(buffer, len + sizeof(size_t));

	// TODO: _ensure_cap should return boolean letting us know if we can
	// or can't write value to in memory buffer, this should simplify the
	// following condition

	// if value can't fit within io's buffer write directly to stream
	// otherwise add it to buffer
	if((len + sizeof(size_t)) <= buffer->cap) {
		// add to buffer
		// write value length to stream
		*((size_t*)(buffer->buffer + buffer->count)) = len;
		buffer->count += sizeof(size_t);

		// write value to buffer
		memcpy(buffer->buffer + buffer->count, value, len);
		buffer->count += len;
	} else {
		// value is too big, flush to stream
		RedisModule_SaveStringBuffer(buffer->stream, value, len);
	}
}

//------------------------------------------------------------------------------
// Buffered Serializer Read API
//------------------------------------------------------------------------------

// load buffer from stream to memory
static void _load_buffer
(
	BufferedIO *buffer  // buffer
) {
	// make sure current buffer is depleted
	ASSERT(buffer->count == buffer->cap);

	// free old buffer
	if(buffer->buffer != NULL) {
		rm_free(buffer->buffer);
		buffer->buffer = NULL;
	}

	// read new buffer from stream
	buffer->buffer =
		(unsigned char*)RedisModule_LoadStringBuffer(buffer->stream,
				&buffer->cap);

	ASSERT(buffer->cap    > 0);
	ASSERT(buffer->buffer != NULL);

	// reset offset
	buffer->count = 0;
}

// read unsigned int from buffer
uint64_t BufferSerializerIO_ReadUnsigned
(
	void *io  // serializer
) {
	BufferedIO *buffer = (BufferedIO*)io;

	// load buffer if depleted
	if(unlikely(buffer->count == buffer->cap)) {
		_load_buffer(buffer);
	}

	// ensure there's at least sizeof(uint64_t) bytes in buffer
	ASSERT((buffer->cap - buffer->count) >= sizeof(uint64_t));

	// read value
	uint64_t v = *(uint64_t*)(buffer->buffer + buffer->count);

	// update offset
	buffer->count += sizeof(uint64_t);

	return v;
}

// read signed int from buffer
int64_t BufferSerializerIO_ReadSigned
(
	void *io  // serializer
) {
	BufferedIO *buffer = (BufferedIO*)io;

	// load buffer if depleted
	if(unlikely(buffer->count == buffer->cap)) {
		_load_buffer(buffer);
	}

	// ensure there's at least sizeof(int64_t) bytes in buffer
	ASSERT((buffer->cap - buffer->count) >= sizeof(int64_t));

	// read value
	int64_t v = *(int64_t*)(buffer->buffer + buffer->count);

	// update offset
	buffer->count += sizeof(int64_t);

	return v;
}

// read double from buffer
double BufferSerializerIO_ReadDouble
(
	void *io  // serializer
) {
	BufferedIO *buffer = (BufferedIO*)io;

	// load buffer if depleted
	if(unlikely(buffer->count == buffer->cap)) {
		_load_buffer(buffer);
	}

	// ensure there's at least sizeof(double) bytes in buffer
	ASSERT((buffer->cap - buffer->count) >= sizeof(double));

	// read value
	double v = *(double*)(buffer->buffer + buffer->count);

	// update offset
	buffer->count += sizeof(double);

	return v;
}

// read float from buffer
float BufferSerializerIO_ReadFloat
(
	void *io  // serializer
) {
	BufferedIO *buffer = (BufferedIO*)io;

	// load buffer if depleted
	if(unlikely(buffer->count == buffer->cap)) {
		_load_buffer(buffer);
	}

	// ensure there's at least sizeof(float) bytes in buffer
	ASSERT((buffer->cap - buffer->count) >= sizeof(float));

	// read value
	float v = *(float*)(buffer->buffer + buffer->count);

	// update offset
	buffer->count += sizeof(float);

	return v;
}

// read long double from buffer
long double BufferSerializerIO_ReadLongDouble
(
	void *io  // serializer
) {
	BufferedIO *buffer = (BufferedIO*)io;

	// load buffer if depleted
	if(unlikely(buffer->count == buffer->cap)) {
		_load_buffer(buffer);
	}

	// ensure there's at least sizeof(long double) bytes in buffer
	ASSERT((buffer->cap - buffer->count) >= sizeof(long double));

	// read value
	long double v = *(long double*)(buffer->buffer + buffer->count);

	// update offset
	buffer->count += sizeof(long double);

	return v;
}

// read buffer from stream
char *BufferSerializerIO_ReadBuffer
(
	void *io,       // stream
	size_t *lenptr  // number of bytes to read
) {
	BufferedIO *buffer = (BufferedIO*)io;

	// load buffer if depleted
	if(unlikely(buffer->count == buffer->cap)) {
		_load_buffer(buffer);
	}

	// check for large string
	if(unlikely(buffer->cap > BUFFER_SIZE)) {
		// large string stand on their own
		// they're not encoded within the buffer, they are the buffer
		ASSERT(buffer->count == 0);

		char *ret = (char*)buffer->buffer;

		if(lenptr != NULL) {
			*lenptr = buffer->cap;
		}

		// reset serializer buffer
		buffer->cap    = 0;
		buffer->buffer = NULL;

		return ret;
	}

	// expecting at least the string length
	ASSERT((buffer->cap - buffer->count) >= sizeof(uint64_t));

	// read buffer len
	uint64_t l = *(uint64_t*)(buffer->buffer + buffer->count);
	buffer->count += sizeof(uint64_t);

	// copy buffer
	char *v = rm_malloc(sizeof(char) * l);
	memcpy(v, buffer->buffer + buffer->count, l);

	buffer->count += l;

	return v;
}

//------------------------------------------------------------------------------
// Serializer Create API
//------------------------------------------------------------------------------

// create a serializer which uses stream
SerializerIO SerializerIO_FromStream
(
	FILE *f,      // stream
	bool encoder  // true for encoder, false decoder
) {
	ASSERT(f != NULL);

	SerializerIO serializer = rm_calloc(1, sizeof(struct SerializerIO_Opaque));

	serializer->stream  = (void*)f;
	serializer->encoder = encoder;

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
	RedisModuleIO *io,  // redis module io
	bool encoder        // true for encoder, false decoder
) {
	ASSERT(io != NULL);

	SerializerIO serializer = rm_calloc(1, sizeof(struct SerializerIO_Opaque));

	serializer->stream  = io;
	serializer->encoder = encoder;

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

// create a buffered serializer which uses RedisIO
SerializerIO SerializerIO_FromBufferedRedisModuleIO
(
	RedisModuleIO *io,  // redis module io
	bool encoder        // true for encoder, false decoder
) {
	ASSERT(io != NULL);

	BufferedIO *buffer_io = rm_malloc(sizeof(BufferedIO));

	buffer_io->stream = io;

	if(encoder == true) {
		// serializer used for graph encoding
		buffer_io->cap    = BUFFER_SIZE;
		buffer_io->count  = 0;
		buffer_io->buffer = rm_malloc(sizeof(unsigned char) * BUFFER_SIZE);
	} else {
		// serializer used for graph decoding
		buffer_io->cap    = 0;
		buffer_io->count  = 0;
		buffer_io->buffer = NULL;
	}

	SerializerIO serializer = rm_calloc(1, sizeof(struct SerializerIO_Opaque));

	// set serializer function pointers
	serializer->WriteUnsigned   = BufferSerializerIO_WriteUnsigned;
	serializer->WriteSigned     = BufferSerializerIO_WriteSigned;
	serializer->WriteString     = (void (*)(void*, RedisModuleString*))RedisModule_SaveString;
	serializer->WriteBuffer     = BufferSerializerIO_WriteBuffer;
	serializer->WriteDouble     = BufferSerializerIO_WriteDouble;
	serializer->WriteFloat      = BufferSerializerIO_WriteFloat;
	serializer->WriteLongDouble = BufferSerializerIO_WriteLongDouble;

	serializer->ReadFloat       = BufferSerializerIO_ReadFloat;
	serializer->ReadDouble      = BufferSerializerIO_ReadDouble;
	serializer->ReadSigned      = BufferSerializerIO_ReadSigned;
	serializer->ReadBuffer      = BufferSerializerIO_ReadBuffer;
	serializer->ReadString      = (RedisModuleString* (*)(void*))RedisModule_LoadString;
	serializer->ReadUnsigned    = BufferSerializerIO_ReadUnsigned;
	serializer->ReadLongDouble  = BufferSerializerIO_ReadLongDouble;

	serializer->stream  = buffer_io;
	serializer->encoder = encoder;

	return serializer;
}

// free serializer
void SerializerIO_Free
(
	SerializerIO *io  // serializer to free
) {
	ASSERT(io  != NULL);
	ASSERT(*io != NULL);

	SerializerIO _io = *io;

	// free internal buffer
	if(_io->encoder == true &&
       _io->WriteUnsigned == BufferSerializerIO_WriteUnsigned) {
		// flush remaining content before free
		_flush_buffer(_io->stream);
		rm_free(_io->stream);
	}

	rm_free(*io);
	*io = NULL;
}

