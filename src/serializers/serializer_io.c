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
	void (*WriteBuffer)(void*, const void*, size_t);  // write bytes

	float (*ReadFloat)(void*);                        // read float
	double (*ReadDouble)(void*);                      // read dobule
	int64_t (*ReadSigned)(void*);                     // read signed int
	uint64_t (*ReadUnsigned)(void*);                  // read unsigned int
	void* (*ReadBuffer)(void*, size_t*);              // read bytes
	long double	(*ReadLongDouble)(void*);             // read long double
	RedisModuleString* (*ReadString)(void*);          // read RedisModuleString

	bool encoder;   // true is serializer is used for encoding, false decoding
	void *stream;   // RedisModuleIO* or a Stream descriptor
	bool free_buff; // true if serializer has a buffer to free
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
	const void *buff,  // buffer
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
	const void *buff,  // buffer
	size_t n           // number of bytes to write
) {
	FILE *f = (FILE*)stream;

	// write size
	size_t written = fwrite(&n, sizeof(size_t), 1, f);
	ASSERT(written == 1);

	// write data
	if (n > 0) {
		written = fwrite(buff, n, 1, f);
		ASSERT(written == 1);
	}
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
static void *Stream_ReadBuffer
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

	void *data = rm_malloc(sizeof(char) * len);

	// read data
	if (len > 0) {
		read = fread(data, len, 1, f);
		ASSERT(read == 1);
	}

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
void *SerializerIO_ReadBuffer
(
	SerializerIO io,  // stream
	size_t *lenptr    // number of bytes to read
) {
	return io->ReadBuffer(io->stream, lenptr);
}

//------------------------------------------------------------------------------
// Buffered Serializer Read & Write API Version 2
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

// try to accommodate n additional bytes
// if there's no room the buffer is flushed
static inline bool _accommodate
(
	BufferedIO *buffer,  // buffer
	size_t n             // number of bytes to write
) {
	ASSERT(n > 0);

	// flush in case there's not enough room in buffer
	if((buffer->cap - buffer->count) < n) {
		_flush_buffer(buffer);

		// once flushed we can accommodate only if n <= buffer's capacity
		return n <= buffer->cap;
	}

	// there's enough space in buffer to accommodate additional n bytes
	return true;
}

// Must declare as a struct because C _Generic will not tell apart the following
// typedef char* blob_t
typedef struct { char* ptr; } blob_t;

// enum flaging the encoded types
// primitive types are prefixed by their serializer_type_t byte
typedef enum {
	SERIALIZER_TYPE_BYTES       = 0, // byte buffers have a length prefix after
	                                 // their type prefix
	SERIALIZER_TYPE_FLOAT       = 1,
	SERIALIZER_TYPE_DOUBLE      = 2,
	SERIALIZER_TYPE_SIGNED      = 3,
	SERIALIZER_TYPE_UNSIGNED    = 4,
	SERIALIZER_TYPE_LONG_DOUBLE = 5,
	SERIALIZER_TYPE_BLOB        = 6  // found at the end of a buffer, signals
	                                 // signals that the next write will be a
	                                 // standalone blob
} serializer_type_t;

#undef TYPE_ENCODE
#undef REQUIRED_SIZE

// Helper macro to encode the type
#define TYPE_ENCODE(t) ((uint8_t) _Generic((t){0},       \
	char*            : SERIALIZER_TYPE_BYTES,            \
	float            : SERIALIZER_TYPE_FLOAT,            \
	double           : SERIALIZER_TYPE_DOUBLE,           \
	int64_t          : SERIALIZER_TYPE_SIGNED,           \
	uint64_t         : SERIALIZER_TYPE_UNSIGNED,         \
	long double      : SERIALIZER_TYPE_LONG_DOUBLE,      \
	blob_t           : SERIALIZER_TYPE_BLOB))

// write the given type to buffer
#define SERIALIZER_WRITE_TYPE(t)                                \
ASSERT (buffer->count < buffer->cap);                           \
*((uint8_t*)(buffer->buffer + buffer->count)) = TYPE_ENCODE(t); \
buffer->count++;                                                \

// read the type
#define SERIALIZER_READ_TYPE(s)                                 \
s = *((uint8_t*)(buffer->buffer + buffer->count));              \
buffer->count++;

// check that the type being read matches expectations.
#define SERIALIZER_VALIDATE_TYPE(t)                             \
do {                                                            \
	assert (buffer->count < buffer->cap);                       \
	uint8_t s = *((uint8_t*)(buffer->buffer + buffer->count));  \
	buffer->count++;                                            \
	assert (s == TYPE_ENCODE(t));                               \
} while (0);


// Encode the size required to write PRIMITIVE types
#define REQUIRED_SIZE(t) (sizeof(t) + sizeof(uint8_t))

// macro for creating both read and write buffered RDB serializer functions
#define BUFFERED_SERIALIZERV2_READ_WRITE(suffix, t)             \
/* buffer serializerio write function*/                         \
static void BufferSerializerIOv2_Write##suffix(void *io, t v) { \
	BufferedIO *buffer = (BufferedIO*)io;                       \
                                                                \
	/* make sure buffer has enough room */                      \
	_accommodate(buffer, REQUIRED_SIZE(t));                     \
                                                                \
	/* in DEBUG mode we write the value type */                 \
	SERIALIZER_WRITE_TYPE(t)                                    \
                                                                \
	/* write value to buffer */                                 \
	*((t*)(buffer->buffer + buffer->count)) = v;                \
                                                                \
	/* update buffer offset */                                  \
	buffer->count += sizeof(t);                                 \
}                                                               \
                                                                \
/* buffer serializerio read function*/                          \
static t BufferSerializerIOv2_Read##suffix(void *io) {          \
	BufferedIO *buffer = (BufferedIO*)io;                       \
                                                                \
	/* load buffer if depleted */                               \
	if(unlikely(buffer->count == buffer->cap)) {                \
		_load_buffer(buffer);                                   \
	}                                                           \
                                                                \
	/* ensure there's at least sizeof(t) bytes in buffer */     \
	ASSERT((buffer->cap - buffer->count) >= REQUIRED_SIZE(t));  \
                                                                \
	/* validate type */                                         \
	SERIALIZER_VALIDATE_TYPE(t)                                 \
                                                                \
	/* read value */                                            \
	t v = *(t*)(buffer->buffer + buffer->count);                \
                                                                \
	/* update offset */                                         \
	buffer->count += sizeof(t);                                 \
                                                                \
	return v;                                                   \
}

//------------------------------------------------------------------------------
// create buffer serializer read & write functions
//------------------------------------------------------------------------------

// BufferSerializerIO_ReadFloat & BufferSerializerIO_WriteFloat
BUFFERED_SERIALIZERV2_READ_WRITE(Float, float)

// BufferSerializerIO_ReadDouble & BufferSerializerIO_WriteDouble
BUFFERED_SERIALIZERV2_READ_WRITE(Double, double)

// BufferSerializerIO_ReadSigned & BufferSerializerIO_WriteSigned
BUFFERED_SERIALIZERV2_READ_WRITE(Signed, int64_t)

// BufferSerializerIO_ReadUnsigned & BufferSerializerIO_WriteUnsigned
BUFFERED_SERIALIZERV2_READ_WRITE(Unsigned, uint64_t)

// BufferSerializerIO_ReadLongDouble & BufferSerializerIO_WriteLongDouble
BUFFERED_SERIALIZERV2_READ_WRITE(LongDouble, long double)

// write buffer to stream
void BufferSerializerIOv2_WriteBuffer
(
	void *io,           // serializer
	const void *value,  // value
	size_t len          // value size
) {
	ASSERT(io != NULL);

	BufferedIO *buffer = (BufferedIO*)io;

	if (REQUIRED_SIZE(size_t) + len > buffer->cap) {
		// value is too big, we will write as a blob
		// signal that a standalone buffer is coming then flush
		bool res = _accommodate(buffer, sizeof(uint8_t));
		ASSERT (res == true);
		SERIALIZER_WRITE_TYPE(blob_t);
		_flush_buffer(buffer);
		RedisModule_SaveStringBuffer(buffer->stream, value, len);
	} else {
		// make sure value has enough room
		bool res = _accommodate(buffer, REQUIRED_SIZE(size_t) + len);
		ASSERT (res == true) ;
		// write the value type
		SERIALIZER_WRITE_TYPE(char*);

		// add to buffer
		// write value length to stream
		*((size_t*)(buffer->buffer + buffer->count)) = len;
		buffer->count += sizeof(size_t);

		// write value to buffer
		memcpy(buffer->buffer + buffer->count, value, len);
		buffer->count += len;
	}
}

// read buffer from stream
void *BufferSerializerIOv2_ReadBuffer
(
	void *io,       // stream
	size_t *lenptr  // number of bytes to read
) {
	BufferedIO *buffer = (BufferedIO*)io;

	// load buffer if depleted
	if(unlikely(buffer->count == buffer->cap)) {
		_load_buffer(buffer);
	}

	ASSERT(buffer->cap > 0);

	// find the type
	uint8_t info_type;
	SERIALIZER_READ_TYPE(info_type);

	void *ret = NULL;

	// large string stand on their own
	// they're not encoded within the buffer, they are the buffer
	if (info_type == TYPE_ENCODE(blob_t)) {
		ASSERT (buffer->count == buffer->cap) ;
		_load_buffer(buffer);

		ret = buffer->buffer;

		if(lenptr != NULL) {
			*lenptr = buffer->cap;
		}

		// reset serializer buffer
		buffer->cap    = 0;
		buffer->count  = 0;
		buffer->buffer = NULL;
	} else {
	    ASSERT(info_type == TYPE_ENCODE(char *));
		// expecting at least the string length
		ASSERT((buffer->cap - buffer->count) >= sizeof(size_t));

		// read buffer len
		size_t l = *(size_t*)(buffer->buffer + buffer->count);

		buffer->count += sizeof(size_t);

		ASSERT((buffer->cap - buffer->count) >= l) ;

		// copy buffer
		ret = rm_malloc(sizeof(char) * l) ;
		memcpy(ret, buffer->buffer + buffer->count, l);

		buffer->count += l;
		if(lenptr != NULL) {
			*lenptr = l;
		}
	}
	return ret;
}

static void BufferSerializerIO_WriteUnsigned(void *io, uint64_t v);

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

	serializer->stream    = (void*)f;
	serializer->encoder   = encoder;
	serializer->free_buff = false;

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

	serializer->stream    = io;
	serializer->encoder   = encoder;
	serializer->free_buff = false;

	// set serializer function pointers
	serializer->WriteUnsigned   = (void (*)(void*, uint64_t))RedisModule_SaveUnsigned;
	serializer->WriteSigned     = (void (*)(void*, int64_t))RedisModule_SaveSigned;
	serializer->WriteString     = (void (*)(void*, RedisModuleString*))RedisModule_SaveString;
	serializer->WriteBuffer     = (void (*)(void*, const void*, size_t))RedisModule_SaveStringBuffer;
	serializer->WriteDouble     = (void (*)(void*, double))RedisModule_SaveDouble;
	serializer->WriteFloat      = (void (*)(void*, float))RedisModule_SaveFloat;
	serializer->WriteLongDouble = (void (*)(void*, long double))RedisModule_SaveLongDouble;

	serializer->ReadFloat       = (float (*)(void*))RedisModule_LoadFloat;
	serializer->ReadDouble      = (double (*)(void*))RedisModule_LoadDouble;
	serializer->ReadSigned      = (int64_t (*)(void*))RedisModule_LoadSigned;
	serializer->ReadBuffer      = (void* (*)(void*, size_t*))RedisModule_LoadStringBuffer;
	serializer->ReadString      = (RedisModuleString* (*)(void*))RedisModule_LoadString;
	serializer->ReadUnsigned    = (uint64_t (*)(void*))RedisModule_LoadUnsigned;
	serializer->ReadLongDouble  = (long double (*)(void*))RedisModule_LoadLongDouble;

	return serializer;
}

// create a buffered serializer which uses RedisIO
SerializerIO SerializerIOv2_FromBufferedRedisModuleIO
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
	serializer->WriteUnsigned   = BufferSerializerIOv2_WriteUnsigned;
	serializer->WriteSigned     = BufferSerializerIOv2_WriteSigned;
	serializer->WriteString     = (void (*)(void*, RedisModuleString*))RedisModule_SaveString;
	serializer->WriteBuffer     = BufferSerializerIOv2_WriteBuffer;
	serializer->WriteDouble     = BufferSerializerIOv2_WriteDouble;
	serializer->WriteFloat      = BufferSerializerIOv2_WriteFloat;
	serializer->WriteLongDouble = BufferSerializerIOv2_WriteLongDouble;

	serializer->ReadFloat       = BufferSerializerIOv2_ReadFloat;
	serializer->ReadDouble      = BufferSerializerIOv2_ReadDouble;
	serializer->ReadSigned      = BufferSerializerIOv2_ReadSigned;
	serializer->ReadBuffer      = BufferSerializerIOv2_ReadBuffer;
	serializer->ReadString      = (RedisModuleString* (*)(void*))RedisModule_LoadString;
	serializer->ReadUnsigned    = BufferSerializerIOv2_ReadUnsigned;
	serializer->ReadLongDouble  = BufferSerializerIOv2_ReadLongDouble;

	serializer->stream    = buffer_io;
	serializer->encoder   = encoder;
	serializer->free_buff = true;

	return serializer;
}

// free BufferedIO
void _BufferedIO_Free
(
	BufferedIO **io
) {
	ASSERT(io != NULL && *io != NULL);

	BufferedIO *_io = *io;

	// free internal buffer if allocated
	if(_io->buffer != NULL) {
		rm_free(_io->buffer);
	}

	// free buffer io and nullify
	rm_free(_io);
	*io = NULL;
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
	if(_io->free_buff) {

		// free bufferedIO
		BufferedIO *buffer_io = (BufferedIO*)_io->stream;

		if(_io->encoder == true) {
			// flush remaining content before free
			_flush_buffer(buffer_io);
		}

		// free buffer io
		_BufferedIO_Free((BufferedIO**) &_io->stream);
	}

	rm_free(*io);
	*io = NULL;
}

