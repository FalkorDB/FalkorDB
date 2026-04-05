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

	bool encoder;  // true is serializer is used for encoding, false decoding
	void *stream;  // RedisModuleIO* or a Stream descriptor
	bool free_buff; // true if serializer has a buffer to free
};

//------------------------------------------------------------------------------
// PREVIOUS Buffered Serializer Read & Write API
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

#if SERIALIZER_DEBUG

	// encoded value types, used for debugging purposes
	static char Bytes      = 0;
	static char Float      = 1;
	static char Double     = 2;
	static char Signed     = 3;
	static char Unsigned   = 4;
	static char LongDouble = 5;

	// macro to map types to encoded values
	#define TYPE_ENCODE(t)                 \
		_Generic((t)0,                     \
				char*       : Bytes,       \
				float       : Float,       \
				double      : Double,      \
				int64_t     : Signed,      \
				uint64_t    : Unsigned,    \
				long double : LongDouble,  \
				default: Bytes)

	#define DEBUG_WRITE_TYPE(t)                                       \
		/* write type to buffer */                                    \
		*((char*)(buffer->buffer + buffer->count)) = TYPE_ENCODE(t);  \
                                                                      \
		/* update buffer offset */                                    \
		buffer->count++;

	#define DEBUG_VALIDATE_TYPE(t)                                    \
		/* validate type */                                           \
		char s = *(char*)(buffer->buffer + buffer->count);            \
		buffer->count++;                                              \
		ASSERT(s == TYPE_ENCODE(t));

	#define REQUIRED_SIZE(t) (sizeof(t) + 1)
#else
	#define DEBUG_WRITE_TYPE(t)           /* nop */
	#define DEBUG_VALIDATE_TYPE(t)        /* nop */
	#define REQUIRED_SIZE(t) (sizeof(t))
#endif

// macro for creating both read and write buffered RDB serializer functions
#define BUFFERED_SERIALIZER_READ_WRITE(suffix, t)               \
/* buffer serializerio write function*/                         \
static void BufferSerializerIO_Write##suffix(void *io, t v) {   \
	BufferedIO *buffer = (BufferedIO*)io;                       \
                                                                \
	/* make sure buffer has enough room */                      \
	_accommodate(buffer, REQUIRED_SIZE(t));                     \
                                                                \
	/* in DEBUG mode we write the value type */                 \
	DEBUG_WRITE_TYPE(t)                                         \
                                                                \
	/* write value to buffer */                                 \
	*((t*)(buffer->buffer + buffer->count)) = v;                \
                                                                \
	/* update buffer offset */                                  \
	buffer->count += sizeof(t);                                 \
}                                                               \
                                                                \
/* buffer serializerio read function*/                          \
static t BufferSerializerIO_Read##suffix(void *io) {            \
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
	DEBUG_VALIDATE_TYPE(t)                                      \
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
BUFFERED_SERIALIZER_READ_WRITE(Float, float)

// BufferSerializerIO_ReadDouble & BufferSerializerIO_WriteDouble
BUFFERED_SERIALIZER_READ_WRITE(Double, double)

// BufferSerializerIO_ReadSigned & BufferSerializerIO_WriteSigned
BUFFERED_SERIALIZER_READ_WRITE(Signed, int64_t)

// BufferSerializerIO_ReadUnsigned & BufferSerializerIO_WriteUnsigned
BUFFERED_SERIALIZER_READ_WRITE(Unsigned, uint64_t)

// BufferSerializerIO_ReadLongDouble & BufferSerializerIO_WriteLongDouble
BUFFERED_SERIALIZER_READ_WRITE(LongDouble, long double)

// write buffer to stream
void BufferSerializerIO_WriteBuffer
(
	void *io,           // serializer
	const void *value,  // value
	size_t len          // value size
) {
	ASSERT(io != NULL);

	BufferedIO *buffer = (BufferedIO*)io;

	// make sure value has enough room
	if(_accommodate(buffer, len + REQUIRED_SIZE(size_t))) {
		// in DEBUG mode we write the value type
		DEBUG_WRITE_TYPE(char*)

		// add to buffer
		// write value length to stream
		*((size_t*)(buffer->buffer + buffer->count)) = len;
		buffer->count += sizeof(size_t);

		// write value to buffer
		if (len > 0) {
			memcpy(buffer->buffer + buffer->count, value, len);
			buffer->count += len;
		}
	} else {
		// value is too big
		// buffer had been flushed by '_accommodate'
		RedisModule_SaveStringBuffer(buffer->stream, value, len);
	}
}

// read buffer from stream
void *BufferSerializerIO_ReadBuffer
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

		void *ret = buffer->buffer;

		if(lenptr != NULL) {
			*lenptr = buffer->cap;
		}

		// reset serializer buffer
		buffer->cap    = 0;
		buffer->buffer = NULL;

		return ret;
	}

	// expecting at least the string length
	ASSERT((buffer->cap - buffer->count) >= REQUIRED_SIZE(size_t));

	// in DEBUG mode we validate the value type
	DEBUG_VALIDATE_TYPE(char*);

	// read buffer len
	size_t l = *(size_t*)(buffer->buffer + buffer->count);

	// bug mitigation
	if (unlikely(buffer->count == 0 &&
		         buffer->cap >= (BUFFER_SIZE - sizeof (size_t)))) {
		// printf ("buffer->cap: %zu\n", buffer->cap) ;
		// printf ("l: %zu\n", l) ;
		if (l > buffer->cap) {
			printf ("detected a large string!\n") ;
			// this is a large string
			// large string stand on their own
			// they're not encoded within the buffer, they are the buffer
			void *ret = buffer->buffer ;

			if(lenptr != NULL) {
				*lenptr = buffer->cap ;
			}

			// reset serializer buffer
			buffer->cap    = 0 ;
			buffer->count  = 0 ;
			buffer->buffer = NULL ;

			return ret ;
		}
	}

	buffer->count += sizeof(size_t);

	// copy buffer
	void *v = rm_malloc(sizeof(char) * l);

	if (l > 0) {
		ASSERT ((buffer->cap - buffer->count) >= l) ;
		memcpy(v, buffer->buffer + buffer->count, l);
	}

	buffer->count += l;

	if(lenptr != NULL) {
		*lenptr = l;
	}

	return v;
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

	serializer->stream    = buffer_io;
	serializer->encoder   = encoder;
	serializer->free_buff = true;

	return serializer;
}

