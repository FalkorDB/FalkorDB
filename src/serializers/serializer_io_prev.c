/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "serializer_io.h"
#include "serializer_io_internal.h"
#include "../util/rmalloc.h"

#include <stdio.h>
#include <stdint.h>
#include <unistd.h>

#define BUFFER_SIZE 256000  // buffered searializer buffer size 256KB

//------------------------------------------------------------------------------
// PREVIOUS Buffered Serializer Read & Write API
//------------------------------------------------------------------------------

// load buffer from stream to memory
// returns false on a short read (the underlying RedisModuleIO latched an error)
static bool _load_buffer
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
	size_t cap = 0;
	buffer->buffer =
		(unsigned char*)RedisModule_LoadStringBuffer(buffer->stream, &cap);

	// reset offset
	buffer->count = 0;

	// a short read yields a NULL buffer and latches the IO error on the stream
	if(buffer->buffer == NULL || RedisModule_IsIOError(buffer->stream)) {
		buffer->cap = 0;
		return false;
	}

	// the encoder never flushes an empty buffer
	if(unlikely(cap == 0)) {
		Buffered_SetCorrupt(buffer, "empty serializer buffer");
		buffer->cap = 0;
		return false;
	}

	buffer->cap = cap;
	return true;
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

	#define DEBUG_VALIDATE_TYPE(t, ret)                               \
		/* validate type */                                           \
		char s = *(char*)(buffer->buffer + buffer->count);            \
		buffer->count++;                                              \
		if(unlikely(s != TYPE_ENCODE(t))) {                           \
			Buffered_SetCorrupt(buffer, "unexpected value type");     \
			return ret;                                               \
		}

	#define REQUIRED_SIZE(t) (sizeof(t) + 1)
#else
	#define DEBUG_WRITE_TYPE(t)           /* nop */
	#define DEBUG_VALIDATE_TYPE(t, ret)   /* nop */
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
		if(!_load_buffer(buffer)) {                             \
			/* short read - latched via Buffered_IsError */     \
			return (t)0;                                        \
		}                                                       \
	}                                                           \
                                                                \
	/* a malformed buffer may not hold a full value */          \
	if(unlikely((buffer->cap - buffer->count) < REQUIRED_SIZE(t))) { \
		Buffered_SetCorrupt(buffer, "value overruns its buffer"); \
		return (t)0;                                            \
	}                                                           \
                                                                \
	/* validate type */                                         \
	DEBUG_VALIDATE_TYPE(t, (t)0)                                \
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
		if(!_load_buffer(buffer)) {
			// short read - the read layer latches via Buffered_IsError
			if(lenptr != NULL) *lenptr = 0;
			return NULL;
		}
	}

	// check for large string
	if(unlikely(buffer->cap > BUFFER_SIZE)) {
		// large string stand on their own
		// they're not encoded within the buffer, they are the buffer
		if(unlikely(buffer->count != 0)) {
			Buffered_SetCorrupt(buffer, "misplaced large string");
			if(lenptr != NULL) *lenptr = 0;
			return NULL;
		}

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
	if(unlikely((buffer->cap - buffer->count) < REQUIRED_SIZE(size_t))) {
		Buffered_SetCorrupt(buffer, "no room for a buffer length");
		if(lenptr != NULL) *lenptr = 0;
		return NULL;
	}

	// in DEBUG mode we validate the value type
	DEBUG_VALIDATE_TYPE(char*, NULL);

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

	// the declared length must fit in what is left of the buffer
	if(unlikely(l > (buffer->cap - buffer->count))) {
		Buffered_SetCorrupt(buffer, "buffer length overruns its buffer");
		if(lenptr != NULL) *lenptr = 0;
		return NULL;
	}

	// copy buffer
	void *v = rm_malloc(sizeof(char) * l);

	if (l > 0) {
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

	BufferedIO *buffer_io = rm_calloc(1, sizeof(BufferedIO));

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

	// decoders latch short reads via the backend error probe
	if(!encoder) serializer->IsError = Buffered_IsError;

	return serializer;
}

