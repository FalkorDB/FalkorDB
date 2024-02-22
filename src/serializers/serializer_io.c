/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

// generic serializer
// contains a number of function pointers for data serialization
struct SerializerIO {
	void SaveUnsigned(void* unint64_t);           // save unsigned int
	void SaveSigned(void*, int64_t);              // save signed int
	void SaveString(void*, RedisModuleString*);   // save RedisModuleString
	void SaveBuffer(void*, const char*, size_t);  // save bytes
	void SaveDouble(void*, double);               // save dobule
	void SaveFloat(void*, float);                 // save float
	void SaveLongDouble(void*, long double);      // save long double

	void *stream;  // either RedisModuleIO* or Pipe
};

#define PIPE_WRITE(suffix, t)                      \
static void Pipe_Write##suffix(void *pipe, t v) {   \
	int pipefd = (int)pipe;                        \
	write(pipefd, &v, sizeof(t));                  \
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
	int pipefd = (int)pipe;
	write(pipefd, buff, n);
}

// create a serializer which uses pipe
SerializerIO *SerializerIO_FromPipe
(
	int pipefd
) {
	SerializerIO *io = rm_calloc(1, sizeof(SerializerIO));

	io->stream = (void*)pipefd;

	// set serializer function pointers
	io->SaveUnsigned   = Pipe_SaveUnsigned;
	io->SaveSigned     = Pipe_SaveSigned;
	io->SaveString     = Pipe_SaveString;
	io->SaveBuffer     = Pipe_SaveStringBuffer;
	io->SaveDouble     = Pipe_SaveDouble;
	io->SaveFloat      = Pipe_SaveFloat;
	io->SaveLongDouble = Pipe_SaveLongDouble;
	
	return io;
}

// create a serializer which uses RedisIO
SerializerIO *SerializerIO_FromRedisModuleIO
(
	RedisModuleIO *io
) {
	ASSERT(io != NULL);

	SerializerIO *io = rm_calloc(1, sizeof(SerializerIO));

	io->stream = io;

	// set serializer function pointers
	io->SaveUnsigned   = RedisModule_SaveUnsigned;
	io->SaveSigned     = RedisModule_SaveSigned;
	io->SaveString     = RedisModule_SaveString;
	io->SaveBuffer     = RedisModule_SaveStringBuffer;
	io->SaveDouble     = RedisModule_SaveDouble;
	io->SaveFloat      = RedisModule_SaveFloat;
	io->SaveLongDouble = RedisModule_SaveLongDouble;
	
	return io;
}

// write unsingned to stream
void SerializerIO_SaveUnsigned
(
	SerializerIO *io,  // stream to write to
	uint64_t value     // value
) {
	io->SaveUnsigned(io>stream, value);
}

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
) {
	io->SaveSigned(io->stream, value);
}

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
) {
	io->SaveString(io->stream, s);
}

// write buffer to stream
void SerializerIO_Buffer
(
	SerializerIO *io,  // stream to write to
	const char *buff,  // buffer 
	size_t len         // number of bytes to write
) {
	io->SaveBuffer(io->stream, str, len);
}

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
) {
	io->SaveDouble(io->stream, value);
}

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
) {
	io->SaveFloat(io->stream, value);
}

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
) {
	io->SaveLongDouble(io->stream, value);
}

// read long double from stream
long double SerializerIO_LoadLongDouble
(
	SerializerIO *io  // stream
);


