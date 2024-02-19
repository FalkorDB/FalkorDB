struct SerializerIO {
	void SaveUnsigned(void* unint64_t);
	void SaveSigned(void*, int64_t);
	void SaveString(void*, RedisModuleString*);
	void SaveBuffer(void*, const char*, size_t);
	void SaveDouble(void*, double);
	void SaveFloat(void*, float);
	void SaveLongDouble(void*, long double);

	void *stream;  // either RedisModuleIO* or ...
};

SerializerIO *SeializerIO_FromRedisModuleIO
(
	RedisModuleIO *io
) {
	SerializerIO *io = rm_calloc(1, sizeof(SerializerIO));

	io->stream         = io;
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
