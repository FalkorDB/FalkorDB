/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "src/util/rmalloc.h"
#include "src/serializers/serializer_io.h"

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

void setup() {
	Alloc_Reset();
}

#define TEST_INIT setup();
#include "acutest.h"

void test_serializer(void) {
	// create pipe
	int pipefd[2];  // read and write ends of a pipe
	TEST_ASSERT(pipe(pipefd) != -1);

	// create a FILE* stream both ends of the pipe
	FILE *fs_read_stream  = fdopen(pipefd[0], "r");
	FILE *fs_write_stream = fdopen(pipefd[1], "w");
	TEST_ASSERT(fs_read_stream  != NULL);
	TEST_ASSERT(fs_write_stream != NULL);
	
	SerializerIO reader = SerializerIO_FromStream(fs_read_stream);
	SerializerIO writer = SerializerIO_FromStream(fs_write_stream);
	TEST_ASSERT(reader != NULL);
	TEST_ASSERT(writer != NULL);

	//--------------------------------------------------------------------------
	// write to stream
	//--------------------------------------------------------------------------

	// write unsigned
	uint64_t unsigned_v = 2;
	SerializerIO_WriteUnsigned(writer, unsigned_v);

	// write signed
	int64_t signed_v = 3;
	SerializerIO_WriteSigned(writer, signed_v);

	// write buffer
	const char* write_buff = "data";
	size_t write_len = strlen(write_buff);
	SerializerIO_WriteBuffer(writer, write_buff, write_len);

	// write double
	double double_v = 4.5;
	SerializerIO_WriteDouble(writer, double_v);

	// write float
	float float_v = 6.7;
	SerializerIO_WriteFloat(writer, float_v);

	// write long double
	long double longdouble_v = 8.9;
	SerializerIO_WriteLongDouble(writer, longdouble_v);

	fclose(fs_write_stream);     // close write end
	close(pipefd[1]);            // close pipe
	SerializerIO_Free(&writer);  // free serializer

	//--------------------------------------------------------------------------
	// read from stream
	//--------------------------------------------------------------------------

	// read unsigned
	TEST_ASSERT(SerializerIO_ReadUnsigned(reader) == unsigned_v);

	// read signed
	TEST_ASSERT(SerializerIO_ReadSigned(reader) == signed_v);

	// read buffer
	size_t read_len;
	char *read_buff = SerializerIO_ReadBuffer(reader, &read_len);
	TEST_ASSERT(read_len == write_len);
	TEST_ASSERT(strcmp(read_buff, write_buff) == 0);
	rm_free(read_buff);

	// read double
	TEST_ASSERT(SerializerIO_ReadDouble(reader) == double_v);

	// read float
	TEST_ASSERT(SerializerIO_ReadFloat(reader) == float_v);

	// read long double
	TEST_ASSERT(SerializerIO_ReadLongDouble(reader) == longdouble_v);

	fclose(fs_read_stream);      // close file streams
	close(pipefd[0]);            // close pipe
	SerializerIO_Free(&reader);  // free serializer
}

void test_serializer_generic_write(void) {
	// create pipe
	int pipefd[2];  // read and write ends of a pipe
	TEST_ASSERT(pipe(pipefd) != -1);

	// create a FILE* stream both ends of the pipe
    FILE *fs_read_stream  = fdopen(pipefd[0], "r");
    FILE *fs_write_stream = fdopen(pipefd[1], "w");
	TEST_ASSERT(fs_read_stream  != NULL);
	TEST_ASSERT(fs_write_stream != NULL);
	
	SerializerIO reader = SerializerIO_FromStream(fs_read_stream);
	SerializerIO writer = SerializerIO_FromStream(fs_write_stream);
	TEST_ASSERT(reader != NULL);
	TEST_ASSERT(writer != NULL);

	//--------------------------------------------------------------------------
	// write to stream
	//--------------------------------------------------------------------------

	// write unsigned
	uint64_t unsigned_v = 2;
	SerializerIO_Write(writer, unsigned_v);

	// write signed
	int64_t signed_v = 3;
	SerializerIO_Write(writer, signed_v);

	// write double
	double double_v = 4.5;
	SerializerIO_Write(writer, double_v);

	// write float
	float float_v = 6.7;
	SerializerIO_Write(writer, float_v);

	// write long double
	long double longdouble_v = 8.9;
	SerializerIO_Write(writer, longdouble_v);

	fclose(fs_write_stream);     // close file streams
	close(pipefd[1]);            // close pipe
	SerializerIO_Free(&writer);  // free serializer

	//--------------------------------------------------------------------------
	// read from stream
	//--------------------------------------------------------------------------

	// read unsigned
	TEST_ASSERT(SerializerIO_ReadUnsigned(reader) == unsigned_v);

	// read signed
	TEST_ASSERT(SerializerIO_ReadSigned(reader) == signed_v);

	// read double
	TEST_ASSERT(SerializerIO_ReadDouble(reader) == double_v);

	// read float
	TEST_ASSERT(SerializerIO_ReadFloat(reader) == float_v);

	// read long double
	TEST_ASSERT(SerializerIO_ReadLongDouble(reader) == longdouble_v);

	fclose(fs_read_stream);      // close file streams
	close(pipefd[0]);	         // close pipe
	SerializerIO_Free(&reader);  // free serializer
}

TEST_LIST = {
	{ "serializer", test_serializer},
	{ "serializer_generic_write", test_serializer_generic_write},
	{ NULL, NULL }
};

