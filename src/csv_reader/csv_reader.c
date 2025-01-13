/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "csv_reader.h"
#include "libcsv/csv.h"
#include "../util/arr.h"
#include "../errors/errors.h"
#include "../datatypes/map.h"
#include "../datatypes/array.h"

#include <stdint.h>
#include <stdio.h>

#define DEFAULT_STEP 4096

typedef void (*field_cb)  (void *data, size_t n, void *pdata);
typedef void (*record_cb) (int t, void *pdata);

static char* empty_string = "";

struct Opaque_CSVReader {
	FILE *stream;               // CSV stream handle
	struct csv_parser parser;   // CSV parser
	char delimiter;             // CSV delimiter
	SIValue row;                // parsed row
	SIValue *rows;              // parsed rows
	bool reached_eof;           // processed entire stream
	field_cb cell_cb;           // function called for each cell
	record_cb row_cb;           // function called for each row
	SIValue *columns;           // CSV columns
	int col_idx;                // current processed column idx
	int col_count;              // number of columns in the last row
	int step;                   // number of bytes to read in each call to fread
	char buffer[DEFAULT_STEP];  // input buffer
};

//------------------------------------------------------------------------------
// cell & row callbacks
//------------------------------------------------------------------------------

// handle cell by adding it to an array
static void _array_cell_cb
(
    void *data,   // field data
    size_t n,     // number of bytes in data
    void *pdata   // original buffer
) {
	CSVReader reader = (CSVReader)pdata;

	if(unlikely(n == 0)) {
		// empty cell is treated as NULL
		ASSERT(data == NULL);
		SIArray_Append(&reader->row, SI_NullVal());
	} else {
		// append cell to current row
		SIArray_Append(&reader->row, SI_ConstStringVal((char*)data));
	}
}

// handle row by accumulating the row into an array of rows
static void _array_row_cb
(
    int t,       // record termination chracter
    void *pdata  // original buffer
) {
	CSVReader reader = (CSVReader)pdata;

	//--------------------------------------------------------------------------
	// verify that this new row has the same length as the previous row
	//--------------------------------------------------------------------------

	// on first row set column count
	if(unlikely(reader->col_count == -1)) {
		reader->col_count = SIArray_Length(reader->row);
	}

	if(unlikely(SIArray_Length(reader->row) != reader->col_count)) {
		ErrorCtx_RaiseRuntimeException("CSV row of unexpected length");
		return;
	}

	// done parsing row
	// add row to rows array and create a new empty row
	array_append(reader->rows, reader->row);
	reader->row = SIArray_New(SIArray_Length(reader->row));
}

// handle cell by adding it to a map
static void _map_cell_cb
(
    void *data,   // field data
    size_t n,     // number of bytes in data
    void *pdata   // original buffer
) {
	CSVReader reader = (CSVReader)pdata;

	if(unlikely(reader->col_idx >= reader->col_count)) {
		ErrorCtx_RaiseRuntimeException("CSV row of unexpected length");
		return;
	}

	SIValue key = reader->columns[reader->col_idx++];

	// empty cell is treated as an empty string
	if(unlikely(n == 0)) {
		ASSERT(data == NULL);
		// do not add key to map for missing cells
		return;
	}

	// append cell to current map
	Map_Add(&reader->row, key, SI_ConstStringVal((char*)data));
}

// handle row by accumulating the row into an array of maps
static void _map_row_cb
(
    int t,       // record termination chracter
    void *pdata  // original buffer
) {
	CSVReader reader = (CSVReader)pdata;

	//--------------------------------------------------------------------------
	// verify that this new row has the same length as the previous row
	//--------------------------------------------------------------------------

	if(unlikely(reader->col_idx != reader->col_count)) {
		ErrorCtx_RaiseRuntimeException("CSV row of unexpected length");
		return;
	}

	// done parsing row
	// add row to rows array and create a new empty row
	array_append(reader->rows, reader->row);

	// reset state
	reader->row     = Map_New(array_len(reader->columns));
	reader->col_idx = 0;
}

// handle cell when processing CSV header row
// add column name to reader's headers array
static void _header_cell_cb
(
    void *data,   // field data
    size_t n,     // number of bytes in data
    void *pdata   // original buffer
) {
	CSVReader reader = (CSVReader)pdata;

	// empty cell is treated as an empty string
	if(unlikely(n == 0)) {
		// invalid header
		ErrorCtx_RaiseRuntimeException("CSV empty column name");
		return;
	}

	// append cell to current row
	SIValue col = SI_DuplicateStringVal((char*)data);
	array_append(reader->columns, col);
}

// handle row when processing CSV header row
// update readers cell & row callbacks to map generation
static void _header_row_cb
(
    int t,       // record termination chracter
    void *pdata  // original buffer
) {
	CSVReader reader = (CSVReader)pdata;

	// update step, map cell and row callbacks
	reader->col_count = array_len(reader->columns);
	reader->step      = DEFAULT_STEP;
	reader->row_cb    = _map_row_cb;
	reader->cell_cb   = _map_cell_cb;

	array_append(reader->rows, SI_NullVal());
}

// read header row from CSV
static bool _read_header
(
	CSVReader reader
) {
	// processing header row should not return a row
	SIValue row = CSVReader_GetRow(reader);
	ASSERT(SI_TYPE(row) == T_NULL);

	if(array_len(reader->columns) == 0) {
		return false;
	}

	return true;
}

// create a new CSV reader
CSVReader CSVReader_New
(
	FILE *stream,      // CSV stream handle
	bool has_headers,  // first row is a header row
	char delimiter     // column delimiter character
) {
	ASSERT(stream != NULL);

	CSVReader reader = rm_calloc(1, sizeof(struct Opaque_CSVReader));

	reader->rows        = array_new(SIValue, 0);
	reader->stream      = stream;
	reader->delimiter   = delimiter;
	reader->reached_eof = false;
	reader->col_count   = -1;  // unknown number of columns

	//--------------------------------------------------------------------------
	// init csv parser
	//--------------------------------------------------------------------------

	// enables strict mode
	unsigned char options = CSV_STRICT | CSV_APPEND_NULL | CSV_EMPTY_IS_NULL;
	int res = csv_init(&(reader->parser), options);
	ASSERT(res == 0);

	// CSV has a header row
	// rows will return as maps
	if(has_headers) {
		reader->row     = Map_New(0);
		reader->columns = array_new(SIValue, 0);
		reader->row_cb  = _header_row_cb;
		reader->cell_cb = _header_cell_cb;
		reader->step    = 1;  // read one byte at a time when processing header

		if(!_read_header(reader)) {
			CSVReader_Free(&reader);
			ErrorCtx_RaiseRuntimeException("Failed reading CSV header row");
			return NULL;
		}
	} else {
		// CSV doesn't contains a header row
		// rows will return as arrays
		reader->row     = SIArray_New(0);
		reader->row_cb  = _array_row_cb;
		reader->cell_cb = _array_cell_cb;
		reader->step    = DEFAULT_STEP;
	}

	return reader;
}

// extract the current row
// returns either
// SIArray when CSV doesn't contains a header row
// SIMap when CSV does contains a header row
SIValue CSVReader_GetRow
(
	CSVReader reader  // CSV reader
) {
	ASSERT(reader != NULL);
	
	// try to parse additional data
	while(!reader->reached_eof && array_len(reader->rows) == 0) {
		// read up to step bytes from the file
		size_t bytesRead = fread(reader->buffer, sizeof(char), reader->step,
				reader->stream);

		// check if an error occurred during reading
		if(ferror(reader->stream)) {
			ErrorCtx_RaiseRuntimeException("Error reading file");
			return SI_NullVal();
		}

		// no data was read
		if(bytesRead == 0) {
			// reached end of file
			ASSERT(feof(reader->stream));
			reader->reached_eof = true;

			// last call to csv parser
			int res = csv_fini(&reader->parser, reader->cell_cb, reader->row_cb,
					reader);
			ASSERT(res == 0);

			break;
		}

		// process buffer
		size_t bytesProcessed =
			csv_parse(&(reader->parser), reader->buffer, bytesRead,
					reader->cell_cb, reader->row_cb, (void*)reader);

		// expecting number of bytes processed to equal number of bytes read
		if(bytesRead != bytesProcessed) {
			ErrorCtx_RaiseRuntimeException("csv reader error: %s\n",
					csv_strerror(csv_error(&reader->parser)));
			return SI_NullVal();
		}
	}

	if(array_len(reader->rows) > 0) {
		return array_pop(reader->rows);
	}

	return SI_NullVal();
}

// free CSV reader
void CSVReader_Free
(
	CSVReader *reader  // CSV reader to free
) {
	ASSERT(reader != NULL && *reader != NULL);
	CSVReader _reader = *reader;

	// close stream
	fclose(_reader->stream);

	// free header columns
	if(_reader->columns != NULL) {
		array_free_cb(_reader->columns, SIValue_Free);
	}

	// free current parsed row
	SIValue_Free(_reader->row);

	// free remaining rows
	array_free_cb(_reader->rows, SIValue_Free);

	// free csv parser
	csv_free(&_reader->parser);

	rm_free(_reader);

	*reader = NULL;
}

