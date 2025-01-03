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

typedef void (*field_cb)  (void *data, size_t n, void *pdata);
typedef void (*record_cb) (int t, void *pdata);

struct Opaque_CSVReader {
	FILE *file;                // CSV file handle
	struct csv_parser parser;  // CSV parser
	char delimiter;            // CSV delimiter
	SIValue row;               // parsed row
	SIValue *rows;             // parsed rows
	bool reached_eof;          // processed entire file
	field_cb cell_cb;          // function called for each cell
	record_cb row_cb;          // function called for each row
	SIValue *columns;          // CSV columns
	int col_idx;               // current processed column idx
	int step;                  // number of bytes to read in each call to fread
	char buffer[1024];         // input buffer
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

	// append cell to current row
	SIArray_Append(&reader->row, SI_ConstStringVal((char*)data));
}

// handle row by accumulating the row into an array of rows
static void _array_row_cb
(
    int t,       // record termination chracter
    void *pdata  // original buffer
) {
	CSVReader reader = (CSVReader)pdata;

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

	// append cell to current row
	SIValue key = reader->columns[reader->col_idx++];
	Map_Add(&reader->row, key, SI_ConstStringVal((char*)data));
}

// handle row by accumulating the row into an array of maps
static void _map_row_cb
(
    int t,       // record termination chracter
    void *pdata  // original buffer
) {
	CSVReader reader = (CSVReader)pdata;

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
	reader->step    = 1023;
	reader->row_cb  = _map_row_cb;
	reader->cell_cb = _map_cell_cb;

	array_append(reader->rows, SI_NullVal());
}

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
    const char *file_name,  // URI to CSV
    bool has_headers,       // first row is a header row
    char delimiter          // column delimiter character
) {
	ASSERT(file_name != NULL);

	//--------------------------------------------------------------------------
	// open the file in read mode
	//--------------------------------------------------------------------------

	FILE *file = fopen(file_name, "r");
	if (file == NULL) {
		ErrorCtx_RaiseRuntimeException("Error opening file");
		return NULL;
	}

	CSVReader reader = rm_calloc(1, sizeof(struct Opaque_CSVReader));

	reader->file        = file;
	reader->rows        = array_new(SIValue, 0);
	reader->delimiter   = delimiter;
	reader->reached_eof = false;

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
			ErrorCtx_RaiseRuntimeException("Failed reading CSV header row");
			CSVReader_Free(reader);
			return NULL;
		}
	} else {
		// CSV doesn't contains a header row
		// rows will return as arrays
		reader->row     = SIArray_New(0);
		reader->row_cb  = _array_row_cb;
		reader->cell_cb = _array_cell_cb;
		reader->step    = 1023;
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
				reader->file);

		// check if an error occurred during reading
		if(ferror(reader->file)) {
			ErrorCtx_RaiseRuntimeException("Error reading file");
			return SI_NullVal();
		}

		// no data was read
		if(bytesRead == 0) {
			// reached end of file
			ASSERT(feof(reader->file));
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
	CSVReader reader  // CSV reader to free
) {
	// close input file
	fclose(reader->file);

	int n;

	if(reader->columns != NULL) {
		n = array_len(reader->columns);
		for(int i = 0; i < n; i++) {
			SIValue_Free(reader->columns[i]);
		}
		array_free(reader->columns);
	}

	// free remaining rows
	n = array_len(reader->rows);
	for(int i = 0; i < n; i++) {
		SIValue_Free(reader->rows[i]);
	}
	array_free(reader->rows);

	// free current parsed row
	SIValue_Free(reader->row);

	// free csv parser
	csv_free(&reader->parser);

	rm_free(reader);
}

