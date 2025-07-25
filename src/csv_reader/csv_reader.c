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

#include <poll.h>
#include <errno.h>
#include <stdio.h>
#include <unistd.h>


#define DEFAULT_STEP 4096

static const unsigned char BOM[3] = {0xef, 0xbb, 0xbf};

typedef void (*field_cb)  (void *data, size_t n, void *pdata);
typedef void (*record_cb) (int t, void *pdata);

static char* empty_string = "";

struct Opaque_CSVReader {
	FILE *stream;               // CSV stream handle
	bool search_for_bom;        // flag indicating if we need to search for BOM
	int fd;                     // stream file descriptor
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
	unsigned char buffer[DEFAULT_STEP];  // input buffer
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
		ErrorCtx_SetError("CSV row of unexpected length");

		// adding NULL to the rows array will cause our reader to stop
		// pulling additional rows
		array_append(reader->rows, SI_NullVal());
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
		ErrorCtx_SetError("CSV row of unexpected length");
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
		ErrorCtx_SetError("CSV row of unexpected length");

		// adding NULL to the rows array will cause our reader to stop
		// pulling additional rows
		array_append(reader->rows, SI_NullVal());
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
		ErrorCtx_SetError("CSV empty column name");
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

	// disable buffering on the stream
	int res = setvbuf(stream, NULL, _IONBF, 0);  // use unbuffered mode
	assert(res == 0);

	CSVReader reader = rm_calloc(1, sizeof(struct Opaque_CSVReader));

	reader->rows           = array_new(SIValue, 0);
	reader->stream         = stream;
	reader->fd             = fileno(stream);
	reader->delimiter      = delimiter;
	reader->col_count      = -1;    // unknown number of columns
	reader->reached_eof    = false;
	reader->search_for_bom = true;

	//--------------------------------------------------------------------------
	// init csv parser
	//--------------------------------------------------------------------------

	// enables strict mode
	unsigned char options = CSV_STRICT | CSV_APPEND_NULL | CSV_EMPTY_IS_NULL;
	res = csv_init(&(reader->parser), options);
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

	short bom_idx = 0;
	
	// try to parse additional data
	while(!reader->reached_eof && array_len(reader->rows) == 0) {

		//----------------------------------------------------------------------
		// pool on stream
		//----------------------------------------------------------------------

		struct pollfd pfd = {
			.fd = reader->fd,
			.events = POLLIN
		};

		ssize_t bytesRead = 0;
		int ret = poll(&pfd, 1, 5000);  // wait up to 5 second
		if (ret < 0) {
			RedisModule_Log(NULL, "warning",
					"CSV reader: poll failed: %s\n", strerror(errno));
			return SI_NullVal();
		} else if (ret == 0) {
			RedisModule_Log(NULL, "warning",
					"CSV reader: timeout while waiting for data (5s)");
			return SI_NullVal();
		} else if (pfd.revents & POLLIN || pfd.revents & POLLHUP) {
			// read up to step bytes from the file
			bytesRead = read(reader->fd, reader->buffer, reader->step);
		} else {
			RedisModule_Log(NULL, "warning", "Unexpected poll revents: 0x%x",
					pfd.revents);
			return SI_NullVal();
		}

		// read error
		if(bytesRead < 0) {
			RedisModule_Log(NULL, "warning", "read failed: %s", strerror(errno));
			return SI_NullVal();
		}

		// no data was read
		if(bytesRead == 0) {
			// reached end of file
			reader->reached_eof = true;

			// last call to csv parser
			int res = csv_fini(&reader->parser, reader->cell_cb, reader->row_cb,
					reader);
			ASSERT(res == 0);

			break;
		}

		size_t offset = 0;

		// try to consume BOM bytes
		if (reader->search_for_bom) {
			int i = 0;
			int n = MIN(bytesRead, 3 - bom_idx);

			for (; i < n; i++, bom_idx++) {
				if (reader->buffer[i] != BOM[bom_idx]) {
					// BOM mismatch; stop searching
					reader->search_for_bom = false;
					break;
				}
			}

			// some BOM bytes been skipped
			// but the entire BOM sequence wasn't matched
			if (reader->search_for_bom == false && bom_idx > 0 && bom_idx < 3) {
				// in this case we decide to raise an exception as we're not
				// replaying the skipped bytes
				RedisModule_Log(NULL, "warning", "BOM partial match");
				return SI_NullVal();
			}

			if (reader->search_for_bom) {
				offset    += n;  // skip BOM bytes
				bytesRead -= n;  // update number of non-BOM bytes remaining

				if (bom_idx == 3) {
					reader->search_for_bom = false; // fully matched
				}
			}
		}

		// process buffer
		size_t bytesProcessed =
			csv_parse(&(reader->parser), reader->buffer + offset, bytesRead,
					reader->cell_cb, reader->row_cb, (void*)reader);

		// expecting number of bytes processed to equal number of bytes read
		if(bytesRead != bytesProcessed) {
			ErrorCtx_SetError("csv reader error: %s\n",
					csv_strerror(csv_error(&reader->parser)));
			return SI_NullVal();
		}
	}

	// either at leaset one row was generated or we've reached EOF
	// either way no need to continue searching for bom
	reader->search_for_bom = false;

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

