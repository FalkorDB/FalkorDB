/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include <stdint.h>
#include <stdio.h>

typedef struct Opaque_CSVReader *CSVReader;

// create a new CSV reader
CSVReader CSVReader_New
(
    const char *file_name,  // URI to CSV
    bool has_headers,       // first row is a header row
    char delimiter          // column delimiter character
);

// returns the number of columns in CSV file
size_t CSVReader_ColumnCount
(
    const CSVReader reader  // CSV reader
);

// extracts the header row
// length of 'values' and 'lengths' arrays must be the same
// returns true on success false otherwise
bool CSVReader_GetHeaders
(
    const CSVReader reader,  // CSV reader
	const char **values,     // header values
	const size_t *lengths    // length of each value
);

// extract the current row
// length of 'values' and 'lengths' arrays must be the same
// returns true on success false indicates either an error or EOF
bool CSVReader_GetRow
(
	const CSVReader reader,  // CSV reader
	const char **values,     // row values
	const size_t *lengths    // length of each value
);

// free CSV reader
void CSVReader_Free
(
	CSVReader reader  // CSV reader to free
);
