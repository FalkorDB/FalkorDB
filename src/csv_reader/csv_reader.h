/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "../value.h"

typedef struct Opaque_CSVReader *CSVReader;

// create a new CSV reader
CSVReader CSVReader_New
(
	FILE *stream,      // CSV stream handle
	bool has_headers,  // first row is a header row
	char delimiter     // column delimiter character
);

// extract the current row
// returns either
// SIArray when CSV doesn't contains a header row
// SIMap when CSV does contains a header row
// SINull value upon failure to produce a row
SIValue CSVReader_GetRow
(
	const CSVReader reader  // CSV reader
);

// free CSV reader
void CSVReader_Free
(
	CSVReader reader  // CSV reader to free
);

