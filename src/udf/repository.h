/*
* Copyright FalkorDB Ltd. 2023 - present
* Licensed under the Server Side Public License v1 (SSPLv1).
*/

#pragma once

// initialize UDF repository
bool UDF_RepoInit(void);

// returns script from UDF repository
const char *UDF_RepoGetScript
(
	const unsigned char *hash  // script SHA1 hash to retrieve
);

// checks if UDF repository contains script
bool UDF_RepoContainsScript
(
	const unsigned char *hash  // script SHA1 hash to look for
);

// register a new UDF script
bool UDF_RepoRegisterScript
(
	const char *script  // script
);

// removes a script from UDF repository
bool UDF_RepoRemoveScript
(
	const char *hash  // script SHA1 hash to remove
);

// free UDF repository
void UDF_RepoFree(void);

