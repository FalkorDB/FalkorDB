/*
* Copyright FalkorDB Ltd. 2023 - present
* Licensed under the Server Side Public License v1 (SSPLv1).
*/

#pragma once

// initialize UDF repository
bool UDF_RepoInit(void);

// search UDF repository for function
const char *UDF_RepoGetScript
(
	const char *func_name  // function name to look for
);

// returns true if UDF repository contains script
bool UDF_RepoContainsScript
(
	const char *func_name  // function name to look for
);

// register a new UDF script
bool UDF_RepoSetScript
(
	const char *func_name,  // function name
	const char *script      // script
);

// free UDF repository
void UDF_RepoFree(void);

