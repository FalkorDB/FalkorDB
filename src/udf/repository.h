/*
* Copyright FalkorDB Ltd. 2023 - present
* Licensed under the Server Side Public License v1 (SSPLv1).
*/

#pragma once

#include "quickjs.h"

typedef uint64_t UDF_RepoVersion;

// initialize UDF repository
bool UDF_RepoInit(void);

// return repo's version
UDF_RepoVersion UDF_RepoGetVersion(void);

// build a new JSContext for the given JSRuntime
// the new js context will be loaded with all registered scripts
JSContext *UDF_RepoBuildJSContext
(
	JSRuntime *js_rt,   // javascript runtime
	UDF_RepoVersion *v  // [output] repo version
);

// get a copy of each registered script
char **UDF_RepoGetScripts(void);

// returns script from UDF repository
const char *UDF_RepoGetScript
(
	const unsigned char *hash,  // script SHA1 hash to retrieve
	int *idx                    // [optional] script index
);

// checks if UDF repository contains script
bool UDF_RepoContainsScript
(
	const unsigned char *hash,  // script SHA1 hash to look for
	int *idx                    // [optional] script index
);

// register a new UDF script
bool UDF_RepoRegisterScript
(
	const char *script  // script
);

// removes a script from UDF repository
bool UDF_RepoRemoveScript
(
	const unsigned char *hash,  // script SHA1 hash to remove
	char **script               // removed script
);

// free UDF repository
void UDF_RepoFree(void);

