/*
* Copyright FalkorDB Ltd. 2023 - present
* Licensed under the Server Side Public License v1 (SSPLv1).
*/

#pragma once

#include "quickjs.h"

typedef uint64_t UDF_RepoVersion;

// initialize UDF repository
bool UDF_RepoInit(void);

// bumps repo's version by 1
// function is exposed to allow quickjs runtime heap & stack size adjustment
// via runtime re-configuration, once the user changes either the heap or stack
// size limit the configuration handler would bump the UDFs repo version
// without changing its internal content, this will force each query execution
// thread to reload a new quickjs runtime picking up on the new limits
void UDF_RepoBumpVersion(void);

// return repo's version
UDF_RepoVersion UDF_RepoGetVersion(void);

// populate the JSContext with registered libs
void UDF_RepoPopulateJSContext
(
	JSContext *js_ctx,  // context to populate
	UDF_RepoVersion *v  // [output] repo version
);

// returns number of registered libs
unsigned int UDF_RepoLibsCount(void);

// get lib by name
bool UDF_RepoGetLib
(
	const char *name,         // lib's name
	const char ***functions,  // [optional] [output] lib's functions
	const char **script       // [optional] [output] lib's script
);

// get lib by index
void UDF_RepoGetLibIdx
(
	unsigned int idx,         // lib's index
	const char **name,        // [optional] [output] lib's name
	const char ***functions,  // [optional] [output] lib's functions
	const char **script       // [optional] [output] lib's script
);

// returns script from UDF repository
const char *UDF_RepoGetScript
(
	const char *lib  // UDF library
);

// checks if UDF repository contains function
bool UDF_RepoContainsFunc
(
	const char *lib,  // UDF library
	const char *func  // UDF function
);

// checks if UDF repository contains lib
bool UDF_RepoContainsLib
(
	const char *lib,   // UDF library
	unsigned int *idx  // [optional] library index
);

// register a new UDF library
bool UDF_RepoRegisterLib
(
	const char *lib,    // library
	const char *script  // script
);

// register a new function for library
bool UDF_RepoRegisterFunc
(
	const char *lib,  // library
	const char *func  // function
);

// removes a UDF library from repository
bool UDF_RepoRemoveLib
(
	const char *lib,  // UDF library
	char **script     // [optional] [output] removed script
);

// expose library by:
// 1. bumping repository version (causing others to pick up the latest version)
// 2. introduce library's functions to the global UDF functions repo
void UDF_RepoExposeLib
(
	const char *lib  // library to expose
) ;

// free UDF repository
void UDF_RepoFree(void);

