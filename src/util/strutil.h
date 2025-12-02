/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

// return utf8 encoded length of string
int str_length
(
	const char *str
);

// return true if utf8 string is valid
bool str_utf8_validate
(
    const char *str
);

// convert ascii str to a lower case string and save it in lower
void str_tolower_ascii
(
	const char *str,
	char *lower,
	size_t *lower_len
);

// return utf8 string converted to lower case
char *str_tolower
(
	const char *str
);

// return utf8 string converted to upper case
char *str_toupper
(
	const char *str
);

// utility function to increase the size of a buffer
void str_ExtendBuffer
(
	char **buf,           // buffer to populate
	size_t *bufferLen,    // size of buffer
	size_t extensionLen   // number of bytes to add
);

// Utility function to check if the string matches
// the regex pattern.
bool str_MatchRegex
(
	const char* regex,   // regex pattern to match with
	const char* str      // string to match
);

// truncate string to max_len
// truncated = str[:min (n, max_len)]
//
// create a newly allocated truncated copy of `str`
// copies at most `max_len` bytes from `str` (not including null terminator)
// the returned string is always null-terminated
void str_truncate
(
	char **truncated,  // [output] truncated string
	const char *str,   // string to truncate
	uint len,          // string length
	uint max_len       // string max length
);

