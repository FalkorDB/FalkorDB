/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include <string.h>
#include <ctype.h>
#include "rmalloc.h"
#include "utf8proc/utf8proc.h"
#include "oniguruma/src/oniguruma.h"

// convert ascii str to a lower case string and save it in lower
void str_tolower_ascii
(
	const char *str,
	char *lower,
	size_t *lower_len
) {
	size_t str_len = strlen(str);
	// avoid overflow
	ASSERT(*lower_len >= str_len);

	// update lower len
	*lower_len = str_len;

	for(size_t i = 0; i < str_len; i++) lower[i] = tolower(str[i]);
	lower[str_len] = 0;
}

// return true if utf8 string is valid
bool str_utf8_validate
(
	const char *str
) {
	// hold current Unicode character
	utf8proc_int32_t c;

	// while we didn't get to the end of the string
	while(str[0] != 0) {
		// increment current position by number of bytes in Unicode character
		str += utf8proc_iterate((const utf8proc_uint8_t *)str, -1, &c);
		if(c == -1) {
			// string is not valid
			return false;
		}
	}

	// string is valid
	return true;
}

// determine the length of the string
// not in terms of the number of bytes in the string
// but in terms of the number of Unicode characters in the string
int str_length
(
	const char *str
) {
	// hold current Unicode character
	utf8proc_int32_t c;

	int len = 0;

	// while we didn't get to the end of the string
	while(str[0] != 0) {
		// increment current position by number of bytes in Unicode character
		str += utf8proc_iterate((const utf8proc_uint8_t *)str, -1, &c);
		// increment length of the string in terms of Unicode characters
		len++;
	}

	// return the length of the string in terms of Unicode characters
	return len;
}

char *str_tolower
(
	const char *str
) {
	// hold current Unicode character
	utf8proc_int32_t c;
	const utf8proc_uint8_t *str_i = (const utf8proc_uint8_t *)str;
	size_t lower_len = 0;
	utf8proc_uint8_t buffer[4];
	while(str_i[0] != 0) {
		str_i     += utf8proc_iterate(str_i, -1, &c);
		if(c == -1) {
			// Unicode character is not valid
			return NULL;
		}
		lower_len += utf8proc_encode_char(utf8proc_tolower(c), buffer);
	}
	
	utf8proc_uint8_t *lower = rm_malloc(lower_len + 1);

	str_i = (const utf8proc_uint8_t *)str;
	// while we didn't get to the end of the string
	while(str_i[0] != 0) {
		// increment current position by number of bytes in Unicode character
		str_i += utf8proc_iterate(str_i, -1, &c);
		// write the Unicode character to the buffer
		lower += utf8proc_encode_char(utf8proc_tolower(c), lower);
	}
	lower[0] = 0;

	return (char *)(lower - lower_len);
}

char *str_toupper
(
	const char *str
) {
	// hold current Unicode character
	utf8proc_int32_t c;
	const utf8proc_uint8_t *str_i = (const utf8proc_uint8_t *)str;
	size_t upper_len = 0;
	utf8proc_uint8_t buffer[4];
	while(str_i[0] != 0) {
		str_i     += utf8proc_iterate(str_i, -1, &c);
		if(c == -1) {
			// Unicode character is not valid
			return NULL;
		}
		upper_len += utf8proc_encode_char(utf8proc_toupper(c), buffer);
	}
	
	utf8proc_uint8_t *upper = rm_malloc(upper_len + 1);

	str_i = (const utf8proc_uint8_t *)str;
	// while we didn't get to the end of the strings
	while(str_i[0] != 0) {
		// increment current position by number of bytes in Unicode character
		str_i += utf8proc_iterate(str_i, -1, &c);
		// write the Unicode character to the buffer
		upper += utf8proc_encode_char(utf8proc_toupper(c), upper);
	}
	upper[0] = 0;

	return (char *)(upper - upper_len);
}

// Utility function to increase the size of a buffer.
void str_ExtendBuffer
(
	char **buf,           // buffer to populate
	size_t *bufferLen,    // size of buffer
	size_t extensionLen   // number of bytes to add
) {
	*bufferLen += extensionLen;
	*buf = rm_realloc(*buf, sizeof(char) * *bufferLen);
}

// Utility function to check if the string matches
// the regex pattern.
bool str_MatchRegex
(
	const char* regex,   // regex pattern to match with
	const char* str      // string to match
) {
	const int len = strlen(str);
	regex_t *reg;
	OnigErrorInfo einfo;
	OnigRegion *region = onig_region_new();

	bool match = true;

	int rv = onig_new(&reg, (const UChar *)regex,
		(const UChar *)(regex + strlen(regex)), ONIG_OPTION_DEFAULT,
		ONIG_ENCODING_UTF8, ONIG_SYNTAX_JAVA, &einfo);
	if(unlikely(rv != ONIG_NORMAL)) {
		ASSERT(rv == ONIG_NORMAL);
		onig_free(reg);
		onig_region_free(region, 1);
		return false;
	}

	rv = onig_search(reg, (const UChar*)str, (UChar* )(str + len),
					(const UChar* )str, (const UChar* )(str + len),
					region, ONIG_OPTION_NONE);
	if (rv < ONIG_MISMATCH) {
		ASSERT(rv >= ONIG_MISMATCH);
		onig_free(reg);
		onig_region_free(region, 1);
		return false;
	}

	if (rv == ONIG_MISMATCH || region->beg[0] != 0 || region->end[0] != len) {
		match = false;
	} 

	onig_free(reg);
	onig_region_free(region, 1);

	return match;
}

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
) {
	ASSERT (str       != NULL) ;
	ASSERT (truncated != NULL) ;

	if (len >= max_len) {
		asprintf (truncated, "%.*s...", max_len, str) ;
	} else {
		*truncated = strndup (str, len) ;
	}
}

