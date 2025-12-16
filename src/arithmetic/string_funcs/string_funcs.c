/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "string_funcs.h"
#include "../func_desc.h"
#include "../../util/arr.h"
#include "../../util/uuid.h"
#include "utf8proc/utf8proc.h"
#include "../../util/rmalloc.h"
#include "../../util/strutil.h"
#include "../../errors/errors.h"
#include "../../util/math_util.h"
#include "../../datatypes/array.h"
#include "../../util/json_encoder.h"
#include "../deps/oniguruma/src/oniguruma.h"

// toString supports only integer, float, string, boolean, point & temporals
#define STRINGABLE (SI_NUMERIC | T_POINT | T_STRING | T_BOOL | SI_TEMPORAL)

// returns a string containing the specified number of leftmost characters of
// the original string
SIValue AR_LEFT
(
	SIValue *argv,
	int argc,
	void *private_data
) {
	if(SIValue_IsNull(argv[0])) return SI_NullVal();
	
	int64_t newlen = -1;
	if(SI_TYPE(argv[1]) == T_INT64) {
		newlen = argv[1].longval;
	} 
	if(newlen < 0) {
		ErrorCtx_SetError(EMSG_MUST_BE_NON_NEGATIVE, "length");
		return SI_NullVal();
	}

	if(strlen(argv[0].stringval) <= newlen) {
		// No need to truncate this string based on the requested length
		return SI_DuplicateStringVal(argv[0].stringval);
	}

	// determine new string byte size
	utf8proc_int32_t c;
	int64_t newlen_bytes = 0;
	const char *str = argv[0].stringval;
	for (int i = 0; i < newlen; i++) {
		newlen_bytes += utf8proc_iterate((const utf8proc_uint8_t *)(str+newlen_bytes), -1, &c);
	}

	char *left_str = rm_malloc((newlen_bytes + 1) * sizeof(char));
 	strncpy(left_str, str, newlen_bytes * sizeof(char));
 	left_str[newlen_bytes] = '\0';

	return SI_TransferStringVal(left_str);
}

// returns the original string with leading whitespace removed.
SIValue AR_LTRIM
(
	SIValue *argv,
	int argc,
	void *private_data
) {
	if(SIValue_IsNull(argv[0])) return SI_NullVal();

	char *trimmed = argv[0].stringval;

	while(*trimmed == ' ') {
		trimmed ++;
	}

	return SI_DuplicateStringVal(trimmed);
}

// returns a string containing the specified number of rightmost characters of
// the original string
SIValue AR_RIGHT
(
	SIValue *argv,
	int argc,
	void *private_data
) {
	if(SIValue_IsNull(argv[0])) return SI_NullVal();
	
	int64_t newlen = -1;
	if(SI_TYPE(argv[1]) == T_INT64) {
		newlen = argv[1].longval;
	}
	if(newlen < 0) {
		ErrorCtx_SetError(EMSG_MUST_BE_NON_NEGATIVE, "length");
		return SI_NullVal();
	}

	const char *str = argv[0].stringval;
	int64_t start   = str_length(str) - newlen;

	if(start <= 0) {
		// No need to truncate this string based on the requested length
		return SI_DuplicateStringVal(str);
	}

	utf8proc_int32_t c;
	int64_t start_bytes = 0;
	for (int i = 0; i < start; i++) {
		start_bytes += utf8proc_iterate((const utf8proc_uint8_t *)(str+start_bytes), -1, &c);
	}

	return SI_DuplicateStringVal(str + start_bytes);
}

// returns the original string with trailing whitespace removed
SIValue AR_RTRIM
(
	SIValue *argv,
	int argc,
	void *private_data
) {
	if(SIValue_IsNull(argv[0])) return SI_NullVal();

	char *str = argv[0].stringval;

	size_t i = strlen(str);
	while(i > 0 && str[i - 1] == ' ') {
		i --;
	}

	char *trimmed = rm_malloc((i + 1) * sizeof(char));
	strncpy(trimmed, str, i);
	trimmed[i] = '\0';

	return SI_TransferStringVal(trimmed);
}

// incase the parameter type is 
// 1. string - returns a string in which the order of all characters in the
//             original string have been reversed
// 2. array  - returns an array in which the order of all elements in the
//             original array have been reversed
SIValue AR_REVERSE
(
	SIValue *argv,
	int argc,
	void *private_data
) {
	if(SIValue_IsNull(argv[0])) return SI_NullVal();

	SIValue value = argv[0];
	if(SI_TYPE(value) & T_STRING) {
		// string reverse
		char *str = value.stringval;
		size_t str_len = strlen(str);
		char *reverse = rm_malloc((str_len + 1) * sizeof(char));

		char *reverse_i = reverse + str_len;
		utf8proc_int32_t c;
		utf8proc_ssize_t w;
		while(str[0] != 0) {
			w = utf8proc_iterate((const utf8proc_uint8_t *)str, -1, &c);
			str += w;
			reverse_i -= w;
			utf8proc_encode_char(c, (utf8proc_uint8_t *)reverse_i);
		}
		reverse[str_len] = '\0';
		return SI_TransferStringVal(reverse);
	} else {
		SIValue reverse = SI_CloneValue(value);
		array_reverse(reverse.array);
		return reverse;
	}
}

// returns a substring of the original string, beginning with a 0-based index start and length.
SIValue AR_SUBSTRING
(
	SIValue *argv,
	int argc,
	void *private_data
) {
	// argv[0] - original string
	// argv[1] - start position
	// argv[2] - length
	// if length is omitted, the function returns the substring starting at the
	// position given by start and extending to the end of original
	// if either start or length is null or a negative integer
	// an error is raised
	// if start is 0, the substring will start at the beginning of original
	// if length is 0, the empty string will be returned

	if(SIValue_IsNull(argv[0])) return SI_NullVal();

	int64_t length;
	const char   *original     = argv[0].stringval;
	const int64_t original_len = strlen(original);
	const int64_t start        = argv[1].longval;

	// make sure start doesn't overreach
	if(start < 0) {
		ErrorCtx_SetError(EMSG_MUST_BE_NON_NEGATIVE, "start");
		return SI_NullVal();
	}

	if(start >= original_len) {
		return SI_ConstStringVal("");
	}

	const int64_t suffix_len = original_len - start;
	if(argc == 2) {
		length = suffix_len;
	} else {
		length = argv[2].longval;
		if(length < 0) {
			ErrorCtx_SetError(EMSG_MUST_BE_NON_NEGATIVE, "length");
			return SI_ConstStringVal("");
		}

		// make sure length does not overreach
		length = MIN(length, suffix_len);
	}

	utf8proc_int32_t c;
	// find the start position to copy from
	const char *start_p = original;
	for (int i = 0; i < start; i++) {
		start_p += utf8proc_iterate((const utf8proc_uint8_t *)start_p, -1, &c);
	}

	// find the end position
	const char *end_p = start_p;
	for (int i = 0; i < length; i++) {
		end_p += utf8proc_iterate((const utf8proc_uint8_t *)end_p, -1, &c);
	}

	int len = end_p - start_p;
	char *substring = rm_malloc((len + 1) * sizeof(char));
	strncpy(substring, start_p, len);
	substring[len] = '\0';

	return SI_TransferStringVal(substring);
}

// given a list of strings and an optional delimiter
// return a concatenation of all the strings using the given delimiter
// string.join(list, delimiter = '') -> string
SIValue AR_JOIN
(
	SIValue *argv,
	int argc,
	void *private_data
) {
	SIValue list = argv[0];
	if(SI_TYPE(list) == T_NULL) {
		return SI_NullVal();
	}

	char *delimiter = "";
	if(argc == 2) {
		delimiter = argv[1].stringval;
	}

	int str_len = 0;                           // output string length
	uint32_t n = SIArray_Length(list);         // number of strings to join
	size_t delimeter_len = strlen(delimiter);  // length of the delimiter

	// empty list
	if(unlikely(n) == 0) {
		return SI_ConstStringVal("");
	}

	//--------------------------------------------------------------------------
	// compute required string length
	//--------------------------------------------------------------------------

	// delimeter length is added between each string
	if(n >= 2) {
		if(safe_mul(delimeter_len, n - 1, &str_len)) {
			ErrorCtx_SetError(EMSG_STRING_OVERFLOW);
			return SI_NullVal();
		}
	}

	// acount for each string length
	for(uint i = 0; i < n; i++) {
		SIValue str = SIArray_Get(list, i);
		if(!(SI_TYPE(str) & T_STRING)) {
			// all elements in the list should be string.
			Error_SITypeMismatch(str, T_STRING);
			return SI_NullVal();
		}

		if(safe_add(str_len, strlen(str.stringval), &str_len)) {
			ErrorCtx_SetError(EMSG_STRING_OVERFLOW);
			return SI_NullVal();
		}
	}

	// acoun for null terminator
	if(safe_add(str_len, 1, &str_len)) {
		ErrorCtx_SetError(EMSG_STRING_OVERFLOW);
		return SI_NullVal();
	}

	//--------------------------------------------------------------------------
	// join strings
	//--------------------------------------------------------------------------

	int l = 0;                       // current string length
	int cur_len = 0;                 // offset into output string
	char *res = rm_malloc(str_len);  // output string

	for(uint i = 0; i < n - 1; i++) {
		SIValue str = SIArray_Get(list, i);
		l = strlen(str.stringval);
		memcpy(res + cur_len, str.stringval, l);
		cur_len += l;
		memcpy(res + cur_len, delimiter, delimeter_len);
		cur_len += delimeter_len;
	}

	// write the last string
	SIValue str = SIArray_Get(list, n - 1);
	l = strlen(str.stringval);
	memcpy(res + cur_len, str.stringval, l);
	cur_len += l;

	// place null terminator
	res[cur_len] = '\0';

	return SI_TransferStringVal(res);
}

typedef struct {
	SIValue *list;
	const char *str;
} match_regex_scan_cb_args;

static int match_regex_scan_cb
(
	int n,
	int pos,
	OnigRegion *region,
	void *arg
) {
	match_regex_scan_cb_args *args = (match_regex_scan_cb_args *)arg;
	SIValue *list = args->list;
	const char *str = args->str;
	SIValue subList = SIArray_New(region->num_regs);
	assert(region->num_regs > 0);

	for (int i = 0; i < region->num_regs; i++) {
		int substr_len = region->end[i] - region->beg[i];
		char *substr = rm_strndup(str + region->beg[i], substr_len);
		SIArray_Append(&subList, SI_TransferStringVal(substr));
		rm_free(substr);
	}

	SIArray_Append(list, subList);
	SIValue_Free(subList);
	return 0;
}

// given a string and a regular expression,
// return an array of all matches and matching regions
// string.matchRegEx(str, regex) -> array(array(string))
SIValue AR_MATCHREGEX
(
	SIValue *argv,
	int argc,
	void *private_data
) {
	SIValue list = SIArray_New(0);
	if(SI_TYPE(argv[0]) == T_NULL || SI_TYPE(argv[1]) == T_NULL) {
		return list;
	}

	regex_t *regex;
	OnigErrorInfo einfo;
	OnigRegion *region    = onig_region_new();
	const char *str       = argv[0].stringval;
	const char *regex_str = argv[1].stringval;

	int rv = onig_new(&regex, (const UChar *)regex_str, 
		(const UChar *)(regex_str + strlen(regex_str)), ONIG_OPTION_DEFAULT,
		ONIG_ENCODING_UTF8, ONIG_SYNTAX_JAVA, &einfo);
	if(rv != ONIG_NORMAL) {
		char s[ONIG_MAX_ERROR_MESSAGE_LEN];
		onig_error_code_to_str((UChar* )s, rv, &einfo);
		ErrorCtx_SetError(EMSG_INVALID_REGEX, s);
		onig_free(regex);
		onig_region_free(region, 1);
		SIValue_Free(list);
		return SI_NullVal();
	}

	match_regex_scan_cb_args args = {
		.list = &list,
		.str = str
	};

	rv = onig_scan(regex, (const UChar *)str,
		(const UChar *)(str + strlen(str)), region, ONIG_OPTION_DEFAULT,
		match_regex_scan_cb, &args);
	if(rv < 0) {
		char s[ONIG_MAX_ERROR_MESSAGE_LEN];
		onig_error_code_to_str((OnigUChar* )s, rv);
		ErrorCtx_SetError(EMSG_INVALID_REGEX, s);
		onig_free(regex);
		onig_region_free(region, 1);
		SIValue_Free(list);
		return SI_NullVal();
	}

	onig_free(regex);
	onig_region_free(region, 1);

	return list;
}

typedef struct {
	char *res;
	uint32_t res_len;
	const char *str;
	uint32_t str_ind; // current index in str
	const char *replacement;
	uint32_t replacement_len;
} replace_regex_scan_cb_args;

static int replace_regex_scan_cb
(
	int n,
	int pos,
	OnigRegion *region,
	void *arg
) {
	replace_regex_scan_cb_args *args = (replace_regex_scan_cb_args *)arg;
	const char *str = args->str;
	assert(region->num_regs > 0);

	// reallocate new str size
	int str_copy_len = region->beg[0] - args->str_ind;
	int str_size = args->res_len + str_copy_len + args->replacement_len + 1;
	args->res = rm_realloc(args->res, str_size);

	// copy the string between the last match and the current match
	memcpy(args->res + args->res_len, str + args->str_ind, str_copy_len);
	args->str_ind = region->end[0];
	args->res_len += str_copy_len;

	// copy the replacement string
	memcpy(args->res + args->res_len, args->replacement, args->replacement_len);
	args->res_len += args->replacement_len;

	args->res[args->res_len] = '\0';

	return 0;
}

// given a string and a regular expression,
// return a string after replacing each regex match with a given replacement.
// string.replaceRegEx(str, regex, replacement) -> string
SIValue AR_REPLACEREGEX
(
	SIValue *argv,
	int argc,
	void *private_data
) {
	if(SI_TYPE(argv[0]) == T_NULL || SI_TYPE(argv[1]) == T_NULL) {
		return SI_NullVal();
	}

	char       *replacement = "";
	const char *str         = argv[0].stringval;
	const char *regex_str   = argv[1].stringval;

	if(argc == 3) {
		if(SI_TYPE(argv[2]) == T_NULL) {
			return SI_NullVal();
		}

		replacement = argv[2].stringval;
	}

	regex_t *regex;
	OnigErrorInfo einfo;
	OnigRegion *region = onig_region_new();

	int rv = onig_new(&regex, (const UChar *)regex_str, 
		(const UChar *)(regex_str + strlen(regex_str)), ONIG_OPTION_DEFAULT,
		ONIG_ENCODING_UTF8, ONIG_SYNTAX_JAVA, &einfo);
	if(rv != ONIG_NORMAL) {
		char s[ONIG_MAX_ERROR_MESSAGE_LEN];
		onig_error_code_to_str((UChar* )s, rv, &einfo);
		ErrorCtx_SetError(EMSG_INVALID_REGEX, s);
		onig_free(regex);
		onig_region_free(region, 1);
		return SI_NullVal();
	}

	replace_regex_scan_cb_args args = {
		.res = NULL,
		.res_len = 0,
		.str = str,
		.str_ind = 0,
		.replacement = replacement,
		.replacement_len = strlen(replacement)
	};

	rv = onig_scan(regex, (const UChar *)str,
		(const UChar *)(str + strlen(str)), region, ONIG_OPTION_DEFAULT,
		replace_regex_scan_cb, &args);
	if(rv < 0) {
		char s[ONIG_MAX_ERROR_MESSAGE_LEN];
		onig_error_code_to_str((OnigUChar* )s, rv);
		ErrorCtx_SetError(EMSG_INVALID_REGEX, s);
		onig_free(regex);
		onig_region_free(region, 1);
		return SI_NullVal();
	}

	onig_free(regex);
	onig_region_free(region, 1);

	// copy the remaining string
	int str_copy_len = strlen(str) - args.str_ind;
	args.res = rm_realloc(args.res, (args.res_len + str_copy_len + 1)*sizeof(char));
	memcpy(args.res + args.res_len, str + args.str_ind, str_copy_len*sizeof(char));
	args.res[args.res_len + str_copy_len] = '\0';

	return SI_TransferStringVal(args.res);
}

// returns the original string in lowercase.
SIValue AR_TOLOWER(SIValue *argv, int argc, void *private_data) {
	if(SIValue_IsNull(argv[0])) return SI_NullVal();
	char *original = argv[0].stringval;
	char *lower = str_tolower(original);
	if(lower == NULL) {
		ErrorCtx_SetError(EMSG_INVALID_UTF8);
		return SI_NullVal();
	}
	return SI_TransferStringVal(lower);
}

// returns the original string in uppercase.
SIValue AR_TOUPPER
(
	SIValue *argv,
	int argc,
	void *private_data
) {
	if(SIValue_IsNull(argv[0])) return SI_NullVal();
	char *original = argv[0].stringval;
	char *upper = str_toupper(original);
	if(upper == NULL) {
		ErrorCtx_SetError(EMSG_INVALID_UTF8);
		return SI_NullVal();
	}
	return SI_TransferStringVal(upper);
}

// converts an integer, float or boolean value to a string
SIValue AR_TOSTRING
(
	SIValue *argv,
	int argc,
	void *private_data
) {
	if(SI_TYPE(argv[0]) & STRINGABLE) {
		size_t len = SIValue_StringJoinLen(argv, 1, "");
		char *str = rm_malloc(len * sizeof(char));
		size_t bytesWritten = 0;
		SIValue_ToString(argv[0], &str, &len, &bytesWritten);
		return SI_TransferStringVal(str);
	} else {
		return SI_NullVal();
	}
}

// returns a JSON string representation of a map value
SIValue AR_TOJSON
(
	SIValue *argv,
	int argc,
	void *private_data
) {
	if(SIValue_IsNull(argv[0])) return SI_NullVal();
	char *buf = JsonEncoder_SIValue(argv[0]);
	return SI_TransferStringVal(buf);
}

// returns the original string with leading and trailing whitespace removed.
SIValue AR_TRIM
(
	SIValue *argv,
	int argc,
	void *private_data
) {
	if(SIValue_IsNull(argv[0])) {
		return SI_NullVal();
	}

	SIValue ltrim   = AR_LTRIM(argv, argc, NULL);
	SIValue trimmed = AR_RTRIM(&ltrim, 1, NULL);

	SIValue_Free(ltrim);
	return trimmed;
}

// returns true if argv[1] is a substring of argv[0].
SIValue AR_CONTAINS(SIValue *argv, int argc, void *private_data) {
	// No string contains null.
	if(SIValue_IsNull(argv[0]) || SIValue_IsNull(argv[1])) return SI_NullVal();

	const char *hay = argv[0].stringval;
	const char *needle = argv[1].stringval;

	// See if needle is in hay.
	bool found = (strstr(hay, needle) != NULL);
	return SI_BoolVal(found);
}

// returns true if argv[0] starts with argv[1].
SIValue AR_STARTSWITH
(
	SIValue *argv,
	int argc,
	void *private_data
) {
	// no string contains null
	if(SIValue_IsNull(argv[0]) || SIValue_IsNull(argv[1])) return SI_NullVal();

	const char *str = argv[0].stringval;
	const char *sub_string = argv[1].stringval;
	size_t str_len = strlen(str);
	size_t sub_string_len = strlen(sub_string);

	// if sub-string is longer then string return quickly
	if(sub_string_len > str_len) return SI_BoolVal(false);

	// compare character by character, see if there's a match
	for(int i = 0; i < sub_string_len; i++) {
		if(str[i] != sub_string[i]) return SI_BoolVal(false);
	}

	return SI_BoolVal(true);
}

// returns true if argv[0] ends with argv[1].
SIValue AR_ENDSWITH
(
	SIValue *argv,
	int argc,
	void *private_data
) {
	// no string contains null
	if(SIValue_IsNull(argv[0]) || SIValue_IsNull(argv[1])) return SI_NullVal();

	const char *str = argv[0].stringval;
	const char *sub_string = argv[1].stringval;
	size_t str_len = strlen(str);
	size_t sub_string_len = strlen(sub_string);

	// if sub-string is longer then string return quickly
	if(sub_string_len > str_len) return SI_BoolVal(false);

	// advance str to the "end"
	str += (str_len - sub_string_len);
	// compare character by character, see if there's a match
	for(int i = 0; i < sub_string_len; i++) {
		if(str[i] != sub_string[i]) return SI_BoolVal(false);
	}

	return SI_BoolVal(true);
}

// returns a string in which all occurrences of a specified string in the
// original string have been replaced by ANOTHER (specified) string
// for example:
// RETURN replace('Well I wish I was in the land of cotton', 'cotton', 'the free')
// the result is Well I wish I was in the land of the free
SIValue AR_REPLACE
(
	SIValue *argv,
	int argc,
	void *private_data
) {
	// no string contains null
	if(SIValue_IsNull(argv[0]) ||
	   SIValue_IsNull(argv[1]) ||
	   SIValue_IsNull(argv[2])) return SI_NullVal();

	// argv[0] is the original string to be manipulated
	// argv[1] is the search sub string to be replaced
	// argv[2] is the string to be replaced with
	const char *str            = argv[0].stringval;
	const char *old_string     = argv[1].stringval;
	const char *new_string     = argv[2].stringval;
	size_t      str_len        = strlen(str);
	size_t      old_string_len = strlen(old_string);
	size_t      new_string_len = strlen(new_string);

	const char *ptr  = str;
	const char **arr = array_new(const char *, 0);

	// if any parameter is not a valid utf8 string return the original string
	if(!str_utf8_validate(old_string) || !str_utf8_validate(new_string) ||
		!str_utf8_validate(str)) {
		return SI_DuplicateStringVal(str);
	}

	while(ptr <= str + str_len) {
		// find pointer to next substring
		ptr = strstr(ptr, old_string);

		// if no substring found, then break from the loop
		if(ptr == NULL) break;

		// store ptr for replace use
		array_append(arr, ptr);

		// increment our string pointer
		// in case search string is empty move one char
		ptr += old_string_len == 0 ? 1 : old_string_len;
	}

	int occurrences = array_len(arr);

	// if sub string not found return original string
	if(occurrences == 0) {
		array_free(arr);
		return SI_DuplicateStringVal(str);
	}

	// calculate new buffer size
	size_t buffer_size =
		strlen(str)                    +
		(occurrences * new_string_len) -
		(occurrences * old_string_len);

	// allocate buffer
	char *buffer = (char*) rm_malloc(sizeof(char) * buffer_size + 1);

	// set pointers to start point
	ptr = str;
	char *buffer_ptr = buffer;

	// iterate occurrences
	for (int i = 0; i < occurrences; i++) {
		// calculate len to copy from last to current occurance
		int len = arr[i] - ptr;

		// copy part from original string
		strncpy(buffer_ptr, ptr, len);

		// move forward to copy more data to the buffer
		buffer_ptr += len;

		// copy new string instead of old string
		strcpy(buffer_ptr, new_string);

		// move forward to copy more data to the buffer
		buffer_ptr += new_string_len;

		// move forwart to copy more data from the original string
		ptr = arr[i] + old_string_len;
	}

	// copy rest of the string from the original string
	strcpy(buffer_ptr, ptr);

	buffer[buffer_size] = '\0';

	array_free(arr);

	return SI_TransferStringVal(buffer);
}

// returns a list of strings resulting from the splitting of the original string
// around matches of the given delimiter
SIValue AR_SPLIT
(
	SIValue *argv,
	int argc,
	void *private_data
) {
	if(SIValue_IsNull(argv[0]) || SIValue_IsNull(argv[1])) {
		return SI_NullVal();
	}

	char       *str           = argv[0].stringval;
	const char *delimiter     = argv[1].stringval;
	size_t      str_len       = strlen(str);
	size_t      delimiter_len = strlen(delimiter);
	SIValue     tokens        = SIArray_New(1);

	if(delimiter_len == 0) {
		if(strlen(str) == 0) {
			SIArray_Append(&tokens, SI_ConstStringVal(""));
		} else {
			utf8proc_int32_t c;
			utf8proc_uint8_t token[5];
			const utf8proc_uint8_t *str_i = (const utf8proc_uint8_t *)str;
			while(str_i[0] != 0) {
				str_i += utf8proc_iterate(str_i, -1, &c);
				int i  = utf8proc_encode_char(c, token);
				token[i] = '\0';
				SIArray_Append(&tokens, SI_ConstStringVal((const char *)token));
			}
		}
	} else if(str_len == 0) {
		SIArray_Append(&tokens, SI_ConstStringVal(""));
	} else {
		size_t rest_len   = str_len;
		const char *start = str;
		bool delimiter_found = false;
		while(rest_len >= delimiter_len) {
			// find bytes length from start to delimiter
			int len = 0;
			delimiter_found = false;
			while(len <= rest_len - delimiter_len) {
				if(strncmp(start + len, delimiter, delimiter_len) == 0) {
					delimiter_found = true;
					break;
				}
				len++;
			}
			if(!delimiter_found) {
				break;
			}
			SIValue si_token = SI_TransferStringVal(rm_strndup(start, len));
			SIArray_Append(&tokens, si_token);
			SIValue_Free(si_token);
			start += len + delimiter_len;
			rest_len -= len + delimiter_len;
		}
		if(rest_len > 0 || delimiter_found) {
			SIValue si_token = SI_ConstStringVal(start);
			SIArray_Append(&tokens, si_token);
		}
	}

	return tokens;
}

// create an intern string
SIValue AR_INTERN
(
	SIValue *argv,      // arguments
	int argc,           // number of arguments
	void *private_data  // private data
) {
	// in case of a NULL value, return NULL
	if(SIValue_IsNull(argv[0])) {
		return SI_NullVal();
	}

	// create and return an intern string
	char *str = argv[0].stringval;
	return SI_InternStringVal(str);
}

//------------------------------------------------------------------------------
// Scalar functions
//------------------------------------------------------------------------------

SIValue AR_RANDOMUUID
(
	SIValue *argv,
	int argc,
	void *private_data
) {
	char *uuid = UUID_New();
	return SI_TransferStringVal(uuid);
}

void Register_StringFuncs() {
	SIType *types;
	SIType ret_type;
	AR_FuncDesc *func_desc;

	types = array_new(SIType, 2);
	array_append(types, (T_STRING | T_NULL));
	array_append(types, T_INT64 | T_NULL);
	ret_type = T_STRING | T_NULL;
	func_desc = AR_FuncDescNew("left", AR_LEFT, 2, 2, types, ret_type, false,
			true, true);
	AR_RegFunc(func_desc);

	types = array_new(SIType, 1);
	array_append(types, (T_STRING | T_NULL));
	ret_type = T_STRING | T_NULL;
	func_desc = AR_FuncDescNew("ltrim", AR_LTRIM, 1, 1, types, ret_type, false,
			true, true);
	AR_RegFunc(func_desc);

	types = array_new(SIType, 2);
	array_append(types, (T_STRING | T_NULL));
	array_append(types, T_INT64 | T_NULL);
	ret_type = T_STRING | T_NULL;
	func_desc = AR_FuncDescNew("right", AR_RIGHT, 2, 2, types, ret_type, false,
			true, true);
	AR_RegFunc(func_desc);

	types = array_new(SIType, 1);
	array_append(types, (T_STRING | T_NULL));
	ret_type = T_STRING | T_NULL;
	func_desc = AR_FuncDescNew("rtrim", AR_RTRIM, 1, 1, types, ret_type, false,
			true, true);
	AR_RegFunc(func_desc);

	types = array_new(SIType, 1);
	array_append(types, (T_STRING | T_ARRAY | T_NULL));
	ret_type = T_STRING | T_ARRAY | T_NULL;
	func_desc = AR_FuncDescNew("reverse", AR_REVERSE, 1, 1, types, ret_type,
			false, true, true);
	AR_RegFunc(func_desc);

	types = array_new(SIType, 3);
	array_append(types, (T_STRING | T_NULL));
	array_append(types, T_INT64);
	array_append(types, T_INT64);
	ret_type = T_STRING | T_NULL;
	func_desc = AR_FuncDescNew("substring", AR_SUBSTRING, 2, 3, types, ret_type,
			false, true, true);
	AR_RegFunc(func_desc);

	types = array_new(SIType, 2);
	array_append(types, (T_ARRAY | T_NULL));
	array_append(types, T_STRING);
	ret_type = T_STRING | T_NULL;
	func_desc = AR_FuncDescNew("string.join", AR_JOIN, 1, 2, types, ret_type,
			false, true, true);
	AR_RegFunc(func_desc);

	types = array_new(SIType, 2);
	array_append(types, (T_STRING | T_NULL));
	array_append(types, (T_STRING | T_NULL));
	ret_type = T_ARRAY | T_NULL;
	func_desc = AR_FuncDescNew("string.matchRegEx", AR_MATCHREGEX, 2, 2, types,
			ret_type, false, true, true);
	AR_RegFunc(func_desc);

	types = array_new(SIType, 3);
	array_append(types, (T_STRING | T_NULL));
	array_append(types, (T_STRING | T_NULL));
	array_append(types, (T_STRING | T_NULL));
	ret_type = T_STRING | T_NULL;
	func_desc = AR_FuncDescNew("string.replaceRegEx", AR_REPLACEREGEX, 2, 3,
			types, ret_type, false, true, true);
	AR_RegFunc(func_desc);

	types = array_new(SIType, 1);
	array_append(types, (T_STRING | T_NULL));
	ret_type = T_STRING | T_NULL;
	func_desc = AR_FuncDescNew("tolower", AR_TOLOWER, 1, 1, types, ret_type,
			false, true, true);
	AR_RegFunc(func_desc);

	types = array_new(SIType, 1);
	array_append(types, (T_STRING | T_NULL));
	ret_type = T_STRING | T_NULL;
	func_desc = AR_FuncDescNew("toupper", AR_TOUPPER, 1, 1, types, ret_type,
			false, true, true);
	AR_RegFunc(func_desc);

	types = array_new(SIType, 1);
	array_append(types, STRINGABLE | T_NULL);
	ret_type = T_STRING | T_NULL;
	func_desc = AR_FuncDescNew("tostring", AR_TOSTRING, 1, 1, types, ret_type,
			false, true, true);
	AR_RegFunc(func_desc);

	types = array_new(SIType, 1);
	array_append(types, SI_ALL);
	ret_type = T_STRING | T_NULL;
	func_desc = AR_FuncDescNew("tostringornull", AR_TOSTRING, 1, 1, types,
			ret_type, false, true, true);
	AR_RegFunc(func_desc);

	types = array_new(SIType, 1);
	array_append(types, SI_ALL);
	ret_type = T_STRING | T_NULL;
	func_desc = AR_FuncDescNew("tojson", AR_TOJSON, 1, 1, types, ret_type,
			false, true, true);
	AR_RegFunc(func_desc);

	types = array_new(SIType, 1);
	array_append(types, (T_STRING | T_NULL));
	ret_type = T_STRING | T_NULL;
	func_desc = AR_FuncDescNew("trim", AR_TRIM, 1, 1, types, ret_type, false,
			true, true);
	AR_RegFunc(func_desc);

	types = array_new(SIType, 2);
	array_append(types, (T_STRING | T_NULL));
	array_append(types, (T_STRING | T_NULL));
	ret_type = T_BOOL | T_NULL;
	func_desc = AR_FuncDescNew("contains", AR_CONTAINS, 2, 2, types, ret_type,
			true, true, true);
	AR_RegFunc(func_desc);

	types = array_new(SIType, 2);
	array_append(types, (T_STRING | T_NULL));
	array_append(types, (T_STRING | T_NULL));
	ret_type = T_BOOL | T_NULL;
	func_desc = AR_FuncDescNew("starts with", AR_STARTSWITH, 2, 2, types,
			ret_type, true, true, true);
	AR_RegFunc(func_desc);

	types = array_new(SIType, 2);
	array_append(types, (T_STRING | T_NULL));
	array_append(types, (T_STRING | T_NULL));
	ret_type = T_BOOL | T_NULL;
	func_desc = AR_FuncDescNew("ends with", AR_ENDSWITH, 2, 2, types, ret_type,
			true, true, true);
	AR_RegFunc(func_desc);

	types = array_new(SIType, 0);
	ret_type = T_STRING;
	func_desc = AR_FuncDescNew("randomuuid", AR_RANDOMUUID, 0, 0, types,
			ret_type, false, false, false);
	AR_RegFunc(func_desc);

	types = array_new(SIType, 3);
	array_append(types, (T_STRING | T_NULL));
	array_append(types, (T_STRING | T_NULL));
	array_append(types, (T_STRING | T_NULL));
	ret_type = T_STRING | T_NULL;
	func_desc = AR_FuncDescNew("replace", AR_REPLACE, 3, 3, types, ret_type,
			false, true, true);
	AR_RegFunc(func_desc);

	types = array_new(SIType, 2);
	array_append(types, (T_STRING | T_NULL));
	array_append(types, (T_STRING | T_NULL));
	ret_type = T_ARRAY | T_NULL;
	func_desc = AR_FuncDescNew("split", AR_SPLIT, 2, 2, types, ret_type, false,
			true, true);
	AR_RegFunc(func_desc);

	types = array_new(SIType, 1);
	array_append(types, (T_STRING | T_NULL));
	ret_type = T_INTERN_STRING | T_NULL;
	func_desc = AR_FuncDescNew("intern", AR_INTERN, 1, 1, types, ret_type,
			false, false, true);
	AR_RegFunc(func_desc);
}

