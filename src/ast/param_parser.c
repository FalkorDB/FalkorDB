/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "param_parser.h"

#include "../value.h"
#include "../datatypes/map.h"
#include "../datatypes/array.h"

#include <ctype.h>

// hash function
// turning string into hash
static uint64_t _hashFunction
(
	const void *key  // string key
) {
	return XXH64(key, strlen(key), 0);
}

// hash compare function
// checking if two keys are the same using the strcmp function
static int _hashCompare
(
	dict *d,
	const void *key1,
	const void *key2
) {
	const char *a = (const char*)key1;
	const char *b = (const char*)key2;
	return (strcmp(a, b) == 0);
}

static void _keyDestructor
(
	dict *d,
	void *key
) {
	char *_key = key;
	rm_free(_key);
}

static void _valDestructor
(
	dict *d,
	void *obj
) {
	SIValue *v = obj;
	SIValue_Free(*v);
}

static dictType dt = {
	.hashFunction           =  _hashFunction,
	.keyDup                 =  NULL,
	.valDup                 =  NULL,
	.keyCompare             =  _hashCompare,
	.keyDestructor          =  _keyDestructor,
	.valDestructor          =  _valDestructor,
	.expandAllowed          =  NULL,
	.dictEntryMetadataBytes =  NULL,
	.dictMetadataBytes      =  NULL,
	.afterReplaceEntry      =  NULL
};

// forward declaration
static bool parse_value(char **head, SIValue *v);

// try accept given string
static bool accept
(
	char **head,      // parser header
	const char *str,  // string to accept
	size_t n          // length of str
) {
	if (strncasecmp(*head, str, n) == 0) {
		*head += n;
		return true;
	}

	return false;
}

// move head beyond the last space character
static inline void skip_spaces
(
	char **head  // parser head to advance
) {
	char *_head = *head;

	// skip all spaces
	while(_head[0] == ' '  ||
		  _head[0] == '\n' ||
		  _head[0] == '\t' ||
		  _head[0] == '\r') {
		_head++;
	}

	*head = _head;
}

// consume a single token
static char consume_token
(
	char **head  // parser head
) {
	char token = **head;

	// advance head
	if (token != '\0') {
		(*head)++;
	}

	return token;
}

// consumes a single digit
static void consume_digits
(
	char **head  // parser head
) {
	char *_head = *head;

	while (isdigit(*_head)) {
		_head++;
	}

	*head = _head;
}

// consume a valid Cypher parameter name
// e.g. budget42, _x99, user_name
static bool consume_name
(
    char **head  // parser head
) {
    char *_head = *head;

    // first character: must be a letter or underscore
    if (!((*_head >= 'a' && *_head <= 'z') ||
          (*_head >= 'A' && *_head <= 'Z') ||
          (*_head == '_'))) {
        return false;
    }
    _head++;

    // subsequent characters: letters, digits, or underscore
    while (*_head != '\0' &&
           ((*_head >= 'a' && *_head <= 'z') ||
            (*_head >= 'A' && *_head <= 'Z') ||
            (*_head >= '0' && *_head <= '9') ||
            (*_head == '_'))) {
        _head++;
    }

    *head = _head;
    return true;
}

// parse parameter name
// e.g. budget = 4
static bool parse_param_name
(
	char **head,  // parser head
	char **param  // [output] param name
) {
	char *_head = *head;
	skip_spaces(&_head);

	*param = _head;

	if(!consume_name(&_head)) {
		return false;
	}

	size_t len = _head - *param;

	skip_spaces(&_head);

	if(consume_token(&_head) != '=') {
		return false;
	}

	(*param)[len] = '\0';

	skip_spaces(&_head);

	// update parser head
	*head = _head;

	return true;
}

// parse map key
// e.g. key:
static bool parse_key
(
	char **head,  // parser head
	char **key    // [output] map key
) {
	char *_head = *head;
	skip_spaces(&_head);

	bool backtick = false;
	*key = _head;

	// key might start with ` (backtick) in which case we consume all chars
	// up to the closing backtick
	if (_head[0] == '`') {
		backtick = true;
		consume_token(&_head);

		char t;
		while ((t = consume_token(&_head)) != '\0') {
			if (t == '`') {
				break;
			}
		}
		// reached end of string before finding a closing backtick
		if (t == '\0') {
			return false;
		}
	} else {
		if (!consume_name(&_head)) {
			return false;
		}
	}

	size_t len = _head - *key;

	// remove enclosing backticks
	if (backtick) {
		len -= 2;
		(*key)++;
	}

	skip_spaces(&_head);

	if (consume_token(&_head) != ':') {
		return false;
	}

	(*key)[len] = '\0';

	skip_spaces(&_head);

	// update parser head
	*head = _head;

	return true;
}

#define APPEND_CHAR(buf, len, cap, c)              \
    do {                                           \
        if ((len) + 1 >= (cap)) {                  \
            (cap) = ((cap) == 0) ? 64 : (cap) * 2; \
            (buf) = rm_realloc((buf), (cap));      \
        }                                          \
        (buf)[(len)++] = (c);                      \
    } while (0)

static bool parse_escaped_string
(
	char **head,  // parser head
	char **str,   // [out] parsed string
	size_t len,   // string length
	size_t cap,   // string cap
	char quote    // enclosing quotation mark
) {
	char t;
	bool escape = false;
	char *_head = *head;

	while ((t = consume_token(&_head)) != '\0') {
		if (!escape) {
            if (t == '\\') {
                escape = true;
                continue;
            }

            if (t == quote) break; // end of quoted string

			APPEND_CHAR(*str, len, cap, t);
            continue;
        }

		// handle escape
		switch(t) {
			case 'a':  APPEND_CHAR(*str, len, cap, '\a'); break;
			case 'b':  APPEND_CHAR(*str, len, cap, '\b'); break;
			case 'f':  APPEND_CHAR(*str, len, cap, '\f'); break;
			case 'n':  APPEND_CHAR(*str, len, cap, '\n'); break;
			case 'r':  APPEND_CHAR(*str, len, cap, '\r'); break;
			case 't':  APPEND_CHAR(*str, len, cap, '\t'); break;
			case 'v':  APPEND_CHAR(*str, len, cap, '\v'); break;
			case '\\': APPEND_CHAR(*str, len, cap, '\\'); break;
			case '\'': APPEND_CHAR(*str, len, cap, '\''); break;
			case '\"': APPEND_CHAR(*str, len, cap, '\"'); break;
			case '?':  APPEND_CHAR(*str, len, cap, '?');  break;
			default:
				// Unrecognized escape — keep as-is
				APPEND_CHAR(*str, len, cap, '\\');
				APPEND_CHAR(*str, len, cap, t);
				break;
		}

		escape = false;
	}

	// did we found matching closing quote
	if (t != quote) {
		rm_free(*str);
		return false;
	}

	// null terminate
	APPEND_CHAR(*str, len, cap, '\0');

	// update head
	*head = _head;

	return true;
}

// parse a quoted string
static bool parse_string
(
	char **head,  // parser header
	SIValue *v    // [out] parsed string
) {
	ASSERT(**head == '\'' || **head == '"');

	char t;  // read token
	char *_head = *head;
	char quote  = consume_token(&_head);
	char *str   = _head;
	
	while ((t = consume_token(&_head)) != '\0') {
		// escape
		if (t == '\\') {
			_head--;  // backtrack
			size_t len = _head - str;
			size_t cap = len + 16;
			char *buf  = rm_malloc(cap);

			memcpy(buf, str, len);

			if (parse_escaped_string(&_head, &buf, len, cap, quote)) {
				*v = SI_TransferStringVal(buf);
				*head = _head;
				return true;
			}

			return false;
		}

		// found closing quotes
		if (t == quote) {
			break;
		}
	}
	
	// did we found matching closing quote
	if (t != quote) {
		return false;
	}

	// set string
	size_t len = _head - str - 1; // do not count closing quote
	(str)[len] = '\0';

	*v = SI_ConstStringVal(str);

	// update head
	*head = _head;

	return true;
}

// return current charecter
static char peek
(
	char *head  // parser head
) {
	return head[0];
}

static bool parse_number
(
	char **head,  // parser head
	SIValue *d    // [output] number
) {
	char *_head = *head;
	char *s = _head;
	char t;

	// consume the whole number part
	//t = consume_token(&_head);
	t = peek(_head);

	ASSERT(t == '-' || t == '.' || isdigit(t));

	// bool decimal_point;

	accept(&_head, "-", 1);  // skip - if present
	bool decimal_point = accept(&_head, ".", 1);  // skip . if present

	// require at least one digist
	t = consume_token(&_head);
	if (!isdigit(t)) {
		return false;
	}

	// consume additional digits
	consume_digits(&_head);

	//--------------------------------------------------------------------------
	// decimal point
	//--------------------------------------------------------------------------

	if (accept(&_head, ".", 1)) {
		decimal_point = true;

		// consume first digit
		t = consume_token(&_head);

		// expecting at least a single digit
		if (!isdigit(t)) {
			return false;
		}

		// consume additional digits
		consume_digits(&_head);
	}

	//--------------------------------------------------------------------------
	// exponent
	//--------------------------------------------------------------------------

	t = peek(_head);
	bool exponent = (t == 'e' || t == 'E');

	if (exponent) {
		consume_token(&_head);   // skip 'e'
		accept(&_head, "+", 1);  // skip + if present
		accept(&_head, "-", 1);  // skip - if present

		// expecting at least a single digit
		t = consume_token(&_head);
		if (!isdigit(t)) {
			return false;
		}

		// consume additional digits
		consume_digits(&_head);
	}

	t = peek(_head);  // backup char

	size_t len = _head - s;
	_head[0] = '\0';

	if(decimal_point || exponent) {
		char *endptr;
		double v = strtod(s, &endptr);
		if (*endptr != '\0') {
			return false;
		}
		*d = SI_DoubleVal(v);
	} else {
		char *endptr;
		long long v = strtoll(s, &endptr, 10);
		if (*endptr != '\0') {
			return false;
		}
		*d = SI_LongVal(v);
	}

	// restore head token
	_head[0] = t;

	*head = _head;

	return true;
}

static bool parse_array
(
	char **head,  // parser head
	SIValue *v    // [output] array
) {
	char *_head = *head;

	char t = consume_token(&_head);
	ASSERT(t == '[');

	skip_spaces(&_head);

	t = peek(_head);

	// empty array
	if(t == ']') {
		consume_token(&_head);  // advance
		*head = _head;          // accept
		*v = SIArray_New(0);    // set output
		return true;
	}

	SIValue arr = SIArray_New(16);

	while (t != ']') {
		SIValue v;

		if (!parse_value(&_head, &v)) {
			SIArray_Free(arr);
			return false;
		}

		SIArray_AppendAsOwner(&arr, &v);

		skip_spaces(&_head);

		t = consume_token(&_head);
		if (t != ',') {
			break;
		}
	}

	if (t == ']') {
		*head = _head;
		*v = arr;

		return true;
	}

	SIArray_Free(arr);
	return false;
}

static bool parse_map
(
	char **head,  // parser head
	SIValue *v    // [output] map
) {
	char *_head = *head;

	char t = consume_token(&_head);
	ASSERT(t == '{');

	skip_spaces(&_head);

	t = peek(_head);

	// empty map
	if(t == '}') {
		consume_token(&_head);  // advance
		*head = _head;          // accept
		*v = Map_New(0);        // set output
		return true;
	}

	SIValue map = Map_New(16);

	while (t != '}') {
		char *k;
		SIValue key;
		SIValue val;

		if (!parse_key(&_head, &k)) {
			Map_Free(map);
			return false;
		}

		if (!parse_value(&_head, &val)) {
			Map_Free(map);
			return false;
		}

		key = SI_ConstStringVal(k);
		Map_AddNoClone(&map, key, val);

		skip_spaces(&_head);

		t = consume_token(&_head);
		if (t != ',') {
			break;
		}
	}

	if (t == '}') {
		*head = _head;
		*v = map;

		return true;
	}

	Map_Free(map);
	return false;
}

static bool parse_value
(
	char **head,  // parser head
	SIValue *v    // [output] parsed value
) {
	skip_spaces(head);

	char p = peek(*head);

	if (p == '\0') {
		return false;
	}

	else if (p == '"' || p == '\'') {
		// parse string
		if (!parse_string(head, v)) {
			return false;
		}

		return true;
	}

	else if (p == '-' || p == '.' || (p >= '0' && p <= '9')) {
		if (!parse_number(head, v)) {
			return false;	
		}

		return true;
	}

	else if (p == '[') {
		if (!parse_array(head, v)) {
			return false;
		}

		return true;
	}

	else if (p == '{') {
		if (!parse_map(head, v)) {
			return false;
		}

		return true;
	}

	else if (accept(head, "true", 4)) {
		*v = SI_BoolVal(true);
		return true;
	}

	else if (accept(head, "false", 5)) {
		*v = SI_BoolVal(false);
		return true;
	}
	
	else if (accept(head, "null", 4)) {
		*v = SI_NullVal();
		return true;
	}

	return false;
}

dict *ParamParser_Parse
(
	char **input  // input to parse
) {
	ASSERT(input  != NULL && *input != NULL);

	char *head = *input;  // parser head
	
	skip_spaces(&head);

	// expecting keyword: CYPHER
	if(!accept(&head, "CYPHER", 6)) {
		return NULL;
	}

	dict *params = HashTableCreate(&dt);

	while(*head != '\0') {
		char *param = NULL;
		if(!parse_param_name(&head, &param)) {
			break;
		}

		SIValue *v = rm_malloc(sizeof(SIValue));
		if(!parse_value(&head, v)) {
			rm_free(v);
			// todo: release individual values
			HashTableRelease(params);
			return NULL;
		}

		// repeat param name
		size_t l = strlen(param) * 2;
		char *duplicated_param_name = rm_malloc(l + 1);
		strcpy(duplicated_param_name, param);
		strcpy(duplicated_param_name + l/2, param);
		duplicated_param_name[l] = '\0';
		HashTableAdd(params, duplicated_param_name, v);
	}

	*input = head;
	return params;
}

