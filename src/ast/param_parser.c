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
	while(_head[0] == ' ') {
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

	*key = _head;

	if(!consume_name(&_head)) {
		return false;
	}

	size_t len = _head - *key;

	skip_spaces(&_head);

	if(consume_token(&_head) != ':') {
		return false;
	}

	(*key)[len] = '\0';

	skip_spaces(&_head);

	// update parser head
	*head = _head;

	return true;
}

// parse a quoted string
static bool parse_string
(
	char **head,  // parser header
	char **str    // [out] parsed string
) {
	ASSERT(**head == '\'' || **head == '"');

	char t;  // read token
	char *_head = *head;
	char quote = consume_token(&_head);
	char *s = _head;
	
	while ((t = consume_token(&_head)) != '\0') {
		// escape
		if (t == '\\') {
			if (consume_token(&_head) == '\0') {
				return false;
			}
			// skip escaped character
			continue;
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
	size_t len = _head - s - 1; // do not count closing quote
	*str = s;
	(*str)[len] = '\0';

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
	t = consume_token(&_head);
	ASSERT(t == '-' || isdigit(t));

	consume_digits(&_head);

	bool decimal_point = (peek(_head) == '.');
	if (decimal_point) {
		// skip over '.'
		consume_token(&_head);

		// expecting at least a single digit
		if (!isdigit(peek(_head))) {
			return false;
		}

		consume_digits(&_head);
	}

	t = peek(_head);

	size_t len = _head - s;
	_head[0] = '\0';

	if(decimal_point) {
		// todo: check for error
		char *endptr;
		double v = strtod(s, &endptr);
		if (*endptr != '\0') return false;
		*d = SI_DoubleVal(v);
	} else {
		char *endptr;
		long long v = strtoll(s, &endptr, 10);
		if (*endptr != '\0') return false;
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
		char *s = NULL;
		if (!parse_string(head, &s)) {
			return false;
		}

		*v = SI_ConstStringVal(s);
		return true;
	}

	else if (p == '-' || (p >= '0' && p <= '9')) {
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

	dict *params = HashTableCreate(&string_dt);

	while(*head != '\0') {
		char *param = NULL;
		if(!parse_param_name(&head, &param)) {
			break;
		}

		SIValue *v = rm_malloc(sizeof(SIValue));
		if(!parse_value(&head, v)) {
			rm_free(v);
			break;
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

