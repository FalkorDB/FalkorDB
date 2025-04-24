# 🗄️ json.h

[![Actions Status](https://github.com/sheredom/json.h/workflows/CMake/badge.svg)](https://github.com/sheredom/json.h/actions)
[![Build status](https://ci.appveyor.com/api/projects/status/piell6hcrlwrcxp9?svg=true)](https://ci.appveyor.com/project/sheredom/json-h)
[![Sponsor](https://img.shields.io/badge/💜-sponsor-blueviolet)](https://github.com/sponsors/sheredom)

A simple single header solution to parsing JSON in C and C++.

JSON is parsed into a read-only, single allocation buffer.

The current supported compilers are gcc, clang and msvc.

The current supported platforms are Windows, mac OS and Linux.

## Usage

Just `#include "json.h"` in your code!

### json_parse

Parse a json string into a DOM.

```c
struct json_value_s *json_parse(
    const void *src,
    size_t src_size);
```

- `src` - a utf-8 json string to parse.
- `src_size` - the size of `src` in bytes.

Returns a `struct json_value_s*` pointing the root of the json DOM.

### struct json_value_s

The main struct for interacting with a parsed JSON Document Object Model (DOM)
is the `struct json_value_s`.

```c
struct json_value_s {
  void *payload;
  size_t type;
};
```

- `payload` - a pointer to the contents of the value.
- `type` - the type of struct `payload` points to, one of `json_type_e`. Note:
  if type is `json_type_true`, `json_type_false`, or `json_type_null`, payload
  will be NULL.

### json_parse_ex

Extended parse a json string into a DOM.

```c
struct json_value_s *json_parse_ex(
    const void *src,
    size_t src_size,
    size_t flags_bitset,
    void*(*alloc_func_ptr)(void *, size_t),
    void *user_data,
    struct json_parse_result_s *result);
```

- `src` - a utf-8 json string to parse.
- `src_size` - the size of `src` in bytes.
- `flags_bitset` - extra parsing flags, a bitset of flags specified in
  `enum json_parse_flags_e`.
- `alloc_func_ptr` - a callback function to use for doing the single allocation.
  If NULL, `malloc()` is used.
- `user_data` - user data to be passed as the first argument to
  `alloc_func_ptr`.
- `result` - the result of the parsing. If a parsing error occurred this will
  contain what type of error, and where in the source it occurred. Can be NULL.

Returns a `struct json_value_s*` pointing the root of the json DOM.

### enum json_parse_flags_e

The extra parsing flags that can be specified to `json_parse_ex()` are as
follows:

```c
enum json_parse_flags_e {
  json_parse_flags_default = 0,
  json_parse_flags_allow_trailing_comma = 0x1,
  json_parse_flags_allow_unquoted_keys = 0x2,
  json_parse_flags_allow_global_object = 0x4,
  json_parse_flags_allow_equals_in_object = 0x8,
  json_parse_flags_allow_no_commas = 0x10,
  json_parse_flags_allow_c_style_comments = 0x20,
  json_parse_flags_deprecated = 0x40,
  json_parse_flags_allow_location_information = 0x80,
  json_parse_flags_allow_single_quoted_strings = 0x100,
  json_parse_flags_allow_hexadecimal_numbers = 0x200,
  json_parse_flags_allow_leading_plus_sign = 0x400,
  json_parse_flags_allow_leading_or_trailing_decimal_point = 0x800,
  json_parse_flags_allow_inf_and_nan = 0x1000,
  json_parse_flags_allow_multi_line_strings = 0x2000,
  json_parse_flags_allow_simplified_json =
      (json_parse_flags_allow_trailing_comma |
       json_parse_flags_allow_unquoted_keys |
       json_parse_flags_allow_global_object |
       json_parse_flags_allow_equals_in_object |
       json_parse_flags_allow_no_commas),
  json_parse_flags_allow_json5 =
      (json_parse_flags_allow_trailing_comma |
       json_parse_flags_allow_unquoted_keys |
       json_parse_flags_allow_c_style_comments |
       json_parse_flags_allow_single_quoted_strings |
       json_parse_flags_allow_hexadecimal_numbers |
       json_parse_flags_allow_leading_plus_sign |
       json_parse_flags_allow_leading_or_trailing_decimal_point |
       json_parse_flags_allow_inf_and_nan |
       json_parse_flags_allow_multi_line_strings)
};
```

- `json_parse_flags_default` - the default, no special behaviour is enabled.
- `json_parse_flags_allow_trailing_comma` - allow trailing commas in objects and
  arrays. For example, both `[true,]` and `{"a" : null,}` would be allowed with
  this option on.
- `json_parse_flags_allow_unquoted_keys` - allow unquoted keys for objects. For
  example, `{a : null}` would be allowed with this option on.
- `json_parse_flags_allow_global_object` - allow a global unbracketed object. For
  example, `a : null, b : true, c : {}` would be allowed with this option on.
- `json_parse_flags_allow_equals_in_object` - allow objects to use '=' as well as
  ':' between key/value pairs. For example, `{"a" = null, "b" : true}` would be
  allowed with this option on.
- `json_parse_flags_allow_no_commas` - allow that objects don't have to have
  comma separators between key/value pairs. For example,
  `{"a" : null "b" : true}` would be allowed with this option on.
- `json_parse_flags_allow_c_style_comments` - allow c-style comments (`//` or
  `/* */`) to be ignored in the input JSON file.
- `json_parse_flags_deprecated` - a deprecated option.
- `json_parse_flags_allow_location_information` - allow location information to
  be tracked for where values are in the input JSON. Useful for alerting users to
  errors with precise location information pertaining to the original source.
  When this option is enabled, all `json_value_s*`'s can be casted to
  `json_value_ex_s*`, and the `json_string_s*` of `json_object_element_s*`'s
  name member can be casted to `json_string_ex_s*` to retrieve specific
  locations on all the values and keys. Note this option will increase the
  memory budget required for the DOM used to record the JSON.
- `json_parse_flags_allow_single_quoted_strings` - allows strings to be in
  `'single quotes'`.
- `json_parse_flags_allow_hexadecimal_numbers` - allows hexadecimal numbers to
  be used `0x42`.
- `json_parse_flags_allow_leading_plus_sign` - allows a leading '+' sign on
  numbers `+42`.
- `json_parse_flags_allow_leading_or_trailing_decimal_point` - allows decimal
  points to be lead or trailed by 0 digits `.42` or `42.`.
- `json_parse_flags_allow_inf_and_nan` - allows using infinity and NaN
  identifiers `Infinity` or `NaN`.
- `json_parse_flags_allow_multi_line_strings` - allows strings to span multiple
  lines.
- `json_parse_flags_allow_simplified_json` - allow simplified JSON to be parsed.
  Simplified JSON is an enabling of a set of other parsing options.
  [See the Bitsquid blog introducing this here.](http://bitsquid.blogspot.com/2009/10/simplified-json-notation.html)
- `json_parse_flags_allow_json5` - allow JSON5 to be parsed. JSON5 is an
  enabling of a set of other parsing options.
  [See the website defining this extension here.](https://json5.org)

## Examples

### Parsing with `json_parse`

Lets say we had the JSON string  *'{"a" : true, "b" : [false, null, "foo"]}'*.
To get to each part of the parsed JSON we'd do:

```c
const char json[] = "{\"a\" : true, \"b\" : [false, null, \"foo\"]}";
struct json_value_s* root = json_parse(json, strlen(json));
assert(root->type == json_type_object);

struct json_object_s* object = (struct json_object_s*)root->payload;
assert(object->length == 2);

struct json_object_element_s* a = object->start;

struct json_string_s* a_name = a->name;
assert(0 == strcmp(a_name->string, "a"));
assert(a_name->string_size == strlen("a"));

struct json_value_s* a_value = a->value;
assert(a_value->type == json_type_true);
assert(a_value->payload == NULL);

struct json_object_element_s* b = a->next;
assert(b->next == NULL);

struct json_string_s* b_name = b->name;
assert(0 == strcmp(b_name->string, "b"));
assert(b_name->string_size == strlen("b"));

struct json_value_s* b_value = b->value;
assert(b_value->type == json_type_array);

struct json_array_s* array = (struct json_array_s*)b_value->payload;
assert(array->length == 3);

struct json_array_element_s* b_1st = array->start;

struct json_value_s* b_1st_value = b_1st->value;
assert(b_1st_value->type == json_type_false);
assert(b_1st_value->payload == NULL);

struct json_array_element_s* b_2nd = b_1st->next;

struct json_value_s* b_2nd_value = b_2nd->value;
assert(b_2nd_value->type == json_type_null);
assert(b_2nd_value->payload == NULL);

struct json_array_element_s* b_3rd = b_2nd->next;
assert(b_3rd->next == NULL);

struct json_value_s* b_3rd_value = b_3rd->value;
assert(b_3rd_value->type == json_type_string);

struct json_string_s* string = (struct json_string_s*)b_3rd_value->payload;
assert(0 == strcmp(string->string, "foo"));
assert(string->string_size == strlen("foo"));

/* Don't forget to free the one allocation! */
free(root);
```

### Iterator Helpers

There are some functions that serve no purpose other than to make it nicer to
iterate through the produced JSON DOM:

- `json_value_as_string` - returns a value as a string, or null if it wasn't a
  string.
- `json_value_as_number` - returns a value as a number, or null if it wasn't a
  number.
- `json_value_as_object` - returns a value as an object, or null if it wasn't an
  object.
- `json_value_as_array` - returns a value as an array, or null if it wasn't an
  array.
- `json_value_is_true` - returns non-zero is a value was true, zero otherwise.
- `json_value_is_false` - returns non-zero is a value was false, zero otherwise.
- `json_value_is_null` - returns non-zero is a value was null, zero otherwise.

Lets look at the same example from above but using these helper iterators
instead:

```c
const char json[] = "{\"a\" : true, \"b\" : [false, null, \"foo\"]}";
struct json_value_s* root = json_parse(json, strlen(json));

struct json_object_s* object = json_value_as_object(root);
assert(object != NULL);
assert(object->length == 2);

struct json_object_element_s* a = object->start;

struct json_string_s* a_name = a->name;
assert(0 == strcmp(a_name->string, "a"));
assert(a_name->string_size == strlen("a"));

struct json_value_s* a_value = a->value;
assert(json_value_is_true(a_value));

struct json_object_element_s* b = a->next;
assert(b->next == NULL);

struct json_string_s* b_name = b->name;
assert(0 == strcmp(b_name->string, "b"));
assert(b_name->string_size == strlen("b"));

struct json_array_s* array = json_value_as_array(b->value);
assert(array->length == 3);

struct json_array_element_s* b_1st = array->start;

struct json_value_s* b_1st_value = b_1st->value;
assert(json_value_is_false(b_1st_value));

struct json_array_element_s* b_2nd = b_1st->next;

struct json_value_s* b_2nd_value = b_2nd->value;
assert(json_value_is_null(b_2nd_value));

struct json_array_element_s* b_3rd = b_2nd->next;
assert(b_3rd->next == NULL);

struct json_string_s* string = json_value_as_string(b_3rd->value);
assert(string != NULL);
assert(0 == strcmp(string->string, "foo"));
assert(string->string_size == strlen("foo"));

/* Don't forget to free the one allocation! */
free(root);
```

As you can see it makes iterating through the DOM a little more pleasant.

### Extracting a Value from a DOM

If you want to extract a value from a DOM into a new allocation then
`json_extract_value` and `json_extract_value_ex` are you friends. These
functions let you take any value and its subtree from a DOM and clone it
into a new allocation - either a single `malloc` or a user-provided
allocation region.

```c
const char json[] = "{\"foo\" : { \"bar\" : [123, false, null, true], \"haz\" : \"haha\" }}";
struct json_value_s* root = json_parse(json, strlen(json));
assert(root);

struct json_value_s* foo = json_value_as_object(root)->start->value;
assert(foo);

struct json_value_s* extracted = json_extract_value(foo);

/* We can free root now because we've got a new allocation for extracted! */
free(root);

assert(json_value_as_object(extracted));

/* Don't forget to free the one allocation! */
free(extracted);
```

## Design

The json_parse function calls malloc once, and then slices up this single
allocation to support all the weird and wonderful JSON structures you can
imagine!

The structure of the data is always the JSON structs first (which encode the
structure of the original JSON), followed by the data.

## Todo

- Add debug output to specify why the printer failed (as suggested by
  [@hugin84](https://twitter.com/hugin84) in
  https://twitter.com/hugin84/status/668506811595677696).

## License

This is free and unencumbered software released into the public domain.

Anyone is free to copy, modify, publish, use, compile, sell, or
distribute this software, either in source code form or as a compiled
binary, for any purpose, commercial or non-commercial, and by any
means.

In jurisdictions that recognize copyright laws, the author or authors
of this software dedicate any and all copyright interest in the
software to the public domain. We make this dedication for the benefit
of the public at large and to the detriment of our heirs and
successors. We intend this dedication to be an overt act of
relinquishment in perpetuity of all present and future rights to this
software under copyright law.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.

For more information, please refer to <http://unlicense.org/>
