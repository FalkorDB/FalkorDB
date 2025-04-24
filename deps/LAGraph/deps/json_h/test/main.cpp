// This is free and unencumbered software released into the public domain.
//
// Anyone is free to copy, modify, publish, use, compile, sell, or
// distribute this software, either in source code form or as a compiled
// binary, for any purpose, commercial or non-commercial, and by any
// means.
//
// In jurisdictions that recognize copyright laws, the author or authors
// of this software dedicate any and all copyright interest in the
// software to the public domain. We make this dedication for the benefit
// of the public at large and to the detriment of our heirs and
// successors. We intend this dedication to be an overt act of
// relinquishment in perpetuity of all present and future rights to this
// software under copyright law.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
// IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
// OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
// OTHER DEALINGS IN THE SOFTWARE.
//
// For more information, please refer to <http://unlicense.org/>

#include "utest.h"

#include "json.h"

UTEST(object, empty) {
  const char payload[] = "{}";
  struct json_value_s *value = json_parse(payload, strlen(payload));
  struct json_object_s *object = 0;

  ASSERT_TRUE(value);
  ASSERT_TRUE(value->payload);
  ASSERT_EQ(json_type_object, value->type);

  object = (struct json_object_s *)value->payload;

  ASSERT_FALSE(object->start);
  ASSERT_EQ(0, object->length);

  free(value);
}

UTEST(object, string) {
  const char payload[] = "{\"foo\" : \"Heyo, gaia?\"}";
  struct json_value_s *value = json_parse(payload, strlen(payload));
  struct json_object_s *object = 0;
  struct json_value_s *value2 = 0;
  struct json_string_s *string = 0;

  ASSERT_TRUE(value);
  ASSERT_TRUE(value->payload);
  ASSERT_EQ(json_type_object, value->type);

  object = (struct json_object_s *)value->payload;

  ASSERT_TRUE(object->start);
  ASSERT_EQ(1, object->length);

  ASSERT_TRUE(object->start->name);
  ASSERT_TRUE(object->start->value);
  ASSERT_FALSE(object->start->next); // we have only one element

  ASSERT_TRUE(object->start->name->string);
  ASSERT_STREQ("foo", object->start->name->string);
  ASSERT_EQ(strlen("foo"), object->start->name->string_size);
  ASSERT_EQ(strlen(object->start->name->string),
            object->start->name->string_size);

  value2 = object->start->value;

  ASSERT_TRUE(value2->payload);
  ASSERT_EQ(json_type_string, value2->type);

  string = (struct json_string_s *)value2->payload;

  ASSERT_TRUE(string->string);
  ASSERT_STREQ("Heyo, gaia?", string->string);
  ASSERT_EQ(strlen("Heyo, gaia?"), string->string_size);
  ASSERT_EQ(strlen(string->string), string->string_size);

  free(value);
}

UTEST(object, number) {
  const char payload[] = "{\"foo\" : -0.123e-42}";
  struct json_value_s *value = json_parse(payload, strlen(payload));
  struct json_object_s *object = 0;
  struct json_value_s *value2 = 0;
  struct json_number_s *number = 0;

  ASSERT_TRUE(value);
  ASSERT_TRUE(value->payload);
  ASSERT_EQ(json_type_object, value->type);

  object = (struct json_object_s *)value->payload;

  ASSERT_TRUE(object->start);
  ASSERT_EQ(1, object->length);

  ASSERT_TRUE(object->start->name);
  ASSERT_TRUE(object->start->value);
  ASSERT_FALSE(object->start->next); // we have only one element

  ASSERT_TRUE(object->start->name->string);
  ASSERT_STREQ("foo", object->start->name->string);
  ASSERT_EQ(strlen("foo"), object->start->name->string_size);
  ASSERT_EQ(strlen(object->start->name->string),
            object->start->name->string_size);

  value2 = object->start->value;

  ASSERT_TRUE(value2->payload);
  ASSERT_EQ(json_type_number, value2->type);

  number = (struct json_number_s *)value2->payload;

  ASSERT_TRUE(number->number);
  ASSERT_STREQ("-0.123e-42", number->number);
  ASSERT_EQ(strlen("-0.123e-42"), number->number_size);
  ASSERT_EQ(strlen(number->number), number->number_size);

  free(value);
}

UTEST(object, object) {
  const char payload[] = "{\"foo\" : {}}";
  struct json_value_s *value = json_parse(payload, strlen(payload));
  struct json_object_s *object = 0;
  struct json_value_s *value2 = 0;
  struct json_object_s *object2 = 0;

  ASSERT_TRUE(value);
  ASSERT_TRUE(value->payload);
  ASSERT_EQ(json_type_object, value->type);

  object = (struct json_object_s *)value->payload;

  ASSERT_TRUE(object->start);
  ASSERT_EQ(1, object->length);

  ASSERT_TRUE(object->start->name);
  ASSERT_TRUE(object->start->value);
  ASSERT_FALSE(object->start->next); // we have only one element

  ASSERT_TRUE(object->start->name->string);
  ASSERT_STREQ("foo", object->start->name->string);
  ASSERT_EQ(strlen("foo"), object->start->name->string_size);
  ASSERT_EQ(strlen(object->start->name->string),
            object->start->name->string_size);

  value2 = object->start->value;

  ASSERT_TRUE(value2->payload);
  ASSERT_EQ(json_type_object, value2->type);

  object2 = (struct json_object_s *)value2->payload;

  ASSERT_FALSE(object2->start);
  ASSERT_EQ(0, object2->length);

  free(value);
}

UTEST(object, array) {
  const char payload[] = "{\"foo\" : []}";
  struct json_value_s *value = json_parse(payload, strlen(payload));
  struct json_object_s *object = 0;
  struct json_value_s *value2 = 0;
  struct json_array_s *array = 0;

  ASSERT_TRUE(value);
  ASSERT_TRUE(value->payload);
  ASSERT_EQ(json_type_object, value->type);

  object = (struct json_object_s *)value->payload;

  ASSERT_TRUE(object->start);
  ASSERT_EQ(1, object->length);

  ASSERT_TRUE(object->start->name);
  ASSERT_TRUE(object->start->value);
  ASSERT_FALSE(object->start->next); // we have only one element

  ASSERT_TRUE(object->start->name->string);
  ASSERT_STREQ("foo", object->start->name->string);
  ASSERT_EQ(strlen("foo"), object->start->name->string_size);
  ASSERT_EQ(strlen(object->start->name->string),
            object->start->name->string_size);

  value2 = object->start->value;

  ASSERT_TRUE(value2->payload);
  ASSERT_EQ(json_type_array, value2->type);

  array = (struct json_array_s *)value2->payload;

  ASSERT_FALSE(array->start);
  ASSERT_EQ(0, array->length);

  free(value);
}

UTEST(object, true) {
  const char payload[] = "{\"foo\" : true}";
  struct json_value_s *value = json_parse(payload, strlen(payload));
  struct json_object_s *object = 0;
  struct json_value_s *value2 = 0;

  ASSERT_TRUE(value);
  ASSERT_TRUE(value->payload);
  ASSERT_EQ(json_type_object, value->type);

  object = (struct json_object_s *)value->payload;

  ASSERT_TRUE(object->start);
  ASSERT_EQ(1, object->length);

  ASSERT_TRUE(object->start->name);
  ASSERT_TRUE(object->start->value);
  ASSERT_FALSE(object->start->next); // we have only one element

  ASSERT_TRUE(object->start->name->string);
  ASSERT_STREQ("foo", object->start->name->string);
  ASSERT_EQ(strlen("foo"), object->start->name->string_size);
  ASSERT_EQ(strlen(object->start->name->string),
            object->start->name->string_size);

  value2 = object->start->value;

  ASSERT_FALSE(value2->payload);
  ASSERT_EQ(json_type_true, value2->type);

  free(value);
}

UTEST(object, false) {
  const char payload[] = "{\"foo\" : false}";
  struct json_value_s *value = json_parse(payload, strlen(payload));
  struct json_object_s *object = 0;
  struct json_value_s *value2 = 0;

  ASSERT_TRUE(value);
  ASSERT_TRUE(value->payload);
  ASSERT_EQ(json_type_object, value->type);

  object = (struct json_object_s *)value->payload;

  ASSERT_TRUE(object->start);
  ASSERT_EQ(1, object->length);

  ASSERT_TRUE(object->start->name);
  ASSERT_TRUE(object->start->value);
  ASSERT_FALSE(object->start->next); // we have only one element

  ASSERT_TRUE(object->start->name->string);
  ASSERT_STREQ("foo", object->start->name->string);
  ASSERT_EQ(strlen("foo"), object->start->name->string_size);
  ASSERT_EQ(strlen(object->start->name->string),
            object->start->name->string_size);

  value2 = object->start->value;

  ASSERT_FALSE(value2->payload);
  ASSERT_EQ(json_type_false, value2->type);

  free(value);
}

UTEST(object, null) {
  const char payload[] = "{\"foo\" : null}";
  struct json_value_s *value = json_parse(payload, strlen(payload));
  struct json_object_s *object = 0;
  struct json_value_s *value2 = 0;

  ASSERT_TRUE(value);
  ASSERT_TRUE(value->payload);
  ASSERT_EQ(json_type_object, value->type);

  object = (struct json_object_s *)value->payload;

  ASSERT_TRUE(object->start);
  ASSERT_EQ(1, object->length);

  ASSERT_TRUE(object->start->name);
  ASSERT_TRUE(object->start->value);
  ASSERT_FALSE(object->start->next); // we have only one element

  ASSERT_TRUE(object->start->name->string);
  ASSERT_STREQ("foo", object->start->name->string);
  ASSERT_EQ(strlen("foo"), object->start->name->string_size);
  ASSERT_EQ(strlen(object->start->name->string),
            object->start->name->string_size);

  value2 = object->start->value;

  ASSERT_FALSE(value2->payload);
  ASSERT_EQ(json_type_null, value2->type);

  free(value);
}

UTEST(array, empty) {
  const char payload[] = "[]";
  struct json_value_s *value = json_parse(payload, strlen(payload));
  struct json_array_s *array = 0;

  ASSERT_TRUE(value);
  ASSERT_TRUE(value->payload);
  ASSERT_EQ(json_type_array, value->type);

  array = (struct json_array_s *)value->payload;

  ASSERT_FALSE(array->start);
  ASSERT_EQ(0, array->length);

  free(value);
}

UTEST(array, string) {
  const char payload[] = "[\"Heyo, gaia?\"]";
  struct json_value_s *value = json_parse(payload, strlen(payload));
  struct json_array_s *array = 0;
  struct json_value_s *value2 = 0;
  struct json_string_s *string = 0;

  ASSERT_TRUE(value);
  ASSERT_TRUE(value->payload);
  ASSERT_EQ(json_type_array, value->type);

  array = (struct json_array_s *)value->payload;

  ASSERT_TRUE(array->start);
  ASSERT_EQ(1, array->length);

  ASSERT_TRUE(array->start->value);
  ASSERT_FALSE(array->start->next); // we have only one element

  value2 = array->start->value;

  ASSERT_TRUE(value2->payload);
  ASSERT_EQ(json_type_string, value2->type);

  string = (struct json_string_s *)value2->payload;

  ASSERT_TRUE(string->string);
  ASSERT_STREQ("Heyo, gaia?", string->string);
  ASSERT_EQ(strlen("Heyo, gaia?"), string->string_size);
  ASSERT_EQ(strlen(string->string), string->string_size);

  free(value);
}

UTEST(array, number) {
  const char payload[] = "[-0.123e-42]";
  struct json_value_s *value = json_parse(payload, strlen(payload));
  struct json_array_s *array = 0;
  struct json_value_s *value2 = 0;
  struct json_number_s *number = 0;

  ASSERT_TRUE(value);
  ASSERT_TRUE(value->payload);
  ASSERT_EQ(json_type_array, value->type);

  array = (struct json_array_s *)value->payload;

  ASSERT_TRUE(array->start);
  ASSERT_EQ(1, array->length);

  ASSERT_TRUE(array->start->value);
  ASSERT_FALSE(array->start->next); // we have only one element

  value2 = array->start->value;

  ASSERT_TRUE(value2->payload);
  ASSERT_EQ(json_type_number, value2->type);

  number = (struct json_number_s *)value2->payload;

  ASSERT_TRUE(number->number);
  ASSERT_STREQ("-0.123e-42", number->number);
  ASSERT_EQ(strlen("-0.123e-42"), number->number_size);
  ASSERT_EQ(strlen(number->number), number->number_size);

  free(value);
}

UTEST(array, true) {
  const char payload[] = "[true]";
  struct json_value_s *value = json_parse(payload, strlen(payload));
  struct json_array_s *array = 0;
  struct json_value_s *value2 = 0;

  ASSERT_TRUE(value);
  ASSERT_TRUE(value->payload);
  ASSERT_EQ(json_type_array, value->type);

  array = (struct json_array_s *)value->payload;

  ASSERT_TRUE(array->start);
  ASSERT_EQ(1, array->length);

  ASSERT_TRUE(array->start->value);
  ASSERT_FALSE(array->start->next); // we have only one element

  value2 = array->start->value;

  ASSERT_FALSE(value2->payload);
  ASSERT_EQ(json_type_true, value2->type);

  free(value);
}

UTEST(array, false) {
  const char payload[] = "[false]";
  struct json_value_s *value = json_parse(payload, strlen(payload));
  struct json_array_s *array = 0;
  struct json_value_s *value2 = 0;

  ASSERT_TRUE(value);
  ASSERT_TRUE(value->payload);
  ASSERT_EQ(json_type_array, value->type);

  array = (struct json_array_s *)value->payload;

  ASSERT_TRUE(array->start);
  ASSERT_EQ(1, array->length);

  ASSERT_TRUE(array->start->value);
  ASSERT_FALSE(array->start->next); // we have only one element

  value2 = array->start->value;

  ASSERT_FALSE(value2->payload);
  ASSERT_EQ(json_type_false, value2->type);

  free(value);
}

UTEST(array, null) {
  const char payload[] = "[null]";
  struct json_value_s *value = json_parse(payload, strlen(payload));
  struct json_array_s *array = 0;
  struct json_value_s *value2 = 0;

  ASSERT_TRUE(value);
  ASSERT_TRUE(value->payload);
  ASSERT_EQ(json_type_array, value->type);

  array = (struct json_array_s *)value->payload;

  ASSERT_TRUE(array->start);
  ASSERT_EQ(1, array->length);

  ASSERT_TRUE(array->start->value);
  ASSERT_FALSE(array->start->next); // we have only one element

  value2 = array->start->value;

  ASSERT_FALSE(value2->payload);
  ASSERT_EQ(json_type_null, value2->type);

  free(value);
}

UTEST(no_global_object, empty) {
  const char payload[] = "";
  struct json_value_s *value = json_parse(payload, strlen(payload));
  ASSERT_FALSE(value);
}

UTEST(number, zero) {
  const char payload[] = "[0]";
  struct json_value_s *value = json_parse(payload, strlen(payload));
  struct json_array_s *array = 0;
  struct json_number_s *number = 0;

  ASSERT_TRUE(value);
  ASSERT_TRUE(value->payload);
  ASSERT_EQ(json_type_array, value->type);

  array = (struct json_array_s *)value->payload;

  ASSERT_TRUE(array->start);
  ASSERT_EQ(1, array->length);

  ASSERT_TRUE(array->start->value);
  ASSERT_TRUE(array->start->value->payload);
  ASSERT_EQ(json_type_number, array->start->value->type);

  number = (struct json_number_s *)array->start->value->payload;

  ASSERT_TRUE(number->number);

  ASSERT_STREQ("0", number->number);
  ASSERT_EQ(strlen("0"), number->number_size);
  ASSERT_EQ(strlen(number->number), number->number_size);

  free(value);
}

UTEST(number, positive) {
  const char payload[] = "[42]";
  struct json_value_s *value = json_parse(payload, strlen(payload));
  struct json_array_s *array = 0;
  struct json_number_s *number = 0;

  ASSERT_TRUE(value);
  ASSERT_TRUE(value->payload);
  ASSERT_EQ(json_type_array, value->type);

  array = (struct json_array_s *)value->payload;

  ASSERT_TRUE(array->start);
  ASSERT_EQ(1, array->length);

  ASSERT_TRUE(array->start->value);
  ASSERT_TRUE(array->start->value->payload);
  ASSERT_EQ(json_type_number, array->start->value->type);

  number = (struct json_number_s *)array->start->value->payload;

  ASSERT_TRUE(number->number);

  ASSERT_STREQ("42", number->number);
  ASSERT_EQ(strlen("42"), number->number_size);
  ASSERT_EQ(strlen(number->number), number->number_size);

  free(value);
}

UTEST(number, minus) {
  const char payload[] = "[-0]";
  struct json_value_s *value = json_parse(payload, strlen(payload));
  struct json_array_s *array = 0;
  struct json_number_s *number = 0;

  ASSERT_TRUE(value);
  ASSERT_TRUE(value->payload);
  ASSERT_EQ(json_type_array, value->type);

  array = (struct json_array_s *)value->payload;

  ASSERT_TRUE(array->start);
  ASSERT_EQ(1, array->length);

  ASSERT_TRUE(array->start->value);
  ASSERT_TRUE(array->start->value->payload);
  ASSERT_EQ(json_type_number, array->start->value->type);

  number = (struct json_number_s *)array->start->value->payload;

  ASSERT_TRUE(number->number);

  ASSERT_STREQ("-0", number->number);
  ASSERT_EQ(strlen("-0"), number->number_size);
  ASSERT_EQ(strlen(number->number), number->number_size);

  free(value);
}

UTEST(number, decimal) {
  const char payload[] = "[0.4]";
  struct json_value_s *value = json_parse(payload, strlen(payload));
  struct json_array_s *array = 0;
  struct json_number_s *number = 0;

  ASSERT_TRUE(value);
  ASSERT_TRUE(value->payload);
  ASSERT_EQ(json_type_array, value->type);

  array = (struct json_array_s *)value->payload;

  ASSERT_TRUE(array->start);
  ASSERT_EQ(1, array->length);

  ASSERT_TRUE(array->start->value);
  ASSERT_TRUE(array->start->value->payload);
  ASSERT_EQ(json_type_number, array->start->value->type);

  number = (struct json_number_s *)array->start->value->payload;

  ASSERT_TRUE(number->number);

  ASSERT_STREQ("0.4", number->number);
  ASSERT_EQ(strlen("0.4"), number->number_size);
  ASSERT_EQ(strlen(number->number), number->number_size);

  free(value);
}

UTEST(number, smalle) {
  const char payload[] = "[1e4]";
  struct json_value_s *value = json_parse(payload, strlen(payload));
  struct json_array_s *array = 0;
  struct json_number_s *number = 0;

  ASSERT_TRUE(value);
  ASSERT_TRUE(value->payload);
  ASSERT_EQ(json_type_array, value->type);

  array = (struct json_array_s *)value->payload;

  ASSERT_TRUE(array->start);
  ASSERT_EQ(1, array->length);

  ASSERT_TRUE(array->start->value);
  ASSERT_TRUE(array->start->value->payload);
  ASSERT_EQ(json_type_number, array->start->value->type);

  number = (struct json_number_s *)array->start->value->payload;

  ASSERT_TRUE(number->number);

  ASSERT_STREQ("1e4", number->number);
  ASSERT_EQ(strlen("1e4"), number->number_size);
  ASSERT_EQ(strlen(number->number), number->number_size);

  free(value);
}

UTEST(number, bige) {
  const char payload[] = "[1E4]";
  struct json_value_s *value = json_parse(payload, strlen(payload));
  struct json_array_s *array = 0;
  struct json_number_s *number = 0;

  ASSERT_TRUE(value);
  ASSERT_TRUE(value->payload);
  ASSERT_EQ(json_type_array, value->type);

  array = (struct json_array_s *)value->payload;

  ASSERT_TRUE(array->start);
  ASSERT_EQ(1, array->length);

  ASSERT_TRUE(array->start->value);
  ASSERT_TRUE(array->start->value->payload);
  ASSERT_EQ(json_type_number, array->start->value->type);

  number = (struct json_number_s *)array->start->value->payload;

  ASSERT_TRUE(number->number);

  ASSERT_STREQ("1E4", number->number);
  ASSERT_EQ(strlen("1E4"), number->number_size);
  ASSERT_EQ(strlen(number->number), number->number_size);

  free(value);
}

UTEST(number, eplus) {
  const char payload[] = "[1e+4]";
  struct json_value_s *value = json_parse(payload, strlen(payload));
  struct json_array_s *array = 0;
  struct json_number_s *number = 0;

  ASSERT_TRUE(value);
  ASSERT_TRUE(value->payload);
  ASSERT_EQ(json_type_array, value->type);

  array = (struct json_array_s *)value->payload;

  ASSERT_TRUE(array->start);
  ASSERT_EQ(1, array->length);

  ASSERT_TRUE(array->start->value);
  ASSERT_TRUE(array->start->value->payload);
  ASSERT_EQ(json_type_number, array->start->value->type);

  number = (struct json_number_s *)array->start->value->payload;

  ASSERT_TRUE(number->number);

  ASSERT_STREQ("1e+4", number->number);
  ASSERT_EQ(strlen("1e+4"), number->number_size);
  ASSERT_EQ(strlen(number->number), number->number_size);

  free(value);
}

UTEST(number, eminus) {
  const char payload[] = "[1e-4]";
  struct json_value_s *value = json_parse(payload, strlen(payload));
  struct json_array_s *array = 0;
  struct json_number_s *number = 0;

  ASSERT_TRUE(value);
  ASSERT_TRUE(value->payload);
  ASSERT_EQ(json_type_array, value->type);

  array = (struct json_array_s *)value->payload;

  ASSERT_TRUE(array->start);
  ASSERT_EQ(1, array->length);

  ASSERT_TRUE(array->start->value);
  ASSERT_TRUE(array->start->value->payload);
  ASSERT_EQ(json_type_number, array->start->value->type);

  number = (struct json_number_s *)array->start->value->payload;

  ASSERT_TRUE(number->number);

  ASSERT_STREQ("1e-4", number->number);
  ASSERT_EQ(strlen("1e-4"), number->number_size);
  ASSERT_EQ(strlen(number->number), number->number_size);

  free(value);
}

UTEST(object, missing_closing_bracket) {
  const char payload[] = "{\n  \"dps\":[1, 2, {\"a\" : true]\n}";

  struct json_parse_result_s result;

  struct json_value_s *value =
      json_parse_ex(payload, strlen(payload), 0, 0, 0, &result);

  ASSERT_FALSE(value);

  ASSERT_EQ(json_parse_error_expected_comma_or_closing_bracket, result.error);
  ASSERT_EQ(28, result.error_offset);
  ASSERT_EQ(2, result.error_line_no);
  ASSERT_EQ(27, result.error_row_no);
}

UTEST(array, missing_closing_bracket) {
  const char payload[] = "{\n  \"dps\":[1, 2, 3\n}";

  struct json_parse_result_s result;

  struct json_value_s *value =
      json_parse_ex(payload, strlen(payload), 0, 0, 0, &result);

  ASSERT_FALSE(value);

  ASSERT_EQ(json_parse_error_expected_comma_or_closing_bracket, result.error);
  ASSERT_EQ(19, result.error_offset);
  ASSERT_EQ(3, result.error_line_no);
  ASSERT_EQ(1, result.error_row_no);
}

UTEST(object, empty_strings) {
  const char payload[] = "{\"foo\": \"\", \"bar\": \"\"}";
  struct json_value_s *value = json_parse(payload, strlen(payload));
  struct json_object_s *object = 0;
  struct json_object_element_s *el1 = 0;
  struct json_object_element_s *el2 = 0;
  struct json_string_s *s1 = 0;
  struct json_string_s *s2 = 0;

  ASSERT_TRUE(value);
  ASSERT_TRUE(value->payload);
  ASSERT_EQ(json_type_object, value->type);

  object = (struct json_object_s *)value->payload;

  ASSERT_EQ(2, object->length);

  el1 = object->start;
  ASSERT_TRUE(el1);
  el2 = el1->next;
  ASSERT_TRUE(el2);

  ASSERT_FALSE(el2->next); // we have only one element

  ASSERT_TRUE(el1->name);
  ASSERT_TRUE(el1->name->string);
  ASSERT_STREQ("foo", el1->name->string);
  ASSERT_EQ(strlen("foo"), el1->name->string_size);
  ASSERT_EQ(strlen(el1->name->string), el1->name->string_size);

  ASSERT_TRUE(el1->value);
  ASSERT_EQ(json_type_string, el1->value->type);
  s1 = (struct json_string_s *)el1->value->payload;
  ASSERT_TRUE(s1);
  ASSERT_TRUE(s1->string);
  ASSERT_STREQ("", s1->string);
  ASSERT_EQ(strlen(""), s1->string_size);
  ASSERT_EQ(strlen(s1->string), s1->string_size);

  ASSERT_TRUE(el2->name);
  ASSERT_TRUE(el2->name->string);
  ASSERT_STREQ("bar", el2->name->string);
  ASSERT_EQ(strlen("bar"), el2->name->string_size);
  ASSERT_EQ(strlen(el2->name->string), el2->name->string_size);

  ASSERT_TRUE(el2->value);
  ASSERT_EQ(json_type_string, el2->value->type);
  s2 = (struct json_string_s *)el2->value->payload;
  ASSERT_TRUE(s2);
  ASSERT_TRUE(s2->string);
  ASSERT_STREQ("", s2->string);
  ASSERT_EQ(strlen(""), s2->string_size);
  ASSERT_EQ(strlen(s2->string), s2->string_size);

  free(value);
}

UTEST(string, unicode_escape) {
  const char expected_str[] =
      "\xEA\x83\x8A"
      "ABC"
      "\xC3\x8A"
      "DEF"
      "\n"
      " ,\xC5\xBD,\xE0\xA0\x80,\xE0\xA6\xA8,\xE2\x99\x9E,\xEF\xBF\xBD,\xD0\xA8,"
      "\xE4\x93\x8D,\xF0\x90\x80\x80,\xF0\x9F\x98\x83.";
  const char payload[] = "[\"\\ua0caABC\\u00caDEF\\u000a"
                         "\\u0020,\\u017D,\\u0800,\\u09A8,\\u265E,\\uFFFD,"
                         "\\u0428,\\u44CD,\\uD800\\uDC00,\\uD83D\\uDE03.\"]";
  struct json_value_s *value = json_parse(payload, strlen(payload));
  struct json_array_s *array = 0;
  struct json_string_s *str = 0;

  ASSERT_TRUE(value);
  ASSERT_TRUE(value->payload);
  ASSERT_EQ(json_type_array, value->type);

  array = (struct json_array_s *)value->payload;

  ASSERT_TRUE(array->start);
  ASSERT_EQ(1, array->length);

  ASSERT_TRUE(array->start->value);
  ASSERT_TRUE(array->start->value->payload);
  ASSERT_EQ(json_type_string, array->start->value->type);

  str = (struct json_string_s *)array->start->value->payload;

  ASSERT_TRUE(str->string);

  ASSERT_STREQ(expected_str, str->string);
  ASSERT_EQ(strlen(expected_str), str->string_size);
  ASSERT_EQ(strlen(str->string), str->string_size);

  free(value);
}

UTEST(helpers, all) {
  const char payload[] = "{\"foo\" : [ null, true, false, \"bar\", 42 ]}";
  struct json_value_s *const root = json_parse(payload, strlen(payload));
  struct json_object_s *object = 0;
  struct json_object_element_s *object_element = 0;
  struct json_array_s *array = 0;
  struct json_array_element_s *array_element = 0;
  struct json_number_s *number = 0;
  struct json_string_s *string = 0;

  object = json_value_as_object(root);
  ASSERT_TRUE(object);
  ASSERT_EQ(object->length, 1);

  object_element = object->start;
  ASSERT_TRUE(object_element);
  ASSERT_STREQ(object_element->name->string, "foo");
  ASSERT_FALSE(object_element->next);

  array = json_value_as_array(object_element->value);
  ASSERT_TRUE(array);
  ASSERT_EQ(array->length, 5);

  // null
  array_element = array->start;
  ASSERT_TRUE(array_element);
  ASSERT_TRUE(json_value_is_null(array_element->value));

  // true
  array_element = array_element->next;
  ASSERT_TRUE(array_element);
  ASSERT_TRUE(json_value_is_true(array_element->value));

  // false
  array_element = array_element->next;
  ASSERT_TRUE(array_element);
  ASSERT_TRUE(json_value_is_false(array_element->value));

  // string
  array_element = array_element->next;
  ASSERT_TRUE(array_element);

  string = json_value_as_string(array_element->value);
  ASSERT_TRUE(string);
  ASSERT_STREQ(string->string, "bar");

  // number
  array_element = array_element->next;
  ASSERT_TRUE(array_element);

  number = json_value_as_number(array_element->value);
  ASSERT_TRUE(number);
  ASSERT_STREQ(number->number, "42");

  ASSERT_FALSE(array_element->next);

  free(root);
}

UTEST(random, overflow) {
  const char payload[] = "\n\t\t\n\t\t\t\t\"\x00";
  struct json_value_s *const root = json_parse(payload, sizeof(payload));
  ASSERT_FALSE(root);
}

#define assert(x) ASSERT_TRUE(x)

UTEST(generated, readme){
#include "generated.h"
}

UTEST_MAIN();
