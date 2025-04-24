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

UTEST(allow_global_object, empty) {
  const char payload[] = "";
  struct json_value_s *value = json_parse_ex(
      payload, strlen(payload), json_parse_flags_allow_global_object, 0, 0, 0);
  struct json_object_s *object = 0;

  ASSERT_TRUE(value);
  ASSERT_TRUE(value->payload);
  ASSERT_EQ(json_type_object, value->type);

  object = (struct json_object_s *)value->payload;

  ASSERT_FALSE(object->start);
  ASSERT_EQ(0, object->length);

  free(value);
}

UTEST(allow_global_object, string) {
  const char payload[] = "\"foo\" : \"Heyo, gaia?\"";
  struct json_value_s *value = json_parse_ex(
      payload, strlen(payload), json_parse_flags_allow_global_object, 0, 0, 0);
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

UTEST(allow_global_object, number) {
  const char payload[] = "\"foo\" : -0.123e-42";
  struct json_value_s *value = json_parse_ex(
      payload, strlen(payload), json_parse_flags_allow_global_object, 0, 0, 0);
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

UTEST(allow_global_object, object) {
  const char payload[] = "\"foo\" : {}";
  struct json_value_s *value = json_parse_ex(
      payload, strlen(payload), json_parse_flags_allow_global_object, 0, 0, 0);
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

UTEST(allow_global_object, array) {
  const char payload[] = "\"foo\" : []";
  struct json_value_s *value = json_parse_ex(
      payload, strlen(payload), json_parse_flags_allow_global_object, 0, 0, 0);
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

UTEST(allow_global_object, true) {
  const char payload[] = "\"foo\" : true";
  struct json_value_s *value = json_parse_ex(
      payload, strlen(payload), json_parse_flags_allow_global_object, 0, 0, 0);
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

UTEST(allow_global_object, false) {
  const char payload[] = "\"foo\" : false";
  struct json_value_s *value = json_parse_ex(
      payload, strlen(payload), json_parse_flags_allow_global_object, 0, 0, 0);
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

UTEST(allow_global_object, null) {
  const char payload[] = "\"foo\" : null";
  struct json_value_s *value = json_parse_ex(
      payload, strlen(payload), json_parse_flags_allow_global_object, 0, 0, 0);
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

UTEST(allow_global_object, not_a_global_object_afterall) {
  const char payload[] = "{}";
  struct json_value_s *value = json_parse_ex(
      payload, strlen(payload), json_parse_flags_allow_global_object, 0, 0, 0);
  struct json_object_s *object = 0;

  ASSERT_TRUE(value);
  ASSERT_TRUE(value->payload);
  ASSERT_EQ(json_type_object, value->type);

  object = (struct json_object_s *)value->payload;

  ASSERT_FALSE(object->start);
  ASSERT_EQ(0, object->length);

  free(value);
}

struct allow_global_object {
  struct json_value_s *value;
};

UTEST_F_SETUP(allow_global_object) {
  const char payload[] = "\"foo\" : null";
  utest_fixture->value = json_parse_ex(
      payload, strlen(payload), json_parse_flags_allow_global_object, 0, 0, 0);

  ASSERT_TRUE(utest_fixture->value);
}

UTEST_F_TEARDOWN(allow_global_object) {
  struct json_value_s *value = utest_fixture->value;
  struct json_object_s *object = 0;
  struct json_value_s *value2 = 0;
  size_t size = 0;
  void *json = 0;

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

UTEST_F(allow_global_object, read_write_pretty_read) {
  size_t size = 0;
  void *json = json_write_pretty(utest_fixture->value, "  ", "\n", &size);

  free(utest_fixture->value);

  utest_fixture->value = json_parse(json, size - 1);

  free(json);
}

UTEST_F(allow_global_object, read_write_minified_read) {
  size_t size = 0;
  void *json = json_write_minified(utest_fixture->value, &size);

  free(utest_fixture->value);

  utest_fixture->value = json_parse(json, size - 1);

  free(json);
}
