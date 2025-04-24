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

UTEST(allow_no_commas, object_one) {
  const char payload[] = "{\"foo\" : true \"bar\" : false}";
  struct json_value_s *value = json_parse_ex(
      payload, strlen(payload), json_parse_flags_allow_no_commas, 0, 0, 0);
  struct json_object_s *object = 0;
  struct json_object_element_s *element = 0;
  struct json_value_s *value2 = 0;

  ASSERT_TRUE(value);
  ASSERT_TRUE(value->payload);
  ASSERT_EQ(json_type_object, value->type);

  object = (struct json_object_s *)value->payload;

  ASSERT_TRUE(object->start);
  ASSERT_EQ(2, object->length);

  element = object->start;

  ASSERT_TRUE(element->name);
  ASSERT_TRUE(element->value);
  ASSERT_TRUE(element->next);

  ASSERT_TRUE(element->name->string);
  ASSERT_STREQ("foo", element->name->string);
  ASSERT_EQ(strlen("foo"), element->name->string_size);
  ASSERT_EQ(strlen(element->name->string), element->name->string_size);

  value2 = element->value;

  ASSERT_FALSE(value2->payload);
  ASSERT_EQ(json_type_true, value2->type);

  element = element->next;

  ASSERT_FALSE(element->next);

  ASSERT_TRUE(element->name->string);
  ASSERT_STREQ("bar", element->name->string);
  ASSERT_EQ(strlen("bar"), element->name->string_size);
  ASSERT_EQ(strlen(element->name->string), element->name->string_size);

  value2 = element->value;

  ASSERT_FALSE(value2->payload);
  ASSERT_EQ(json_type_false, value2->type);

  free(value);
}

UTEST(allow_no_commas, object_two) {
  const char payload[] = "{\"foo\" : \"yada\"\"bar\" : null}";
  struct json_value_s *value = json_parse_ex(
      payload, strlen(payload), json_parse_flags_allow_no_commas, 0, 0, 0);
  struct json_object_s *object = 0;
  struct json_object_element_s *element = 0;
  struct json_value_s *value2 = 0;
  struct json_string_s *string = 0;

  ASSERT_TRUE(value);
  ASSERT_TRUE(value->payload);
  ASSERT_EQ(json_type_object, value->type);

  object = (struct json_object_s *)value->payload;

  ASSERT_TRUE(object->start);
  ASSERT_EQ(2, object->length);

  element = object->start;

  ASSERT_TRUE(element->name);
  ASSERT_TRUE(element->value);
  ASSERT_TRUE(element->next);

  ASSERT_TRUE(element->name->string);
  ASSERT_STREQ("foo", element->name->string);
  ASSERT_EQ(strlen("foo"), element->name->string_size);
  ASSERT_EQ(strlen(element->name->string), element->name->string_size);

  value2 = element->value;

  ASSERT_TRUE(value2->payload);
  ASSERT_EQ(json_type_string, value2->type);

  string = (struct json_string_s *)value2->payload;

  ASSERT_TRUE(string->string);

  ASSERT_TRUE(string->string);
  ASSERT_STREQ("yada", string->string);
  ASSERT_EQ(strlen("yada"), string->string_size);
  ASSERT_EQ(strlen(string->string), string->string_size);

  element = element->next;

  ASSERT_FALSE(element->next);

  ASSERT_TRUE(element->name->string);
  ASSERT_STREQ("bar", element->name->string);
  ASSERT_EQ(strlen("bar"), element->name->string_size);
  ASSERT_EQ(strlen(element->name->string), element->name->string_size);

  value2 = element->value;

  ASSERT_FALSE(value2->payload);
  ASSERT_EQ(json_type_null, value2->type);

  free(value);
}

UTEST(allow_no_commas, array_one) {
  const char payload[] = "[false true]";
  struct json_value_s *value = json_parse_ex(
      payload, strlen(payload), json_parse_flags_allow_no_commas, 0, 0, 0);
  struct json_array_s *array = 0;
  struct json_array_element_s *element = 0;
  struct json_value_s *value2 = 0;

  ASSERT_TRUE(value);
  ASSERT_TRUE(value->payload);
  ASSERT_EQ(json_type_array, value->type);

  array = (struct json_array_s *)value->payload;

  ASSERT_TRUE(array->start);
  ASSERT_EQ(2, array->length);

  element = array->start;

  ASSERT_TRUE(element->value);
  ASSERT_TRUE(element->next);

  ASSERT_EQ(json_type_false, element->value->type);
  ASSERT_FALSE(element->value->payload);

  element = element->next;

  ASSERT_TRUE(element->value);
  ASSERT_FALSE(element->next);

  ASSERT_EQ(json_type_true, element->value->type);
  ASSERT_FALSE(element->value->payload);

  free(value);
}

struct allow_no_commas {
  struct json_value_s *value;
};

UTEST_F_SETUP(allow_no_commas) {
  const char payload[] = "[false true]";
  utest_fixture->value = json_parse_ex(
      payload, strlen(payload), json_parse_flags_allow_no_commas, 0, 0, 0);

  ASSERT_TRUE(utest_fixture->value);
}

UTEST_F_TEARDOWN(allow_no_commas) {
  struct json_value_s *value = utest_fixture->value;
  struct json_array_s *array = 0;
  struct json_array_element_s *element = 0;
  struct json_value_s *value2 = 0;

  ASSERT_TRUE(value);
  ASSERT_TRUE(value->payload);
  ASSERT_EQ(json_type_array, value->type);

  array = (struct json_array_s *)value->payload;

  ASSERT_TRUE(array->start);
  ASSERT_EQ(2, array->length);

  element = array->start;

  ASSERT_TRUE(element->value);
  ASSERT_TRUE(element->next);

  ASSERT_EQ(json_type_false, element->value->type);
  ASSERT_FALSE(element->value->payload);

  element = element->next;

  ASSERT_TRUE(element->value);
  ASSERT_FALSE(element->next);

  ASSERT_EQ(json_type_true, element->value->type);
  ASSERT_FALSE(element->value->payload);

  free(value);
}

UTEST_F(allow_no_commas, read_write_pretty_read) {
  size_t size = 0;
  void *json = json_write_pretty(utest_fixture->value, "  ", "\n", &size);

  free(utest_fixture->value);

  utest_fixture->value = json_parse(json, size - 1);

  free(json);
}

UTEST_F(allow_no_commas, read_write_minified_read) {
  size_t size = 0;
  void *json = json_write_minified(utest_fixture->value, &size);

  free(utest_fixture->value);

  utest_fixture->value = json_parse(json, size - 1);

  free(json);
}
