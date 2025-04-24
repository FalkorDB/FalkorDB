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

UTEST(write_minified, object_empty) {
  struct json_object_s object = {0, 0};
  struct json_value_s value = {&object, json_type_object};
  size_t size = 0;
  void *minified = json_write_minified(&value, &size);

  ASSERT_TRUE(minified);
  ASSERT_EQ(strlen(static_cast<char *>(minified)) + 1, size);
  ASSERT_STREQ("{}", static_cast<char *>(minified));

  free(minified);
}

UTEST(write_minified, object_string) {
  struct json_string_s sub = {"yaba daba", strlen("yaba daba")};
  struct json_value_s sub_value = {&sub, json_type_string};
  struct json_string_s sub_string = {"sub", strlen("sub")};
  struct json_object_element_s element = {&sub_string, &sub_value, 0};
  struct json_object_s object = {&element, 1};
  struct json_value_s value = {&object, json_type_object};
  size_t size = 0;
  void *minified = json_write_minified(&value, &size);

  ASSERT_TRUE(minified);
  ASSERT_EQ(strlen(static_cast<char *>(minified)) + 1, size);
  ASSERT_STREQ("{\"sub\":\"yaba daba\"}", static_cast<char *>(minified));

  free(minified);
}

UTEST(write_minified, object_number) {
  struct json_number_s sub = {"-0.234e+42", strlen("-0.234e+42")};
  struct json_value_s sub_value = {&sub, json_type_number};
  struct json_string_s sub_string = {"sub", strlen("sub")};
  struct json_object_element_s element = {&sub_string, &sub_value, 0};
  struct json_object_s object = {&element, 1};
  struct json_value_s value = {&object, json_type_object};
  size_t size = 0;
  void *minified = json_write_minified(&value, &size);

  ASSERT_TRUE(minified);
  ASSERT_EQ(strlen(static_cast<char *>(minified)) + 1, size);
  ASSERT_STREQ("{\"sub\":-0.234e+42}", static_cast<char *>(minified));

  free(minified);
}

UTEST(write_minified, object_object) {
  struct json_object_s sub = {0, 0};
  struct json_value_s sub_value = {&sub, json_type_object};
  struct json_string_s sub_string = {"sub", strlen("sub")};
  struct json_object_element_s element = {&sub_string, &sub_value, 0};
  struct json_object_s object = {&element, 1};
  struct json_value_s value = {&object, json_type_object};
  size_t size = 0;
  void *minified = json_write_minified(&value, &size);

  ASSERT_TRUE(minified);
  ASSERT_EQ(strlen(static_cast<char *>(minified)) + 1, size);
  ASSERT_STREQ("{\"sub\":{}}", static_cast<char *>(minified));

  free(minified);
}

UTEST(write_minified, object_array) {
  struct json_array_s sub = {0, 0};
  struct json_value_s sub_value = {&sub, json_type_array};
  struct json_string_s sub_string = {"sub", strlen("sub")};
  struct json_object_element_s element = {&sub_string, &sub_value, 0};
  struct json_object_s object = {&element, 1};
  struct json_value_s value = {&object, json_type_object};
  size_t size = 0;
  void *minified = json_write_minified(&value, &size);

  ASSERT_TRUE(minified);
  ASSERT_EQ(strlen(static_cast<char *>(minified)) + 1, size);
  ASSERT_STREQ("{\"sub\":[]}", static_cast<char *>(minified));

  free(minified);
}

UTEST(write_minified, object_true) {
  struct json_value_s sub_value = {0, json_type_true};
  struct json_string_s sub_string = {"sub", strlen("sub")};
  struct json_object_element_s element = {&sub_string, &sub_value, 0};
  struct json_object_s object = {&element, 1};
  struct json_value_s value = {&object, json_type_object};
  size_t size = 0;
  void *minified = json_write_minified(&value, &size);

  ASSERT_TRUE(minified);
  ASSERT_EQ(strlen(static_cast<char *>(minified)) + 1, size);
  ASSERT_STREQ("{\"sub\":true}", static_cast<char *>(minified));

  free(minified);
}

UTEST(write_minified, object_false) {
  struct json_value_s sub_value = {0, json_type_false};
  struct json_string_s sub_string = {"sub", strlen("sub")};
  struct json_object_element_s element = {&sub_string, &sub_value, 0};
  struct json_object_s object = {&element, 1};
  struct json_value_s value = {&object, json_type_object};
  size_t size = 0;
  void *minified = json_write_minified(&value, &size);

  ASSERT_TRUE(minified);
  ASSERT_EQ(strlen(static_cast<char *>(minified)) + 1, size);
  ASSERT_STREQ("{\"sub\":false}", static_cast<char *>(minified));

  free(minified);
}

UTEST(write_minified, object_null) {
  struct json_value_s sub_value = {0, json_type_null};
  struct json_string_s sub_string = {"sub", strlen("sub")};
  struct json_object_element_s element = {&sub_string, &sub_value, 0};
  struct json_object_s object = {&element, 1};
  struct json_value_s value = {&object, json_type_object};
  size_t size = 0;
  void *minified = json_write_minified(&value, &size);

  ASSERT_TRUE(minified);
  ASSERT_EQ(strlen(static_cast<char *>(minified)) + 1, size);
  ASSERT_STREQ("{\"sub\":null}", static_cast<char *>(minified));

  free(minified);
}

UTEST(write_minified, array_empty) {
  struct json_array_s array = {0, 0};
  struct json_value_s value = {&array, json_type_array};
  size_t size = 0;
  void *minified = json_write_minified(&value, &size);

  ASSERT_TRUE(minified);
  ASSERT_EQ(strlen(static_cast<char *>(minified)) + 1, size);
  ASSERT_STREQ("[]", static_cast<char *>(minified));

  free(minified);
}

UTEST(write_minified, array_string) {
  struct json_string_s sub = {"yaba daba", strlen("yaba daba")};
  struct json_value_s sub_value = {&sub, json_type_string};
  struct json_array_element_s element = {&sub_value, 0};
  struct json_array_s object = {&element, 1};
  struct json_value_s value = {&object, json_type_array};
  size_t size = 0;
  void *minified = json_write_minified(&value, &size);

  ASSERT_TRUE(minified);
  ASSERT_EQ(strlen(static_cast<char *>(minified)) + 1, size);
  ASSERT_STREQ("[\"yaba daba\"]", static_cast<char *>(minified));

  free(minified);
}

UTEST(write_minified, array_number) {
  struct json_number_s sub = {"-0.234e+42", strlen("-0.234e+42")};
  struct json_value_s sub_value = {&sub, json_type_number};
  struct json_array_element_s element = {&sub_value, 0};
  struct json_array_s object = {&element, 1};
  struct json_value_s value = {&object, json_type_array};
  size_t size = 0;
  void *minified = json_write_minified(&value, &size);

  ASSERT_TRUE(minified);
  ASSERT_EQ(strlen(static_cast<char *>(minified)) + 1, size);
  ASSERT_STREQ("[-0.234e+42]", static_cast<char *>(minified));

  free(minified);
}

UTEST(write_minified, array_object) {
  struct json_object_s sub = {0, 0};
  struct json_value_s sub_value = {&sub, json_type_object};
  struct json_array_element_s element = {&sub_value, 0};
  struct json_array_s object = {&element, 1};
  struct json_value_s value = {&object, json_type_array};
  size_t size = 0;
  void *minified = json_write_minified(&value, &size);

  ASSERT_TRUE(minified);
  ASSERT_EQ(strlen(static_cast<char *>(minified)) + 1, size);
  ASSERT_STREQ("[{}]", static_cast<char *>(minified));

  free(minified);
}

UTEST(write_minified, array_array) {
  struct json_array_s sub = {0, 0};
  struct json_value_s sub_value = {&sub, json_type_array};
  struct json_array_element_s element = {&sub_value, 0};
  struct json_array_s object = {&element, 1};
  struct json_value_s value = {&object, json_type_array};
  size_t size = 0;
  void *minified = json_write_minified(&value, &size);

  ASSERT_TRUE(minified);
  ASSERT_EQ(strlen(static_cast<char *>(minified)) + 1, size);
  ASSERT_STREQ("[[]]", static_cast<char *>(minified));

  free(minified);
}

UTEST(write_minified, array_true) {
  struct json_value_s sub_value = {0, json_type_true};
  struct json_array_element_s element = {&sub_value, 0};
  struct json_array_s object = {&element, 1};
  struct json_value_s value = {&object, json_type_array};
  size_t size = 0;
  void *minified = json_write_minified(&value, &size);

  ASSERT_TRUE(minified);
  ASSERT_EQ(strlen(static_cast<char *>(minified)) + 1, size);
  ASSERT_STREQ("[true]", static_cast<char *>(minified));

  free(minified);
}

UTEST(write_minified, array_false) {
  struct json_value_s sub_value = {0, json_type_false};
  struct json_array_element_s element = {&sub_value, 0};
  struct json_array_s object = {&element, 1};
  struct json_value_s value = {&object, json_type_array};
  size_t size = 0;
  void *minified = json_write_minified(&value, &size);

  ASSERT_TRUE(minified);
  ASSERT_EQ(strlen(static_cast<char *>(minified)) + 1, size);
  ASSERT_STREQ("[false]", static_cast<char *>(minified));

  free(minified);
}

UTEST(write_minified, array_null) {
  struct json_value_s sub_value = {0, json_type_null};
  struct json_array_element_s element = {&sub_value, 0};
  struct json_array_s object = {&element, 1};
  struct json_value_s value = {&object, json_type_array};
  size_t size = 0;
  void *minified = json_write_minified(&value, &size);

  ASSERT_TRUE(minified);
  ASSERT_EQ(strlen(static_cast<char *>(minified)) + 1, size);
  ASSERT_STREQ("[null]", static_cast<char *>(minified));

  free(minified);
}
