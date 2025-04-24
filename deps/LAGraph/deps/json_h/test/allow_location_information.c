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

UTEST(allow_location_information, object_one) {
  const char payload[] = "{\"foo\" : true,\n\"bar\" : false}";
  struct json_value_ex_s *value_ex = (struct json_value_ex_s *)json_parse_ex(
      payload, strlen(payload), json_parse_flags_allow_location_information, 0,
      0, 0);
  struct json_object_s *object = 0;
  struct json_object_element_s *element = 0;
  struct json_string_ex_s *string_ex = 0;
  struct json_value_ex_s *value_ex2 = 0;

  ASSERT_TRUE(value_ex);

  ASSERT_EQ(0, value_ex->offset);
  ASSERT_EQ(1, value_ex->line_no);
  ASSERT_EQ(0, value_ex->row_no);

  object = (struct json_object_s *)value_ex->value.payload;

  ASSERT_TRUE(object->start);
  ASSERT_EQ(2, object->length);

  element = object->start;

  ASSERT_TRUE(element->name);
  ASSERT_TRUE(element->value);
  ASSERT_TRUE(element->next);

  string_ex = (struct json_string_ex_s *)element->name;

  ASSERT_TRUE(string_ex->string.string);
  ASSERT_STREQ("foo", string_ex->string.string);
  ASSERT_EQ(strlen("foo"), string_ex->string.string_size);
  ASSERT_EQ(strlen(string_ex->string.string), string_ex->string.string_size);

  ASSERT_EQ(1, string_ex->offset);
  ASSERT_EQ(1, string_ex->line_no);
  ASSERT_EQ(1, string_ex->row_no);

  value_ex2 = (struct json_value_ex_s *)element->value;

  ASSERT_FALSE(value_ex2->value.payload);
  ASSERT_EQ(json_type_true, value_ex2->value.type);

  ASSERT_EQ(9, value_ex2->offset);
  ASSERT_EQ(1, value_ex2->line_no);
  ASSERT_EQ(9, value_ex2->row_no);

  element = element->next;

  ASSERT_FALSE(element->next);

  string_ex = (struct json_string_ex_s *)element->name;

  ASSERT_TRUE(string_ex->string.string);
  ASSERT_STREQ("bar", string_ex->string.string);
  ASSERT_EQ(strlen("bar"), string_ex->string.string_size);
  ASSERT_EQ(strlen(string_ex->string.string), string_ex->string.string_size);

  ASSERT_EQ(15, string_ex->offset);
  ASSERT_EQ(2, string_ex->line_no);
  ASSERT_EQ(1, string_ex->row_no);

  value_ex2 = (struct json_value_ex_s *)element->value;

  ASSERT_FALSE(value_ex2->value.payload);
  ASSERT_EQ(json_type_false, value_ex2->value.type);

  ASSERT_EQ(23, value_ex2->offset);
  ASSERT_EQ(2, value_ex2->line_no);
  ASSERT_EQ(9, value_ex2->row_no);

  free(value_ex);
}
