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

UTEST(allow_c_style_comments, single_line) {
  const char payload[] =
      "// a \n { // b \n \"foo\" // c \n : // d \n null // e \n } // f";
  struct json_value_s *value =
      json_parse_ex(payload, strlen(payload),
                    json_parse_flags_allow_c_style_comments, 0, 0, 0);
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

UTEST(allow_c_style_comments, multi_line) {
  const char payload[] = "/* a */ { /* b */ \"foo\" /* c1 \n c2 */ : /* d */ "
                         "null /* e1 \n e2 */ } /* f */";
  struct json_value_s *value =
      json_parse_ex(payload, strlen(payload),
                    json_parse_flags_allow_c_style_comments, 0, 0, 0);
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

UTEST(allow_c_style_comments, multiple) {
  const char payload[] = "{/**/ /**/\"foo\" : null}";
  struct json_value_s *value =
      json_parse_ex(payload, strlen(payload),
                    json_parse_flags_allow_c_style_comments, 0, 0, 0);
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

struct allow_c_style_comments {
  struct json_value_s *value;
};

UTEST_F_SETUP(allow_c_style_comments) {
  const char payload[] = "{/**/ /**/\"foo\" : null // abc \n}";
  utest_fixture->value =
      json_parse_ex(payload, strlen(payload),
                    json_parse_flags_allow_c_style_comments, 0, 0, 0);

  ASSERT_TRUE(utest_fixture->value);
}

UTEST_F_TEARDOWN(allow_c_style_comments) {
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

UTEST_F(allow_c_style_comments, read_write_pretty_read) {
  size_t size = 0;
  void *json = json_write_pretty(utest_fixture->value, "  ", "\n", &size);

  free(utest_fixture->value);

  utest_fixture->value = json_parse(json, size - 1);

  free(json);
}

UTEST_F(allow_c_style_comments, read_write_minified_read) {
  size_t size = 0;
  void *json = json_write_minified(utest_fixture->value, &size);

  free(utest_fixture->value);

  utest_fixture->value = json_parse(json, size - 1);

  free(json);
}
