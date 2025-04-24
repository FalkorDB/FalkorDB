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

UTEST(extract, all) {
  const char payload[] =
      "{\"foo\" : { \"bar\" : [123, false, null, true], \"haz\" : \"haha\" }}";
  struct json_value_s *const value = json_parse(payload, strlen(payload));
  ASSERT_TRUE(value);

  ASSERT_TRUE(json_value_as_object(value));
  ASSERT_TRUE(json_value_as_object(value)->start);

  struct json_value_s *const extract =
      json_extract_value(json_value_as_object(value)->start->value);
  free(value);

  ASSERT_TRUE(extract);
  ASSERT_TRUE(extract->payload);

  struct json_object_s *const foo = json_value_as_object(extract);
  ASSERT_TRUE(foo);
  ASSERT_EQ(2, foo->length);

  struct json_object_element_s *object_element = foo->start;
  ASSERT_TRUE(object_element);

  ASSERT_TRUE(object_element->name);
  ASSERT_STREQ("bar", object_element->name->string);
  ASSERT_EQ(strlen("bar"), object_element->name->string_size);

  struct json_array_s *const bar = json_value_as_array(object_element->value);
  ASSERT_TRUE(bar);

  struct json_array_element_s *array_element = bar->start;
  ASSERT_TRUE(array_element);

  struct json_number_s *const oneTwoThree =
      json_value_as_number(array_element->value);
  ASSERT_TRUE(oneTwoThree);
  ASSERT_STRNEQ("123", oneTwoThree->number, oneTwoThree->number_size);
  ASSERT_EQ(strlen("123"), oneTwoThree->number_size);

  array_element = array_element->next;

  ASSERT_TRUE(json_value_is_false(array_element->value));

  array_element = array_element->next;

  ASSERT_TRUE(json_value_is_null(array_element->value));

  array_element = array_element->next;

  ASSERT_TRUE(json_value_is_true(array_element->value));

  ASSERT_FALSE(array_element->next);

  object_element = object_element->next;
  ASSERT_TRUE(object_element);

  ASSERT_TRUE(object_element->name);
  ASSERT_STREQ("haz", object_element->name->string);
  ASSERT_EQ(strlen("haz"), object_element->name->string_size);

  struct json_string_s *const haz = json_value_as_string(object_element->value);
  ASSERT_TRUE(haz);

  ASSERT_STREQ("haha", haz->string);
  ASSERT_EQ(strlen("haha"), haz->string_size);

  free(extract);
}