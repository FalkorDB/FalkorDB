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

#include <stdlib.h>

#include "utest.h"

#include "json.h"

UTEST(allocator, malloc) {
  struct _ {
    static void *alloc(void *, size_t size) { return malloc(size); }
  };

  const char payload[] = "{}";
  struct json_value_s *value =
      json_parse_ex(payload, strlen(payload), 0, &_::alloc, 0, 0);
  struct json_object_s *object = 0;

  ASSERT_TRUE(value);
  ASSERT_EQ(json_type_object, value->type);

  object = (struct json_object_s *)value->payload;

  ASSERT_FALSE(object->start);
  ASSERT_EQ(0, object->length);

  free(value);
}

UTEST(allocator, static_data) {
  struct _ {
    static void *alloc(void *, size_t size) {
      static char data[256];
      return data;
    }
  };

  const char payload[] = "{}";
  struct json_value_s *value =
      json_parse_ex(payload, strlen(payload), 0, &_::alloc, 0, 0);
  struct json_object_s *object = 0;

  ASSERT_TRUE(value);
  ASSERT_EQ(json_type_object, value->type);

  object = (struct json_object_s *)value->payload;

  ASSERT_FALSE(object->start);
  ASSERT_EQ(0, object->length);
}

UTEST(allocator, null) {
  struct _ {
    static void *alloc(void *, size_t) { return 0; }
  };

  const char payload[] = "{}";
  struct json_parse_result_s result;
  struct json_value_s *value =
      json_parse_ex(payload, strlen(payload), 0, &_::alloc, 0, &result);
  struct json_object_s *object = 0;

  ASSERT_FALSE(value);

  ASSERT_EQ(json_parse_error_allocator_failed, result.error);
  ASSERT_EQ(0, result.error_offset);
  ASSERT_EQ(0, result.error_line_no);
  ASSERT_EQ(0, result.error_row_no);
}

UTEST(allocator, user_data) {
  struct _ {
    static void *alloc(void *user_data, size_t) { return user_data; }
  };

  char data[128];
  const char payload[] = "{}";
  struct json_value_s *value =
      json_parse_ex(payload, strlen(payload), 0, &_::alloc, data, 0);
  struct json_object_s *object = 0;

  ASSERT_TRUE(value);
  ASSERT_EQ(json_type_object, value->type);

  object = (struct json_object_s *)value->payload;

  ASSERT_FALSE(object->start);
  ASSERT_EQ(0, object->length);
}
