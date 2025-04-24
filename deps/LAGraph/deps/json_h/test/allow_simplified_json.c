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

struct allow_simplified_json {
  struct json_value_s *value;
};

UTEST_F_SETUP(allow_simplified_json) {
  const char payload[] = "description = \"A sample, json document.\"\n"
                         "author = {\n"
                         "  name = \"Autodesk\"\n"
                         "}\n"
                         "keywords = [\"sample\", \"sjson\"]\n";

  utest_fixture->value =
      json_parse_ex(payload, strlen(payload),
                    json_parse_flags_allow_simplified_json, 0, 0, 0);

  ASSERT_TRUE(utest_fixture->value);
}

UTEST_F_TEARDOWN(allow_simplified_json) {
  ASSERT_TRUE(utest_fixture->value);
  free(utest_fixture->value);
}

UTEST_F(allow_simplified_json, read_write_pretty_read) {
  size_t size = 0;
  void *json = json_write_pretty(utest_fixture->value, "  ", "\n", &size);

  free(utest_fixture->value);

  utest_fixture->value = json_parse(json, size - 1);

  free(json);
}

UTEST_F(allow_simplified_json, read_write_minified_read) {
  size_t size = 0;
  void *json = json_write_minified(utest_fixture->value, &size);

  free(utest_fixture->value);

  utest_fixture->value = json_parse(json, size - 1);

  free(json);
}
