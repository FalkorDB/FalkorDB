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

#include "JSONTestSuite.inc"

struct JSONTestSuiteTests {
  const unsigned char *string;
  size_t length;
  ExpectedResult expected;
  bool skip;
};

UTEST_I_SETUP(JSONTestSuiteTests) {
  utest_fixture->string = JSONTestSuite[utest_index].string;
  utest_fixture->length = JSONTestSuite[utest_index].length;
  utest_fixture->expected = JSONTestSuite[utest_index].expected;
  utest_fixture->skip = false;

  switch (utest_index) {
  default:
    break;
  case 174:
    // 100,000 array openings
  case 200:
    // tons of open array objects
    utest_fixture->skip = true;
    break;
  case 333:
    // High continuation with no follow-up
  case 335:
    // Two high continuations
  case 337:
    // Three high continuations
    utest_fixture->expected = ExpectedResultFail;
    break;
  }
}

UTEST_I_TEARDOWN(JSONTestSuiteTests) {}

UTEST_I(JSONTestSuiteTests, all, JSONTESTSUITE_TESTS) {
  if (utest_fixture->skip) {
    return;
  }

  struct json_value_s *value =
      json_parse(utest_fixture->string, utest_fixture->length);

  switch (utest_fixture->expected) {
  case ExpectedResultPass:
    ASSERT_TRUE(value);
    break;
  case ExpectedResultFail:
    ASSERT_FALSE(value);
    break;
  case ExpectedResultUndefined:
    break;
  }

  free(value);
}
