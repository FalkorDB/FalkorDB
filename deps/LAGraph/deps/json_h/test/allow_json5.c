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

UTEST(allow_json5, example1) {
  const char payload[] = "{\n"
                         "  foo: 'bar',\n"
                         "  while: true,\n"
                         "\n"
                         "  this: 'is a \\n"
                         "  multi-line string',\n"
                         "\n"
                         "  // this is an inline comment\n"
                         "  here: 'is another', // inline comment\n"
                         "\n"
                         "  /* this is a block comment\n"
                         "     that continues on another line */\n"
                         "\n"
                         "  hex: 0xDEADbeef,\n"
                         "  half: .5,\n"
                         "  delta: +10,\n"
                         "  to: Infinity,   // and beyond!\n"
                         "\n"
                         "  finally: 'a trailing comma',\n"
                         "  oh: [\n"
                         "        \"we shouldn't forget\",\n"
                         "        'arrays can have',\n"
                         "        'trailing commas too',\n"
                         "      ],\n"
                         "}";
  struct json_value_s *value = json_parse_ex(
      payload, strlen(payload), json_parse_flags_allow_json5, 0, 0, 0);
  struct json_object_s *object = 0;
  struct json_value_s *value2 = 0;
  struct json_string_s *string = 0;

  ASSERT_TRUE(value);

  free(value);
}

UTEST(allow_json5, example2) {
  const char payload[] =
      "// This file is written in JSON5 syntax, naturally, but npm needs a "
      "regular\n"
      "// JSON file, so compile via `npm run build`. Be sure to keep both in "
      "sync!\n"
      "\n"
      "{\n"
      "    name: 'json5',\n"
      "    version: '0.5.0',\n"
      "    description: 'JSON for the ES5 era.',\n"
      "    keywords: ['json', 'es5'],\n"
      "    author: 'Aseem Kishore <aseem.kishore@gmail.com>',\n"
      "    contributors: [\n"
      "        // todo: Should we remove this section in favor of GitHub's "
      "list?\n"
      "        // https://github.com/json5/json5/contributors\n"
      "        'Max Nanasy <max.nanasy@gmail.com>',\n"
      "        'Andrew Eisenberg <andrew@eisenberg.as>',\n"
      "        'Jordan Tucker <jordanbtucker@gmail.com>',\n"
      "    ],\n"
      "    main: 'lib/json5.js',\n"
      "    bin: 'lib/cli.js',\n"
      "    files: [\"lib/\"],\n"
      "    dependencies: {},\n"
      "    devDependencies: {\n"
      "        gulp: \"^3.9.1\",\n"
      "        'gulp-jshint': \"^2.0.0\",\n"
      "        jshint: \"^2.9.1\",\n"
      "        'jshint-stylish': \"^2.1.0\",\n"
      "        mocha: \"^2.4.5\"\n"
      "    },\n"
      "    scripts: {\n"
      "        build: 'node ./lib/cli.js -c package.json5',\n"
      "        test: 'mocha --ui exports --reporter spec',\n"
      "            // todo: Would it be better to define these in a mocha.opts "
      "file?\n"
      "    },\n"
      "    homepage: 'http://json5.org/',\n"
      "    license: 'MIT',\n"
      "    repository: {\n"
      "        type: 'git',\n"
      "        url: 'https://github.com/json5/json5',\n"
      "    },\n"
      "}\n";
  struct json_value_s *value = json_parse_ex(
      payload, strlen(payload), json_parse_flags_allow_json5, 0, 0, 0);
  struct json_object_s *object = 0;
  struct json_value_s *value2 = 0;
  struct json_string_s *string = 0;

  ASSERT_TRUE(value);

  free(value);
}

struct allow_json5 {
  struct json_value_s *value;
};

UTEST_F_SETUP(allow_json5) {
  const char payload[] =
      "// This file is written in JSON5 syntax, naturally, but npm needs a "
      "regular\n"
      "// JSON file, so compile via `npm run build`. Be sure to keep both in "
      "sync!\n"
      "\n"
      "{\n"
      "    name: 'json5',\n"
      "    version: '0.5.0',\n"
      "    description: 'JSON for the ES5 era.',\n"
      "    keywords: ['json', 'es5'],\n"
      "    author: 'Aseem Kishore <aseem.kishore@gmail.com>',\n"
      "    contributors: [\n"
      "        // todo: Should we remove this section in favor of GitHub's "
      "list?\n"
      "        // https://github.com/json5/json5/contributors\n"
      "        'Max Nanasy <max.nanasy@gmail.com>',\n"
      "        'Andrew Eisenberg <andrew@eisenberg.as>',\n"
      "        'Jordan Tucker <jordanbtucker@gmail.com>',\n"
      "    ],\n"
      "    main: 'lib/json5.js',\n"
      "    bin: 'lib/cli.js',\n"
      "    files: [\"lib/\"],\n"
      "    dependencies: {},\n"
      "    devDependencies: {\n"
      "        gulp: \"^3.9.1\",\n"
      "        'gulp-jshint': \"^2.0.0\",\n"
      "        jshint: \"^2.9.1\",\n"
      "        'jshint-stylish': \"^2.1.0\",\n"
      "        mocha: \"^2.4.5\"\n"
      "    },\n"
      "    scripts: {\n"
      "        build: 'node ./lib/cli.js -c package.json5',\n"
      "        test: 'mocha --ui exports --reporter spec',\n"
      "            // todo: Would it be better to define these in a mocha.opts "
      "file?\n"
      "    },\n"
      "    homepage: 'http://json5.org/',\n"
      "    license: 'MIT',\n"
      "    repository: {\n"
      "        type: 'git',\n"
      "        url: 'https://github.com/json5/json5',\n"
      "    },\n"
      "}\n";
  utest_fixture->value = json_parse_ex(payload, strlen(payload),
                                       json_parse_flags_allow_json5, 0, 0, 0);

  ASSERT_TRUE(utest_fixture->value);
}

UTEST_F_TEARDOWN(allow_json5) {
  ASSERT_TRUE(utest_fixture->value);
  free(utest_fixture->value);
}

UTEST_F(allow_json5, read_write_pretty_read) {
  size_t size = 0;
  void *json = json_write_pretty(utest_fixture->value, "  ", "\n", &size);

  free(utest_fixture->value);

  utest_fixture->value = json_parse(json, size - 1);

  free(json);
}

UTEST_F(allow_json5, read_write_minified_read) {
  size_t size = 0;
  void *json = json_write_minified(utest_fixture->value, &size);

  free(utest_fixture->value);

  utest_fixture->value = json_parse(json, size - 1);

  free(json);
}
