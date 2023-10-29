/* vi:set ts=4 sw=4 expandtab:
 *
 * Copyright 2016, Chris Leishman (http://github.com/cleishm)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "../../config.h"
#include "../../lib/src/cypher-parser.h"
#include <check.h>
#include <errno.h>
#include <unistd.h>


#define MAX_SEGMENTS 8

static FILE *in;
static FILE *out;
static unsigned int nsegments;
static bool is_statement[MAX_SEGMENTS];
static char *segments[MAX_SEGMENTS];
static struct cypher_input_range ranges[MAX_SEGMENTS];
static bool eofs[MAX_SEGMENTS];


static void setup(void)
{
    int fds[2];
    ck_assert(pipe(fds) == 0);
    in = fdopen(fds[0], "r");
    ck_assert_ptr_ne(in, NULL);
    out = fdopen(fds[1], "w");
    ck_assert_ptr_ne(out, NULL);

    nsegments = 0;
    memset(is_statement, 0, sizeof(is_statement));
    memset(segments, 0, sizeof(segments));
    memset(ranges, 0, sizeof(ranges));
    memset(eofs, 0, sizeof(eofs));
}


static void teardown(void)
{
    if (out != NULL)
    {
        fclose(out);
    }
    fclose(in);

    for (unsigned int i = 0; i < nsegments; ++i)
    {
        free(segments[i]);
    }
}


static void fill_stream(const char *content)
{
    fputs(content, out);
    fclose(out);
    out = NULL;
}


static int segment_callback(void *data,
        const cypher_quick_parse_segment_t *segment)
{
    ck_assert(nsegments < MAX_SEGMENTS);
    is_statement[nsegments] = cypher_quick_parse_segment_is_statement(segment);
    size_t n;
    const char *s = cypher_quick_parse_segment_get_text(segment, &n);
    segments[nsegments] = strndup(s, n);
    ck_assert(segments[nsegments] != NULL);
    ranges[nsegments] = cypher_quick_parse_segment_get_range(segment);
    eofs[nsegments] = cypher_quick_parse_segment_is_eof(segment);
    ++nsegments;
    return 0;
}


START_TEST (fparse_none)
{
    fill_stream("");

    int result = cypher_quick_fparse(in, segment_callback, NULL, 0);
    ck_assert_int_eq(result, 0);
    ck_assert_int_eq(nsegments, 0);
}
END_TEST


START_TEST (fparse_empty_statement)
{
    fill_stream(";");

    int result = cypher_quick_fparse(in, segment_callback, NULL, 0);
    ck_assert_int_eq(result, 0);
    ck_assert_int_eq(nsegments, 1);

    ck_assert(is_statement[0]);
    ck_assert_str_eq(segments[0], "");
    ck_assert_int_eq(ranges[0].start.line, 1);
    ck_assert_int_eq(ranges[0].start.column, 1);
    ck_assert_int_eq(ranges[0].start.offset, 0);
    ck_assert_int_eq(ranges[0].end.line, 1);
    ck_assert_int_eq(ranges[0].end.column, 1);
    ck_assert_int_eq(ranges[0].end.offset, 0);
}
END_TEST


START_TEST (fparse_whitespace_statement)
{
    fill_stream("  ; ");

    int result = cypher_quick_fparse(in, segment_callback, NULL, 0);
    ck_assert_int_eq(result, 0);
    ck_assert_int_eq(nsegments, 2);

    ck_assert(is_statement[0]);
    ck_assert_str_eq(segments[0], "");
    ck_assert_int_eq(ranges[0].start.line, 1);
    ck_assert_int_eq(ranges[0].start.column, 3);
    ck_assert_int_eq(ranges[0].start.offset, 2);
    ck_assert_int_eq(ranges[0].end.line, 1);
    ck_assert_int_eq(ranges[0].end.column, 3);
    ck_assert_int_eq(ranges[0].end.offset, 2);

    ck_assert(is_statement[1]);
    ck_assert_str_eq(segments[1], "");
    ck_assert_int_eq(ranges[1].start.line, 1);
    ck_assert_int_eq(ranges[1].start.column, 5);
    ck_assert_int_eq(ranges[1].start.offset, 4);
    ck_assert_int_eq(ranges[1].end.line, 1);
    ck_assert_int_eq(ranges[1].end.column, 5);
    ck_assert_int_eq(ranges[1].end.offset, 4);
}
END_TEST


START_TEST (fparse_single)
{
    fill_stream("return 1;");

    int result = cypher_quick_fparse(in, segment_callback, NULL, 0);
    ck_assert_int_eq(result, 0);
    ck_assert_int_eq(nsegments, 1);

    ck_assert(is_statement[0]);
    ck_assert_str_eq(segments[0], "return 1");
    ck_assert_int_eq(ranges[0].start.line, 1);
    ck_assert_int_eq(ranges[0].start.column, 1);
    ck_assert_int_eq(ranges[0].start.offset, 0);
    ck_assert_int_eq(ranges[0].end.line, 1);
    ck_assert_int_eq(ranges[0].end.column, 9);
    ck_assert_int_eq(ranges[0].end.offset, 8);
    ck_assert(!eofs[0]);
}
END_TEST


START_TEST (fparse_multiple)
{
    fill_stream("return 1; return 2;\n   return 3    ;");

    int result = cypher_quick_fparse(in, segment_callback, NULL, 0);
    ck_assert_int_eq(result, 0);
    ck_assert_int_eq(nsegments, 3);

    ck_assert(is_statement[0]);
    ck_assert_str_eq(segments[0], "return 1");
    ck_assert_int_eq(ranges[0].start.line, 1);
    ck_assert_int_eq(ranges[0].start.column, 1);
    ck_assert_int_eq(ranges[0].start.offset, 0);
    ck_assert_int_eq(ranges[0].end.line, 1);
    ck_assert_int_eq(ranges[0].end.column, 9);
    ck_assert_int_eq(ranges[0].end.offset, 8);
    ck_assert(!eofs[0]);

    ck_assert(is_statement[1]);
    ck_assert_str_eq(segments[1], "return 2");
    ck_assert_int_eq(ranges[1].start.line, 1);
    ck_assert_int_eq(ranges[1].start.column, 11);
    ck_assert_int_eq(ranges[1].start.offset, 10);
    ck_assert_int_eq(ranges[1].end.line, 1);
    ck_assert_int_eq(ranges[1].end.column, 19);
    ck_assert_int_eq(ranges[1].end.offset, 18);
    ck_assert(!eofs[1]);

    ck_assert(is_statement[2]);
    ck_assert_str_eq(segments[2], "return 3");
    ck_assert_int_eq(ranges[2].start.line, 2);
    ck_assert_int_eq(ranges[2].start.column, 4);
    ck_assert_int_eq(ranges[2].start.offset, 23);
    ck_assert_int_eq(ranges[2].end.line, 2);
    ck_assert_int_eq(ranges[2].end.column, 12);
    ck_assert_int_eq(ranges[2].end.offset, 31);
    ck_assert(!eofs[2]);
}
END_TEST


START_TEST (fparse_commands)
{
    fill_stream(":foo bar\"baz\"\n");

    int result = cypher_quick_fparse(in, segment_callback, NULL, 0);
    ck_assert_int_eq(result, 0);
    ck_assert_int_eq(nsegments, 1);

    ck_assert(!is_statement[0]);
    ck_assert_str_eq(segments[0], ":foo bar\"baz\"");
    ck_assert_int_eq(ranges[0].start.line, 1);
    ck_assert_int_eq(ranges[0].start.column, 1);
    ck_assert_int_eq(ranges[0].start.offset, 0);
    ck_assert_int_eq(ranges[0].end.line, 1);
    ck_assert_int_eq(ranges[0].end.column, 14);
    ck_assert_int_eq(ranges[0].end.offset, 13);
    ck_assert(!eofs[0]);
}
END_TEST


START_TEST (fparse_statements_only)
{
    fill_stream("return 1; :foo bar\"baz\"\n return 2;");

    int result = cypher_quick_fparse(in, segment_callback, NULL,
            CYPHER_PARSE_ONLY_STATEMENTS);
    ck_assert_int_eq(result, 0);
    ck_assert_int_eq(nsegments, 2);

    ck_assert(is_statement[0]);
    ck_assert_str_eq(segments[0], "return 1");
    ck_assert_int_eq(ranges[0].start.line, 1);
    ck_assert_int_eq(ranges[0].start.column, 1);
    ck_assert_int_eq(ranges[0].start.offset, 0);
    ck_assert_int_eq(ranges[0].end.line, 1);
    ck_assert_int_eq(ranges[0].end.column, 9);
    ck_assert_int_eq(ranges[0].end.offset, 8);
    ck_assert(!eofs[0]);

    ck_assert(is_statement[1]);
    ck_assert_str_eq(segments[1], ":foo bar\"baz\"\n return 2");
    ck_assert_int_eq(ranges[1].start.line, 1);
    ck_assert_int_eq(ranges[1].start.column, 11);
    ck_assert_int_eq(ranges[1].start.offset, 10);
    ck_assert_int_eq(ranges[1].end.line, 2);
    ck_assert_int_eq(ranges[1].end.column, 10);
    ck_assert_int_eq(ranges[1].end.offset, 33);
    ck_assert(!eofs[1]);
}
END_TEST


START_TEST (fparse_eof_statement)
{
    fill_stream("return 1; return 2");

    int result = cypher_quick_fparse(in, segment_callback, NULL, 0);
    ck_assert_int_eq(result, 0);
    ck_assert_int_eq(nsegments, 2);

    ck_assert(is_statement[0]);
    ck_assert_str_eq(segments[0], "return 1");
    ck_assert_int_eq(ranges[0].start.line, 1);
    ck_assert_int_eq(ranges[0].start.column, 1);
    ck_assert_int_eq(ranges[0].start.offset, 0);
    ck_assert_int_eq(ranges[0].end.line, 1);
    ck_assert_int_eq(ranges[0].end.column, 9);
    ck_assert_int_eq(ranges[0].end.offset, 8);
    ck_assert(!eofs[0]);

    ck_assert(is_statement[1]);
    ck_assert_str_eq(segments[1], "return 2");
    ck_assert_int_eq(ranges[1].start.line, 1);
    ck_assert_int_eq(ranges[1].start.column, 11);
    ck_assert_int_eq(ranges[1].start.offset, 10);
    ck_assert_int_eq(ranges[1].end.line, 1);
    ck_assert_int_eq(ranges[1].end.column, 19);
    ck_assert_int_eq(ranges[1].end.offset, 18);
    ck_assert(eofs[1]);
}
END_TEST


START_TEST (fparse_eof_command)
{
    fill_stream(":bar\n:foo bar\"baz\"");

    int result = cypher_quick_fparse(in, segment_callback, NULL, 0);
    ck_assert_int_eq(result, 0);
    ck_assert_int_eq(nsegments, 2);

    ck_assert(!is_statement[0]);
    ck_assert_str_eq(segments[0], ":bar");
    ck_assert_int_eq(ranges[0].start.line, 1);
    ck_assert_int_eq(ranges[0].start.column, 1);
    ck_assert_int_eq(ranges[0].start.offset, 0);
    ck_assert_int_eq(ranges[0].end.line, 1);
    ck_assert_int_eq(ranges[0].end.column, 5);
    ck_assert_int_eq(ranges[0].end.offset, 4);
    ck_assert(!eofs[0]);

    ck_assert(!is_statement[1]);
    ck_assert_str_eq(segments[1], ":foo bar\"baz\"");
    ck_assert_int_eq(ranges[1].start.line, 2);
    ck_assert_int_eq(ranges[1].start.column, 1);
    ck_assert_int_eq(ranges[1].start.offset, 5);
    ck_assert_int_eq(ranges[1].end.line, 2);
    ck_assert_int_eq(ranges[1].end.column, 14);
    ck_assert_int_eq(ranges[1].end.offset, 18);
    ck_assert(eofs[1]);
}
END_TEST


TCase* quick_fparse_tcase(void)
{
    TCase *tc = tcase_create("quick_fparse");
    tcase_add_checked_fixture(tc, setup, teardown);
    tcase_add_test(tc, fparse_none);
    tcase_add_test(tc, fparse_empty_statement);
    tcase_add_test(tc, fparse_whitespace_statement);
    tcase_add_test(tc, fparse_single);
    tcase_add_test(tc, fparse_multiple);
    tcase_add_test(tc, fparse_commands);
    tcase_add_test(tc, fparse_statements_only);
    tcase_add_test(tc, fparse_eof_statement);
    tcase_add_test(tc, fparse_eof_command);
    return tc;
}
