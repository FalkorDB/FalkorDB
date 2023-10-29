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
#include "memstream.h"
#include <check.h>
#include <errno.h>
#include <unistd.h>


static cypher_parse_result_t *result;
static char *memstream_buffer;
static size_t memstream_size;
static FILE *memstream;


static void setup(void)
{
    result = NULL;
    memstream = open_memstream(&memstream_buffer, &memstream_size);
    fputc('\n', memstream);
}


static void teardown(void)
{
    cypher_parse_result_free(result);
    fclose(memstream);
    free(memstream_buffer);
}


START_TEST (empty_input)
{
    struct cypher_input_position last = cypher_input_position_zero;
    result = cypher_parse(" ", &last, NULL, 0);
    ck_assert_ptr_ne(result, NULL);
    ck_assert_int_eq(last.offset, 1);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n";
    ck_assert_str_eq(memstream_buffer, expected);

    ck_assert_int_eq(cypher_parse_result_ndirectives(result), 0);
    ck_assert_int_eq(cypher_parse_result_nerrors(result), 0);
    ck_assert(!cypher_parse_result_eof(result));
}
END_TEST


START_TEST (empty_input_with_terminator)
{
    struct cypher_input_position last = cypher_input_position_zero;
    result = cypher_parse(";", &last, NULL, 0);
    ck_assert_ptr_ne(result, NULL);
    ck_assert_int_eq(last.offset, 1);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n";
    ck_assert_str_eq(memstream_buffer, expected);

    ck_assert_int_eq(cypher_parse_result_ndirectives(result), 0);
    ck_assert_int_eq(cypher_parse_result_nerrors(result), 0);
    ck_assert(!cypher_parse_result_eof(result));
}
END_TEST


START_TEST (comment_only)
{
    struct cypher_input_position last = cypher_input_position_zero;
    result = cypher_parse("/*foo*/", &last, NULL, 0);
    ck_assert_ptr_ne(result, NULL);
    ck_assert_int_eq(last.offset, 7);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
"@0  2..5  block_comment  /*foo*/\n";
    ck_assert_str_eq(memstream_buffer, expected);

    ck_assert_int_eq(cypher_parse_result_ndirectives(result), 0);
    ck_assert_int_eq(cypher_parse_result_nerrors(result), 0);
    ck_assert(!cypher_parse_result_eof(result));
}
END_TEST


START_TEST (comment_only_with_terminator)
{
    struct cypher_input_position last = cypher_input_position_zero;
    result = cypher_parse("/*foo*/;", &last, NULL, 0);
    ck_assert_ptr_ne(result, NULL);
    ck_assert_int_eq(last.offset, 8);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
"@0  2..5  block_comment  /*foo*/\n";
    ck_assert_str_eq(memstream_buffer, expected);

    ck_assert_int_eq(cypher_parse_result_ndirectives(result), 0);
    ck_assert_int_eq(cypher_parse_result_nerrors(result), 0);
    ck_assert(!cypher_parse_result_eof(result));
}
END_TEST


START_TEST (error_only)
{
    struct cypher_input_position last = cypher_input_position_zero;
    result = cypher_parse("foo", &last, NULL, 0);
    ck_assert_ptr_ne(result, NULL);
    ck_assert_int_eq(last.offset, 3);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
"@0  0..3  error  >>foo<<\n";
    ck_assert_str_eq(memstream_buffer, expected);

    ck_assert_int_eq(cypher_parse_result_ndirectives(result), 0);
    ck_assert_int_eq(cypher_parse_result_nerrors(result), 1);
    ck_assert(cypher_parse_result_eof(result));
}
END_TEST


START_TEST (error_only_with_terminator)
{
    struct cypher_input_position last = cypher_input_position_zero;
    result = cypher_parse("foo;", &last, NULL, 0);
    ck_assert_ptr_ne(result, NULL);
    ck_assert_int_eq(last.offset, 4);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
"@0  0..3  error  >>foo<<\n";
    ck_assert_str_eq(memstream_buffer, expected);

    ck_assert_int_eq(cypher_parse_result_ndirectives(result), 0);
    ck_assert_int_eq(cypher_parse_result_nerrors(result), 1);
    ck_assert(!cypher_parse_result_eof(result));
}
END_TEST


START_TEST (single_statement)
{
    struct cypher_input_position last = cypher_input_position_zero;
    result = cypher_parse("return 1", &last, NULL, 0);
    ck_assert_ptr_ne(result, NULL);
    ck_assert_int_eq(last.offset, 8);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
"@0  0..8  statement           body=@1\n"
"@1  0..8  > query             clauses=[@2]\n"
"@2  0..8  > > RETURN          projections=[@3]\n"
"@3  7..8  > > > projection    expression=@4, alias=@5\n"
"@4  7..8  > > > > integer     1\n"
"@5  7..8  > > > > identifier  `1`\n";
    ck_assert_str_eq(memstream_buffer, expected);

    ck_assert_int_eq(cypher_parse_result_ndirectives(result), 1);
    ck_assert_int_eq(cypher_parse_result_nerrors(result), 0);
    ck_assert(cypher_parse_result_eof(result));
}
END_TEST


START_TEST (single_statement_with_terminator)
{
    struct cypher_input_position last = cypher_input_position_zero;
    result = cypher_parse("return 1;", &last, NULL, 0);
    ck_assert_ptr_ne(result, NULL);
    ck_assert_int_eq(last.offset, 9);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
"@0  0..9  statement           body=@1\n"
"@1  0..9  > query             clauses=[@2]\n"
"@2  0..8  > > RETURN          projections=[@3]\n"
"@3  7..8  > > > projection    expression=@4, alias=@5\n"
"@4  7..8  > > > > integer     1\n"
"@5  7..8  > > > > identifier  `1`\n";
    ck_assert_str_eq(memstream_buffer, expected);

    ck_assert_int_eq(cypher_parse_result_ndirectives(result), 1);
    ck_assert_int_eq(cypher_parse_result_nerrors(result), 0);
    ck_assert(!cypher_parse_result_eof(result));
}
END_TEST


START_TEST (single_command)
{
    struct cypher_input_position last = cypher_input_position_zero;
    result = cypher_parse(":foo bar", &last, NULL, 0);
    ck_assert_ptr_ne(result, NULL);
    ck_assert_int_eq(last.offset, 8);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
"@0  0..8  command   name=@1, args=[@2]\n"
"@1  1..4  > string  \"foo\"\n"
"@2  5..8  > string  \"bar\"\n";
    ck_assert_str_eq(memstream_buffer, expected);

    ck_assert_int_eq(cypher_parse_result_ndirectives(result), 1);
    ck_assert_int_eq(cypher_parse_result_nerrors(result), 0);
    ck_assert(cypher_parse_result_eof(result));
}
END_TEST


START_TEST (single_command_with_terminator)
{
    struct cypher_input_position last = cypher_input_position_zero;
    result = cypher_parse(":foo bar\n", &last, NULL, 0);
    ck_assert_ptr_ne(result, NULL);
    ck_assert_int_eq(last.offset, 9);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
"@0  0..8  command   name=@1, args=[@2]\n"
"@1  1..4  > string  \"foo\"\n"
"@2  5..8  > string  \"bar\"\n";
    ck_assert_str_eq(memstream_buffer, expected);

    ck_assert_int_eq(cypher_parse_result_ndirectives(result), 1);
    ck_assert_int_eq(cypher_parse_result_nerrors(result), 0);
    ck_assert(!cypher_parse_result_eof(result));
}
END_TEST


START_TEST (single_command_ending_in_line_comment)
{
    struct cypher_input_position last = cypher_input_position_zero;
    result = cypher_parse(":foo bar // baz", &last, NULL, 0);
    ck_assert_ptr_ne(result, NULL);
    ck_assert_int_eq(last.offset, 15);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
"@0   0..9   command       name=@1, args=[@2]\n"
"@1   1..4   > string      \"foo\"\n"
"@2   5..8   > string      \"bar\"\n"
"@3  11..15  line_comment  // baz\n";
    ck_assert_str_eq(memstream_buffer, expected);

    ck_assert_int_eq(cypher_parse_result_ndirectives(result), 1);
    ck_assert_int_eq(cypher_parse_result_nerrors(result), 0);
    ck_assert(cypher_parse_result_eof(result));
}
END_TEST


START_TEST (single_command_ending_in_line_comment_with_terminator)
{
    struct cypher_input_position last = cypher_input_position_zero;
    result = cypher_parse(":foo bar // baz\n", &last, NULL, 0);
    ck_assert_ptr_ne(result, NULL);
    ck_assert_int_eq(last.offset, 16);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
"@0   0..9   command       name=@1, args=[@2]\n"
"@1   1..4   > string      \"foo\"\n"
"@2   5..8   > string      \"bar\"\n"
"@3  11..15  line_comment  // baz\n";
    ck_assert_str_eq(memstream_buffer, expected);

    ck_assert_int_eq(cypher_parse_result_ndirectives(result), 1);
    ck_assert_int_eq(cypher_parse_result_nerrors(result), 0);
    ck_assert(!cypher_parse_result_eof(result));
}
END_TEST


START_TEST (mixed_ending_in_empty)
{
    struct cypher_input_position last = cypher_input_position_zero;
    result = cypher_parse("return 1; return 2; ; ;", &last, NULL, 0);
    ck_assert_ptr_ne(result, NULL);
    ck_assert_int_eq(last.offset, 23);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
" @0   0..9   statement           body=@1\n"
" @1   0..9   > query             clauses=[@2]\n"
" @2   0..8   > > RETURN          projections=[@3]\n"
" @3   7..8   > > > projection    expression=@4, alias=@5\n"
" @4   7..8   > > > > integer     1\n"
" @5   7..8   > > > > identifier  `1`\n"
" @6  10..19  statement           body=@7\n"
" @7  10..19  > query             clauses=[@8]\n"
" @8  10..18  > > RETURN          projections=[@9]\n"
" @9  17..18  > > > projection    expression=@10, alias=@11\n"
"@10  17..18  > > > > integer     2\n"
"@11  17..18  > > > > identifier  `2`\n";
    ck_assert_str_eq(memstream_buffer, expected);

    ck_assert_int_eq(cypher_parse_result_ndirectives(result), 2);
    ck_assert_int_eq(cypher_parse_result_nerrors(result), 0);
    ck_assert(!cypher_parse_result_eof(result));
}
END_TEST


START_TEST (mixed_ending_in_comment)
{
    struct cypher_input_position last = cypher_input_position_zero;
    result = cypher_parse("return 1; return 2; ; // foo", &last, NULL, 0);
    ck_assert_ptr_ne(result, NULL);
    ck_assert_int_eq(last.offset, 28);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
" @0   0..9   statement           body=@1\n"
" @1   0..9   > query             clauses=[@2]\n"
" @2   0..8   > > RETURN          projections=[@3]\n"
" @3   7..8   > > > projection    expression=@4, alias=@5\n"
" @4   7..8   > > > > integer     1\n"
" @5   7..8   > > > > identifier  `1`\n"
" @6  10..19  statement           body=@7\n"
" @7  10..19  > query             clauses=[@8]\n"
" @8  10..18  > > RETURN          projections=[@9]\n"
" @9  17..18  > > > projection    expression=@10, alias=@11\n"
"@10  17..18  > > > > integer     2\n"
"@11  17..18  > > > > identifier  `2`\n"
"@12  24..28  line_comment        // foo\n";
    ck_assert_str_eq(memstream_buffer, expected);

    ck_assert_int_eq(cypher_parse_result_ndirectives(result), 2);
    ck_assert_int_eq(cypher_parse_result_nerrors(result), 0);
    ck_assert(!cypher_parse_result_eof(result));
}
END_TEST


START_TEST (mixed_ending_in_error)
{
    struct cypher_input_position last = cypher_input_position_zero;
    result = cypher_parse("return 1; return 2; ; foo", &last, NULL, 0);
    ck_assert_ptr_ne(result, NULL);
    ck_assert_int_eq(last.offset, 25);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
" @0   0..9   statement           body=@1\n"
" @1   0..9   > query             clauses=[@2]\n"
" @2   0..8   > > RETURN          projections=[@3]\n"
" @3   7..8   > > > projection    expression=@4, alias=@5\n"
" @4   7..8   > > > > integer     1\n"
" @5   7..8   > > > > identifier  `1`\n"
" @6  10..19  statement           body=@7\n"
" @7  10..19  > query             clauses=[@8]\n"
" @8  10..18  > > RETURN          projections=[@9]\n"
" @9  17..18  > > > projection    expression=@10, alias=@11\n"
"@10  17..18  > > > > integer     2\n"
"@11  17..18  > > > > identifier  `2`\n"
"@12  22..25  error               >>foo<<\n";
    ck_assert_str_eq(memstream_buffer, expected);

    ck_assert_int_eq(cypher_parse_result_ndirectives(result), 2);
    ck_assert_int_eq(cypher_parse_result_nerrors(result), 1);
    ck_assert(cypher_parse_result_eof(result));
}
END_TEST


START_TEST (mixed_ending_in_unterminated)
{
    struct cypher_input_position last = cypher_input_position_zero;
    result = cypher_parse("return 1; return 2; return 3", &last, NULL, 0);
    ck_assert_ptr_ne(result, NULL);
    ck_assert_int_eq(last.offset, 28);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
" @0   0..9   statement           body=@1\n"
" @1   0..9   > query             clauses=[@2]\n"
" @2   0..8   > > RETURN          projections=[@3]\n"
" @3   7..8   > > > projection    expression=@4, alias=@5\n"
" @4   7..8   > > > > integer     1\n"
" @5   7..8   > > > > identifier  `1`\n"
" @6  10..19  statement           body=@7\n"
" @7  10..19  > query             clauses=[@8]\n"
" @8  10..18  > > RETURN          projections=[@9]\n"
" @9  17..18  > > > projection    expression=@10, alias=@11\n"
"@10  17..18  > > > > integer     2\n"
"@11  17..18  > > > > identifier  `2`\n"
"@12  20..28  statement           body=@13\n"
"@13  20..28  > query             clauses=[@14]\n"
"@14  20..28  > > RETURN          projections=[@15]\n"
"@15  27..28  > > > projection    expression=@16, alias=@17\n"
"@16  27..28  > > > > integer     3\n"
"@17  27..28  > > > > identifier  `3`\n";
    ck_assert_str_eq(memstream_buffer, expected);

    ck_assert_int_eq(cypher_parse_result_ndirectives(result), 3);
    ck_assert_int_eq(cypher_parse_result_nerrors(result), 0);
    ck_assert(cypher_parse_result_eof(result));
}
END_TEST


TCase* eof_tcase(void)
{
    TCase *tc = tcase_create("eof");
    tcase_add_checked_fixture(tc, setup, teardown);
    tcase_add_test(tc, empty_input);
    tcase_add_test(tc, empty_input_with_terminator);
    tcase_add_test(tc, comment_only);
    tcase_add_test(tc, comment_only_with_terminator);
    tcase_add_test(tc, error_only);
    tcase_add_test(tc, error_only_with_terminator);
    tcase_add_test(tc, single_statement);
    tcase_add_test(tc, single_statement_with_terminator);
    tcase_add_test(tc, single_command);
    tcase_add_test(tc, single_command_with_terminator);
    tcase_add_test(tc, single_command_ending_in_line_comment);
    tcase_add_test(tc, single_command_ending_in_line_comment_with_terminator);
    tcase_add_test(tc, mixed_ending_in_empty);
    tcase_add_test(tc, mixed_ending_in_comment);
    tcase_add_test(tc, mixed_ending_in_error);
    tcase_add_test(tc, mixed_ending_in_unterminated);
    return tc;
}
