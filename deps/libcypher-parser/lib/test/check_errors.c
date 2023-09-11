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


START_TEST (parse_unterminated_string)
{
    struct cypher_input_position last = cypher_input_position_zero;
    result = cypher_parse("RETURN 'foo", &last, NULL, 0);
    ck_assert_ptr_ne(result, NULL);
    ck_assert_int_eq(last.offset, 11);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
"@0  0..11  error  >>RETURN 'foo<<\n";
    ck_assert_str_eq(memstream_buffer, expected);

    ck_assert_int_eq(cypher_parse_result_ndirectives(result), 0);
    ck_assert_ptr_eq(cypher_parse_result_get_directive(result, 0), NULL);

    ck_assert_int_eq(cypher_parse_result_nerrors(result), 1);
    const cypher_parse_error_t *err = cypher_parse_result_get_error(result, 0);
    struct cypher_input_position pos = cypher_parse_error_position(err);
    ck_assert_int_eq(pos.line, 1);
    ck_assert_int_eq(pos.column, 12);
    ck_assert_int_eq(pos.offset, 11);

    ck_assert_str_eq(cypher_parse_error_message(err),
            "Invalid input at end of input: expected '");
}
END_TEST


START_TEST (parse_invalid_directive)
{
    struct cypher_input_position last = cypher_input_position_zero;
    result = cypher_parse("RETURN 1; [1,2,3]", &last, NULL, 0);
    ck_assert_ptr_ne(result, NULL);
    ck_assert_int_eq(last.offset, 17);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
"@0   0..9   statement           body=@1\n"
"@1   0..9   > query             clauses=[@2]\n"
"@2   0..8   > > RETURN          projections=[@3]\n"
"@3   7..8   > > > projection    expression=@4, alias=@5\n"
"@4   7..8   > > > > integer     1\n"
"@5   7..8   > > > > identifier  `1`\n"
"@6  10..17  error               >>[1,2,3]<<\n";
    ck_assert_str_eq(memstream_buffer, expected);

    ck_assert_int_eq(cypher_parse_result_ndirectives(result), 1);

    ck_assert_int_eq(cypher_parse_result_nerrors(result), 1);
    const cypher_parse_error_t *err = cypher_parse_result_get_error(result, 0);
    struct cypher_input_position pos = cypher_parse_error_position(err);
    ck_assert_int_eq(pos.line, 1);
    ck_assert_int_eq(pos.column, 11);
    ck_assert_int_eq(pos.offset, 10);

    ck_assert_str_eq(cypher_parse_error_message(err),
            "Invalid input '[': expected ';', ':', a statement option, a query hint, a clause or a schema command");
}
END_TEST


START_TEST (parse_invalid_clause)
{
    struct cypher_input_position last = cypher_input_position_zero;
    result = cypher_parse(
            "MATCH (n)\n"
            "[1,2,3]\n"
            "RETURN n",
            &last, NULL, 0);
    ck_assert_ptr_ne(result, NULL);
    ck_assert_int_eq(last.offset, 26);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
" @0   0..26  statement               body=@1\n"
" @1   0..26  > query                 clauses=[@2, @8]\n"
" @2   0..10  > > MATCH               pattern=@3\n"
" @3   6..9   > > > pattern           paths=[@4]\n"
" @4   6..9   > > > > pattern path    (@5)\n"
" @5   6..9   > > > > > node pattern  (@6)\n"
" @6   7..8   > > > > > > identifier  `n`\n"
" @7  10..17  > > error               >>[1,2,3]<<\n"
" @8  18..26  > > RETURN              projections=[@9]\n"
" @9  25..26  > > > projection        expression=@10\n"
"@10  25..26  > > > > identifier      `n`\n";
    ck_assert_str_eq(memstream_buffer, expected);

    ck_assert_int_eq(cypher_parse_result_ndirectives(result), 1);

    ck_assert_int_eq(cypher_parse_result_nerrors(result), 1);
    const cypher_parse_error_t *err = cypher_parse_result_get_error(result, 0);
    struct cypher_input_position pos = cypher_parse_error_position(err);
    ck_assert_int_eq(pos.line, 2);
    ck_assert_int_eq(pos.column, 1);
    ck_assert_int_eq(pos.offset, 10);

    ck_assert_str_eq(cypher_parse_error_message(err),
            "Invalid input '[': expected '<', '-', USING INDEX, USING JOIN ON, USING SCAN, WHERE, ';' or a clause");
}
END_TEST


START_TEST (parse_invalid_query_and_resync)
{
    struct cypher_input_position last = cypher_input_position_zero;
    result = cypher_parse("MATCH n; ; MATCH (n) RETURN n;", &last, NULL, 0);
    ck_assert_ptr_ne(result, NULL);
    ck_assert_int_eq(last.offset, 30);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
" @0   0..7   error                   >>MATCH n<<\n"
" @1  11..30  statement               body=@2\n"
" @2  11..30  > query                 clauses=[@3, @8]\n"
" @3  11..21  > > MATCH               pattern=@4\n"
" @4  17..20  > > > pattern           paths=[@5]\n"
" @5  17..20  > > > > pattern path    (@6)\n"
" @6  17..20  > > > > > node pattern  (@7)\n"
" @7  18..19  > > > > > > identifier  `n`\n"
" @8  21..29  > > RETURN              projections=[@9]\n"
" @9  28..29  > > > projection        expression=@10\n"
"@10  28..29  > > > > identifier      `n`\n";
    ck_assert_str_eq(memstream_buffer, expected);

    ck_assert_int_eq(cypher_parse_result_ndirectives(result), 1);
}
END_TEST


START_TEST (parse_single_invalid_query)
{
    struct cypher_input_position last = cypher_input_position_zero;
    result = cypher_parse("MATCH n; ; MATCH (n) RETURN n;",
            &last, NULL, CYPHER_PARSE_SINGLE);
    ck_assert_ptr_ne(result, NULL);
    ck_assert_int_eq(last.offset, 8);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
"@0  0..7  error  >>MATCH n<<\n";
    ck_assert_str_eq(memstream_buffer, expected);

    ck_assert_int_eq(cypher_parse_result_ndirectives(result), 0);
    ck_assert(!cypher_parse_result_eof(result));
}
END_TEST


START_TEST (track_error_position_over_embedded_newline)
{
    struct cypher_input_position last = cypher_input_position_zero;
    result = cypher_parse("RETURN \"foo\nbar\" MA TCH (n)",
            &last, NULL, CYPHER_PARSE_SINGLE);
    ck_assert_ptr_ne(result, NULL);
    ck_assert_int_eq(last.offset, 27);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
"@0   0..27  statement           body=@1\n"
"@1   0..27  > query             clauses=[@2]\n"
"@2   0..17  > > RETURN          projections=[@3]\n"
"@3   7..17  > > > projection    expression=@4, alias=@5\n"
"@4   7..16  > > > > string      \"foo\\nbar\"\n"
"@5   7..17  > > > > identifier  `\"foo\\nbar\"`\n"
"@6  17..27  > > error           >>MA TCH (n)<<\n";
    ck_assert_str_eq(memstream_buffer, expected);

    ck_assert_int_eq(cypher_parse_result_nerrors(result), 1);
    const cypher_parse_error_t *err = cypher_parse_result_get_error(result, 0);
    struct cypher_input_position pos = cypher_parse_error_position(err);
    ck_assert_int_eq(pos.line, 2);
    ck_assert_int_eq(pos.column, 8);
    ck_assert_int_eq(pos.offset, 19);

    ck_assert_str_eq(cypher_parse_error_message(err), 
            "Invalid input ' ': expected MATCH");
}
END_TEST


START_TEST (track_error_position_across_statements)
{
    struct cypher_input_position last = cypher_input_position_zero;
    result = cypher_parse(
            "match (n) return n;\n"
            "m atch (n) return n;\n"
            "match (d) return d;\n"
            "match (bc e) return a;\n",
            &last, NULL, 0);
    ck_assert_ptr_ne(result, NULL);
    ck_assert_int_eq(last.offset, 84);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
" @0   0..19  statement               body=@1\n"
" @1   0..19  > query                 clauses=[@2, @7]\n"
" @2   0..10  > > MATCH               pattern=@3\n"
" @3   6..9   > > > pattern           paths=[@4]\n"
" @4   6..9   > > > > pattern path    (@5)\n"
" @5   6..9   > > > > > node pattern  (@6)\n"
" @6   7..8   > > > > > > identifier  `n`\n"
" @7  10..18  > > RETURN              projections=[@8]\n"
" @8  17..18  > > > projection        expression=@9\n"
" @9  17..18  > > > > identifier      `n`\n"
"@10  20..30  error                   >>m atch (n)<<\n"
"@11  30..40  statement               body=@12\n"
"@12  31..40  > query                 clauses=[@13]\n"
"@13  31..39  > > RETURN              projections=[@14]\n"
"@14  38..39  > > > projection        expression=@15\n"
"@15  38..39  > > > > identifier      `n`\n"
"@16  41..60  statement               body=@17\n"
"@17  41..60  > query                 clauses=[@18, @23]\n"
"@18  41..51  > > MATCH               pattern=@19\n"
"@19  47..50  > > > pattern           paths=[@20]\n"
"@20  47..50  > > > > pattern path    (@21)\n"
"@21  47..50  > > > > > node pattern  (@22)\n"
"@22  48..49  > > > > > > identifier  `d`\n"
"@23  51..59  > > RETURN              projections=[@24]\n"
"@24  58..59  > > > projection        expression=@25\n"
"@25  58..59  > > > > identifier      `d`\n"
"@26  61..73  error                   >>match (bc e)<<\n"
"@27  73..83  statement               body=@28\n"
"@28  74..83  > query                 clauses=[@29]\n"
"@29  74..82  > > RETURN              projections=[@30]\n"
"@30  81..82  > > > projection        expression=@31\n"
"@31  81..82  > > > > identifier      `a`\n";
    ck_assert_str_eq(memstream_buffer, expected);

    ck_assert_int_eq(cypher_parse_result_nerrors(result), 2);

    const cypher_parse_error_t *err = cypher_parse_result_get_error(result, 0);
    struct cypher_input_position pos = cypher_parse_error_position(err);
    ck_assert_int_eq(pos.line, 2);
    ck_assert_int_eq(pos.column, 2);
    ck_assert_int_eq(pos.offset, 21);
    ck_assert_str_eq(cypher_parse_error_message(err),
            "Invalid input ' ': expected MATCH or MERGE");

    err = cypher_parse_result_get_error(result, 1);
    pos = cypher_parse_error_position(err);
    ck_assert_int_eq(pos.line, 4);
    ck_assert_int_eq(pos.column, 11);
    ck_assert_int_eq(pos.offset, 71);
    ck_assert_str_eq(cypher_parse_error_message(err),
            "Invalid input 'e': expected a label, '{', a parameter or ')'");
}
END_TEST


TCase* errors_tcase(void)
{
    TCase *tc = tcase_create("errors");
    tcase_add_checked_fixture(tc, setup, teardown);
    tcase_add_test(tc, parse_unterminated_string);
    tcase_add_test(tc, parse_invalid_directive);
    tcase_add_test(tc, parse_invalid_clause);
    tcase_add_test(tc, parse_invalid_query_and_resync);
    tcase_add_test(tc, parse_single_invalid_query);
    tcase_add_test(tc, track_error_position_over_embedded_newline);
    tcase_add_test(tc, track_error_position_across_statements);
    return tc;
}
