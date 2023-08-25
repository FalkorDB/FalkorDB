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


START_TEST (parse_simple_call)
{
    struct cypher_input_position last = cypher_input_position_zero;
    result = cypher_parse("CALL foo. bar.baz();", &last, NULL, 0);
    ck_assert_ptr_ne(result, NULL);
    ck_assert_int_eq(last.offset, 20);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
"@0  0..20  statement        body=@1\n"
"@1  0..20  > query          clauses=[@2]\n"
"@2  0..19  > > CALL         name=@3\n"
"@3  5..17  > > > proc name  `foo.bar.baz`\n";
    ck_assert_str_eq(memstream_buffer, expected);

    const cypher_astnode_t *ast = cypher_parse_result_get_directive(result, 0);
    const cypher_astnode_t *query = cypher_ast_statement_get_body(ast);
    const cypher_astnode_t *clause = cypher_ast_query_get_clause(query, 0);
    ck_assert_int_eq(cypher_astnode_type(clause), CYPHER_AST_CALL);

    const cypher_astnode_t *proc = cypher_ast_call_get_proc_name(clause);
    ck_assert_int_eq(cypher_astnode_type(proc), CYPHER_AST_PROC_NAME);
    ck_assert_str_eq(cypher_ast_proc_name_get_value(proc), "foo.bar.baz");

    ck_assert_int_eq(cypher_ast_call_narguments(clause), 0);
    ck_assert_ptr_eq(cypher_ast_call_get_argument(clause, 0), NULL);
    ck_assert_int_eq(cypher_ast_call_nprojections(clause), 0);
    ck_assert_ptr_eq(cypher_ast_call_get_projection(clause, 0), NULL);
}
END_TEST


START_TEST (parse_call_with_args)
{
    struct cypher_input_position last = cypher_input_position_zero;
    result = cypher_parse("CALL foo.bar.baz(1+n, 'foo');", &last, NULL, 0);
    ck_assert_ptr_ne(result, NULL);
    ck_assert_int_eq(last.offset, 29);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
"@0   0..29  statement              body=@1\n"
"@1   0..29  > query                clauses=[@2]\n"
"@2   0..28  > > CALL               name=@3, args=[@4, @7]\n"
"@3   5..16  > > > proc name        `foo.bar.baz`\n"
"@4  17..20  > > > binary operator  @5 + @6\n"
"@5  17..18  > > > > integer        1\n"
"@6  19..20  > > > > identifier     `n`\n"
"@7  22..27  > > > string           \"foo\"\n";
    ck_assert_str_eq(memstream_buffer, expected);

    const cypher_astnode_t *ast = cypher_parse_result_get_directive(result, 0);
    const cypher_astnode_t *query = cypher_ast_statement_get_body(ast);
    const cypher_astnode_t *clause = cypher_ast_query_get_clause(query, 0);
    ck_assert_int_eq(cypher_astnode_type(clause), CYPHER_AST_CALL);

    const cypher_astnode_t *proc = cypher_ast_call_get_proc_name(clause);
    ck_assert_int_eq(cypher_astnode_type(proc), CYPHER_AST_PROC_NAME);
    ck_assert_str_eq(cypher_ast_proc_name_get_value(proc), "foo.bar.baz");

    ck_assert_int_eq(cypher_ast_call_narguments(clause), 2);
    ck_assert_ptr_eq(cypher_ast_call_get_argument(clause, 2), NULL);

    const cypher_astnode_t *arg = cypher_ast_call_get_argument(clause, 0);
    ck_assert_int_eq(cypher_astnode_type(arg), CYPHER_AST_BINARY_OPERATOR);
    arg = cypher_ast_call_get_argument(clause, 1);
    ck_assert_int_eq(cypher_astnode_type(arg), CYPHER_AST_STRING);

    ck_assert_int_eq(cypher_ast_call_nprojections(clause), 0);
    ck_assert_ptr_eq(cypher_ast_call_get_projection(clause, 0), NULL);
}
END_TEST


START_TEST (parse_call_with_projections)
{
    struct cypher_input_position last = cypher_input_position_zero;
    result = cypher_parse(
            "CALL foo.bar.baz(1, 2) YIELD a AS x, b AS y;",
            &last, NULL, 0);
    ck_assert_ptr_ne(result, NULL);
    ck_assert_int_eq(last.offset, 44);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
" @0   0..44  statement           body=@1\n"
" @1   0..44  > query             clauses=[@2]\n"
" @2   0..43  > > CALL            name=@3, args=[@4, @5], YIELD=[@6, @9]\n"
" @3   5..16  > > > proc name     `foo.bar.baz`\n"
" @4  17..18  > > > integer       1\n"
" @5  20..21  > > > integer       2\n"
" @6  29..35  > > > projection    expression=@7, alias=@8\n"
" @7  29..30  > > > > identifier  `a`\n"
" @8  34..35  > > > > identifier  `x`\n"
" @9  37..43  > > > projection    expression=@10, alias=@11\n"
"@10  37..38  > > > > identifier  `b`\n"
"@11  42..43  > > > > identifier  `y`\n";
    ck_assert_str_eq(memstream_buffer, expected);

    const cypher_astnode_t *ast = cypher_parse_result_get_directive(result, 0);
    const cypher_astnode_t *query = cypher_ast_statement_get_body(ast);
    const cypher_astnode_t *clause = cypher_ast_query_get_clause(query, 0);
    ck_assert_int_eq(cypher_astnode_type(clause), CYPHER_AST_CALL);

    const cypher_astnode_t *proc = cypher_ast_call_get_proc_name(clause);
    ck_assert_int_eq(cypher_astnode_type(proc), CYPHER_AST_PROC_NAME);
    ck_assert_str_eq(cypher_ast_proc_name_get_value(proc), "foo.bar.baz");

    ck_assert_int_eq(cypher_ast_call_narguments(clause), 2);
    ck_assert_int_eq(cypher_ast_call_nprojections(clause), 2);
    ck_assert_ptr_eq(cypher_ast_call_get_projection(clause, 2), NULL);

    const cypher_astnode_t *proj = cypher_ast_call_get_projection(clause, 0);
    ck_assert_int_eq(cypher_astnode_type(proj), CYPHER_AST_PROJECTION);
    const cypher_astnode_t *id = cypher_ast_projection_get_expression(proj);
    ck_assert_int_eq(cypher_astnode_type(id), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(id), "a");
    id = cypher_ast_projection_get_alias(proj);
    ck_assert_int_eq(cypher_astnode_type(id), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(id), "x");

    proj = cypher_ast_call_get_projection(clause, 1);
    ck_assert_int_eq(cypher_astnode_type(proj), CYPHER_AST_PROJECTION);
    id = cypher_ast_projection_get_expression(proj);
    ck_assert_int_eq(cypher_astnode_type(id), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(id), "b");
    id = cypher_ast_projection_get_alias(proj);
    ck_assert_int_eq(cypher_astnode_type(id), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(id), "y");
}
END_TEST


START_TEST (parse_call_with_blank_projection)
{
    struct cypher_input_position last = cypher_input_position_zero;
    result = cypher_parse(
            "CALL foo.bar.baz(1, 2) YIELD -;",
            &last, NULL, 0);
    ck_assert_ptr_ne(result, NULL);
    ck_assert_int_eq(last.offset, 31);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
"@0   0..31  statement        body=@1\n"
"@1   0..31  > query          clauses=[@2]\n"
"@2   0..30  > > CALL         name=@3, args=[@4, @5]\n"
"@3   5..16  > > > proc name  `foo.bar.baz`\n"
"@4  17..18  > > > integer    1\n"
"@5  20..21  > > > integer    2\n";
    ck_assert_str_eq(memstream_buffer, expected);

    const cypher_astnode_t *ast = cypher_parse_result_get_directive(result, 0);
    const cypher_astnode_t *query = cypher_ast_statement_get_body(ast);
    const cypher_astnode_t *clause = cypher_ast_query_get_clause(query, 0);
    ck_assert_int_eq(cypher_astnode_type(clause), CYPHER_AST_CALL);

    const cypher_astnode_t *proc = cypher_ast_call_get_proc_name(clause);
    ck_assert_int_eq(cypher_astnode_type(proc), CYPHER_AST_PROC_NAME);
    ck_assert_str_eq(cypher_ast_proc_name_get_value(proc), "foo.bar.baz");

    ck_assert_int_eq(cypher_ast_call_narguments(clause), 2);
    ck_assert_int_eq(cypher_ast_call_nprojections(clause), 0);

    const cypher_astnode_t *pred = cypher_ast_call_get_predicate(clause);
    ck_assert_ptr_eq(pred, NULL);
}
END_TEST


START_TEST (parse_call_with_predicate)
{
    struct cypher_input_position last = cypher_input_position_zero;
    result = cypher_parse(
            "CALL foo.bar.baz(1, 2) YIELD a WHERE a = 5;",
            &last, NULL, 0);
    ck_assert_ptr_ne(result, NULL);
    ck_assert_int_eq(last.offset, 43);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
" @0   0..43  statement              body=@1\n"
" @1   0..43  > query                clauses=[@2]\n"
" @2   0..42  > > CALL               name=@3, args=[@4, @5], YIELD=[@6], WHERE=@8\n"
" @3   5..16  > > > proc name        `foo.bar.baz`\n"
" @4  17..18  > > > integer          1\n"
" @5  20..21  > > > integer          2\n"
" @6  29..31  > > > projection       expression=@7\n"
" @7  29..30  > > > > identifier     `a`\n"
" @8  37..42  > > > binary operator  @9 = @10\n"
" @9  37..38  > > > > identifier     `a`\n"
"@10  41..42  > > > > integer        5\n";
    ck_assert_str_eq(memstream_buffer, expected);

    const cypher_astnode_t *ast = cypher_parse_result_get_directive(result, 0);
    const cypher_astnode_t *query = cypher_ast_statement_get_body(ast);
    const cypher_astnode_t *clause = cypher_ast_query_get_clause(query, 0);
    ck_assert_int_eq(cypher_astnode_type(clause), CYPHER_AST_CALL);

    const cypher_astnode_t *proc = cypher_ast_call_get_proc_name(clause);
    ck_assert_int_eq(cypher_astnode_type(proc), CYPHER_AST_PROC_NAME);
    ck_assert_str_eq(cypher_ast_proc_name_get_value(proc), "foo.bar.baz");

    ck_assert_int_eq(cypher_ast_call_narguments(clause), 2);
    ck_assert_ptr_eq(cypher_ast_call_get_argument(clause, 2), NULL);

    ck_assert_int_eq(cypher_ast_call_nprojections(clause), 1);
    ck_assert_ptr_eq(cypher_ast_call_get_projection(clause, 1), NULL);

    const cypher_astnode_t *proj = cypher_ast_call_get_projection(clause, 0);
    ck_assert_int_eq(cypher_astnode_type(proj), CYPHER_AST_PROJECTION);
    const cypher_astnode_t *id = cypher_ast_projection_get_expression(proj);
    ck_assert_int_eq(cypher_astnode_type(id), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(id), "a");

    const cypher_astnode_t *pred = cypher_ast_call_get_predicate(clause);
    ck_assert_int_eq(cypher_astnode_type(pred), CYPHER_AST_BINARY_OPERATOR);
}
END_TEST


TCase* call_tcase(void)
{
    TCase *tc = tcase_create("call");
    tcase_add_checked_fixture(tc, setup, teardown);
    tcase_add_test(tc, parse_simple_call);
    tcase_add_test(tc, parse_call_with_args);
    tcase_add_test(tc, parse_call_with_projections);
    tcase_add_test(tc, parse_call_with_blank_projection);
    tcase_add_test(tc, parse_call_with_predicate);
    return tc;
}
