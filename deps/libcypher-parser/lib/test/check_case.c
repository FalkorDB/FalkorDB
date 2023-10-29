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


START_TEST (parse_simple_case)
{
    struct cypher_input_position last = cypher_input_position_zero;
    result = cypher_parse(
            "RETURN CASE x WHEN 1 THEN y WHEN 2 THEN z ELSE d END AS r",
            &last, NULL, 0);
    ck_assert_ptr_ne(result, NULL);
    ck_assert_int_eq(last.offset, 57);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
" @0   0..57  statement             body=@1\n"
" @1   0..57  > query               clauses=[@2]\n"
" @2   0..57  > > RETURN            projections=[@3]\n"
" @3   7..57  > > > projection      expression=@4, alias=@11\n"
" @4   7..52  > > > > case          expression=@5, alternatives=[(@6:@7), (@8:@9)], default=@10\n"
" @5  12..13  > > > > > identifier  `x`\n"
" @6  19..20  > > > > > integer     1\n"
" @7  26..27  > > > > > identifier  `y`\n"
" @8  33..34  > > > > > integer     2\n"
" @9  40..41  > > > > > identifier  `z`\n"
"@10  47..48  > > > > > identifier  `d`\n"
"@11  56..57  > > > > identifier    `r`\n";
    ck_assert_str_eq(memstream_buffer, expected);

    const cypher_astnode_t *ast = cypher_parse_result_get_directive(result, 0);
    const cypher_astnode_t *query = cypher_ast_statement_get_body(ast);
    const cypher_astnode_t *clause = cypher_ast_query_get_clause(query, 0);
    ck_assert_int_eq(cypher_astnode_type(clause), CYPHER_AST_RETURN);
    ck_assert_int_eq(cypher_ast_return_nprojections(clause), 1);

    const cypher_astnode_t *proj = cypher_ast_return_get_projection(clause, 0);
    ck_assert_int_eq(cypher_astnode_type(proj), CYPHER_AST_PROJECTION);
    const cypher_astnode_t *exp = cypher_ast_projection_get_expression(proj);
    ck_assert_int_eq(cypher_astnode_type(exp), CYPHER_AST_CASE);

    ck_assert_int_eq(cypher_ast_case_nalternatives(exp), 2);
    ck_assert_ptr_eq(cypher_ast_case_get_predicate(exp, 2), NULL);
    ck_assert_ptr_eq(cypher_ast_case_get_value(exp, 2), NULL);

    const cypher_astnode_t *id = cypher_ast_case_get_expression(exp);
    ck_assert_int_eq(cypher_astnode_type(id), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(id), "x");

    const cypher_astnode_t *pred = cypher_ast_case_get_predicate(exp, 0);
    ck_assert_int_eq(cypher_astnode_type(pred), CYPHER_AST_INTEGER);
    ck_assert_str_eq(cypher_ast_integer_get_valuestr(pred), "1");

    const cypher_astnode_t *value = cypher_ast_case_get_value(exp, 0);
    ck_assert_int_eq(cypher_astnode_type(value), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(value), "y");

    pred = cypher_ast_case_get_predicate(exp, 1);
    ck_assert_int_eq(cypher_astnode_type(pred), CYPHER_AST_INTEGER);
    ck_assert_str_eq(cypher_ast_integer_get_valuestr(pred), "2");

    value = cypher_ast_case_get_value(exp, 1);
    ck_assert_int_eq(cypher_astnode_type(value), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(value), "z");

    const cypher_astnode_t *deflt = cypher_ast_case_get_default(exp);
    ck_assert_int_eq(cypher_astnode_type(deflt), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(deflt), "d");
}
END_TEST


START_TEST (parse_simple_case_without_default)
{
    struct cypher_input_position last = cypher_input_position_zero;
    result = cypher_parse(
            "RETURN CASE x WHEN 1 THEN y WHEN 2 THEN z END AS r",
            &last, NULL, 0);
    ck_assert_ptr_ne(result, NULL);
    ck_assert_int_eq(last.offset, 50);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
" @0   0..50  statement             body=@1\n"
" @1   0..50  > query               clauses=[@2]\n"
" @2   0..50  > > RETURN            projections=[@3]\n"
" @3   7..50  > > > projection      expression=@4, alias=@10\n"
" @4   7..45  > > > > case          expression=@5, alternatives=[(@6:@7), (@8:@9)]\n"
" @5  12..13  > > > > > identifier  `x`\n"
" @6  19..20  > > > > > integer     1\n"
" @7  26..27  > > > > > identifier  `y`\n"
" @8  33..34  > > > > > integer     2\n"
" @9  40..41  > > > > > identifier  `z`\n"
"@10  49..50  > > > > identifier    `r`\n";
    ck_assert_str_eq(memstream_buffer, expected);

    const cypher_astnode_t *ast = cypher_parse_result_get_directive(result, 0);
    const cypher_astnode_t *query = cypher_ast_statement_get_body(ast);
    const cypher_astnode_t *clause = cypher_ast_query_get_clause(query, 0);
    ck_assert_int_eq(cypher_astnode_type(clause), CYPHER_AST_RETURN);
    ck_assert_int_eq(cypher_ast_return_nprojections(clause), 1);

    const cypher_astnode_t *proj = cypher_ast_return_get_projection(clause, 0);
    ck_assert_int_eq(cypher_astnode_type(proj), CYPHER_AST_PROJECTION);
    const cypher_astnode_t *exp = cypher_ast_projection_get_expression(proj);
    ck_assert_int_eq(cypher_astnode_type(exp), CYPHER_AST_CASE);

    ck_assert_int_eq(cypher_ast_case_nalternatives(exp), 2);
    ck_assert_ptr_eq(cypher_ast_case_get_predicate(exp, 2), NULL);
    ck_assert_ptr_eq(cypher_ast_case_get_value(exp, 2), NULL);

    const cypher_astnode_t *id = cypher_ast_case_get_expression(exp);
    ck_assert_int_eq(cypher_astnode_type(id), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(id), "x");

    const cypher_astnode_t *pred = cypher_ast_case_get_predicate(exp, 0);
    ck_assert_int_eq(cypher_astnode_type(pred), CYPHER_AST_INTEGER);
    ck_assert_str_eq(cypher_ast_integer_get_valuestr(pred), "1");

    const cypher_astnode_t *value = cypher_ast_case_get_value(exp, 0);
    ck_assert_int_eq(cypher_astnode_type(value), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(value), "y");

    pred = cypher_ast_case_get_predicate(exp, 1);
    ck_assert_int_eq(cypher_astnode_type(pred), CYPHER_AST_INTEGER);
    ck_assert_str_eq(cypher_ast_integer_get_valuestr(pred), "2");

    value = cypher_ast_case_get_value(exp, 1);
    ck_assert_int_eq(cypher_astnode_type(value), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(value), "z");

    ck_assert_ptr_eq(cypher_ast_case_get_default(exp), NULL);
}
END_TEST


START_TEST (parse_generic_case)
{
    struct cypher_input_position last = cypher_input_position_zero;
    result = cypher_parse(
            "RETURN CASE WHEN x=1 THEN y WHEN x=2 THEN z ELSE d END AS r",
            &last, NULL, 0);
    ck_assert_ptr_ne(result, NULL);
    ck_assert_int_eq(last.offset, 59);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
" @0   0..59  statement                  body=@1\n"
" @1   0..59  > query                    clauses=[@2]\n"
" @2   0..59  > > RETURN                 projections=[@3]\n"
" @3   7..59  > > > projection           expression=@4, alias=@14\n"
" @4   7..54  > > > > case               alternatives=[(@5:@8), (@9:@12)], default=@13\n"
" @5  17..21  > > > > > binary operator  @6 = @7\n"
" @6  17..18  > > > > > > identifier     `x`\n"
" @7  19..20  > > > > > > integer        1\n"
" @8  26..27  > > > > > identifier       `y`\n"
" @9  33..37  > > > > > binary operator  @10 = @11\n"
"@10  33..34  > > > > > > identifier     `x`\n"
"@11  35..36  > > > > > > integer        2\n"
"@12  42..43  > > > > > identifier       `z`\n"
"@13  49..50  > > > > > identifier       `d`\n"
"@14  58..59  > > > > identifier         `r`\n";
    ck_assert_str_eq(memstream_buffer, expected);

    const cypher_astnode_t *ast = cypher_parse_result_get_directive(result, 0);
    const cypher_astnode_t *query = cypher_ast_statement_get_body(ast);
    const cypher_astnode_t *clause = cypher_ast_query_get_clause(query, 0);
    ck_assert_int_eq(cypher_astnode_type(clause), CYPHER_AST_RETURN);
    ck_assert_int_eq(cypher_ast_return_nprojections(clause), 1);

    const cypher_astnode_t *proj = cypher_ast_return_get_projection(clause, 0);
    ck_assert_int_eq(cypher_astnode_type(proj), CYPHER_AST_PROJECTION);
    const cypher_astnode_t *exp = cypher_ast_projection_get_expression(proj);
    ck_assert_int_eq(cypher_astnode_type(exp), CYPHER_AST_CASE);

    ck_assert_int_eq(cypher_ast_case_nalternatives(exp), 2);
    ck_assert_ptr_eq(cypher_ast_case_get_predicate(exp, 2), NULL);
    ck_assert_ptr_eq(cypher_ast_case_get_value(exp, 2), NULL);

    ck_assert_ptr_eq(cypher_ast_case_get_expression(exp), NULL);

    const cypher_astnode_t *pred = cypher_ast_case_get_predicate(exp, 0);
    ck_assert_int_eq(cypher_astnode_type(pred), CYPHER_AST_BINARY_OPERATOR);
    const cypher_astnode_t *id = cypher_ast_binary_operator_get_argument2(pred);
    ck_assert_int_eq(cypher_astnode_type(id), CYPHER_AST_INTEGER);
    ck_assert_str_eq(cypher_ast_integer_get_valuestr(id), "1");

    const cypher_astnode_t *value = cypher_ast_case_get_value(exp, 0);
    ck_assert_int_eq(cypher_astnode_type(value), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(value), "y");

    pred = cypher_ast_case_get_predicate(exp, 1);
    ck_assert_int_eq(cypher_astnode_type(pred), CYPHER_AST_BINARY_OPERATOR);
    id = cypher_ast_binary_operator_get_argument2(pred);
    ck_assert_int_eq(cypher_astnode_type(id), CYPHER_AST_INTEGER);
    ck_assert_str_eq(cypher_ast_integer_get_valuestr(id), "2");

    value = cypher_ast_case_get_value(exp, 1);
    ck_assert_int_eq(cypher_astnode_type(value), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(value), "z");

    const cypher_astnode_t *deflt = cypher_ast_case_get_default(exp);
    ck_assert_int_eq(cypher_astnode_type(deflt), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(deflt), "d");
}
END_TEST


START_TEST (parse_generic_case_without_default)
{
    struct cypher_input_position last = cypher_input_position_zero;
    result = cypher_parse(
            "RETURN CASE WHEN x=1 THEN y WHEN x=2 THEN z END AS r",
            &last, NULL, 0);
    ck_assert_ptr_ne(result, NULL);
    ck_assert_int_eq(last.offset, 52);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
" @0   0..52  statement                  body=@1\n"
" @1   0..52  > query                    clauses=[@2]\n"
" @2   0..52  > > RETURN                 projections=[@3]\n"
" @3   7..52  > > > projection           expression=@4, alias=@13\n"
" @4   7..47  > > > > case               alternatives=[(@5:@8), (@9:@12)]\n"
" @5  17..21  > > > > > binary operator  @6 = @7\n"
" @6  17..18  > > > > > > identifier     `x`\n"
" @7  19..20  > > > > > > integer        1\n"
" @8  26..27  > > > > > identifier       `y`\n"
" @9  33..37  > > > > > binary operator  @10 = @11\n"
"@10  33..34  > > > > > > identifier     `x`\n"
"@11  35..36  > > > > > > integer        2\n"
"@12  42..43  > > > > > identifier       `z`\n"
"@13  51..52  > > > > identifier         `r`\n";
    ck_assert_str_eq(memstream_buffer, expected);

    const cypher_astnode_t *ast = cypher_parse_result_get_directive(result, 0);
    const cypher_astnode_t *query = cypher_ast_statement_get_body(ast);
    const cypher_astnode_t *clause = cypher_ast_query_get_clause(query, 0);
    ck_assert_int_eq(cypher_astnode_type(clause), CYPHER_AST_RETURN);
    ck_assert_int_eq(cypher_ast_return_nprojections(clause), 1);

    const cypher_astnode_t *proj = cypher_ast_return_get_projection(clause, 0);
    ck_assert_int_eq(cypher_astnode_type(proj), CYPHER_AST_PROJECTION);
    const cypher_astnode_t *exp = cypher_ast_projection_get_expression(proj);
    ck_assert_int_eq(cypher_astnode_type(exp), CYPHER_AST_CASE);

    ck_assert_int_eq(cypher_ast_case_nalternatives(exp), 2);
    ck_assert_ptr_eq(cypher_ast_case_get_predicate(exp, 2), NULL);
    ck_assert_ptr_eq(cypher_ast_case_get_value(exp, 2), NULL);

    ck_assert_ptr_eq(cypher_ast_case_get_expression(exp), NULL);

    const cypher_astnode_t *pred = cypher_ast_case_get_predicate(exp, 0);
    ck_assert_int_eq(cypher_astnode_type(pred), CYPHER_AST_BINARY_OPERATOR);
    const cypher_astnode_t *id = cypher_ast_binary_operator_get_argument2(pred);
    ck_assert_int_eq(cypher_astnode_type(id), CYPHER_AST_INTEGER);
    ck_assert_str_eq(cypher_ast_integer_get_valuestr(id), "1");

    const cypher_astnode_t *value = cypher_ast_case_get_value(exp, 0);
    ck_assert_int_eq(cypher_astnode_type(value), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(value), "y");

    pred = cypher_ast_case_get_predicate(exp, 1);
    ck_assert_int_eq(cypher_astnode_type(pred), CYPHER_AST_BINARY_OPERATOR);
    id = cypher_ast_binary_operator_get_argument2(pred);
    ck_assert_int_eq(cypher_astnode_type(id), CYPHER_AST_INTEGER);
    ck_assert_str_eq(cypher_ast_integer_get_valuestr(id), "2");

    value = cypher_ast_case_get_value(exp, 1);
    ck_assert_int_eq(cypher_astnode_type(value), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(value), "z");

    ck_assert_ptr_eq(cypher_ast_case_get_default(exp), NULL);
}
END_TEST


TCase* case_tcase(void)
{
    TCase *tc = tcase_create("case");
    tcase_add_checked_fixture(tc, setup, teardown);
    tcase_add_test(tc, parse_simple_case);
    tcase_add_test(tc, parse_simple_case_without_default);
    tcase_add_test(tc, parse_generic_case);
    tcase_add_test(tc, parse_generic_case_without_default);
    return tc;
}
