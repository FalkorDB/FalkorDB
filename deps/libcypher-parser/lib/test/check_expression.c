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


START_TEST (parse_simple_unary_operators)
{
    struct cypher_input_position last = cypher_input_position_zero;
    result = cypher_parse("RETURN -1, +1, NOT false;", &last, NULL, 0);
    ck_assert_ptr_ne(result, NULL);
    ck_assert_int_eq(last.offset, 25);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
" @0   0..25  statement               body=@1\n"
" @1   0..25  > query                 clauses=[@2]\n"
" @2   0..24  > > RETURN              projections=[@3, @7, @11]\n"
" @3   7..9   > > > projection        expression=@4, alias=@6\n"
" @4   7..9   > > > > unary operator  - @5\n"
" @5   8..9   > > > > > integer       1\n"
" @6   7..9   > > > > identifier      `-1`\n"
" @7  11..13  > > > projection        expression=@8, alias=@10\n"
" @8  11..13  > > > > unary operator  + @9\n"
" @9  12..13  > > > > > integer       1\n"
"@10  11..13  > > > > identifier      `+1`\n"
"@11  15..24  > > > projection        expression=@12, alias=@14\n"
"@12  15..24  > > > > unary operator  NOT @13\n"
"@13  19..24  > > > > > FALSE\n"
"@14  15..24  > > > > identifier      `NOT false`\n";
    ck_assert_str_eq(memstream_buffer, expected);

    const cypher_astnode_t *ast = cypher_parse_result_get_directive(result, 0);
    const cypher_astnode_t *query = cypher_ast_statement_get_body(ast);
    const cypher_astnode_t *clause = cypher_ast_query_get_clause(query, 0);
    ck_assert_int_eq(cypher_astnode_type(clause), CYPHER_AST_RETURN);

    ck_assert_int_eq(cypher_ast_return_nprojections(clause), 3);

    const cypher_astnode_t *proj = cypher_ast_return_get_projection(clause, 0);
    ck_assert_int_eq(cypher_astnode_type(proj), CYPHER_AST_PROJECTION);
    const cypher_astnode_t *exp = cypher_ast_projection_get_expression(proj);
    ck_assert_int_eq(cypher_astnode_type(exp), CYPHER_AST_UNARY_OPERATOR);
    ck_assert_ptr_eq(cypher_ast_unary_operator_get_operator(exp),
            CYPHER_OP_UNARY_MINUS);
    const cypher_astnode_t *arg = cypher_ast_unary_operator_get_argument(exp);
    ck_assert_int_eq(cypher_astnode_type(arg), CYPHER_AST_INTEGER);

    proj = cypher_ast_return_get_projection(clause, 1);
    ck_assert_int_eq(cypher_astnode_type(proj), CYPHER_AST_PROJECTION);
    exp = cypher_ast_projection_get_expression(proj);
    ck_assert_int_eq(cypher_astnode_type(exp), CYPHER_AST_UNARY_OPERATOR);
    ck_assert_ptr_eq(cypher_ast_unary_operator_get_operator(exp),
            CYPHER_OP_UNARY_PLUS);
    arg = cypher_ast_unary_operator_get_argument(exp);
    ck_assert_int_eq(cypher_astnode_type(arg), CYPHER_AST_INTEGER);

    proj = cypher_ast_return_get_projection(clause, 2);
    ck_assert_int_eq(cypher_astnode_type(proj), CYPHER_AST_PROJECTION);
    exp = cypher_ast_projection_get_expression(proj);
    ck_assert_int_eq(cypher_astnode_type(exp), CYPHER_AST_UNARY_OPERATOR);
    ck_assert_ptr_eq(cypher_ast_unary_operator_get_operator(exp),
            CYPHER_OP_NOT);
    arg = cypher_ast_unary_operator_get_argument(exp);
    ck_assert(cypher_astnode_instanceof(arg, CYPHER_AST_BOOLEAN));
    ck_assert_int_eq(cypher_astnode_type(arg), CYPHER_AST_FALSE);
}
END_TEST


START_TEST (parse_simple_binary_operators)
{
    struct cypher_input_position last = cypher_input_position_zero;
    result = cypher_parse(
            "RETURN a-1, 1 / b, c STARTS WITH 'foo', d =~ '.*', e = 1;",
            &last, NULL, 0);
    ck_assert_ptr_ne(result, NULL);
    ck_assert_int_eq(last.offset, 57);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
" @0   0..57  statement                body=@1\n"
" @1   0..57  > query                  clauses=[@2]\n"
" @2   0..56  > > RETURN               projections=[@3, @8, @13, @18, @23]\n"
" @3   7..10  > > > projection         expression=@4, alias=@7\n"
" @4   7..10  > > > > binary operator  @5 - @6\n"
" @5   7..8   > > > > > identifier     `a`\n"
" @6   9..10  > > > > > integer        1\n"
" @7   7..10  > > > > identifier       `a-1`\n"
" @8  12..17  > > > projection         expression=@9, alias=@12\n"
" @9  12..17  > > > > binary operator  @10 / @11\n"
"@10  12..13  > > > > > integer        1\n"
"@11  16..17  > > > > > identifier     `b`\n"
"@12  12..17  > > > > identifier       `1 / b`\n"
"@13  19..38  > > > projection         expression=@14, alias=@17\n"
"@14  19..38  > > > > binary operator  @15 STARTS WITH @16\n"
"@15  19..20  > > > > > identifier     `c`\n"
"@16  33..38  > > > > > string         \"foo\"\n"
"@17  19..38  > > > > identifier       `c STARTS WITH 'foo'`\n"
"@18  40..49  > > > projection         expression=@19, alias=@22\n"
"@19  40..49  > > > > binary operator  @20 =~ @21\n"
"@20  40..41  > > > > > identifier     `d`\n"
"@21  45..49  > > > > > string         \".*\"\n"
"@22  40..49  > > > > identifier       `d =~ '.*'`\n"
"@23  51..56  > > > projection         expression=@24, alias=@27\n"
"@24  51..56  > > > > binary operator  @25 = @26\n"
"@25  51..52  > > > > > identifier     `e`\n"
"@26  55..56  > > > > > integer        1\n"
"@27  51..56  > > > > identifier       `e = 1`\n";
    ck_assert_str_eq(memstream_buffer, expected);

    const cypher_astnode_t *ast = cypher_parse_result_get_directive(result, 0);
    const cypher_astnode_t *query = cypher_ast_statement_get_body(ast);
    const cypher_astnode_t *clause = cypher_ast_query_get_clause(query, 0);
    ck_assert_int_eq(cypher_astnode_type(clause), CYPHER_AST_RETURN);

    ck_assert_int_eq(cypher_ast_return_nprojections(clause), 5);

    const cypher_astnode_t *proj = cypher_ast_return_get_projection(clause, 0);
    ck_assert_int_eq(cypher_astnode_type(proj), CYPHER_AST_PROJECTION);
    const cypher_astnode_t *exp = cypher_ast_projection_get_expression(proj);
    ck_assert_int_eq(cypher_astnode_type(exp), CYPHER_AST_BINARY_OPERATOR);
    ck_assert_ptr_eq(cypher_ast_binary_operator_get_operator(exp),
            CYPHER_OP_MINUS);
    const cypher_astnode_t *arg = cypher_ast_binary_operator_get_argument1(exp);
    ck_assert_int_eq(cypher_astnode_type(arg), CYPHER_AST_IDENTIFIER);
    arg = cypher_ast_binary_operator_get_argument2(exp);
    ck_assert_int_eq(cypher_astnode_type(arg), CYPHER_AST_INTEGER);

    proj = cypher_ast_return_get_projection(clause, 1);
    ck_assert_int_eq(cypher_astnode_type(proj), CYPHER_AST_PROJECTION);
    exp = cypher_ast_projection_get_expression(proj);
    ck_assert_int_eq(cypher_astnode_type(exp), CYPHER_AST_BINARY_OPERATOR);
    ck_assert_ptr_eq(cypher_ast_binary_operator_get_operator(exp),
            CYPHER_OP_DIV);
    arg = cypher_ast_binary_operator_get_argument1(exp);
    ck_assert_int_eq(cypher_astnode_type(arg), CYPHER_AST_INTEGER);
    arg = cypher_ast_binary_operator_get_argument2(exp);
    ck_assert_int_eq(cypher_astnode_type(arg), CYPHER_AST_IDENTIFIER);

    proj = cypher_ast_return_get_projection(clause, 2);
    ck_assert_int_eq(cypher_astnode_type(proj), CYPHER_AST_PROJECTION);
    exp = cypher_ast_projection_get_expression(proj);
    ck_assert_int_eq(cypher_astnode_type(exp), CYPHER_AST_BINARY_OPERATOR);
    ck_assert_ptr_eq(cypher_ast_binary_operator_get_operator(exp),
            CYPHER_OP_STARTS_WITH);
    arg = cypher_ast_binary_operator_get_argument1(exp);
    ck_assert_int_eq(cypher_astnode_type(arg), CYPHER_AST_IDENTIFIER);
    arg = cypher_ast_binary_operator_get_argument2(exp);
    ck_assert_int_eq(cypher_astnode_type(arg), CYPHER_AST_STRING);

    proj = cypher_ast_return_get_projection(clause, 3);
    ck_assert_int_eq(cypher_astnode_type(proj), CYPHER_AST_PROJECTION);
    exp = cypher_ast_projection_get_expression(proj);
    ck_assert_int_eq(cypher_astnode_type(exp), CYPHER_AST_BINARY_OPERATOR);
    ck_assert_ptr_eq(cypher_ast_binary_operator_get_operator(exp),
            CYPHER_OP_REGEX);
    arg = cypher_ast_binary_operator_get_argument1(exp);
    ck_assert_int_eq(cypher_astnode_type(arg), CYPHER_AST_IDENTIFIER);
    arg = cypher_ast_binary_operator_get_argument2(exp);
    ck_assert_int_eq(cypher_astnode_type(arg), CYPHER_AST_STRING);

    proj = cypher_ast_return_get_projection(clause, 4);
    ck_assert_int_eq(cypher_astnode_type(proj), CYPHER_AST_PROJECTION);
    exp = cypher_ast_projection_get_expression(proj);
    ck_assert_int_eq(cypher_astnode_type(exp), CYPHER_AST_BINARY_OPERATOR);
    ck_assert_ptr_eq(cypher_ast_binary_operator_get_operator(exp),
            CYPHER_OP_EQUAL);
    arg = cypher_ast_binary_operator_get_argument1(exp);
    ck_assert_int_eq(cypher_astnode_type(arg), CYPHER_AST_IDENTIFIER);
    arg = cypher_ast_binary_operator_get_argument2(exp);
    ck_assert_int_eq(cypher_astnode_type(arg), CYPHER_AST_INTEGER);
}
END_TEST


START_TEST (parse_unary_and_binary_operators)
{
    struct cypher_input_position last = cypher_input_position_zero;
    result = cypher_parse("RETURN NOT 1 - 2 AND 3;", &last, NULL, 0);
    ck_assert_ptr_ne(result, NULL);
    ck_assert_int_eq(last.offset, 23);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
" @0   0..23  statement                    body=@1\n"
" @1   0..23  > query                      clauses=[@2]\n"
" @2   0..22  > > RETURN                   projections=[@3]\n"
" @3   7..22  > > > projection             expression=@4, alias=@10\n"
" @4   7..22  > > > > binary operator      @5 AND @9\n"
" @5   7..17  > > > > > unary operator     NOT @6\n"
" @6  11..17  > > > > > > binary operator  @7 - @8\n"
" @7  11..12  > > > > > > > integer        1\n"
" @8  15..16  > > > > > > > integer        2\n"
" @9  21..22  > > > > > integer            3\n"
"@10   7..22  > > > > identifier           `NOT 1 - 2 AND 3`\n";
    ck_assert_str_eq(memstream_buffer, expected);
}
END_TEST


START_TEST (parse_comparison_operators)
{
    struct cypher_input_position last = cypher_input_position_zero;
    result = cypher_parse("RETURN a < 1, 4 > b > 2, 2 <= c >= 5;",
            &last, NULL, 0);
    ck_assert_ptr_ne(result, NULL);
    ck_assert_int_eq(last.offset, 37);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
" @0   0..37  statement             body=@1\n"
" @1   0..37  > query               clauses=[@2]\n"
" @2   0..36  > > RETURN            projections=[@3, @8, @14]\n"
" @3   7..12  > > > projection      expression=@4, alias=@7\n"
" @4   7..12  > > > > comparison    @5 < @6\n"
" @5   7..8   > > > > > identifier  `a`\n"
" @6  11..12  > > > > > integer     1\n"
" @7   7..12  > > > > identifier    `a < 1`\n"
" @8  14..23  > > > projection      expression=@9, alias=@13\n"
" @9  14..23  > > > > comparison    @10 > @11 > @12\n"
"@10  14..15  > > > > > integer     4\n"
"@11  18..19  > > > > > identifier  `b`\n"
"@12  22..23  > > > > > integer     2\n"
"@13  14..23  > > > > identifier    `4 > b > 2`\n"
"@14  25..36  > > > projection      expression=@15, alias=@19\n"
"@15  25..36  > > > > comparison    @16 <= @17 >= @18\n"
"@16  25..26  > > > > > integer     2\n"
"@17  30..31  > > > > > identifier  `c`\n"
"@18  35..36  > > > > > integer     5\n"
"@19  25..36  > > > > identifier    `2 <= c >= 5`\n";
    ck_assert_str_eq(memstream_buffer, expected);

    const cypher_astnode_t *ast = cypher_parse_result_get_directive(result, 0);
    const cypher_astnode_t *query = cypher_ast_statement_get_body(ast);
    const cypher_astnode_t *clause = cypher_ast_query_get_clause(query, 0);
    ck_assert_int_eq(cypher_astnode_type(clause), CYPHER_AST_RETURN);

    ck_assert_int_eq(cypher_ast_return_nprojections(clause), 3);

    const cypher_astnode_t *proj = cypher_ast_return_get_projection(clause, 0);
    ck_assert_int_eq(cypher_astnode_type(proj), CYPHER_AST_PROJECTION);
    const cypher_astnode_t *exp = cypher_ast_projection_get_expression(proj);
    ck_assert_int_eq(cypher_astnode_type(exp), CYPHER_AST_COMPARISON);
    ck_assert_int_eq(cypher_ast_comparison_get_length(exp), 1);
    const cypher_astnode_t *arg = cypher_ast_comparison_get_argument(exp, 0);
    ck_assert_int_eq(cypher_astnode_type(arg), CYPHER_AST_IDENTIFIER);
    ck_assert_ptr_eq(cypher_ast_comparison_get_operator(exp, 0),
            CYPHER_OP_LT);
    arg = cypher_ast_comparison_get_argument(exp, 1);
    ck_assert_int_eq(cypher_astnode_type(arg), CYPHER_AST_INTEGER);
    ck_assert_ptr_eq(cypher_ast_comparison_get_operator(exp, 1), NULL);
    ck_assert_ptr_eq(cypher_ast_comparison_get_argument(exp, 2), NULL);

    proj = cypher_ast_return_get_projection(clause, 1);
    ck_assert_int_eq(cypher_astnode_type(proj), CYPHER_AST_PROJECTION);
    exp = cypher_ast_projection_get_expression(proj);
    ck_assert_int_eq(cypher_astnode_type(exp), CYPHER_AST_COMPARISON);
    ck_assert_int_eq(cypher_ast_comparison_get_length(exp), 2);
    arg = cypher_ast_comparison_get_argument(exp, 0);
    ck_assert_int_eq(cypher_astnode_type(arg), CYPHER_AST_INTEGER);
    ck_assert_ptr_eq(cypher_ast_comparison_get_operator(exp, 0),
            CYPHER_OP_GT);
    arg = cypher_ast_comparison_get_argument(exp, 1);
    ck_assert_int_eq(cypher_astnode_type(arg), CYPHER_AST_IDENTIFIER);
    ck_assert_ptr_eq(cypher_ast_comparison_get_operator(exp, 1),
            CYPHER_OP_GT);
    arg = cypher_ast_comparison_get_argument(exp, 2);
    ck_assert_int_eq(cypher_astnode_type(arg), CYPHER_AST_INTEGER);
    ck_assert_ptr_eq(cypher_ast_comparison_get_operator(exp, 2), NULL);
    ck_assert_ptr_eq(cypher_ast_comparison_get_argument(exp, 3), NULL);

    proj = cypher_ast_return_get_projection(clause, 2);
    ck_assert_int_eq(cypher_astnode_type(proj), CYPHER_AST_PROJECTION);
    exp = cypher_ast_projection_get_expression(proj);
    ck_assert_int_eq(cypher_astnode_type(exp), CYPHER_AST_COMPARISON);
    ck_assert_int_eq(cypher_ast_comparison_get_length(exp), 2);
    arg = cypher_ast_comparison_get_argument(exp, 0);
    ck_assert_int_eq(cypher_astnode_type(arg), CYPHER_AST_INTEGER);
    ck_assert_ptr_eq(cypher_ast_comparison_get_operator(exp, 0),
            CYPHER_OP_LTE);
    arg = cypher_ast_comparison_get_argument(exp, 1);
    ck_assert_int_eq(cypher_astnode_type(arg), CYPHER_AST_IDENTIFIER);
    ck_assert_ptr_eq(cypher_ast_comparison_get_operator(exp, 1),
            CYPHER_OP_GTE);
    arg = cypher_ast_comparison_get_argument(exp, 2);
    ck_assert_int_eq(cypher_astnode_type(arg), CYPHER_AST_INTEGER);
    ck_assert_ptr_eq(cypher_ast_comparison_get_operator(exp, 2), NULL);
    ck_assert_ptr_eq(cypher_ast_comparison_get_argument(exp, 3), NULL);
}
END_TEST


START_TEST (parse_apply)
{
    struct cypher_input_position last = cypher_input_position_zero;
    result = cypher_parse(
            "RETURN foo(bar, baz), sum(DISTINCT a), count(*), count(DISTINCT *);",
            &last, NULL, 0);
    ck_assert_ptr_ne(result, NULL);
    ck_assert_int_eq(last.offset, 67);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
" @0   0..67  statement                body=@1\n"
" @1   0..67  > query                  clauses=[@2]\n"
" @2   0..66  > > RETURN               projections=[@3, @9, @14, @18]\n"
" @3   7..20  > > > projection         expression=@4, alias=@8\n"
" @4   7..20  > > > > apply            @5(@6, @7)\n"
" @5   7..10  > > > > > function name  `foo`\n"
" @6  11..14  > > > > > identifier     `bar`\n"
" @7  16..19  > > > > > identifier     `baz`\n"
" @8   7..20  > > > > identifier       `foo(bar, baz)`\n"
" @9  22..37  > > > projection         expression=@10, alias=@13\n"
"@10  22..37  > > > > apply            @11(DISTINCT @12)\n"
"@11  22..25  > > > > > function name  `sum`\n"
"@12  35..36  > > > > > identifier     `a`\n"
"@13  22..37  > > > > identifier       `sum(DISTINCT a)`\n"
"@14  39..47  > > > projection         expression=@15, alias=@17\n"
"@15  39..47  > > > > apply all        @16(*)\n"
"@16  39..44  > > > > > function name  `count`\n"
"@17  39..47  > > > > identifier       `count(*)`\n"
"@18  49..66  > > > projection         expression=@19, alias=@21\n"
"@19  49..66  > > > > apply all        @20(DISTINCT *)\n"
"@20  49..54  > > > > > function name  `count`\n"
"@21  49..66  > > > > identifier       `count(DISTINCT *)`\n";
    ck_assert_str_eq(memstream_buffer, expected);

    const cypher_astnode_t *ast = cypher_parse_result_get_directive(result, 0);
    const cypher_astnode_t *query = cypher_ast_statement_get_body(ast);
    const cypher_astnode_t *clause = cypher_ast_query_get_clause(query, 0);
    ck_assert_int_eq(cypher_astnode_type(clause), CYPHER_AST_RETURN);

    ck_assert_int_eq(cypher_ast_return_nprojections(clause), 4);

    const cypher_astnode_t *proj = cypher_ast_return_get_projection(clause, 0);
    ck_assert_int_eq(cypher_astnode_type(proj), CYPHER_AST_PROJECTION);
    const cypher_astnode_t *exp = cypher_ast_projection_get_expression(proj);
    ck_assert_int_eq(cypher_astnode_type(exp), CYPHER_AST_APPLY_OPERATOR);

    const cypher_astnode_t *func_name =
            cypher_ast_apply_operator_get_func_name(exp);
    ck_assert_int_eq(cypher_astnode_type(func_name), CYPHER_AST_FUNCTION_NAME);
    ck_assert_str_eq(cypher_ast_function_name_get_value(func_name), "foo");

    ck_assert(!cypher_ast_apply_operator_get_distinct(exp));
    ck_assert_int_eq(cypher_ast_apply_operator_narguments(exp), 2);
    const cypher_astnode_t *arg = cypher_ast_apply_operator_get_argument(exp, 0);
    ck_assert_int_eq(cypher_astnode_type(arg), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(arg), "bar");
    arg = cypher_ast_apply_operator_get_argument(exp, 1);
    ck_assert_int_eq(cypher_astnode_type(arg), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(arg), "baz");
    ck_assert_ptr_eq(cypher_ast_apply_operator_get_argument(exp, 2), NULL);

    proj = cypher_ast_return_get_projection(clause, 1);
    ck_assert_int_eq(cypher_astnode_type(proj), CYPHER_AST_PROJECTION);
    exp = cypher_ast_projection_get_expression(proj);
    ck_assert_int_eq(cypher_astnode_type(exp), CYPHER_AST_APPLY_OPERATOR);

    func_name = cypher_ast_apply_operator_get_func_name(exp);
    ck_assert_int_eq(cypher_astnode_type(func_name), CYPHER_AST_FUNCTION_NAME);
    ck_assert_str_eq(cypher_ast_function_name_get_value(func_name), "sum");

    ck_assert(cypher_ast_apply_operator_get_distinct(exp));
    ck_assert_int_eq(cypher_ast_apply_operator_narguments(exp), 1);
    arg = cypher_ast_apply_operator_get_argument(exp, 0);
    ck_assert_int_eq(cypher_astnode_type(arg), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(arg), "a");
    ck_assert_ptr_eq(cypher_ast_apply_operator_get_argument(exp, 1), NULL);

    proj = cypher_ast_return_get_projection(clause, 2);
    ck_assert_int_eq(cypher_astnode_type(proj), CYPHER_AST_PROJECTION);
    exp = cypher_ast_projection_get_expression(proj);
    ck_assert_int_eq(cypher_astnode_type(exp), CYPHER_AST_APPLY_ALL_OPERATOR);

    func_name = cypher_ast_apply_all_operator_get_func_name(exp);
    ck_assert_int_eq(cypher_astnode_type(func_name), CYPHER_AST_FUNCTION_NAME);
    ck_assert_str_eq(cypher_ast_function_name_get_value(func_name), "count");

    ck_assert(!cypher_ast_apply_all_operator_get_distinct(exp));

    proj = cypher_ast_return_get_projection(clause, 3);
    ck_assert_int_eq(cypher_astnode_type(proj), CYPHER_AST_PROJECTION);
    exp = cypher_ast_projection_get_expression(proj);
    ck_assert_int_eq(cypher_astnode_type(exp), CYPHER_AST_APPLY_ALL_OPERATOR);

    func_name = cypher_ast_apply_all_operator_get_func_name(exp);
    ck_assert_int_eq(cypher_astnode_type(func_name), CYPHER_AST_FUNCTION_NAME);
    ck_assert_str_eq(cypher_ast_function_name_get_value(func_name), "count");

    ck_assert(cypher_ast_apply_all_operator_get_distinct(exp));

    ck_assert_ptr_eq(cypher_ast_return_get_projection(clause, 4), NULL);
}
END_TEST


START_TEST (parse_namespaced_apply)
{
    struct cypher_input_position last = cypher_input_position_zero;
    result = cypher_parse("RETURN fuz .foo(bar, baz);", &last, NULL, 0);
    ck_assert_ptr_ne(result, NULL);
    ck_assert_int_eq(last.offset, 26);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
"@0   0..26  statement                body=@1\n"
"@1   0..26  > query                  clauses=[@2]\n"
"@2   0..25  > > RETURN               projections=[@3]\n"
"@3   7..25  > > > projection         expression=@4, alias=@8\n"
"@4   7..25  > > > > apply            @5(@6, @7)\n"
"@5   7..15  > > > > > function name  `fuz.foo`\n"
"@6  16..19  > > > > > identifier     `bar`\n"
"@7  21..24  > > > > > identifier     `baz`\n"
"@8   7..25  > > > > identifier       `fuz .foo(bar, baz)`\n";
    ck_assert_str_eq(memstream_buffer, expected);

    const cypher_astnode_t *ast = cypher_parse_result_get_directive(result, 0);
    const cypher_astnode_t *query = cypher_ast_statement_get_body(ast);
    const cypher_astnode_t *clause = cypher_ast_query_get_clause(query, 0);
    ck_assert_int_eq(cypher_astnode_type(clause), CYPHER_AST_RETURN);

    ck_assert_int_eq(cypher_ast_return_nprojections(clause), 1);

    const cypher_astnode_t *proj = cypher_ast_return_get_projection(clause, 0);
    ck_assert_int_eq(cypher_astnode_type(proj), CYPHER_AST_PROJECTION);
    const cypher_astnode_t *exp = cypher_ast_projection_get_expression(proj);
    ck_assert_int_eq(cypher_astnode_type(exp), CYPHER_AST_APPLY_OPERATOR);

    const cypher_astnode_t *func_name =
            cypher_ast_apply_operator_get_func_name(exp);
    ck_assert_int_eq(cypher_astnode_type(func_name), CYPHER_AST_FUNCTION_NAME);
    ck_assert_str_eq(cypher_ast_function_name_get_value(func_name), "fuz.foo");

    ck_assert(!cypher_ast_apply_operator_get_distinct(exp));
    ck_assert_int_eq(cypher_ast_apply_operator_narguments(exp), 2);
    const cypher_astnode_t *arg = cypher_ast_apply_operator_get_argument(exp, 0);
    ck_assert_int_eq(cypher_astnode_type(arg), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(arg), "bar");
    arg = cypher_ast_apply_operator_get_argument(exp, 1);
    ck_assert_int_eq(cypher_astnode_type(arg), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(arg), "baz");
    ck_assert_ptr_eq(cypher_ast_apply_operator_get_argument(exp, 2), NULL);
}
END_TEST


START_TEST (parse_subscript)
{
    struct cypher_input_position last = cypher_input_position_zero;
    result = cypher_parse("RETURN foo[n];", &last, NULL, 0);
    ck_assert_ptr_ne(result, NULL);
    ck_assert_int_eq(last.offset, 14);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
"@0   0..14  statement             body=@1\n"
"@1   0..14  > query               clauses=[@2]\n"
"@2   0..13  > > RETURN            projections=[@3]\n"
"@3   7..13  > > > projection      expression=@4, alias=@7\n"
"@4   7..13  > > > > subscript     @5[@6]\n"
"@5   7..10  > > > > > identifier  `foo`\n"
"@6  11..12  > > > > > identifier  `n`\n"
"@7   7..13  > > > > identifier    `foo[n]`\n";
    ck_assert_str_eq(memstream_buffer, expected);

    const cypher_astnode_t *ast = cypher_parse_result_get_directive(result, 0);
    const cypher_astnode_t *query = cypher_ast_statement_get_body(ast);
    const cypher_astnode_t *clause = cypher_ast_query_get_clause(query, 0);
    ck_assert_int_eq(cypher_astnode_type(clause), CYPHER_AST_RETURN);

    ck_assert_int_eq(cypher_ast_return_nprojections(clause), 1);

    const cypher_astnode_t *proj = cypher_ast_return_get_projection(clause, 0);
    ck_assert_int_eq(cypher_astnode_type(proj), CYPHER_AST_PROJECTION);
    const cypher_astnode_t *exp = cypher_ast_projection_get_expression(proj);
    ck_assert_int_eq(cypher_astnode_type(exp), CYPHER_AST_SUBSCRIPT_OPERATOR);

    const cypher_astnode_t *node = cypher_ast_subscript_operator_get_expression(exp);
    ck_assert_int_eq(cypher_astnode_type(node), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(node), "foo");

    node = cypher_ast_subscript_operator_get_subscript(exp);
    ck_assert_int_eq(cypher_astnode_type(node), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(node), "n");
}
END_TEST


START_TEST (parse_slice)
{
    struct cypher_input_position last = cypher_input_position_zero;
    result = cypher_parse("RETURN foo[1..5], bar[..n+5], baz[..];", &last, NULL, 0);
    ck_assert_ptr_ne(result, NULL);
    ck_assert_int_eq(last.offset, 38);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
" @0   0..38  statement                  body=@1\n"
" @1   0..38  > query                    clauses=[@2]\n"
" @2   0..37  > > RETURN                 projections=[@3, @9, @16]\n"
" @3   7..16  > > > projection           expression=@4, alias=@8\n"
" @4   7..16  > > > > slice              @5[@6..@7]\n"
" @5   7..10  > > > > > identifier       `foo`\n"
" @6  11..12  > > > > > integer          1\n"
" @7  14..15  > > > > > integer          5\n"
" @8   7..16  > > > > identifier         `foo[1..5]`\n"
" @9  18..28  > > > projection           expression=@10, alias=@15\n"
"@10  18..28  > > > > slice              @11[..@12]\n"
"@11  18..21  > > > > > identifier       `bar`\n"
"@12  24..27  > > > > > binary operator  @13 + @14\n"
"@13  24..25  > > > > > > identifier     `n`\n"
"@14  26..27  > > > > > > integer        5\n"
"@15  18..28  > > > > identifier         `bar[..n+5]`\n"
"@16  30..37  > > > projection           expression=@17, alias=@19\n"
"@17  30..37  > > > > slice              @18[..]\n"
"@18  30..33  > > > > > identifier       `baz`\n"
"@19  30..37  > > > > identifier         `baz[..]`\n";
    ck_assert_str_eq(memstream_buffer, expected);

    const cypher_astnode_t *ast = cypher_parse_result_get_directive(result, 0);
    const cypher_astnode_t *query = cypher_ast_statement_get_body(ast);
    const cypher_astnode_t *clause = cypher_ast_query_get_clause(query, 0);
    ck_assert_int_eq(cypher_astnode_type(clause), CYPHER_AST_RETURN);

    ck_assert_int_eq(cypher_ast_return_nprojections(clause), 3);

    const cypher_astnode_t *proj = cypher_ast_return_get_projection(clause, 0);
    ck_assert_int_eq(cypher_astnode_type(proj), CYPHER_AST_PROJECTION);
    const cypher_astnode_t *exp = cypher_ast_projection_get_expression(proj);
    ck_assert_int_eq(cypher_astnode_type(exp), CYPHER_AST_SLICE_OPERATOR);

    const cypher_astnode_t *node = cypher_ast_slice_operator_get_expression(exp);
    ck_assert_int_eq(cypher_astnode_type(node), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(node), "foo");

    const cypher_astnode_t *start = cypher_ast_slice_operator_get_start(exp);
    ck_assert_int_eq(cypher_astnode_type(start), CYPHER_AST_INTEGER);
    ck_assert_str_eq(cypher_ast_integer_get_valuestr(start), "1");
    const cypher_astnode_t *end = cypher_ast_slice_operator_get_end(exp);
    ck_assert_int_eq(cypher_astnode_type(end), CYPHER_AST_INTEGER);
    ck_assert_str_eq(cypher_ast_integer_get_valuestr(end), "5");

    proj = cypher_ast_return_get_projection(clause, 1);
    ck_assert_int_eq(cypher_astnode_type(proj), CYPHER_AST_PROJECTION);
    exp = cypher_ast_projection_get_expression(proj);
    ck_assert_int_eq(cypher_astnode_type(exp), CYPHER_AST_SLICE_OPERATOR);

    node = cypher_ast_slice_operator_get_expression(exp);
    ck_assert_int_eq(cypher_astnode_type(node), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(node), "bar");

    start = cypher_ast_slice_operator_get_start(exp);
    ck_assert_ptr_eq(start, NULL);
    end = cypher_ast_slice_operator_get_end(exp);
    ck_assert_int_eq(cypher_astnode_type(end), CYPHER_AST_BINARY_OPERATOR);

    proj = cypher_ast_return_get_projection(clause, 2);
    ck_assert_int_eq(cypher_astnode_type(proj), CYPHER_AST_PROJECTION);
    exp = cypher_ast_projection_get_expression(proj);
    ck_assert_int_eq(cypher_astnode_type(exp), CYPHER_AST_SLICE_OPERATOR);

    node = cypher_ast_slice_operator_get_expression(exp);
    ck_assert_int_eq(cypher_astnode_type(node), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(node), "baz");

    start = cypher_ast_slice_operator_get_start(exp);
    ck_assert_ptr_eq(start, NULL);
    end = cypher_ast_slice_operator_get_end(exp);
    ck_assert_ptr_eq(end, NULL);
}
END_TEST

START_TEST (parse_subscript_list_with_in_operator)
{
    struct cypher_input_position last = cypher_input_position_zero;
    result = cypher_parse("RETURN 3 IN [[1,2,3]][0] AS r",
            &last, NULL, 0);
    ck_assert_ptr_ne(result, NULL);
    ck_assert_int_eq(last.offset, 29);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
" @0   0..29  statement                 body=@1\n"
" @1   0..29  > query                   clauses=[@2]\n"
" @2   0..29  > > RETURN                projections=[@3]\n"
" @3   7..29  > > > projection          expression=@4, alias=@13\n"
" @4   7..25  > > > > binary operator   @5 IN @6\n"
" @5   7..8   > > > > > integer         3\n"
" @6  12..25  > > > > > subscript       @7[@12]\n"
" @7  12..21  > > > > > > collection    [@8]\n"
" @8  13..20  > > > > > > > collection  [@9, @10, @11]\n"
" @9  14..15  > > > > > > > > integer   1\n"
"@10  16..17  > > > > > > > > integer   2\n"
"@11  18..19  > > > > > > > > integer   3\n"
"@12  22..23  > > > > > > integer       0\n"
"@13  28..29  > > > > identifier        `r`\n";
    ck_assert_str_eq(memstream_buffer, expected);

    const cypher_astnode_t *ast = cypher_parse_result_get_directive(result, 0);
    const cypher_astnode_t *query = cypher_ast_statement_get_body(ast);
    const cypher_astnode_t *clause = cypher_ast_query_get_clause(query, 0);
    ck_assert_int_eq(cypher_astnode_type(clause), CYPHER_AST_RETURN);
    ck_assert_int_eq(cypher_ast_return_nprojections(clause), 1);

    const cypher_astnode_t *proj = cypher_ast_return_get_projection(clause, 0);
    ck_assert_int_eq(cypher_astnode_type(proj), CYPHER_AST_PROJECTION);

    const cypher_astnode_t *identifier = cypher_ast_projection_get_alias(proj);
    ck_assert_int_eq(cypher_astnode_type(identifier), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(identifier), "r");

    const cypher_astnode_t *exp = cypher_ast_projection_get_expression(proj);
    ck_assert_int_eq(cypher_astnode_type(exp), CYPHER_AST_BINARY_OPERATOR);

    const cypher_operator_t *op = cypher_ast_binary_operator_get_operator(exp);
    ck_assert_ptr_eq(op, CYPHER_OP_IN);

    const cypher_astnode_t *arg1 = cypher_ast_binary_operator_get_argument1(exp);
    ck_assert_int_eq(cypher_astnode_type(arg1), CYPHER_AST_INTEGER);
    ck_assert_str_eq(cypher_ast_integer_get_valuestr(arg1), "3");

    const cypher_astnode_t *arg2 = cypher_ast_binary_operator_get_argument2(exp);
    ck_assert_int_eq(cypher_astnode_type(arg2), CYPHER_AST_SUBSCRIPT_OPERATOR);

    const cypher_astnode_t *subscript = cypher_ast_subscript_operator_get_subscript(arg2);
    ck_assert_int_eq(cypher_astnode_type(subscript), CYPHER_AST_INTEGER);
    ck_assert_str_eq(cypher_ast_integer_get_valuestr(subscript), "0");

    exp = cypher_ast_subscript_operator_get_expression(arg2);
    ck_assert_int_eq(cypher_astnode_type(exp), CYPHER_AST_COLLECTION);
    ck_assert_int_eq(cypher_ast_collection_length(exp), 1);

    exp = cypher_ast_collection_get(exp, 0);
    ck_assert_int_eq(cypher_astnode_type(exp), CYPHER_AST_COLLECTION);
    ck_assert_int_eq(cypher_ast_collection_length(exp), 3);

    for(uint i = 0; i < 3; i++) {
        char snum[2];
        sprintf(snum, "%d", i+1);
        const cypher_astnode_t *constant = cypher_ast_collection_get(exp, i);
        ck_assert_int_eq(cypher_astnode_type(constant), CYPHER_AST_INTEGER);
        ck_assert_str_eq(cypher_ast_integer_get_valuestr(constant), snum);
    }
}
END_TEST

TCase* expression_tcase(void)
{
    TCase *tc = tcase_create("expression");
    tcase_add_checked_fixture(tc, setup, teardown);
    tcase_add_test(tc, parse_simple_unary_operators);
    tcase_add_test(tc, parse_simple_binary_operators);
    tcase_add_test(tc, parse_unary_and_binary_operators);
    tcase_add_test(tc, parse_comparison_operators);
    tcase_add_test(tc, parse_apply);
    tcase_add_test(tc, parse_namespaced_apply);
    tcase_add_test(tc, parse_subscript);
    tcase_add_test(tc, parse_slice);
    tcase_add_test(tc, parse_subscript_list_with_in_operator);
    return tc;
}
