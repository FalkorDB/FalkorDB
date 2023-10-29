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


START_TEST (parse_empty_list_comprehension)
{
    struct cypher_input_position last = cypher_input_position_zero;
    result = cypher_parse("RETURN [ x in list /* nothing */ ];",
            &last, NULL, 0);
    ck_assert_ptr_ne(result, NULL);
    ck_assert_int_eq(last.offset, 35);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
"@0   0..35  statement                   body=@1\n"
"@1   0..35  > query                     clauses=[@2]\n"
"@2   0..34  > > RETURN                  projections=[@3]\n"
"@3   7..34  > > > projection            expression=@4, alias=@8\n"
"@4   7..34  > > > > list comprehension  [@5 IN @6]\n"
"@5   9..10  > > > > > identifier        `x`\n"
"@6  14..18  > > > > > identifier        `list`\n"
"@7  21..30  > > > > > block_comment     /* nothing */\n"
"@8   7..34  > > > > identifier          `[ x in list /* nothing */ ]`\n";
    ck_assert_str_eq(memstream_buffer, expected);

    const cypher_astnode_t *ast = cypher_parse_result_get_directive(result, 0);
    const cypher_astnode_t *query = cypher_ast_statement_get_body(ast);
    const cypher_astnode_t *clause = cypher_ast_query_get_clause(query, 0);
    ck_assert_int_eq(cypher_astnode_type(clause), CYPHER_AST_RETURN);
    ck_assert_int_eq(cypher_ast_return_nprojections(clause), 1);

    const cypher_astnode_t *proj = cypher_ast_return_get_projection(clause, 0);
    ck_assert_int_eq(cypher_astnode_type(proj), CYPHER_AST_PROJECTION);
    const cypher_astnode_t *exp = cypher_ast_projection_get_expression(proj);
    ck_assert_int_eq(cypher_astnode_type(exp), CYPHER_AST_LIST_COMPREHENSION);

    const cypher_astnode_t *id = cypher_ast_list_comprehension_get_identifier(exp);
    ck_assert_int_eq(cypher_astnode_type(id), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(id), "x");

    const cypher_astnode_t *node = cypher_ast_list_comprehension_get_expression(exp);
    ck_assert_int_eq(cypher_astnode_type(node), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(node), "list");

    ck_assert_ptr_eq(cypher_ast_list_comprehension_get_predicate(exp), NULL);
    ck_assert_ptr_eq(cypher_ast_list_comprehension_get_eval(exp), NULL);
}
END_TEST


START_TEST (parse_full_list_comprehension)
{
    struct cypher_input_position last = cypher_input_position_zero;
    result = cypher_parse("RETURN [x in list WHERE x.foo < 10 | x.bar ];",
            &last, NULL, 0);
    ck_assert_ptr_ne(result, NULL);
    ck_assert_int_eq(last.offset, 45);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
" @0   0..45  statement                   body=@1\n"
" @1   0..45  > query                     clauses=[@2]\n"
" @2   0..44  > > RETURN                  projections=[@3]\n"
" @3   7..44  > > > projection            expression=@4, alias=@15\n"
" @4   7..44  > > > > list comprehension  [@5 IN @6 WHERE @7 | @12]\n"
" @5   8..9   > > > > > identifier        `x`\n"
" @6  13..17  > > > > > identifier        `list`\n"
" @7  24..35  > > > > > comparison        @8 < @11\n"
" @8  24..30  > > > > > > property        @9.@10\n"
" @9  24..25  > > > > > > > identifier    `x`\n"
"@10  26..29  > > > > > > > prop name     `foo`\n"
"@11  32..34  > > > > > > integer         10\n"
"@12  37..43  > > > > > property          @13.@14\n"
"@13  37..38  > > > > > > identifier      `x`\n"
"@14  39..42  > > > > > > prop name       `bar`\n"
"@15   7..44  > > > > identifier          `[x in list WHERE x.foo < 10 | x.bar ]`\n";
    ck_assert_str_eq(memstream_buffer, expected);

    const cypher_astnode_t *ast = cypher_parse_result_get_directive(result, 0);
    const cypher_astnode_t *query = cypher_ast_statement_get_body(ast);
    const cypher_astnode_t *clause = cypher_ast_query_get_clause(query, 0);
    ck_assert_int_eq(cypher_astnode_type(clause), CYPHER_AST_RETURN);
    ck_assert_int_eq(cypher_ast_return_nprojections(clause), 1);

    const cypher_astnode_t *proj = cypher_ast_return_get_projection(clause, 0);
    ck_assert_int_eq(cypher_astnode_type(proj), CYPHER_AST_PROJECTION);
    const cypher_astnode_t *exp = cypher_ast_projection_get_expression(proj);
    ck_assert_int_eq(cypher_astnode_type(exp), CYPHER_AST_LIST_COMPREHENSION);

    const cypher_astnode_t *id = cypher_ast_list_comprehension_get_identifier(exp);
    ck_assert_int_eq(cypher_astnode_type(id), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(id), "x");

    const cypher_astnode_t *node = cypher_ast_list_comprehension_get_expression(exp);
    ck_assert_int_eq(cypher_astnode_type(node), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(node), "list");

    const cypher_astnode_t *pred = cypher_ast_list_comprehension_get_predicate(exp);
    ck_assert_int_eq(cypher_astnode_type(pred), CYPHER_AST_COMPARISON);

    const cypher_astnode_t *eval = cypher_ast_list_comprehension_get_eval(exp);
    ck_assert_int_eq(cypher_astnode_type(eval), CYPHER_AST_PROPERTY_OPERATOR);
}
END_TEST


START_TEST (parse_filter)
{
    struct cypher_input_position last = cypher_input_position_zero;
    result = cypher_parse("RETURN filter(x in list WHERE x.foo < 10);",
            &last, NULL, 0);
    ck_assert_ptr_ne(result, NULL);
    ck_assert_int_eq(last.offset, 42);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
" @0   0..42  statement                 body=@1\n"
" @1   0..42  > query                   clauses=[@2]\n"
" @2   0..41  > > RETURN                projections=[@3]\n"
" @3   7..41  > > > projection          expression=@4, alias=@12\n"
" @4   7..41  > > > > filter            [@5 IN @6 WHERE @7]\n"
" @5  14..15  > > > > > identifier      `x`\n"
" @6  19..23  > > > > > identifier      `list`\n"
" @7  30..40  > > > > > comparison      @8 < @11\n"
" @8  30..36  > > > > > > property      @9.@10\n"
" @9  30..31  > > > > > > > identifier  `x`\n"
"@10  32..35  > > > > > > > prop name   `foo`\n"
"@11  38..40  > > > > > > integer       10\n"
"@12   7..41  > > > > identifier        `filter(x in list WHERE x.foo < 10)`\n";
    ck_assert_str_eq(memstream_buffer, expected);

    const cypher_astnode_t *ast = cypher_parse_result_get_directive(result, 0);
    const cypher_astnode_t *query = cypher_ast_statement_get_body(ast);
    const cypher_astnode_t *clause = cypher_ast_query_get_clause(query, 0);
    ck_assert_int_eq(cypher_astnode_type(clause), CYPHER_AST_RETURN);
    ck_assert_int_eq(cypher_ast_return_nprojections(clause), 1);

    const cypher_astnode_t *proj = cypher_ast_return_get_projection(clause, 0);
    ck_assert_int_eq(cypher_astnode_type(proj), CYPHER_AST_PROJECTION);
    const cypher_astnode_t *exp = cypher_ast_projection_get_expression(proj);
    ck_assert(cypher_astnode_instanceof(exp, CYPHER_AST_LIST_COMPREHENSION));
    ck_assert_int_eq(cypher_astnode_type(exp), CYPHER_AST_FILTER);

    const cypher_astnode_t *id = cypher_ast_list_comprehension_get_identifier(exp);
    ck_assert_int_eq(cypher_astnode_type(id), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(id), "x");

    const cypher_astnode_t *node = cypher_ast_list_comprehension_get_expression(exp);
    ck_assert_int_eq(cypher_astnode_type(node), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(node), "list");

    const cypher_astnode_t *pred = cypher_ast_list_comprehension_get_predicate(exp);
    ck_assert_int_eq(cypher_astnode_type(pred), CYPHER_AST_COMPARISON);

    ck_assert_ptr_eq(cypher_ast_list_comprehension_get_eval(exp), NULL);
}
END_TEST


START_TEST (parse_extract)
{
    struct cypher_input_position last = cypher_input_position_zero;
    result = cypher_parse("RETURN extract(x in list | x.bar);",
            &last, NULL, 0);
    ck_assert_ptr_ne(result, NULL);
    ck_assert_int_eq(last.offset, 34);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
" @0   0..34  statement               body=@1\n"
" @1   0..34  > query                 clauses=[@2]\n"
" @2   0..33  > > RETURN              projections=[@3]\n"
" @3   7..33  > > > projection        expression=@4, alias=@10\n"
" @4   7..33  > > > > extract         [@5 IN @6 | @7]\n"
" @5  15..16  > > > > > identifier    `x`\n"
" @6  20..24  > > > > > identifier    `list`\n"
" @7  27..32  > > > > > property      @8.@9\n"
" @8  27..28  > > > > > > identifier  `x`\n"
" @9  29..32  > > > > > > prop name   `bar`\n"
"@10   7..33  > > > > identifier      `extract(x in list | x.bar)`\n";
    ck_assert_str_eq(memstream_buffer, expected);

    const cypher_astnode_t *ast = cypher_parse_result_get_directive(result, 0);
    const cypher_astnode_t *query = cypher_ast_statement_get_body(ast);
    const cypher_astnode_t *clause = cypher_ast_query_get_clause(query, 0);
    ck_assert_int_eq(cypher_astnode_type(clause), CYPHER_AST_RETURN);
    ck_assert_int_eq(cypher_ast_return_nprojections(clause), 1);

    const cypher_astnode_t *proj = cypher_ast_return_get_projection(clause, 0);
    ck_assert_int_eq(cypher_astnode_type(proj), CYPHER_AST_PROJECTION);
    const cypher_astnode_t *exp = cypher_ast_projection_get_expression(proj);
    ck_assert(cypher_astnode_instanceof(exp, CYPHER_AST_LIST_COMPREHENSION));
    ck_assert_int_eq(cypher_astnode_type(exp), CYPHER_AST_EXTRACT);

    const cypher_astnode_t *id = cypher_ast_list_comprehension_get_identifier(exp);
    ck_assert_int_eq(cypher_astnode_type(id), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(id), "x");

    const cypher_astnode_t *node = cypher_ast_list_comprehension_get_expression(exp);
    ck_assert_int_eq(cypher_astnode_type(node), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(node), "list");

    ck_assert_ptr_eq(cypher_ast_list_comprehension_get_predicate(exp), NULL);

    const cypher_astnode_t *eval = cypher_ast_list_comprehension_get_eval(exp);
    ck_assert_int_eq(cypher_astnode_type(eval), CYPHER_AST_PROPERTY_OPERATOR);
}
END_TEST


START_TEST (parse_all)
{
    struct cypher_input_position last = cypher_input_position_zero;
    result = cypher_parse("RETURN all(x in list WHERE x.foo < 10);",
            &last, NULL, 0);
    ck_assert_ptr_ne(result, NULL);
    ck_assert_int_eq(last.offset, 39);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
" @0   0..39  statement                 body=@1\n"
" @1   0..39  > query                   clauses=[@2]\n"
" @2   0..38  > > RETURN                projections=[@3]\n"
" @3   7..38  > > > projection          expression=@4, alias=@12\n"
" @4   7..38  > > > > all               [@5 IN @6 WHERE @7]\n"
" @5  11..12  > > > > > identifier      `x`\n"
" @6  16..20  > > > > > identifier      `list`\n"
" @7  27..37  > > > > > comparison      @8 < @11\n"
" @8  27..33  > > > > > > property      @9.@10\n"
" @9  27..28  > > > > > > > identifier  `x`\n"
"@10  29..32  > > > > > > > prop name   `foo`\n"
"@11  35..37  > > > > > > integer       10\n"
"@12   7..38  > > > > identifier        `all(x in list WHERE x.foo < 10)`\n";
    ck_assert_str_eq(memstream_buffer, expected);

    const cypher_astnode_t *ast = cypher_parse_result_get_directive(result, 0);
    const cypher_astnode_t *query = cypher_ast_statement_get_body(ast);
    const cypher_astnode_t *clause = cypher_ast_query_get_clause(query, 0);
    ck_assert_int_eq(cypher_astnode_type(clause), CYPHER_AST_RETURN);
    ck_assert_int_eq(cypher_ast_return_nprojections(clause), 1);

    const cypher_astnode_t *proj = cypher_ast_return_get_projection(clause, 0);
    ck_assert_int_eq(cypher_astnode_type(proj), CYPHER_AST_PROJECTION);
    const cypher_astnode_t *exp = cypher_ast_projection_get_expression(proj);
    ck_assert(cypher_astnode_instanceof(exp, CYPHER_AST_LIST_COMPREHENSION));
    ck_assert_int_eq(cypher_astnode_type(exp), CYPHER_AST_ALL);

    const cypher_astnode_t *id = cypher_ast_list_comprehension_get_identifier(exp);
    ck_assert_int_eq(cypher_astnode_type(id), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(id), "x");

    const cypher_astnode_t *node = cypher_ast_list_comprehension_get_expression(exp);
    ck_assert_int_eq(cypher_astnode_type(node), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(node), "list");

    const cypher_astnode_t *pred = cypher_ast_list_comprehension_get_predicate(exp);
    ck_assert_int_eq(cypher_astnode_type(pred), CYPHER_AST_COMPARISON);

    ck_assert_ptr_eq(cypher_ast_list_comprehension_get_eval(exp), NULL);
}
END_TEST


START_TEST (parse_any)
{
    struct cypher_input_position last = cypher_input_position_zero;
    result = cypher_parse("RETURN any(x in list WHERE x.foo < 10);",
            &last, NULL, 0);
    ck_assert_ptr_ne(result, NULL);
    ck_assert_int_eq(last.offset, 39);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
" @0   0..39  statement                 body=@1\n"
" @1   0..39  > query                   clauses=[@2]\n"
" @2   0..38  > > RETURN                projections=[@3]\n"
" @3   7..38  > > > projection          expression=@4, alias=@12\n"
" @4   7..38  > > > > any               [@5 IN @6 WHERE @7]\n"
" @5  11..12  > > > > > identifier      `x`\n"
" @6  16..20  > > > > > identifier      `list`\n"
" @7  27..37  > > > > > comparison      @8 < @11\n"
" @8  27..33  > > > > > > property      @9.@10\n"
" @9  27..28  > > > > > > > identifier  `x`\n"
"@10  29..32  > > > > > > > prop name   `foo`\n"
"@11  35..37  > > > > > > integer       10\n"
"@12   7..38  > > > > identifier        `any(x in list WHERE x.foo < 10)`\n";
    ck_assert_str_eq(memstream_buffer, expected);

    const cypher_astnode_t *ast = cypher_parse_result_get_directive(result, 0);
    const cypher_astnode_t *query = cypher_ast_statement_get_body(ast);
    const cypher_astnode_t *clause = cypher_ast_query_get_clause(query, 0);
    ck_assert_int_eq(cypher_astnode_type(clause), CYPHER_AST_RETURN);
    ck_assert_int_eq(cypher_ast_return_nprojections(clause), 1);

    const cypher_astnode_t *proj = cypher_ast_return_get_projection(clause, 0);
    ck_assert_int_eq(cypher_astnode_type(proj), CYPHER_AST_PROJECTION);
    const cypher_astnode_t *exp = cypher_ast_projection_get_expression(proj);
    ck_assert(cypher_astnode_instanceof(exp, CYPHER_AST_LIST_COMPREHENSION));
    ck_assert_int_eq(cypher_astnode_type(exp), CYPHER_AST_ANY);

    const cypher_astnode_t *id = cypher_ast_list_comprehension_get_identifier(exp);
    ck_assert_int_eq(cypher_astnode_type(id), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(id), "x");

    const cypher_astnode_t *node = cypher_ast_list_comprehension_get_expression(exp);
    ck_assert_int_eq(cypher_astnode_type(node), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(node), "list");

    const cypher_astnode_t *pred = cypher_ast_list_comprehension_get_predicate(exp);
    ck_assert_int_eq(cypher_astnode_type(pred), CYPHER_AST_COMPARISON);

    ck_assert_ptr_eq(cypher_ast_list_comprehension_get_eval(exp), NULL);
}
END_TEST


START_TEST (parse_single)
{
    struct cypher_input_position last = cypher_input_position_zero;
    result = cypher_parse("RETURN single(x in list WHERE x.foo < 10);",
            &last, NULL, 0);
    ck_assert_ptr_ne(result, NULL);
    ck_assert_int_eq(last.offset, 42);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
" @0   0..42  statement                 body=@1\n"
" @1   0..42  > query                   clauses=[@2]\n"
" @2   0..41  > > RETURN                projections=[@3]\n"
" @3   7..41  > > > projection          expression=@4, alias=@12\n"
" @4   7..41  > > > > single            [@5 IN @6 WHERE @7]\n"
" @5  14..15  > > > > > identifier      `x`\n"
" @6  19..23  > > > > > identifier      `list`\n"
" @7  30..40  > > > > > comparison      @8 < @11\n"
" @8  30..36  > > > > > > property      @9.@10\n"
" @9  30..31  > > > > > > > identifier  `x`\n"
"@10  32..35  > > > > > > > prop name   `foo`\n"
"@11  38..40  > > > > > > integer       10\n"
"@12   7..41  > > > > identifier        `single(x in list WHERE x.foo < 10)`\n";
    ck_assert_str_eq(memstream_buffer, expected);

    const cypher_astnode_t *ast = cypher_parse_result_get_directive(result, 0);
    const cypher_astnode_t *query = cypher_ast_statement_get_body(ast);
    const cypher_astnode_t *clause = cypher_ast_query_get_clause(query, 0);
    ck_assert_int_eq(cypher_astnode_type(clause), CYPHER_AST_RETURN);
    ck_assert_int_eq(cypher_ast_return_nprojections(clause), 1);

    const cypher_astnode_t *proj = cypher_ast_return_get_projection(clause, 0);
    ck_assert_int_eq(cypher_astnode_type(proj), CYPHER_AST_PROJECTION);
    const cypher_astnode_t *exp = cypher_ast_projection_get_expression(proj);
    ck_assert(cypher_astnode_instanceof(exp, CYPHER_AST_LIST_COMPREHENSION));
    ck_assert_int_eq(cypher_astnode_type(exp), CYPHER_AST_SINGLE);

    const cypher_astnode_t *id = cypher_ast_list_comprehension_get_identifier(exp);
    ck_assert_int_eq(cypher_astnode_type(id), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(id), "x");

    const cypher_astnode_t *node = cypher_ast_list_comprehension_get_expression(exp);
    ck_assert_int_eq(cypher_astnode_type(node), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(node), "list");

    const cypher_astnode_t *pred = cypher_ast_list_comprehension_get_predicate(exp);
    ck_assert_int_eq(cypher_astnode_type(pred), CYPHER_AST_COMPARISON);

    ck_assert_ptr_eq(cypher_ast_list_comprehension_get_eval(exp), NULL);
}
END_TEST


START_TEST (parse_none)
{
    struct cypher_input_position last = cypher_input_position_zero;
    result = cypher_parse("RETURN none(x in list WHERE x.foo < 10);",
            &last, NULL, 0);
    ck_assert_ptr_ne(result, NULL);
    ck_assert_int_eq(last.offset, 40);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
" @0   0..40  statement                 body=@1\n"
" @1   0..40  > query                   clauses=[@2]\n"
" @2   0..39  > > RETURN                projections=[@3]\n"
" @3   7..39  > > > projection          expression=@4, alias=@12\n"
" @4   7..39  > > > > none              [@5 IN @6 WHERE @7]\n"
" @5  12..13  > > > > > identifier      `x`\n"
" @6  17..21  > > > > > identifier      `list`\n"
" @7  28..38  > > > > > comparison      @8 < @11\n"
" @8  28..34  > > > > > > property      @9.@10\n"
" @9  28..29  > > > > > > > identifier  `x`\n"
"@10  30..33  > > > > > > > prop name   `foo`\n"
"@11  36..38  > > > > > > integer       10\n"
"@12   7..39  > > > > identifier        `none(x in list WHERE x.foo < 10)`\n";
    ck_assert_str_eq(memstream_buffer, expected);

    const cypher_astnode_t *ast = cypher_parse_result_get_directive(result, 0);
    const cypher_astnode_t *query = cypher_ast_statement_get_body(ast);
    const cypher_astnode_t *clause = cypher_ast_query_get_clause(query, 0);
    ck_assert_int_eq(cypher_astnode_type(clause), CYPHER_AST_RETURN);
    ck_assert_int_eq(cypher_ast_return_nprojections(clause), 1);

    const cypher_astnode_t *proj = cypher_ast_return_get_projection(clause, 0);
    ck_assert_int_eq(cypher_astnode_type(proj), CYPHER_AST_PROJECTION);
    const cypher_astnode_t *exp = cypher_ast_projection_get_expression(proj);
    ck_assert(cypher_astnode_instanceof(exp, CYPHER_AST_LIST_COMPREHENSION));
    ck_assert_int_eq(cypher_astnode_type(exp), CYPHER_AST_NONE);

    const cypher_astnode_t *id = cypher_ast_list_comprehension_get_identifier(exp);
    ck_assert_int_eq(cypher_astnode_type(id), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(id), "x");

    const cypher_astnode_t *node = cypher_ast_list_comprehension_get_expression(exp);
    ck_assert_int_eq(cypher_astnode_type(node), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(node), "list");

    const cypher_astnode_t *pred = cypher_ast_list_comprehension_get_predicate(exp);
    ck_assert_int_eq(cypher_astnode_type(pred), CYPHER_AST_COMPARISON);

    ck_assert_ptr_eq(cypher_ast_list_comprehension_get_eval(exp), NULL);
}
END_TEST


TCase* list_comprehensions_tcase(void)
{
    TCase *tc = tcase_create("list comprehensions");
    tcase_add_checked_fixture(tc, setup, teardown);
    tcase_add_test(tc, parse_empty_list_comprehension);
    tcase_add_test(tc, parse_full_list_comprehension);
    tcase_add_test(tc, parse_filter);
    tcase_add_test(tc, parse_extract);
    tcase_add_test(tc, parse_all);
    tcase_add_test(tc, parse_any);
    tcase_add_test(tc, parse_single);
    tcase_add_test(tc, parse_none);
    return tc;
}
