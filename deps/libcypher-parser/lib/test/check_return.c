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


START_TEST (parse_simple_return)
{
    struct cypher_input_position last = cypher_input_position_zero;
    result = cypher_parse(
            "/* MATCH */ RETURN 1 AS x, 'bar' AS y",
            &last, NULL, 0);
    ck_assert_ptr_ne(result, NULL);
    ck_assert_int_eq(last.offset, 37);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
"@0   2..9   block_comment       /* MATCH */\n"
"@1  12..37  statement           body=@2\n"
"@2  12..37  > query             clauses=[@3]\n"
"@3  12..37  > > RETURN          projections=[@4, @7]\n"
"@4  19..25  > > > projection    expression=@5, alias=@6\n"
"@5  19..20  > > > > integer     1\n"
"@6  24..25  > > > > identifier  `x`\n"
"@7  27..37  > > > projection    expression=@8, alias=@9\n"
"@8  27..32  > > > > string      \"bar\"\n"
"@9  36..37  > > > > identifier  `y`\n";
    ck_assert_str_eq(memstream_buffer, expected);

    const cypher_astnode_t *ast = cypher_parse_result_get_directive(result, 0);
    const cypher_astnode_t *query = cypher_ast_statement_get_body(ast);
    const cypher_astnode_t *clause = cypher_ast_query_get_clause(query, 0);
    ck_assert_int_eq(cypher_astnode_type(clause), CYPHER_AST_RETURN);

    ck_assert(!cypher_ast_return_is_distinct(clause));
    ck_assert(!cypher_ast_return_has_include_existing(clause));

    ck_assert_int_eq(cypher_ast_return_nprojections(clause), 2);
    ck_assert_ptr_eq(cypher_ast_return_get_projection(clause, 2), NULL);

    const cypher_astnode_t *proj = cypher_ast_return_get_projection(clause, 0);
    ck_assert_int_eq(cypher_astnode_type(proj), CYPHER_AST_PROJECTION);
    const cypher_astnode_t *exp = cypher_ast_projection_get_expression(proj);
    ck_assert_int_eq(cypher_astnode_type(exp), CYPHER_AST_INTEGER);
    ck_assert_str_eq(cypher_ast_integer_get_valuestr(exp), "1");
    const cypher_astnode_t *alias = cypher_ast_projection_get_alias(proj);
    ck_assert_int_eq(cypher_astnode_type(alias), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(alias), "x");

    proj = cypher_ast_return_get_projection(clause, 1);
    ck_assert_int_eq(cypher_astnode_type(proj), CYPHER_AST_PROJECTION);
    exp = cypher_ast_projection_get_expression(proj);
    ck_assert_int_eq(cypher_astnode_type(exp), CYPHER_AST_STRING);
    ck_assert_str_eq(cypher_ast_string_get_value(exp), "bar");
    alias = cypher_ast_projection_get_alias(proj);
    ck_assert_int_eq(cypher_astnode_type(alias), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(alias), "y");

    ck_assert_ptr_eq(cypher_ast_return_get_order_by(clause), NULL);
    ck_assert_ptr_eq(cypher_ast_return_get_skip(clause), NULL);
    ck_assert_ptr_eq(cypher_ast_return_get_limit(clause), NULL);
}
END_TEST


START_TEST (parse_distinct_return)
{
    struct cypher_input_position last = cypher_input_position_zero;
    result = cypher_parse(
            "/* MATCH */ RETURN DISTINCT 1 AS x, 'bar' AS y",
            &last, NULL, 0);
    ck_assert_ptr_ne(result, NULL);
    ck_assert_int_eq(last.offset, 46);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
"@0   2..9   block_comment       /* MATCH */\n"
"@1  12..46  statement           body=@2\n"
"@2  12..46  > query             clauses=[@3]\n"
"@3  12..46  > > RETURN          DISTINCT, projections=[@4, @7]\n"
"@4  28..34  > > > projection    expression=@5, alias=@6\n"
"@5  28..29  > > > > integer     1\n"
"@6  33..34  > > > > identifier  `x`\n"
"@7  36..46  > > > projection    expression=@8, alias=@9\n"
"@8  36..41  > > > > string      \"bar\"\n"
"@9  45..46  > > > > identifier  `y`\n";
    ck_assert_str_eq(memstream_buffer, expected);

    const cypher_astnode_t *ast = cypher_parse_result_get_directive(result, 0);
    const cypher_astnode_t *query = cypher_ast_statement_get_body(ast);
    const cypher_astnode_t *clause = cypher_ast_query_get_clause(query, 0);
    ck_assert_int_eq(cypher_astnode_type(clause), CYPHER_AST_RETURN);

    ck_assert(cypher_ast_return_is_distinct(clause));
    ck_assert(!cypher_ast_return_has_include_existing(clause));

    ck_assert_int_eq(cypher_ast_return_nprojections(clause), 2);
    ck_assert_ptr_eq(cypher_ast_return_get_projection(clause, 2), NULL);

    const cypher_astnode_t *proj = cypher_ast_return_get_projection(clause, 0);
    ck_assert_int_eq(cypher_astnode_type(proj), CYPHER_AST_PROJECTION);
    const cypher_astnode_t *exp = cypher_ast_projection_get_expression(proj);
    ck_assert_int_eq(cypher_astnode_type(exp), CYPHER_AST_INTEGER);
    ck_assert_str_eq(cypher_ast_integer_get_valuestr(exp), "1");
    const cypher_astnode_t *alias = cypher_ast_projection_get_alias(proj);
    ck_assert_int_eq(cypher_astnode_type(alias), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(alias), "x");

    proj = cypher_ast_return_get_projection(clause, 1);
    ck_assert_int_eq(cypher_astnode_type(proj), CYPHER_AST_PROJECTION);
    exp = cypher_ast_projection_get_expression(proj);
    ck_assert_int_eq(cypher_astnode_type(exp), CYPHER_AST_STRING);
    ck_assert_str_eq(cypher_ast_string_get_value(exp), "bar");
    alias = cypher_ast_projection_get_alias(proj);
    ck_assert_int_eq(cypher_astnode_type(alias), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(alias), "y");

    ck_assert_ptr_eq(cypher_ast_return_get_order_by(clause), NULL);
    ck_assert_ptr_eq(cypher_ast_return_get_skip(clause), NULL);
    ck_assert_ptr_eq(cypher_ast_return_get_limit(clause), NULL);
}
END_TEST


START_TEST (parse_return_including_existing)
{
    struct cypher_input_position last = cypher_input_position_zero;
    result = cypher_parse(
            "/* MATCH */ RETURN *, 1 AS x, 'bar' AS y",
            &last, NULL, 0);
    ck_assert_ptr_ne(result, NULL);
    ck_assert_int_eq(last.offset, 40);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
"@0   2..9   block_comment       /* MATCH */\n"
"@1  12..40  statement           body=@2\n"
"@2  12..40  > query             clauses=[@3]\n"
"@3  12..40  > > RETURN          *, projections=[@4, @7]\n"
"@4  22..28  > > > projection    expression=@5, alias=@6\n"
"@5  22..23  > > > > integer     1\n"
"@6  27..28  > > > > identifier  `x`\n"
"@7  30..40  > > > projection    expression=@8, alias=@9\n"
"@8  30..35  > > > > string      \"bar\"\n"
"@9  39..40  > > > > identifier  `y`\n";
    ck_assert_str_eq(memstream_buffer, expected);

    const cypher_astnode_t *ast = cypher_parse_result_get_directive(result, 0);
    const cypher_astnode_t *query = cypher_ast_statement_get_body(ast);
    const cypher_astnode_t *clause = cypher_ast_query_get_clause(query, 0);
    ck_assert_int_eq(cypher_astnode_type(clause), CYPHER_AST_RETURN);

    ck_assert(!cypher_ast_return_is_distinct(clause));
    ck_assert(cypher_ast_return_has_include_existing(clause));

    ck_assert_int_eq(cypher_ast_return_nprojections(clause), 2);
    ck_assert_ptr_eq(cypher_ast_return_get_projection(clause, 2), NULL);

    const cypher_astnode_t *proj = cypher_ast_return_get_projection(clause, 0);
    ck_assert_int_eq(cypher_astnode_type(proj), CYPHER_AST_PROJECTION);
    const cypher_astnode_t *exp = cypher_ast_projection_get_expression(proj);
    ck_assert_int_eq(cypher_astnode_type(exp), CYPHER_AST_INTEGER);
    ck_assert_str_eq(cypher_ast_integer_get_valuestr(exp), "1");
    const cypher_astnode_t *alias = cypher_ast_projection_get_alias(proj);
    ck_assert_int_eq(cypher_astnode_type(alias), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(alias), "x");

    proj = cypher_ast_return_get_projection(clause, 1);
    ck_assert_int_eq(cypher_astnode_type(proj), CYPHER_AST_PROJECTION);
    exp = cypher_ast_projection_get_expression(proj);
    ck_assert_int_eq(cypher_astnode_type(exp), CYPHER_AST_STRING);
    ck_assert_str_eq(cypher_ast_string_get_value(exp), "bar");
    alias = cypher_ast_projection_get_alias(proj);
    ck_assert_int_eq(cypher_astnode_type(alias), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(alias), "y");

    ck_assert_ptr_eq(cypher_ast_return_get_order_by(clause), NULL);
    ck_assert_ptr_eq(cypher_ast_return_get_skip(clause), NULL);
    ck_assert_ptr_eq(cypher_ast_return_get_limit(clause), NULL);
}
END_TEST


START_TEST (parse_return_and_order_by)
{
    struct cypher_input_position last = cypher_input_position_zero;
    result = cypher_parse(
            "RETURN 1 AS x, 'bar' AS y ORDER BY x DESC, y ASC, z.prop + 10",
            &last, NULL, 0);
    ck_assert_ptr_ne(result, NULL);
    ck_assert_int_eq(last.offset, 61);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
" @0   0..61  statement                  body=@1\n"
" @1   0..61  > query                    clauses=[@2]\n"
" @2   0..61  > > RETURN                 projections=[@3, @6], ORDER BY=@9\n"
" @3   7..13  > > > projection           expression=@4, alias=@5\n"
" @4   7..8   > > > > integer            1\n"
" @5  12..13  > > > > identifier         `x`\n"
" @6  15..26  > > > projection           expression=@7, alias=@8\n"
" @7  15..20  > > > > string             \"bar\"\n"
" @8  24..25  > > > > identifier         `y`\n"
" @9  26..61  > > > ORDER BY             items=[@10, @12, @14]\n"
"@10  35..41  > > > > sort item          expression=@11, DESCENDING\n"
"@11  35..36  > > > > > identifier       `x`\n"
"@12  43..48  > > > > sort item          expression=@13, ASCENDING\n"
"@13  43..44  > > > > > identifier       `y`\n"
"@14  50..61  > > > > sort item          expression=@15, ASCENDING\n"
"@15  50..61  > > > > > binary operator  @16 + @19\n"
"@16  50..57  > > > > > > property       @17.@18\n"
"@17  50..51  > > > > > > > identifier   `z`\n"
"@18  52..56  > > > > > > > prop name    `prop`\n"
"@19  59..61  > > > > > > integer        10\n";
    ck_assert_str_eq(memstream_buffer, expected);

    const cypher_astnode_t *ast = cypher_parse_result_get_directive(result, 0);
    const cypher_astnode_t *query = cypher_ast_statement_get_body(ast);
    const cypher_astnode_t *clause = cypher_ast_query_get_clause(query, 0);
    ck_assert_int_eq(cypher_astnode_type(clause), CYPHER_AST_RETURN);

    ck_assert(!cypher_ast_return_is_distinct(clause));
    ck_assert(!cypher_ast_return_has_include_existing(clause));

    ck_assert_int_eq(cypher_ast_return_nprojections(clause), 2);
    ck_assert_ptr_eq(cypher_ast_return_get_projection(clause, 2), NULL);

    const cypher_astnode_t *proj = cypher_ast_return_get_projection(clause, 0);
    ck_assert_int_eq(cypher_astnode_type(proj), CYPHER_AST_PROJECTION);
    const cypher_astnode_t *exp = cypher_ast_projection_get_expression(proj);
    ck_assert_int_eq(cypher_astnode_type(exp), CYPHER_AST_INTEGER);
    ck_assert_str_eq(cypher_ast_integer_get_valuestr(exp), "1");
    const cypher_astnode_t *alias = cypher_ast_projection_get_alias(proj);
    ck_assert_int_eq(cypher_astnode_type(alias), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(alias), "x");

    proj = cypher_ast_return_get_projection(clause, 1);
    ck_assert_int_eq(cypher_astnode_type(proj), CYPHER_AST_PROJECTION);
    exp = cypher_ast_projection_get_expression(proj);
    ck_assert_int_eq(cypher_astnode_type(exp), CYPHER_AST_STRING);
    ck_assert_str_eq(cypher_ast_string_get_value(exp), "bar");
    alias = cypher_ast_projection_get_alias(proj);
    ck_assert_int_eq(cypher_astnode_type(alias), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(alias), "y");

    const cypher_astnode_t *order = cypher_ast_return_get_order_by(clause);
    ck_assert_int_eq(cypher_astnode_type(order), CYPHER_AST_ORDER_BY);
    ck_assert_int_eq(cypher_ast_order_by_nitems(order), 3);
    ck_assert_ptr_eq(cypher_ast_order_by_get_item(order, 3), NULL);

    const cypher_astnode_t *item = cypher_ast_order_by_get_item(order, 0);
    ck_assert_int_eq(cypher_astnode_type(item), CYPHER_AST_SORT_ITEM);
    exp = cypher_ast_sort_item_get_expression(item);
    ck_assert_int_eq(cypher_astnode_type(exp), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(exp), "x");
    ck_assert(!cypher_ast_sort_item_is_ascending(item));

    item = cypher_ast_order_by_get_item(order, 1);
    ck_assert_int_eq(cypher_astnode_type(item), CYPHER_AST_SORT_ITEM);
    exp = cypher_ast_sort_item_get_expression(item);
    ck_assert_int_eq(cypher_astnode_type(exp), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(exp), "y");
    ck_assert(cypher_ast_sort_item_is_ascending(item));

    item = cypher_ast_order_by_get_item(order, 2);
    ck_assert_int_eq(cypher_astnode_type(item), CYPHER_AST_SORT_ITEM);
    exp = cypher_ast_sort_item_get_expression(item);
    ck_assert_int_eq(cypher_astnode_type(exp), CYPHER_AST_BINARY_OPERATOR);
    ck_assert(cypher_ast_sort_item_is_ascending(item));

    ck_assert_ptr_eq(cypher_ast_return_get_skip(clause), NULL);
    ck_assert_ptr_eq(cypher_ast_return_get_limit(clause), NULL);
}
END_TEST


START_TEST (parse_return_and_skip)
{
    struct cypher_input_position last = cypher_input_position_zero;
    result = cypher_parse(
            "RETURN *, 1 AS x, 'bar' AS y SKIP 10",
            &last, NULL, 0);
    ck_assert_ptr_ne(result, NULL);
    ck_assert_int_eq(last.offset, 36);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
"@0   0..36  statement           body=@1\n"
"@1   0..36  > query             clauses=[@2]\n"
"@2   0..36  > > RETURN          *, projections=[@3, @6], SKIP=@9\n"
"@3  10..16  > > > projection    expression=@4, alias=@5\n"
"@4  10..11  > > > > integer     1\n"
"@5  15..16  > > > > identifier  `x`\n"
"@6  18..29  > > > projection    expression=@7, alias=@8\n"
"@7  18..23  > > > > string      \"bar\"\n"
"@8  27..28  > > > > identifier  `y`\n"
"@9  34..36  > > > integer       10\n";
    ck_assert_str_eq(memstream_buffer, expected);

    const cypher_astnode_t *ast = cypher_parse_result_get_directive(result, 0);
    const cypher_astnode_t *query = cypher_ast_statement_get_body(ast);
    const cypher_astnode_t *clause = cypher_ast_query_get_clause(query, 0);
    ck_assert_int_eq(cypher_astnode_type(clause), CYPHER_AST_RETURN);

    ck_assert(!cypher_ast_return_is_distinct(clause));
    ck_assert(cypher_ast_return_has_include_existing(clause));

    ck_assert_int_eq(cypher_ast_return_nprojections(clause), 2);
    ck_assert_ptr_eq(cypher_ast_return_get_projection(clause, 2), NULL);

    const cypher_astnode_t *proj = cypher_ast_return_get_projection(clause, 0);
    ck_assert_int_eq(cypher_astnode_type(proj), CYPHER_AST_PROJECTION);
    const cypher_astnode_t *exp = cypher_ast_projection_get_expression(proj);
    ck_assert_int_eq(cypher_astnode_type(exp), CYPHER_AST_INTEGER);
    ck_assert_str_eq(cypher_ast_integer_get_valuestr(exp), "1");
    const cypher_astnode_t *alias = cypher_ast_projection_get_alias(proj);
    ck_assert_int_eq(cypher_astnode_type(alias), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(alias), "x");

    proj = cypher_ast_return_get_projection(clause, 1);
    ck_assert_int_eq(cypher_astnode_type(proj), CYPHER_AST_PROJECTION);
    exp = cypher_ast_projection_get_expression(proj);
    ck_assert_int_eq(cypher_astnode_type(exp), CYPHER_AST_STRING);
    ck_assert_str_eq(cypher_ast_string_get_value(exp), "bar");
    alias = cypher_ast_projection_get_alias(proj);
    ck_assert_int_eq(cypher_astnode_type(alias), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(alias), "y");

    ck_assert_ptr_eq(cypher_ast_return_get_order_by(clause), NULL);

    const cypher_astnode_t *skip = cypher_ast_return_get_skip(clause);
    ck_assert_int_eq(cypher_astnode_type(skip), CYPHER_AST_INTEGER);
    ck_assert_str_eq(cypher_ast_integer_get_valuestr(skip), "10");

    ck_assert_ptr_eq(cypher_ast_return_get_limit(clause), NULL);
}
END_TEST


START_TEST (parse_return_and_skip_limit)
{
    struct cypher_input_position last = cypher_input_position_zero;
    result = cypher_parse(
            "RETURN *, 1 AS x, 'bar' AS y SKIP 10 LIMIT 5",
            &last, NULL, 0);
    ck_assert_ptr_ne(result, NULL);
    ck_assert_int_eq(last.offset, 44);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
" @0   0..44  statement           body=@1\n"
" @1   0..44  > query             clauses=[@2]\n"
" @2   0..44  > > RETURN          *, projections=[@3, @6], SKIP=@9, LIMIT=@10\n"
" @3  10..16  > > > projection    expression=@4, alias=@5\n"
" @4  10..11  > > > > integer     1\n"
" @5  15..16  > > > > identifier  `x`\n"
" @6  18..29  > > > projection    expression=@7, alias=@8\n"
" @7  18..23  > > > > string      \"bar\"\n"
" @8  27..28  > > > > identifier  `y`\n"
" @9  34..36  > > > integer       10\n"
"@10  43..44  > > > integer       5\n";
    ck_assert_str_eq(memstream_buffer, expected);

    const cypher_astnode_t *ast = cypher_parse_result_get_directive(result, 0);
    const cypher_astnode_t *query = cypher_ast_statement_get_body(ast);
    const cypher_astnode_t *clause = cypher_ast_query_get_clause(query, 0);
    ck_assert_int_eq(cypher_astnode_type(clause), CYPHER_AST_RETURN);

    ck_assert(!cypher_ast_return_is_distinct(clause));
    ck_assert(cypher_ast_return_has_include_existing(clause));

    ck_assert_int_eq(cypher_ast_return_nprojections(clause), 2);
    ck_assert_ptr_eq(cypher_ast_return_get_projection(clause, 2), NULL);

    const cypher_astnode_t *proj = cypher_ast_return_get_projection(clause, 0);
    ck_assert_int_eq(cypher_astnode_type(proj), CYPHER_AST_PROJECTION);
    const cypher_astnode_t *exp = cypher_ast_projection_get_expression(proj);
    ck_assert_int_eq(cypher_astnode_type(exp), CYPHER_AST_INTEGER);
    ck_assert_str_eq(cypher_ast_integer_get_valuestr(exp), "1");
    const cypher_astnode_t *alias = cypher_ast_projection_get_alias(proj);
    ck_assert_int_eq(cypher_astnode_type(alias), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(alias), "x");

    proj = cypher_ast_return_get_projection(clause, 1);
    ck_assert_int_eq(cypher_astnode_type(proj), CYPHER_AST_PROJECTION);
    exp = cypher_ast_projection_get_expression(proj);
    ck_assert_int_eq(cypher_astnode_type(exp), CYPHER_AST_STRING);
    ck_assert_str_eq(cypher_ast_string_get_value(exp), "bar");
    alias = cypher_ast_projection_get_alias(proj);
    ck_assert_int_eq(cypher_astnode_type(alias), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(alias), "y");

    ck_assert_ptr_eq(cypher_ast_return_get_order_by(clause), NULL);

    const cypher_astnode_t *skip = cypher_ast_return_get_skip(clause);
    ck_assert_int_eq(cypher_astnode_type(skip), CYPHER_AST_INTEGER);
    ck_assert_str_eq(cypher_ast_integer_get_valuestr(skip), "10");

    const cypher_astnode_t *limit = cypher_ast_return_get_limit(clause);
    ck_assert_int_eq(cypher_astnode_type(limit), CYPHER_AST_INTEGER);
    ck_assert_str_eq(cypher_ast_integer_get_valuestr(limit), "5");
}
END_TEST


TCase* return_tcase(void)
{
    TCase *tc = tcase_create("return");
    tcase_add_checked_fixture(tc, setup, teardown);
    tcase_add_test(tc, parse_simple_return);
    tcase_add_test(tc, parse_distinct_return);
    tcase_add_test(tc, parse_return_including_existing);
    tcase_add_test(tc, parse_return_and_order_by);
    tcase_add_test(tc, parse_return_and_skip);
    tcase_add_test(tc, parse_return_and_skip_limit);
    return tc;
}
