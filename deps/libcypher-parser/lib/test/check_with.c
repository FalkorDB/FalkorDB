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


START_TEST (parse_simple_with)
{
    struct cypher_input_position last = cypher_input_position_zero;
    result = cypher_parse(
            "WITH 1 AS x, 'bar' AS y RETURN x, y",
            &last, NULL, 0);
    ck_assert_ptr_ne(result, NULL);
    ck_assert_int_eq(last.offset, 35);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
" @0   0..35  statement           body=@1\n"
" @1   0..35  > query             clauses=[@2, @9]\n"
" @2   0..24  > > WITH            projections=[@3, @6]\n"
" @3   5..11  > > > projection    expression=@4, alias=@5\n"
" @4   5..6   > > > > integer     1\n"
" @5  10..11  > > > > identifier  `x`\n"
" @6  13..24  > > > projection    expression=@7, alias=@8\n"
" @7  13..18  > > > > string      \"bar\"\n"
" @8  22..23  > > > > identifier  `y`\n"
" @9  24..35  > > RETURN          projections=[@10, @12]\n"
"@10  31..32  > > > projection    expression=@11\n"
"@11  31..32  > > > > identifier  `x`\n"
"@12  34..35  > > > projection    expression=@13\n"
"@13  34..35  > > > > identifier  `y`\n";
    ck_assert_str_eq(memstream_buffer, expected);

    const cypher_astnode_t *ast = cypher_parse_result_get_directive(result, 0);
    const cypher_astnode_t *query = cypher_ast_statement_get_body(ast);
    const cypher_astnode_t *clause = cypher_ast_query_get_clause(query, 0);
    ck_assert_int_eq(cypher_astnode_type(clause), CYPHER_AST_WITH);

    ck_assert(!cypher_ast_with_is_distinct(clause));
    ck_assert(!cypher_ast_with_has_include_existing(clause));

    ck_assert_int_eq(cypher_ast_with_nprojections(clause), 2);
    ck_assert_ptr_eq(cypher_ast_with_get_projection(clause, 2), NULL);

    const cypher_astnode_t *proj = cypher_ast_with_get_projection(clause, 0);
    ck_assert_int_eq(cypher_astnode_type(proj), CYPHER_AST_PROJECTION);
    const cypher_astnode_t *exp = cypher_ast_projection_get_expression(proj);
    ck_assert_int_eq(cypher_astnode_type(exp), CYPHER_AST_INTEGER);
    ck_assert_str_eq(cypher_ast_integer_get_valuestr(exp), "1");
    const cypher_astnode_t *alias = cypher_ast_projection_get_alias(proj);
    ck_assert_int_eq(cypher_astnode_type(alias), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(alias), "x");

    proj = cypher_ast_with_get_projection(clause, 1);
    ck_assert_int_eq(cypher_astnode_type(proj), CYPHER_AST_PROJECTION);
    exp = cypher_ast_projection_get_expression(proj);
    ck_assert_int_eq(cypher_astnode_type(exp), CYPHER_AST_STRING);
    ck_assert_str_eq(cypher_ast_string_get_value(exp), "bar");
    alias = cypher_ast_projection_get_alias(proj);
    ck_assert_int_eq(cypher_astnode_type(alias), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(alias), "y");

    ck_assert_ptr_eq(cypher_ast_with_get_order_by(clause), NULL);
    ck_assert_ptr_eq(cypher_ast_with_get_skip(clause), NULL);
    ck_assert_ptr_eq(cypher_ast_with_get_limit(clause), NULL);
    ck_assert_ptr_eq(cypher_ast_with_get_predicate(clause), NULL);
}
END_TEST


START_TEST (parse_distinct_with)
{
    struct cypher_input_position last = cypher_input_position_zero;
    result = cypher_parse(
            "WITH DISTINCT 1 AS x, 'bar' AS y RETURN x, y",
            &last, NULL, 0);
    ck_assert_ptr_ne(result, NULL);
    ck_assert_int_eq(last.offset, 44);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
" @0   0..44  statement           body=@1\n"
" @1   0..44  > query             clauses=[@2, @9]\n"
" @2   0..33  > > WITH            DISTINCT, projections=[@3, @6]\n"
" @3  14..20  > > > projection    expression=@4, alias=@5\n"
" @4  14..15  > > > > integer     1\n"
" @5  19..20  > > > > identifier  `x`\n"
" @6  22..33  > > > projection    expression=@7, alias=@8\n"
" @7  22..27  > > > > string      \"bar\"\n"
" @8  31..32  > > > > identifier  `y`\n"
" @9  33..44  > > RETURN          projections=[@10, @12]\n"
"@10  40..41  > > > projection    expression=@11\n"
"@11  40..41  > > > > identifier  `x`\n"
"@12  43..44  > > > projection    expression=@13\n"
"@13  43..44  > > > > identifier  `y`\n";
    ck_assert_str_eq(memstream_buffer, expected);

    const cypher_astnode_t *ast = cypher_parse_result_get_directive(result, 0);
    const cypher_astnode_t *query = cypher_ast_statement_get_body(ast);
    const cypher_astnode_t *clause = cypher_ast_query_get_clause(query, 0);
    ck_assert_int_eq(cypher_astnode_type(clause), CYPHER_AST_WITH);

    ck_assert(cypher_ast_with_is_distinct(clause));
    ck_assert(!cypher_ast_with_has_include_existing(clause));

    ck_assert_int_eq(cypher_ast_with_nprojections(clause), 2);
    ck_assert_ptr_eq(cypher_ast_with_get_projection(clause, 2), NULL);

    const cypher_astnode_t *proj = cypher_ast_with_get_projection(clause, 0);
    ck_assert_int_eq(cypher_astnode_type(proj), CYPHER_AST_PROJECTION);
    const cypher_astnode_t *exp = cypher_ast_projection_get_expression(proj);
    ck_assert_int_eq(cypher_astnode_type(exp), CYPHER_AST_INTEGER);
    ck_assert_str_eq(cypher_ast_integer_get_valuestr(exp), "1");
    const cypher_astnode_t *alias = cypher_ast_projection_get_alias(proj);
    ck_assert_int_eq(cypher_astnode_type(alias), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(alias), "x");

    proj = cypher_ast_with_get_projection(clause, 1);
    ck_assert_int_eq(cypher_astnode_type(proj), CYPHER_AST_PROJECTION);
    exp = cypher_ast_projection_get_expression(proj);
    ck_assert_int_eq(cypher_astnode_type(exp), CYPHER_AST_STRING);
    ck_assert_str_eq(cypher_ast_string_get_value(exp), "bar");
    alias = cypher_ast_projection_get_alias(proj);
    ck_assert_int_eq(cypher_astnode_type(alias), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(alias), "y");

    ck_assert_ptr_eq(cypher_ast_with_get_order_by(clause), NULL);
    ck_assert_ptr_eq(cypher_ast_with_get_skip(clause), NULL);
    ck_assert_ptr_eq(cypher_ast_with_get_limit(clause), NULL);
    ck_assert_ptr_eq(cypher_ast_with_get_predicate(clause), NULL);
}
END_TEST


START_TEST (parse_with_including_existing)
{
    struct cypher_input_position last = cypher_input_position_zero;
    result = cypher_parse(
            "WITH *, 1 AS x, 'bar' AS y RETURN x, y",
            &last, NULL, 0);
    ck_assert_ptr_ne(result, NULL);
    ck_assert_int_eq(last.offset, 38);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
" @0   0..38  statement           body=@1\n"
" @1   0..38  > query             clauses=[@2, @9]\n"
" @2   0..27  > > WITH            *, projections=[@3, @6]\n"
" @3   8..14  > > > projection    expression=@4, alias=@5\n"
" @4   8..9   > > > > integer     1\n"
" @5  13..14  > > > > identifier  `x`\n"
" @6  16..27  > > > projection    expression=@7, alias=@8\n"
" @7  16..21  > > > > string      \"bar\"\n"
" @8  25..26  > > > > identifier  `y`\n"
" @9  27..38  > > RETURN          projections=[@10, @12]\n"
"@10  34..35  > > > projection    expression=@11\n"
"@11  34..35  > > > > identifier  `x`\n"
"@12  37..38  > > > projection    expression=@13\n"
"@13  37..38  > > > > identifier  `y`\n";
    ck_assert_str_eq(memstream_buffer, expected);

    const cypher_astnode_t *ast = cypher_parse_result_get_directive(result, 0);
    const cypher_astnode_t *query = cypher_ast_statement_get_body(ast);
    const cypher_astnode_t *clause = cypher_ast_query_get_clause(query, 0);
    ck_assert_int_eq(cypher_astnode_type(clause), CYPHER_AST_WITH);

    ck_assert(!cypher_ast_with_is_distinct(clause));
    ck_assert(cypher_ast_with_has_include_existing(clause));

    ck_assert_int_eq(cypher_ast_with_nprojections(clause), 2);
    ck_assert_ptr_eq(cypher_ast_with_get_projection(clause, 2), NULL);

    const cypher_astnode_t *proj = cypher_ast_with_get_projection(clause, 0);
    ck_assert_int_eq(cypher_astnode_type(proj), CYPHER_AST_PROJECTION);
    const cypher_astnode_t *exp = cypher_ast_projection_get_expression(proj);
    ck_assert_int_eq(cypher_astnode_type(exp), CYPHER_AST_INTEGER);
    ck_assert_str_eq(cypher_ast_integer_get_valuestr(exp), "1");
    const cypher_astnode_t *alias = cypher_ast_projection_get_alias(proj);
    ck_assert_int_eq(cypher_astnode_type(alias), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(alias), "x");

    proj = cypher_ast_with_get_projection(clause, 1);
    ck_assert_int_eq(cypher_astnode_type(proj), CYPHER_AST_PROJECTION);
    exp = cypher_ast_projection_get_expression(proj);
    ck_assert_int_eq(cypher_astnode_type(exp), CYPHER_AST_STRING);
    ck_assert_str_eq(cypher_ast_string_get_value(exp), "bar");
    alias = cypher_ast_projection_get_alias(proj);
    ck_assert_int_eq(cypher_astnode_type(alias), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(alias), "y");

    ck_assert_ptr_eq(cypher_ast_with_get_order_by(clause), NULL);
    ck_assert_ptr_eq(cypher_ast_with_get_skip(clause), NULL);
    ck_assert_ptr_eq(cypher_ast_with_get_limit(clause), NULL);
    ck_assert_ptr_eq(cypher_ast_with_get_predicate(clause), NULL);
}
END_TEST


START_TEST (parse_with_and_order_by)
{
    struct cypher_input_position last = cypher_input_position_zero;
    result = cypher_parse(
            "WITH 1 AS x, 'bar' AS y ORDER BY x DESC, y ASC, z.prop + 10",
            &last, NULL, 0);
    ck_assert_ptr_ne(result, NULL);
    ck_assert_int_eq(last.offset, 59);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
" @0   0..59  statement                  body=@1\n"
" @1   0..59  > query                    clauses=[@2]\n"
" @2   0..59  > > WITH                   projections=[@3, @6], ORDER BY=@9\n"
" @3   5..11  > > > projection           expression=@4, alias=@5\n"
" @4   5..6   > > > > integer            1\n"
" @5  10..11  > > > > identifier         `x`\n"
" @6  13..24  > > > projection           expression=@7, alias=@8\n"
" @7  13..18  > > > > string             \"bar\"\n"
" @8  22..23  > > > > identifier         `y`\n"
" @9  24..59  > > > ORDER BY             items=[@10, @12, @14]\n"
"@10  33..39  > > > > sort item          expression=@11, DESCENDING\n"
"@11  33..34  > > > > > identifier       `x`\n"
"@12  41..46  > > > > sort item          expression=@13, ASCENDING\n"
"@13  41..42  > > > > > identifier       `y`\n"
"@14  48..59  > > > > sort item          expression=@15, ASCENDING\n"
"@15  48..59  > > > > > binary operator  @16 + @19\n"
"@16  48..55  > > > > > > property       @17.@18\n"
"@17  48..49  > > > > > > > identifier   `z`\n"
"@18  50..54  > > > > > > > prop name    `prop`\n"
"@19  57..59  > > > > > > integer        10\n";
    ck_assert_str_eq(memstream_buffer, expected);

    const cypher_astnode_t *ast = cypher_parse_result_get_directive(result, 0);
    const cypher_astnode_t *query = cypher_ast_statement_get_body(ast);
    const cypher_astnode_t *clause = cypher_ast_query_get_clause(query, 0);
    ck_assert_int_eq(cypher_astnode_type(clause), CYPHER_AST_WITH);

    ck_assert(!cypher_ast_with_is_distinct(clause));
    ck_assert(!cypher_ast_with_has_include_existing(clause));

    ck_assert_int_eq(cypher_ast_with_nprojections(clause), 2);
    ck_assert_ptr_eq(cypher_ast_with_get_projection(clause, 2), NULL);

    const cypher_astnode_t *proj = cypher_ast_with_get_projection(clause, 0);
    ck_assert_int_eq(cypher_astnode_type(proj), CYPHER_AST_PROJECTION);
    const cypher_astnode_t *exp = cypher_ast_projection_get_expression(proj);
    ck_assert_int_eq(cypher_astnode_type(exp), CYPHER_AST_INTEGER);
    ck_assert_str_eq(cypher_ast_integer_get_valuestr(exp), "1");
    const cypher_astnode_t *alias = cypher_ast_projection_get_alias(proj);
    ck_assert_int_eq(cypher_astnode_type(alias), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(alias), "x");

    proj = cypher_ast_with_get_projection(clause, 1);
    ck_assert_int_eq(cypher_astnode_type(proj), CYPHER_AST_PROJECTION);
    exp = cypher_ast_projection_get_expression(proj);
    ck_assert_int_eq(cypher_astnode_type(exp), CYPHER_AST_STRING);
    ck_assert_str_eq(cypher_ast_string_get_value(exp), "bar");
    alias = cypher_ast_projection_get_alias(proj);
    ck_assert_int_eq(cypher_astnode_type(alias), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(alias), "y");

    const cypher_astnode_t *order = cypher_ast_with_get_order_by(clause);
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

    ck_assert_ptr_eq(cypher_ast_with_get_skip(clause), NULL);
    ck_assert_ptr_eq(cypher_ast_with_get_limit(clause), NULL);
    ck_assert_ptr_eq(cypher_ast_with_get_predicate(clause), NULL);
}
END_TEST


START_TEST (parse_with_and_skip)
{
    struct cypher_input_position last = cypher_input_position_zero;
    result = cypher_parse(
            "WITH *, 1 AS x, 'bar' AS y SKIP 10",
            &last, NULL, 0);
    ck_assert_ptr_ne(result, NULL);
    ck_assert_int_eq(last.offset, 34);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
"@0   0..34  statement           body=@1\n"
"@1   0..34  > query             clauses=[@2]\n"
"@2   0..34  > > WITH            *, projections=[@3, @6], SKIP=@9\n"
"@3   8..14  > > > projection    expression=@4, alias=@5\n"
"@4   8..9   > > > > integer     1\n"
"@5  13..14  > > > > identifier  `x`\n"
"@6  16..27  > > > projection    expression=@7, alias=@8\n"
"@7  16..21  > > > > string      \"bar\"\n"
"@8  25..26  > > > > identifier  `y`\n"
"@9  32..34  > > > integer       10\n";
    ck_assert_str_eq(memstream_buffer, expected);

    const cypher_astnode_t *ast = cypher_parse_result_get_directive(result, 0);
    const cypher_astnode_t *query = cypher_ast_statement_get_body(ast);
    const cypher_astnode_t *clause = cypher_ast_query_get_clause(query, 0);
    ck_assert_int_eq(cypher_astnode_type(clause), CYPHER_AST_WITH);

    ck_assert(!cypher_ast_with_is_distinct(clause));
    ck_assert(cypher_ast_with_has_include_existing(clause));

    ck_assert_int_eq(cypher_ast_with_nprojections(clause), 2);
    ck_assert_ptr_eq(cypher_ast_with_get_projection(clause, 2), NULL);

    const cypher_astnode_t *proj = cypher_ast_with_get_projection(clause, 0);
    ck_assert_int_eq(cypher_astnode_type(proj), CYPHER_AST_PROJECTION);
    const cypher_astnode_t *exp = cypher_ast_projection_get_expression(proj);
    ck_assert_int_eq(cypher_astnode_type(exp), CYPHER_AST_INTEGER);
    ck_assert_str_eq(cypher_ast_integer_get_valuestr(exp), "1");
    const cypher_astnode_t *alias = cypher_ast_projection_get_alias(proj);
    ck_assert_int_eq(cypher_astnode_type(alias), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(alias), "x");

    proj = cypher_ast_with_get_projection(clause, 1);
    ck_assert_int_eq(cypher_astnode_type(proj), CYPHER_AST_PROJECTION);
    exp = cypher_ast_projection_get_expression(proj);
    ck_assert_int_eq(cypher_astnode_type(exp), CYPHER_AST_STRING);
    ck_assert_str_eq(cypher_ast_string_get_value(exp), "bar");
    alias = cypher_ast_projection_get_alias(proj);
    ck_assert_int_eq(cypher_astnode_type(alias), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(alias), "y");

    ck_assert_ptr_eq(cypher_ast_with_get_order_by(clause), NULL);

    const cypher_astnode_t *skip = cypher_ast_with_get_skip(clause);
    ck_assert_int_eq(cypher_astnode_type(skip), CYPHER_AST_INTEGER);
    ck_assert_str_eq(cypher_ast_integer_get_valuestr(skip), "10");

    ck_assert_ptr_eq(cypher_ast_with_get_limit(clause), NULL);
    ck_assert_ptr_eq(cypher_ast_with_get_predicate(clause), NULL);
}
END_TEST


START_TEST (parse_with_and_skip_limit)
{
    struct cypher_input_position last = cypher_input_position_zero;
    result = cypher_parse(
            "WITH *, 1 AS x, 'bar' AS y SKIP 10 LIMIT 5",
            &last, NULL, 0);
    ck_assert_ptr_ne(result, NULL);
    ck_assert_int_eq(last.offset, 42);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
" @0   0..42  statement           body=@1\n"
" @1   0..42  > query             clauses=[@2]\n"
" @2   0..42  > > WITH            *, projections=[@3, @6], SKIP=@9, LIMIT=@10\n"
" @3   8..14  > > > projection    expression=@4, alias=@5\n"
" @4   8..9   > > > > integer     1\n"
" @5  13..14  > > > > identifier  `x`\n"
" @6  16..27  > > > projection    expression=@7, alias=@8\n"
" @7  16..21  > > > > string      \"bar\"\n"
" @8  25..26  > > > > identifier  `y`\n"
" @9  32..34  > > > integer       10\n"
"@10  41..42  > > > integer       5\n";
    ck_assert_str_eq(memstream_buffer, expected);

    const cypher_astnode_t *ast = cypher_parse_result_get_directive(result, 0);
    const cypher_astnode_t *query = cypher_ast_statement_get_body(ast);
    const cypher_astnode_t *clause = cypher_ast_query_get_clause(query, 0);
    ck_assert_int_eq(cypher_astnode_type(clause), CYPHER_AST_WITH);

    ck_assert(!cypher_ast_with_is_distinct(clause));
    ck_assert(cypher_ast_with_has_include_existing(clause));

    ck_assert_int_eq(cypher_ast_with_nprojections(clause), 2);
    ck_assert_ptr_eq(cypher_ast_with_get_projection(clause, 2), NULL);

    const cypher_astnode_t *proj = cypher_ast_with_get_projection(clause, 0);
    ck_assert_int_eq(cypher_astnode_type(proj), CYPHER_AST_PROJECTION);
    const cypher_astnode_t *exp = cypher_ast_projection_get_expression(proj);
    ck_assert_int_eq(cypher_astnode_type(exp), CYPHER_AST_INTEGER);
    ck_assert_str_eq(cypher_ast_integer_get_valuestr(exp), "1");
    const cypher_astnode_t *alias = cypher_ast_projection_get_alias(proj);
    ck_assert_int_eq(cypher_astnode_type(alias), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(alias), "x");

    proj = cypher_ast_with_get_projection(clause, 1);
    ck_assert_int_eq(cypher_astnode_type(proj), CYPHER_AST_PROJECTION);
    exp = cypher_ast_projection_get_expression(proj);
    ck_assert_int_eq(cypher_astnode_type(exp), CYPHER_AST_STRING);
    ck_assert_str_eq(cypher_ast_string_get_value(exp), "bar");
    alias = cypher_ast_projection_get_alias(proj);
    ck_assert_int_eq(cypher_astnode_type(alias), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(alias), "y");

    ck_assert_ptr_eq(cypher_ast_with_get_order_by(clause), NULL);

    const cypher_astnode_t *skip = cypher_ast_with_get_skip(clause);
    ck_assert_int_eq(cypher_astnode_type(skip), CYPHER_AST_INTEGER);
    ck_assert_str_eq(cypher_ast_integer_get_valuestr(skip), "10");

    const cypher_astnode_t *limit = cypher_ast_with_get_limit(clause);
    ck_assert_int_eq(cypher_astnode_type(limit), CYPHER_AST_INTEGER);
    ck_assert_str_eq(cypher_ast_integer_get_valuestr(limit), "5");

    ck_assert_ptr_eq(cypher_ast_with_get_predicate(clause), NULL);
}
END_TEST


START_TEST (parse_with_and_predicate)
{
    struct cypher_input_position last = cypher_input_position_zero;
    result = cypher_parse("WITH * WHERE n.foo > 10", &last, NULL, 0);
    ck_assert_ptr_ne(result, NULL);
    ck_assert_int_eq(last.offset, 23);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
"@0   0..23  statement             body=@1\n"
"@1   0..23  > query               clauses=[@2]\n"
"@2   0..23  > > WITH              *, projections=[], WHERE=@3\n"
"@3  13..23  > > > comparison      @4 > @7\n"
"@4  13..19  > > > > property      @5.@6\n"
"@5  13..14  > > > > > identifier  `n`\n"
"@6  15..18  > > > > > prop name   `foo`\n"
"@7  21..23  > > > > integer       10\n";
    ck_assert_str_eq(memstream_buffer, expected);

    const cypher_astnode_t *ast = cypher_parse_result_get_directive(result, 0);
    const cypher_astnode_t *query = cypher_ast_statement_get_body(ast);
    const cypher_astnode_t *clause = cypher_ast_query_get_clause(query, 0);
    ck_assert_int_eq(cypher_astnode_type(clause), CYPHER_AST_WITH);

    ck_assert(!cypher_ast_with_is_distinct(clause));
    ck_assert(cypher_ast_with_has_include_existing(clause));

    ck_assert_int_eq(cypher_ast_with_nprojections(clause), 0);
    ck_assert_ptr_eq(cypher_ast_with_get_projection(clause, 0), NULL);

    ck_assert_ptr_eq(cypher_ast_with_get_order_by(clause), NULL);
    ck_assert_ptr_eq(cypher_ast_with_get_skip(clause), NULL);
    ck_assert_ptr_eq(cypher_ast_with_get_limit(clause), NULL);

    const cypher_astnode_t *pred = cypher_ast_with_get_predicate(clause);
    ck_assert_int_eq(cypher_astnode_type(pred), CYPHER_AST_COMPARISON);
}
END_TEST


TCase* with_tcase(void)
{
    TCase *tc = tcase_create("with");
    tcase_add_checked_fixture(tc, setup, teardown);
    tcase_add_test(tc, parse_simple_with);
    tcase_add_test(tc, parse_distinct_with);
    tcase_add_test(tc, parse_with_including_existing);
    tcase_add_test(tc, parse_with_and_order_by);
    tcase_add_test(tc, parse_with_and_skip);
    tcase_add_test(tc, parse_with_and_skip_limit);
    tcase_add_test(tc, parse_with_and_predicate);
    return tc;
}
