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


START_TEST (parse_simple_merge)
{
    struct cypher_input_position last = cypher_input_position_zero;
    result = cypher_parse("MERGE (n)-[:KNOWS]->(f) RETURN f;", &last, NULL, 0);
    ck_assert_ptr_ne(result, NULL);
    ck_assert_int_eq(last.offset, 33);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
" @0   0..33  statement             body=@1\n"
" @1   0..33  > query               clauses=[@2, @10]\n"
" @2   0..24  > > MERGE             path=@3\n"
" @3   6..23  > > > pattern path    (@4)-[@6]-(@8)\n"
" @4   6..9   > > > > node pattern  (@5)\n"
" @5   7..8   > > > > > identifier  `n`\n"
" @6   9..20  > > > > rel pattern   -[:@7]->\n"
" @7  11..17  > > > > > rel type    :`KNOWS`\n"
" @8  20..23  > > > > node pattern  (@9)\n"
" @9  21..22  > > > > > identifier  `f`\n"
"@10  24..32  > > RETURN            projections=[@11]\n"
"@11  31..32  > > > projection      expression=@12\n"
"@12  31..32  > > > > identifier    `f`\n";
    ck_assert_str_eq(memstream_buffer, expected);

    const cypher_astnode_t *ast = cypher_parse_result_get_directive(result, 0);
    ck_assert_int_eq(cypher_astnode_type(ast), CYPHER_AST_STATEMENT);
    const cypher_astnode_t *query = cypher_ast_statement_get_body(ast);
    ck_assert_int_eq(cypher_astnode_type(query), CYPHER_AST_QUERY);
    const cypher_astnode_t *merge = cypher_ast_query_get_clause(query, 0);
    ck_assert_int_eq(cypher_astnode_type(merge), CYPHER_AST_MERGE);

    const cypher_astnode_t *path = cypher_ast_merge_get_pattern_path(merge);
    ck_assert_int_eq(cypher_astnode_type(path), CYPHER_AST_PATTERN_PATH);
    ck_assert_int_eq(cypher_ast_pattern_path_nelements(path), 3);

    ck_assert_int_eq(cypher_ast_merge_nactions(merge), 0);
    ck_assert_ptr_eq(cypher_ast_merge_get_action(merge, 0), NULL);
}
END_TEST


START_TEST (parse_merge_with_on_match_action)
{
    struct cypher_input_position last = cypher_input_position_zero;
    result = cypher_parse("MERGE (n:Foo) ON MATCH SET n.bar = 'baz' RETURN n;",
            &last, NULL, 0);
    ck_assert_ptr_ne(result, NULL);
    ck_assert_int_eq(last.offset, 50);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
" @0   0..50  statement               body=@1\n"
" @1   0..50  > query                 clauses=[@2, @13]\n"
" @2   0..41  > > MERGE               path=@3, action[@7]\n"
" @3   6..13  > > > pattern path      (@4)\n"
" @4   6..13  > > > > node pattern    (@5:@6)\n"
" @5   7..8   > > > > > identifier    `n`\n"
" @6   8..12  > > > > > label         :`Foo`\n"
" @7  14..41  > > > ON MATCH          items=[@8]\n"
" @8  27..41  > > > > set property    @9 = @12\n"
" @9  27..33  > > > > > property      @10.@11\n"
"@10  27..28  > > > > > > identifier  `n`\n"
"@11  29..32  > > > > > > prop name   `bar`\n"
"@12  35..40  > > > > > string        \"baz\"\n"
"@13  41..49  > > RETURN              projections=[@14]\n"
"@14  48..49  > > > projection        expression=@15\n"
"@15  48..49  > > > > identifier      `n`\n";
    ck_assert_str_eq(memstream_buffer, expected);

    const cypher_astnode_t *ast = cypher_parse_result_get_directive(result, 0);
    ck_assert_int_eq(cypher_astnode_type(ast), CYPHER_AST_STATEMENT);
    const cypher_astnode_t *query = cypher_ast_statement_get_body(ast);
    ck_assert_int_eq(cypher_astnode_type(query), CYPHER_AST_QUERY);
    const cypher_astnode_t *merge = cypher_ast_query_get_clause(query, 0);
    ck_assert_int_eq(cypher_astnode_type(merge), CYPHER_AST_MERGE);

    const cypher_astnode_t *path = cypher_ast_merge_get_pattern_path(merge);
    ck_assert_int_eq(cypher_astnode_type(path), CYPHER_AST_PATTERN_PATH);
    ck_assert_int_eq(cypher_ast_pattern_path_nelements(path), 1);

    ck_assert_int_eq(cypher_ast_merge_nactions(merge), 1);
    ck_assert_ptr_eq(cypher_ast_merge_get_action(merge, 1), NULL);

    const cypher_astnode_t *action = cypher_ast_merge_get_action(merge, 0);
    ck_assert(cypher_astnode_instanceof(action, CYPHER_AST_MERGE_ACTION));
    ck_assert_int_eq(cypher_astnode_type(action), CYPHER_AST_ON_MATCH);

    ck_assert_int_eq(cypher_ast_on_match_nitems(action), 1);
    ck_assert_ptr_eq(cypher_ast_on_match_get_item(action, 1), NULL);

    const cypher_astnode_t *item = cypher_ast_on_match_get_item(action, 0);
    ck_assert_int_eq(cypher_astnode_type(item), CYPHER_AST_SET_PROPERTY);
}
END_TEST


START_TEST (parse_merge_with_on_create_action)
{
    struct cypher_input_position last = cypher_input_position_zero;
    result = cypher_parse(
            "MERGE (n:Foo) ON CREATE SET n.bar = 'baz', n:Bar RETURN n;",
            &last, NULL, 0);
    ck_assert_ptr_ne(result, NULL);
    ck_assert_int_eq(last.offset, 58);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
" @0   0..58  statement               body=@1\n"
" @1   0..58  > query                 clauses=[@2, @16]\n"
" @2   0..49  > > MERGE               path=@3, action[@7]\n"
" @3   6..13  > > > pattern path      (@4)\n"
" @4   6..13  > > > > node pattern    (@5:@6)\n"
" @5   7..8   > > > > > identifier    `n`\n"
" @6   8..12  > > > > > label         :`Foo`\n"
" @7  14..49  > > > ON CREATE         items=[@8, @13]\n"
" @8  28..41  > > > > set property    @9 = @12\n"
" @9  28..34  > > > > > property      @10.@11\n"
"@10  28..29  > > > > > > identifier  `n`\n"
"@11  30..33  > > > > > > prop name   `bar`\n"
"@12  36..41  > > > > > string        \"baz\"\n"
"@13  43..49  > > > > set labels      @14:@15\n"
"@14  43..44  > > > > > identifier    `n`\n"
"@15  44..48  > > > > > label         :`Bar`\n"
"@16  49..57  > > RETURN              projections=[@17]\n"
"@17  56..57  > > > projection        expression=@18\n"
"@18  56..57  > > > > identifier      `n`\n";
    ck_assert_str_eq(memstream_buffer, expected);

    const cypher_astnode_t *ast = cypher_parse_result_get_directive(result, 0);
    ck_assert_int_eq(cypher_astnode_type(ast), CYPHER_AST_STATEMENT);
    const cypher_astnode_t *query = cypher_ast_statement_get_body(ast);
    ck_assert_int_eq(cypher_astnode_type(query), CYPHER_AST_QUERY);
    const cypher_astnode_t *merge = cypher_ast_query_get_clause(query, 0);
    ck_assert_int_eq(cypher_astnode_type(merge), CYPHER_AST_MERGE);

    const cypher_astnode_t *path = cypher_ast_merge_get_pattern_path(merge);
    ck_assert_int_eq(cypher_astnode_type(path), CYPHER_AST_PATTERN_PATH);
    ck_assert_int_eq(cypher_ast_pattern_path_nelements(path), 1);

    ck_assert_int_eq(cypher_ast_merge_nactions(merge), 1);
    ck_assert_ptr_eq(cypher_ast_merge_get_action(merge, 1), NULL);

    const cypher_astnode_t *action = cypher_ast_merge_get_action(merge, 0);
    ck_assert(cypher_astnode_instanceof(action, CYPHER_AST_MERGE_ACTION));
    ck_assert_int_eq(cypher_astnode_type(action), CYPHER_AST_ON_CREATE);

    ck_assert_int_eq(cypher_ast_on_create_nitems(action), 2);
    ck_assert_ptr_eq(cypher_ast_on_create_get_item(action, 2), NULL);

    const cypher_astnode_t *item = cypher_ast_on_create_get_item(action, 0);
    ck_assert_int_eq(cypher_astnode_type(item), CYPHER_AST_SET_PROPERTY);

    item = cypher_ast_on_create_get_item(action, 1);
    ck_assert_int_eq(cypher_astnode_type(item), CYPHER_AST_SET_LABELS);
}
END_TEST


START_TEST (parse_merge_with_multiple_actions)
{
    struct cypher_input_position last = cypher_input_position_zero;
    result = cypher_parse(
            "MERGE (n:Foo) ON CREATE SET n.bar = 'baz' ON MATCH SET n.bar = 'foo' RETURN n;",
            &last, NULL, 0);
    ck_assert_ptr_ne(result, NULL);
    ck_assert_int_eq(last.offset, 78);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
" @0   0..78  statement               body=@1\n"
" @1   0..78  > query                 clauses=[@2, @19]\n"
" @2   0..69  > > MERGE               path=@3, action[@7, @13]\n"
" @3   6..13  > > > pattern path      (@4)\n"
" @4   6..13  > > > > node pattern    (@5:@6)\n"
" @5   7..8   > > > > > identifier    `n`\n"
" @6   8..12  > > > > > label         :`Foo`\n"
" @7  14..42  > > > ON CREATE         items=[@8]\n"
" @8  28..42  > > > > set property    @9 = @12\n"
" @9  28..34  > > > > > property      @10.@11\n"
"@10  28..29  > > > > > > identifier  `n`\n"
"@11  30..33  > > > > > > prop name   `bar`\n"
"@12  36..41  > > > > > string        \"baz\"\n"
"@13  42..69  > > > ON MATCH          items=[@14]\n"
"@14  55..69  > > > > set property    @15 = @18\n"
"@15  55..61  > > > > > property      @16.@17\n"
"@16  55..56  > > > > > > identifier  `n`\n"
"@17  57..60  > > > > > > prop name   `bar`\n"
"@18  63..68  > > > > > string        \"foo\"\n"
"@19  69..77  > > RETURN              projections=[@20]\n"
"@20  76..77  > > > projection        expression=@21\n"
"@21  76..77  > > > > identifier      `n`\n";
    ck_assert_str_eq(memstream_buffer, expected);

    const cypher_astnode_t *ast = cypher_parse_result_get_directive(result, 0);
    ck_assert_int_eq(cypher_astnode_type(ast), CYPHER_AST_STATEMENT);
    const cypher_astnode_t *query = cypher_ast_statement_get_body(ast);
    ck_assert_int_eq(cypher_astnode_type(query), CYPHER_AST_QUERY);
    const cypher_astnode_t *merge = cypher_ast_query_get_clause(query, 0);
    ck_assert_int_eq(cypher_astnode_type(merge), CYPHER_AST_MERGE);

    const cypher_astnode_t *path = cypher_ast_merge_get_pattern_path(merge);
    ck_assert_int_eq(cypher_astnode_type(path), CYPHER_AST_PATTERN_PATH);
    ck_assert_int_eq(cypher_ast_pattern_path_nelements(path), 1);

    ck_assert_int_eq(cypher_ast_merge_nactions(merge), 2);
    ck_assert_ptr_eq(cypher_ast_merge_get_action(merge, 2), NULL);

    const cypher_astnode_t *action = cypher_ast_merge_get_action(merge, 0);
    ck_assert(cypher_astnode_instanceof(action, CYPHER_AST_MERGE_ACTION));
    ck_assert_int_eq(cypher_astnode_type(action), CYPHER_AST_ON_CREATE);

    ck_assert_int_eq(cypher_ast_on_create_nitems(action), 1);
    ck_assert_ptr_eq(cypher_ast_on_create_get_item(action, 1), NULL);

    const cypher_astnode_t *item = cypher_ast_on_create_get_item(action, 0);
    ck_assert_int_eq(cypher_astnode_type(item), CYPHER_AST_SET_PROPERTY);

    action = cypher_ast_merge_get_action(merge, 1);
    ck_assert(cypher_astnode_instanceof(action, CYPHER_AST_MERGE_ACTION));
    ck_assert_int_eq(cypher_astnode_type(action), CYPHER_AST_ON_MATCH);

    ck_assert_int_eq(cypher_ast_on_match_nitems(action), 1);
    ck_assert_ptr_eq(cypher_ast_on_match_get_item(action, 1), NULL);

    item = cypher_ast_on_match_get_item(action, 0);
    ck_assert_int_eq(cypher_astnode_type(item), CYPHER_AST_SET_PROPERTY);
}
END_TEST


TCase* merge_tcase(void)
{
    TCase *tc = tcase_create("merge");
    tcase_add_checked_fixture(tc, setup, teardown);
    tcase_add_test(tc, parse_simple_merge);
    tcase_add_test(tc, parse_merge_with_on_match_action);
    tcase_add_test(tc, parse_merge_with_on_create_action);
    tcase_add_test(tc, parse_merge_with_multiple_actions);
    return tc;
}
