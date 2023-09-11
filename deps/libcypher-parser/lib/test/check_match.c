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


START_TEST (parse_simple_match)
{
    struct cypher_input_position last = cypher_input_position_zero;
    result = cypher_parse("MATCH (n)-[:KNOWS]->(f) RETURN f;", &last, NULL, 0);
    ck_assert_ptr_ne(result, NULL);
    ck_assert_int_eq(last.offset, 33);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
" @0   0..33  statement               body=@1\n"
" @1   0..33  > query                 clauses=[@2, @11]\n"
" @2   0..24  > > MATCH               pattern=@3\n"
" @3   6..23  > > > pattern           paths=[@4]\n"
" @4   6..23  > > > > pattern path    (@5)-[@7]-(@9)\n"
" @5   6..9   > > > > > node pattern  (@6)\n"
" @6   7..8   > > > > > > identifier  `n`\n"
" @7   9..20  > > > > > rel pattern   -[:@8]->\n"
" @8  11..17  > > > > > > rel type    :`KNOWS`\n"
" @9  20..23  > > > > > node pattern  (@10)\n"
"@10  21..22  > > > > > > identifier  `f`\n"
"@11  24..32  > > RETURN              projections=[@12]\n"
"@12  31..32  > > > projection        expression=@13\n"
"@13  31..32  > > > > identifier      `f`\n";
    ck_assert_str_eq(memstream_buffer, expected);

    const cypher_astnode_t *ast = cypher_parse_result_get_directive(result, 0);
    ck_assert_int_eq(cypher_astnode_type(ast), CYPHER_AST_STATEMENT);
    const cypher_astnode_t *query = cypher_ast_statement_get_body(ast);
    ck_assert_int_eq(cypher_astnode_type(query), CYPHER_AST_QUERY);
    const cypher_astnode_t *match = cypher_ast_query_get_clause(query, 0);
    ck_assert_int_eq(cypher_astnode_type(match), CYPHER_AST_MATCH);

    ck_assert(!cypher_ast_match_is_optional(match));

    const cypher_astnode_t *pattern = cypher_ast_match_get_pattern(match);
    ck_assert_int_eq(cypher_astnode_type(pattern), CYPHER_AST_PATTERN);
    ck_assert_int_eq(cypher_ast_pattern_npaths(pattern), 1);

    ck_assert_int_eq(cypher_ast_match_nhints(match), 0);
    ck_assert_ptr_eq(cypher_ast_match_get_hint(match, 0), NULL);
    ck_assert_ptr_eq(cypher_ast_match_get_predicate(match), NULL);
}
END_TEST


START_TEST (parse_simple_optional_match)
{
    struct cypher_input_position last = cypher_input_position_zero;
    result = cypher_parse("OPTIONAL MATCH (n) RETURN n;", &last, NULL, 0);
    ck_assert_ptr_ne(result, NULL);
    ck_assert_int_eq(last.offset, 28);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
"@0   0..28  statement               body=@1\n"
"@1   0..28  > query                 clauses=[@2, @7]\n"
"@2   0..19  > > MATCH               OPTIONAL, pattern=@3\n"
"@3  15..18  > > > pattern           paths=[@4]\n"
"@4  15..18  > > > > pattern path    (@5)\n"
"@5  15..18  > > > > > node pattern  (@6)\n"
"@6  16..17  > > > > > > identifier  `n`\n"
"@7  19..27  > > RETURN              projections=[@8]\n"
"@8  26..27  > > > projection        expression=@9\n"
"@9  26..27  > > > > identifier      `n`\n";
    ck_assert_str_eq(memstream_buffer, expected);

    const cypher_astnode_t *ast = cypher_parse_result_get_directive(result, 0);
    ck_assert_int_eq(cypher_astnode_type(ast), CYPHER_AST_STATEMENT);
    const cypher_astnode_t *query = cypher_ast_statement_get_body(ast);
    ck_assert_int_eq(cypher_astnode_type(query), CYPHER_AST_QUERY);
    const cypher_astnode_t *match = cypher_ast_query_get_clause(query, 0);
    ck_assert_int_eq(cypher_astnode_type(match), CYPHER_AST_MATCH);

    ck_assert(cypher_ast_match_is_optional(match));

    const cypher_astnode_t *pattern = cypher_ast_match_get_pattern(match);
    ck_assert_int_eq(cypher_astnode_type(pattern), CYPHER_AST_PATTERN);
    ck_assert_int_eq(cypher_ast_pattern_npaths(pattern), 1);

    ck_assert_int_eq(cypher_ast_match_nhints(match), 0);
    ck_assert_ptr_eq(cypher_ast_match_get_hint(match, 0), NULL);
    ck_assert_ptr_eq(cypher_ast_match_get_predicate(match), NULL);
}
END_TEST


START_TEST (parse_match_with_predicate)
{
    struct cypher_input_position last = cypher_input_position_zero;
    result = cypher_parse("MATCH (n) WHERE n:Foo RETURN n;", &last, NULL, 0);
    ck_assert_ptr_ne(result, NULL);
    ck_assert_int_eq(last.offset, 31);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
" @0   0..31  statement               body=@1\n"
" @1   0..31  > query                 clauses=[@2, @10]\n"
" @2   0..22  > > MATCH               pattern=@3, where=@7\n"
" @3   6..9   > > > pattern           paths=[@4]\n"
" @4   6..9   > > > > pattern path    (@5)\n"
" @5   6..9   > > > > > node pattern  (@6)\n"
" @6   7..8   > > > > > > identifier  `n`\n"
" @7  16..22  > > > has labels        @8:@9\n"
" @8  16..17  > > > > identifier      `n`\n"
" @9  17..21  > > > > label           :`Foo`\n"
"@10  22..30  > > RETURN              projections=[@11]\n"
"@11  29..30  > > > projection        expression=@12\n"
"@12  29..30  > > > > identifier      `n`\n";
    ck_assert_str_eq(memstream_buffer, expected);

    const cypher_astnode_t *ast = cypher_parse_result_get_directive(result, 0);
    ck_assert_int_eq(cypher_astnode_type(ast), CYPHER_AST_STATEMENT);
    const cypher_astnode_t *query = cypher_ast_statement_get_body(ast);
    ck_assert_int_eq(cypher_astnode_type(query), CYPHER_AST_QUERY);
    const cypher_astnode_t *match = cypher_ast_query_get_clause(query, 0);
    ck_assert_int_eq(cypher_astnode_type(match), CYPHER_AST_MATCH);

    ck_assert(!cypher_ast_match_is_optional(match));

    const cypher_astnode_t *pattern = cypher_ast_match_get_pattern(match);
    ck_assert_int_eq(cypher_astnode_type(pattern), CYPHER_AST_PATTERN);
    ck_assert_int_eq(cypher_ast_pattern_npaths(pattern), 1);

    ck_assert_int_eq(cypher_ast_match_nhints(match), 0);
    ck_assert_ptr_eq(cypher_ast_match_get_hint(match, 0), NULL);

    const cypher_astnode_t *pred = cypher_ast_match_get_predicate(match);
    ck_assert_int_eq(cypher_astnode_type(pred), CYPHER_AST_LABELS_OPERATOR);

    const cypher_astnode_t *id = cypher_ast_labels_operator_get_expression(pred);
    ck_assert_int_eq(cypher_astnode_type(id), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(id), "n");

    ck_assert_int_eq(cypher_ast_labels_operator_nlabels(pred), 1);
    const cypher_astnode_t *label = cypher_ast_labels_operator_get_label(pred, 0);
    ck_assert_int_eq(cypher_astnode_type(label), CYPHER_AST_LABEL);
    ck_assert_str_eq(cypher_ast_label_get_name(label), "Foo");
}
END_TEST


START_TEST (parse_match_with_using_index_hint)
{
    struct cypher_input_position last = cypher_input_position_zero;
    result = cypher_parse("MATCH (n:Foo) USING INDEX n:Foo(bar) RETURN n;",
            &last, NULL, 0);
    ck_assert_ptr_ne(result, NULL);
    ck_assert_int_eq(last.offset, 46);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
" @0   0..46  statement               body=@1\n"
" @1   0..46  > query                 clauses=[@2, @12]\n"
" @2   0..37  > > MATCH               pattern=@3, hints=[@8]\n"
" @3   6..13  > > > pattern           paths=[@4]\n"
" @4   6..13  > > > > pattern path    (@5)\n"
" @5   6..13  > > > > > node pattern  (@6:@7)\n"
" @6   7..8   > > > > > > identifier  `n`\n"
" @7   8..12  > > > > > > label       :`Foo`\n"
" @8  14..37  > > > USING INDEX       @9:@10(@11)\n"
" @9  26..27  > > > > identifier      `n`\n"
"@10  27..31  > > > > label           :`Foo`\n"
"@11  32..35  > > > > prop name       `bar`\n"
"@12  37..45  > > RETURN              projections=[@13]\n"
"@13  44..45  > > > projection        expression=@14\n"
"@14  44..45  > > > > identifier      `n`\n";
    ck_assert_str_eq(memstream_buffer, expected);

    const cypher_astnode_t *ast = cypher_parse_result_get_directive(result, 0);
    ck_assert_int_eq(cypher_astnode_type(ast), CYPHER_AST_STATEMENT);
    const cypher_astnode_t *query = cypher_ast_statement_get_body(ast);
    ck_assert_int_eq(cypher_astnode_type(query), CYPHER_AST_QUERY);
    const cypher_astnode_t *match = cypher_ast_query_get_clause(query, 0);
    ck_assert_int_eq(cypher_astnode_type(match), CYPHER_AST_MATCH);

    ck_assert(!cypher_ast_match_is_optional(match));

    const cypher_astnode_t *pattern = cypher_ast_match_get_pattern(match);
    ck_assert_int_eq(cypher_astnode_type(pattern), CYPHER_AST_PATTERN);
    ck_assert_int_eq(cypher_ast_pattern_npaths(pattern), 1);

    const cypher_astnode_t *path = cypher_ast_pattern_get_path(pattern, 0);
    ck_assert_int_eq(cypher_astnode_type(path), CYPHER_AST_PATTERN_PATH);
    ck_assert_int_eq(cypher_ast_pattern_path_nelements(path), 1);

    ck_assert_int_eq(cypher_ast_match_nhints(match), 1);

    const cypher_astnode_t *hint = cypher_ast_match_get_hint(match, 0);
    ck_assert(cypher_astnode_instanceof(hint, CYPHER_AST_MATCH_HINT));
    ck_assert_int_eq(cypher_astnode_type(hint), CYPHER_AST_USING_INDEX);

    const cypher_astnode_t *id = cypher_ast_using_index_get_identifier(hint);
    ck_assert_int_eq(cypher_astnode_type(id), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(id), "n");

    const cypher_astnode_t *label = cypher_ast_using_index_get_label(hint);
    ck_assert_int_eq(cypher_astnode_type(label), CYPHER_AST_LABEL);
    ck_assert_str_eq(cypher_ast_label_get_name(label), "Foo");

    const cypher_astnode_t *prop_name = cypher_ast_using_index_get_prop_name(hint);
    ck_assert_int_eq(cypher_astnode_type(prop_name), CYPHER_AST_PROP_NAME);
    ck_assert_str_eq(cypher_ast_prop_name_get_value(prop_name), "bar");

    ck_assert_ptr_eq(cypher_ast_match_get_hint(match, 1), NULL);
    ck_assert_ptr_eq(cypher_ast_match_get_predicate(match), NULL);
}
END_TEST


START_TEST (parse_match_with_using_join_hint)
{
    struct cypher_input_position last = cypher_input_position_zero;
    result = cypher_parse("MATCH (n), (m) USING JOIN ON n, m RETURN n;",
            &last, NULL, 0);
    ck_assert_ptr_ne(result, NULL);
    ck_assert_int_eq(last.offset, 43);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
" @0   0..43  statement               body=@1\n"
" @1   0..43  > query                 clauses=[@2, @13]\n"
" @2   0..34  > > MATCH               pattern=@3, hints=[@10]\n"
" @3   6..14  > > > pattern           paths=[@4, @7]\n"
" @4   6..9   > > > > pattern path    (@5)\n"
" @5   6..9   > > > > > node pattern  (@6)\n"
" @6   7..8   > > > > > > identifier  `n`\n"
" @7  11..14  > > > > pattern path    (@8)\n"
" @8  11..14  > > > > > node pattern  (@9)\n"
" @9  12..13  > > > > > > identifier  `m`\n"
"@10  15..34  > > > USING JOIN        on=[@11, @12]\n"
"@11  29..30  > > > > identifier      `n`\n"
"@12  32..33  > > > > identifier      `m`\n"
"@13  34..42  > > RETURN              projections=[@14]\n"
"@14  41..42  > > > projection        expression=@15\n"
"@15  41..42  > > > > identifier      `n`\n";
    ck_assert_str_eq(memstream_buffer, expected);

    const cypher_astnode_t *ast = cypher_parse_result_get_directive(result, 0);
    ck_assert_int_eq(cypher_astnode_type(ast), CYPHER_AST_STATEMENT);
    const cypher_astnode_t *query = cypher_ast_statement_get_body(ast);
    ck_assert_int_eq(cypher_astnode_type(query), CYPHER_AST_QUERY);
    const cypher_astnode_t *match = cypher_ast_query_get_clause(query, 0);
    ck_assert_int_eq(cypher_astnode_type(match), CYPHER_AST_MATCH);

    ck_assert(!cypher_ast_match_is_optional(match));

    const cypher_astnode_t *pattern = cypher_ast_match_get_pattern(match);
    ck_assert_int_eq(cypher_astnode_type(pattern), CYPHER_AST_PATTERN);
    ck_assert_int_eq(cypher_ast_pattern_npaths(pattern), 2);
    const cypher_astnode_t *path = cypher_ast_pattern_get_path(pattern, 0);
    ck_assert_int_eq(cypher_astnode_type(path), CYPHER_AST_PATTERN_PATH);
    path = cypher_ast_pattern_get_path(pattern, 1);
    ck_assert_int_eq(cypher_astnode_type(path), CYPHER_AST_PATTERN_PATH);
    ck_assert_ptr_eq(cypher_ast_pattern_get_path(pattern, 2), NULL);

    ck_assert_int_eq(cypher_ast_match_nhints(match), 1);

    const cypher_astnode_t *hint = cypher_ast_match_get_hint(match, 0);
    ck_assert(cypher_astnode_instanceof(hint, CYPHER_AST_MATCH_HINT));
    ck_assert_int_eq(cypher_astnode_type(hint), CYPHER_AST_USING_JOIN);

    ck_assert_int_eq(cypher_ast_using_join_nidentifiers(hint), 2);

    const cypher_astnode_t *id = cypher_ast_using_join_get_identifier(hint, 0);
    ck_assert_int_eq(cypher_astnode_type(id), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(id), "n");

    id = cypher_ast_using_join_get_identifier(hint, 1);
    ck_assert_int_eq(cypher_astnode_type(id), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(id), "m");

    ck_assert_ptr_eq(cypher_ast_match_get_hint(match, 1), NULL);
    ck_assert_ptr_eq(cypher_ast_match_get_predicate(match), NULL);
}
END_TEST


START_TEST (parse_match_with_using_scan_hint)
{
    struct cypher_input_position last = cypher_input_position_zero;
    result = cypher_parse("MATCH (n:Foo) USING SCAN n:Foo RETURN n;",
            &last, NULL, 0);
    ck_assert_ptr_ne(result, NULL);
    ck_assert_int_eq(last.offset, 40);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
" @0   0..40  statement               body=@1\n"
" @1   0..40  > query                 clauses=[@2, @11]\n"
" @2   0..31  > > MATCH               pattern=@3, hints=[@8]\n"
" @3   6..13  > > > pattern           paths=[@4]\n"
" @4   6..13  > > > > pattern path    (@5)\n"
" @5   6..13  > > > > > node pattern  (@6:@7)\n"
" @6   7..8   > > > > > > identifier  `n`\n"
" @7   8..12  > > > > > > label       :`Foo`\n"
" @8  14..31  > > > USING SCAN        @9:@10\n"
" @9  25..26  > > > > identifier      `n`\n"
"@10  26..30  > > > > label           :`Foo`\n"
"@11  31..39  > > RETURN              projections=[@12]\n"
"@12  38..39  > > > projection        expression=@13\n"
"@13  38..39  > > > > identifier      `n`\n";
    ck_assert_str_eq(memstream_buffer, expected);

    const cypher_astnode_t *ast = cypher_parse_result_get_directive(result, 0);
    ck_assert_int_eq(cypher_astnode_type(ast), CYPHER_AST_STATEMENT);
    const cypher_astnode_t *query = cypher_ast_statement_get_body(ast);
    ck_assert_int_eq(cypher_astnode_type(query), CYPHER_AST_QUERY);
    const cypher_astnode_t *match = cypher_ast_query_get_clause(query, 0);
    ck_assert_int_eq(cypher_astnode_type(match), CYPHER_AST_MATCH);

    ck_assert(!cypher_ast_match_is_optional(match));

    const cypher_astnode_t *pattern = cypher_ast_match_get_pattern(match);
    ck_assert_int_eq(cypher_astnode_type(pattern), CYPHER_AST_PATTERN);
    ck_assert_int_eq(cypher_ast_pattern_npaths(pattern), 1);

    ck_assert_int_eq(cypher_ast_match_nhints(match), 1);

    const cypher_astnode_t *hint = cypher_ast_match_get_hint(match, 0);
    ck_assert(cypher_astnode_instanceof(hint, CYPHER_AST_MATCH_HINT));
    ck_assert_int_eq(cypher_astnode_type(hint), CYPHER_AST_USING_SCAN);

    const cypher_astnode_t *id = cypher_ast_using_scan_get_identifier(hint);
    ck_assert_int_eq(cypher_astnode_type(id), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(id), "n");

    const cypher_astnode_t *label = cypher_ast_using_scan_get_label(hint);
    ck_assert_int_eq(cypher_astnode_type(label), CYPHER_AST_LABEL);
    ck_assert_str_eq(cypher_ast_label_get_name(label), "Foo");

    ck_assert_ptr_eq(cypher_ast_match_get_hint(match, 1), NULL);
    ck_assert_ptr_eq(cypher_ast_match_get_predicate(match), NULL);
}
END_TEST


TCase* match_tcase(void)
{
    TCase *tc = tcase_create("match");
    tcase_add_checked_fixture(tc, setup, teardown);
    tcase_add_test(tc, parse_simple_match);
    tcase_add_test(tc, parse_simple_optional_match);
    tcase_add_test(tc, parse_match_with_predicate);
    tcase_add_test(tc, parse_match_with_using_index_hint);
    tcase_add_test(tc, parse_match_with_using_join_hint);
    tcase_add_test(tc, parse_match_with_using_scan_hint);
    return tc;
}
