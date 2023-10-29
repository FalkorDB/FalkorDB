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


START_TEST (parse_single_node)
{
    struct cypher_input_position last = cypher_input_position_zero;
    result = cypher_parse("MATCH (n) RETURN n;", &last, NULL, 0);
    ck_assert_ptr_ne(result, NULL);
    ck_assert_int_eq(last.offset, 19);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
"@0   0..19  statement               body=@1\n"
"@1   0..19  > query                 clauses=[@2, @7]\n"
"@2   0..10  > > MATCH               pattern=@3\n"
"@3   6..9   > > > pattern           paths=[@4]\n"
"@4   6..9   > > > > pattern path    (@5)\n"
"@5   6..9   > > > > > node pattern  (@6)\n"
"@6   7..8   > > > > > > identifier  `n`\n"
"@7  10..18  > > RETURN              projections=[@8]\n"
"@8  17..18  > > > projection        expression=@9\n"
"@9  17..18  > > > > identifier      `n`\n";
    ck_assert_str_eq(memstream_buffer, expected);

    const cypher_astnode_t *ast = cypher_parse_result_get_directive(result, 0);
    const cypher_astnode_t *query = cypher_ast_statement_get_body(ast);
    const cypher_astnode_t *match = cypher_ast_query_get_clause(query, 0);

    const cypher_astnode_t *pattern = cypher_ast_match_get_pattern(match);
    ck_assert_int_eq(cypher_astnode_type(pattern), CYPHER_AST_PATTERN);

    ck_assert_int_eq(cypher_ast_pattern_npaths(pattern), 1);
    ck_assert_ptr_eq(cypher_ast_pattern_get_path(pattern, 1), NULL);
    const cypher_astnode_t *path = cypher_ast_pattern_get_path(pattern, 0);
    ck_assert_int_eq(cypher_astnode_type(path), CYPHER_AST_PATTERN_PATH);

    ck_assert_int_eq(cypher_ast_pattern_path_nelements(path), 1);
    ck_assert_ptr_eq(cypher_ast_pattern_path_get_element(path, 1), NULL);
    const cypher_astnode_t *node = cypher_ast_pattern_path_get_element(path, 0);
    ck_assert_int_eq(cypher_astnode_type(node), CYPHER_AST_NODE_PATTERN);

    const cypher_astnode_t *id = cypher_ast_node_pattern_get_identifier(node);
    ck_assert_int_eq(cypher_astnode_type(id), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(id), "n");

    ck_assert_int_eq(cypher_ast_node_pattern_nlabels(node), 0);
    ck_assert_ptr_eq(cypher_ast_node_pattern_get_label(node, 0), NULL);
    ck_assert_ptr_eq(cypher_ast_node_pattern_get_properties(node), NULL);
}
END_TEST


START_TEST (parse_labeled_node)
{
    struct cypher_input_position last = cypher_input_position_zero;
    result = cypher_parse("MATCH (n:Foo) RETURN n;", &last, NULL, 0);
    ck_assert_ptr_ne(result, NULL);
    ck_assert_int_eq(last.offset, 23);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
" @0   0..23  statement               body=@1\n"
" @1   0..23  > query                 clauses=[@2, @8]\n"
" @2   0..14  > > MATCH               pattern=@3\n"
" @3   6..13  > > > pattern           paths=[@4]\n"
" @4   6..13  > > > > pattern path    (@5)\n"
" @5   6..13  > > > > > node pattern  (@6:@7)\n"
" @6   7..8   > > > > > > identifier  `n`\n"
" @7   8..12  > > > > > > label       :`Foo`\n"
" @8  14..22  > > RETURN              projections=[@9]\n"
" @9  21..22  > > > projection        expression=@10\n"
"@10  21..22  > > > > identifier      `n`\n";
    ck_assert_str_eq(memstream_buffer, expected);

    const cypher_astnode_t *ast = cypher_parse_result_get_directive(result, 0);
    const cypher_astnode_t *query = cypher_ast_statement_get_body(ast);
    const cypher_astnode_t *match = cypher_ast_query_get_clause(query, 0);

    const cypher_astnode_t *pattern = cypher_ast_match_get_pattern(match);
    ck_assert_int_eq(cypher_astnode_type(pattern), CYPHER_AST_PATTERN);

    ck_assert_int_eq(cypher_ast_pattern_npaths(pattern), 1);
    ck_assert_ptr_eq(cypher_ast_pattern_get_path(pattern, 1), NULL);
    const cypher_astnode_t *path = cypher_ast_pattern_get_path(pattern, 0);
    ck_assert_int_eq(cypher_astnode_type(path), CYPHER_AST_PATTERN_PATH);

    ck_assert_int_eq(cypher_ast_pattern_path_nelements(path), 1);
    ck_assert_ptr_eq(cypher_ast_pattern_path_get_element(path, 1), NULL);
    const cypher_astnode_t *node = cypher_ast_pattern_path_get_element(path, 0);
    ck_assert_int_eq(cypher_astnode_type(node), CYPHER_AST_NODE_PATTERN);

    const cypher_astnode_t *id = cypher_ast_node_pattern_get_identifier(node);
    ck_assert_int_eq(cypher_astnode_type(id), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(id), "n");

    ck_assert_int_eq(cypher_ast_node_pattern_nlabels(node), 1);
    ck_assert_ptr_eq(cypher_ast_node_pattern_get_label(node, 1), NULL);
    const cypher_astnode_t *label = cypher_ast_node_pattern_get_label(node, 0);
    ck_assert_int_eq(cypher_astnode_type(label), CYPHER_AST_LABEL);
    ck_assert_str_eq(cypher_ast_label_get_name(label), "Foo");

    ck_assert_ptr_eq(cypher_ast_node_pattern_get_properties(node), NULL);
}
END_TEST


START_TEST (parse_multiple_labeled_node)
{
    struct cypher_input_position last = cypher_input_position_zero;
    result = cypher_parse("MATCH (n:Foo:Bar) RETURN n;", &last, NULL, 0);
    ck_assert_ptr_ne(result, NULL);
    ck_assert_int_eq(last.offset, 27);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
" @0   0..27  statement               body=@1\n"
" @1   0..27  > query                 clauses=[@2, @9]\n"
" @2   0..18  > > MATCH               pattern=@3\n"
" @3   6..17  > > > pattern           paths=[@4]\n"
" @4   6..17  > > > > pattern path    (@5)\n"
" @5   6..17  > > > > > node pattern  (@6:@7:@8)\n"
" @6   7..8   > > > > > > identifier  `n`\n"
" @7   8..12  > > > > > > label       :`Foo`\n"
" @8  12..16  > > > > > > label       :`Bar`\n"
" @9  18..26  > > RETURN              projections=[@10]\n"
"@10  25..26  > > > projection        expression=@11\n"
"@11  25..26  > > > > identifier      `n`\n";
    ck_assert_str_eq(memstream_buffer, expected);

    const cypher_astnode_t *ast = cypher_parse_result_get_directive(result, 0);
    const cypher_astnode_t *query = cypher_ast_statement_get_body(ast);
    const cypher_astnode_t *match = cypher_ast_query_get_clause(query, 0);

    const cypher_astnode_t *pattern = cypher_ast_match_get_pattern(match);
    ck_assert_int_eq(cypher_astnode_type(pattern), CYPHER_AST_PATTERN);

    ck_assert_int_eq(cypher_ast_pattern_npaths(pattern), 1);
    ck_assert_ptr_eq(cypher_ast_pattern_get_path(pattern, 1), NULL);
    const cypher_astnode_t *path = cypher_ast_pattern_get_path(pattern, 0);
    ck_assert_int_eq(cypher_astnode_type(path), CYPHER_AST_PATTERN_PATH);

    ck_assert_int_eq(cypher_ast_pattern_path_nelements(path), 1);
    ck_assert_ptr_eq(cypher_ast_pattern_path_get_element(path, 1), NULL);
    const cypher_astnode_t *node = cypher_ast_pattern_path_get_element(path, 0);
    ck_assert_int_eq(cypher_astnode_type(node), CYPHER_AST_NODE_PATTERN);

    const cypher_astnode_t *id = cypher_ast_node_pattern_get_identifier(node);
    ck_assert_int_eq(cypher_astnode_type(id), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(id), "n");

    ck_assert_int_eq(cypher_ast_node_pattern_nlabels(node), 2);
    ck_assert_ptr_eq(cypher_ast_node_pattern_get_label(node, 2), NULL);
    const cypher_astnode_t *label = cypher_ast_node_pattern_get_label(node, 0);
    ck_assert_int_eq(cypher_astnode_type(label), CYPHER_AST_LABEL);
    ck_assert_str_eq(cypher_ast_label_get_name(label), "Foo");
    label = cypher_ast_node_pattern_get_label(node, 1);
    ck_assert_int_eq(cypher_astnode_type(label), CYPHER_AST_LABEL);
    ck_assert_str_eq(cypher_ast_label_get_name(label), "Bar");

    ck_assert_ptr_eq(cypher_ast_node_pattern_get_properties(node), NULL);
}
END_TEST


START_TEST (parse_node_with_map_props)
{
    struct cypher_input_position last = cypher_input_position_zero;
    result = cypher_parse("MATCH (n:Person {name: 'Hunter'}) RETURN n;",
            &last, NULL, 0);
    ck_assert_ptr_ne(result, NULL);
    ck_assert_int_eq(last.offset, 43);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
" @0   0..43  statement                body=@1\n"
" @1   0..43  > query                  clauses=[@2, @11]\n"
" @2   0..34  > > MATCH                pattern=@3\n"
" @3   6..33  > > > pattern            paths=[@4]\n"
" @4   6..33  > > > > pattern path     (@5)\n"
" @5   6..33  > > > > > node pattern   (@6:@7 {@8})\n"
" @6   7..8   > > > > > > identifier   `n`\n"
" @7   8..15  > > > > > > label        :`Person`\n"
" @8  16..32  > > > > > > map          {@9:@10}\n"
" @9  17..21  > > > > > > > prop name  `name`\n"
"@10  23..31  > > > > > > > string     \"Hunter\"\n"
"@11  34..42  > > RETURN               projections=[@12]\n"
"@12  41..42  > > > projection         expression=@13\n"
"@13  41..42  > > > > identifier       `n`\n";
    ck_assert_str_eq(memstream_buffer, expected);

    const cypher_astnode_t *ast = cypher_parse_result_get_directive(result, 0);
    const cypher_astnode_t *query = cypher_ast_statement_get_body(ast);
    const cypher_astnode_t *match = cypher_ast_query_get_clause(query, 0);

    const cypher_astnode_t *pattern = cypher_ast_match_get_pattern(match);
    ck_assert_int_eq(cypher_astnode_type(pattern), CYPHER_AST_PATTERN);

    ck_assert_int_eq(cypher_ast_pattern_npaths(pattern), 1);
    ck_assert_ptr_eq(cypher_ast_pattern_get_path(pattern, 1), NULL);
    const cypher_astnode_t *path = cypher_ast_pattern_get_path(pattern, 0);
    ck_assert_int_eq(cypher_astnode_type(path), CYPHER_AST_PATTERN_PATH);

    ck_assert_int_eq(cypher_ast_pattern_path_nelements(path), 1);
    ck_assert_ptr_eq(cypher_ast_pattern_path_get_element(path, 1), NULL);
    const cypher_astnode_t *node = cypher_ast_pattern_path_get_element(path, 0);
    ck_assert_int_eq(cypher_astnode_type(node), CYPHER_AST_NODE_PATTERN);

    const cypher_astnode_t *id = cypher_ast_node_pattern_get_identifier(node);
    ck_assert_int_eq(cypher_astnode_type(id), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(id), "n");

    ck_assert_int_eq(cypher_ast_node_pattern_nlabels(node), 1);
    ck_assert_ptr_eq(cypher_ast_node_pattern_get_label(node, 1), NULL);
    const cypher_astnode_t *label = cypher_ast_node_pattern_get_label(node, 0);
    ck_assert_int_eq(cypher_astnode_type(label), CYPHER_AST_LABEL);
    ck_assert_str_eq(cypher_ast_label_get_name(label), "Person");

    const cypher_astnode_t *props = cypher_ast_node_pattern_get_properties(node);
    ck_assert_int_eq(cypher_astnode_type(props), CYPHER_AST_MAP);

    ck_assert_int_eq(cypher_ast_map_nentries(props), 1);
    ck_assert_ptr_eq(cypher_ast_map_get_key(props, 1), NULL);
    ck_assert_ptr_eq(cypher_ast_map_get_value(props, 1), NULL);
    const cypher_astnode_t *key = cypher_ast_map_get_key(props, 0);
    ck_assert_int_eq(cypher_astnode_type(key), CYPHER_AST_PROP_NAME);
    ck_assert_str_eq(cypher_ast_prop_name_get_value(key), "name");
    const cypher_astnode_t *value = cypher_ast_map_get_value(props, 0);
    ck_assert_int_eq(cypher_astnode_type(value), CYPHER_AST_STRING);
    ck_assert_str_eq(cypher_ast_string_get_value(value), "Hunter");
}
END_TEST


START_TEST (parse_node_with_param_props)
{
    struct cypher_input_position last = cypher_input_position_zero;
    result = cypher_parse("MATCH (n:Person {param}) RETURN n;",
            &last, NULL, 0);
    ck_assert_ptr_ne(result, NULL);
    ck_assert_int_eq(last.offset, 34);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
" @0   0..34  statement               body=@1\n"
" @1   0..34  > query                 clauses=[@2, @9]\n"
" @2   0..25  > > MATCH               pattern=@3\n"
" @3   6..24  > > > pattern           paths=[@4]\n"
" @4   6..24  > > > > pattern path    (@5)\n"
" @5   6..24  > > > > > node pattern  (@6:@7 {@8})\n"
" @6   7..8   > > > > > > identifier  `n`\n"
" @7   8..15  > > > > > > label       :`Person`\n"
" @8  16..23  > > > > > > parameter   $`param`\n"
" @9  25..33  > > RETURN              projections=[@10]\n"
"@10  32..33  > > > projection        expression=@11\n"
"@11  32..33  > > > > identifier      `n`\n";
    ck_assert_str_eq(memstream_buffer, expected);

    const cypher_astnode_t *ast = cypher_parse_result_get_directive(result, 0);
    const cypher_astnode_t *query = cypher_ast_statement_get_body(ast);
    const cypher_astnode_t *match = cypher_ast_query_get_clause(query, 0);

    const cypher_astnode_t *pattern = cypher_ast_match_get_pattern(match);
    ck_assert_int_eq(cypher_astnode_type(pattern), CYPHER_AST_PATTERN);

    ck_assert_int_eq(cypher_ast_pattern_npaths(pattern), 1);
    ck_assert_ptr_eq(cypher_ast_pattern_get_path(pattern, 1), NULL);
    const cypher_astnode_t *path = cypher_ast_pattern_get_path(pattern, 0);
    ck_assert_int_eq(cypher_astnode_type(path), CYPHER_AST_PATTERN_PATH);

    ck_assert_int_eq(cypher_ast_pattern_path_nelements(path), 1);
    ck_assert_ptr_eq(cypher_ast_pattern_path_get_element(path, 1), NULL);
    const cypher_astnode_t *node = cypher_ast_pattern_path_get_element(path, 0);
    ck_assert_int_eq(cypher_astnode_type(node), CYPHER_AST_NODE_PATTERN);

    const cypher_astnode_t *id = cypher_ast_node_pattern_get_identifier(node);
    ck_assert_int_eq(cypher_astnode_type(id), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(id), "n");

    ck_assert_int_eq(cypher_ast_node_pattern_nlabels(node), 1);
    ck_assert_ptr_eq(cypher_ast_node_pattern_get_label(node, 1), NULL);
    const cypher_astnode_t *label = cypher_ast_node_pattern_get_label(node, 0);
    ck_assert_int_eq(cypher_astnode_type(label), CYPHER_AST_LABEL);
    ck_assert_str_eq(cypher_ast_label_get_name(label), "Person");

    const cypher_astnode_t *props = cypher_ast_node_pattern_get_properties(node);
    ck_assert_int_eq(cypher_astnode_type(props), CYPHER_AST_PARAMETER);
    ck_assert_str_eq(cypher_ast_parameter_get_name(props), "param");
}
END_TEST


START_TEST (parse_single_rel)
{
    struct cypher_input_position last = cypher_input_position_zero;
    result = cypher_parse("MATCH (n)-[:Foo]->(m) RETURN n;", &last, NULL, 0);
    ck_assert_ptr_ne(result, NULL);
    ck_assert_int_eq(last.offset, 31);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
" @0   0..31  statement               body=@1\n"
" @1   0..31  > query                 clauses=[@2, @11]\n"
" @2   0..22  > > MATCH               pattern=@3\n"
" @3   6..21  > > > pattern           paths=[@4]\n"
" @4   6..21  > > > > pattern path    (@5)-[@7]-(@9)\n"
" @5   6..9   > > > > > node pattern  (@6)\n"
" @6   7..8   > > > > > > identifier  `n`\n"
" @7   9..18  > > > > > rel pattern   -[:@8]->\n"
" @8  11..15  > > > > > > rel type    :`Foo`\n"
" @9  18..21  > > > > > node pattern  (@10)\n"
"@10  19..20  > > > > > > identifier  `m`\n"
"@11  22..30  > > RETURN              projections=[@12]\n"
"@12  29..30  > > > projection        expression=@13\n"
"@13  29..30  > > > > identifier      `n`\n";
    ck_assert_str_eq(memstream_buffer, expected);

    const cypher_astnode_t *ast = cypher_parse_result_get_directive(result, 0);
    const cypher_astnode_t *query = cypher_ast_statement_get_body(ast);
    const cypher_astnode_t *match = cypher_ast_query_get_clause(query, 0);

    const cypher_astnode_t *pattern = cypher_ast_match_get_pattern(match);
    ck_assert_int_eq(cypher_astnode_type(pattern), CYPHER_AST_PATTERN);

    ck_assert_int_eq(cypher_ast_pattern_npaths(pattern), 1);
    ck_assert_ptr_eq(cypher_ast_pattern_get_path(pattern, 1), NULL);
    const cypher_astnode_t *path = cypher_ast_pattern_get_path(pattern, 0);
    ck_assert_int_eq(cypher_astnode_type(path), CYPHER_AST_PATTERN_PATH);

    ck_assert_int_eq(cypher_ast_pattern_path_nelements(path), 3);
    ck_assert_ptr_eq(cypher_ast_pattern_path_get_element(path, 3), NULL);
    const cypher_astnode_t *node = cypher_ast_pattern_path_get_element(path, 0);
    ck_assert_int_eq(cypher_astnode_type(node), CYPHER_AST_NODE_PATTERN);

    const cypher_astnode_t *id = cypher_ast_node_pattern_get_identifier(node);
    ck_assert_int_eq(cypher_astnode_type(id), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(id), "n");

    ck_assert_int_eq(cypher_ast_node_pattern_nlabels(node), 0);
    ck_assert_ptr_eq(cypher_ast_node_pattern_get_label(node, 0), NULL);
    ck_assert_ptr_eq(cypher_ast_node_pattern_get_properties(node), NULL);

    const cypher_astnode_t *rel = cypher_ast_pattern_path_get_element(path, 1);
    ck_assert_int_eq(cypher_astnode_type(rel), CYPHER_AST_REL_PATTERN);

    ck_assert_int_eq(cypher_ast_rel_pattern_get_direction(rel), CYPHER_REL_OUTBOUND);

    ck_assert_ptr_eq(cypher_ast_rel_pattern_get_identifier(rel), NULL);

    ck_assert_int_eq(cypher_ast_rel_pattern_nreltypes(rel), 1);
    const cypher_astnode_t *reltype = cypher_ast_rel_pattern_get_reltype(rel, 0);
    ck_assert_int_eq(cypher_astnode_type(reltype), CYPHER_AST_RELTYPE);
    ck_assert_str_eq(cypher_ast_reltype_get_name(reltype), "Foo");
    ck_assert_ptr_eq(cypher_ast_rel_pattern_get_reltype(rel, 2), NULL);

    ck_assert_ptr_eq(cypher_ast_rel_pattern_get_varlength(rel), NULL);

    node = cypher_ast_pattern_path_get_element(path, 2);
    ck_assert_int_eq(cypher_astnode_type(node), CYPHER_AST_NODE_PATTERN);
}
END_TEST


START_TEST (parse_varlength_rel)
{
    struct cypher_input_position last = cypher_input_position_zero;
    result = cypher_parse("MATCH (n)-[r:Foo*]-(m) RETURN n;", &last, NULL, 0);
    ck_assert_ptr_ne(result, NULL);
    ck_assert_int_eq(last.offset, 32);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
" @0   0..32  statement               body=@1\n"
" @1   0..32  > query                 clauses=[@2, @13]\n"
" @2   0..23  > > MATCH               pattern=@3\n"
" @3   6..22  > > > pattern           paths=[@4]\n"
" @4   6..22  > > > > pattern path    (@5)-[@7]-(@11)\n"
" @5   6..9   > > > > > node pattern  (@6)\n"
" @6   7..8   > > > > > > identifier  `n`\n"
" @7   9..19  > > > > > rel pattern   -[@8:@9*@10]-\n"
" @8  11..12  > > > > > > identifier  `r`\n"
" @9  12..16  > > > > > > rel type    :`Foo`\n"
"@10  17..17  > > > > > > range       *\n"
"@11  19..22  > > > > > node pattern  (@12)\n"
"@12  20..21  > > > > > > identifier  `m`\n"
"@13  23..31  > > RETURN              projections=[@14]\n"
"@14  30..31  > > > projection        expression=@15\n"
"@15  30..31  > > > > identifier      `n`\n";
    ck_assert_str_eq(memstream_buffer, expected);

    const cypher_astnode_t *ast = cypher_parse_result_get_directive(result, 0);
    const cypher_astnode_t *query = cypher_ast_statement_get_body(ast);
    const cypher_astnode_t *match = cypher_ast_query_get_clause(query, 0);

    const cypher_astnode_t *pattern = cypher_ast_match_get_pattern(match);
    ck_assert_int_eq(cypher_astnode_type(pattern), CYPHER_AST_PATTERN);

    ck_assert_int_eq(cypher_ast_pattern_npaths(pattern), 1);
    ck_assert_ptr_eq(cypher_ast_pattern_get_path(pattern, 1), NULL);
    const cypher_astnode_t *path = cypher_ast_pattern_get_path(pattern, 0);
    ck_assert_int_eq(cypher_astnode_type(path), CYPHER_AST_PATTERN_PATH);

    ck_assert_int_eq(cypher_ast_pattern_path_nelements(path), 3);
    ck_assert_ptr_eq(cypher_ast_pattern_path_get_element(path, 3), NULL);
    const cypher_astnode_t *node = cypher_ast_pattern_path_get_element(path, 0);
    ck_assert_int_eq(cypher_astnode_type(node), CYPHER_AST_NODE_PATTERN);

    const cypher_astnode_t *id = cypher_ast_node_pattern_get_identifier(node);
    ck_assert_int_eq(cypher_astnode_type(id), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(id), "n");

    ck_assert_int_eq(cypher_ast_node_pattern_nlabels(node), 0);
    ck_assert_ptr_eq(cypher_ast_node_pattern_get_label(node, 0), NULL);
    ck_assert_ptr_eq(cypher_ast_node_pattern_get_properties(node), NULL);

    const cypher_astnode_t *rel = cypher_ast_pattern_path_get_element(path, 1);
    ck_assert_int_eq(cypher_astnode_type(rel), CYPHER_AST_REL_PATTERN);

    ck_assert_int_eq(cypher_ast_rel_pattern_get_direction(rel),
            CYPHER_REL_BIDIRECTIONAL);

    id = cypher_ast_rel_pattern_get_identifier(rel);
    ck_assert_int_eq(cypher_astnode_type(id), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(id), "r");

    ck_assert_int_eq(cypher_ast_rel_pattern_nreltypes(rel), 1);
    const cypher_astnode_t *reltype = cypher_ast_rel_pattern_get_reltype(rel, 0);
    ck_assert_int_eq(cypher_astnode_type(reltype), CYPHER_AST_RELTYPE);
    ck_assert_str_eq(cypher_ast_reltype_get_name(reltype), "Foo");
    ck_assert_ptr_eq(cypher_ast_rel_pattern_get_reltype(rel, 2), NULL);

    const cypher_astnode_t *range = cypher_ast_rel_pattern_get_varlength(rel);
    ck_assert_int_eq(cypher_astnode_type(range), CYPHER_AST_RANGE);
    ck_assert_ptr_eq(cypher_ast_range_get_start(range), NULL);
    ck_assert_ptr_eq(cypher_ast_range_get_end(range), NULL);

    node = cypher_ast_pattern_path_get_element(path, 2);
    ck_assert_int_eq(cypher_astnode_type(node), CYPHER_AST_NODE_PATTERN);
}
END_TEST


START_TEST (parse_varlength_rel_with_bounded_start)
{
    struct cypher_input_position last = cypher_input_position_zero;
    result = cypher_parse("MATCH (n)<-[r:Foo*5..]-(m) RETURN n;", &last, NULL, 0);
    ck_assert_ptr_ne(result, NULL);
    ck_assert_int_eq(last.offset, 36);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
" @0   0..36  statement               body=@1\n"
" @1   0..36  > query                 clauses=[@2, @14]\n"
" @2   0..27  > > MATCH               pattern=@3\n"
" @3   6..26  > > > pattern           paths=[@4]\n"
" @4   6..26  > > > > pattern path    (@5)-[@7]-(@12)\n"
" @5   6..9   > > > > > node pattern  (@6)\n"
" @6   7..8   > > > > > > identifier  `n`\n"
" @7   9..23  > > > > > rel pattern   <-[@8:@9*@10]-\n"
" @8  12..13  > > > > > > identifier  `r`\n"
" @9  13..17  > > > > > > rel type    :`Foo`\n"
"@10  17..21  > > > > > > range       *@11..\n"
"@11  18..19  > > > > > > > integer   5\n"
"@12  23..26  > > > > > node pattern  (@13)\n"
"@13  24..25  > > > > > > identifier  `m`\n"
"@14  27..35  > > RETURN              projections=[@15]\n"
"@15  34..35  > > > projection        expression=@16\n"
"@16  34..35  > > > > identifier      `n`\n";
    ck_assert_str_eq(memstream_buffer, expected);

    const cypher_astnode_t *ast = cypher_parse_result_get_directive(result, 0);
    const cypher_astnode_t *query = cypher_ast_statement_get_body(ast);
    const cypher_astnode_t *match = cypher_ast_query_get_clause(query, 0);

    const cypher_astnode_t *pattern = cypher_ast_match_get_pattern(match);
    ck_assert_int_eq(cypher_astnode_type(pattern), CYPHER_AST_PATTERN);

    ck_assert_int_eq(cypher_ast_pattern_npaths(pattern), 1);
    const cypher_astnode_t *path = cypher_ast_pattern_get_path(pattern, 0);
    ck_assert_int_eq(cypher_astnode_type(path), CYPHER_AST_PATTERN_PATH);

    ck_assert_int_eq(cypher_ast_pattern_path_nelements(path), 3);
    const cypher_astnode_t *node = cypher_ast_pattern_path_get_element(path, 0);
    ck_assert_int_eq(cypher_astnode_type(node), CYPHER_AST_NODE_PATTERN);

    const cypher_astnode_t *id = cypher_ast_node_pattern_get_identifier(node);
    ck_assert_int_eq(cypher_astnode_type(id), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(id), "n");

    ck_assert_int_eq(cypher_ast_node_pattern_nlabels(node), 0);
    ck_assert_ptr_eq(cypher_ast_node_pattern_get_label(node, 0), NULL);
    ck_assert_ptr_eq(cypher_ast_node_pattern_get_properties(node), NULL);

    const cypher_astnode_t *rel = cypher_ast_pattern_path_get_element(path, 1);
    ck_assert_int_eq(cypher_astnode_type(rel), CYPHER_AST_REL_PATTERN);

    ck_assert_int_eq(cypher_ast_rel_pattern_get_direction(rel), CYPHER_REL_INBOUND);

    id = cypher_ast_rel_pattern_get_identifier(rel);
    ck_assert_int_eq(cypher_astnode_type(id), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(id), "r");

    ck_assert_int_eq(cypher_ast_rel_pattern_nreltypes(rel), 1);
    const cypher_astnode_t *reltype = cypher_ast_rel_pattern_get_reltype(rel, 0);
    ck_assert_int_eq(cypher_astnode_type(reltype), CYPHER_AST_RELTYPE);
    ck_assert_str_eq(cypher_ast_reltype_get_name(reltype), "Foo");
    ck_assert_ptr_eq(cypher_ast_rel_pattern_get_reltype(rel, 2), NULL);

    const cypher_astnode_t *range = cypher_ast_rel_pattern_get_varlength(rel);
    ck_assert_int_eq(cypher_astnode_type(range), CYPHER_AST_RANGE);
    const cypher_astnode_t *start = cypher_ast_range_get_start(range);
    ck_assert_int_eq(cypher_astnode_type(start), CYPHER_AST_INTEGER);
    ck_assert_str_eq(cypher_ast_integer_get_valuestr(start), "5");
    ck_assert_ptr_eq(cypher_ast_range_get_end(range), NULL);

    node = cypher_ast_pattern_path_get_element(path, 2);
    ck_assert_int_eq(cypher_astnode_type(node), CYPHER_AST_NODE_PATTERN);
}
END_TEST


START_TEST (parse_varlength_rel_with_bounded_end)
{
    struct cypher_input_position last = cypher_input_position_zero;
    result = cypher_parse("MATCH (n)<-[r:Foo*..9]->(m) RETURN n;", &last, NULL, 0);
    ck_assert_ptr_ne(result, NULL);
    ck_assert_int_eq(last.offset, 37);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
" @0   0..37  statement               body=@1\n"
" @1   0..37  > query                 clauses=[@2, @14]\n"
" @2   0..28  > > MATCH               pattern=@3\n"
" @3   6..27  > > > pattern           paths=[@4]\n"
" @4   6..27  > > > > pattern path    (@5)-[@7]-(@12)\n"
" @5   6..9   > > > > > node pattern  (@6)\n"
" @6   7..8   > > > > > > identifier  `n`\n"
" @7   9..24  > > > > > rel pattern   -[@8:@9*@10]-\n"
" @8  12..13  > > > > > > identifier  `r`\n"
" @9  13..17  > > > > > > rel type    :`Foo`\n"
"@10  17..21  > > > > > > range       *..@11\n"
"@11  20..21  > > > > > > > integer   9\n"
"@12  24..27  > > > > > node pattern  (@13)\n"
"@13  25..26  > > > > > > identifier  `m`\n"
"@14  28..36  > > RETURN              projections=[@15]\n"
"@15  35..36  > > > projection        expression=@16\n"
"@16  35..36  > > > > identifier      `n`\n";
    ck_assert_str_eq(memstream_buffer, expected);

    const cypher_astnode_t *ast = cypher_parse_result_get_directive(result, 0);
    const cypher_astnode_t *query = cypher_ast_statement_get_body(ast);
    const cypher_astnode_t *match = cypher_ast_query_get_clause(query, 0);

    const cypher_astnode_t *pattern = cypher_ast_match_get_pattern(match);
    ck_assert_int_eq(cypher_astnode_type(pattern), CYPHER_AST_PATTERN);

    ck_assert_int_eq(cypher_ast_pattern_npaths(pattern), 1);
    const cypher_astnode_t *path = cypher_ast_pattern_get_path(pattern, 0);
    ck_assert_int_eq(cypher_astnode_type(path), CYPHER_AST_PATTERN_PATH);

    ck_assert_int_eq(cypher_ast_pattern_path_nelements(path), 3);
    const cypher_astnode_t *node = cypher_ast_pattern_path_get_element(path, 0);
    ck_assert_int_eq(cypher_astnode_type(node), CYPHER_AST_NODE_PATTERN);

    const cypher_astnode_t *id = cypher_ast_node_pattern_get_identifier(node);
    ck_assert_int_eq(cypher_astnode_type(id), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(id), "n");

    ck_assert_int_eq(cypher_ast_node_pattern_nlabels(node), 0);
    ck_assert_ptr_eq(cypher_ast_node_pattern_get_label(node, 0), NULL);
    ck_assert_ptr_eq(cypher_ast_node_pattern_get_properties(node), NULL);

    const cypher_astnode_t *rel = cypher_ast_pattern_path_get_element(path, 1);
    ck_assert_int_eq(cypher_astnode_type(rel), CYPHER_AST_REL_PATTERN);

    ck_assert_int_eq(cypher_ast_rel_pattern_get_direction(rel),
            CYPHER_REL_BIDIRECTIONAL);

    id = cypher_ast_rel_pattern_get_identifier(rel);
    ck_assert_int_eq(cypher_astnode_type(id), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(id), "r");

    ck_assert_int_eq(cypher_ast_rel_pattern_nreltypes(rel), 1);
    const cypher_astnode_t *reltype = cypher_ast_rel_pattern_get_reltype(rel, 0);
    ck_assert_int_eq(cypher_astnode_type(reltype), CYPHER_AST_RELTYPE);
    ck_assert_str_eq(cypher_ast_reltype_get_name(reltype), "Foo");
    ck_assert_ptr_eq(cypher_ast_rel_pattern_get_reltype(rel, 2), NULL);

    const cypher_astnode_t *range = cypher_ast_rel_pattern_get_varlength(rel);
    ck_assert_ptr_eq(cypher_ast_range_get_start(range), NULL);
    ck_assert_int_eq(cypher_astnode_type(range), CYPHER_AST_RANGE);
    const cypher_astnode_t *end = cypher_ast_range_get_end(range);
    ck_assert_int_eq(cypher_astnode_type(end), CYPHER_AST_INTEGER);
    ck_assert_str_eq(cypher_ast_integer_get_valuestr(end), "9");

    node = cypher_ast_pattern_path_get_element(path, 2);
    ck_assert_int_eq(cypher_astnode_type(node), CYPHER_AST_NODE_PATTERN);
}
END_TEST


START_TEST (parse_varlength_rel_with_fixed_range)
{
    struct cypher_input_position last = cypher_input_position_zero;
    result = cypher_parse("MATCH (n)<-[r:Foo*7]->(m) RETURN n;", &last, NULL, 0);
    ck_assert_ptr_ne(result, NULL);
    ck_assert_int_eq(last.offset, 35);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
" @0   0..35  statement               body=@1\n"
" @1   0..35  > query                 clauses=[@2, @14]\n"
" @2   0..26  > > MATCH               pattern=@3\n"
" @3   6..25  > > > pattern           paths=[@4]\n"
" @4   6..25  > > > > pattern path    (@5)-[@7]-(@12)\n"
" @5   6..9   > > > > > node pattern  (@6)\n"
" @6   7..8   > > > > > > identifier  `n`\n"
" @7   9..22  > > > > > rel pattern   -[@8:@9*@10]-\n"
" @8  12..13  > > > > > > identifier  `r`\n"
" @9  13..17  > > > > > > rel type    :`Foo`\n"
"@10  18..19  > > > > > > range       *@11..@11\n"
"@11  18..19  > > > > > > > integer   7\n"
"@12  22..25  > > > > > node pattern  (@13)\n"
"@13  23..24  > > > > > > identifier  `m`\n"
"@14  26..34  > > RETURN              projections=[@15]\n"
"@15  33..34  > > > projection        expression=@16\n"
"@16  33..34  > > > > identifier      `n`\n";
    ck_assert_str_eq(memstream_buffer, expected);

    const cypher_astnode_t *ast = cypher_parse_result_get_directive(result, 0);
    const cypher_astnode_t *query = cypher_ast_statement_get_body(ast);
    const cypher_astnode_t *match = cypher_ast_query_get_clause(query, 0);

    const cypher_astnode_t *pattern = cypher_ast_match_get_pattern(match);
    ck_assert_int_eq(cypher_astnode_type(pattern), CYPHER_AST_PATTERN);

    ck_assert_int_eq(cypher_ast_pattern_npaths(pattern), 1);
    const cypher_astnode_t *path = cypher_ast_pattern_get_path(pattern, 0);
    ck_assert_int_eq(cypher_astnode_type(path), CYPHER_AST_PATTERN_PATH);

    ck_assert_int_eq(cypher_ast_pattern_path_nelements(path), 3);
    const cypher_astnode_t *node = cypher_ast_pattern_path_get_element(path, 0);
    ck_assert_int_eq(cypher_astnode_type(node), CYPHER_AST_NODE_PATTERN);

    const cypher_astnode_t *id = cypher_ast_node_pattern_get_identifier(node);
    ck_assert_int_eq(cypher_astnode_type(id), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(id), "n");

    ck_assert_int_eq(cypher_ast_node_pattern_nlabels(node), 0);
    ck_assert_ptr_eq(cypher_ast_node_pattern_get_label(node, 0), NULL);
    ck_assert_ptr_eq(cypher_ast_node_pattern_get_properties(node), NULL);

    const cypher_astnode_t *rel = cypher_ast_pattern_path_get_element(path, 1);
    ck_assert_int_eq(cypher_astnode_type(rel), CYPHER_AST_REL_PATTERN);

    ck_assert_int_eq(cypher_ast_rel_pattern_get_direction(rel),
            CYPHER_REL_BIDIRECTIONAL);

    id = cypher_ast_rel_pattern_get_identifier(rel);
    ck_assert_int_eq(cypher_astnode_type(id), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(id), "r");

    ck_assert_int_eq(cypher_ast_rel_pattern_nreltypes(rel), 1);
    const cypher_astnode_t *reltype = cypher_ast_rel_pattern_get_reltype(rel, 0);
    ck_assert_int_eq(cypher_astnode_type(reltype), CYPHER_AST_RELTYPE);
    ck_assert_str_eq(cypher_ast_reltype_get_name(reltype), "Foo");
    ck_assert_ptr_eq(cypher_ast_rel_pattern_get_reltype(rel, 2), NULL);

    const cypher_astnode_t *range = cypher_ast_rel_pattern_get_varlength(rel);
    ck_assert_int_eq(cypher_astnode_type(range), CYPHER_AST_RANGE);
    const cypher_astnode_t *start = cypher_ast_range_get_end(range);
    const cypher_astnode_t *end = cypher_ast_range_get_end(range);
    ck_assert_int_eq(cypher_astnode_type(start), CYPHER_AST_INTEGER);
    ck_assert_str_eq(cypher_ast_integer_get_valuestr(start), "7");
    ck_assert_ptr_eq(start, end);

    node = cypher_ast_pattern_path_get_element(path, 2);
    ck_assert_int_eq(cypher_astnode_type(node), CYPHER_AST_NODE_PATTERN);
}
END_TEST


START_TEST (parse_rel_with_map_props)
{
    struct cypher_input_position last = cypher_input_position_zero;
    result = cypher_parse(
            "RETURN (n)-[:Foo {start:1999, end:2000}]->(m) AS p;",
            &last, NULL, 0);
    ck_assert_ptr_ne(result, NULL);
    ck_assert_int_eq(last.offset, 51);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
" @0   0..51  statement                body=@1\n"
" @1   0..51  > query                  clauses=[@2]\n"
" @2   0..50  > > RETURN               projections=[@3]\n"
" @3   7..50  > > > projection         expression=@4, alias=@16\n"
" @4   7..45  > > > > pattern path     (@5)-[@7]-(@14)\n"
" @5   7..10  > > > > > node pattern   (@6)\n"
" @6   8..9   > > > > > > identifier   `n`\n"
" @7  10..42  > > > > > rel pattern    -[:@8 {@9}]->\n"
" @8  12..16  > > > > > > rel type     :`Foo`\n"
" @9  17..39  > > > > > > map          {@10:@11, @12:@13}\n"
"@10  18..23  > > > > > > > prop name  `start`\n"
"@11  24..28  > > > > > > > integer    1999\n"
"@12  30..33  > > > > > > > prop name  `end`\n"
"@13  34..38  > > > > > > > integer    2000\n"
"@14  42..45  > > > > > node pattern   (@15)\n"
"@15  43..44  > > > > > > identifier   `m`\n"
"@16  49..50  > > > > identifier       `p`\n";
    ck_assert_str_eq(memstream_buffer, expected);

    const cypher_astnode_t *ast = cypher_parse_result_get_directive(result, 0);
    const cypher_astnode_t *query = cypher_ast_statement_get_body(ast);
    const cypher_astnode_t *clause = cypher_ast_query_get_clause(query, 0);
    ck_assert_int_eq(cypher_astnode_type(clause), CYPHER_AST_RETURN);

    const cypher_astnode_t *projection = cypher_ast_return_get_projection(clause, 0);
    ck_assert_int_eq(cypher_astnode_type(projection), CYPHER_AST_PROJECTION);
    const cypher_astnode_t *ppath =
            cypher_ast_projection_get_expression(projection);
    ck_assert_int_eq(cypher_astnode_type(ppath), CYPHER_AST_PATTERN_PATH);

    ck_assert_int_eq(cypher_ast_pattern_path_nelements(ppath), 3);
    ck_assert_ptr_eq(cypher_ast_pattern_path_get_element(ppath, 3), NULL);
    const cypher_astnode_t *node = cypher_ast_pattern_path_get_element(ppath, 0);
    ck_assert_int_eq(cypher_astnode_type(node), CYPHER_AST_NODE_PATTERN);

    const cypher_astnode_t *id = cypher_ast_node_pattern_get_identifier(node);
    ck_assert_int_eq(cypher_astnode_type(id), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(id), "n");

    ck_assert_int_eq(cypher_ast_node_pattern_nlabels(node), 0);
    ck_assert_ptr_eq(cypher_ast_node_pattern_get_label(node, 0), NULL);
    ck_assert_ptr_eq(cypher_ast_node_pattern_get_properties(node), NULL);

    const cypher_astnode_t *rel = cypher_ast_pattern_path_get_element(ppath, 1);
    ck_assert_int_eq(cypher_astnode_type(rel), CYPHER_AST_REL_PATTERN);

    ck_assert_int_eq(cypher_ast_rel_pattern_get_direction(rel), CYPHER_REL_OUTBOUND);

    ck_assert_ptr_eq(cypher_ast_rel_pattern_get_identifier(rel), NULL);

    ck_assert_int_eq(cypher_ast_rel_pattern_nreltypes(rel), 1);
    const cypher_astnode_t *reltype = cypher_ast_rel_pattern_get_reltype(rel, 0);
    ck_assert_int_eq(cypher_astnode_type(reltype), CYPHER_AST_RELTYPE);
    ck_assert_str_eq(cypher_ast_reltype_get_name(reltype), "Foo");
    ck_assert_ptr_eq(cypher_ast_rel_pattern_get_reltype(rel, 2), NULL);

    ck_assert_ptr_eq(cypher_ast_rel_pattern_get_varlength(rel), NULL);

    const cypher_astnode_t *props = cypher_ast_rel_pattern_get_properties(rel);
    ck_assert_int_eq(cypher_astnode_type(props), CYPHER_AST_MAP);

    ck_assert_int_eq(cypher_ast_map_nentries(props), 2);
    ck_assert_ptr_eq(cypher_ast_map_get_key(props, 2), NULL);
    ck_assert_ptr_eq(cypher_ast_map_get_value(props, 2), NULL);

    const cypher_astnode_t *key = cypher_ast_map_get_key(props, 0);
    ck_assert_int_eq(cypher_astnode_type(key), CYPHER_AST_PROP_NAME);
    ck_assert_str_eq(cypher_ast_prop_name_get_value(key), "start");
    const cypher_astnode_t *value = cypher_ast_map_get_value(props, 0);
    ck_assert_int_eq(cypher_astnode_type(value), CYPHER_AST_INTEGER);
    ck_assert_str_eq(cypher_ast_integer_get_valuestr(value), "1999");

    key = cypher_ast_map_get_key(props, 1);
    ck_assert_int_eq(cypher_astnode_type(key), CYPHER_AST_PROP_NAME);
    ck_assert_str_eq(cypher_ast_prop_name_get_value(key), "end");
    value = cypher_ast_map_get_value(props, 1);
    ck_assert_int_eq(cypher_astnode_type(value), CYPHER_AST_INTEGER);
    ck_assert_str_eq(cypher_ast_integer_get_valuestr(value), "2000");

    node = cypher_ast_pattern_path_get_element(ppath, 2);
    ck_assert_int_eq(cypher_astnode_type(node), CYPHER_AST_NODE_PATTERN);
}
END_TEST


START_TEST (parse_rel_with_param_props)
{
    struct cypher_input_position last = cypher_input_position_zero;
    result = cypher_parse("RETURN (n)-[:Foo {param}]->(m) AS p;",
            &last, NULL, 0);
    ck_assert_ptr_ne(result, NULL);
    ck_assert_int_eq(last.offset, 36);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
" @0   0..36  statement               body=@1\n"
" @1   0..36  > query                 clauses=[@2]\n"
" @2   0..35  > > RETURN              projections=[@3]\n"
" @3   7..35  > > > projection        expression=@4, alias=@12\n"
" @4   7..30  > > > > pattern path    (@5)-[@7]-(@10)\n"
" @5   7..10  > > > > > node pattern  (@6)\n"
" @6   8..9   > > > > > > identifier  `n`\n"
" @7  10..27  > > > > > rel pattern   -[:@8 {@9}]->\n"
" @8  12..16  > > > > > > rel type    :`Foo`\n"
" @9  17..24  > > > > > > parameter   $`param`\n"
"@10  27..30  > > > > > node pattern  (@11)\n"
"@11  28..29  > > > > > > identifier  `m`\n"
"@12  34..35  > > > > identifier      `p`\n";
    ck_assert_str_eq(memstream_buffer, expected);

    const cypher_astnode_t *ast = cypher_parse_result_get_directive(result, 0);
    const cypher_astnode_t *query = cypher_ast_statement_get_body(ast);
    const cypher_astnode_t *clause = cypher_ast_query_get_clause(query, 0);
    ck_assert_int_eq(cypher_astnode_type(clause), CYPHER_AST_RETURN);

    const cypher_astnode_t *projection = cypher_ast_return_get_projection(clause, 0);
    ck_assert_int_eq(cypher_astnode_type(projection), CYPHER_AST_PROJECTION);
    const cypher_astnode_t *ppath =
            cypher_ast_projection_get_expression(projection);
    ck_assert_int_eq(cypher_astnode_type(ppath), CYPHER_AST_PATTERN_PATH);

    ck_assert_int_eq(cypher_ast_pattern_path_nelements(ppath), 3);
    ck_assert_ptr_eq(cypher_ast_pattern_path_get_element(ppath, 3), NULL);
    const cypher_astnode_t *node = cypher_ast_pattern_path_get_element(ppath, 0);
    ck_assert_int_eq(cypher_astnode_type(node), CYPHER_AST_NODE_PATTERN);

    const cypher_astnode_t *id = cypher_ast_node_pattern_get_identifier(node);
    ck_assert_int_eq(cypher_astnode_type(id), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(id), "n");

    ck_assert_int_eq(cypher_ast_node_pattern_nlabels(node), 0);
    ck_assert_ptr_eq(cypher_ast_node_pattern_get_label(node, 0), NULL);
    ck_assert_ptr_eq(cypher_ast_node_pattern_get_properties(node), NULL);

    const cypher_astnode_t *rel = cypher_ast_pattern_path_get_element(ppath, 1);
    ck_assert_int_eq(cypher_astnode_type(rel), CYPHER_AST_REL_PATTERN);

    ck_assert_int_eq(cypher_ast_rel_pattern_get_direction(rel), CYPHER_REL_OUTBOUND);

    ck_assert_ptr_eq(cypher_ast_rel_pattern_get_identifier(rel), NULL);

    ck_assert_int_eq(cypher_ast_rel_pattern_nreltypes(rel), 1);
    const cypher_astnode_t *reltype = cypher_ast_rel_pattern_get_reltype(rel, 0);
    ck_assert_int_eq(cypher_astnode_type(reltype), CYPHER_AST_RELTYPE);
    ck_assert_str_eq(cypher_ast_reltype_get_name(reltype), "Foo");
    ck_assert_ptr_eq(cypher_ast_rel_pattern_get_reltype(rel, 2), NULL);

    ck_assert_ptr_eq(cypher_ast_rel_pattern_get_varlength(rel), NULL);

    const cypher_astnode_t *param = cypher_ast_rel_pattern_get_properties(rel);
    ck_assert_int_eq(cypher_astnode_type(param), CYPHER_AST_PARAMETER);
    ck_assert_str_eq(cypher_ast_parameter_get_name(param), "param");

    node = cypher_ast_pattern_path_get_element(ppath, 2);
    ck_assert_int_eq(cypher_astnode_type(node), CYPHER_AST_NODE_PATTERN);
}
END_TEST


START_TEST (parse_named_path)
{
    struct cypher_input_position last = cypher_input_position_zero;
    result = cypher_parse("MATCH p = (n)-[:Foo]->(m) RETURN p;",
            &last, NULL, 0);
    ck_assert_ptr_ne(result, NULL);
    ck_assert_int_eq(last.offset, 35);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
" @0   0..35  statement                 body=@1\n"
" @1   0..35  > query                   clauses=[@2, @13]\n"
" @2   0..26  > > MATCH                 pattern=@3\n"
" @3   6..25  > > > pattern             paths=[@4]\n"
" @4   6..25  > > > > named path        @5 = @6\n"
" @5   6..7   > > > > > identifier      `p`\n"
" @6  10..25  > > > > > pattern path    (@7)-[@9]-(@11)\n"
" @7  10..13  > > > > > > node pattern  (@8)\n"
" @8  11..12  > > > > > > > identifier  `n`\n"
" @9  13..22  > > > > > > rel pattern   -[:@10]->\n"
"@10  15..19  > > > > > > > rel type    :`Foo`\n"
"@11  22..25  > > > > > > node pattern  (@12)\n"
"@12  23..24  > > > > > > > identifier  `m`\n"
"@13  26..34  > > RETURN                projections=[@14]\n"
"@14  33..34  > > > projection          expression=@15\n"
"@15  33..34  > > > > identifier        `p`\n";
    ck_assert_str_eq(memstream_buffer, expected);

    const cypher_astnode_t *ast = cypher_parse_result_get_directive(result, 0);
    const cypher_astnode_t *query = cypher_ast_statement_get_body(ast);
    const cypher_astnode_t *match = cypher_ast_query_get_clause(query, 0);

    const cypher_astnode_t *pattern = cypher_ast_match_get_pattern(match);
    ck_assert_int_eq(cypher_astnode_type(pattern), CYPHER_AST_PATTERN);

    ck_assert_int_eq(cypher_ast_pattern_npaths(pattern), 1);
    ck_assert_ptr_eq(cypher_ast_pattern_get_path(pattern, 1), NULL);
    const cypher_astnode_t *path = cypher_ast_pattern_get_path(pattern, 0);
    ck_assert_int_eq(cypher_astnode_type(path), CYPHER_AST_NAMED_PATH);

    const cypher_astnode_t *id = cypher_ast_named_path_get_identifier(path);
    ck_assert_int_eq(cypher_astnode_type(id), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(id), "p");

    ck_assert_int_eq(cypher_ast_pattern_path_nelements(path), 3);
    ck_assert_ptr_eq(cypher_ast_pattern_path_get_element(path, 3), NULL);
    const cypher_astnode_t *node = cypher_ast_pattern_path_get_element(path, 0);
    ck_assert_int_eq(cypher_astnode_type(node), CYPHER_AST_NODE_PATTERN);

    const cypher_astnode_t *unnamed_path = cypher_ast_named_path_get_path(path);
    ck_assert_int_eq(cypher_astnode_type(unnamed_path), CYPHER_AST_PATTERN_PATH);
    ck_assert_int_eq(cypher_ast_pattern_path_nelements(unnamed_path), 3);
    ck_assert_ptr_eq(cypher_ast_pattern_path_get_element(unnamed_path, 0), node);

    id = cypher_ast_node_pattern_get_identifier(node);
    ck_assert_int_eq(cypher_astnode_type(id), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(id), "n");

    ck_assert_int_eq(cypher_ast_node_pattern_nlabels(node), 0);
    ck_assert_ptr_eq(cypher_ast_node_pattern_get_label(node, 0), NULL);
    ck_assert_ptr_eq(cypher_ast_node_pattern_get_properties(node), NULL);

    const cypher_astnode_t *rel = cypher_ast_pattern_path_get_element(path, 1);
    ck_assert_int_eq(cypher_astnode_type(rel), CYPHER_AST_REL_PATTERN);

    ck_assert_int_eq(cypher_ast_rel_pattern_get_direction(rel), CYPHER_REL_OUTBOUND);

    ck_assert_ptr_eq(cypher_ast_rel_pattern_get_identifier(rel), NULL);

    ck_assert_int_eq(cypher_ast_rel_pattern_nreltypes(rel), 1);
    const cypher_astnode_t *reltype = cypher_ast_rel_pattern_get_reltype(rel, 0);
    ck_assert_int_eq(cypher_astnode_type(reltype), CYPHER_AST_RELTYPE);
    ck_assert_str_eq(cypher_ast_reltype_get_name(reltype), "Foo");
    ck_assert_ptr_eq(cypher_ast_rel_pattern_get_reltype(rel, 2), NULL);

    ck_assert_ptr_eq(cypher_ast_rel_pattern_get_varlength(rel), NULL);

    node = cypher_ast_pattern_path_get_element(path, 2);
    ck_assert_int_eq(cypher_astnode_type(node), CYPHER_AST_NODE_PATTERN);
}
END_TEST


START_TEST (parse_shortest_path)
{
    struct cypher_input_position last = cypher_input_position_zero;
    result = cypher_parse("MATCH p = shortestPath((n)-[:Foo]->(m)) RETURN n;",
            &last, NULL, 0);
    ck_assert_ptr_ne(result, NULL);
    ck_assert_int_eq(last.offset, 49);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
" @0   0..49  statement                   body=@1\n"
" @1   0..49  > query                     clauses=[@2, @14]\n"
" @2   0..40  > > MATCH                   pattern=@3\n"
" @3   6..39  > > > pattern               paths=[@4]\n"
" @4   6..39  > > > > named path          @5 = @6\n"
" @5   6..7   > > > > > identifier        `p`\n"
" @6  10..39  > > > > > shortestPath      single=true, path=@7\n"
" @7  23..38  > > > > > > pattern path    (@8)-[@10]-(@12)\n"
" @8  23..26  > > > > > > > node pattern  (@9)\n"
" @9  24..25  > > > > > > > > identifier  `n`\n"
"@10  26..35  > > > > > > > rel pattern   -[:@11]->\n"
"@11  28..32  > > > > > > > > rel type    :`Foo`\n"
"@12  35..38  > > > > > > > node pattern  (@13)\n"
"@13  36..37  > > > > > > > > identifier  `m`\n"
"@14  40..48  > > RETURN                  projections=[@15]\n"
"@15  47..48  > > > projection            expression=@16\n"
"@16  47..48  > > > > identifier          `n`\n";
    ck_assert_str_eq(memstream_buffer, expected);

    const cypher_astnode_t *ast = cypher_parse_result_get_directive(result, 0);
    const cypher_astnode_t *query = cypher_ast_statement_get_body(ast);
    const cypher_astnode_t *match = cypher_ast_query_get_clause(query, 0);

    const cypher_astnode_t *pattern = cypher_ast_match_get_pattern(match);
    ck_assert_int_eq(cypher_astnode_type(pattern), CYPHER_AST_PATTERN);

    ck_assert_int_eq(cypher_ast_pattern_npaths(pattern), 1);
    ck_assert_ptr_eq(cypher_ast_pattern_get_path(pattern, 1), NULL);
    const cypher_astnode_t *npath = cypher_ast_pattern_get_path(pattern, 0);
    ck_assert_int_eq(cypher_astnode_type(npath), CYPHER_AST_NAMED_PATH);

    const cypher_astnode_t *id = cypher_ast_named_path_get_identifier(npath);
    ck_assert_int_eq(cypher_astnode_type(id), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(id), "p");

    ck_assert_int_eq(cypher_ast_pattern_path_nelements(npath), 3);
    ck_assert_ptr_eq(cypher_ast_pattern_path_get_element(npath, 3), NULL);
    const cypher_astnode_t *node = cypher_ast_pattern_path_get_element(npath, 0);
    ck_assert_int_eq(cypher_astnode_type(node), CYPHER_AST_NODE_PATTERN);

    const cypher_astnode_t *spath = cypher_ast_named_path_get_path(npath);
    ck_assert_int_eq(cypher_astnode_type(spath), CYPHER_AST_SHORTEST_PATH);
    ck_assert_int_eq(cypher_ast_pattern_path_nelements(spath), 3);
    ck_assert_ptr_eq(cypher_ast_pattern_path_get_element(spath, 0), node);

    ck_assert(cypher_ast_shortest_path_is_single(spath));

    const cypher_astnode_t *unnamed_path = cypher_ast_shortest_path_get_path(spath);
    ck_assert_int_eq(cypher_astnode_type(unnamed_path), CYPHER_AST_PATTERN_PATH);
    ck_assert_int_eq(cypher_ast_pattern_path_nelements(unnamed_path), 3);
    ck_assert_ptr_eq(cypher_ast_pattern_path_get_element(unnamed_path, 0), node);
}
END_TEST


START_TEST (parse_all_shortest_paths)
{
    struct cypher_input_position last = cypher_input_position_zero;
    result = cypher_parse("RETURN allShortestPaths((n)-[:Foo]->(m)) AS p;",
            &last, NULL, 0);
    ck_assert_ptr_ne(result, NULL);
    ck_assert_int_eq(last.offset, 46);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
" @0   0..46  statement                 body=@1\n"
" @1   0..46  > query                   clauses=[@2]\n"
" @2   0..45  > > RETURN                projections=[@3]\n"
" @3   7..45  > > > projection          expression=@4, alias=@12\n"
" @4   7..40  > > > > shortestPath      single=false, path=@5\n"
" @5  24..39  > > > > > pattern path    (@6)-[@8]-(@10)\n"
" @6  24..27  > > > > > > node pattern  (@7)\n"
" @7  25..26  > > > > > > > identifier  `n`\n"
" @8  27..36  > > > > > > rel pattern   -[:@9]->\n"
" @9  29..33  > > > > > > > rel type    :`Foo`\n"
"@10  36..39  > > > > > > node pattern  (@11)\n"
"@11  37..38  > > > > > > > identifier  `m`\n"
"@12  44..45  > > > > identifier        `p`\n";
    ck_assert_str_eq(memstream_buffer, expected);

    const cypher_astnode_t *ast = cypher_parse_result_get_directive(result, 0);
    const cypher_astnode_t *query = cypher_ast_statement_get_body(ast);
    const cypher_astnode_t *clause = cypher_ast_query_get_clause(query, 0);
    ck_assert_int_eq(cypher_astnode_type(clause), CYPHER_AST_RETURN);

    const cypher_astnode_t *projection = cypher_ast_return_get_projection(clause, 0);
    ck_assert_int_eq(cypher_astnode_type(projection), CYPHER_AST_PROJECTION);
    const cypher_astnode_t *spath =
            cypher_ast_projection_get_expression(projection);
    ck_assert_int_eq(cypher_astnode_type(spath), CYPHER_AST_SHORTEST_PATH);

    ck_assert_int_eq(cypher_ast_pattern_path_nelements(spath), 3);
    ck_assert_ptr_eq(cypher_ast_pattern_path_get_element(spath, 3), NULL);
    const cypher_astnode_t *node = cypher_ast_pattern_path_get_element(spath, 0);
    ck_assert_int_eq(cypher_astnode_type(node), CYPHER_AST_NODE_PATTERN);

    ck_assert(!cypher_ast_shortest_path_is_single(spath));

    const cypher_astnode_t *unnamed_path = cypher_ast_shortest_path_get_path(spath);
    ck_assert_int_eq(cypher_astnode_type(unnamed_path), CYPHER_AST_PATTERN_PATH);
    ck_assert_int_eq(cypher_ast_pattern_path_nelements(unnamed_path), 3);
    ck_assert_ptr_eq(cypher_ast_pattern_path_get_element(unnamed_path, 0), node);
}
END_TEST


TCase* pattern_tcase(void)
{
    TCase *tc = tcase_create("pattern");
    tcase_add_checked_fixture(tc, setup, teardown);
    tcase_add_test(tc, parse_single_node);
    tcase_add_test(tc, parse_labeled_node);
    tcase_add_test(tc, parse_node_with_map_props);
    tcase_add_test(tc, parse_node_with_param_props);
    tcase_add_test(tc, parse_multiple_labeled_node);
    tcase_add_test(tc, parse_single_rel);
    tcase_add_test(tc, parse_varlength_rel);
    tcase_add_test(tc, parse_varlength_rel_with_bounded_start);
    tcase_add_test(tc, parse_varlength_rel_with_bounded_end);
    tcase_add_test(tc, parse_varlength_rel_with_fixed_range);
    tcase_add_test(tc, parse_rel_with_map_props);
    tcase_add_test(tc, parse_rel_with_param_props);
    tcase_add_test(tc, parse_named_path);
    tcase_add_test(tc, parse_shortest_path);
    tcase_add_test(tc, parse_all_shortest_paths);
    return tc;
}
