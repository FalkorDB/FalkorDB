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


START_TEST (parse_remove_label)
{
    struct cypher_input_position last = cypher_input_position_zero;
    result = cypher_parse("/*MATCH*/ REMOVE n:Foo, m:Bar:Baz", &last, NULL, 0);
    ck_assert_ptr_ne(result, NULL);
    ck_assert_int_eq(last.offset, 33);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
" @0   2..7   block_comment        /*MATCH*/\n"
" @1  10..33  statement            body=@2\n"
" @2  10..33  > query              clauses=[@3]\n"
" @3  10..33  > > REMOVE           items=[@4, @7]\n"
" @4  17..22  > > > remove labels  @5:@6\n"
" @5  17..18  > > > > identifier   `n`\n"
" @6  18..22  > > > > label        :`Foo`\n"
" @7  24..33  > > > remove labels  @8:@9:@10\n"
" @8  24..25  > > > > identifier   `m`\n"
" @9  25..29  > > > > label        :`Bar`\n"
"@10  29..33  > > > > label        :`Baz`\n";
    ck_assert_str_eq(memstream_buffer, expected);

    const cypher_astnode_t *ast = cypher_parse_result_get_directive(result, 0);
    ck_assert_int_eq(cypher_astnode_type(ast), CYPHER_AST_STATEMENT);
    const cypher_astnode_t *query = cypher_ast_statement_get_body(ast);
    ck_assert_int_eq(cypher_astnode_type(query), CYPHER_AST_QUERY);
    const cypher_astnode_t *rem = cypher_ast_query_get_clause(query, 0);
    ck_assert_int_eq(cypher_astnode_type(rem), CYPHER_AST_REMOVE);

    ck_assert_int_eq(cypher_ast_remove_nitems(rem), 2);
    ck_assert_ptr_eq(cypher_ast_remove_get_item(rem, 2), NULL);

    const cypher_astnode_t *item = cypher_ast_remove_get_item(rem, 0);
    ck_assert(cypher_astnode_instanceof(item, CYPHER_AST_REMOVE_ITEM));
    ck_assert_int_eq(cypher_astnode_type(item), CYPHER_AST_REMOVE_LABELS);

    const cypher_astnode_t *id = cypher_ast_remove_labels_get_identifier(item);
    ck_assert_int_eq(cypher_astnode_type(id), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(id), "n");

    ck_assert_int_eq(cypher_ast_remove_labels_nlabels(item), 1);
    ck_assert_ptr_eq(cypher_ast_remove_labels_get_label(item, 1), NULL);

    const cypher_astnode_t *label = cypher_ast_remove_labels_get_label(item, 0);
    ck_assert_int_eq(cypher_astnode_type(label), CYPHER_AST_LABEL);
    ck_assert_str_eq(cypher_ast_label_get_name(label), "Foo");

    item = cypher_ast_remove_get_item(rem, 1);
    ck_assert(cypher_astnode_instanceof(item, CYPHER_AST_REMOVE_ITEM));
    ck_assert_int_eq(cypher_astnode_type(item), CYPHER_AST_REMOVE_LABELS);

    id = cypher_ast_remove_labels_get_identifier(item);
    ck_assert_int_eq(cypher_astnode_type(id), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(id), "m");

    ck_assert_int_eq(cypher_ast_remove_labels_nlabels(item), 2);
    ck_assert_ptr_eq(cypher_ast_remove_labels_get_label(item, 2), NULL);

    label = cypher_ast_remove_labels_get_label(item, 0);
    ck_assert_int_eq(cypher_astnode_type(label), CYPHER_AST_LABEL);
    ck_assert_str_eq(cypher_ast_label_get_name(label), "Bar");

    label = cypher_ast_remove_labels_get_label(item, 1);
    ck_assert_int_eq(cypher_astnode_type(label), CYPHER_AST_LABEL);
    ck_assert_str_eq(cypher_ast_label_get_name(label), "Baz");
}
END_TEST


START_TEST (parse_remove_property)
{
    struct cypher_input_position last = cypher_input_position_zero;
    result = cypher_parse("/*MATCH*/ REMOVE n.foo, m.bar.baz", &last, NULL, 0);
    ck_assert_ptr_ne(result, NULL);
    ck_assert_int_eq(last.offset, 33);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
" @0   2..7   block_comment           /*MATCH*/\n"
" @1  10..33  statement               body=@2\n"
" @2  10..33  > query                 clauses=[@3]\n"
" @3  10..33  > > REMOVE              items=[@4, @8]\n"
" @4  17..22  > > > remove property   prop=@5\n"
" @5  17..22  > > > > property        @6.@7\n"
" @6  17..18  > > > > > identifier    `n`\n"
" @7  19..22  > > > > > prop name     `foo`\n"
" @8  24..33  > > > remove property   prop=@9\n"
" @9  24..33  > > > > property        @10.@13\n"
"@10  24..29  > > > > > property      @11.@12\n"
"@11  24..25  > > > > > > identifier  `m`\n"
"@12  26..29  > > > > > > prop name   `bar`\n"
"@13  30..33  > > > > > prop name     `baz`\n";
    ck_assert_str_eq(memstream_buffer, expected);

    const cypher_astnode_t *ast = cypher_parse_result_get_directive(result, 0);
    ck_assert_int_eq(cypher_astnode_type(ast), CYPHER_AST_STATEMENT);
    const cypher_astnode_t *query = cypher_ast_statement_get_body(ast);
    ck_assert_int_eq(cypher_astnode_type(query), CYPHER_AST_QUERY);
    const cypher_astnode_t *rem = cypher_ast_query_get_clause(query, 0);
    ck_assert_int_eq(cypher_astnode_type(rem), CYPHER_AST_REMOVE);

    ck_assert_int_eq(cypher_ast_remove_nitems(rem), 2);
    ck_assert_ptr_eq(cypher_ast_remove_get_item(rem, 2), NULL);

    const cypher_astnode_t *item = cypher_ast_remove_get_item(rem, 0);
    ck_assert(cypher_astnode_instanceof(item, CYPHER_AST_REMOVE_ITEM));
    ck_assert_int_eq(cypher_astnode_type(item), CYPHER_AST_REMOVE_PROPERTY);

    const cypher_astnode_t *prop = cypher_ast_remove_property_get_property(item);
    ck_assert_int_eq(cypher_astnode_type(prop), CYPHER_AST_PROPERTY_OPERATOR);
    const cypher_astnode_t *id = cypher_ast_property_operator_get_expression(prop);
    ck_assert_int_eq(cypher_astnode_type(id), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(id), "n");
    const cypher_astnode_t *propname = cypher_ast_property_operator_get_prop_name(prop);
    ck_assert_int_eq(cypher_astnode_type(propname), CYPHER_AST_PROP_NAME);
    ck_assert_str_eq(cypher_ast_prop_name_get_value(propname), "foo");

    item = cypher_ast_remove_get_item(rem, 1);
    ck_assert(cypher_astnode_instanceof(item, CYPHER_AST_REMOVE_ITEM));
    ck_assert_int_eq(cypher_astnode_type(item), CYPHER_AST_REMOVE_PROPERTY);

    prop = cypher_ast_remove_property_get_property(item);
    ck_assert_int_eq(cypher_astnode_type(prop), CYPHER_AST_PROPERTY_OPERATOR);

    const cypher_astnode_t *expr = cypher_ast_property_operator_get_expression(prop);
    ck_assert_int_eq(cypher_astnode_type(expr), CYPHER_AST_PROPERTY_OPERATOR);

    id = cypher_ast_property_operator_get_expression(expr);
    ck_assert_int_eq(cypher_astnode_type(id), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(id), "m");

    propname = cypher_ast_property_operator_get_prop_name(expr);
    ck_assert_int_eq(cypher_astnode_type(propname), CYPHER_AST_PROP_NAME);
    ck_assert_str_eq(cypher_ast_prop_name_get_value(propname), "bar");

    propname = cypher_ast_property_operator_get_prop_name(prop);
    ck_assert_int_eq(cypher_astnode_type(propname), CYPHER_AST_PROP_NAME);
    ck_assert_str_eq(cypher_ast_prop_name_get_value(propname), "baz");
}
END_TEST


TCase* remove_tcase(void)
{
    TCase *tc = tcase_create("remove");
    tcase_add_checked_fixture(tc, setup, teardown);
    tcase_add_test(tc, parse_remove_label);
    tcase_add_test(tc, parse_remove_property);
    return tc;
}
