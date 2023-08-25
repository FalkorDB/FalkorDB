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


START_TEST (parse_set_property)
{
    struct cypher_input_position last = cypher_input_position_zero;
    result = cypher_parse("/*MATCH*/ SET n.foo = bar;", &last, NULL, 0);
    ck_assert_ptr_ne(result, NULL);
    ck_assert_int_eq(last.offset, 26);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
"@0   2..7   block_comment         /*MATCH*/\n"
"@1  10..26  statement             body=@2\n"
"@2  10..26  > query               clauses=[@3]\n"
"@3  10..25  > > SET               items=[@4]\n"
"@4  14..25  > > > set property    @5 = @8\n"
"@5  14..20  > > > > property      @6.@7\n"
"@6  14..15  > > > > > identifier  `n`\n"
"@7  16..19  > > > > > prop name   `foo`\n"
"@8  22..25  > > > > identifier    `bar`\n";
    ck_assert_str_eq(memstream_buffer, expected);

    const cypher_astnode_t *ast = cypher_parse_result_get_directive(result, 0);
    ck_assert_int_eq(cypher_astnode_type(ast), CYPHER_AST_STATEMENT);
    const cypher_astnode_t *query = cypher_ast_statement_get_body(ast);
    ck_assert_int_eq(cypher_astnode_type(query), CYPHER_AST_QUERY);
    const cypher_astnode_t *set = cypher_ast_query_get_clause(query, 0);
    ck_assert_int_eq(cypher_astnode_type(set), CYPHER_AST_SET);

    ck_assert_int_eq(cypher_ast_set_nitems(set), 1);
    ck_assert_ptr_eq(cypher_ast_set_get_item(set, 1), NULL);

    const cypher_astnode_t *item = cypher_ast_set_get_item(set, 0);
    ck_assert_int_eq(cypher_astnode_type(item), CYPHER_AST_SET_PROPERTY);

    const cypher_astnode_t *prop = cypher_ast_set_property_get_property(item);
    ck_assert_int_eq(cypher_astnode_type(prop), CYPHER_AST_PROPERTY_OPERATOR);
    const cypher_astnode_t *id = cypher_ast_property_operator_get_expression(prop);
    ck_assert_int_eq(cypher_astnode_type(id), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(id), "n");
    const cypher_astnode_t *propname =
            cypher_ast_property_operator_get_prop_name(prop);
    ck_assert_int_eq(cypher_astnode_type(propname), CYPHER_AST_PROP_NAME);
    ck_assert_str_eq(cypher_ast_prop_name_get_value(propname), "foo");

    const cypher_astnode_t *expr = cypher_ast_set_property_get_expression(item);
    ck_assert_int_eq(cypher_astnode_type(expr), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(expr), "bar");
}
END_TEST


START_TEST (parse_set_all_properties)
{
    struct cypher_input_position last = cypher_input_position_zero;
    result = cypher_parse("/*MATCH*/ SET n = {foo: bar};", &last, NULL, 0);
    ck_assert_ptr_ne(result, NULL);
    ck_assert_int_eq(last.offset, 29);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
"@0   2..7   block_comment             /*MATCH*/\n"
"@1  10..29  statement                 body=@2\n"
"@2  10..29  > query                   clauses=[@3]\n"
"@3  10..28  > > SET                   items=[@4]\n"
"@4  14..28  > > > set all properties  @5 = @6\n"
"@5  14..15  > > > > identifier        `n`\n"
"@6  18..28  > > > > map               {@7:@8}\n"
"@7  19..22  > > > > > prop name       `foo`\n"
"@8  24..27  > > > > > identifier      `bar`\n";
    ck_assert_str_eq(memstream_buffer, expected);

    const cypher_astnode_t *ast = cypher_parse_result_get_directive(result, 0);
    ck_assert_int_eq(cypher_astnode_type(ast), CYPHER_AST_STATEMENT);
    const cypher_astnode_t *query = cypher_ast_statement_get_body(ast);
    ck_assert_int_eq(cypher_astnode_type(query), CYPHER_AST_QUERY);
    const cypher_astnode_t *set = cypher_ast_query_get_clause(query, 0);
    ck_assert_int_eq(cypher_astnode_type(set), CYPHER_AST_SET);

    ck_assert_int_eq(cypher_ast_set_nitems(set), 1);
    ck_assert_ptr_eq(cypher_ast_set_get_item(set, 1), NULL);

    const cypher_astnode_t *item = cypher_ast_set_get_item(set, 0);
    ck_assert_int_eq(cypher_astnode_type(item), CYPHER_AST_SET_ALL_PROPERTIES);

    const cypher_astnode_t *id = cypher_ast_set_all_properties_get_identifier(item);
    ck_assert_int_eq(cypher_astnode_type(id), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(id), "n");

    const cypher_astnode_t *expr = cypher_ast_set_all_properties_get_expression(item);
    ck_assert_int_eq(cypher_astnode_type(expr), CYPHER_AST_MAP);
}
END_TEST


START_TEST (parse_set_nested_property)
{
    struct cypher_input_position last = cypher_input_position_zero;
    result = cypher_parse("/*MATCH*/ SET n.foo.bar = baz;", &last, NULL, 0);
    ck_assert_ptr_ne(result, NULL);
    ck_assert_int_eq(last.offset, 30);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
" @0   2..7   block_comment           /*MATCH*/\n"
" @1  10..30  statement               body=@2\n"
" @2  10..30  > query                 clauses=[@3]\n"
" @3  10..29  > > SET                 items=[@4]\n"
" @4  14..29  > > > set property      @5 = @10\n"
" @5  14..24  > > > > property        @6.@9\n"
" @6  14..19  > > > > > property      @7.@8\n"
" @7  14..15  > > > > > > identifier  `n`\n"
" @8  16..19  > > > > > > prop name   `foo`\n"
" @9  20..23  > > > > > prop name     `bar`\n"
"@10  26..29  > > > > identifier      `baz`\n";
    ck_assert_str_eq(memstream_buffer, expected);

    const cypher_astnode_t *ast = cypher_parse_result_get_directive(result, 0);
    ck_assert_int_eq(cypher_astnode_type(ast), CYPHER_AST_STATEMENT);
    const cypher_astnode_t *query = cypher_ast_statement_get_body(ast);
    ck_assert_int_eq(cypher_astnode_type(query), CYPHER_AST_QUERY);
    const cypher_astnode_t *set = cypher_ast_query_get_clause(query, 0);
    ck_assert_int_eq(cypher_astnode_type(set), CYPHER_AST_SET);

    ck_assert_int_eq(cypher_ast_set_nitems(set), 1);
    ck_assert_ptr_eq(cypher_ast_set_get_item(set, 1), NULL);

    const cypher_astnode_t *item = cypher_ast_set_get_item(set, 0);
    ck_assert_int_eq(cypher_astnode_type(item), CYPHER_AST_SET_PROPERTY);

    const cypher_astnode_t *prop = cypher_ast_set_property_get_property(item);
    ck_assert_int_eq(cypher_astnode_type(prop), CYPHER_AST_PROPERTY_OPERATOR);

    const cypher_astnode_t *exp = cypher_ast_property_operator_get_expression(prop);
    ck_assert_int_eq(cypher_astnode_type(exp), CYPHER_AST_PROPERTY_OPERATOR);

    const cypher_astnode_t *id = cypher_ast_property_operator_get_expression(exp);
    ck_assert_int_eq(cypher_astnode_type(id), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(id), "n");
    const cypher_astnode_t *propname =
            cypher_ast_property_operator_get_prop_name(exp);
    ck_assert_int_eq(cypher_astnode_type(propname), CYPHER_AST_PROP_NAME);
    ck_assert_str_eq(cypher_ast_prop_name_get_value(propname), "foo");

    propname = cypher_ast_property_operator_get_prop_name(prop);
    ck_assert_int_eq(cypher_astnode_type(propname), CYPHER_AST_PROP_NAME);
    ck_assert_str_eq(cypher_ast_prop_name_get_value(propname), "bar");

    const cypher_astnode_t *expr = cypher_ast_set_property_get_expression(item);
    ck_assert_int_eq(cypher_astnode_type(expr), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(expr), "baz");
}
END_TEST


START_TEST (parse_set_expression_property)
{
    struct cypher_input_position last = cypher_input_position_zero;
    result = cypher_parse("/*MATCH*/ SET (n.foo).bar = baz;", &last, NULL, 0);
    ck_assert_ptr_ne(result, NULL);
    ck_assert_int_eq(last.offset, 32);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
" @0   2..7   block_comment           /*MATCH*/\n"
" @1  10..32  statement               body=@2\n"
" @2  10..32  > query                 clauses=[@3]\n"
" @3  10..31  > > SET                 items=[@4]\n"
" @4  14..31  > > > set property      @5 = @10\n"
" @5  14..26  > > > > property        @6.@9\n"
" @6  15..20  > > > > > property      @7.@8\n"
" @7  15..16  > > > > > > identifier  `n`\n"
" @8  17..20  > > > > > > prop name   `foo`\n"
" @9  22..25  > > > > > prop name     `bar`\n"
"@10  28..31  > > > > identifier      `baz`\n";
    ck_assert_str_eq(memstream_buffer, expected);

    const cypher_astnode_t *ast = cypher_parse_result_get_directive(result, 0);
    ck_assert_int_eq(cypher_astnode_type(ast), CYPHER_AST_STATEMENT);
    const cypher_astnode_t *query = cypher_ast_statement_get_body(ast);
    ck_assert_int_eq(cypher_astnode_type(query), CYPHER_AST_QUERY);
    const cypher_astnode_t *set = cypher_ast_query_get_clause(query, 0);
    ck_assert_int_eq(cypher_astnode_type(set), CYPHER_AST_SET);

    ck_assert_int_eq(cypher_ast_set_nitems(set), 1);
    ck_assert_ptr_eq(cypher_ast_set_get_item(set, 1), NULL);

    const cypher_astnode_t *item = cypher_ast_set_get_item(set, 0);
    ck_assert_int_eq(cypher_astnode_type(item), CYPHER_AST_SET_PROPERTY);

    const cypher_astnode_t *prop = cypher_ast_set_property_get_property(item);
    ck_assert_int_eq(cypher_astnode_type(prop), CYPHER_AST_PROPERTY_OPERATOR);

    const cypher_astnode_t *exp = cypher_ast_property_operator_get_expression(prop);
    ck_assert_int_eq(cypher_astnode_type(exp), CYPHER_AST_PROPERTY_OPERATOR);

    const cypher_astnode_t *id = cypher_ast_property_operator_get_expression(exp);
    ck_assert_int_eq(cypher_astnode_type(id), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(id), "n");
    const cypher_astnode_t *propname =
            cypher_ast_property_operator_get_prop_name(exp);
    ck_assert_int_eq(cypher_astnode_type(propname), CYPHER_AST_PROP_NAME);
    ck_assert_str_eq(cypher_ast_prop_name_get_value(propname), "foo");

    propname = cypher_ast_property_operator_get_prop_name(prop);
    ck_assert_int_eq(cypher_astnode_type(propname), CYPHER_AST_PROP_NAME);
    ck_assert_str_eq(cypher_ast_prop_name_get_value(propname), "bar");

    const cypher_astnode_t *expr = cypher_ast_set_property_get_expression(item);
    ck_assert_int_eq(cypher_astnode_type(expr), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(expr), "baz");
}
END_TEST


START_TEST (parse_merge_properties)
{
    struct cypher_input_position last = cypher_input_position_zero;
    result = cypher_parse("/*MATCH*/ SET n += {foo: bar};", &last, NULL, 0);
    ck_assert_ptr_ne(result, NULL);
    ck_assert_int_eq(last.offset, 30);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
"@0   2..7   block_comment           /*MATCH*/\n"
"@1  10..30  statement               body=@2\n"
"@2  10..30  > query                 clauses=[@3]\n"
"@3  10..29  > > SET                 items=[@4]\n"
"@4  14..29  > > > merge properties  @5 += @6\n"
"@5  14..15  > > > > identifier      `n`\n"
"@6  19..29  > > > > map             {@7:@8}\n"
"@7  20..23  > > > > > prop name     `foo`\n"
"@8  25..28  > > > > > identifier    `bar`\n";
    ck_assert_str_eq(memstream_buffer, expected);

    const cypher_astnode_t *ast = cypher_parse_result_get_directive(result, 0);
    ck_assert_int_eq(cypher_astnode_type(ast), CYPHER_AST_STATEMENT);
    const cypher_astnode_t *query = cypher_ast_statement_get_body(ast);
    ck_assert_int_eq(cypher_astnode_type(query), CYPHER_AST_QUERY);
    const cypher_astnode_t *set = cypher_ast_query_get_clause(query, 0);
    ck_assert_int_eq(cypher_astnode_type(set), CYPHER_AST_SET);

    ck_assert_int_eq(cypher_ast_set_nitems(set), 1);
    ck_assert_ptr_eq(cypher_ast_set_get_item(set, 1), NULL);

    const cypher_astnode_t *item = cypher_ast_set_get_item(set, 0);
    ck_assert_int_eq(cypher_astnode_type(item), CYPHER_AST_MERGE_PROPERTIES);

    const cypher_astnode_t *id = cypher_ast_merge_properties_get_identifier(item);
    ck_assert_int_eq(cypher_astnode_type(id), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(id), "n");

    const cypher_astnode_t *expr = cypher_ast_merge_properties_get_expression(item);
    ck_assert_int_eq(cypher_astnode_type(expr), CYPHER_AST_MAP);
}
END_TEST


START_TEST (parse_set_labels)
{
    struct cypher_input_position last = cypher_input_position_zero;
    result = cypher_parse("/*MATCH*/ SET n:Foo:Bar;", &last, NULL, 0);
    ck_assert_ptr_ne(result, NULL);
    ck_assert_int_eq(last.offset, 24);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
"@0   2..7   block_comment       /*MATCH*/\n"
"@1  10..24  statement           body=@2\n"
"@2  10..24  > query             clauses=[@3]\n"
"@3  10..23  > > SET             items=[@4]\n"
"@4  14..23  > > > set labels    @5:@6:@7\n"
"@5  14..15  > > > > identifier  `n`\n"
"@6  15..19  > > > > label       :`Foo`\n"
"@7  19..23  > > > > label       :`Bar`\n";
    ck_assert_str_eq(memstream_buffer, expected);

    const cypher_astnode_t *ast = cypher_parse_result_get_directive(result, 0);
    ck_assert_int_eq(cypher_astnode_type(ast), CYPHER_AST_STATEMENT);
    const cypher_astnode_t *query = cypher_ast_statement_get_body(ast);
    ck_assert_int_eq(cypher_astnode_type(query), CYPHER_AST_QUERY);
    const cypher_astnode_t *set = cypher_ast_query_get_clause(query, 0);
    ck_assert_int_eq(cypher_astnode_type(set), CYPHER_AST_SET);

    ck_assert_int_eq(cypher_ast_set_nitems(set), 1);
    ck_assert_ptr_eq(cypher_ast_set_get_item(set, 1), NULL);

    const cypher_astnode_t *item = cypher_ast_set_get_item(set, 0);
    ck_assert_int_eq(cypher_astnode_type(item), CYPHER_AST_SET_LABELS);

    ck_assert_int_eq(cypher_ast_set_labels_nlabels(item), 2);
    ck_assert_ptr_eq(cypher_ast_set_labels_get_label(item, 2), NULL);

    const cypher_astnode_t *label = cypher_ast_set_labels_get_label(item, 0);
    ck_assert_int_eq(cypher_astnode_type(label), CYPHER_AST_LABEL);
    ck_assert_str_eq(cypher_ast_label_get_name(label), "Foo");

    label = cypher_ast_set_labels_get_label(item, 1);
    ck_assert_int_eq(cypher_astnode_type(label), CYPHER_AST_LABEL);
    ck_assert_str_eq(cypher_ast_label_get_name(label), "Bar");
}
END_TEST


TCase* set_tcase(void)
{
    TCase *tc = tcase_create("set");
    tcase_add_checked_fixture(tc, setup, teardown);
    tcase_add_test(tc, parse_set_property);
    tcase_add_test(tc, parse_set_all_properties);
    tcase_add_test(tc, parse_set_nested_property);
    tcase_add_test(tc, parse_set_expression_property);
    tcase_add_test(tc, parse_merge_properties);
    tcase_add_test(tc, parse_set_labels);
    return tc;
}
