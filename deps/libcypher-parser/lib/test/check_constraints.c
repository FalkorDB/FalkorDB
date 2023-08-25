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


START_TEST (parse_create_unique_node_prop_constraint)
{
    struct cypher_input_position last = cypher_input_position_zero;
    result = cypher_parse(
            "CREATE CONSTRAINT ON (f:Foo) ASSERT f.bar IS UNIQUE;",
            &last, NULL, 0);
    ck_assert_ptr_ne(result, NULL);
    ck_assert_int_eq(last.offset, 52);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
"@0   0..52  statement                      body=@1\n"
"@1   0..51  > create node prop constraint  ON=(@2:@3), expression=@4, IS UNIQUE\n"
"@2  22..23  > > identifier                 `f`\n"
"@3  23..27  > > label                      :`Foo`\n"
"@4  36..42  > > property                   @5.@6\n"
"@5  36..37  > > > identifier               `f`\n"
"@6  38..41  > > > prop name                `bar`\n";
    ck_assert_str_eq(memstream_buffer, expected);

    ck_assert_int_eq(cypher_parse_result_ndirectives(result), 1);
    const cypher_astnode_t *ast = cypher_parse_result_get_directive(result, 0);
    ck_assert_int_eq(cypher_astnode_type(ast), CYPHER_AST_STATEMENT);
    ck_assert_int_eq(cypher_astnode_range(ast).start.offset, 0);
    ck_assert_int_eq(cypher_astnode_range(ast).end.offset, 52);

    ck_assert_int_eq(cypher_ast_statement_noptions(ast), 0);
    ck_assert_ptr_eq(cypher_ast_statement_get_option(ast, 0), NULL);

    const cypher_astnode_t *body = cypher_ast_statement_get_body(ast);
    ck_assert_int_eq(cypher_astnode_type(body),
            CYPHER_AST_CREATE_NODE_PROP_CONSTRAINT);
    ck_assert_int_eq(cypher_astnode_range(body).start.offset, 0);
    ck_assert_int_eq(cypher_astnode_range(body).end.offset, 51);

    const cypher_astnode_t *id =
            cypher_ast_create_node_prop_constraint_get_identifier(body);
    ck_assert_int_eq(cypher_astnode_type(id), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(id), "f");

    const cypher_astnode_t *label =
            cypher_ast_create_node_prop_constraint_get_label(body);
    ck_assert_int_eq(cypher_astnode_type(label), CYPHER_AST_LABEL);
    ck_assert_str_eq(cypher_ast_label_get_name(label), "Foo");

    const cypher_astnode_t *expr =
            cypher_ast_create_node_prop_constraint_get_expression(body);
    ck_assert(cypher_astnode_instanceof(expr, CYPHER_AST_EXPRESSION));
    ck_assert_int_eq(cypher_astnode_type(expr), CYPHER_AST_PROPERTY_OPERATOR);

    const cypher_astnode_t *eid =
            cypher_ast_property_operator_get_expression(expr);
    ck_assert_int_eq(cypher_astnode_type(eid), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(eid), "f");

    const cypher_astnode_t *prop_name =
            cypher_ast_property_operator_get_prop_name(expr);
    ck_assert_int_eq(cypher_astnode_type(prop_name), CYPHER_AST_PROP_NAME);
    ck_assert_str_eq(cypher_ast_prop_name_get_value(prop_name), "bar");

    ck_assert(cypher_ast_create_node_prop_constraint_is_unique(body));
}
END_TEST


START_TEST (parse_drop_unique_node_prop_constraint)
{
    struct cypher_input_position last = cypher_input_position_zero;
    result = cypher_parse(
            "DROP CONSTRAINT ON (f:Foo) ASSERT f.bar IS UNIQUE;",
            &last, NULL, 0);
    ck_assert_ptr_ne(result, NULL);
    ck_assert_int_eq(last.offset, 50);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
"@0   0..50  statement                    body=@1\n"
"@1   0..49  > drop node prop constraint  ON=(@2:@3), expression=@4, IS UNIQUE\n"
"@2  20..21  > > identifier               `f`\n"
"@3  21..25  > > label                    :`Foo`\n"
"@4  34..40  > > property                 @5.@6\n"
"@5  34..35  > > > identifier             `f`\n"
"@6  36..39  > > > prop name              `bar`\n";
    ck_assert_str_eq(memstream_buffer, expected);

    ck_assert_int_eq(cypher_parse_result_ndirectives(result), 1);
    const cypher_astnode_t *ast = cypher_parse_result_get_directive(result, 0);
    ck_assert_int_eq(cypher_astnode_type(ast), CYPHER_AST_STATEMENT);
    ck_assert_int_eq(cypher_astnode_range(ast).start.offset, 0);
    ck_assert_int_eq(cypher_astnode_range(ast).end.offset, 50);

    ck_assert_int_eq(cypher_ast_statement_noptions(ast), 0);
    ck_assert_ptr_eq(cypher_ast_statement_get_option(ast, 0), NULL);

    const cypher_astnode_t *body = cypher_ast_statement_get_body(ast);
    ck_assert_int_eq(cypher_astnode_type(body),
            CYPHER_AST_DROP_NODE_PROP_CONSTRAINT);
    ck_assert_int_eq(cypher_astnode_range(body).start.offset, 0);
    ck_assert_int_eq(cypher_astnode_range(body).end.offset, 49);

    const cypher_astnode_t *id =
            cypher_ast_drop_node_prop_constraint_get_identifier(body);
    ck_assert_int_eq(cypher_astnode_type(id), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(id), "f");

    const cypher_astnode_t *label =
            cypher_ast_drop_node_prop_constraint_get_label(body);
    ck_assert_int_eq(cypher_astnode_type(label), CYPHER_AST_LABEL);
    ck_assert_str_eq(cypher_ast_label_get_name(label), "Foo");

    const cypher_astnode_t *expr =
            cypher_ast_drop_node_prop_constraint_get_expression(body);
    ck_assert(cypher_astnode_instanceof(expr, CYPHER_AST_EXPRESSION));
    ck_assert_int_eq(cypher_astnode_type(expr), CYPHER_AST_PROPERTY_OPERATOR);

    const cypher_astnode_t *eid =
            cypher_ast_property_operator_get_expression(expr);
    ck_assert_int_eq(cypher_astnode_type(eid), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(eid), "f");

    const cypher_astnode_t *prop_name =
            cypher_ast_property_operator_get_prop_name(expr);
    ck_assert_int_eq(cypher_astnode_type(prop_name), CYPHER_AST_PROP_NAME);
    ck_assert_str_eq(cypher_ast_prop_name_get_value(prop_name), "bar");

    ck_assert(cypher_ast_drop_node_prop_constraint_is_unique(body));
}
END_TEST


START_TEST (parse_create_node_prop_constraint)
{
    struct cypher_input_position last = cypher_input_position_zero;
    result = cypher_parse(
            "CREATE CONSTRAINT ON (f:Foo) ASSERT exists(f.bar);",
            &last, NULL, 0);
    ck_assert_ptr_ne(result, NULL);
    ck_assert_int_eq(last.offset, 50);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
"@0   0..50  statement                      body=@1\n"
"@1   0..49  > create node prop constraint  ON=(@2:@3), expression=@4\n"
"@2  22..23  > > identifier                 `f`\n"
"@3  23..27  > > label                      :`Foo`\n"
"@4  36..49  > > apply                      @5(@6)\n"
"@5  36..42  > > > function name            `exists`\n"
"@6  43..48  > > > property                 @7.@8\n"
"@7  43..44  > > > > identifier             `f`\n"
"@8  45..48  > > > > prop name              `bar`\n";
    ck_assert_str_eq(memstream_buffer, expected);

    ck_assert_int_eq(cypher_parse_result_ndirectives(result), 1);
    const cypher_astnode_t *ast = cypher_parse_result_get_directive(result, 0);
    ck_assert_int_eq(cypher_astnode_type(ast), CYPHER_AST_STATEMENT);
    ck_assert_int_eq(cypher_astnode_range(ast).start.offset, 0);
    ck_assert_int_eq(cypher_astnode_range(ast).end.offset, 50);

    ck_assert_int_eq(cypher_ast_statement_noptions(ast), 0);
    ck_assert_ptr_eq(cypher_ast_statement_get_option(ast, 0), NULL);

    const cypher_astnode_t *body = cypher_ast_statement_get_body(ast);
    ck_assert_int_eq(cypher_astnode_type(body),
            CYPHER_AST_CREATE_NODE_PROP_CONSTRAINT);
    ck_assert_int_eq(cypher_astnode_range(body).start.offset, 0);
    ck_assert_int_eq(cypher_astnode_range(body).end.offset, 49);

    const cypher_astnode_t *id =
            cypher_ast_create_node_prop_constraint_get_identifier(body);
    ck_assert_int_eq(cypher_astnode_type(id), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(id), "f");

    const cypher_astnode_t *label =
            cypher_ast_create_node_prop_constraint_get_label(body);
    ck_assert_int_eq(cypher_astnode_type(label), CYPHER_AST_LABEL);
    ck_assert_str_eq(cypher_ast_label_get_name(label), "Foo");

    const cypher_astnode_t *apply =
            cypher_ast_create_node_prop_constraint_get_expression(body);
    ck_assert_int_eq(cypher_astnode_type(apply), CYPHER_AST_APPLY_OPERATOR);

    const cypher_astnode_t *func_name =
            cypher_ast_apply_operator_get_func_name(apply);
    ck_assert_int_eq(cypher_astnode_type(func_name), CYPHER_AST_FUNCTION_NAME);
    ck_assert_str_eq(cypher_ast_function_name_get_value(func_name), "exists");

    const cypher_astnode_t *property =
            cypher_ast_apply_operator_get_argument(apply, 0);
    ck_assert_int_eq(cypher_astnode_type(property),
            CYPHER_AST_PROPERTY_OPERATOR);

    const cypher_astnode_t *expr =
            cypher_ast_property_operator_get_expression(property);
    ck_assert_int_eq(cypher_astnode_type(expr), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(expr), "f");

    const cypher_astnode_t *prop_name =
            cypher_ast_property_operator_get_prop_name(property);
    ck_assert_int_eq(cypher_astnode_type(prop_name), CYPHER_AST_PROP_NAME);
    ck_assert_str_eq(cypher_ast_prop_name_get_value(prop_name), "bar");

    ck_assert(!cypher_ast_create_node_prop_constraint_is_unique(body));
}
END_TEST


START_TEST (parse_drop_node_prop_constraint)
{
    struct cypher_input_position last = cypher_input_position_zero;
    result = cypher_parse(
            "DROP CONSTRAINT ON (f:Foo) ASSERT exists(f.bar);",
            &last, NULL, 0);
    ck_assert_ptr_ne(result, NULL);
    ck_assert_int_eq(last.offset, 48);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
"@0   0..48  statement                    body=@1\n"
"@1   0..47  > drop node prop constraint  ON=(@2:@3), expression=@4\n"
"@2  20..21  > > identifier               `f`\n"
"@3  21..25  > > label                    :`Foo`\n"
"@4  34..47  > > apply                    @5(@6)\n"
"@5  34..40  > > > function name          `exists`\n"
"@6  41..46  > > > property               @7.@8\n"
"@7  41..42  > > > > identifier           `f`\n"
"@8  43..46  > > > > prop name            `bar`\n";
    ck_assert_str_eq(memstream_buffer, expected);

    ck_assert_int_eq(cypher_parse_result_ndirectives(result), 1);
    const cypher_astnode_t *ast = cypher_parse_result_get_directive(result, 0);
    ck_assert_int_eq(cypher_astnode_type(ast), CYPHER_AST_STATEMENT);
    ck_assert_int_eq(cypher_astnode_range(ast).start.offset, 0);
    ck_assert_int_eq(cypher_astnode_range(ast).end.offset, 48);

    ck_assert_int_eq(cypher_ast_statement_noptions(ast), 0);
    ck_assert_ptr_eq(cypher_ast_statement_get_option(ast, 0), NULL);

    const cypher_astnode_t *body = cypher_ast_statement_get_body(ast);
    ck_assert_int_eq(cypher_astnode_type(body),
            CYPHER_AST_DROP_NODE_PROP_CONSTRAINT);
    ck_assert_int_eq(cypher_astnode_range(body).start.offset, 0);
    ck_assert_int_eq(cypher_astnode_range(body).end.offset, 47);

    const cypher_astnode_t *id =
            cypher_ast_drop_node_prop_constraint_get_identifier(body);
    ck_assert_int_eq(cypher_astnode_type(id), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(id), "f");

    const cypher_astnode_t *label =
            cypher_ast_drop_node_prop_constraint_get_label(body);
    ck_assert_int_eq(cypher_astnode_type(label), CYPHER_AST_LABEL);
    ck_assert_str_eq(cypher_ast_label_get_name(label), "Foo");

    const cypher_astnode_t *apply =
            cypher_ast_drop_node_prop_constraint_get_expression(body);
    ck_assert_int_eq(cypher_astnode_type(apply), CYPHER_AST_APPLY_OPERATOR);

    const cypher_astnode_t *func_name =
            cypher_ast_apply_operator_get_func_name(apply);
    ck_assert_int_eq(cypher_astnode_type(func_name), CYPHER_AST_FUNCTION_NAME);
    ck_assert_str_eq(cypher_ast_function_name_get_value(func_name), "exists");

    const cypher_astnode_t *property =
            cypher_ast_apply_operator_get_argument(apply, 0);
    ck_assert_int_eq(cypher_astnode_type(property),
            CYPHER_AST_PROPERTY_OPERATOR);

    const cypher_astnode_t *expr =
            cypher_ast_property_operator_get_expression(property);
    ck_assert_int_eq(cypher_astnode_type(expr), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(expr), "f");

    const cypher_astnode_t *prop_name =
            cypher_ast_property_operator_get_prop_name(property);
    ck_assert_int_eq(cypher_astnode_type(prop_name), CYPHER_AST_PROP_NAME);
    ck_assert_str_eq(cypher_ast_prop_name_get_value(prop_name), "bar");

    ck_assert(!cypher_ast_drop_node_prop_constraint_is_unique(body));
}
END_TEST


START_TEST (parse_create_rel_prop_constraint)
{
    struct cypher_input_position last = cypher_input_position_zero;
    result = cypher_parse(
            "CREATE CONSTRAINT ON ()-[f:Foo]-() ASSERT exists(f.bar);",
            &last, NULL, 0);
    ck_assert_ptr_ne(result, NULL);
    ck_assert_int_eq(last.offset, 56);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
"@0   0..56  statement                     body=@1\n"
"@1   0..55  > create rel prop constraint  ON=(@2:@3), expression=@4\n"
"@2  25..26  > > identifier                `f`\n"
"@3  26..30  > > rel type                  :`Foo`\n"
"@4  42..55  > > apply                     @5(@6)\n"
"@5  42..48  > > > function name           `exists`\n"
"@6  49..54  > > > property                @7.@8\n"
"@7  49..50  > > > > identifier            `f`\n"
"@8  51..54  > > > > prop name             `bar`\n";
    ck_assert_str_eq(memstream_buffer, expected);

    ck_assert_int_eq(cypher_parse_result_ndirectives(result), 1);
    const cypher_astnode_t *ast = cypher_parse_result_get_directive(result, 0);
    ck_assert_int_eq(cypher_astnode_type(ast), CYPHER_AST_STATEMENT);
    ck_assert_int_eq(cypher_astnode_range(ast).start.offset, 0);
    ck_assert_int_eq(cypher_astnode_range(ast).end.offset, 56);

    ck_assert_int_eq(cypher_ast_statement_noptions(ast), 0);
    ck_assert_ptr_eq(cypher_ast_statement_get_option(ast, 0), NULL);

    const cypher_astnode_t *body = cypher_ast_statement_get_body(ast);
    ck_assert_int_eq(cypher_astnode_type(body),
            CYPHER_AST_CREATE_REL_PROP_CONSTRAINT);
    ck_assert_int_eq(cypher_astnode_range(body).start.offset, 0);
    ck_assert_int_eq(cypher_astnode_range(body).end.offset, 55);

    const cypher_astnode_t *id =
            cypher_ast_create_rel_prop_constraint_get_identifier(body);
    ck_assert_int_eq(cypher_astnode_type(id), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(id), "f");

    const cypher_astnode_t *reltype =
            cypher_ast_create_rel_prop_constraint_get_reltype(body);
    ck_assert_int_eq(cypher_astnode_type(reltype), CYPHER_AST_RELTYPE);
    ck_assert_str_eq(cypher_ast_reltype_get_name(reltype), "Foo");

    const cypher_astnode_t *apply =
            cypher_ast_create_rel_prop_constraint_get_expression(body);
    ck_assert_int_eq(cypher_astnode_type(apply), CYPHER_AST_APPLY_OPERATOR);

    const cypher_astnode_t *func_name =
            cypher_ast_apply_operator_get_func_name(apply);
    ck_assert_int_eq(cypher_astnode_type(func_name), CYPHER_AST_FUNCTION_NAME);
    ck_assert_str_eq(cypher_ast_function_name_get_value(func_name), "exists");

    const cypher_astnode_t *property =
            cypher_ast_apply_operator_get_argument(apply, 0);
    ck_assert_int_eq(cypher_astnode_type(property),
            CYPHER_AST_PROPERTY_OPERATOR);

    const cypher_astnode_t *expr =
            cypher_ast_property_operator_get_expression(property);
    ck_assert_int_eq(cypher_astnode_type(expr), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(expr), "f");

    const cypher_astnode_t *prop_name =
            cypher_ast_property_operator_get_prop_name(property);
    ck_assert_int_eq(cypher_astnode_type(prop_name), CYPHER_AST_PROP_NAME);
    ck_assert_str_eq(cypher_ast_prop_name_get_value(prop_name), "bar");

    ck_assert(!cypher_ast_create_rel_prop_constraint_is_unique(body));
}
END_TEST


START_TEST (parse_drop_rel_prop_constraint)
{
    struct cypher_input_position last = cypher_input_position_zero;
    result = cypher_parse(
            "DROP CONSTRAINT ON ()-[f:Foo]-() ASSERT exists(f.bar);",
            &last, NULL, 0);
    ck_assert_ptr_ne(result, NULL);
    ck_assert_int_eq(last.offset, 54);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
"@0   0..54  statement                   body=@1\n"
"@1   0..53  > drop rel prop constraint  ON=(@2:@3), expression=@4\n"
"@2  23..24  > > identifier              `f`\n"
"@3  24..28  > > rel type                :`Foo`\n"
"@4  40..53  > > apply                   @5(@6)\n"
"@5  40..46  > > > function name         `exists`\n"
"@6  47..52  > > > property              @7.@8\n"
"@7  47..48  > > > > identifier          `f`\n"
"@8  49..52  > > > > prop name           `bar`\n";
    ck_assert_str_eq(memstream_buffer, expected);

    ck_assert_int_eq(cypher_parse_result_ndirectives(result), 1);
    const cypher_astnode_t *ast = cypher_parse_result_get_directive(result, 0);
    ck_assert_int_eq(cypher_astnode_type(ast), CYPHER_AST_STATEMENT);
    ck_assert_int_eq(cypher_astnode_range(ast).start.offset, 0);
    ck_assert_int_eq(cypher_astnode_range(ast).end.offset, 54);

    ck_assert_int_eq(cypher_ast_statement_noptions(ast), 0);
    ck_assert_ptr_eq(cypher_ast_statement_get_option(ast, 0), NULL);

    const cypher_astnode_t *body = cypher_ast_statement_get_body(ast);
    ck_assert_int_eq(cypher_astnode_type(body),
            CYPHER_AST_DROP_REL_PROP_CONSTRAINT);
    ck_assert_int_eq(cypher_astnode_range(body).start.offset, 0);
    ck_assert_int_eq(cypher_astnode_range(body).end.offset, 53);

    const cypher_astnode_t *id =
            cypher_ast_drop_rel_prop_constraint_get_identifier(body);
    ck_assert_int_eq(cypher_astnode_type(id), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(id), "f");

    const cypher_astnode_t *reltype =
            cypher_ast_drop_rel_prop_constraint_get_reltype(body);
    ck_assert_int_eq(cypher_astnode_type(reltype), CYPHER_AST_RELTYPE);
    ck_assert_str_eq(cypher_ast_reltype_get_name(reltype), "Foo");

    const cypher_astnode_t *apply =
            cypher_ast_drop_rel_prop_constraint_get_expression(body);
    ck_assert_int_eq(cypher_astnode_type(apply), CYPHER_AST_APPLY_OPERATOR);

    const cypher_astnode_t *func_name =
            cypher_ast_apply_operator_get_func_name(apply);
    ck_assert_int_eq(cypher_astnode_type(func_name), CYPHER_AST_FUNCTION_NAME);
    ck_assert_str_eq(cypher_ast_function_name_get_value(func_name), "exists");

    const cypher_astnode_t *property =
            cypher_ast_apply_operator_get_argument(apply, 0);
    ck_assert_int_eq(cypher_astnode_type(property),
            CYPHER_AST_PROPERTY_OPERATOR);

    const cypher_astnode_t *expr =
            cypher_ast_property_operator_get_expression(property);
    ck_assert_int_eq(cypher_astnode_type(expr), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(expr), "f");

    const cypher_astnode_t *prop_name =
            cypher_ast_property_operator_get_prop_name(property);
    ck_assert_int_eq(cypher_astnode_type(prop_name), CYPHER_AST_PROP_NAME);
    ck_assert_str_eq(cypher_ast_prop_name_get_value(prop_name), "bar");

    ck_assert(!cypher_ast_drop_rel_prop_constraint_is_unique(body));
}
END_TEST


TCase* constraints_tcase(void)
{
    TCase *tc = tcase_create("constraints");
    tcase_add_checked_fixture(tc, setup, teardown);
    tcase_add_test(tc, parse_create_unique_node_prop_constraint);
    tcase_add_test(tc, parse_drop_unique_node_prop_constraint);
    tcase_add_test(tc, parse_create_node_prop_constraint);
    tcase_add_test(tc, parse_drop_node_prop_constraint);
    tcase_add_test(tc, parse_create_rel_prop_constraint);
    tcase_add_test(tc, parse_drop_rel_prop_constraint);
    return tc;
}
