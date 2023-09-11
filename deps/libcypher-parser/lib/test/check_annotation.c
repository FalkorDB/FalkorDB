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
static const cypher_astnode_t *ast;
static const cypher_astnode_t *query;
static const cypher_astnode_t *match;
static unsigned released;


static void setup(void)
{
    result = cypher_parse("MATCH (n:Label) RETURN n", NULL, NULL, 0);
    ck_assert_ptr_ne(result, NULL);

    ast = cypher_parse_result_get_directive(result, 0);
    query = cypher_ast_statement_get_body(ast);
    match = cypher_ast_query_get_clause(query, 0);

    released = 0;
}


static void teardown(void)
{
    cypher_parse_result_free(result);
}


START_TEST (annotate_single_node)
{
    cypher_ast_annotation_context_t *ctx = cypher_ast_annotation_context();

    ck_assert_ptr_eq(cypher_astnode_get_annotation(ctx, match), NULL);

    void *ptr1 = (void *)"foo";
    ck_assert_int_eq(cypher_astnode_attach_annotation(ctx, match, ptr1, NULL), 0);
    ck_assert_ptr_eq(cypher_astnode_get_annotation(ctx, match), ptr1);

    void *ptr2 = (void *)"bar";
    void *ptr3 = ptr2;
    ck_assert_int_eq(cypher_astnode_attach_annotation(ctx, match, ptr2, &ptr3), 0);
    ck_assert_ptr_eq(ptr3, ptr1);
    ck_assert_ptr_eq(cypher_astnode_get_annotation(ctx, match), ptr2);

    cypher_astnode_remove_annotation(ctx, match);
    ck_assert_ptr_eq(cypher_astnode_get_annotation(ctx, match), NULL);

    cypher_ast_annotation_context_free(ctx);
}
END_TEST


START_TEST (annotate_multiple_nodes)
{
    cypher_ast_annotation_context_t *ctx = cypher_ast_annotation_context();

    ck_assert_ptr_eq(cypher_astnode_get_annotation(ctx, query), NULL);
    ck_assert_ptr_eq(cypher_astnode_get_annotation(ctx, match), NULL);

    void *ptr1 = (void *)"foo";
    ck_assert_int_eq(cypher_astnode_attach_annotation(ctx, query, ptr1, NULL), 0);
    ck_assert_ptr_eq(cypher_astnode_get_annotation(ctx, query), ptr1);
    ck_assert_ptr_eq(cypher_astnode_get_annotation(ctx, match), NULL);

    void *ptr2 = (void *)"bar";
    void *ptr3 = ptr2;
    ck_assert_int_eq(cypher_astnode_attach_annotation(ctx, match, ptr2, &ptr3), 0);
    ck_assert_ptr_eq(ptr3, NULL);
    ck_assert_ptr_eq(cypher_astnode_get_annotation(ctx, query), ptr1);
    ck_assert_ptr_eq(cypher_astnode_get_annotation(ctx, match), ptr2);

    cypher_ast_annotation_context_free(ctx);

    ck_assert_ptr_eq(cypher_astnode_get_annotation(ctx, query), NULL);
    ck_assert_ptr_eq(cypher_astnode_get_annotation(ctx, match), NULL);
}
END_TEST


static void release_handler(void *userdata, const cypher_astnode_t *node,
        void *annotation)
{
    released++;
    ck_assert_ptr_eq(annotation, userdata);
}


START_TEST (annotations_are_released_on_context_free)
{
    cypher_ast_annotation_context_t *ctx = cypher_ast_annotation_context();

    void *ptr1 = (void *)"foo";

    cypher_ast_annotation_context_set_release_handler(ctx, release_handler, ptr1);
    ck_assert_int_eq(cypher_astnode_attach_annotation(ctx, query, ptr1, NULL), 0);
    cypher_ast_annotation_context_free(ctx);
    ck_assert_int_eq(released, 1);
}
END_TEST


START_TEST (annotations_are_released_on_ast_free)
{
    cypher_ast_annotation_context_t *ctx = cypher_ast_annotation_context();

    void *ptr1 = (void *)"foo";

    cypher_ast_annotation_context_set_release_handler(ctx, release_handler, ptr1);
    ck_assert_int_eq(cypher_astnode_attach_annotation(ctx, query, ptr1, NULL), 0);
    cypher_parse_result_free(result);
    result = NULL;
    ck_assert_int_eq(released, 1);

    cypher_ast_annotation_context_free(ctx);
    ck_assert_int_eq(released, 1);
}
END_TEST


TCase* annotation_tcase(void)
{
    TCase *tc = tcase_create("annotation");
    tcase_add_checked_fixture(tc, setup, teardown);
    tcase_add_test(tc, annotate_single_node);
    tcase_add_test(tc, annotate_multiple_nodes);
    tcase_add_test(tc, annotations_are_released_on_context_free);
    tcase_add_test(tc, annotations_are_released_on_ast_free);
    return tc;
}
