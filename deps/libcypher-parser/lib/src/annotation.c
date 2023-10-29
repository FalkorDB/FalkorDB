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
#include "annotation.h"
#include "astnode.h"
#include "util.h"
#include <assert.h>
#include <stdlib.h>
#include <string.h>


static struct cypher_astnode_annotation *find_annotation(
        const cypher_ast_annotation_context_t *context,
        const cypher_astnode_t *node);
static void attach_annotation_to_astnode(
        const cypher_astnode_t *node,
        struct cypher_astnode_annotation *annotation);
static void detach_annotation_from_astnode(
        struct cypher_astnode_annotation *annotation);
static void attach_annotation_to_context(
        cypher_ast_annotation_context_t *context,
        struct cypher_astnode_annotation *annotation);
static void detach_annotation_from_context(
        struct cypher_astnode_annotation *annotation);


cypher_ast_annotation_context_t *cypher_ast_annotation_context(void)
{
    return calloc(1, sizeof(cypher_ast_annotation_context_t));
}


void cypher_ast_annotation_context_set_release_handler(
        cypher_ast_annotation_context_t *context,
        cypher_ast_annotation_context_release_handler_t handler,
        void *userdata)
{
    context->release_cb = handler;
    context->release_cb_userdata = (handler == NULL) ? NULL : userdata;
}


void cypher_ast_annotation_context_free(
        cypher_ast_annotation_context_t *context)
{
    if (context == NULL)
    {
        return;
    }

    while (context->annotations != NULL)
    {
        cp_release_annotation(context->annotations);
    }
    free(context);
}


int cypher_astnode_attach_annotation(cypher_ast_annotation_context_t *context,
        const cypher_astnode_t *node, void *annotation,
        void **previous_annotation)
{
    REQUIRE(context != NULL, -1);
    REQUIRE(node != NULL, -1);
    REQUIRE(annotation != NULL, -1);

    struct cypher_astnode_annotation *annotation_node = find_annotation(
            context, node);
    if (annotation_node != NULL)
    {
        if (previous_annotation != NULL)
        {
            *previous_annotation = annotation_node->data;
        }
        annotation_node->data = annotation;
        return 0;
    }

    annotation_node = malloc(sizeof(struct cypher_astnode_annotation));
    if (annotation_node == NULL)
    {
        return -1;
    }
    memset(annotation_node, 0, sizeof(struct cypher_astnode_annotation));
    annotation_node->data = annotation;

    attach_annotation_to_astnode(node, annotation_node);
    attach_annotation_to_context(context, annotation_node);

    if (previous_annotation != NULL)
    {
        *previous_annotation = NULL;
    }

    return 0;
}


void *cypher_astnode_remove_annotation(cypher_ast_annotation_context_t *context,
        const cypher_astnode_t *node)
{
    REQUIRE(context != NULL, NULL);
    REQUIRE(node != NULL, NULL);

    struct cypher_astnode_annotation *annotation = find_annotation(
            context, node);
    if (annotation == NULL)
    {
        return NULL;
    }

    assert(node == annotation->astnode);
    assert(context == annotation->context);

    detach_annotation_from_astnode(annotation);
    detach_annotation_from_context(annotation);

    void *data = annotation->data;
    free(annotation);
    return data;
}


void *cypher_astnode_get_annotation(
        const cypher_ast_annotation_context_t *context,
        const cypher_astnode_t *node)
{
    REQUIRE(context != NULL, NULL);
    REQUIRE(node != NULL, NULL);

    struct cypher_astnode_annotation *annotation = find_annotation(
            context, node);
    if (annotation == NULL)
    {
        return NULL;
    }
    return annotation->data;
}


struct cypher_astnode_annotation *find_annotation(
        const cypher_ast_annotation_context_t *context,
        const cypher_astnode_t *node)
{
    // search using the astnode as it will typically have less items
    struct cypher_astnode_annotation *annotation = node->annotations;
    while (annotation != NULL && annotation->context != context)
    {
        annotation = annotation->node_next;
    }
    return annotation;
}


void attach_annotation_to_astnode(const cypher_astnode_t *node,
        struct cypher_astnode_annotation *annotation)
{
    annotation->astnode = node;

    // insert at head of list on the astnode (overriding the const qualifier)
    cypher_astnode_t *astnode = (cypher_astnode_t *)(uintptr_t)node;
    annotation->node_next = astnode->annotations;
    if (astnode->annotations != NULL)
    {
        astnode->annotations->node_prev = annotation;
    }
    astnode->annotations = annotation;
}


void detach_annotation_from_astnode(
        struct cypher_astnode_annotation *annotation)
{
    if (annotation->node_next != NULL)
    {
        annotation->node_next->node_prev = annotation->node_prev;
    }
    if (annotation->node_prev == NULL)
    {
        cypher_astnode_t *astnode =
            (cypher_astnode_t *)(uintptr_t)annotation->astnode;
        astnode->annotations = annotation->node_next;
    }
    else
    {
        annotation->node_prev->node_next = annotation->node_next;
    }
    annotation->astnode = NULL;
    annotation->node_next = NULL;
    annotation->node_prev = NULL;
}


void attach_annotation_to_context(cypher_ast_annotation_context_t *context,
        struct cypher_astnode_annotation *annotation)
{
    annotation->context = context;

    // insert at head of list on the context
    annotation->ctx_next = context->annotations;
    if (context->annotations != NULL)
    {
        context->annotations->ctx_prev = annotation;
    }
    context->annotations = annotation;
}


void detach_annotation_from_context(
        struct cypher_astnode_annotation *annotation)
{
    // remove from context list
    if (annotation->ctx_next != NULL)
    {
        annotation->ctx_next->ctx_prev = annotation->ctx_prev;
    }
    if (annotation->ctx_prev == NULL)
    {
        annotation->context->annotations = annotation->ctx_next;
    }
    else
    {
        annotation->ctx_prev->ctx_next = annotation->ctx_next;
    }
    annotation->context = NULL;
    annotation->ctx_next = NULL;
    annotation->ctx_prev = NULL;
}


void cp_release_annotation(struct cypher_astnode_annotation *annotation)
{
    assert(annotation != NULL);

    cypher_ast_annotation_context_t *context = annotation->context;

    detach_annotation_from_astnode(annotation);
    detach_annotation_from_context(annotation);
    if (context->release_cb != NULL)
    {
        context->release_cb(context->release_cb_userdata,
                annotation->astnode, annotation->data);
    }

    free(annotation);
}
