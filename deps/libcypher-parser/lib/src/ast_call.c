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
#include "astnode.h"
#include "util.h"
#include <assert.h>


struct call_clause
{
    cypher_astnode_t _astnode;
    const cypher_astnode_t *proc_name;
    const cypher_astnode_t **args;
    unsigned int nargs;
    const cypher_astnode_t *predicate;
    unsigned int nprojections;
    const cypher_astnode_t *projections[];
};


static cypher_astnode_t *clone(const cypher_astnode_t *self,
        cypher_astnode_t **children);
static ssize_t detailstr(const cypher_astnode_t *self, char *str, size_t size);
static void call_release(cypher_astnode_t *self);


static const struct cypher_astnode_vt *parents[] =
    { &cypher_query_clause_astnode_vt };

const struct cypher_astnode_vt cypher_call_astnode_vt =
    { .parents = parents,
      .nparents = 1,
      .name = "CALL",
      .detailstr = detailstr,
      .release = call_release,
      .clone = clone };


cypher_astnode_t *cypher_ast_call(const cypher_astnode_t *proc_name,
        cypher_astnode_t * const *args, unsigned int nargs,
        cypher_astnode_t * const *projections, unsigned int nprojections,
        const cypher_astnode_t *predicate, cypher_astnode_t **children,
        unsigned int nchildren, struct cypher_input_range range)
{
    REQUIRE_CHILD(children, nchildren, proc_name, CYPHER_AST_PROC_NAME, NULL);
    REQUIRE_CHILD_ALL(children, nchildren, args, nargs,
            CYPHER_AST_EXPRESSION, NULL);
    REQUIRE_CHILD_ALL(children, nchildren, projections, nprojections,
            CYPHER_AST_PROJECTION, NULL);
    REQUIRE_CHILD_OPTIONAL(children, nchildren, predicate,
            CYPHER_AST_EXPRESSION, NULL);

    struct call_clause *node = calloc(1, sizeof(struct call_clause) +
            nprojections * sizeof(cypher_astnode_t *));
    if (node == NULL)
    {
        return NULL;
    }
    if (cypher_astnode_init(&(node->_astnode), CYPHER_AST_CALL,
            children, nchildren, range))
    {
        goto cleanup;
    }
    node->proc_name = proc_name;
    if (nargs > 0)
    {
        node->args = mdup(args, nargs * sizeof(cypher_astnode_t *));
        if (node->args == NULL)
        {
            goto cleanup;
        }
        node->nargs = nargs;
    }
    node->predicate = predicate;
    memcpy(node->projections, projections,
            nprojections * sizeof(cypher_astnode_t *));
    node->nprojections = nprojections;
    return &(node->_astnode);

    int errsv;
cleanup:
    errsv = errno;
    free(node);
    errno = errsv;
    return NULL;
}


void call_release(cypher_astnode_t *self)
{
    struct call_clause *node = container_of(self, struct call_clause, _astnode);
    free(node->args);
    cypher_astnode_release(self);
}


cypher_astnode_t *clone(const cypher_astnode_t *self,
        cypher_astnode_t **children)
{
    REQUIRE_TYPE(self, CYPHER_AST_CALL, NULL);
    struct call_clause *node = container_of(self, struct call_clause, _astnode);

    cypher_astnode_t *proc_name = children[child_index(self, node->proc_name)];
    cypher_astnode_t **args = calloc(node->nargs, sizeof(cypher_astnode_t *));
    if (args == NULL)
    {
        return NULL;
    }
    for (unsigned int i = 0; i < node->nargs; ++i)
    {
        args[i] = children[child_index(self, node->args[i])];
    }
    cypher_astnode_t **projections =
            calloc(node->nprojections, sizeof(cypher_astnode_t *));
    if (projections == NULL)
    {
        return NULL;
    }
    for (unsigned int i = 0; i < node->nprojections; ++i)
    {
        projections[i] = children[child_index(self, node->projections[i])];
    }

    cypher_astnode_t *predicate = (node->predicate == NULL) ? NULL :
            children[child_index(self, node->predicate)];

    cypher_astnode_t *clone = cypher_ast_call(proc_name, args, node->nargs,
            projections, node->nprojections, predicate,
            children, self->nchildren, self->range);
    int errsv = errno;
    free(args);
    free(projections);
    errno = errsv;
    return clone;
}


const cypher_astnode_t *cypher_ast_call_get_proc_name(
        const cypher_astnode_t *astnode)
{
    REQUIRE_TYPE(astnode, CYPHER_AST_CALL, NULL);
    struct call_clause *node = container_of(astnode, struct call_clause, _astnode);
    return node->proc_name;
}


unsigned int cypher_ast_call_narguments(const cypher_astnode_t *astnode)
{
    REQUIRE_TYPE(astnode, CYPHER_AST_CALL, 0);
    struct call_clause *node = container_of(astnode, struct call_clause, _astnode);
    return node->nargs;
}


const cypher_astnode_t *cypher_ast_call_get_argument(
        const cypher_astnode_t *astnode, unsigned int index)
{
    REQUIRE_TYPE(astnode, CYPHER_AST_CALL, NULL);
    struct call_clause *node = container_of(astnode, struct call_clause, _astnode);
    if (index >= node->nargs)
    {
        return NULL;
    }
    return node->args[index];
}


unsigned int cypher_ast_call_nprojections(const cypher_astnode_t *astnode)
{
    REQUIRE_TYPE(astnode, CYPHER_AST_CALL, 0);
    struct call_clause *node = container_of(astnode, struct call_clause, _astnode);
    return node->nprojections;
}


const cypher_astnode_t *cypher_ast_call_get_projection(
        const cypher_astnode_t *astnode, unsigned int index)
{
    REQUIRE_TYPE(astnode, CYPHER_AST_CALL, NULL);
    struct call_clause *node = container_of(astnode, struct call_clause, _astnode);
    if (index >= node->nprojections)
    {
        return NULL;
    }
    return node->projections[index];
}


const cypher_astnode_t *cypher_ast_call_get_predicate(
        const cypher_astnode_t *astnode)
{
    REQUIRE_TYPE(astnode, CYPHER_AST_CALL, NULL);
    struct call_clause *node =
            container_of(astnode, struct call_clause, _astnode);
    return node->predicate;
}


ssize_t detailstr(const cypher_astnode_t *self, char *str, size_t size)
{
    REQUIRE_TYPE(self, CYPHER_AST_CALL, -1);
    struct call_clause *node = container_of(self, struct call_clause, _astnode);

    size_t n = 0;
    ssize_t r = snprintf(str, size, "name=@%u", node->proc_name->ordinal);
    if (r < 0)
    {
        return -1;
    }
    n += r;

    if (node->nargs > 0)
    {
        if (n < size)
        {
            strncpy(str + n, ", args=", size-n);
            str[size-1] = '\0';
        }
        n += 7;

        r = snprint_sequence(str + n, (n < size)? size-n : 0,
                node->args, node->nargs);
        if (r < 0)
        {
            return -1;
        }
        n += r;
    }

    if (node->nprojections > 0)
    {
        if (n < size)
        {
            strncpy(str + n, ", YIELD=", size-n);
            str[size-1] = '\0';
        }
        n += 8;

        r = snprint_sequence(str + n, (n < size)? size-n : 0,
                node->projections, node->nprojections);
        if (r < 0)
        {
            return -1;
        }
        n += r;
    }

    if (node->predicate != NULL)
    {
        r = snprintf(str + n, (n < size)? size-n : 0, ", WHERE=@%u",
                node->predicate->ordinal);
        if (r < 0)
        {
            return -1;
        }
        n += r;
    }

    return n;
}
