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


struct return_clause
{
    cypher_astnode_t _astnode;
    bool distinct;
    bool include_existing;
    const cypher_astnode_t *order_by;
    const cypher_astnode_t *skip;
    const cypher_astnode_t *limit;
    unsigned int nprojections;
    const cypher_astnode_t *projections[];
};


static cypher_astnode_t *clone(const cypher_astnode_t *self,
        cypher_astnode_t **children);
static ssize_t detailstr(const cypher_astnode_t *self, char *str, size_t size);


static const struct cypher_astnode_vt *parents[] =
    { &cypher_query_clause_astnode_vt };

const struct cypher_astnode_vt cypher_return_astnode_vt =
    { .parents = parents,
      .nparents = 1,
      .name = "RETURN",
      .detailstr = detailstr,
      .release = cypher_astnode_release,
      .clone = clone };


cypher_astnode_t *cypher_ast_return(bool distinct, bool include_existing,
        cypher_astnode_t * const *projections, unsigned int nprojections,
        const cypher_astnode_t *order_by, const cypher_astnode_t *skip,
        const cypher_astnode_t *limit, cypher_astnode_t **children,
        unsigned int nchildren, struct cypher_input_range range)
{
    REQUIRE(include_existing || nprojections > 0, NULL);
    REQUIRE_CHILD_ALL(children, nchildren, projections, nprojections,
            CYPHER_AST_PROJECTION, NULL);
    REQUIRE_CHILD_OPTIONAL(children, nchildren, order_by,
            CYPHER_AST_ORDER_BY, NULL);
    REQUIRE_CHILD_OPTIONAL(children, nchildren, skip,
            CYPHER_AST_EXPRESSION, NULL);
    REQUIRE_CHILD_OPTIONAL(children, nchildren, limit,
            CYPHER_AST_EXPRESSION, NULL);

    struct return_clause *node = calloc(1, sizeof(struct return_clause) +
            nprojections * sizeof(cypher_astnode_t *));
    if (node == NULL)
    {
        return NULL;
    }
    if (cypher_astnode_init(&(node->_astnode), CYPHER_AST_RETURN,
            children, nchildren, range))
    {
        goto cleanup;
    }
    node->distinct = distinct;
    node->include_existing = include_existing;
    memcpy(node->projections, projections,
            nprojections * sizeof(cypher_astnode_t *));
    node->nprojections = nprojections;
    node->order_by = order_by;
    node->skip = skip;
    node->limit = limit;
    return &(node->_astnode);

    int errsv;
cleanup:
    errsv = errno;
    free(node);
    errno = errsv;
    return NULL;
}


cypher_astnode_t *clone(const cypher_astnode_t *self,
        cypher_astnode_t **children)
{
    REQUIRE_TYPE(self, CYPHER_AST_RETURN, NULL);
    struct return_clause *node =
            container_of(self, struct return_clause, _astnode);

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
    cypher_astnode_t *order_by = (node->order_by == NULL) ? NULL :
            children[child_index(self, node->order_by)];
    cypher_astnode_t *skip = (node->skip == NULL) ? NULL :
            children[child_index(self, node->skip)];
    cypher_astnode_t *limit = (node->limit == NULL) ? NULL :
            children[child_index(self, node->limit)];

    cypher_astnode_t *clone = cypher_ast_return(node->distinct,
            node->include_existing, projections, node->nprojections,
            order_by, skip, limit, children, self->nchildren, self->range);
    int errsv = errno;
    free(projections);
    errno = errsv;
    return clone;
}


bool cypher_ast_return_is_distinct(const cypher_astnode_t *astnode)
{
    REQUIRE_TYPE(astnode, CYPHER_AST_RETURN, false);
    struct return_clause *node =
            container_of(astnode, struct return_clause, _astnode);
    return node->distinct;
}


bool cypher_ast_return_has_include_existing(const cypher_astnode_t *astnode)
{
    REQUIRE_TYPE(astnode, CYPHER_AST_RETURN, false);
    struct return_clause *node =
            container_of(astnode, struct return_clause, _astnode);
    return node->include_existing;
}


unsigned int cypher_ast_return_nprojections(const cypher_astnode_t *astnode)
{
    REQUIRE_TYPE(astnode, CYPHER_AST_RETURN, 0);
    struct return_clause *node =
            container_of(astnode, struct return_clause, _astnode);
    return node->nprojections;
}


const cypher_astnode_t *cypher_ast_return_get_projection(
        const cypher_astnode_t *astnode, unsigned int index)
{
    REQUIRE_TYPE(astnode, CYPHER_AST_RETURN, NULL);
    struct return_clause *node =
            container_of(astnode, struct return_clause, _astnode);
    if (index >= node->nprojections)
    {
        return NULL;
    }
    return node->projections[index];
}


const cypher_astnode_t *cypher_ast_return_get_order_by(
        const cypher_astnode_t *astnode)
{
    REQUIRE_TYPE(astnode, CYPHER_AST_RETURN, NULL);
    struct return_clause *node =
            container_of(astnode, struct return_clause, _astnode);
    return node->order_by;
}


const cypher_astnode_t *cypher_ast_return_get_skip(
        const cypher_astnode_t *astnode)
{
    REQUIRE_TYPE(astnode, CYPHER_AST_RETURN, NULL);
    struct return_clause *node =
            container_of(astnode, struct return_clause, _astnode);
    return node->skip;
}


const cypher_astnode_t *cypher_ast_return_get_limit(
        const cypher_astnode_t *astnode)
{
    REQUIRE_TYPE(astnode, CYPHER_AST_RETURN, NULL);
    struct return_clause *node =
            container_of(astnode, struct return_clause, _astnode);
    return node->limit;
}


ssize_t detailstr(const cypher_astnode_t *self, char *str, size_t size)
{
    REQUIRE_TYPE(self, CYPHER_AST_RETURN, -1);
    struct return_clause *node =
            container_of(self, struct return_clause, _astnode);

    ssize_t r = snprintf(str, size, "%s%s%s%sprojections=",
            node->distinct? "DISTINCT":"",
            (node->distinct && node->include_existing)? ", ":"",
            node->include_existing? "*":"",
            (node->distinct || node->include_existing)? ", ":"");
    if (r < 0)
    {
        return -1;
    }
    size_t n = r;
    r = snprint_sequence(str + n, (n < size)? size-n : 0,
            node->projections, node->nprojections);
    if (r < 0)
    {
        return -1;
    }
    n += r;

    if (node->order_by != NULL)
    {
        r = snprintf(str + n, (n < size)? size-n : 0, ", ORDER BY=@%u",
                node->order_by->ordinal);
        if (r < 0)
        {
            return -1;
        }
        n += r;
    }

    if (node->skip != NULL)
    {
        r = snprintf(str + n, (n < size)? size-n : 0, ", SKIP=@%u",
                node->skip->ordinal);
        if (r < 0)
        {
            return -1;
        }
        n += r;
    }

    if (node->limit != NULL)
    {
        r = snprintf(str + n, (n < size)? size-n : 0, ", LIMIT=@%u",
                node->limit->ordinal);
        if (r < 0)
        {
            return -1;
        }
        n += r;
    }

    return n;
}
