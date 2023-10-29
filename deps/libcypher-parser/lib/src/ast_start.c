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


struct start
{
    cypher_astnode_t _astnode;
    const cypher_astnode_t *predicate;
    unsigned int npoints;
    const cypher_astnode_t *points[];
};


static cypher_astnode_t *clone(const cypher_astnode_t *self,
        cypher_astnode_t **children);
static ssize_t detailstr(const cypher_astnode_t *self, char *str, size_t size);


static const struct cypher_astnode_vt *parents[] =
    { &cypher_query_clause_astnode_vt };

const struct cypher_astnode_vt cypher_start_astnode_vt =
    { .parents = parents,
      .nparents = 1,
      .name = "START",
      .detailstr = detailstr,
      .release = cypher_astnode_release,
      .clone = clone };


cypher_astnode_t *cypher_ast_start(cypher_astnode_t * const *points,
        unsigned int npoints, const cypher_astnode_t *predicate,
        cypher_astnode_t **children, unsigned int nchildren,
        struct cypher_input_range range)
{
    REQUIRE(npoints > 0, NULL);
    REQUIRE_CHILD_ALL(children, nchildren, points, npoints,
            CYPHER_AST_START_POINT, NULL);
    REQUIRE_CHILD_OPTIONAL(children, nchildren, predicate,
            CYPHER_AST_EXPRESSION, NULL);

    struct start *node = calloc(1, sizeof(struct start) +
            npoints * sizeof(cypher_astnode_t *));
    if (node == NULL)
    {
        return NULL;
    }
    if (cypher_astnode_init(&(node->_astnode), CYPHER_AST_START,
            children, nchildren, range))
    {
        goto cleanup;
    }
    memcpy(node->points, points, npoints * sizeof(cypher_astnode_t *));
    node->npoints = npoints;
    node->predicate = predicate;
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
    REQUIRE_TYPE(self, CYPHER_AST_START, NULL);
    struct start *node = container_of(self, struct start, _astnode);

    cypher_astnode_t **points = calloc(node->npoints,
            sizeof(cypher_astnode_t *));
    if (points == NULL)
    {
        return NULL;
    }
    for (unsigned int i = 0; i < node->npoints; ++i)
    {
        points[i] = children[child_index(self, node->points[i])];
    }
    cypher_astnode_t *predicate = (node->predicate == NULL) ? NULL :
            children[child_index(self, node->predicate)];

    cypher_astnode_t *clone = cypher_ast_start(points, node->npoints,
            predicate, children, self->nchildren, self->range);
    int errsv = errno;
    free(points);
    errno = errsv;
    return clone;
}


unsigned int cypher_ast_start_npoints(const cypher_astnode_t *astnode)
{
    REQUIRE_TYPE(astnode, CYPHER_AST_START, 0);
    struct start *node = container_of(astnode, struct start, _astnode);
    return node->npoints;
}


const cypher_astnode_t *cypher_ast_start_get_point(
        const cypher_astnode_t *astnode, unsigned int index)
{
    REQUIRE_TYPE(astnode, CYPHER_AST_START, NULL);
    struct start *node = container_of(astnode, struct start, _astnode);
    if (index >= node->npoints)
    {
        return NULL;
    }
    return node->points[index];
}


const cypher_astnode_t *cypher_ast_start_get_predicate(
        const cypher_astnode_t *astnode)
{
    REQUIRE_TYPE(astnode, CYPHER_AST_START, NULL);
    struct start *node = container_of(astnode, struct start, _astnode);
    return node->predicate;
}


ssize_t detailstr(const cypher_astnode_t *self, char *str, size_t size)
{
    REQUIRE_TYPE(self, CYPHER_AST_START, -1);
    struct start *node = container_of(self, struct start, _astnode);

    size_t n = 0;
    if (n < size)
    {
        strncpy(str+n, "points=", size-n);
        str[size-1] = '\0';
    }
    n += 7;

    ssize_t r = snprint_sequence(str+n, (n < size)? size-n : 0,
            node->points, node->npoints);
    if (r < 0)
    {
        return -1;
    }
    n += r;

    if (node->predicate != NULL)
    {
        r = snprintf(str+n, (n < size)? size-n : 0, ", WHERE=@%u",
                node->predicate->ordinal);
        if (r < 0)
        {
            return -1;
        }
        n += r;
    }

    return n;
}
