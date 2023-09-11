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
#include "operators.h"
#include "util.h"
#include <assert.h>


struct map_projection
{
    cypher_astnode_t _astnode;
    const cypher_astnode_t *expression;
    unsigned int nselectors;
    const cypher_astnode_t *selectors[];
};


static cypher_astnode_t *clone(const cypher_astnode_t *self,
        cypher_astnode_t **children);
static ssize_t detailstr(const cypher_astnode_t *self, char *str, size_t size);


static const struct cypher_astnode_vt *parents[] =
    { &cypher_expression_astnode_vt };

const struct cypher_astnode_vt cypher_map_projection_astnode_vt =
    { .parents = parents,
      .nparents = 1,
      .name = "map projection",
      .detailstr = detailstr,
      .release = cypher_astnode_release,
      .clone = clone };


cypher_astnode_t *cypher_ast_map_projection(
        const cypher_astnode_t *expression,
        cypher_astnode_t * const *selectors, unsigned int nselectors,
        cypher_astnode_t **children, unsigned int nchildren,
        struct cypher_input_range range)
{
    REQUIRE_CHILD(children, nchildren, expression, CYPHER_AST_EXPRESSION, NULL);

    struct map_projection *node = calloc(1, sizeof(struct map_projection) +
            nselectors * sizeof(cypher_astnode_t *));
    if (node == NULL)
    {
        return NULL;
    }
    if (cypher_astnode_init(&(node->_astnode), CYPHER_AST_MAP_PROJECTION,
            children, nchildren, range))
    {
        free(node);
        return NULL;
    }
    node->expression = expression;
    memcpy(node->selectors, selectors, nselectors * sizeof(cypher_astnode_t *));
    node->nselectors = nselectors;
    return &(node->_astnode);
}


cypher_astnode_t *clone(const cypher_astnode_t *self,
        cypher_astnode_t **children)
{
    REQUIRE_TYPE(self, CYPHER_AST_MAP_PROJECTION, NULL);
    struct map_projection *node =
            container_of(self, struct map_projection, _astnode);

    cypher_astnode_t *expression = children[child_index(self, node->expression)];
    cypher_astnode_t **selectors = calloc(node->nselectors,
            sizeof(cypher_astnode_t *));
    if (selectors == NULL)
    {
        return NULL;
    }
    for (unsigned int i = 0; i < node->nselectors; ++i)
    {
        selectors[i] = children[child_index(self, node->selectors[i])];
    }

    cypher_astnode_t *clone = cypher_ast_map_projection(expression,
            selectors, node->nselectors, children, self->nchildren,
            self->range);
    int errsv = errno;
    free(selectors);
    errno = errsv;
    return clone;
}


const cypher_astnode_t *cypher_ast_map_projection_get_expression(
        const cypher_astnode_t *astnode)
{
    REQUIRE_TYPE(astnode, CYPHER_AST_MAP_PROJECTION, NULL);
    struct map_projection *node =
            container_of(astnode, struct map_projection, _astnode);
    return node->expression;
}


unsigned int cypher_ast_map_projection_nselectors(
        const cypher_astnode_t *astnode)
{
    REQUIRE_TYPE(astnode, CYPHER_AST_MAP_PROJECTION, 0);
    struct map_projection *node =
            container_of(astnode, struct map_projection, _astnode);
    return node->nselectors;
}


const cypher_astnode_t *cypher_ast_map_projection_get_selector(
        const cypher_astnode_t *astnode, unsigned int index)
{
    REQUIRE_TYPE(astnode, CYPHER_AST_MAP_PROJECTION, NULL);
    struct map_projection *node =
            container_of(astnode, struct map_projection, _astnode);
    if (index >= node->nselectors)
    {
        return NULL;
    }
    return node->selectors[index];
}


ssize_t detailstr(const cypher_astnode_t *self, char *str, size_t size)
{
    REQUIRE_TYPE(self, CYPHER_AST_MAP_PROJECTION, -1);
    struct map_projection *node =
            container_of(self, struct map_projection, _astnode);

    size_t n = 0;
    ssize_t w = snprintf(str, size, "@%u{", node->expression->ordinal);
    if (w < 0)
    {
        return -1;
    }
    n += w;

    for (unsigned int i = 0; i < node->nselectors; )
    {
        ssize_t r = snprintf(str+n, (n < size)? size-n : 0,
                "@%u", node->selectors[i]->ordinal);
        if (r < 0)
        {
            return -1;
        }
        n += r;
        if (++i < node->nselectors)
        {
            if (n < size)
            {
                str[n] = ',';
            }
            n++;
            if (n < size)
            {
                str[n] = ' ';
            }
            n++;
        }
    }

    if (n+1 < size)
    {
        str[n] = '}';
        str[n+1] = '\0';
    }
    n++;

    return n;
}
