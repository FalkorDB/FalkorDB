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


struct map
{
    cypher_astnode_t _astnode;
    size_t nentries;
    const cypher_astnode_t *pairs[];
};


static cypher_astnode_t *clone(const cypher_astnode_t *self,
        cypher_astnode_t **children);
static ssize_t detailstr(const cypher_astnode_t *self, char *str, size_t size);


static const struct cypher_astnode_vt *parents[] =
    { &cypher_expression_astnode_vt };

const struct cypher_astnode_vt cypher_map_astnode_vt =
    { .parents = parents,
      .nparents = 1,
      .name = "map",
      .detailstr = detailstr,
      .release = cypher_astnode_release,
      .clone = clone };


static struct map *map_init(unsigned int nentries,
        cypher_astnode_t **children, unsigned int nchildren,
        struct cypher_input_range range)
{
    struct map *node = calloc(1, sizeof(struct map) +
            nentries * 2 * sizeof(cypher_astnode_t *));
    if (node == NULL)
    {
        return NULL;
    }
    if (cypher_astnode_init(&(node->_astnode), CYPHER_AST_MAP,
                children, nchildren, range))
    {
        goto cleanup;
    }
    node->nentries = nentries;
    return node;

    int errsv;
cleanup:
    errsv = errno;
    free(node);
    errno = errsv;
    return NULL;
}


cypher_astnode_t *cypher_ast_map(cypher_astnode_t * const *keys,
        cypher_astnode_t * const *values, unsigned int nentries,
        cypher_astnode_t **children, unsigned int nchildren,
        struct cypher_input_range range)
{
    REQUIRE_TYPE_ALL(keys, nentries, CYPHER_AST_PROP_NAME, NULL);
    REQUIRE_TYPE_ALL(values, nentries, CYPHER_AST_EXPRESSION, NULL);

    struct map *node = map_init(nentries, children, nchildren, range);
    if (node == NULL)
    {
        return NULL;
    }

    for (unsigned int i = 0; i < nentries; ++i)
    {
        node->pairs[i*2] = keys[i];
        node->pairs[i*2 + 1] = values[i];
    }
    return &(node->_astnode);
}


cypher_astnode_t *cypher_ast_pair_map(cypher_astnode_t * const *pairs,
        unsigned int nentries, cypher_astnode_t **children,
        unsigned int nchildren, struct cypher_input_range range)
{
    for (unsigned int i = 0; i < nentries; ++i)
    {
        REQUIRE_TYPE(pairs[i*2], CYPHER_AST_PROP_NAME, NULL);
        REQUIRE_TYPE(pairs[i*2 + 1], CYPHER_AST_EXPRESSION, NULL);
    }

    struct map *node = map_init(nentries, children, nchildren, range);
    if (node == NULL)
    {
        return NULL;
    }
    memcpy(node->pairs, pairs, nentries * 2 * sizeof(cypher_astnode_t *));
    return &(node->_astnode);
}


cypher_astnode_t *clone(const cypher_astnode_t *self,
        cypher_astnode_t **children)
{
    REQUIRE_TYPE(self, CYPHER_AST_MAP, NULL);
    struct map *node = container_of(self, struct map, _astnode);

    cypher_astnode_t **pairs = calloc(node->nentries * 2,
            sizeof(cypher_astnode_t *));
    if (pairs == NULL)
    {
        return NULL;
    }
    for (unsigned int i = 0; i < node->nentries * 2; ++i)
    {
        pairs[i] = children[child_index(self, node->pairs[i])];
    }

    cypher_astnode_t *clone = cypher_ast_pair_map(pairs, node->nentries,
            children, self->nchildren, self->range);
    int errsv = errno;
    free(pairs);
    errno = errsv;
    return clone;
}


unsigned int cypher_ast_map_nentries(const cypher_astnode_t *astnode)
{
    REQUIRE_TYPE(astnode, CYPHER_AST_MAP, 0);
    struct map *node = container_of(astnode, struct map, _astnode);
    return node->nentries;
}


const cypher_astnode_t *cypher_ast_map_get_key(const cypher_astnode_t *astnode,
        unsigned int index)
{
    REQUIRE_TYPE(astnode, CYPHER_AST_MAP, NULL);
    struct map *node = container_of(astnode, struct map, _astnode);
    if (index >= node->nentries)
    {
        return NULL;
    }
    return node->pairs[index*2];
}


const cypher_astnode_t *cypher_ast_map_get_value(const cypher_astnode_t *astnode,
        unsigned int index)
{
    REQUIRE_TYPE(astnode, CYPHER_AST_MAP, NULL);
    struct map *node = container_of(astnode, struct map, _astnode);
    if (index >= node->nentries)
    {
        return NULL;
    }
    return node->pairs[index*2 + 1];
}


ssize_t detailstr(const cypher_astnode_t *self, char *str, size_t size)
{
    REQUIRE_TYPE(self, CYPHER_AST_MAP, -1);
    struct map *node = container_of(self, struct map, _astnode);

    size_t n = 0;
    if (n < size)
    {
        str[n] = '{';
    }
    n++;
    for (unsigned int i = 0; i < node->nentries; ++i)
    {
        ssize_t r = snprintf(str+n, (n < size)? size-n : 0,
                "%s@%u:@%u", (i > 0)? ", ":"",
                node->pairs[i*2]->ordinal, node->pairs[i*2 + 1]->ordinal);
        if (r < 0)
        {
            return -1;
        }
        n += r;
    }
    if (n < size)
    {
        str[n] = '}';
    }
    n++;
    return n;
}
