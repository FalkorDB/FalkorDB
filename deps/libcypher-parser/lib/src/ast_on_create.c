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


struct on_create
{
    cypher_astnode_t _astnode;
    unsigned int nitems;
    const cypher_astnode_t *items[];
};


static cypher_astnode_t *clone(const cypher_astnode_t *self,
        cypher_astnode_t **children);
static ssize_t detailstr(const cypher_astnode_t *self, char *str, size_t size);


static const struct cypher_astnode_vt *parents[] =
    { &cypher_merge_action_astnode_vt };

const struct cypher_astnode_vt cypher_on_create_astnode_vt =
    { .parents = parents,
      .nparents = 1,
      .name = "ON CREATE",
      .detailstr = detailstr,
      .release = cypher_astnode_release,
      .clone = clone };


cypher_astnode_t *cypher_ast_on_create(cypher_astnode_t * const *items,
        unsigned int nitems, cypher_astnode_t **children,
        unsigned int nchildren, struct cypher_input_range range)
{
    REQUIRE(nitems > 0, NULL);
    REQUIRE_CHILD_ALL(children, nchildren, items, nitems,
            CYPHER_AST_SET_ITEM, NULL);

    struct on_create *node = calloc(1, sizeof(struct on_create) +
            nitems * sizeof(cypher_astnode_t *));
    if (node == NULL)
    {
        return NULL;
    }
    if (cypher_astnode_init(&(node->_astnode), CYPHER_AST_ON_CREATE,
            children, nchildren, range))
    {
        goto cleanup;
    }
    memcpy(node->items, items, nitems * sizeof(cypher_astnode_t *));
    node->nitems = nitems;
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
    REQUIRE_TYPE(self, CYPHER_AST_ON_CREATE, NULL);
    struct on_create *node = container_of(self, struct on_create, _astnode);

    cypher_astnode_t **items = calloc(node->nitems,
            sizeof(cypher_astnode_t *));
    if (items == NULL)
    {
        return NULL;
    }
    for (unsigned int i = 0; i < node->nitems; ++i)
    {
        items[i] = children[child_index(self, node->items[i])];
    }

    cypher_astnode_t *clone = cypher_ast_on_create(items, node->nitems,
            children, self->nchildren, self->range);
    int errsv = errno;
    free(items);
    errno = errsv;
    return clone;
}


unsigned int cypher_ast_on_create_nitems(const cypher_astnode_t *astnode)
{
    REQUIRE_TYPE(astnode, CYPHER_AST_ON_CREATE, 0);
    struct on_create *node = container_of(astnode, struct on_create, _astnode);
    return node->nitems;
}


const cypher_astnode_t *cypher_ast_on_create_get_item(
        const cypher_astnode_t *astnode, unsigned int index)
{
    REQUIRE_TYPE(astnode, CYPHER_AST_ON_CREATE, 0);
    struct on_create *node = container_of(astnode, struct on_create, _astnode);
    if (index >= node->nitems)
    {
        return NULL;
    }
    return node->items[index];
}


ssize_t detailstr(const cypher_astnode_t *self, char *str, size_t size)
{
    REQUIRE_TYPE(self, CYPHER_AST_ON_CREATE, -1);
    struct on_create *node = container_of(self, struct on_create, _astnode);

    strncpy(str, "items=", size);
    if (size > 0)
    {
        str[size-1] = '\0';
    }
    size_t n = 6;

    ssize_t r = snprint_sequence(str+n, (n < size)? size-n : 0,
            node->items, node->nitems);
    if (r < 0)
    {
        return -1;
    }
    n += r;
    return n;
}
