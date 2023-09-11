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


struct collection
{
    cypher_astnode_t _astnode;
    size_t nelements;
    const cypher_astnode_t *elements[];
};


static cypher_astnode_t *clone(const cypher_astnode_t *self,
        cypher_astnode_t **children);
static ssize_t detailstr(const cypher_astnode_t *self, char *str, size_t size);


static const struct cypher_astnode_vt *parents[] =
    { &cypher_expression_astnode_vt };

const struct cypher_astnode_vt cypher_collection_astnode_vt =
    { .parents = parents,
      .nparents = 1,
      .name = "collection",
      .detailstr = detailstr,
      .release = cypher_astnode_release,
      .clone = clone };


cypher_astnode_t *cypher_ast_collection(
        cypher_astnode_t * const *elements, unsigned int nelements,
        cypher_astnode_t **children, unsigned int nchildren,
        struct cypher_input_range range)
{
    REQUIRE_CHILD_ALL(children, nchildren, elements, nelements,
            CYPHER_AST_EXPRESSION, NULL);

    struct collection *node = calloc(1, sizeof(struct collection) +
            nelements * sizeof(cypher_astnode_t *));
    if (node == NULL)
    {
        return NULL;
    }
    if (cypher_astnode_init(&(node->_astnode), CYPHER_AST_COLLECTION,
                children, nchildren, range))
    {
        free(node);
        return NULL;
    }
    memcpy(node->elements, elements, nelements * sizeof(cypher_astnode_t *));
    node->nelements = nelements;
    return &(node->_astnode);
}


cypher_astnode_t *clone(const cypher_astnode_t *self,
        cypher_astnode_t **children)
{
    REQUIRE_TYPE(self, CYPHER_AST_COLLECTION, NULL);
    struct collection *node = container_of(self, struct collection, _astnode);

    cypher_astnode_t **elements = calloc(node->nelements,
            sizeof(cypher_astnode_t *));
    if (elements == NULL)
    {
        return NULL;
    }
    for (unsigned int i = 0; i < node->nelements; ++i)
    {
        elements[i] = children[child_index(self, node->elements[i])];
    }

    cypher_astnode_t *clone = cypher_ast_collection(elements, node->nelements,
            children, self->nchildren, self->range);
    int errsv = errno;
    free(elements);
    errno = errsv;
    return clone;
}


unsigned int cypher_ast_collection_length(const cypher_astnode_t *astnode)
{
    REQUIRE_TYPE(astnode, CYPHER_AST_COLLECTION, 0);
    struct collection *node =
            container_of(astnode, struct collection, _astnode);
    return node->nelements;
}


const cypher_astnode_t *cypher_ast_collection_get(
        const cypher_astnode_t *astnode, unsigned int index)
{
    REQUIRE_TYPE(astnode, CYPHER_AST_COLLECTION, NULL);
    struct collection *node =
            container_of(astnode, struct collection, _astnode);

    if (index >= node->nelements)
    {
        return NULL;
    }
    return node->elements[index];
}


ssize_t detailstr(const cypher_astnode_t *self, char *str, size_t size)
{
    REQUIRE_TYPE(self, CYPHER_AST_COLLECTION, -1);
    struct collection *node = container_of(self, struct collection, _astnode);

    return snprint_sequence(str, size, node->elements, node->nelements);
}
