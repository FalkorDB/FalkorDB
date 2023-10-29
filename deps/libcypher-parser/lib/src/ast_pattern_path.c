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


int cypher_pattern_path_astnode_init(cypher_pattern_path_astnode_t *node,
        cypher_astnode_type_t type,
        const struct cypher_pattern_path_astnode_vt *vt,
        cypher_astnode_t **children, unsigned int nchildren,
        struct cypher_input_range range)
{
    assert(vt != NULL);
    if (cypher_astnode_init(&(node->_astnode),type, children, nchildren, range))
    {
        return -1;
    }
    node->_vt = vt;
    return 0;
}


struct pattern_path
{
    cypher_pattern_path_astnode_t _pattern_path_astnode;
    size_t nelements;
    const cypher_astnode_t *elements[];
};


static cypher_astnode_t *clone(const cypher_astnode_t *self,
        cypher_astnode_t **children);
static ssize_t detailstr(const cypher_astnode_t *self, char *str, size_t size);
static unsigned int nelements(const cypher_pattern_path_astnode_t *self);
static const cypher_astnode_t *get_element(
        const cypher_pattern_path_astnode_t *self, unsigned int index);


static const struct cypher_astnode_vt *parents[] =
    { &cypher_expression_astnode_vt };

const struct cypher_astnode_vt cypher_pattern_path_astnode_vt =
    { .parents = parents,
      .nparents = 1,
      .name = "pattern path",
      .detailstr = detailstr,
      .release = cypher_astnode_release,
      .clone = clone };

static const struct cypher_pattern_path_astnode_vt pp_vt =
    { .nelements = nelements,
      .get_element = get_element };


cypher_astnode_t *cypher_ast_pattern_path(cypher_astnode_t * const *elements,
        unsigned int nelements, cypher_astnode_t **children,
        unsigned int nchildren, struct cypher_input_range range)
{
    REQUIRE(nelements % 2 == 1, NULL);
    REQUIRE(elements != NULL, NULL);
    for (unsigned int i = 0; i < nelements; ++i)
    {
        REQUIRE_CHILD(children, nchildren, elements[i],
                (i%2 == 0)? CYPHER_AST_NODE_PATTERN : CYPHER_AST_REL_PATTERN,
                NULL);
    }

    struct pattern_path *node = calloc(1, sizeof(struct pattern_path) +
            nelements * sizeof(cypher_astnode_t *));
    if (node == NULL)
    {
        return NULL;
    }
    if (cypher_pattern_path_astnode_init(&(node->_pattern_path_astnode),
                CYPHER_AST_PATTERN_PATH, &pp_vt, children, nchildren, range))
    {
        goto cleanup;
    }
    memcpy(node->elements, elements, nelements * sizeof(cypher_astnode_t *));
    node->nelements = nelements;
    return &(node->_pattern_path_astnode._astnode);

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
    REQUIRE_TYPE(self, CYPHER_AST_PATTERN_PATH, NULL);
    const cypher_pattern_path_astnode_t *ppnode =
            container_of(self, cypher_pattern_path_astnode_t, _astnode);
    const struct pattern_path *node =
            container_of(ppnode, struct pattern_path, _pattern_path_astnode);

    cypher_astnode_t **elements =
            calloc(node->nelements, sizeof(cypher_astnode_t *));
    if (elements == NULL)
    {
        return NULL;
    }
    for (unsigned int i = 0; i < node->nelements; ++i)
    {
        elements[i] = children[child_index(self, node->elements[i])];
    }

    cypher_astnode_t *clone = cypher_ast_pattern_path(elements, node->nelements,
            children, self->nchildren, self->range);
    int errsv = errno;
    free(elements);
    errno = errsv;
    return clone;
}


unsigned int cypher_ast_pattern_path_nelements(const cypher_astnode_t *astnode)
{
    REQUIRE_TYPE(astnode, CYPHER_AST_PATTERN_PATH, 0);
    const cypher_pattern_path_astnode_t *ppnode =
            container_of(astnode, cypher_pattern_path_astnode_t, _astnode);
    assert(ppnode->_vt != NULL && ppnode->_vt->nelements != NULL);
    return ppnode->_vt->nelements(ppnode);
}


const cypher_astnode_t *cypher_ast_pattern_path_get_element(
        const cypher_astnode_t *astnode, unsigned int index)
{
    REQUIRE_TYPE(astnode, CYPHER_AST_PATTERN_PATH, NULL);
    const cypher_pattern_path_astnode_t *ppnode =
            container_of(astnode, cypher_pattern_path_astnode_t, _astnode);
    assert(ppnode->_vt != NULL && ppnode->_vt->get_element != NULL);
    return ppnode->_vt->get_element(ppnode, index);
}


unsigned int nelements(const cypher_pattern_path_astnode_t *self)
{
    const struct pattern_path *node =
            container_of(self, struct pattern_path, _pattern_path_astnode);
    return node->nelements;
}


const cypher_astnode_t *get_element(const cypher_pattern_path_astnode_t *self,
        unsigned int index)
{
    const struct pattern_path *node =
            container_of(self, struct pattern_path, _pattern_path_astnode);
    if (index >= node->nelements)
    {
        return NULL;
    }
    return node->elements[index];
}


ssize_t detailstr(const cypher_astnode_t *self, char *str, size_t size)
{
    REQUIRE_TYPE(self, CYPHER_AST_PATTERN_PATH, -1);
    const cypher_pattern_path_astnode_t *ppnode =
            container_of(self, cypher_pattern_path_astnode_t, _astnode);
    const struct pattern_path *node =
            container_of(ppnode, struct pattern_path, _pattern_path_astnode);
    assert(node->nelements % 2 == 1);

    size_t n = 0;
    for (unsigned int i = 0; i < node->nelements; ++i)
    {
        ssize_t r;
        if (i % 2 == 0)
        {
            r = snprintf(str+n, (n < size)? size-n : 0, "(@%u)",
                    node->elements[i]->ordinal);
        }
        else
        {
            r = snprintf(str+n, (n < size)? size-n : 0, "-[@%u]-",
                    node->elements[i]->ordinal);
        }
        if (r < 0)
        {
            return -1;
        }
        n += r;
    }
    return n;
}
