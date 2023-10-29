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


struct using_join
{
    cypher_astnode_t _astnode;
    unsigned int nidentifiers;
    const cypher_astnode_t *identifiers[];
};


static cypher_astnode_t *clone(const cypher_astnode_t *self,
        cypher_astnode_t **children);
static ssize_t detailstr(const cypher_astnode_t *self, char *str, size_t size);


static const struct cypher_astnode_vt *parents[] =
    { &cypher_match_hint_astnode_vt };

const struct cypher_astnode_vt cypher_using_join_astnode_vt =
    { .parents = parents,
      .nparents = 1,
      .name = "USING JOIN",
      .detailstr = detailstr,
      .release = cypher_astnode_release,
      .clone = clone };


cypher_astnode_t *cypher_ast_using_join(
        cypher_astnode_t * const *identifiers, unsigned int nidentifiers,
        cypher_astnode_t **children, unsigned int nchildren,
        struct cypher_input_range range)
{
    REQUIRE(nidentifiers > 0, NULL);
    REQUIRE_CHILD_ALL(children, nchildren, identifiers, nidentifiers,
            CYPHER_AST_IDENTIFIER, NULL);

    struct using_join *node = calloc(1, sizeof(struct using_join) +
            nidentifiers * sizeof(cypher_astnode_t *));
    if (node == NULL)
    {
        return NULL;
    }
    if (cypher_astnode_init(&(node->_astnode), CYPHER_AST_USING_JOIN,
            children, nchildren, range))
    {
        goto cleanup;
    }
    memcpy(node->identifiers, identifiers,
            nidentifiers * sizeof(cypher_astnode_t *));
    node->nidentifiers = nidentifiers;
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
    REQUIRE_TYPE(self, CYPHER_AST_USING_JOIN, NULL);
    struct using_join *node = container_of(self, struct using_join, _astnode);

    cypher_astnode_t **identifiers = calloc(node->nidentifiers,
            sizeof(cypher_astnode_t *));
    if (identifiers == NULL)
    {
        return NULL;
    }
    for (unsigned int i = 0; i < node->nidentifiers; ++i)
    {
        identifiers[i] = children[child_index(self, node->identifiers[i])];
    }

    cypher_astnode_t *clone = cypher_ast_using_join(identifiers,
            node->nidentifiers, children, self->nchildren, self->range);
    int errsv = errno;
    free(identifiers);
    errno = errsv;
    return clone;
}


unsigned int cypher_ast_using_join_nidentifiers(const cypher_astnode_t *astnode)
{
    REQUIRE_TYPE(astnode, CYPHER_AST_USING_JOIN, 0);
    struct using_join *node = container_of(astnode,
            struct using_join, _astnode);
    return node->nidentifiers;
}


const cypher_astnode_t *cypher_ast_using_join_get_identifier(
        const cypher_astnode_t *astnode, unsigned int index)
{
    REQUIRE_TYPE(astnode, CYPHER_AST_USING_JOIN, NULL);
    struct using_join *node = container_of(astnode,
            struct using_join, _astnode);
    if (index >= node->nidentifiers)
    {
        return NULL;
    }
    return node->identifiers[index];
}


ssize_t detailstr(const cypher_astnode_t *self, char *str, size_t size)
{
    REQUIRE_TYPE(self, CYPHER_AST_USING_JOIN, -1);
    struct using_join *node = container_of(self, struct using_join, _astnode);

    strncpy(str, "on=", size);
    if (size > 0)
    {
        str[size-1] = '\0';
    }
    size_t n = 3;

    ssize_t r = snprint_sequence(str+n, (n < size)? size-n : 0,
            node->identifiers, node->nidentifiers);
    if (r < 0)
    {
        return -1;
    }
    return n + r;
}
