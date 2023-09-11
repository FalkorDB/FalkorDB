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


struct rel_id_lookup
{
    cypher_astnode_t _astnode;
    const cypher_astnode_t *identifier;
    unsigned int nids;
    const cypher_astnode_t *ids[];
};


static cypher_astnode_t *clone(const cypher_astnode_t *self,
        cypher_astnode_t **children);
static ssize_t detailstr(const cypher_astnode_t *self, char *str, size_t size);


static const struct cypher_astnode_vt *parents[] =
    { &cypher_start_point_astnode_vt };

const struct cypher_astnode_vt cypher_rel_id_lookup_astnode_vt =
    { .parents = parents,
      .nparents = 1,
      .name = "rel id lookup",
      .detailstr = detailstr,
      .release = cypher_astnode_release,
      .clone = clone };


cypher_astnode_t *cypher_ast_rel_id_lookup(const cypher_astnode_t *identifier,
        cypher_astnode_t * const *ids, unsigned int nids,
        cypher_astnode_t **children, unsigned int nchildren,
        struct cypher_input_range range)
{
    REQUIRE_CHILD(children, nchildren, identifier, CYPHER_AST_IDENTIFIER, NULL);
    REQUIRE(nids > 0, NULL);
    REQUIRE_CHILD_ALL(children, nchildren, ids, nids, CYPHER_AST_INTEGER, NULL);

    struct rel_id_lookup *node = calloc(1, sizeof(struct rel_id_lookup) +
            nids * sizeof(cypher_astnode_t *));
    if (node == NULL)
    {
        return NULL;
    }
    if (cypher_astnode_init(&(node->_astnode), CYPHER_AST_REL_ID_LOOKUP,
            children, nchildren, range))
    {
        goto cleanup;
    }
    node->identifier = identifier;
    memcpy(node->ids, ids, nids * sizeof(cypher_astnode_t *));
    node->nids = nids;
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
    REQUIRE_TYPE(self, CYPHER_AST_REL_ID_LOOKUP, NULL);
    struct rel_id_lookup *node =
            container_of(self, struct rel_id_lookup, _astnode);

    cypher_astnode_t *identifier = children[child_index(self, node->identifier)];
    cypher_astnode_t **ids = calloc(node->nids, sizeof(cypher_astnode_t *));
    if (ids == NULL)
    {
        return NULL;
    }
    for (unsigned int i = 0; i < node->nids; ++i)
    {
        ids[i] = children[child_index(self, node->ids[i])];
    }

    cypher_astnode_t *clone = cypher_ast_rel_id_lookup(identifier, ids,
            node->nids, children, self->nchildren, self->range);
    int errsv = errno;
    free(ids);
    errno = errsv;
    return clone;
}


const cypher_astnode_t *cypher_ast_rel_id_lookup_get_identifier(
        const cypher_astnode_t *astnode)
{
    REQUIRE_TYPE(astnode, CYPHER_AST_REL_ID_LOOKUP, NULL);
    struct rel_id_lookup *node =
            container_of(astnode, struct rel_id_lookup, _astnode);
    return node->identifier;
}


unsigned int cypher_ast_rel_id_lookup_nids(const cypher_astnode_t *astnode)
{
    REQUIRE_TYPE(astnode, CYPHER_AST_REL_ID_LOOKUP, 0);
    struct rel_id_lookup *node =
            container_of(astnode, struct rel_id_lookup, _astnode);
    return node->nids;
}


const cypher_astnode_t *cypher_ast_rel_id_lookup_get_id(
        const cypher_astnode_t *astnode, unsigned int index)
{
    REQUIRE_TYPE(astnode, CYPHER_AST_REL_ID_LOOKUP, NULL);
    struct rel_id_lookup *node =
            container_of(astnode, struct rel_id_lookup, _astnode);
    if (index >= node->nids)
    {
        return NULL;
    }
    return node->ids[index];
}


ssize_t detailstr(const cypher_astnode_t *self, char *str, size_t size)
{
    REQUIRE_TYPE(self, CYPHER_AST_REL_ID_LOOKUP, -1);
    struct rel_id_lookup *node =
            container_of(self, struct rel_id_lookup, _astnode);

    size_t n = 0;
    ssize_t r = snprintf(str, size, "@%u = rel(", node->identifier->ordinal);
    if (r < 0)
    {
        return -1;
    }
    n += r;

    for (unsigned int i = 0; i < node->nids; ++i)
    {
        r = snprintf(str+n, (n < size)? size-n : 0, "%s@%u",
                (i > 0)? ", " : "", node->ids[i]->ordinal);
        if (r < 0)
        {
            return -1;
        }
        n += r;
    }

    if ((n+1) < size)
    {
        str[n] = ')';
        str[size-1] = '\0';
    }
    ++n;
    return n;
}
