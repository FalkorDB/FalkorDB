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


struct node_index_lookup
{
    cypher_astnode_t _astnode;
    const cypher_astnode_t *identifier;
    const cypher_astnode_t *index_name;
    const cypher_astnode_t *prop_name;
    const cypher_astnode_t *lookup;
};


static cypher_astnode_t *clone(const cypher_astnode_t *self,
        cypher_astnode_t **children);
static ssize_t detailstr(const cypher_astnode_t *self, char *str, size_t size);


static const struct cypher_astnode_vt *parents[] =
    { &cypher_start_point_astnode_vt };

const struct cypher_astnode_vt cypher_node_index_lookup_astnode_vt =
    { .parents = parents,
      .nparents = 1,
      .name = "node index lookup",
      .detailstr = detailstr,
      .release = cypher_astnode_release,
      .clone = clone };


cypher_astnode_t *cypher_ast_node_index_lookup(
        const cypher_astnode_t *identifier, const cypher_astnode_t *index_name,
        const cypher_astnode_t *prop_name, const cypher_astnode_t *lookup,
        cypher_astnode_t **children, unsigned int nchildren,
        struct cypher_input_range range)
{
    REQUIRE_CHILD(children, nchildren, identifier, CYPHER_AST_IDENTIFIER, NULL);
    REQUIRE_CHILD(children, nchildren, index_name, CYPHER_AST_INDEX_NAME, NULL);
    REQUIRE_CHILD(children, nchildren, prop_name, CYPHER_AST_PROP_NAME, NULL);
    REQUIRE(cypher_astnode_instanceof(lookup, CYPHER_AST_STRING) ||
            cypher_astnode_instanceof(lookup, CYPHER_AST_PARAMETER), NULL);
    REQUIRE_CONTAINS(children, nchildren, lookup, NULL);

    struct node_index_lookup *node = calloc(1, sizeof(struct node_index_lookup));
    if (node == NULL)
    {
        return NULL;
    }
    if (cypher_astnode_init(&(node->_astnode), CYPHER_AST_NODE_INDEX_LOOKUP,
            children, nchildren, range))
    {
        free(node);
        return NULL;
    }
    node->identifier = identifier;
    node->index_name = index_name;
    node->prop_name = prop_name;
    node->lookup = lookup;
    return &(node->_astnode);
}


cypher_astnode_t *clone(const cypher_astnode_t *self,
        cypher_astnode_t **children)
{
    REQUIRE_TYPE(self, CYPHER_AST_NODE_INDEX_LOOKUP, NULL);
    struct node_index_lookup *node =
            container_of(self, struct node_index_lookup, _astnode);

    cypher_astnode_t *identifier = children[child_index(self, node->identifier)];
    cypher_astnode_t *index_name = children[child_index(self, node->index_name)];
    cypher_astnode_t *prop_name = children[child_index(self, node->prop_name)];
    cypher_astnode_t *lookup = children[child_index(self, node->lookup)];

    return cypher_ast_node_index_lookup(identifier, index_name, prop_name,
            lookup, children, self->nchildren, self->range);
}


const cypher_astnode_t *cypher_ast_node_index_lookup_get_identifier(
        const cypher_astnode_t *astnode)
{
    REQUIRE_TYPE(astnode, CYPHER_AST_NODE_INDEX_LOOKUP, NULL);
    struct node_index_lookup *node =
            container_of(astnode, struct node_index_lookup, _astnode);
    return node->identifier;
}


const cypher_astnode_t *cypher_ast_node_index_lookup_get_index_name(
        const cypher_astnode_t *astnode)
{
    REQUIRE_TYPE(astnode, CYPHER_AST_NODE_INDEX_LOOKUP, NULL);
    struct node_index_lookup *node =
            container_of(astnode, struct node_index_lookup, _astnode);
    return node->index_name;
}


const cypher_astnode_t *cypher_ast_node_index_lookup_get_prop_name(
        const cypher_astnode_t *astnode)
{
    REQUIRE_TYPE(astnode, CYPHER_AST_NODE_INDEX_LOOKUP, NULL);
    struct node_index_lookup *node =
            container_of(astnode, struct node_index_lookup, _astnode);
    return node->prop_name;
}


const cypher_astnode_t *cypher_ast_node_index_lookup_get_lookup(
        const cypher_astnode_t *astnode)
{
    REQUIRE_TYPE(astnode, CYPHER_AST_NODE_INDEX_LOOKUP, NULL);
    struct node_index_lookup *node =
            container_of(astnode, struct node_index_lookup, _astnode);
    return node->lookup;
}


ssize_t detailstr(const cypher_astnode_t *self, char *str, size_t size)
{
    REQUIRE_TYPE(self, CYPHER_AST_NODE_INDEX_LOOKUP, -1);
    struct node_index_lookup *node =
            container_of(self, struct node_index_lookup, _astnode);
    return snprintf(str, size, "@%u = node:@%u(@%u = @%u)",
                node->identifier->ordinal, node->index_name->ordinal,
                node->prop_name->ordinal, node->lookup->ordinal);
}
