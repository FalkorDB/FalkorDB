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


struct rel_index_query
{
    cypher_astnode_t _astnode;
    const cypher_astnode_t *identifier;
    const cypher_astnode_t *index_name;
    const cypher_astnode_t *query;
};


static cypher_astnode_t *clone(const cypher_astnode_t *self,
        cypher_astnode_t **children);
static ssize_t detailstr(const cypher_astnode_t *self, char *str, size_t size);


static const struct cypher_astnode_vt *parents[] =
    { &cypher_start_point_astnode_vt };

const struct cypher_astnode_vt cypher_rel_index_query_astnode_vt =
    { .parents = parents,
      .nparents = 1,
      .name = "rel index query",
      .detailstr = detailstr,
      .release = cypher_astnode_release,
      .clone = clone };


cypher_astnode_t *cypher_ast_rel_index_query(
        const cypher_astnode_t *identifier, const cypher_astnode_t *index_name,
        const cypher_astnode_t *query, cypher_astnode_t **children,
        unsigned int nchildren, struct cypher_input_range range)
{
    REQUIRE_CHILD(children, nchildren, identifier, CYPHER_AST_IDENTIFIER, NULL);
    REQUIRE_CHILD(children, nchildren, index_name, CYPHER_AST_INDEX_NAME, NULL);
    REQUIRE(cypher_astnode_instanceof(query, CYPHER_AST_STRING) ||
            cypher_astnode_instanceof(query, CYPHER_AST_PARAMETER), NULL);
    REQUIRE_CONTAINS(children, nchildren, query, NULL);

    struct rel_index_query *node = calloc(1, sizeof(struct rel_index_query));
    if (node == NULL)
    {
        return NULL;
    }
    if (cypher_astnode_init(&(node->_astnode), CYPHER_AST_REL_INDEX_QUERY,
            children, nchildren, range))
    {
        free(node);
        return NULL;
    }
    node->identifier = identifier;
    node->index_name = index_name;
    node->query = query;
    return &(node->_astnode);
}


cypher_astnode_t *clone(const cypher_astnode_t *self,
        cypher_astnode_t **children)
{
    REQUIRE_TYPE(self, CYPHER_AST_REL_INDEX_QUERY, NULL);
    struct rel_index_query *node =
            container_of(self, struct rel_index_query, _astnode);

    cypher_astnode_t *identifier = children[child_index(self, node->identifier)];
    cypher_astnode_t *index_name = children[child_index(self, node->index_name)];
    cypher_astnode_t *query = children[child_index(self, node->query)];

    return cypher_ast_rel_index_query(identifier, index_name, query, children,
            self->nchildren, self->range);
}


const cypher_astnode_t *cypher_ast_rel_index_query_get_identifier(
        const cypher_astnode_t *astnode)
{
    REQUIRE_TYPE(astnode, CYPHER_AST_REL_INDEX_QUERY, NULL);
    struct rel_index_query *node =
            container_of(astnode, struct rel_index_query, _astnode);
    return node->identifier;
}


const cypher_astnode_t *cypher_ast_rel_index_query_get_index_name(
        const cypher_astnode_t *astnode)
{
    REQUIRE_TYPE(astnode, CYPHER_AST_REL_INDEX_QUERY, NULL);
    struct rel_index_query *node =
            container_of(astnode, struct rel_index_query, _astnode);
    return node->index_name;
}


const cypher_astnode_t *cypher_ast_rel_index_query_get_query(
        const cypher_astnode_t *astnode)
{
    REQUIRE_TYPE(astnode, CYPHER_AST_REL_INDEX_QUERY, NULL);
    struct rel_index_query *node =
            container_of(astnode, struct rel_index_query, _astnode);
    return node->query;
}


ssize_t detailstr(const cypher_astnode_t *self, char *str, size_t size)
{
    REQUIRE_TYPE(self, CYPHER_AST_REL_INDEX_QUERY, -1);
    struct rel_index_query *node =
            container_of(self, struct rel_index_query, _astnode);
    return snprintf(str, size, "@%u = rel:@%u(@%u)",
                node->identifier->ordinal, node->index_name->ordinal,
                node->query->ordinal);
}
