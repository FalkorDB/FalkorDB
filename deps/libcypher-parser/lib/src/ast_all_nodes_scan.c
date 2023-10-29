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


struct all_nodes_scan
{
    cypher_astnode_t _astnode;
    const cypher_astnode_t *identifier;
};


static cypher_astnode_t *clone(const cypher_astnode_t *self,
        cypher_astnode_t **children);
static ssize_t detailstr(const cypher_astnode_t *self, char *str, size_t size);


static const struct cypher_astnode_vt *parents[] =
    { &cypher_start_point_astnode_vt };

const struct cypher_astnode_vt cypher_all_nodes_scan_astnode_vt =
    { .parents = parents,
      .nparents = 1,
      .name = "all nodes scan",
      .detailstr = detailstr,
      .release = cypher_astnode_release,
      .clone = clone };


cypher_astnode_t *cypher_ast_all_nodes_scan(const cypher_astnode_t *identifier,
        cypher_astnode_t **children, unsigned int nchildren,
        struct cypher_input_range range)
{
    REQUIRE_CHILD(children, nchildren, identifier, CYPHER_AST_IDENTIFIER, NULL);

    struct all_nodes_scan *node = calloc(1, sizeof(struct all_nodes_scan));
    if (node == NULL)
    {
        return NULL;
    }
    if (cypher_astnode_init(&(node->_astnode), CYPHER_AST_ALL_NODES_SCAN,
            children, nchildren, range))
    {
        free(node);
        return NULL;
    }
    node->identifier = identifier;
    return &(node->_astnode);
}


cypher_astnode_t *clone(const cypher_astnode_t *self,
        cypher_astnode_t **children)
{
    REQUIRE_TYPE(self, CYPHER_AST_ALL_NODES_SCAN, NULL);
    struct all_nodes_scan *node =
            container_of(self, struct all_nodes_scan, _astnode);

    cypher_astnode_t *identifier = children[child_index(self, node->identifier)];

    return cypher_ast_all_nodes_scan(identifier, children, self->nchildren,
            self->range);
}


const cypher_astnode_t *cypher_ast_all_nodes_scan_get_identifier(
        const cypher_astnode_t *astnode)
{
    REQUIRE_TYPE(astnode, CYPHER_AST_ALL_NODES_SCAN, NULL);
    struct all_nodes_scan *node =
            container_of(astnode, struct all_nodes_scan, _astnode);
    return node->identifier;
}


ssize_t detailstr(const cypher_astnode_t *self, char *str, size_t size)
{
    REQUIRE_TYPE(self, CYPHER_AST_ALL_NODES_SCAN, -1);
    struct all_nodes_scan *node =
            container_of(self, struct all_nodes_scan, _astnode);
    return snprintf(str, size, "identifier=@%u", node->identifier->ordinal);
}
