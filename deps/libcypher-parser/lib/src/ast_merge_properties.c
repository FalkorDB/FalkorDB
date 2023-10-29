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


struct merge_properties
{
    cypher_astnode_t _astnode;
    const cypher_astnode_t *identifier;
    const cypher_astnode_t *expression;
};


static cypher_astnode_t *clone(const cypher_astnode_t *self,
        cypher_astnode_t **children);
static ssize_t detailstr(const cypher_astnode_t *self, char *str, size_t size);


static const struct cypher_astnode_vt *parents[] =
    { &cypher_set_item_astnode_vt };

const struct cypher_astnode_vt cypher_merge_properties_astnode_vt =
    { .parents = parents,
      .nparents = 1,
      .name = "merge properties",
      .detailstr = detailstr,
      .release = cypher_astnode_release,
      .clone = clone };


cypher_astnode_t *cypher_ast_merge_properties(
        const cypher_astnode_t *identifier, const cypher_astnode_t *expression,
        cypher_astnode_t **children, unsigned int nchildren,
        struct cypher_input_range range)
{
    REQUIRE_CHILD(children, nchildren, identifier, CYPHER_AST_IDENTIFIER, NULL);
    REQUIRE_CHILD(children, nchildren, expression, CYPHER_AST_EXPRESSION, NULL);

    struct merge_properties *node = calloc(1, sizeof(struct merge_properties));
    if (node == NULL)
    {
        return NULL;
    }
    if (cypher_astnode_init(&(node->_astnode), CYPHER_AST_MERGE_PROPERTIES,
            children, nchildren, range))
    {
        free(node);
        return NULL;
    }
    node->identifier = identifier;
    node->expression = expression;
    return &(node->_astnode);
}


cypher_astnode_t *clone(const cypher_astnode_t *self,
        cypher_astnode_t **children)
{
    REQUIRE_TYPE(self, CYPHER_AST_MERGE_PROPERTIES, NULL);
    struct merge_properties *node =
            container_of(self, struct merge_properties, _astnode);

    cypher_astnode_t *identifier = children[child_index(self, node->identifier)];
    cypher_astnode_t *expression = children[child_index(self, node->expression)];

    return cypher_ast_merge_properties(identifier, expression, children,
            self->nchildren, self->range);
}


const cypher_astnode_t *cypher_ast_merge_properties_get_identifier(
        const cypher_astnode_t *astnode)
{
    REQUIRE_TYPE(astnode, CYPHER_AST_MERGE_PROPERTIES, NULL);
    struct merge_properties *node =
            container_of(astnode, struct merge_properties, _astnode);
    return node->identifier;
}


const cypher_astnode_t *cypher_ast_merge_properties_get_expression(
        const cypher_astnode_t *astnode)
{
    REQUIRE_TYPE(astnode, CYPHER_AST_MERGE_PROPERTIES, NULL);
    struct merge_properties *node =
            container_of(astnode, struct merge_properties, _astnode);
    return node->expression;
}


ssize_t detailstr(const cypher_astnode_t *self, char *str, size_t size)
{
    REQUIRE_TYPE(self, CYPHER_AST_MERGE_PROPERTIES, -1);
    struct merge_properties *node =
            container_of(self, struct merge_properties, _astnode);

    return snprintf(str, size, "@%u += @%u", node->identifier->ordinal,
            node->expression->ordinal);
}
