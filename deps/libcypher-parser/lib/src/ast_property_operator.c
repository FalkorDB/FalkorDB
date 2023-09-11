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


struct property_operator
{
    cypher_astnode_t _astnode;
    const cypher_astnode_t *expression;
    const cypher_astnode_t *prop_name;
};


static cypher_astnode_t *clone(const cypher_astnode_t *self,
        cypher_astnode_t **children);
static ssize_t detailstr(const cypher_astnode_t *self, char *str, size_t size);


static const struct cypher_astnode_vt *parents[] =
    { &cypher_expression_astnode_vt };

const struct cypher_astnode_vt cypher_property_operator_astnode_vt =
    { .parents = parents,
      .nparents = 1,
      .name = "property",
      .detailstr = detailstr,
      .release = cypher_astnode_release,
      .clone = clone };


cypher_astnode_t *cypher_ast_property_operator(
        const cypher_astnode_t *expression, const cypher_astnode_t *prop_name,
        cypher_astnode_t **children, unsigned int nchildren,
        struct cypher_input_range range)
{
    REQUIRE_CHILD(children, nchildren, expression, CYPHER_AST_EXPRESSION, NULL);
    REQUIRE_CHILD(children, nchildren, prop_name, CYPHER_AST_PROP_NAME, NULL);

    struct property_operator *node =
            calloc(1, sizeof(struct property_operator));
    if (node == NULL)
    {
        return NULL;
    }
    if (cypher_astnode_init(&(node->_astnode), CYPHER_AST_PROPERTY_OPERATOR,
                children, nchildren, range))
    {
        free(node);
        return NULL;
    }
    node->expression = expression;
    node->prop_name = prop_name;
    return &(node->_astnode);
}


cypher_astnode_t *clone(const cypher_astnode_t *self,
        cypher_astnode_t **children)
{
    REQUIRE_TYPE(self, CYPHER_AST_PROPERTY_OPERATOR, NULL);
    struct property_operator *node =
            container_of(self, struct property_operator, _astnode);

    cypher_astnode_t *expression = children[child_index(self, node->expression)];
    cypher_astnode_t *prop_name = children[child_index(self, node->prop_name)];

    return cypher_ast_property_operator(expression, prop_name, children,
            self->nchildren, self->range);
}


const cypher_astnode_t *cypher_ast_property_operator_get_expression(
                const cypher_astnode_t *astnode)
{
    REQUIRE_TYPE(astnode, CYPHER_AST_PROPERTY_OPERATOR, NULL);
    struct property_operator *node =
            container_of(astnode, struct property_operator, _astnode);
    return node->expression;
}


const cypher_astnode_t *cypher_ast_property_operator_get_prop_name(
                const cypher_astnode_t *astnode)
{
    REQUIRE_TYPE(astnode, CYPHER_AST_PROPERTY_OPERATOR, NULL);
    struct property_operator *node =
            container_of(astnode, struct property_operator, _astnode);
    return node->prop_name;
}


ssize_t detailstr(const cypher_astnode_t *self, char *str, size_t size)
{
    REQUIRE_TYPE(self, CYPHER_AST_PROPERTY_OPERATOR, -1);
    struct property_operator *node =
        container_of(self, struct property_operator, _astnode);
    return snprintf(str, size, "@%u.@%u", node->expression->ordinal,
                node->prop_name->ordinal);
}
