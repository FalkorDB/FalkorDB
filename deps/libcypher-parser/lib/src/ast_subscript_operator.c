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


struct subscript_operator
{
    cypher_astnode_t _astnode;
    const cypher_astnode_t *expression;
    const cypher_astnode_t *subscript;
};


static cypher_astnode_t *clone(const cypher_astnode_t *self,
        cypher_astnode_t **children);
static ssize_t detailstr(const cypher_astnode_t *self, char *str, size_t size);


static const struct cypher_astnode_vt *parents[] =
    { &cypher_expression_astnode_vt };

const struct cypher_astnode_vt cypher_subscript_operator_astnode_vt =
    { .parents = parents,
      .nparents = 1,
      .name = "subscript",
      .detailstr = detailstr,
      .release = cypher_astnode_release,
      .clone = clone };


cypher_astnode_t *cypher_ast_subscript_operator(
        const cypher_astnode_t *expression, const cypher_astnode_t *subscript,
        cypher_astnode_t **children, unsigned int nchildren,
        struct cypher_input_range range)
{
    REQUIRE_CHILD(children, nchildren, expression, CYPHER_AST_EXPRESSION, NULL);
    REQUIRE_CHILD(children, nchildren, subscript, CYPHER_AST_EXPRESSION, NULL);

    struct subscript_operator *node =
            calloc(1, sizeof(struct subscript_operator));
    if (node == NULL)
    {
        return NULL;
    }
    if (cypher_astnode_init(&(node->_astnode), CYPHER_AST_SUBSCRIPT_OPERATOR,
            children, nchildren, range))
    {
        free(node);
        return NULL;
    }
    node->expression = expression;
    node->subscript = subscript;
    return &(node->_astnode);
}


cypher_astnode_t *clone(const cypher_astnode_t *self,
        cypher_astnode_t **children)
{
    REQUIRE_TYPE(self, CYPHER_AST_SUBSCRIPT_OPERATOR, NULL);
    struct subscript_operator *node =
            container_of(self, struct subscript_operator, _astnode);

    cypher_astnode_t *expression = children[child_index(self, node->expression)];
    cypher_astnode_t *subscript = children[child_index(self, node->subscript)];

    return cypher_ast_subscript_operator(expression, subscript, children,
            self->nchildren, self->range);
}


const cypher_astnode_t *cypher_ast_subscript_operator_get_expression(
        const cypher_astnode_t *astnode)
{
    REQUIRE_TYPE(astnode, CYPHER_AST_SUBSCRIPT_OPERATOR, NULL);
    struct subscript_operator *node =
            container_of(astnode, struct subscript_operator, _astnode);
    return node->expression;
}


const cypher_astnode_t *cypher_ast_subscript_operator_get_subscript(
        const cypher_astnode_t *astnode)
{
    REQUIRE_TYPE(astnode, CYPHER_AST_SUBSCRIPT_OPERATOR, NULL);
    struct subscript_operator *node =
            container_of(astnode, struct subscript_operator, _astnode);
    return node->subscript;
}


ssize_t detailstr(const cypher_astnode_t *self, char *str, size_t size)
{
    REQUIRE_TYPE(self, CYPHER_AST_SUBSCRIPT_OPERATOR, -1);
    struct subscript_operator *node =
            container_of(self, struct subscript_operator, _astnode);
    return snprintf(str, size, "@%u[@%u]", node->expression->ordinal,
                node->subscript->ordinal);
}
