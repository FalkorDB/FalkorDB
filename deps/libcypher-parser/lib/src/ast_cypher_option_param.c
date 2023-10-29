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


struct cypher_option_param
{
    cypher_astnode_t _astnode;
    const cypher_astnode_t *name;
    const cypher_astnode_t *value;
};


static cypher_astnode_t *clone(const cypher_astnode_t *self,
        cypher_astnode_t **children);
static ssize_t detailstr(const cypher_astnode_t *self, char *str, size_t size);


const struct cypher_astnode_vt cypher_cypher_option_param_astnode_vt =
    { .name = "cypher parameter",
      .detailstr = detailstr,
      .release = cypher_astnode_release,
      .clone = clone };


cypher_astnode_t *cypher_ast_cypher_option_param(const cypher_astnode_t *name,
        const cypher_astnode_t *value, cypher_astnode_t **children,
        unsigned int nchildren, struct cypher_input_range range)
{
    REQUIRE_CHILD(children, nchildren, name, CYPHER_AST_STRING, NULL);

    struct cypher_option_param *node =
            calloc(1, sizeof(struct cypher_option_param));
    if (node == NULL)
    {
        return NULL;
    }
    if (cypher_astnode_init(&(node->_astnode), CYPHER_AST_CYPHER_OPTION_PARAM,
            children, nchildren, range))
    {
        free(node);
        return NULL;
    }
    node->name = name;
    node->value = value;
    return &(node->_astnode);
}


cypher_astnode_t *clone(const cypher_astnode_t *self,
        cypher_astnode_t **children)
{
    REQUIRE_TYPE(self, CYPHER_AST_CYPHER_OPTION_PARAM, NULL);
    struct cypher_option_param *node = container_of(self,
            struct cypher_option_param, _astnode);

    cypher_astnode_t *name = children[child_index(self, node->name)];
    cypher_astnode_t *value = children[child_index(self, node->value)];

    return cypher_ast_cypher_option_param(name, value, children,
            self->nchildren, self->range);
}


const cypher_astnode_t *cypher_ast_cypher_option_param_get_name(
        const cypher_astnode_t *astnode)
{
    REQUIRE_TYPE(astnode, CYPHER_AST_CYPHER_OPTION_PARAM, NULL);
    struct cypher_option_param *node = container_of(astnode,
            struct cypher_option_param, _astnode);
    return node->name;
}


const cypher_astnode_t *cypher_ast_cypher_option_param_get_value(
        const cypher_astnode_t *astnode)
{
    REQUIRE_TYPE(astnode, CYPHER_AST_CYPHER_OPTION_PARAM, NULL);
    struct cypher_option_param *node = container_of(astnode,
            struct cypher_option_param, _astnode);
    return node->value;
}


ssize_t detailstr(const cypher_astnode_t *self, char *str, size_t size)
{
    REQUIRE_TYPE(self, CYPHER_AST_CYPHER_OPTION_PARAM, -1);
    struct cypher_option_param *node =
        container_of(self, struct cypher_option_param, _astnode);
    return snprintf(str, size, "@%u = @%u", node->name->ordinal,
                node->value->ordinal);
}
