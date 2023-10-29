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


struct sort_item
{
    cypher_astnode_t _astnode;
    const cypher_astnode_t *expression;
    bool ascending;
};


static cypher_astnode_t *clone(const cypher_astnode_t *self,
        cypher_astnode_t **children);
static ssize_t detailstr(const cypher_astnode_t *self, char *str, size_t size);


const struct cypher_astnode_vt cypher_sort_item_astnode_vt =
    { .name = "sort item",
      .detailstr = detailstr,
      .release = cypher_astnode_release,
      .clone = clone };


cypher_astnode_t *cypher_ast_sort_item(const cypher_astnode_t *expression,
        bool ascending, cypher_astnode_t **children, unsigned int nchildren,
        struct cypher_input_range range)
{
    REQUIRE_CHILD(children, nchildren, expression, CYPHER_AST_EXPRESSION, NULL);

    struct sort_item *node = calloc(1, sizeof(struct sort_item));
    if (node == NULL)
    {
        return NULL;
    }
    if (cypher_astnode_init(&(node->_astnode), CYPHER_AST_SORT_ITEM,
            children, nchildren, range))
    {
        free(node);
        return NULL;
    }
    node->expression = expression;
    node->ascending = ascending;
    return &(node->_astnode);
}


cypher_astnode_t *clone(const cypher_astnode_t *self,
        cypher_astnode_t **children)
{
    REQUIRE_TYPE(self, CYPHER_AST_SORT_ITEM, NULL);
    struct sort_item *node = container_of(self, struct sort_item, _astnode);

    cypher_astnode_t *expression = children[child_index(self, node->expression)];

    return cypher_ast_sort_item(expression, node->ascending, children,
            self->nchildren, self->range);
}


const cypher_astnode_t *cypher_ast_sort_item_get_expression(
        const cypher_astnode_t *astnode)
{
    REQUIRE_TYPE(astnode, CYPHER_AST_SORT_ITEM, NULL);
    struct sort_item *node = container_of(astnode, struct sort_item, _astnode);
    return node->expression;
}


bool cypher_ast_sort_item_is_ascending(const cypher_astnode_t *astnode)
{
    REQUIRE_TYPE(astnode, CYPHER_AST_SORT_ITEM, false);
    struct sort_item *node = container_of(astnode, struct sort_item, _astnode);
    return node->ascending;
}


ssize_t detailstr(const cypher_astnode_t *self, char *str, size_t size)
{
    REQUIRE_TYPE(self, CYPHER_AST_SORT_ITEM, -1);
    struct sort_item *node = container_of(self, struct sort_item, _astnode);
    return snprintf(str, size, "expression=@%u, %s", node->expression->ordinal,
            node->ascending? "ASCENDING" : "DESCENDING");
}
