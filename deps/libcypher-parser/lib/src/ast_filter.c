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


struct filter
{
    cypher_list_comprehension_astnode_t _list_comprehension_astnode;
    const cypher_astnode_t *identifier;
    const cypher_astnode_t *expression;
    const cypher_astnode_t *predicate;
};


static cypher_astnode_t *clone(const cypher_astnode_t *self,
        cypher_astnode_t **children);
static ssize_t detailstr(const cypher_astnode_t *self, char *str, size_t size);
static const cypher_astnode_t *get_identifier(
        const cypher_list_comprehension_astnode_t *self);
static const cypher_astnode_t *get_expression(
        const cypher_list_comprehension_astnode_t *self);
static const cypher_astnode_t *get_predicate(
        const cypher_list_comprehension_astnode_t *self);
static const cypher_astnode_t *get_eval(
        const cypher_list_comprehension_astnode_t *self);


static const struct cypher_astnode_vt *parents[] =
    { &cypher_list_comprehension_astnode_vt };

const struct cypher_astnode_vt cypher_filter_astnode_vt =
    { .parents = parents,
      .nparents = 1,
      .name = "filter",
      .detailstr = detailstr,
      .release = cypher_astnode_release,
      .clone = clone };

static const struct cypher_list_comprehension_astnode_vt lc_vt =
    { .get_identifier = get_identifier,
      .get_expression = get_expression,
      .get_predicate = get_predicate,
      .get_eval = get_eval };


cypher_astnode_t *cypher_ast_filter(const cypher_astnode_t *identifier,
        const cypher_astnode_t *expression, const cypher_astnode_t *predicate,
        cypher_astnode_t **children, unsigned int nchildren,
        struct cypher_input_range range)
{
    REQUIRE_CHILD(children, nchildren, identifier, CYPHER_AST_IDENTIFIER, NULL);
    REQUIRE_CHILD(children, nchildren, expression, CYPHER_AST_EXPRESSION, NULL);
    REQUIRE_CHILD_OPTIONAL(children, nchildren, predicate,
            CYPHER_AST_EXPRESSION, NULL);

    struct filter *node = calloc(1, sizeof(struct filter));
    if (node == NULL)
    {
        return NULL;
    }
    if (cypher_list_comprehension_astnode_init(&(node->_list_comprehension_astnode),
            CYPHER_AST_FILTER, &lc_vt, children, nchildren, range))
    {
        free(node);
        return NULL;
    }
    node->identifier = identifier;
    node->expression = expression;
    node->predicate = predicate;
    return &(node->_list_comprehension_astnode._astnode);
}


cypher_astnode_t *clone(const cypher_astnode_t *self,
        cypher_astnode_t **children)
{
    REQUIRE_TYPE(self, CYPHER_AST_FILTER, NULL);
    const cypher_list_comprehension_astnode_t *lcnode =
            container_of(self, cypher_list_comprehension_astnode_t, _astnode);
    struct filter *node =
            container_of(lcnode, struct filter, _list_comprehension_astnode);

    cypher_astnode_t *identifier = children[child_index(self, node->identifier)];
    cypher_astnode_t *expression = children[child_index(self, node->expression)];
    cypher_astnode_t *predicate = (node->predicate == NULL) ? NULL :
            children[child_index(self, node->predicate)];

    return cypher_ast_filter(identifier, expression, predicate, children,
            self->nchildren, self->range);
}


const cypher_astnode_t *get_identifier(const cypher_list_comprehension_astnode_t *self)
{
    struct filter *node =
        container_of(self, struct filter, _list_comprehension_astnode);
    return node->identifier;
}


const cypher_astnode_t *get_expression(const cypher_list_comprehension_astnode_t *self)
{
    struct filter *node =
        container_of(self, struct filter, _list_comprehension_astnode);
    return node->expression;
}


const cypher_astnode_t *get_predicate(const cypher_list_comprehension_astnode_t *self)
{
    struct filter *node =
        container_of(self, struct filter, _list_comprehension_astnode);
    return node->predicate;
}


const cypher_astnode_t *get_eval(const cypher_list_comprehension_astnode_t *self)
{
    return NULL;
}


ssize_t detailstr(const cypher_astnode_t *self, char *str, size_t size)
{
    REQUIRE_TYPE(self, CYPHER_AST_FILTER, -1);
    const cypher_list_comprehension_astnode_t *lcnode =
            container_of(self, cypher_list_comprehension_astnode_t, _astnode);
    struct filter *node =
            container_of(lcnode, struct filter, _list_comprehension_astnode);

    size_t n = 0;
    ssize_t r = snprintf(str, size, "[@%u IN @%u",
            node->identifier->ordinal,
            node->expression->ordinal);
    if (r < 0)
    {
        return -1;
    }
    n += r;

    if (node->predicate != NULL)
    {
        r = snprintf(str+n, (n < size)? size-n : 0, " WHERE @%u",
                node->predicate->ordinal);
        if (r < 0)
        {
            return -1;
        }
        n += r;
    }

    if (n+1 < size)
    {
        str[n] = ']';
        str[n+1] = '\0';
    }

    return n+1;
}
