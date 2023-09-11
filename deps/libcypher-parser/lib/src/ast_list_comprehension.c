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


int cypher_list_comprehension_astnode_init(
        cypher_list_comprehension_astnode_t *node,
        cypher_astnode_type_t type,
        const struct cypher_list_comprehension_astnode_vt *vt,
        cypher_astnode_t **children, unsigned int nchildren,
        struct cypher_input_range range)
{
    assert(vt != NULL);
    if (cypher_astnode_init(&(node->_astnode),type, children, nchildren, range))
    {
        return -1;
    }
    node->_vt = vt;
    return 0;
}


struct list_comprehension
{
    cypher_list_comprehension_astnode_t _list_comprehension_astnode;
    const cypher_astnode_t *identifier;
    const cypher_astnode_t *expression;
    const cypher_astnode_t *predicate;
    const cypher_astnode_t *eval;
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
    { &cypher_expression_astnode_vt };

const struct cypher_astnode_vt cypher_list_comprehension_astnode_vt =
    { .parents = parents,
      .nparents = 1,
      .name = "list comprehension",
      .detailstr = detailstr,
      .release = cypher_astnode_release,
      .clone = clone };

static const struct cypher_list_comprehension_astnode_vt lc_vt =
    { .get_identifier = get_identifier,
      .get_expression = get_expression,
      .get_predicate = get_predicate,
      .get_eval = get_eval };


cypher_astnode_t *cypher_ast_list_comprehension(
        const cypher_astnode_t *identifier, const cypher_astnode_t *expression,
        const cypher_astnode_t *predicate, const cypher_astnode_t *eval,
        cypher_astnode_t **children, unsigned int nchildren,
        struct cypher_input_range range)
{
    REQUIRE_CHILD(children, nchildren, identifier, CYPHER_AST_IDENTIFIER, NULL);
    REQUIRE_CHILD(children, nchildren, expression, CYPHER_AST_EXPRESSION, NULL);
    REQUIRE_CHILD_OPTIONAL(children, nchildren, predicate,
            CYPHER_AST_EXPRESSION, NULL);
    REQUIRE_CHILD_OPTIONAL(children, nchildren, eval,
            CYPHER_AST_EXPRESSION, NULL);

    struct list_comprehension *node =
            calloc(1, sizeof(struct list_comprehension));
    if (node == NULL)
    {
        return NULL;
    }
    if (cypher_list_comprehension_astnode_init(&(node->_list_comprehension_astnode),
            CYPHER_AST_LIST_COMPREHENSION, &lc_vt, children, nchildren, range))
    {
        goto cleanup;
    }
    assert(node->_list_comprehension_astnode._vt->get_identifier != NULL);
    node->identifier = identifier;
    node->expression = expression;
    node->predicate = predicate;
    node->eval = eval;
    return &(node->_list_comprehension_astnode._astnode);

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
    REQUIRE_TYPE(self, CYPHER_AST_LIST_COMPREHENSION, NULL);
    const cypher_list_comprehension_astnode_t *lcnode =
            container_of(self, cypher_list_comprehension_astnode_t, _astnode);
    struct list_comprehension *node =
        container_of(lcnode, struct list_comprehension, _list_comprehension_astnode);

    cypher_astnode_t *identifier = children[child_index(self, node->identifier)];
    cypher_astnode_t *expression = children[child_index(self, node->expression)];
    cypher_astnode_t *predicate = (node->predicate == NULL) ? NULL :
        children[child_index(self, node->predicate)];
    cypher_astnode_t *eval = (node->eval == NULL) ? NULL :
        children[child_index(self, node->eval)];

    return cypher_ast_list_comprehension(identifier, expression, predicate,
            eval, children, self->nchildren, self->range);
}


const cypher_astnode_t *cypher_ast_list_comprehension_get_identifier(
        const cypher_astnode_t *astnode)
{
    REQUIRE_TYPE(astnode, CYPHER_AST_LIST_COMPREHENSION, NULL);
    const cypher_list_comprehension_astnode_t *lcnode =
            container_of(astnode, cypher_list_comprehension_astnode_t, _astnode);
    assert(lcnode->_vt != NULL);
    assert(lcnode->_vt->get_identifier != NULL);
    return lcnode->_vt->get_identifier(lcnode);
}


const cypher_astnode_t *cypher_ast_list_comprehension_get_expression(
        const cypher_astnode_t *astnode)
{
    REQUIRE_TYPE(astnode, CYPHER_AST_LIST_COMPREHENSION, NULL);
    const cypher_list_comprehension_astnode_t *lcnode =
            container_of(astnode, cypher_list_comprehension_astnode_t, _astnode);
    assert(lcnode->_vt != NULL && lcnode->_vt->get_expression != NULL);
    return lcnode->_vt->get_expression(lcnode);
}


const cypher_astnode_t *cypher_ast_list_comprehension_get_predicate(
        const cypher_astnode_t *astnode)
{
    REQUIRE_TYPE(astnode, CYPHER_AST_LIST_COMPREHENSION, NULL);
    const cypher_list_comprehension_astnode_t *lcnode =
            container_of(astnode, cypher_list_comprehension_astnode_t, _astnode);
    assert(lcnode->_vt != NULL && lcnode->_vt->get_predicate != NULL);
    return lcnode->_vt->get_predicate(lcnode);
}


const cypher_astnode_t *cypher_ast_list_comprehension_get_eval(
        const cypher_astnode_t *astnode)
{
    REQUIRE_TYPE(astnode, CYPHER_AST_LIST_COMPREHENSION, NULL);
    const cypher_list_comprehension_astnode_t *lcnode =
            container_of(astnode, cypher_list_comprehension_astnode_t, _astnode);
    assert(lcnode->_vt != NULL && lcnode->_vt->get_eval != NULL);
    return lcnode->_vt->get_eval(lcnode);
}


const cypher_astnode_t *get_identifier(const cypher_list_comprehension_astnode_t *self)
{
    struct list_comprehension *node =
        container_of(self, struct list_comprehension, _list_comprehension_astnode);
    return node->identifier;
}


const cypher_astnode_t *get_expression(const cypher_list_comprehension_astnode_t *self)
{
    struct list_comprehension *node =
        container_of(self, struct list_comprehension, _list_comprehension_astnode);
    return node->expression;
}


const cypher_astnode_t *get_predicate(const cypher_list_comprehension_astnode_t *self)
{
    struct list_comprehension *node =
        container_of(self, struct list_comprehension, _list_comprehension_astnode);
    return node->predicate;
}


const cypher_astnode_t *get_eval(const cypher_list_comprehension_astnode_t *self)
{
    struct list_comprehension *node =
        container_of(self, struct list_comprehension, _list_comprehension_astnode);
    return node->eval;
}


ssize_t detailstr(const cypher_astnode_t *self, char *str, size_t size)
{
    REQUIRE_TYPE(self, CYPHER_AST_LIST_COMPREHENSION, -1);
    const cypher_list_comprehension_astnode_t *lcnode =
            container_of(self, cypher_list_comprehension_astnode_t, _astnode);
    struct list_comprehension *node =
        container_of(lcnode, struct list_comprehension, _list_comprehension_astnode);

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

    if (node->eval != NULL)
    {
        r = snprintf(str+n, (n < size)? size-n : 0, " | @%u",
                node->eval->ordinal);
        if (r < 0)
        {
            return -1;
        }
        n += r;
    }

    if (n < size)
    {
        str[n] = ']';
    }
    n++;
    return n;
}
