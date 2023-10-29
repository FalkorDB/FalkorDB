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


struct labels_operator
{
    cypher_astnode_t _astnode;
    const cypher_astnode_t *expression;
    unsigned int nlabels;
    const cypher_astnode_t *labels[];
};


static cypher_astnode_t *clone(const cypher_astnode_t *self,
        cypher_astnode_t **children);
static ssize_t detailstr(const cypher_astnode_t *self, char *str, size_t size);


static const struct cypher_astnode_vt *parents[] =
    { &cypher_expression_astnode_vt };

const struct cypher_astnode_vt cypher_labels_operator_astnode_vt =
    { .parents = parents,
      .nparents = 1,
      .name = "has labels",
      .detailstr = detailstr,
      .release = cypher_astnode_release,
      .clone = clone };


cypher_astnode_t *cypher_ast_labels_operator(const cypher_astnode_t *expression,
        cypher_astnode_t * const *labels, unsigned int nlabels,
        cypher_astnode_t **children, unsigned int nchildren,
        struct cypher_input_range range)
{
    REQUIRE_CHILD(children, nchildren, expression, CYPHER_AST_EXPRESSION, NULL);
    REQUIRE(nlabels > 0, NULL);
    REQUIRE_CHILD_ALL(children, nchildren, labels, nlabels,
            CYPHER_AST_LABEL, NULL);

    struct labels_operator *node = calloc(1, sizeof(struct labels_operator) +
            nlabels * sizeof(cypher_astnode_t *));
    if (node == NULL)
    {
        return NULL;
    }
    if (cypher_astnode_init(&(node->_astnode), CYPHER_AST_LABELS_OPERATOR,
            children, nchildren, range))
    {
        goto cleanup;
    }
    node->expression = expression;
    memcpy(node->labels, labels, nlabels * sizeof(cypher_astnode_t *));
    node->nlabels = nlabels;
    return &(node->_astnode);

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
    REQUIRE_TYPE(self, CYPHER_AST_LABELS_OPERATOR, NULL);
    struct labels_operator *node =
        container_of(self, struct labels_operator, _astnode);

    cypher_astnode_t *expression = children[child_index(self, node->expression)];
    cypher_astnode_t **labels = calloc(node->nlabels,
            sizeof(cypher_astnode_t *));
    if (labels == NULL)
    {
        return NULL;
    }
    for (unsigned int i = 0; i < node->nlabels; ++i)
    {
        labels[i] = children[child_index(self, node->labels[i])];
    }

    cypher_astnode_t *clone = cypher_ast_labels_operator(expression,
            labels, node->nlabels, children, self->nchildren, self->range);
    int errsv = errno;
    free(labels);
    errno = errsv;
    return clone;
}


const cypher_astnode_t *cypher_ast_labels_operator_get_expression(
        const cypher_astnode_t *astnode)
{
    REQUIRE_TYPE(astnode, CYPHER_AST_LABELS_OPERATOR, NULL);
    struct labels_operator *node =
        container_of(astnode, struct labels_operator, _astnode);
    return node->expression;
}


unsigned int cypher_ast_labels_operator_nlabels(const cypher_astnode_t *astnode)
{
    REQUIRE_TYPE(astnode, CYPHER_AST_LABELS_OPERATOR, 0);
    struct labels_operator *node =
        container_of(astnode, struct labels_operator, _astnode);
    return node->nlabels;
}


const cypher_astnode_t *cypher_ast_labels_operator_get_label(
        const cypher_astnode_t *astnode, unsigned int index)
{
    REQUIRE_TYPE(astnode, CYPHER_AST_LABELS_OPERATOR, NULL);
    struct labels_operator *node =
        container_of(astnode, struct labels_operator, _astnode);
    if (index >= node->nlabels)
    {
        return NULL;
    }
    return node->labels[index];
}


ssize_t detailstr(const cypher_astnode_t *self, char *str, size_t size)
{
    REQUIRE_TYPE(self, CYPHER_AST_LABELS_OPERATOR, -1);
    struct labels_operator *node =
        container_of(self, struct labels_operator, _astnode);

    size_t n = 0;
    ssize_t r = snprintf(str, size, "@%u", node->expression->ordinal);
    if (r < 0)
    {
        return -1;
    }
    n += r;

    for (unsigned int l = 0; l < node->nlabels; ++l)
    {
        r = snprintf(str + n, (n < size)? size-n : 0, ":@%u",
                node->labels[l]->ordinal);
        if (r < 0)
        {
            return -1;
        }
        n += r;
    }
    return n;
}
