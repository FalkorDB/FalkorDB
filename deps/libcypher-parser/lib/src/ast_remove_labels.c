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


struct remove_labels
{
    cypher_astnode_t _astnode;
    const cypher_astnode_t *identifier;
    unsigned int nlabels;
    const cypher_astnode_t *labels[];
};


static cypher_astnode_t *clone(const cypher_astnode_t *self,
        cypher_astnode_t **children);
static ssize_t detailstr(const cypher_astnode_t *self, char *str, size_t size);


static const struct cypher_astnode_vt *parents[] =
    { &cypher_remove_item_astnode_vt };

const struct cypher_astnode_vt cypher_remove_labels_astnode_vt =
    { .parents = parents,
      .nparents = 1,
      .name = "remove labels",
      .detailstr = detailstr,
      .release = cypher_astnode_release,
      .clone = clone };


cypher_astnode_t *cypher_ast_remove_labels(const cypher_astnode_t *identifier,
        cypher_astnode_t * const *labels, unsigned int nlabels,
        cypher_astnode_t **children, unsigned int nchildren,
        struct cypher_input_range range)
{
    REQUIRE_CHILD(children, nchildren, identifier, CYPHER_AST_IDENTIFIER, NULL);
    REQUIRE(nlabels > 0, NULL);
    REQUIRE_CHILD_ALL(children, nchildren, labels, nlabels,
            CYPHER_AST_LABEL, NULL);

    struct remove_labels *node = calloc(1, sizeof(struct remove_labels) +
            nlabels * sizeof(cypher_astnode_t *));
    if (node == NULL)
    {
        return NULL;
    }
    if (cypher_astnode_init(&(node->_astnode), CYPHER_AST_REMOVE_LABELS,
            children, nchildren, range))
    {
        goto cleanup;
    }
    node->identifier = identifier;
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
    REQUIRE_TYPE(self, CYPHER_AST_REMOVE_LABELS, NULL);
    struct remove_labels *node =
            container_of(self, struct remove_labels, _astnode);

    cypher_astnode_t *identifier = children[child_index(self, node->identifier)];
    cypher_astnode_t **labels =
            calloc(node->nlabels, sizeof(cypher_astnode_t *));
    if (labels == NULL)
    {
        return NULL;
    }
    for (unsigned int i = 0; i < node->nlabels; ++i)
    {
        labels[i] = children[child_index(self, node->labels[i])];
    }

    cypher_astnode_t *clone = cypher_ast_remove_labels(identifier,
            labels, node->nlabels, children, self->nchildren, self->range);
    int errsv = errno;
    free(labels);
    errno = errsv;
    return clone;
}


const cypher_astnode_t *cypher_ast_remove_labels_get_identifier(
        const cypher_astnode_t *astnode)
{
    REQUIRE_TYPE(astnode, CYPHER_AST_REMOVE_LABELS, NULL);
    struct remove_labels *node =
            container_of(astnode, struct remove_labels, _astnode);
    return node->identifier;
}


unsigned int cypher_ast_remove_labels_nlabels(const cypher_astnode_t *astnode)
{
    REQUIRE_TYPE(astnode, CYPHER_AST_REMOVE_LABELS, 0);
    struct remove_labels *node =
            container_of(astnode, struct remove_labels, _astnode);
    return node->nlabels;
}


const cypher_astnode_t *cypher_ast_remove_labels_get_label(
        const cypher_astnode_t *astnode, unsigned int index)
{
    REQUIRE_TYPE(astnode, CYPHER_AST_REMOVE_LABELS, NULL);
    struct remove_labels *node =
            container_of(astnode, struct remove_labels, _astnode);
    if (index >= node->nlabels)
    {
        return NULL;
    }
    return node->labels[index];
}


ssize_t detailstr(const cypher_astnode_t *self, char *str, size_t size)
{
    REQUIRE_TYPE(self, CYPHER_AST_REMOVE_LABELS, -1);
    struct remove_labels *node =
            container_of(self, struct remove_labels, _astnode);

    size_t n = 0;
    ssize_t r = snprintf(str, size, "@%u", node->identifier->ordinal);
    if (r < 0)
    {
        return -1;
    }
    n += r;

    for (unsigned int i = 0; i < node->nlabels; ++i)
    {
        r = snprintf(str+n, (n < size)? size-n : 0, ":@%u",
                node->labels[i]->ordinal);
        if (r < 0)
        {
            return -1;
        }
        n += r;
    }
    return n;
}
