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


struct using_index
{
    cypher_astnode_t _astnode;
    const cypher_astnode_t *identifier;
    const cypher_astnode_t *label;
    const cypher_astnode_t *prop_name;
};


static cypher_astnode_t *clone(const cypher_astnode_t *self,
        cypher_astnode_t **children);
static ssize_t detailstr(const cypher_astnode_t *self, char *str, size_t size);


static const struct cypher_astnode_vt *parents[] =
    { &cypher_match_hint_astnode_vt };

const struct cypher_astnode_vt cypher_using_index_astnode_vt =
    { .parents = parents,
      .nparents = 1,
      .name = "USING INDEX",
      .detailstr = detailstr,
      .release = cypher_astnode_release,
      .clone = clone };


cypher_astnode_t *cypher_ast_using_index(const cypher_astnode_t *identifier,
        const cypher_astnode_t *label, const cypher_astnode_t *prop_name,
        cypher_astnode_t **children, unsigned int nchildren,
        struct cypher_input_range range)
{
    REQUIRE_CHILD(children, nchildren, identifier, CYPHER_AST_IDENTIFIER, NULL);
    REQUIRE_CHILD(children, nchildren, label, CYPHER_AST_LABEL, NULL);
    REQUIRE_CHILD(children, nchildren, prop_name, CYPHER_AST_PROP_NAME, NULL);

    struct using_index *node = calloc(1, sizeof(struct using_index));
    if (node == NULL)
    {
        return NULL;
    }
    if (cypher_astnode_init(&(node->_astnode), CYPHER_AST_USING_INDEX,
            children, nchildren, range))
    {
        free(node);
        return NULL;
    }
    node->identifier = identifier;
    node->label = label;
    node->prop_name = prop_name;
    return &(node->_astnode);
}


cypher_astnode_t *clone(const cypher_astnode_t *self,
        cypher_astnode_t **children)
{
    REQUIRE_TYPE(self, CYPHER_AST_USING_INDEX, NULL);
    struct using_index *node = container_of(self,
            struct using_index, _astnode);

    cypher_astnode_t *identifier = children[child_index(self, node->identifier)];
    cypher_astnode_t *label = children[child_index(self, node->label)];
    cypher_astnode_t *prop_name = children[child_index(self, node->prop_name)];

    return cypher_ast_using_index(identifier, label, prop_name, children,
            self->nchildren, self->range);
}


const cypher_astnode_t *cypher_ast_using_index_get_identifier(
        const cypher_astnode_t *astnode)
{
    REQUIRE_TYPE(astnode, CYPHER_AST_USING_INDEX, NULL);
    struct using_index *node = container_of(astnode,
            struct using_index, _astnode);
    return node->identifier;
}


const cypher_astnode_t *cypher_ast_using_index_get_label(
        const cypher_astnode_t *astnode)
{
    REQUIRE_TYPE(astnode, CYPHER_AST_USING_INDEX, NULL);
    struct using_index *node = container_of(astnode,
            struct using_index, _astnode);
    return node->label;
}


const cypher_astnode_t *cypher_ast_using_index_get_prop_name(
        const cypher_astnode_t *astnode)
{
    REQUIRE_TYPE(astnode, CYPHER_AST_USING_INDEX, NULL);
    struct using_index *node = container_of(astnode,
            struct using_index, _astnode);
    return node->prop_name;
}


ssize_t detailstr(const cypher_astnode_t *self, char *str, size_t size)
{
    REQUIRE_TYPE(self, CYPHER_AST_USING_INDEX, -1);
    struct using_index *node = container_of(self, struct using_index, _astnode);
    return snprintf(str, size, "@%u:@%u(@%u)", node->identifier->ordinal,
            node->label->ordinal, node->prop_name->ordinal);
}
