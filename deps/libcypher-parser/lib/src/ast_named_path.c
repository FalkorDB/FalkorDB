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


struct named_path
{
    cypher_pattern_path_astnode_t _pattern_path_astnode;
    const cypher_astnode_t *identifier;
    const cypher_astnode_t *path;
};


static cypher_astnode_t *clone(const cypher_astnode_t *self,
        cypher_astnode_t **children);
static ssize_t detailstr(const cypher_astnode_t *self, char *str, size_t size);
static unsigned int nelements(const cypher_pattern_path_astnode_t *self);
static const cypher_astnode_t *get_element(
        const cypher_pattern_path_astnode_t *self, unsigned int index);


static const struct cypher_astnode_vt *parents[] =
    { &cypher_pattern_path_astnode_vt };

const struct cypher_astnode_vt cypher_named_path_astnode_vt =
    { .parents = parents,
      .nparents = 1,
      .name = "named path",
      .detailstr = detailstr,
      .release = cypher_astnode_release,
      .clone = clone };

static const struct cypher_pattern_path_astnode_vt pp_vt =
    { .nelements = nelements,
      .get_element = get_element };


cypher_astnode_t *cypher_ast_named_path(const cypher_astnode_t *identifier,
        const cypher_astnode_t *path, cypher_astnode_t **children,
        unsigned int nchildren, struct cypher_input_range range)
{
    REQUIRE_CHILD(children, nchildren, identifier, CYPHER_AST_IDENTIFIER, NULL);
    REQUIRE_CHILD(children, nchildren, path, CYPHER_AST_PATTERN_PATH, NULL);

    struct named_path *node = calloc(1, sizeof(struct named_path));
    if (node == NULL)
    {
        return NULL;
    }
    if (cypher_pattern_path_astnode_init(&(node->_pattern_path_astnode),
                CYPHER_AST_NAMED_PATH, &pp_vt, children, nchildren, range))
    {
        free(node);
        return NULL;
    }
    node->identifier = identifier;
    node->path = path;
    return &(node->_pattern_path_astnode._astnode);
}


cypher_astnode_t *clone(const cypher_astnode_t *self,
        cypher_astnode_t **children)
{
    REQUIRE_TYPE(self, CYPHER_AST_NAMED_PATH, NULL);
    const cypher_pattern_path_astnode_t *ppnode =
            container_of(self, cypher_pattern_path_astnode_t, _astnode);
    struct named_path *node =
            container_of(ppnode, struct named_path, _pattern_path_astnode);

    cypher_astnode_t *identifier = children[child_index(self, node->identifier)];
    cypher_astnode_t *path = children[child_index(self, node->path)];

    return cypher_ast_named_path(identifier, path, children, self->nchildren,
            self->range);
}


const cypher_astnode_t *cypher_ast_named_path_get_identifier(
        const cypher_astnode_t *astnode)
{
    REQUIRE_TYPE(astnode, CYPHER_AST_NAMED_PATH, NULL);
    const cypher_pattern_path_astnode_t *ppnode =
            container_of(astnode, cypher_pattern_path_astnode_t, _astnode);
    struct named_path *node =
            container_of(ppnode, struct named_path, _pattern_path_astnode);
    return node->identifier;
}


const cypher_astnode_t *cypher_ast_named_path_get_path(
        const cypher_astnode_t *astnode)
{
    REQUIRE_TYPE(astnode, CYPHER_AST_NAMED_PATH, NULL);
    const cypher_pattern_path_astnode_t *ppnode =
            container_of(astnode, cypher_pattern_path_astnode_t, _astnode);
    struct named_path *node =
            container_of(ppnode, struct named_path, _pattern_path_astnode);
    return node->path;
}


unsigned int nelements(const cypher_pattern_path_astnode_t *self)
{
    const struct named_path *node =
            container_of(self, struct named_path, _pattern_path_astnode);
    return cypher_ast_pattern_path_nelements(node->path);
}


const cypher_astnode_t *get_element(const cypher_pattern_path_astnode_t *self,
        unsigned int index)
{
    const struct named_path *node =
            container_of(self, struct named_path, _pattern_path_astnode);
    return cypher_ast_pattern_path_get_element(node->path, index);
}


ssize_t detailstr(const cypher_astnode_t *self, char *str, size_t size)
{
    REQUIRE_TYPE(self, CYPHER_AST_NAMED_PATH, -1);
    const cypher_pattern_path_astnode_t *ppnode =
            container_of(self, cypher_pattern_path_astnode_t, _astnode);
    struct named_path *node =
            container_of(ppnode, struct named_path, _pattern_path_astnode);
    return snprintf(str, size, "@%d = @%d", node->identifier->ordinal,
            node->path->ordinal);
}
