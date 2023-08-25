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


struct map_projection_property
{
    cypher_astnode_t _astnode;
    const cypher_astnode_t *prop_name;
};


static cypher_astnode_t *clone(const cypher_astnode_t *self,
        cypher_astnode_t **children);
static ssize_t detailstr(const cypher_astnode_t *self, char *str, size_t size);


static const struct cypher_astnode_vt *parents[] =
    { &cypher_map_projection_selector_astnode_vt };

const struct cypher_astnode_vt cypher_map_projection_property_astnode_vt =
    { .parents = parents,
      .nparents = 1,
      .name = "property projection",
      .detailstr = detailstr,
      .release = cypher_astnode_release,
      .clone = clone };


cypher_astnode_t *cypher_ast_map_projection_property(
        const cypher_astnode_t *prop_name,
        cypher_astnode_t **children, unsigned int nchildren,
        struct cypher_input_range range)
{
    REQUIRE_CHILD(children, nchildren, prop_name, CYPHER_AST_PROP_NAME, NULL);

    struct map_projection_property *node =
            calloc(1, sizeof(struct map_projection_property));
    if (node == NULL)
    {
        return NULL;
    }
    if (cypher_astnode_init(&(node->_astnode),
            CYPHER_AST_MAP_PROJECTION_PROPERTY, children, nchildren, range))
    {
        free(node);
        return NULL;
    }
    node->prop_name = prop_name;
    return &(node->_astnode);
}


cypher_astnode_t *clone(const cypher_astnode_t *self,
        cypher_astnode_t **children)
{
    REQUIRE_TYPE(self, CYPHER_AST_MAP_PROJECTION_PROPERTY, NULL);
    struct map_projection_property *node =
            container_of(self, struct map_projection_property, _astnode);

    cypher_astnode_t *prop_name = children[child_index(self, node->prop_name)];

    return cypher_ast_map_projection_property(prop_name, children,
            self->nchildren, self->range);
}


const cypher_astnode_t *cypher_ast_map_projection_property_get_prop_name(
        const cypher_astnode_t *astnode)
{
    REQUIRE_TYPE(astnode, CYPHER_AST_MAP_PROJECTION_PROPERTY, NULL);
    struct map_projection_property *node =
            container_of(astnode, struct map_projection_property, _astnode);
    return node->prop_name;
}


ssize_t detailstr(const cypher_astnode_t *self, char *str, size_t size)
{
    REQUIRE_TYPE(self, CYPHER_AST_MAP_PROJECTION_PROPERTY, -1);
    struct map_projection_property *node =
            container_of(self, struct map_projection_property, _astnode);
    return snprintf(str, size, ".@%u", node->prop_name->ordinal);
}
