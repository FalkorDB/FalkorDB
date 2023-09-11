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


struct map_projection_all_properties
{
    cypher_astnode_t _astnode;
};


static cypher_astnode_t *clone(const cypher_astnode_t *self,
        cypher_astnode_t **children);
static ssize_t detailstr(const cypher_astnode_t *self, char *str, size_t size);


static const struct cypher_astnode_vt *parents[] =
    { &cypher_map_projection_selector_astnode_vt };

const struct cypher_astnode_vt cypher_map_projection_all_properties_astnode_vt =
    { .parents = parents,
      .nparents = 1,
      .name = "all properties projection",
      .detailstr = detailstr,
      .release = cypher_astnode_release,
      .clone = clone };


cypher_astnode_t *cypher_ast_map_projection_all_properties(
        cypher_astnode_t **children, unsigned int nchildren,
        struct cypher_input_range range)
{
    struct map_projection_all_properties *node =
            calloc(1, sizeof(struct map_projection_all_properties));
    if (node == NULL)
    {
        return NULL;
    }
    if (cypher_astnode_init(&(node->_astnode),
            CYPHER_AST_MAP_PROJECTION_ALL_PROPERTIES,
            children, nchildren, range))
    {
        free(node);
        return NULL;
    }
    return &(node->_astnode);
}


cypher_astnode_t *clone(const cypher_astnode_t *self,
        cypher_astnode_t **children)
{
    REQUIRE_TYPE(self, CYPHER_AST_MAP_PROJECTION_ALL_PROPERTIES, NULL);
    return cypher_ast_map_projection_all_properties(children, self->nchildren,
            self->range);
}


ssize_t detailstr(const cypher_astnode_t *self, char *str, size_t size)
{
    REQUIRE_TYPE(self, CYPHER_AST_MAP_PROJECTION_ALL_PROPERTIES, -1);
    return snprintf(str, size, ".*");
}
