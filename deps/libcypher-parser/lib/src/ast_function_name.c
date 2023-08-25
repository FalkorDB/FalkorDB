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


struct function_name
{
    cypher_astnode_t _astnode;
    char p[];
};


static cypher_astnode_t *clone(const cypher_astnode_t *self,
        cypher_astnode_t **children);
static ssize_t detailstr(const cypher_astnode_t *self, char *str, size_t size);


const struct cypher_astnode_vt cypher_function_name_astnode_vt =
    { .name = "function name",
      .detailstr = detailstr,
      .release = cypher_astnode_release,
      .clone = clone };


cypher_astnode_t *cypher_ast_function_name(const char *s, size_t n,
        struct cypher_input_range range)
{
    struct function_name *node = calloc(1, sizeof(struct function_name) + n+1);
    if (node == NULL)
    {
        return NULL;
    }
    if (cypher_astnode_init(&(node->_astnode), CYPHER_AST_FUNCTION_NAME,
                NULL, 0, range))
    {
        free(node);
        return NULL;
    }
    memcpy(node->p, s, n);
    node->p[n] = '\0';
    return &(node->_astnode);
}


cypher_astnode_t *clone(const cypher_astnode_t *self,
        cypher_astnode_t **children)
{
    REQUIRE_TYPE(self, CYPHER_AST_FUNCTION_NAME, NULL);
    struct function_name *node =
            container_of(self, struct function_name, _astnode);
    return cypher_ast_function_name(node->p, strlen(node->p), self->range);
}


const char *cypher_ast_function_name_get_value(const cypher_astnode_t *astnode)
{
    REQUIRE_TYPE(astnode, CYPHER_AST_FUNCTION_NAME, NULL);
    struct function_name *node =
            container_of(astnode, struct function_name, _astnode);
    return node->p;
}


ssize_t detailstr(const cypher_astnode_t *self, char *str, size_t size)
{
    REQUIRE_TYPE(self, CYPHER_AST_FUNCTION_NAME, -1);
    struct function_name *node =
            container_of(self, struct function_name, _astnode);
    return snprintf(str, size, "`%s`", node->p);
}
