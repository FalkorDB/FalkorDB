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
#include "vector.h"
#include <assert.h>
#include <stdlib.h>
#include <string.h>

#define CYPHER_VECTOR_BLOCK_SIZE 8


struct cp_vector *cp_vector(size_t element_size)
{
    assert(element_size > 0);
    struct cp_vector *vec = calloc(1, sizeof(struct cp_vector));
    if (vec == NULL)
    {
        return NULL;
    }
    vec->element_size = element_size;
    return vec;
}


void cp_vector_free(struct cp_vector *vec)
{
    cp_vector_cleanup(vec);
    free(vec);
}


void cp_vector_init(struct cp_vector *vec, size_t element_size)
{
    assert(vec != NULL);
    assert(element_size > 0);
    memset(vec, 0, sizeof(struct cp_vector));
    vec->element_size = element_size;
}


void cp_vector_cleanup(struct cp_vector *vec)
{
    free(vec->elements);
    vec->capacity = 0;
    vec->length = 0;
}


int cp_vector_push(struct cp_vector *vec, void *element)
{
    assert(vec->length <= vec->capacity);
    if (vec->length >= vec->capacity)
    {
        unsigned int newcap = (vec->capacity == 0)?
            CYPHER_VECTOR_BLOCK_SIZE : vec->capacity * 2;
        void *elements = realloc(vec->elements, newcap * vec->element_size);
        if (element == NULL)
        {
            return -1;
        }
        vec->elements = elements;
        vec->capacity = newcap;
    }
    assert(vec->length < vec->capacity);
    memcpy(vec->elements + (vec->length * vec->element_size), element,
            vec->element_size);
    ++(vec->length);
    return 0;
}
