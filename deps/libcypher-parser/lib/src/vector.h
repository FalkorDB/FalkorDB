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
#ifndef CYPHER_PARSER_VECTOR_H
#define CYPHER_PARSER_VECTOR_H

#include <stdlib.h>
#include <stdbool.h>
#include "util.h"

#if __GNUC__ > 3
#define __cypherlang_unused __attribute__((unused))
#else
#define __cypherlang_unused /*unused*/
#endif

#define DECLARE_VECTOR(name, type, def) \
    struct name { struct cp_vector vec; }; \
    \
    __cypherlang_unused __cypherlang_must_check \
    static inline struct name *name(void) \
    { return (struct name *)cp_vector(sizeof(type)); } \
    \
    __cypherlang_unused \
    static inline void name##_free(struct name *s) \
    { cp_vector_free(&(s->vec)); } \
    \
    __cypherlang_unused \
    static inline void name##_init(struct name *s) \
    { cp_vector_init(&(s->vec), sizeof(type)); } \
    \
    __cypherlang_unused \
    static inline void name##_cleanup(struct name *s) \
    { cp_vector_cleanup(&(s->vec)); } \
    \
    __cypherlang_unused __cypherlang_must_check \
    static inline int name##_push(struct name *s, type element) \
    { return cp_vector_push(&(s->vec), &element); } \
    \
    __cypherlang_unused \
    static inline type name##_pop(struct name *s) \
    { type *r = cp_vector_pop(&(s->vec)); return (r == NULL)? def:*r; } \
    \
    __cypherlang_unused \
    static inline void name##_npop(struct name *s, unsigned int n) \
    { cp_vector_npop(&(s->vec), n); } \
    \
    __cypherlang_unused \
    static inline type name##_get(struct name *s, unsigned int i) \
    { type *r = cp_vector_get(&(s->vec), i); return (r == NULL)? def:*r; }\
    \
    __cypherlang_unused \
    static inline type name##_last(struct name *s) \
    { type *r = cp_vector_last(&(s->vec)); return (r == NULL)? def:*r; } \
    \
    __cypherlang_unused \
    static inline unsigned int name##_size(struct name *s) \
    { return cp_vector_size(&(s->vec)); } \
    \
    __cypherlang_unused \
    static inline void name##_clear(struct name *s) \
    { cp_vector_clear(&(s->vec)); } \
    \
    __cypherlang_unused \
    static inline type *name##_elements(struct name *s) \
    { return cp_vector_elements(&(s->vec)); } \
    \
    typedef struct name name##_t


struct cp_vector
{
    size_t element_size;
    char *elements;
    unsigned int capacity;
    unsigned int length;
};


struct cp_vector *cp_vector(size_t element_size);

void cp_vector_free(struct cp_vector *vec);

void cp_vector_init(struct cp_vector *vec, size_t element_size);

void cp_vector_cleanup(struct cp_vector *vec);

int cp_vector_push(struct cp_vector *vec, void *element);

static inline void *cp_vector_pop(struct cp_vector *vec)
{
    return (vec->length > 0)?
        (vec->elements + (--(vec->length) * vec->element_size)) : NULL;
}

static inline void cp_vector_npop(struct cp_vector *vec, unsigned int n)
{
    vec->length -= minu(vec->length, n);
}

static inline void *cp_vector_get(struct cp_vector *vec, unsigned int i)
{
    return (i < vec->length)? (vec->elements + (i * vec->element_size)) : NULL;
}

static inline void *cp_vector_last(struct cp_vector *vec)
{
    return (vec->length > 0)?
        (vec->elements + ((vec->length-1) * vec->element_size)) : NULL;
}

static inline unsigned int cp_vector_size(struct cp_vector *vec)
{
    return vec->length;
}

static inline void cp_vector_clear(struct cp_vector *vec)
{
    vec->length = 0;
}

static inline void *cp_vector_elements(struct cp_vector *vec)
{
    return vec->elements;
}

#endif/*CYPHER_PARSER_VECTOR_H*/
