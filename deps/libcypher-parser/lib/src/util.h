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
#ifndef CYPHER_PARSER_UTIL_H
#define CYPHER_PARSER_UTIL_H

#include "cypher-parser.h"
#include <errno.h>
#include <stddef.h>


/**
 * Get the containing structure address.
 */
#define container_of(ptr, type, member) \
        (type *)(void *)( (uint8_t *)(uintptr_t)(ptr) - offsetof(type,member) )

/**
 * Ensure the condition is true, or return the specified result value.
 */
#ifdef NDEBUG
#define REQUIRE(cond, res) \
    if (!(cond)) { errno = EINVAL; return (res); }
#else
#define REQUIRE(cond, res) assert(cond)
#endif

/**
 * Check if the named value is null, and if so then update it to point to a
 * stack variable of the specified type.
 */
#define ENSURE_NOT_NULL(type, name, val) \
    type _##name = (val); \
    if (name == NULL) \
    { \
        name = &_##name; \
    }

#ifdef WIN32
#undef min
#undef max
#endif

/**
 * Determine the minimum of two integers.
 *
 * @param [a] The first integer.
 * @param [b] The second integer.
 * @return The smaller of the two integers.
 */
static inline int min(int a, int b)
{
    return (a <= b)? a : b;
}

/**
 * Determine the minimum of two unsigned integers.
 *
 * @param [a] The first integer.
 * @param [b] The second integer.
 * @return The smaller of the two integers.
 */
static inline unsigned int minu(unsigned int a, unsigned int b)
{
    return (a <= b)? a : b;
}

/**
 * Determine the minimum of two size_t values.
 *
 * @param [a] The first size_t value.
 * @param [b] The second size_t value.
 * @return The smaller of the two size_t values.
 */
static inline size_t minzu(size_t a, size_t b)
{
    return (a <= b)? a : b;
}

/**
 * The maximum of two integers.
 *
 * @param [a] The first integer.
 * @param [b] The second integer.
 * @return The larger of the two integers.
 */
static inline int max(int a, int b)
{
    return (a >= b)? a : b;
}

/**
 * The maximum of two unsigned integers.
 *
 * @param [a] The first integer.
 * @param [b] The second integer.
 * @return The larger of the two integers.
 */
static inline unsigned int maxu(unsigned int a, unsigned int b)
{
    return (a >= b)? a : b;
}

/**
 * Determine the maximum of two size_t values.
 *
 * @param [a] The first size_t value.
 * @param [b] The second size_t value.
 * @return The larger of the two size_t values.
 */
static inline size_t maxzu(size_t a, size_t b)
{
    return (a >= b)? a : b;
}


/**
 * Duplicate a region of memory.
 *
 * @iternal
 *
 * @param [src] The memory to duplicate.
 * @param [n] The number of bytes in the memory region.
 * @return The newly allocated region, or null if an error occurs (errno will
 *         be set).
 */
static inline void *mdup(const void *src, size_t n)
{
    if (n == 0)
    {
        return NULL;
    }

    void *dst = malloc(n);
    if (dst == NULL)
    {
        return NULL;
    }
    return memcpy(dst, src, n);
}


/**
 * Write a formatted string to a buffer, realloc'ing as required.
 *
 * @internal
 *
 * @param [buf] A pointer to the output buffer, or NULL.
 * @param [bufcap] A pointer to the buffer size.
 * @param [format] The output format.
 * @param [...] Arguments to the format.
 * @return The number of characters written to the buffer, or -1 if an
 *         error occurs (errno will be set).
 */
ssize_t snprintf_realloc(char ** restrict buf, size_t *bufcap,
        const char * restrict format, ...) __cypherlang_format(3, 4);

/**
 * Obtain a line of context around a specified point in a buffer.
 *
 * @internal
 *
 * A section of the buffer is identified, which provides context around
 * a specified point.
 *
 * @param [buf] A pointer to a buffer.
 * @param [bufsize] The size of the buffer.
 * @param [offset] A pointer to an size_t, specifying the offset into the
 *         buffer around which context is required. On return, the size_t will
 *         be updated to contain the offset of the same point within the
 *         returned context.
 * @param [max_length] The maximum length of the context to return, which must
 *         be at least 7.
 * @return A pointer to a newly allocated segment of memory, containing the
 *         context terminated by a null character, or NULL if an error occurs
 *         (errno will be set).
 */
char *line_context(const char *buf, size_t bufsize, size_t *offset,
        size_t max_length);

#endif/*CYPHER_PARSER_UTIL_H*/
