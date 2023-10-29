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
#include "memstream.h"

#ifdef WIN32

#include <fmem.h>

fmem fm;

FILE* open_memstream(char** mem, size_t* size)
{
	fmem_init(&fm);
	return fmem_open(&fm, "w+");
}

#else

#ifndef HAVE_OPEN_MEMSTREAM

#define MEMSTREAM_INITIAL_CAPACITY 4096

#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <assert.h>


typedef struct memstream memstream_t;
struct memstream
{
    char **ptr;
    size_t *sizeloc;

    char *buffer;
    size_t capacity;

    size_t position;
};


static int memstream_read(void *cookie, char *buf, int nbytes);
static int memstream_write(void *cookie, const char *buf, int nbytes);
static fpos_t memstream_seek(void *cookie, fpos_t offset, int whence);
static int memstream_close(void *cookie);
static int memstream_ensure_capacity(memstream_t *ms, size_t size);


static inline size_t min(size_t a, size_t b)
{
    return (a <= b)? a : b;
}


FILE *open_memstream(char **ptr, size_t *sizeloc)
{
    if (ptr == NULL || sizeloc == NULL)
    {
        errno = EINVAL;
        return NULL;
    }

    memstream_t *ms = calloc(1, sizeof(memstream_t));
    if (ms == NULL)
    {
        return NULL;
    }

    ms->ptr = ptr;
    ms->sizeloc = sizeloc;

    ms->capacity = MEMSTREAM_INITIAL_CAPACITY;
    ms->buffer = calloc(1, ms->capacity);
    if (ms->buffer == NULL)
    {
        free(ms);
        return NULL;
    }

    FILE *fp = funopen(ms, memstream_read, memstream_write,
            memstream_seek,memstream_close);
    if (fp == NULL)
    {
        free(ms->buffer);
        free(ms);
        return NULL;
    }

    *ptr = ms->buffer;
    *sizeloc = 0;
    return fp;
}


static int memstream_read(void *cookie, char *buf, int nbytes)
{
    if (nbytes < 0 || buf == NULL)
    {
        errno = EINVAL;
        return -1;
    }
    memstream_t *ms = (memstream_t *)cookie;
    size_t n = min(*(ms->sizeloc) - ms->position, nbytes);
    if (n == 0)
    {
        return 0;
    }
    memcpy(buf, ms->buffer + ms->position, n);
    ms->position += n;
    return n;
}


int memstream_write(void *cookie, const char *buf, int nbytes)
{
    if (nbytes < 0 || buf == NULL)
    {
        errno = EINVAL;
        return -1;
    }
    memstream_t *ms = (memstream_t *)cookie;

    if (memstream_ensure_capacity(ms, ms->position + nbytes))
    {
        return -1;
    }
    assert(ms->capacity > (ms->position + nbytes));
    memcpy(ms->buffer + ms->position, buf, nbytes);

    ms->position += nbytes;
    if (*(ms->sizeloc) < ms->position)
    {
        assert((ms->buffer[ms->position]) == 0);
        *(ms->sizeloc) = ms->position;
    }
    return nbytes;
}


fpos_t memstream_seek(void *cookie, fpos_t offset, int whence)
{
    memstream_t *ms = (memstream_t *)cookie;

    size_t position = 0;
    switch (whence)
    {
    case SEEK_SET:
        position = offset;
        break;
    case SEEK_CUR:
        position = ms->position + offset;
        break;
    case SEEK_END:
        position = *(ms->sizeloc) + offset;
        break;
    default:
        errno = EINVAL;
        return -1;
    }

    if (memstream_ensure_capacity(ms, position))
    {
        return -1;
    }
    if (*(ms->sizeloc) < position)
    {
        assert((ms->buffer[position]) == 0);
        *(ms->sizeloc) = position;
    }
    ms->position = position;
    return (fpos_t) position;
}


int memstream_close(void *cookie)
{
    memstream_t *ms = (memstream_t *)cookie;
    free(ms);
    return 0;
}


int memstream_ensure_capacity(memstream_t *ms, size_t size)
{
    size_t capacity;
    for (capacity = ms->capacity; capacity <= size; capacity = capacity*2)
        ;
    assert(capacity > size); // must be at least 1 null byte after content
    if (capacity == ms->capacity)
    {
        return 0;
    }
    ms->buffer = realloc(ms->buffer, capacity);
    if (ms->buffer == NULL)
    {
        return -1;
    }
    memset(ms->buffer + ms->capacity, 0, capacity - ms->capacity);
    ms->capacity = capacity;
    *(ms->ptr) = ms->buffer;
    return 0;
}


#endif
#endif