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
#include "string_buffer.h"
#include <assert.h>
#include <string.h>

#define CYPHER_PARSER_STRING_BUFFER_BLOCK_SIZE 8


int cp_sb_append(struct cp_string_buffer* sb, const char *s, size_t n)
{
    assert(sb->length <= sb->capacity);
    if ((sb->length + n) >= sb->capacity)
    {
        size_t newcap = (((sb->length + n) + 
                (3*CYPHER_PARSER_STRING_BUFFER_BLOCK_SIZE/2)) /
                CYPHER_PARSER_STRING_BUFFER_BLOCK_SIZE
                ) * CYPHER_PARSER_STRING_BUFFER_BLOCK_SIZE;
        void *buf = realloc(sb->buffer, newcap);
        if (buf == NULL)
        {
            return -1;
        }
        sb->buffer = buf;
        sb->capacity = newcap;
    }
    assert((sb->length + n) < sb->capacity);
    memcpy(sb->buffer + sb->length, s, n);
    sb->length += n;
    return 0;
}


void cp_sb_cleanup(struct cp_string_buffer *sb)
{
    free(sb->buffer);
    memset(sb, 0, sizeof(struct cp_string_buffer));
}
