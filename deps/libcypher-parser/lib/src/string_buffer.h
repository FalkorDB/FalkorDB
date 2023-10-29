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
#ifndef CYPHER_PARSER_STRING_BUFFER_H
#define CYPHER_PARSER_STRING_BUFFER_H

#include <stdlib.h>

struct cp_string_buffer
{
    char *buffer;
    size_t capacity;
    size_t length;
};


static inline void cp_sb_reset(struct cp_string_buffer* sb)
{
    sb->length = 0;
}

static inline const char *cp_sb_data(struct cp_string_buffer* sb)
{
    return sb->buffer;
}

static inline size_t cp_sb_length(struct cp_string_buffer* sb)
{
    return sb->length;
}

int cp_sb_append(struct cp_string_buffer* sb, const char *s, size_t n);

void cp_sb_cleanup(struct cp_string_buffer *sb);


#endif/*CYPHER_PARSER_STRING_BUFFER_H*/
