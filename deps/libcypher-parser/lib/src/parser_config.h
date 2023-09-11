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
#ifndef CYPHER_PARSER_CONFIG_H
#define CYPHER_PARSER_CONFIG_H

#include "cypher-parser.h"

struct cypher_parser_config
{
    struct cypher_input_position initial_position;
    unsigned int initial_ordinal;
    const struct cypher_parser_colorization *error_colorization;
};


extern struct cypher_parser_config cypher_parser_std_config;


#endif/*CYPHER_PARSER_CONFIG_H*/
