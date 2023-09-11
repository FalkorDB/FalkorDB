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
#ifndef CYPHER_PARSER_ERRORS_H
#define CYPHER_PARSER_ERRORS_H

#include "cypher-parser.h"


struct cypher_parse_error
{
    struct cypher_input_position position;
    char *msg;
    char *context;
    size_t context_offset;
};


void cp_errors_vcleanup(cypher_parse_error_t *errors, unsigned int n);


typedef struct cp_error_tracking cp_error_tracking_t;
struct cp_error_tracking
{
    const struct cypher_parser_colorization *colorization;

    struct cypher_input_position last_position;
    char last_char;
    const char **labels;
    unsigned int labels_capacity;
    unsigned int nlabels;

    cypher_parse_error_t *errors;
    unsigned int errors_capacity;
    unsigned int nerrors;
    size_t last_error_offset;
};


void cp_et_init(cp_error_tracking_t *et,
        const struct cypher_parser_colorization *colorization);

int cp_et_note_potential_error(cp_error_tracking_t *et,
        struct cypher_input_position position, char c, const char *label);

int cp_et_reify_potentials(cp_error_tracking_t *et);

static inline void cp_et_clear_potentials(cp_error_tracking_t *et)
{
    et->nlabels = 0;
}

static inline unsigned int cp_et_nerrors(const cp_error_tracking_t *et)
{
    return et->nerrors;
}

static inline cypher_parse_error_t *cp_et_errors(cp_error_tracking_t *et)
{
    return et->errors;
}

static inline void cp_et_clear_errors(cp_error_tracking_t *et)
{
    et->nerrors = 0;
}

void cp_et_cleanup(cp_error_tracking_t *et);

#endif/*CYPHER_PARSER_ERRORS_H*/
