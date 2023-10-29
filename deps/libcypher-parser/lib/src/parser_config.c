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
#include "parser_config.h"

#define ANSI_COLOR_RESET "\x1b[0m"
#define ANSI_COLOR_BOLD "\x1b[1m"
#define ANSI_COLOR_RED "\x1b[31m"
#define ANSI_COLOR_GREEN "\x1b[32m"
#define ANSI_COLOR_YELLOW "\x1b[33m"
#define ANSI_COLOR_BLUE "\x1b[34m"
#define ANSI_COLOR_BLUE_BG "\x1b[44m"
#define ANSI_COLOR_MAGENTA "\x1b[35m"
#define ANSI_COLOR_CYAN "\x1b[36m"


const struct cypher_input_position cypher_input_position_zero = { 1, 1, 0 };

static struct cypher_parser_colorization _cypher_parser_no_colorization =
    { .normal = { "", "" },
      .error = { "", "" },
      .error_token = { "", "" },
      .error_message = { "", "" },
      .ast_ordinal = { "", "" },
      .ast_range = { "", "" },
      .ast_indent = { "", "" },
      .ast_type = { "", "" },
      .ast_desc = { "", "" } };

static struct cypher_parser_colorization _cypher_parser_ansi_colorization =
    { .normal = { ANSI_COLOR_RESET, "" },
      .error = { ANSI_COLOR_BOLD ANSI_COLOR_RED, ANSI_COLOR_RESET },
      .error_token = { ANSI_COLOR_BOLD, ANSI_COLOR_RESET },
      .error_message = { ANSI_COLOR_BOLD, ANSI_COLOR_RESET },
      .ast_ordinal = { "", "" },
      .ast_range = { ANSI_COLOR_CYAN, ANSI_COLOR_RESET },
      .ast_indent = { ANSI_COLOR_YELLOW, ANSI_COLOR_RESET },
      .ast_type = { ANSI_COLOR_BOLD, ANSI_COLOR_RESET },
      .ast_desc = { "", "" } };

const struct cypher_parser_colorization *cypher_parser_no_colorization =
    &_cypher_parser_no_colorization;
const struct cypher_parser_colorization *cypher_parser_ansi_colorization =
    &_cypher_parser_ansi_colorization;


struct cypher_parser_config cypher_parser_std_config =
    { .initial_position = { 1, 1, 0 },
      .initial_ordinal = 0,
      .error_colorization = &_cypher_parser_no_colorization };


const char *libcypher_parser_version(void)
{
    return PACKAGE_VERSION;
}


cypher_parser_config_t *cypher_parser_new_config(void)
{
    cypher_parser_config_t *config = calloc(1, sizeof(cypher_parser_config_t));
    if (config == NULL)
    {
        return NULL;
    }
    *config = cypher_parser_std_config;
    return config;
}


void cypher_parser_config_free(cypher_parser_config_t *config)
{
    free(config);
}


void cypher_parser_config_set_initial_position(cypher_parser_config_t *config,
        struct cypher_input_position position)
{
    config->initial_position = position;
}


void cypher_parser_config_set_initial_ordinal(cypher_parser_config_t *config,
        unsigned int n)
{
    config->initial_ordinal = n;
}


void cypher_parser_config_set_error_colorization(cypher_parser_config_t *config,
        const struct cypher_parser_colorization *colorization)
{
    config->error_colorization = colorization;
}
