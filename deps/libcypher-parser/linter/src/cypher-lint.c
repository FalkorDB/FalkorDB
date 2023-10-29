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
#include "cypher-parser.h"
#include <assert.h>
#include <errno.h>
#include <getopt.h>
#include <stdio.h>
#include <string.h>
#ifndef WIN32
#include <libgen.h>
#include <sys/ioctl.h>
#include <unistd.h>
#endif

const char *shortopts = "1ahv";

#define COLORIZE_OPT 1004
#define NO_COLORIZE_OPT 1005
#define ONLY_STATEMENTS_OPT 1006
#define OUTPUT_WIDTH_OPT 1007
#define STREAM_OPT 1008
#define VERSION_OPT 1009

static struct option longopts[] =
    { { "ast", no_argument, NULL, 'a' },
      { "colorize", no_argument, NULL, COLORIZE_OPT },
      { "colorise", no_argument, NULL, COLORIZE_OPT },
      { "colourise", no_argument, NULL, COLORIZE_OPT },
      { "no-colorize", no_argument, NULL, NO_COLORIZE_OPT },
      { "no-colorise", no_argument, NULL, NO_COLORIZE_OPT },
      { "no-colourise", no_argument, NULL, NO_COLORIZE_OPT },
      { "help", no_argument, NULL, 'h' },
      { "only-statements", no_argument, NULL, ONLY_STATEMENTS_OPT },
      { "output-width", required_argument, NULL, OUTPUT_WIDTH_OPT },
      { "stream", no_argument, NULL, STREAM_OPT },
      { "version", no_argument, NULL, VERSION_OPT },
      { NULL, 0, NULL, 0 } };

static void usage(FILE *s, const char *prog_name)
{
    fprintf(s,
"usage: %s [OPTIONS] [file ...]\n"
"options:\n"
" -1                  Only parse the first statement or client-command.\n"
" --ast, -a           Dump the AST to stdout.\n"
" --colorize          Colorize output using ANSI escape sequences.\n"
" --no-colorize       Disable colorization even when outputting to a TTY.\n"
" --help, -h          Output this usage information.\n"
" --only-statements   Only parse statements (and not client commands).\n"
" --output-width <n>  Attempt to limit output to the specified width.\n"
" --stream            Output each statement as it is read, rather than parsing\n"
"                     the entire input first (note: will result in inconsistent\n"
"                     formatting of AST dumps).\n"
" --version           Output the version of cypher-lint and libcypher-parser\n"
"\n"
"If no input files are specified, then input is read from standard input.\n"
"\n",
        prog_name);
}


struct lint_config
{
    unsigned int width;
    int flags;
    bool dump_ast;
    bool colorize_output;
    bool colorize_errors;
    bool stream;
};


static int process(FILE *stream, const char *filename,
        struct lint_config *config);
static int process_streamed(FILE *stream, const char *filename,
        struct lint_config *config, cypher_parser_config_t *cp_config,
        const struct cypher_parser_colorization *error_colorization,
        const struct cypher_parser_colorization *output_colorization);
static int process_all(FILE *stream, const char *filename,
        struct lint_config *config, cypher_parser_config_t *cp_config,
        const struct cypher_parser_colorization *error_colorization,
        const struct cypher_parser_colorization *output_colorization);
static int parse_callback(void *data, cypher_parse_segment_t *segment);
static void print_error(const cypher_parse_error_t *error, const char *filename,
        const struct cypher_parser_colorization *colorization);

#ifdef WIN32
#define DEFINE_CONSOLEV2_PROPERTIES

// System headers
#include <windows.h>
// Standard library C-style
bool EnableVTMode()
{
	// Set output mode to handle virtual terminal sequences
	HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
	if (hOut == INVALID_HANDLE_VALUE)
	{
		return false;
	}

	DWORD dwMode = 0;
	if (!GetConsoleMode(hOut, &dwMode))
	{
		return false;
	}

	dwMode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
	if (!SetConsoleMode(hOut, dwMode))
	{
		return false;
	}
	return true;
}

char* basename(char* arg)
{
	char *prog_name = strrchr(arg, '\\');
	if (!prog_name) prog_name = arg;
	else if (*prog_name == '\\') prog_name++;
	if (!prog_name) prog_name = arg;
	return prog_name;
}

#endif//WIN32

int main(int argc, char *argv[])
{
#ifdef WIN32
	EnableVTMode();
#endif
    char *prog_name = basename(argv[0]);
    if (prog_name == NULL)
    {
        perror("unexpected error");
        exit(EXIT_FAILURE);
    }

    struct lint_config config;
    memset(&config, 0, sizeof(config));

    if (isatty(fileno(stdout)))
    {
        config.colorize_output = true;
    }
    if (isatty(fileno(stderr)))
    {
        config.colorize_errors = true;
    }

    int result = EXIT_FAILURE;

    int c;
    while ((c = getopt_long(argc, argv, shortopts, longopts, NULL)) >= 0)
    {
        switch (c)
        {
        case '1':
            config.flags |= CYPHER_PARSE_SINGLE;
            break;
        case 'a':
            config.dump_ast = true;
            break;
        case COLORIZE_OPT:
            config.colorize_output = true;
            config.colorize_errors = true;
            break;
        case NO_COLORIZE_OPT:
            config.colorize_output = false;
            config.colorize_errors = false;
            break;
        case 'h':
            usage(stdout, prog_name);
            result = EXIT_SUCCESS;
            goto cleanup;
        case ONLY_STATEMENTS_OPT:
            config.flags |= CYPHER_PARSE_ONLY_STATEMENTS;
            break;
        case OUTPUT_WIDTH_OPT:
            config.width = atoi(optarg);
            break;
        case STREAM_OPT:
            config.stream = true;
            break;
        case VERSION_OPT:
            fprintf(stdout, "cypher-lint: %s\n", PACKAGE_VERSION);
            fprintf(stdout, "libcypher-parser: %s\n",
                    libcypher_parser_version());
            result = EXIT_SUCCESS;
            goto cleanup;
        default:
            usage(stderr, prog_name);
            goto cleanup;
        }
    }
    argc -= optind;
    argv += optind;

    // Always stream if ast dumping is disabled
    if (!config.dump_ast)
    {
        config.stream = true;
    }

    if (argc > 0)
    {
        int err = 0;
        for (; argc > 0; --argc, ++argv)
        {
            int res;
            if (strcmp(*argv, "-") == 0)
            {
                res = process(stdin, "<stdin>", &config);
            }
            else
            {
                FILE *stream = fopen(*argv, "r");
                if (stream == NULL)
                {
                    fprintf(stderr, "%s: %s: %s\n", prog_name, *argv,
                            strerror(errno));
                    goto cleanup;
                }
                res = process(stream, *argv, &config);
                if (res < 0)
                {
                    goto cleanup;
                }
                fclose(stream);
            }
            err |= res;
        }
        if (err)
        {
            goto cleanup;
        }
    }
    else
    {
        if (process(stdin, NULL, &config))
        {
            goto cleanup;
        }
    }

    result = EXIT_SUCCESS;

cleanup:
    return result;
}


int process(FILE *stream, const char *filename, struct lint_config *config)
{
    cypher_parser_config_t *cp_config = cypher_parser_new_config();
    if (cp_config == NULL)
    {
        return -1;
    }

    const struct cypher_parser_colorization *error_colorization =
        cypher_parser_no_colorization;
    if (config->colorize_errors)
    {
        error_colorization = cypher_parser_ansi_colorization;
        cypher_parser_config_set_error_colorization(cp_config,
                error_colorization);
    }

    const struct cypher_parser_colorization *output_colorization =
        config->colorize_output? cypher_parser_ansi_colorization : NULL;

    int err = (config->stream)?
        process_streamed(stream, filename, config, cp_config,
              error_colorization, output_colorization) :
        process_all(stream, filename, config, cp_config,
              error_colorization, output_colorization);

    int errsv = errno;
    cypher_parser_config_free(cp_config);
    errno = errsv;
    return err;
}


struct parse_callback_data
{
    const char *filename;
    struct lint_config *config;
    const struct cypher_parser_colorization *error_colorization;
    const struct cypher_parser_colorization *output_colorization;
    unsigned int nerrors;
};


int process_streamed(FILE *stream, const char *filename,
        struct lint_config *config, cypher_parser_config_t *cp_config,
        const struct cypher_parser_colorization *error_colorization,
        const struct cypher_parser_colorization *output_colorization)
{
    struct parse_callback_data callback_data =
        { .filename = filename,
          .config = config,
          .error_colorization = error_colorization,
          .output_colorization = output_colorization,
          .nerrors = 0
        };

    if (cypher_fparse_each(stream, parse_callback, &callback_data, NULL,
                cp_config, config->flags))
    {
        perror("cypher_fparse_each");
        return -1;
    }

    return (callback_data.nerrors == 0)? 0 : 1;
}


int parse_callback(void *data, cypher_parse_segment_t *segment)
{
    struct parse_callback_data *cbdata =
            (struct parse_callback_data *)data;
    struct lint_config *config = cbdata->config;

    unsigned int i = 0;
    const cypher_parse_error_t *error;
    for (; (error = cypher_parse_segment_get_error(segment, i)) != NULL; ++i)
    {
        print_error(error, cbdata->filename, cbdata->error_colorization);
    }

    cbdata->nerrors += i;

    if (config->dump_ast && cypher_parse_segment_fprint_ast(segment, stdout,
            config->width, cbdata->output_colorization, 0) < 0)
    {
        perror("cypher_parse_segment_fprint_ast");
        return -1;
    }

    return 0;
}


int process_all(FILE *stream, const char *filename,
        struct lint_config *config, cypher_parser_config_t *cp_config,
        const struct cypher_parser_colorization *error_colorization,
        const struct cypher_parser_colorization *output_colorization)
{
    cypher_parse_result_t *result =
            cypher_fparse(stream, NULL, cp_config, config->flags);
    if (result == NULL)
    {
        perror("cypher_fparse");
        return -1;
    }

    int err = -1;

    unsigned int i = 0;
    const cypher_parse_error_t *error;
    for (; (error = cypher_parse_result_get_error(result, i)) != NULL; ++i)
    {
        print_error(error, filename, error_colorization);
    }

    if (config->dump_ast)
    {
        if (filename != NULL)
        {
            printf("%s:\n", filename);
        }
        if (cypher_parse_result_fprint_ast(result, stdout,
                config->width, output_colorization, 0) < 0)
        {
            perror("cypher_parse_result_fprint_ast");
            goto cleanup;
        }
    }

    err = (cypher_parse_result_nerrors(result) == 0)? 0 : 1;

    int errsv;
cleanup:
    errsv = errno;
    cypher_parse_result_free(result);
    errno = errsv;
    return err;
}


void print_error(const cypher_parse_error_t *error, const char *filename,
        const struct cypher_parser_colorization *colorization)
{
    struct cypher_input_position pos = cypher_parse_error_position(error);
    const char *msg = cypher_parse_error_message(error);
    const char *context = cypher_parse_error_context(error);
    unsigned int offset = cypher_parse_error_context_offset(error);
    fprintf(stderr, "%s:%u:%u: %s\n", (filename != NULL)? filename : "<stdin>",
            pos.line, pos.column, msg);
    fprintf(stderr, "%s\n%*.*s^\n", context, offset, offset, " ");
}
