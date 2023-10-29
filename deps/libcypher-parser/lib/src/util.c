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
#include "util.h"
#include <assert.h>
#include <stdarg.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>


ssize_t snprintf_realloc(char ** restrict buf, size_t *bufcap,
        const char * restrict format, ...)
{
    va_list ap;
    va_start(ap, format);
    ssize_t width = vsnprintf(*buf, *bufcap, format, ap);
    va_end(ap);
    if (width < 0)
    {
        return -1;
    }
    if ((size_t)width > *bufcap)
    {
        char *newbuf = realloc(*buf, (size_t)width+1);
        if (newbuf == NULL)
        {
            return -1;
        }
        *buf = newbuf;
        *bufcap = (size_t)width+1;
        va_start(ap, format);
        width = vsnprintf(*buf, *bufcap, format, ap);
        va_end(ap);
    }
    return width;
}


char *line_context(const char *buf, size_t bufsize, size_t *offset,
        size_t max_length)
{
    REQUIRE(max_length >= 7, NULL);
    REQUIRE(bufsize > 0, NULL);
    REQUIRE(*offset <= bufsize, NULL);

    assert(*offset <= bufsize);
    const char *errp = buf + *offset;
    const char *buf_end = buf + bufsize;

    if (errp >= buf_end)
    {
        errp = buf_end-1;
    }
    while (errp > buf && (*errp == '\n' || *errp == '\r' || *errp == '\0'))
    {
        --errp;
    }

    unsigned int n = 0;
    const char *startp = errp;
    const char *endp = startp;
    bool found_start = false;
    bool found_end = false;
    while (!found_start || !found_end)
    {
        if (!found_start)
        {
            if (startp == buf || *(startp-1) == '\n' || *(startp-1) == '\r')
            {
                found_start = true;
            }
            else
            {
                --startp;
                if (++n == max_length)
                {
                    break;
                }
            }
        }

        if (!found_end)
        {
            if (endp == buf_end || *endp == '\n' || *endp == '\r')
            {
                found_end = true;
            }
            else
            {
                ++endp;
                if (++n == max_length)
                {
                    break;
                }
            }
        }
    }

    assert((unsigned int)(endp - startp) == n);
    assert(n <= max_length);
    char *context = malloc(n + 1);
    if (context == NULL)
    {
        return NULL;
    }
    memcpy(context, startp, n);
    context[n] = '\0';

    if (!found_start)
    {
        assert(n > 3);
        context[0] = '.';
        context[1] = '.';
        context[2] = '.';
    }

    if (!found_end)
    {
        assert(n > 3);
        context[n-1] = '.';
        context[n-2] = '.';
        context[n-3] = '.';
    }

    *offset -= (startp - buf);
    return context;
}
