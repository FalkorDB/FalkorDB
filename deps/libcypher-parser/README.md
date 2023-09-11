libcypher-parser
================


About
-----

libcypher-parser is a parser library and validation (lint) tool for Cypher, the
graph query language. The parser is written in C, and is intended for building
tools and libraries in various languages.

For more details, see [the project page](https://git.io/libcypher-parser)
and the [FAQ](https://github.com/cleishm/libcypher-parser/wiki/FAQ)


Requirements
------------

libcypher-parser is known to work on GNU/Linux, Mac OS X and FreeBSD.

Note that libcypher-parser is still a beta release, and may change in
incompatible ways before a stable release is made.


Getting Started
---------------

If you're using Mac OS X, libcypher-parser can be installed using homebrew:

```console
$ brew install cleishm/neo4j/libcypher-parser
```

or

```console
$ brew install cleishm/neo4j/cypher-lint
```

If you're using Ubuntu, libcypher-parser can be install using APT:

```console
$ sudo add-apt-repository ppa:cleishm/neo4j
$ sudo apt-get update
$ sudo apt-get install cypher-lint libcypher-parser-dev
```

For other platforms, please see [Building](#building) below.


cypher-lint Usage
-----------------

cypher-lint is a parser and linter for Cypher. It will parse a sequence of
cypher statements from stdin and report any parse errors encountered.
Optionally, it can also output an abstract syntax tree (AST) representation of
the input.

Basic usage:

```console
$ cypher-lint -a < sample.cyp
Invalid input 'S': expected AS (line 3, column 17, offset 124):
RETURN s.name ASS name;
                ^
 @0    2..45   line_comment             // Return all the software that cleishm wrote
 @1   46..132  statement                body=@2
 @2   46..132  > query                  clauses=[@3, @17]
 @3   46..108  > > MATCH                pattern=@4
 @4   52..107  > > > pattern            paths=[@5]
 @5   52..107  > > > > pattern path     (@6)-[@12]-(@14)
 @6   52..84   > > > > > node pattern   (@7:@8 {@9})
 @7   53..54   > > > > > > identifier   `n`
 @8   54..61   > > > > > > label        :`Person`
 @9   62..83   > > > > > > map          {@10:@11}
@10   63..71   > > > > > > > prop name  `username`
@11   73..82   > > > > > > > string     "cleishm"
@12   84..95   > > > > > rel pattern    -[:@13]->
@13   86..92   > > > > > > rel type     :`WROTE`
@14   95..107  > > > > > node pattern   (@15:@16)
@15   96..97   > > > > > > identifier   `s`
@16   97..106  > > > > > > label        :`Software`
@17  108..122  > > RETURN               items=[@18]
@18  115..122  > > > projection         expression=@19, alias=@22
@19  115..122  > > > > property         @20.@21
@20  115..116  > > > > > identifier     `s`
@21  117..121  > > > > > prop name      `name`
@22  115..122  > > > > identifier       `s.name`
@23  122..132  > > error                >>ASS name;\n<<
```

### Editor integration

- Vim and NeoVim
  - The [ALE] linting plugin provides support for `cypher-lint`, it just needs
    to be available in your `$PATH`. ALE will provide instant feedback for
    errors when editing `cypher` files.

[ALE]: https://github.com/w0rp/ale


libcypher-parser Usage
----------------------

libcypher-parser provides a single C header file, `cypher-parser.h`, for
inclusion in source code using the libcypher-parser API. The API is described in
the [API Documentation](#api_documentation).

libcypher-parser can be included in your project by linking the library at
compile time, typically using the linking flag `-lcypher-parser`.
Alternatively, libcypher-parser ships with a [pkg-config](
https://wiki.freedesktop.org/www/Software/pkg-config/)
description file, enabling you to obtain the required flags using
`pkg-config --libs libcypher-parser`.


API Documentation
-----------------

API documentation for the latest release is available at
[https://cleishm.github.io/libcypher-parser/doc/latest/cypher-parser\_8h.html](
[https://cleishm.github.io/libcypher-parser/doc/latest/cypher-parser_8h.html).

Documentation can be built using `make doc`, which will use doxygen to generate
documentation and output it into the `doc/` directory of the libcypher-parser
source tree. See [Building](#building) below.


Example
-------

```C
#include <cypher-parser.h>
#include <errno.h>
#include <stdio.h>

int main(int argc, char *argv[])
{
    cypher_parse_result_t *result = cypher_parse(
            "MATCH (n) RETURN n", NULL, NULL, CYPHER_PARSE_ONLY_STATEMENTS);
    if (result == NULL)
    {
        perror("cypher_parse");
        return EXIT_FAILURE;
    }

    printf("Parsed %d AST nodes\n", cypher_parse_result_nnodes(result));
    printf("Read %d statements\n", cypher_parse_result_ndirectives(result));
    printf("Encountered %d errors\n", cypher_parse_result_nerrors(result));

    cypher_parse_result_free(result);
    return EXIT_SUCCESS;
}
```


Building
--------

To build software using libcypher-parser, consider installing libcypher-parser
using the package management system for your operating system (currently
[Mac OS X](#getting_started),
[Debian](https://mentors.debian.net/package/libcypher-parser) and
[Ubuntu](#getting_started)).

If libcypher-parser is not available via your package management system,
please [download the latest release](
https://github.com/cleishm/libcypher-parser/releases), unpack and then:

```console
$ ./configure
$ make clean check
$ sudo make install
```

Building from the GitHub repository requires a few extra steps. Firstly, some
additional tooling is required, including autoconf, automake, libtool and
[peg/leg](http://piumarta.com/software/peg/). Assuming these are available,
to checkout from GitHub and build:

```console
$ git clone https://github.com/cleishm/libcypher-parser.git
$ cd libcypher-parser
$ ./autogen.sh
$ ./configure
$ make clean check
$ sudo make install
```

If you encounter warnings or errors during the build, please report them at
https://github.com/cleishm/libneo4j-client/issues. If you wish to proceed
dispite warnings, please invoke configure with the `--disable-werror`.


Support
-------

Having trouble with libcypher-parser? Please raise any issues with usage on
[StackOverflow](http://stackoverflow.com/questions/tagged/libcypher-parser). If
you've found a bug in the code for libcypher-parser, please raise an issue on
[GitHub](https://github.com/cleishm/libcypher-parser) and include details of how
to reproduce the bug.


Contributing
------------

Contributions to libcypher-parser and cypher-lint are needed! Contributions
should be made via pull requests made to the [GitHub repository](
https://github.com/cleishm/libcypher-parser). Please include test cases where
possible, and use a style and approach consistent with the rest of the library.

For a list of contributions that would be greatly appreciated, check out
the [TODO list](https://github.com/cleishm/libcypher-parser/wiki/TODO).

For other contributions, it may be worthwhile raising an issue on github for
what you intend to do before developing the code, in order to allow for
discussion and feedback on the requirements.


License
-------

libcypher-parser is licensed under the [Apache License, Version 2.0](
http://www.apache.org/licenses/LICENSE-2.0).

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied.  See the License for the
specific language governing permissions and limitations under the License.
