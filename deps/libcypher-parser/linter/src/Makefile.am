bin_PROGRAMS = cypher-lint

cypher_lint_SOURCES = cypher-lint.c
cypher_lint_CPPFLAGS = -I$(top_srcdir)/lib/src
cypher_lint_LDADD = $(top_builddir)/lib/src/libcypher-parser.la ${LIBEDIT_LIBS}
