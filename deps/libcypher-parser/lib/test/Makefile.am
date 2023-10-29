TESTS = check_libcypher-parser
check_PROGRAMS = check_libcypher-parser
check_libcypher_parser_SOURCES = \
	${check_libcypher_parser_CHECKS} \
	check_libcypher-parser.c \
	check_libcypher-parser_suite.c \
	memstream.c \
	memstream.h

check_libcypher_parser_CHECKS = \
	check_annotation.c \
	check_call.c \
	check_case.c \
	check_command.c \
	check_constraints.c \
	check_create.c \
	check_delete.c \
	check_eof.c \
	check_error_tracking.c \
	check_errors.c \
	check_expression.c \
	check_foreach.c \
	check_indexes.c \
	check_list_comprehensions.c \
	check_load_csv.c \
	check_map_projection.c \
	check_match.c \
	check_merge.c \
	check_pattern.c \
	check_pattern_comprehension.c \
	check_query.c \
	check_quick_parse.c \
	check_quick_fparse.c \
	check_reduce.c \
	check_remove.c \
	check_return.c \
	check_segments.c \
	check_set.c \
	check_start.c \
	check_statement.c \
	check_union.c \
	check_unwind.c \
	check_util.c \
	check_with.c

check_libcypher-parser_suite.c: ${check_libcypher_parser_CHECKS}
	@(names=`for check in $^; do \
		basename $$check .c | sed -e 's/^check_//'; \
	done`; \
	echo "#include <check.h>"; \
	echo; \
	for name in $$names; do \
		echo "TCase* $${name}_tcase(void);"; \
	done; \
	echo; \
	echo "Suite *libcypher_parser_suite(void)"; \
	echo "{"; \
	echo "    Suite *s = suite_create(\"libcypher-parser\");"; \
	for name in $$names; do \
		echo "    suite_add_tcase(s, $${name}_tcase());"; \
	done; \
	echo "    return s;"; \
	echo "}") > $@

check_libcypher_parser_CFLAGS = @CHECK_CFLAGS@
check_libcypher_parser_LDFLAGS = -static
check_libcypher_parser_LDADD = ../src/libcypher-parser.la @CHECK_LIBS@

CLEANFILES = check_libcypher-parser_suite.c
