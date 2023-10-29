include $(top_srcdir)/build-aux/automake/doxygen.am

if DX_COND_doc
all-local:
	$(MAKE) doxygen-doc
endif

libcypher-parser.tag: .deps/cypher-parser.h
.deps/cypher-parser.h: ../src/cypher-parser.h
	@mkdir -p .deps
	(cat "$<" && sed -ne "s/^.*@fn \(.*\)$$/\1;/p" < "$<") >> "$@"

EXTRA_DIST = doxygen.cfg

clean-local:
	rm -rf .deps
