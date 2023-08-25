ACLOCAL_AMFLAGS = -I build-aux/m4
SUBDIRS = lib

if WANT_TOOLS
SUBDIRS += linter
endif

EXTRA_DIST = \
	LICENSE \
	README.md

docker-check:
	@$(MAKE) dist
	@echo "Building docker image..."
	@docker run -t --rm -v $(abs_builddir):/$(PACKAGE) \
		`docker build . -q -f build-aux/build.dockerfile` \
		sh /$(PACKAGE)/build-aux/check-package.sh /$(PACKAGE)/$(DIST_ARCHIVES) $(distdir)

docker-valgrind:
	@$(MAKE) dist
	@echo "Building docker image..."
	@docker run --rm -t -v $(abs_builddir):/$(PACKAGE) \
		`docker build . -q -f build-aux/build.dockerfile` \
		sh /$(PACKAGE)/build-aux/valgrind-package.sh /$(PACKAGE)/$(DIST_ARCHIVES) $(distdir)

MAINTAINERCLEANFILES = \
	build-aux/m4/libtool.m4 \
	build-aux/m4/ltoptions.m4 \
	build-aux/m4/ltsugar.m4 \
	build-aux/m4/ltversion.m4 \
	build-aux/m4/lt~obsolete.m4
