
.NOTPARALLEL:

ROOT=../..
include $(ROOT)/deps/readies/mk/main

MK.configure=1

define HELPTEXT
make build    # configure and compile
make clean    # clean generated sbinaries
  ALL=1       # remote entire binary directory
make source   # fetch sources and generate configure script
endef

#----------------------------------------------------------------------------------------------

MK_ALL_TARGETS=bindirs build

BINDIR=$(BINROOT)/libcsv
SRCDIR=$(ROOT)/deps/libcsv

TARGET=$(BINDIR)/.libs/libcsv.a

CFLAGS += \
	-fPIC \
	-O3 \
	$(CFLAGS.$(OS))

export CFLAGS

#----------------------------------------------------------------------------------------------

MK_CUSTOM_CLEAN=1

include $(MK)/defs
include $(MK)/rules

#----------------------------------------------------------------------------------------------

autoreconf:
	@echo Running autoreconf
	$(SHOW)cd $(SRCDIR); autoreconf

.PHONY: autoreconf

clean:
ifeq ($(ALL),1)
	$(SHOW)-rm -rf $(CONFIGURE_BUILD_DIR)
else
	$(SHOW)[[ -d $(CONFIGURE_BUILD_DIR) ]] && $(MAKE) -C $(CONFIGURE_BUILD_DIR) clean
endif

.PHONY: clean
