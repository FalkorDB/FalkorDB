
.NOTPARALLEL:

ROOT=../..
include $(ROOT)/deps/readies/mk/main

MK.configure=1

define HELPTEXT
make build    # configure and compile
make clean    # clean generated sbinaries
  ALL=1         # remote entire binary directory
  AUTOGEN=1     # clean non-git files from source dir (use with caution!)
make source   # fetch sources and generate configure script
endef

#----------------------------------------------------------------------------------------------

MK_ALL_TARGETS=bindirs build

BINDIR=$(BINROOT)/libcypher-parser
SRCDIR=$(ROOT)/deps/libcypher-parser

TARGET=$(BINDIR)/lib/src/.libs/libcypher-parser.a

CFLAGS += \
	-fPIC \
	-O3 \
	-DYY_BUFFER_SIZE=1048576 \
	-I$(BINDIR)/lib/src \
	$(CFLAGS.$(OS))

ifeq ($(GCC),1)
CFLAGS += -Wno-error=discarded-qualifiers
else
CFLAGS += -Wno-error=ignored-qualifiers -Wno-error=incompatible-pointer-types-discards-qualifiers
endif

export CFLAGS

#----------------------------------------------------------------------------------------------

MK_CUSTOM_CLEAN=1

include $(MK)/defs
include $(MK)/rules

#----------------------------------------------------------------------------------------------

source: $(SRCDIR)/configure

.PHONY: source

$(SRCDIR)/configure: $(SRCDIR)/configure.ac
	$(SHOW)cd $(SRCDIR); ./autogen.sh

clean:
ifeq ($(ALL),1)
	$(SHOW)-rm -rf $(CONFIGURE_BUILD_DIR)
else
	$(SHOW)[[ -d $(CONFIGURE_BUILD_DIR) ]] && $(MAKE) -C $(CONFIGURE_BUILD_DIR) clean
endif
ifeq ($(AUTOGEN),1)
	$(SHOW)cd $(SRCDIR); git ls-files -o | xargs rm
endif

.PHONY: clean
