ROOT=../..

MK.cmake=1

include $(ROOT)/deps/readies/mk/main

define HELPTEXT
make build    # configure and compile
make clean    # clean generated sbinaries
  ALL=1       # remote entire binary directory
endef

#----------------------------------------------------------------------------------------------

BINDIR=$(BINROOT)/oniguruma
SRCDIR=$(ROOT)/deps/oniguruma

TARGET=$(BINDIR)/libonig.a

define CMAKE_DEFS +=
	BUILD_SHARED_LIBS=off
	BUILD_TEST=off
	CMAKE_POSITION_INDEPENDENT_CODE=on
endef

#----------------------------------------------------------------------------------------------

MK_CUSTOM_CLEAN=1

#----------------------------------------------------------------------------------------------

include $(MK)/defs
include $(MK)/rules


clean:
ifeq ($(ALL),1)
	$(SHOW)rm -rf $(BINDIR) $(TARGET)
else
	$(SHOW)rm -f $(TARGET) $(OBJECTS)
endif
