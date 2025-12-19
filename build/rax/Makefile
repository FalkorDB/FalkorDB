
ROOT=../..
include $(ROOT)/deps/readies/mk/main

define HELPTEXT
make build    # configure and compile
make clean    # clean generated sbinaries
  ALL=1       # remote entire binary directory
endef

MK_ALL_TARGETS=build

#----------------------------------------------------------------------------------------------

BINDIR=$(BINROOT)/rax
SRCDIR=$(ROOT)/deps/rax

TARGET=$(BINDIR)/librax.a

#----------------------------------------------------------------------------------------------

MK_CUSTOM_CLEAN=1

include $(MK)/defs

_SOURCES=rax.c

SOURCES=$(addprefix $(SRCDIR)/,$(_SOURCES))
OBJECTS=$(patsubst $(SRCDIR)/%.c,$(BINDIR)/%.o,$(SOURCES))

CC_DEPS = $(patsubst $(SRCDIR)/%.c, $(BINDIR)/%.d, $(SOURCES))

CC_FLAGS += \
	-fPIC \
	-std=gnu99 \
	 -fvisibility=hidden \
	-MMD -MF $(@:.o=.d) \
	-I$(SRCDIR) \
	-I$(BINDIR) \
	-I$(LIBSSL_PREFIX)/include \
	$(CC_FLAGS.coverage)

ifeq ($(DEBUG),1)
CC_FLAGS += -g -O0
LD_FLAGS += -g
else
CC_FLAGS += -O3 -Wno-unused-result
LD_FLAGS += -lto
endif

ifeq ($(OS),macos)
LD_FLAGS += -undefined dynamic_lookup
endif

LD_FLAGS += $(LD_FLAGS.coverage)

#----------------------------------------------------------------------------------------------

include $(MK)/rules

-include $(CC_DEPS)

$(BINDIR)/%.o: $(SRCDIR)/%.c
	@echo Compiling $<...
	$(SHOW)$(CC) $(CC_FLAGS) -c $< -o $@

$(TARGET): $(OBJECTS)
	@echo Creating $@...
	$(SHOW)$(AR) rcs $@ $(OBJECTS)

clean:
ifeq ($(ALL),1)
	$(SHOW)rm -rf $(BINDIR)
else
	$(SHOW)rm -f $(OBJECTS) $(TARGET)
endif
