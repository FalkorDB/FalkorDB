ROOT=../..
include $(ROOT)/deps/readies/mk/main

#----------------------------------------------------------------------------------------------

SRCDIR=$(ROOT)/deps/quickjs
BINDIR=$(BINROOT)/quickjs
TARGET=$(BINDIR)/libquickjs.a

#----------------------------------------------------------------------------------------------

MK_CUSTOM_CLEAN=1

include $(MK)/defs
include $(MK)/rules

#----------------------------------------------------------------------------------------------

# Build libquickjs.a with renamed conflicting symbols
$(TARGET): | $(BINDIR)
	@echo "Building QuickJS library..."
	# Clean first to ensure fresh build
	$(SHOW)$(MAKE) -C $(SRCDIR) clean
	# Create backup of all source files that might contain the symbol
	$(SHOW)cd $(SRCDIR) && \
		find . -name "*.c" -o -name "*.h" | while read file; do \
			if grep -q "unicode_to_utf8" "$$file" 2>/dev/null; then \
				cp "$$file" "$$file.backup"; \
			fi; \
		done
	# Rename the conflicting function in all C and header files
	$(SHOW)cd $(SRCDIR) && \
		find . -name "*.c" -o -name "*.h" | xargs sed -i '' 's/unicode_to_utf8/quickjs_unicode_to_utf8/g'
	# Build the library
	$(SHOW)$(MAKE) -C $(SRCDIR) libquickjs.a
	$(SHOW)cp $(SRCDIR)/libquickjs.a $(TARGET)
	# Restore original source files
	$(SHOW)cd $(SRCDIR) && \
		find . -name "*.backup" | while read backup; do \
			original=$${backup%.backup}; \
			mv "$$backup" "$$original"; \
		done 2>/dev/null || \
		git checkout -- . 2>/dev/null || true

# Ensure output directory exists
$(BINDIR):
	$(SHOW)mkdir -p $(BINDIR)

# Default target
build: $(TARGET)

clean:
ifeq ($(ALL),1)
	$(SHOW)rm -rf $(BINDIR) $(TARGET)
else
	$(SHOW)rm -f $(TARGET) $(OBJECTS)
endif

.PHONY: clean
