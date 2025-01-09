
.NOTPARALLEL:

ROOT=../..
include $(ROOT)/deps/readies/mk/main

MK.configure=1

CONFIGURE_FLAGS.$(OS)=            \
	--with-openssl                \
	--enable-static               \
	--without-zlib                \
	--without-libidn2             \
	--without-libpsl              \
	--without-brotli              \
	--without-librtmp             \
	--without-winidn              \
	--without-libgsasl            \
	--without-nghttp2             \
	--disable-zlib                \
	--disable-debug               \
	--disable-curldebug           \
	--disable-ldap                \
	--disable-shared              \
	--disable-manual              \
	--disable-docs                \
	--disable-rt                  \
	--disable-httpsrr             \
	--disable-threaded-resolver   \
	--disable-ech                 \
	--disable-ftp                 \
	--disable-file                \
	--disable-ipfs                \
	--disable-ldap                \
	--disable-ldaps               \
	--disable-rtsp                \
	--disable-proxy               \
	--disable-dict                \
	--disable-telnet              \
	--disable-tftp                \
	--disable-pop3                \
	--disable-imap                \
	--disable-smb                 \
	--disable-smtp                \
	--disable-gopher              \
	--disable-mqtt                \
	--disable-ca-search           \
	--disable-windows-unicode     \
	--disable-sspi                \
	--disable-kerberos-auth       \
	--disable-ntlm                \
	--disable-cookies             \
	--disable-doh                 \
	--disable-form-api            \
	--disable-alt-svc             \
	--disable-headers-api         \
	--disable-hsts                \
	--disable-netrc               \

define HELPTEXT
make build    # configure and compile
make clean    # clean generated sbinaries
  ALL=1       # remote entire binary directory
make source   # fetch sources and generate configure script
endef

#----------------------------------------------------------------------------------------------

MK_ALL_TARGETS=bindirs build

BINDIR=$(BINROOT)/libcurl
SRCDIR=$(ROOT)/deps/libcurl

TARGET=$(BINDIR)/lib/.libs/libcurl.a

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
	@echo Running configure
	$(SHOW)cd $(SRCDIR); autoreconf -fi

.PHONY: autoreconf

clean:
ifeq ($(ALL),1)
	$(SHOW)-rm -rf $(CONFIGURE_BUILD_DIR)
else
	$(SHOW)[[ -d $(CONFIGURE_BUILD_DIR) ]] && $(MAKE) -C $(CONFIGURE_BUILD_DIR) clean
endif

.PHONY: clean
