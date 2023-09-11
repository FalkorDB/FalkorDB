# Copyright 2016, Chris Leishman (http://github.com/cleishm)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
dnl Process this file with autoconf to produce a configure script
AC_PREREQ([2.69])
AC_INIT([libcypher-parser],[0.6.3~devel])
AC_CONFIG_SRCDIR([lib/src/cypher-parser.h.in])
AC_CONFIG_AUX_DIR([build-aux])
AC_CONFIG_MACRO_DIR([build-aux/m4])
AM_INIT_AUTOMAKE([foreign])
AC_CONFIG_HEADER([config.h])
AM_MAINTAINER_MODE([enable])

DECLARE_VERSIONS
WITH_LIBS

GCC_CFLAGS="-fvisibility=hidden -pipe -Wall -W -Wno-unused-parameter \
    -Wno-missing-field-initializers -Wpointer-arith -Wstrict-prototypes \
    -Wcast-qual -Wcast-align -Wno-error=unused-function \
    -Wno-error=unused-variable -Wno-error=deprecated-declarations"

AC_ARG_ENABLE([werror],
  AS_HELP_STRING([--disable-werror],
  [Do not make warnings into errors even if the compiler supports -Werror.]),
  [werror_flag=$enableval], [werror_flag=yes])
AS_IF([test "X$werror_flag" = "Xyes"], [GCC_CFLAGS="$GCC_CFLAGS -Werror"])

LT_INIT
AC_LANG([C])
AC_PROG_CC
PROG_CC_C11
AC_PROG_CPP
AC_PROG_INSTALL
AC_PROG_LN_S
AC_PROG_MAKE_SET
AC_ARG_VAR([LEG], [leg parser generator command])
AC_CHECK_PROG([LEG], [leg], [leg])
AM_CONDITIONAL([HAVE_LEG], [test -n "$LEG"])

AS_IF([test -n "$LEG" -o -f ./lib/src/parser_leg.c], [],
  [AC_MSG_ERROR([leg is required: please install http://piumarta.com/software/peg/])])

AC_C_CONST
AC_C_INLINE
AC_C_VARARRAYS

AS_IF([test "X$ac_cv_c_vararrays" != "Xyes"],
  [AC_MSG_ERROR([compiler support for variable-length arrays is required])])

AC_HEADER_ASSERT
AC_HEADER_STDC
AC_HEADER_STDBOOL
AC_CHECK_HEADERS([endian.h sys/endian.h libkern/OSByteOrder.h])
AC_TYPE_SIZE_T
AC_TYPE_SSIZE_T
AC_FUNC_STRERROR_R
STATIC_ASSERT
LT_LIB_M


AC_CHECK_FUNCS([open_memstream],[],[],
  [
#include <stdio.h>
  ])


AX_THREAD_LOCAL()
AX_PTHREAD([has_pthreads=yes])
AS_IF([test "X$has_pthreads" = "Xyes"],
  [AC_DEFINE([HAVE_PTHREADS], [1], [Define to 1 if you have pthreads.])])


dnl Check for check test support
PKG_CHECK_MODULES([CHECK], [check >= 0.9.4], [has_check=yes], [has_check=no])
AM_CONDITIONAL([HAVE_CHECK], [test "X$has_check" = "Xyes"])

dnl Check if command line tools should be built
AC_ARG_ENABLE([tools],
  AS_HELP_STRING([--disable-tools], [Do not build command line tools]))
AM_CONDITIONAL([WANT_TOOLS], [test "X$enable_tools" != "Xno"])


dnl Check if assertions should be disabled
AC_ARG_ENABLE([debug], AS_HELP_STRING([--disable-debug], [Disable debug]))
AS_IF([test "X$enable_debug" = "Xno"],
  [AC_DEFINE([NDEBUG], [1], [Define to 1 to disable assertions.])])


dnl Add GCC flags
AS_IF([test "X$GCC" = "Xyes"], [CFLAGS="$CFLAGS $GCC_CFLAGS"])


dnl Capture required dependencies for pkg-check
AC_SUBST([REQUIRES], [`echo "$REQUIRES" | cut -c3-`])


dnl Check for doxygen support
DX_HTML_FEATURE(ON)
DX_CHM_FEATURE(OFF)
DX_CHI_FEATURE(OFF)
DX_MAN_FEATURE(ON)
DX_RTF_FEATURE(OFF)
DX_XML_FEATURE(OFF)
DX_PDF_FEATURE(ON)
DX_PS_FEATURE(OFF)
DX_INIT_DOXYGEN(libcypher-parser, doxygen.cfg, .)


AC_CONFIG_FILES([
    Makefile \
    lib/Makefile \
    lib/cypher-parser.pc \
    lib/doc/Makefile \
    lib/src/Makefile \
    lib/src/cypher-parser.h \
    lib/test/Makefile \
    linter/Makefile \
    linter/doc/Makefile \
    linter/doc/cypher-lint.1 \
    linter/src/Makefile
])
AC_OUTPUT
