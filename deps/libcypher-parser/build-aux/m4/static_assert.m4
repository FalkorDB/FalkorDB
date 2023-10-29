AC_DEFUN([STATIC_ASSERT],
  [AC_CACHE_CHECK([for static_assert],
    [neo4j_cv_macro_static_assert],
    [AC_COMPILE_IFELSE(
      [AC_LANG_PROGRAM([[
#include <assert.h>
static_assert(1, "impossible");
      ]],[[return 0;]])],
      [neo4j_cv_macro_static_assert=yes],
      [neo4j_cv_macro_static_assert=no])])

  if test "X$neo4j_cv_macro_static_assert" = "Xno"; then
    AC_DEFINE([static_assert(c,m)],[typedef void _no_static_assert], [Substitute for static_assert])
  fi
])
