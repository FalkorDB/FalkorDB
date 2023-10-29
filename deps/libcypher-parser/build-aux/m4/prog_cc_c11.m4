AC_DEFUN([PROG_CC_C11],
[AC_CACHE_CHECK([for the _AC_LANG option to accept ISO C11],
  [neo4j_cv_prog_cc_c11], [
  neo4j_save_flags=$[]_AC_LANG_PREFIX[]FLAGS
  _AC_LANG_PREFIX[]FLAGS="$neo4j_save_flags -std=gnu11"
  AC_COMPILE_IFELSE([AC_LANG_PROGRAM()],
    [neo4j_cv_prog_cc_c11="-std=gnu11"], [
    _AC_LANG_PREFIX[]FLAGS="$neo4j_save_flags -std=c11"
    AC_COMPILE_IFELSE([AC_LANG_PROGRAM()],
      [neo4j_cv_prog_cc_c11="-std=c11"],
      [neo4j_cv_prog_cc_c11="none found"])])
  _AC_LANG_PREFIX[]FLAGS=$neo4j_save_flags])

  if test "X$neo4j_cv_prog_cc_c11" != "Xnone found"; then
    CFLAGS="$CFLAGS $neo4j_cv_prog_cc_c11"
  fi
])
