AC_DEFUN([WITH_LIBS],
[AC_ARG_WITH([libs],
  [AC_HELP_STRING(
    [[--with-libs=DIR[:DIR]]],
    [look for libraries installed in DIR/{bin,lib,include}])],
  [neo4j_save_ifs=$IFS
  IFS=:
  for dir in $with_libs; do
    if ! test -d "$dir"; then
      IFS=$neo4j_save_ifs
      AC_MSG_ERROR([lib directory "$dir": not found])
    fi
    CPPFLAGS="-I${dir}/include $CPPFLAGS"
    LDFLAGS="-L${dir}/lib $LDFLAGS"
    PATH="${dir}/bin:$PATH"
    echo "SET path to $PATH"
    PKG_CONFIG_PATH="${PKG_CONFIG_PATH}${PKG_CONFIG_PATH:+:}${dir}/lib/pkgconfig"
  done
  IFS=$neo4j_save_ifs
  ])
])
