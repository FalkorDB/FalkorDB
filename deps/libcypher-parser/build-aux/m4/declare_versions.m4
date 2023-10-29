AC_DEFUN([DECLARE_VERSIONS],
[
  if ! echo $VERSION | grep -Eq '^[[0-9]]+\.[[0-9]]+\.[[0-9]](~[[a-zA-Z0-9_]]+)?$'; then
    AC_MSG_ERROR([[Invalid version "$VERSION" specified in configure.ac. Must follow <major>.<minor>.<patch>[~<devel>] format.]])
  fi

  PACKAGE_MAJOR_VERSION=$(echo $VERSION | awk -F. '{print $[1]}')
  AC_DEFINE_UNQUOTED([PACKAGE_MAJOR_VERSION], $PACKAGE_MAJOR_VERSION,
    [Define to the major version of this package])
  AC_SUBST([PACKAGE_MAJOR_VERSION])

  PACKAGE_MINOR_VERSION=$(echo $VERSION | awk -F. '{print $[2]}')
  AC_DEFINE_UNQUOTED([PACKAGE_MINOR_VERSION], $PACKAGE_MINOR_VERSION,
    [Define to the minor version of this package])
  AC_SUBST([PACKAGE_MINOR_VERSION])

  PACKAGE_PATCH_VERSION=$(echo $VERSION | awk -F'[[.~]]' '{print $[3]}')
  AC_DEFINE_UNQUOTED([PACKAGE_PATCH_VERSION], $PACKAGE_PATCH_VERSION,
    [Define to the patch version of this package])
  AC_SUBST([PACKAGE_PATCH_VERSION])

  PACKAGE_DEVELOPMENT_VERSION=$(echo $VERSION | awk -F~ '{print $[2]}')
  AC_DEFINE_UNQUOTED([PACKAGE_DEVELOPMENT_VERSION], ["]$PACKAGE_DEVELOPMENT_VERSION["],
    [Define to the development version of this package])
  AC_SUBST([PACKAGE_DEVELOPMENT_VERSION])

  if test "X$PACKAGE_DEVELOPMENT_VERSION" = "X"; then
    PACKAGE_STATUS_VERSION=0
  else
    PACKAGE_STATUS_VERSION=1
  fi
  AC_SUBST([PACKAGE_STATUS_VERSION])
])
