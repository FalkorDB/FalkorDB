#!/bin/bash
# vi:ts=2 sw=2 et:

set -eu

if ! [ -d .git ]; then
  echo "Must be run from the project root" >&2
  exit 1
fi

if [ `git symbolic-ref HEAD` != 'refs/heads/master' ]; then
  echo "Must be run on the master branch" >&2
  exit 1
fi

status=`git status --porcelain`
if [ -n "$status" ]; then
  echo "Working directory is not clean" >&2
  exit 1
fi

ac_init=`grep -E 'AC_INIT\(\[[_a-zA-Z-]*\],\[[0-9][0-9]*\.[0-9][0-9]*\.[0-9][0-9]*~devel\]\)$' configure.ac`
if [ `echo $ac_init | wc -l` != 1 ]; then
  echo "Unrecognized or multiple AC_INIT entries in configure.ac" >&2
  exit 1
fi

PACKAGE=`echo $ac_init | sed -e 's/^.*AC_INIT(\[\(.*\)\],.*$/\1/'`
VERSION=`echo $ac_init | sed -e 's/^.*\[\(.*\)~devel\])$/\1/'`
TARBALL="$PACKAGE-$VERSION.tar.gz"

echo "Cleaning working tree"
git clean -fdxq

echo "Removing ~devel suffix from version in configure.ac"
sed -i '' -e "s/\(AC_INIT(.*\)~devel\(\])\)/\1\2/" configure.ac

version_info=`grep -e '-version-info [0-9][0-9]*:[0-9][0-9]*:[0-9][0-9]*$' lib/src/Makefile.am | sed -e 's/^.*-version-info //'`
version_current=`echo $version_info | awk -F: '{ print $1 }'`
version_revision=`echo $version_info | awk -F: '{ print $2 }'`
version_age=`echo $version_info | awk -F: '{ print $3 }'`
echo
echo "Current version-info: $version_info"
echo

while true; do
  read -p "Has the library library source code has changed at all since the last update? [y/n]" yn
  case $yn in
  [Nn]*) break;;
  [Yy]*)
    version_revision=`expr $version_revision + 1`
    while true; do
      read -p "Have any interfaces been added, removed, or changed since the last update? [y/n]" yn
      case $yn in
      [Nn]*) break;;
      [Yy]*)
        version_current=`expr $version_current + 1`
        version_revision=0
        while true; do
          read -p "Have any interfaces been added since the last public release? [y/n]" yn
          case $yn in
          [Nn]*) break;;
          [Yy]*) version_age=`expr $version_age + 1`; break;;
          *)
          esac
        done
        while true; do
          read -p "Have any interfaces been removed or changed since the last public release? [y/n]" yn
          case $yn in
          [Nn]*) break;;
          [Yy]*) version_age=0; break;;
          *)
          esac
        done
        break;;
      *)
      esac
    done
    break;;
  *)
  esac
done

echo "Updating version-info in lib/src/Makefile.am"
sed -i '' -e "s/-version-info $version_info$/-version-info $version_current:$version_revision:$version_age/" lib/src/Makefile.am

echo
echo "Version changes:"
git diff
echo
while true; do
  read -p "Commit? [y/n]" yn
  case $yn in
  [Nn]*) exit;;
  [Yy]*) break;;
  *)
  esac
done
git commit -a -m "Bump version"
git tag -s -m "Release $VERSION" v$VERSION

echo
echo "Building distribution"
./autogen.sh
./configure
make dist

echo
echo "Signing distribution tarball"
gpg --armour --detach-sig "$TARBALL"

echo
echo "Building debian docker image"
debian_docker=`docker build . -q -f build-aux/build-debian.dockerfile`
echo "Built image $debian_docker"

git checkout debian
git clean -fd -e "$TARBALL*"

echo
echo "Switching to debian branch and importing distribution tarball"
docker run --rm -v `pwd`:/home/build/dist -w /home/build/dist -i $debian_docker /bin/sh -s <<EOF
mv "$TARBALL"* ..
gbp import-orig -u$VERSION --no-interactive --rollback ../"$TARBALL"
mv ../"$TARBALL"* .
EOF

echo
echo "Distribution created and debian branch updated"

git checkout master
VERSION_PATCH=`echo $VERSION | awk -F. '{ print $3 }'`
NEXT_VERSION=`echo $VERSION | awk -F. '{ print $1 "." $2 "." }'``expr $VERSION_PATCH + 1`

echo "Setting next ~devel version in configure.ac"
sed -i '' -e "s/\(AC_INIT(\[.*,\[\).*\(\])\)/\1$NEXT_VERSION~devel\2/" configure.ac
git add configure.ac
git commit -m "Bump version"
