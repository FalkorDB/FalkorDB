# -*- coding: utf-8 -*-

# Copyright (c) The python-semanticversion project
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import functools
import re
import warnings


def _has_leading_zero(value):
    return (value
            and value[0] == '0'
            and value.isdigit()
            and value != '0')


class MaxIdentifier(object):
    __slots__ = []  # type: ignore

    def __repr__(self):
        return 'MaxIdentifier()'

    def __eq__(self, other):
        return isinstance(other, self.__class__)


@functools.total_ordering
class NumericIdentifier(object):
    __slots__ = ['value']

    def __init__(self, value):
        self.value = int(value)

    def __repr__(self):
        return 'NumericIdentifier(%r)' % self.value

    def __eq__(self, other):
        if isinstance(other, NumericIdentifier):
            return self.value == other.value
        return NotImplemented

    def __lt__(self, other):
        if isinstance(other, MaxIdentifier):
            return True
        elif isinstance(other, AlphaIdentifier):
            return True
        elif isinstance(other, NumericIdentifier):
            return self.value < other.value
        else:
            return NotImplemented


@functools.total_ordering
class AlphaIdentifier(object):
    __slots__ = ['value']

    def __init__(self, value):
        self.value = value.encode('ascii')

    def __repr__(self):
        return 'AlphaIdentifier(%r)' % self.value

    def __eq__(self, other):
        if isinstance(other, AlphaIdentifier):
            return self.value == other.value
        return NotImplemented

    def __lt__(self, other):
        if isinstance(other, MaxIdentifier):
            return True
        elif isinstance(other, NumericIdentifier):
            return False
        elif isinstance(other, AlphaIdentifier):
            return self.value < other.value
        else:
            return NotImplemented


class Version(object):

    version_re = re.compile(r'^(\d+)\.(\d+)\.(\d+)(?:-([0-9a-zA-Z.-]+))?(?:\+([0-9a-zA-Z.-]+))?$')
    partial_version_re = re.compile(r'^(\d+)(?:\.(\d+)(?:\.(\d+))?)?(?:-([0-9a-zA-Z.-]*))?(?:\+([0-9a-zA-Z.-]*))?$')

    def __init__(
            self,
            version_string=None,
            major=None,
            minor=None,
            patch=None,
            prerelease=None,
            build=None,
            partial=True):
        has_text = version_string is not None
        has_parts = not (major is minor is patch is prerelease is build is None)
        if not has_text ^ has_parts:
            raise ValueError("Call either Version('1.2.3') or Version(major=1, ...).")

        if has_text:
            major, minor, patch, prerelease, build = self.parse(version_string, partial)
        else:
            # Convenience: allow to omit prerelease/build.
            prerelease = tuple(prerelease or ())
            if not partial:
                build = tuple(build or ())
            self._validate_kwargs(major, minor, patch, prerelease, build, partial)

        self.major = major
        self.minor = minor
        self.patch = patch
        self.prerelease = prerelease
        self.build = build

        self.partial = partial

    @classmethod
    def _coerce(cls, value, allow_none=False):
        if value is None and allow_none:
            return value
        return int(value)

    def next_major(self):
        if self.prerelease and self.minor == self.patch == 0:
            return Version(
                major=self.major,
                minor=0,
                patch=0,
                partial=self.partial,
            )
        else:
            return Version(
                major=self.major + 1,
                minor=0,
                patch=0,
                partial=self.partial,
            )

    def next_minor(self):
        if self.prerelease and self.patch == 0:
            return Version(
                major=self.major,
                minor=self.minor,
                patch=0,
                partial=self.partial,
            )
        else:
            return Version(
                major=self.major,
                minor=self.minor + 1,
                patch=0,
                partial=self.partial,
            )

    def next_patch(self):
        if self.prerelease:
            return Version(
                major=self.major,
                minor=self.minor,
                patch=self.patch,
                partial=self.partial,
            )
        else:
            return Version(
                major=self.major,
                minor=self.minor,
                patch=self.patch + 1,
                partial=self.partial,
            )

    def truncate(self, level='patch'):
        """Return a new Version object, truncated up to the selected level."""
        if level == 'build':
            return self
        elif level == 'prerelease':
            return Version(
                major=self.major,
                minor=self.minor,
                patch=self.patch,
                prerelease=self.prerelease,
                partial=self.partial,
            )
        elif level == 'patch':
            return Version(
                major=self.major,
                minor=self.minor,
                patch=self.patch,
                partial=self.partial,
            )
        elif level == 'minor':
            return Version(
                major=self.major,
                minor=self.minor,
                patch=None if self.partial else 0,
                partial=self.partial,
            )
        elif level == 'major':
            return Version(
                major=self.major,
                minor=None if self.partial else 0,
                patch=None if self.partial else 0,
                partial=self.partial,
            )
        else:
            raise ValueError("Invalid truncation level `%s`." % level)

    @classmethod
    def coerce(cls, version_string, partial=True):
        """Coerce an arbitrary version string into a semver-compatible one.

        The rule is:
        - If not enough components, fill minor/patch with zeroes; unless
          partial=True
        - If more than 3 dot-separated components, extra components are "build"
          data. If some "build" data already appeared, append it to the
          extra components

        Examples:
            >>> Version.coerce('0.1')
            Version(0, 1, 0)
            >>> Version.coerce('0.1.2.3')
            Version(0, 1, 2, (), ('3',))
            >>> Version.coerce('0.1.2.3+4')
            Version(0, 1, 2, (), ('3', '4'))
            >>> Version.coerce('0.1+2-3+4_5')
            Version(0, 1, 0, (), ('2-3', '4-5'))
        """
        base_re = re.compile(r'^\d+(?:\.\d+(?:\.\d+)?)?')

        match = base_re.match(version_string)
        if not match:
            raise ValueError(
                "Version string lacks a numerical component: %r"
                % version_string
            )

        version = version_string[:match.end()]
        if not partial:
            # We need a not-partial version.
            while version.count('.') < 2:
                version += '.0'

        # Strip leading zeros in components
        # Version is of the form nn, nn.pp or nn.pp.qq
        version = '.'.join(
            # If the part was '0', we end up with an empty string.
            part.lstrip('0') or '0'
            for part in version.split('.')
        )

        if match.end() == len(version_string):
            return Version(version, partial=partial)

        rest = version_string[match.end():]

        # Cleanup the 'rest'
        rest = re.sub(r'[^a-zA-Z0-9+.-]', '-', rest)

        if rest[0] == '+':
            # A 'build' component
            prerelease = ''
            build = rest[1:]
        elif rest[0] == '.':
            # An extra version component, probably 'build'
            prerelease = ''
            build = rest[1:]
        elif rest[0] == '-':
            rest = rest[1:]
            if '+' in rest:
                prerelease, build = rest.split('+', 1)
            else:
                prerelease, build = rest, ''
        elif '+' in rest:
            prerelease, build = rest.split('+', 1)
        else:
            prerelease, build = rest, ''

        build = build.replace('+', '.')

        if prerelease:
            version = '%s-%s' % (version, prerelease)
        if build:
            version = '%s+%s' % (version, build)

        return cls(version, partial=partial)

    @classmethod
    def parse(cls, version_string, partial=True, coerce=False):
        """Parse a version string into a tuple of components:
           (major, minor, patch, prerelease, build).

        Args:
            version_string (str), the version string to parse
            partial (bool), whether to accept incomplete input
            coerce (bool), whether to try to map the passed in string into a
                valid Version.
        """
        if not version_string:
            raise ValueError('Invalid empty version string: %r' % version_string)

        if partial:
            version_re = cls.partial_version_re
        else:
            version_re = cls.version_re

        match = version_re.match(version_string)
        if not match:
            raise ValueError('Invalid version string: %r' % version_string)

        major, minor, patch, prerelease, build = match.groups()

        if _has_leading_zero(major):
            raise ValueError("Invalid leading zero in major: %r" % version_string)
        if _has_leading_zero(minor):
            raise ValueError("Invalid leading zero in minor: %r" % version_string)
        if _has_leading_zero(patch):
            raise ValueError("Invalid leading zero in patch: %r" % version_string)

        major = int(major)
        minor = cls._coerce(minor, partial)
        patch = cls._coerce(patch, partial)

        if prerelease is None:
            if partial and (build is None):
                # No build info, strip here
                return (major, minor, patch, None, None)
            else:
                prerelease = ()
        elif prerelease == '':
            prerelease = ()
        else:
            prerelease = tuple(prerelease.split('.'))
            cls._validate_identifiers(prerelease, allow_leading_zeroes=False)

        if build is None:
            if partial:
                build = None
            else:
                build = ()
        elif build == '':
            build = ()
        else:
            build = tuple(build.split('.'))
            cls._validate_identifiers(build, allow_leading_zeroes=True)

        return (major, minor, patch, prerelease, build)

    @classmethod
    def _validate_identifiers(cls, identifiers, allow_leading_zeroes=False):
        for item in identifiers:
            if not item:
                raise ValueError(
                    "Invalid empty identifier %r in %r"
                    % (item, '.'.join(identifiers))
                )

            if item[0] == '0' and item.isdigit() and item != '0' and not allow_leading_zeroes:
                raise ValueError("Invalid leading zero in identifier %r" % item)

    @classmethod
    def _validate_kwargs(cls, major, minor, patch, prerelease, build, partial):
        if (
                major != int(major)
                or minor != cls._coerce(minor, partial)
                or patch != cls._coerce(patch, partial)
                or prerelease is None and not partial
                or build is None and not partial
        ):
            raise ValueError(
                "Invalid kwargs to Version(major=%r, minor=%r, patch=%r, "
                "prerelease=%r, build=%r, partial=%r" % (
                    major, minor, patch, prerelease, build, partial
                ))
        if prerelease is not None:
            cls._validate_identifiers(prerelease, allow_leading_zeroes=False)
        if build is not None:
            cls._validate_identifiers(build, allow_leading_zeroes=True)

    def __iter__(self):
        return iter((self.major, self.minor, self.patch, self.prerelease, self.build))

    def __str__(self):
        version = '%d' % self.major
        if self.minor is not None:
            version = '%s.%d' % (version, self.minor)
        if self.patch is not None:
            version = '%s.%d' % (version, self.patch)

        if self.prerelease or (self.partial and self.prerelease == () and self.build is None):
            version = '%s-%s' % (version, '.'.join(self.prerelease))
        if self.build or (self.partial and self.build == ()):
            version = '%s+%s' % (version, '.'.join(self.build))
        return version

    def __repr__(self):
        return '%s(%r%s)' % (
            self.__class__.__name__,
            str(self),
            ', partial=True' if self.partial else '',
        )

    def __hash__(self):
        # We don't include 'partial', since this is strictly equivalent to having
        # at least a field being `None`.
        return hash((self.major, self.minor, self.patch, self.prerelease, self.build))

    @property
    def precedence_key(self):
        def denone(n):
            return 0 if n is None else n

        if self.prerelease:
            prerelease_key = tuple(
                NumericIdentifier(part) if re.match(r'^[0-9]+$', part) else AlphaIdentifier(part)
                for part in self.prerelease
            )
        else:
            prerelease_key = (
                MaxIdentifier(),
            )

        return (
            denone(self.major),
            denone(self.minor),
            denone(self.patch),
            prerelease_key,
        )

    def __cmp__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        if self < other:
            return -1
        elif self > other:
            return 1
        elif self == other:
            return 0
        else:
            return NotImplemented

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return (
            self.major == other.major
            and self.minor == other.minor
            and self.patch == other.patch
            and (self.prerelease or ()) == (other.prerelease or ())
            and (self.build or ()) == (other.build or ())
        )

    def __ne__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return tuple(self) != tuple(other)

    def __lt__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.precedence_key < other.precedence_key

    def __le__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.precedence_key <= other.precedence_key

    def __gt__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.precedence_key > other.precedence_key

    def __ge__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.precedence_key >= other.precedence_key
