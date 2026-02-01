#!/bin/bash

PROGNAME="${BASH_SOURCE[0]}"
HERE="$(cd "$(dirname "$PROGNAME")" &>/dev/null && pwd)"
ROOT=$(cd $HERE/.. && pwd)

# Native helper functions (replacing readies)
eprint() {
    >&2 echo "$@"
}

runn() {
    if [[ $1 == "@" ]]; then
        # Run heredoc command
        shift
        local cmd
        cmd=$(cat)
        [[ $VERBOSE == 1 ]] && echo "$cmd"
        [[ $NOP != 1 ]] && eval "$cmd"
    else
        [[ $VERBOSE == 1 ]] && echo "$@"
        [[ $NOP != 1 ]] && "$@"
    fi
}

SBIN=$ROOT/sbin

export PYTHONWARNINGS=ignore

cd $ROOT

#----------------------------------------------------------------------------------------------

if [[ $1 == --help || $1 == help || $HELP == 1 ]]; then
	cat <<-END
		Generate FalkorDB distribution packages.

		[ARGVARS...] pack.sh [--help|help]

		Argument variables:
		MODULE=path         Path of module .so

		RAMP=0|1            Build RAMP package
		DEPS=0|1            Build dependencies files
		SYM=0|1             Build debug symbols file

		BRANCH=name         Branch name for snapshot packages
		WITH_GITSHA=1       Append Git SHA to shapshot package names
		VARIANT=name        Build variant
		RAMP_VARIANT=name   RAMP variant (e.g. ramp-{name}.yml)

		ARTDIR=dir          Directory in which packages are created (default: bin/artifacts)
		
		RAMP_YAML=path      RAMP configuration file path
		RAMP_ARGS=args      Extra arguments to RAMP

		JUST_PRINT=1        Only print package names, do not generate
		VERBOSE=1           Print commands
		HELP=1              Show help
		NOP=1               Print commands, do not execute

	END
	exit 0
fi

#----------------------------------------------------------------------------------------------

OP=""
[[ $NOP == 1 ]] && OP=echo

# Platform detection (native, replacing readies/bin/platform)
detect_platform() {
    # Detect OS
    local os_name=$(uname -s)
    case "$os_name" in
        Linux*) OS="linux" ;;
        Darwin*) OS="macos" ;;
        *) OS=$(echo "$os_name" | tr '[:upper:]' '[:lower:]') ;;
    esac

    # Detect architecture
    ARCH=$(uname -m)
    case "$ARCH" in
        x86_64) ARCH="x64" ;;
        aarch64|arm64) ARCH="aarch64" ;;
    esac

    # Detect OS nickname
    if [[ "$OS" == "linux" ]]; then
        if [[ -f /etc/os-release ]]; then
            . /etc/os-release
            OSNICK=$(echo "$ID$VERSION_ID" | tr '[:upper:]' '[:lower:]')
        else
            OSNICK="linux"
        fi
    elif [[ "$OS" == "macos" ]]; then
        OSNICK="macos"
    else
        OSNICK="$OS"
    fi
}

detect_platform

# RLEC naming conventions (platform adjustments)
[[ $ARCH == x64 ]] && ARCH=x86_64
[[ $OS == linux ]] && OS=Linux
[[ $OSNICK == trusty ]]  && OSNICK=ubuntu14.04
[[ $OSNICK == xenial ]]  && OSNICK=ubuntu16.04
[[ $OSNICK == bionic ]]  && OSNICK=ubuntu18.04
[[ $OSNICK == focal ]]   && OSNICK=ubuntu20.04
[[ $OSNICK == jammy ]]   && OSNICK=ubuntu22.04
[[ $OSNICK == centos7 ]] && OSNICK=rhel7
[[ $OSNICK == centos8 ]] && OSNICK=rhel8
[[ $OSNICK == ol8 ]]     && OSNICK=rhel8
[[ $OSNICK == rocky8 ]]  && OSNICK=rhel8
[[ $OSNICK == bigsur ]]  && OSNICK=catalina

PLATFORM="$OS-$OSNICK-$ARCH"

#----------------------------------------------------------------------------------------------

RAMP=${RAMP:-1}
# DEPS=${DEPS:-1}
# SYM=${SYM:-1}

RELEASE=${RELEASE:-1}
SNAPSHOT=${SNAPSHOT:-1}

if [[ -z $MODULE || ! -f $MODULE ]]; then
	eprint "MODULE is not defined or does not refer to a file"
	exit 1
fi

[[ -z $ARTDIR ]] && ARTDIR=bin/artifacts
mkdir -p $ARTDIR $ARTDIR/snapshots
ARTDIR=$(cd $ARTDIR && pwd)

MODULE_NAME=${MODULE_NAME:-graph}
PACKAGE_NAME=${PACKAGE_NAME:-redisgraph}

DEP_NAMES=""

RAMP_CMD="python3 -m RAMP.ramp"

#----------------------------------------------------------------------------------------------

pack_ramp() {
	cd $ROOT

	local stem=${PACKAGE_NAME}.${PLATFORM}
	local stem_debug=${PACKAGE_NAME}.debug.${PLATFORM}

	if [[ $SNAPSHOT == 0 ]]; then
		local verspec=${SEMVER}${VARIANT}
		local packdir=.
		local s3base=""
	else
		local verspec=${BRANCH}${VARIANT}
		local packdir=snapshots
		local s3base=snapshots/
	fi
	
	local fq_package=$stem.${verspec}.zip
	local fq_package_debug=$stem_debug.${verspec}.zip

	[[ ! -d $ARTDIR/$packdir ]] && mkdir -p $ARTDIR/$packdir

	local packfile=$ARTDIR/$packdir/$fq_package
	local packfile_debug=$ARTDIR/$packdir/$fq_package_debug

	local xtx_vars=""
	for dep in $DEP_NAMES; do
		eval "export NAME_${dep}=${PACKAGE_NAME}_${dep}"
		local dep_fname="${PACKAGE_NAME}.${dep}.${PLATFORM}.${verspec}.tgz"
		eval "export PATH_${dep}=${s3base}${dep_fname}"
		local dep_sha256="$ARTDIR/$packdir/${dep_fname}.sha256"
		eval "export SHA256_${dep}=$(cat $dep_sha256)"

		xtx_vars+=" -e NAME_$dep -e PATH_$dep -e SHA256_$dep"
	done
	
	if [[ -z $RAMP_YAML ]]; then
		RAMP_YAML=$ROOT/ramp.yml
	elif [[ -z $RAMP_VARIANT ]]; then
		RAMP_YAML=$ROOT/ramp.yml
	else
		RAMP_YAML=$ROOT/ramp${_RAMP_VARIANT}.yml
	fi

	# Export all required variables for envsubst
	export NUMVER SEMVER
	# envsubst handles variable substitution in the template
	envsubst < $RAMP_YAML > /tmp/ramp.yml
	if [[ $VERBOSE == 1 ]]; then
		echo "# ramp.yml:"
		cat /tmp/ramp.yml
	fi

	runn rm -f /tmp/ramp.fname $packfile
	
	# ROOT is required so ramp will detect the right git commit
	cd $ROOT
	runn @ <<-EOF
		$RAMP_CMD pack -m /tmp/ramp.yml \
			$RAMP_ARGS \
			-n $MODULE_NAME \
			--verbose \
			--debug \
			--packname-file /tmp/ramp.fname \
			-o $packfile \
			$MODULE \
			>/tmp/ramp.err 2>&1 || true
		EOF

	if [[ $NOP != 1 ]]; then
		if [[ ! -e $packfile ]]; then
			eprint "Error generating RAMP file:"
			>&2 cat /tmp/ramp.err
			exit 1
		else
			local packname=`cat /tmp/ramp.fname`
			echo "# Created $packname"
		fi
	fi

	if [[ -f $MODULE.debug ]]; then
		runn @ <<-EOF
			$RAMP_CMD pack -m /tmp/ramp.yml \
				$RAMP_ARGS \
				-n $MODULE_NAME \
				--verbose \
				--debug \
				--packname-file /tmp/ramp.fname \
				-o $packfile_debug \
				$MODULE.debug \
				>/tmp/ramp.err 2>&1 || true
			EOF

		if [[ $NOP != 1 ]]; then
			if [[ ! -e $packfile_debug ]]; then
				eprint "Error generating RAMP file:"
				>&2 cat /tmp/ramp.err
				exit 1
			else
				local packname=`cat /tmp/ramp.fname`
				echo "# Created $packname"
			fi
		fi
	fi

	cd $ROOT
}

#----------------------------------------------------------------------------------------------

pack_deps() {
	local dep="$1"
	
	cd $ROOT
	
	local stem=${PACKAGE_NAME}.${dep}.${PLATFORM}
	local verspec=${SEMVER}${VARIANT}
	local fq_package=$stem.${verspec}.tgz

	local depdir=$(cat $ARTDIR/$dep.dir)
	local tar_path=$ARTDIR/$fq_package
	local dep_prefix_dir=$(cat $ARTDIR/$dep.prefix)
	
	rm -f $tar_path
	if [[ $NOP != 1 ]]; then
		{ cd $depdir ;\
		  cat $ARTDIR/$dep.files | \
		  xargs tar -c --sort=name --owner=root:0 --group=root:0 --mtime='UTC 1970-01-01' \
			--transform "s,^,$dep_prefix_dir," 2> /tmp/pack.err | \
		  gzip -n - > $tar_path ; E=$?; } || true
		if [[ ! -e $tar_path || -z $(tar tzf $tar_path) ]]; then
			eprint "Count not create $tar_path. Aborting."
			rm -f $tar_path
			exit 1
		fi
	else
		runn @ <<-EOF
			cd $depdir
			cat $ARTDIR/$dep.files | \
			xargs tar -c --sort=name --owner=root:0 --group=root:0 --mtime='UTC 1970-01-01' \
				--transform "s,^,$dep_prefix_dir," 2>> /tmp/pack.err | \
			gzip -n - > $tar_path ; E=$?; } || true
			EOF
	fi
	runn @ <<-EOF
		sha256sum $tar_path | awk '{print $1}' > $tar_path.sha256
		EOF

	mkdir -p $ARTDIR/snapshots
	cd $ARTDIR/snapshots
	if [[ -n $BRANCH ]]; then
		local snap_package=$stem.${BRANCH}${VARIANT}.tgz
		runn ln -sf ../$fq_package $snap_package
		runn ln -sf ../$fq_package.sha256 $snap_package.sha256
	fi

	cd $ROOT
}

#----------------------------------------------------------------------------------------------

prepare_symbols_dep() {
	if [[ ! -f $MODULE.debug ]]; then return 0; fi
	echo "# Preparing debug symbols dependencies ..."
	dirname "$(realpath "$MODULE")" > "$ARTDIR/debug.dir"
	echo "$(basename "$(realpath "$MODULE")").debug" > "$ARTDIR/debug.files"
	echo "" > $ARTDIR/debug.prefix
	pack_deps debug
	echo "# Done."
}

#----------------------------------------------------------------------------------------------

NUMVER="$(NUMERIC=1 $SBIN/getver)"
SEMVER="$($SBIN/getver)"

if [[ -n $VARIANT ]]; then
	_VARIANT="-${VARIANT}"
fi
if [[ ! -z $RAMP_VARIANT ]]; then
	_RAMP_VARIANT="-${RAMP_VARIANT}"
fi

#----------------------------------------------------------------------------------------------

if [[ -z $BRANCH ]]; then
	BRANCH=$(git rev-parse --abbrev-ref HEAD)
	# this happens of detached HEAD
	if [[ $BRANCH == HEAD ]]; then
		BRANCH="$SEMVER"
	fi
fi
BRANCH=${BRANCH//[^A-Za-z0-9._-]/_}
if [[ $WITH_GITSHA == 1 ]]; then
	GIT_COMMIT=$(git rev-parse --short HEAD)
	BRANCH="${BRANCH}-${GIT_COMMIT}"
fi

#----------------------------------------------------------------------------------------------

RELEASE_ramp=${PACKAGE_NAME}.$OS-$OSNICK-$ARCH.$SEMVER${VARIANT}.zip
SNAPSHOT_ramp=${PACKAGE_NAME}.$OS-$OSNICK-$ARCH.${BRANCH}${VARIANT}.zip

RELEASE_deps=
SNAPSHOT_deps=
for dep in $DEP_NAMES; do
	RELEASE_deps+=" ${PACKAGE_NAME}.${dep}.$OS-$OSNICK-$ARCH.$SEMVER${VARIANT}.tgz"
	SNAPSHOT_deps+=" ${PACKAGE_NAME}.${dep}.$OS-$OSNICK-$ARCH.${BRANCH}${VARIANT}.tgz"
done

#----------------------------------------------------------------------------------------------

if [[ $JUST_PRINT == 1 ]]; then
	if [[ $RAMP == 1 ]]; then
		[[ $RELEASE == 1 ]] && echo $RELEASE_ramp
		[[ $SNAPSHOT == 1 ]] && echo $SNAPSHOT_ramp
	fi
	if [[ $DEPS == 1 ]]; then
		[[ $RELEASE == 1 ]] && echo $RELEASE_deps
		[[ $SNAPSHOT == 1 ]] && echo $SNAPSHOT_deps
	fi
	exit 0
fi

#----------------------------------------------------------------------------------------------

mkdir -p $ARTDIR

if [[ $DEPS == 1 ]]; then
	# set up `debug` dep
	dirname "$(realpath "$MODULE")" > "$ARTDIR/debug.dir"
	echo "$(basename "$(realpath "$MODULE")").debug" > "$ARTDIR/debug.files"
	echo "" > $ARTDIR/debug.prefix

	echo "# Building dependencies ..."

	[[ $SYM == 1 ]] && prepare_symbols_dep

	for dep in $DEP_NAMES; do
		echo "# $dep ..."
		pack_deps $dep
	done
	echo "# Done."
fi

#----------------------------------------------------------------------------------------------

cd $ROOT

if [[ $RAMP == 1 ]]; then
	if ! command -v redis-server > /dev/null; then
		eprint "Cannot find redis-server. Aborting."
		exit 1
	fi

	echo "# Building RAMP $RAMP_VARIANT files ..."

	[[ -z $MODULE ]] && { eprint "Nothing to pack. Aborting."; exit 1; }
	[[ ! -f $MODULE ]] && { eprint "$MODULE does not exist. Aborting."; exit 1; }
	MODULE=$(realpath $MODULE)

	[[ $RELEASE == 1 ]] && SNAPSHOT=0 pack_ramp
	[[ $SNAPSHOT == 1 ]] && pack_ramp
	
	echo "# Done."
fi

if [[ $VERBOSE == 1 ]]; then
	echo "# Artifacts:"
	$OP du -ah --apparent-size $ARTDIR
fi

exit 0
