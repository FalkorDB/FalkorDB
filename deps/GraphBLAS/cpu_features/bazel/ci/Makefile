PROJECT := cpu_features
BRANCH := $(shell git rev-parse --abbrev-ref HEAD)
SHA1 := $(shell git rev-parse --verify HEAD)

# General commands
.PHONY: help
BOLD=\e[1m
RESET=\e[0m

help:
	@echo -e "${BOLD}SYNOPSIS${RESET}"
	@echo -e "\tmake <target> [NOCACHE=1]"
	@echo
	@echo -e "${BOLD}DESCRIPTION${RESET}"
	@echo -e "\ttest build inside docker container to have a reproductible build."
	@echo
	@echo -e "${BOLD}MAKE TARGETS${RESET}"
	@echo -e "\t${BOLD}help${RESET}: display this help and exit."
	@echo
	@echo -e "\t${BOLD}<platform>_<stage>${RESET}: build <stage> docker image using an Ubuntu:latest x86_64 base image."
	@echo -e "\t${BOLD}save_<platform>_<stage>${RESET}: Save the <stage> docker image."
	@echo -e "\t${BOLD}sh_<platform>_<stage>${RESET}: run a container using the <stage> docker image (debug purpose)."
	@echo -e "\t${BOLD}clean_<platform>_<stage>${RESET}: Remove cache and docker image."
	@echo
	@echo -e "\tWith ${BOLD}<platform>${RESET}:"
	@echo -e "\t\t${BOLD}amd64${RESET} (linux/amd64)"
	@echo -e "\t\t${BOLD}arm64${RESET} (linux/arm64)"
	@echo
	@echo -e "\tWith ${BOLD}<stage>${RESET}:"
	@echo -e "\t\t${BOLD}env${RESET}"
	@echo -e "\t\t${BOLD}devel${RESET}"
	@echo -e "\t\t${BOLD}build${RESET}"
	@echo -e "\t\t${BOLD}test${RESET}"
	@echo -e "\te.g. 'make amd64_build'"
	@echo
	@echo -e "\t${BOLD}clean${RESET}: Remove cache and ALL docker images."
	@echo
	@echo -e "\t${BOLD}NOCACHE=1${RESET}: use 'docker build --no-cache' when building container (default use cache)."
	@echo -e "\t${BOLD}VERBOSE=1${RESET}: use 'docker build --progress=plain' when building container."
	@echo
	@echo -e "branch: $(BRANCH)"
	@echo -e "sha1: $(SHA1)"

# Need to add cmd_platform to PHONY otherwise target are ignored since they do not
# contain recipe (using FORCE do not work here)
.PHONY: all
all: build

# Delete all implicit rules to speed up makefile
MAKEFLAGS += --no-builtin-rules
.SUFFIXES:
# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =
# Keep all intermediate files
# ToDo: try to remove it later
.SECONDARY:

# Docker image name prefix.
IMAGE := ${PROJECT}

DOCKER_BUILDX_CMD := docker buildx build
ifdef NOCACHE
DOCKER_BUILDX_CMD := ${DOCKER_BUILDX_CMD} --no-cache
endif
ifdef VERBOSE
DOCKER_BUILDX_CMD := ${DOCKER_BUILDX_CMD} --progress=plain
endif
DOCKER_RUN_CMD := docker run --rm --init --net=host

############
## NATIVE ##
############
# ref: https://go.dev/doc/install/source#environment
# ref: https://github.com/containerd/containerd/blob/269548fa27e0089a8b8278fc4fc781d7f65a939b/platforms/platforms.go#L80-L94
PLATFORMS := amd64 arm64
STAGES := env devel build test

define make-platform-stage-target =
$1_$2: docker/Dockerfile
	${DOCKER_BUILDX_CMD} \
 --platform linux/$1 \
 --build-arg PLATFORM="$1" \
 --target=$2 \
 --tag ${IMAGE}:$1_$2 \
 -f $$< ../..

save_$1_$2: cache/$1/docker_$2.tar
cache/$1/docker_$2.tar: $1_$2
	@rm -f $$@
	mkdir -p cache/$1
	docker save ${IMAGE}:$1_$2 -o $$@

sh_$1_$2: $1_$2
	${DOCKER_RUN_CMD} --platform linux/$1 -it --name ${IMAGE}_$1_$2 ${IMAGE}:$1_$2

clean_$1_$2:
	docker image rm -f ${IMAGE}:$1_$2 2>/dev/null
	rm -f cache/$1/docker_$2.tar
endef

define make-platform-target =
$(foreach stage,$(STAGES),$(eval $(call make-platform-stage-target,$1,$(stage))))

# merge
.PHONY: clean_$1
clean_$1: $(addprefix clean_$1_, $(STAGES))
	-rmdir cache/$1
endef

$(foreach platform,$(PLATFORMS),$(eval $(call make-platform-target,$(platform))))

## MERGE ##
.PHONY: clean
clean: $(addprefix clean_, $(PLATFORMS))
	docker container prune -f
	docker image prune -f
	-rmdir cache

.PHONY: distclean
distclean: clean
	-docker container rm -f $$(docker container ls -aq)
	-docker image rm -f $$(docker image ls -aq)
