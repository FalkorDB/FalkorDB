# The measurement environment, built once instead of 15 times.
#
# Every measure/callgrind job used to reconstruct this inline: `apt-get update`
# against Debian mirrors on the critical path of each of 15 parallel jobs, plus
# valgrind (~100 MB) pulled 12 times, plus the same dpkg-hook workaround pasted
# into three places. Now `preflight` builds this once, tags it by content, and
# the other jobs pull it.
#
# ## Why FROM the C engine image
#
# All three modules — PR, base and C — are measured inside one image, so a
# cross-engine comparison has no environment difference to explain away: same
# redis, same jemalloc build, same library environment. A module copied out of an
# image needs that image's library environment (an earlier attempt that copied
# modules onto a bare runner failed with "server did not start").
#
# ## Why the tools are installed rather than inherited
#
# The tag is `:edge-c`, not `:edge`. `:edge` now resolves to the *Rust* engine —
# it and `:edge-rs` share a digest — so anything relying on `:edge` being the C
# engine is silently measuring Rust against Rust.
#
# That is also why redis-tools is installed explicitly instead of relied upon.
# The harness needs `redis-benchmark`, which the C image happened to ship and the
# Rust images do not; "happened to ship" is what made the tag change invisible.
# apt is a no-op when the base already has it, and correct when it does not.
ARG C_IMAGE=docker.io/falkordb/falkordb-server:edge-c
FROM ${C_IMAGE}

# The image hooks DPkg::Pre-Install-Pkgs to a dpkg-preconfigure binary it does
# not contain, so every apt call prints "/usr/sbin/dpkg-preconfigure: not found"
# and carries on. DEBIAN_FRONTEND does not suppress it (it is a hook, not a
# frontend); removing the hook file does.
RUN rm -f /etc/apt/apt.conf.d/70debconf \
 && apt-get update -qq \
 && apt-get install -y -qq --no-install-recommends \
      valgrind \
      redis-tools \
      ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# uv brings its own pinned interpreter, so there is no `apt-get install python3`
# and no PEP 668 externally-managed-environment refusal to work around.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# Only the dependency set is baked, from the lockfile, so a measurement job does
# no network I/O at all. `--no-install-project` is the important flag: the
# harness *source* is bind-mounted at run time, so installing it here would make
# the image stale on every source edit and defeat the content-addressed tag.
WORKDIR /bench
COPY bench/pyproject.toml bench/uv.lock /bench/
RUN uv sync --frozen --no-dev --no-install-project

# `bench ...` means the same thing inside this image as it does on a laptop, so
# the workflow and the local recipe are not two different invocations. The source
# comes from the bind-mounted workspace via PYTHONPATH.
RUN printf '%s\n' \
      '#!/bin/sh' \
      'exec /bench/.venv/bin/python -m falkorbench.cli "$@"' \
      > /usr/local/bin/bench \
 && chmod +x /usr/local/bin/bench

# Reset the inherited ENTRYPOINT. The base image sets
# `ENTRYPOINT ["/var/lib/falkordb/bin/run.sh"]` with a redis-server CMD, so
# without this `docker run <image> bench measure …` passes "bench measure …" as
# *arguments to the server launcher* — the container comes up running redis and
# the harness never executes. The pre-image workflow sidestepped this with
# `--entrypoint bash` on every invocation; resetting it here means callers do not
# have to know.
ENTRYPOINT []

# THREAD_COUNT is pinned per-invocation via --module-args; OMP_NUM_THREADS is the
# environment half of the same pinning, and it belongs to every job in here
# because thread count is what makes callgrind counts reproducible or not.
ENV PYTHONPATH=/w/bench/src \
    OMP_NUM_THREADS=1

WORKDIR /w
