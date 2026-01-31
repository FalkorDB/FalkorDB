# Create a virtual environment with all tools installed
# ref: https://hub.docker.com/_/ubuntu
FROM ubuntu:latest AS env

# Install system build dependencies
ENV PATH=/usr/local/bin:$PATH
RUN apt-get update -qq \
&& DEBIAN_FRONTEND=noninteractive apt-get install -yq \
 git wget build-essential \
&& apt-get clean \
&& rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
ENTRYPOINT ["/usr/bin/bash", "-c"]
CMD ["/usr/bin/bash"]

# Install Bazelisk
ARG PLATFORM
RUN wget \
 "https://github.com/bazelbuild/bazelisk/releases/download/v1.18.0/bazelisk-linux-${PLATFORM}" \
&& chmod +x "bazelisk-linux-${PLATFORM}" \
&& mv "bazelisk-linux-${PLATFORM}" /usr/local/bin/bazel

FROM env AS devel
WORKDIR /home/project
COPY . .

FROM devel AS build
RUN bazel version
RUN bazel build \
 -c opt \
 --subcommands=true \
 ...

FROM build AS test
RUN bazel test \
 -c opt \
 --test_output=errors \
 ...
