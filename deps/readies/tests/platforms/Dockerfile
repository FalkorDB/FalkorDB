
ARG OS

FROM ${OS}

WORKDIR /readies
ADD . /readies/

RUN set -e ;\
	echo "::group::getpy3" ;\
	V=2 ./bin/getpy3 ;\
	echo "::endgroup::"
