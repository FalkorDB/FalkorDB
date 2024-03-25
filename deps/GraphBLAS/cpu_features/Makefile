# https://github.com/google/cpu_features
#
# Makefile added by Tim Davis

go:
	cmake -S. -Bbuild -DBUILD_TESTING=OFF -DCMAKE_BUILD_TYPE=Release
	cmake --build build --config Release -j

distclean: purge

install:
	( cd build && make install )

purge:
	- rm -rf ./build/*
