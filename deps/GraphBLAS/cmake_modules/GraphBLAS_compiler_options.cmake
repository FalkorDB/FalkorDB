#-------------------------------------------------------------------------------
# GraphBLAS/cmake_modules/GraphBLAS_compiler_options.cmake
#-------------------------------------------------------------------------------

# SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2024, All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

#-------------------------------------------------------------------------------

# check which compiler is being used.  If you need to make
# compiler-specific modifications, here is the place to do it.
if ( "${CMAKE_C_COMPILER_ID}" STREQUAL "GNU" )
    # The -g option is useful for the Intel VTune tool, but it should be
    # removed in production.  Comment this line out if not in use:
    # set ( CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} -g" )
    # cmake 2.8 workaround: gcc needs to be told to do C11.
    # cmake 3.0 doesn't have this problem.
    set ( CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} -Wundef " )
#   uncomment this to check for all warnings:
#   set ( CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} -Wall -Werror -Wno-strict-aliasing " )
    set ( CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} -std=c11 -lm -Wno-pragmas " )
    # operations may be carried out in higher precision
    set ( CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} -fexcess-precision=fast " )
    # faster single complex multiplication and division
    set ( CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} -fcx-limited-range " )
    # math functions do not need to report errno
    set ( CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} -fno-math-errno " )
    # integer operations wrap
    set ( CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} -fwrapv " )
    # check all warnings (uncomment for development only)
#   set ( CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} -Wall -Wextra -Wpedantic " )
    if ( CMAKE_C_COMPILER_VERSION VERSION_LESS 4.9 )
        message ( FATAL_ERROR "gcc version must be at least 4.9" )
    endif ( )
elseif ( "${CMAKE_C_COMPILER_ID}" STREQUAL "Intel" )
    # options for icc: also needs -std=c11
    set ( CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} -diag-disable 10397,15552 " )
#   set ( CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} -qopt-report=5 -qopt-report-phase=vec" )
    # the -mp1 option is important for predictable floating-point results with
    # the icc compiler.  Without, ((float) 1.)/((float) 0.) produces NaN,
    # instead of the correct result, Inf.
    set ( CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} -std=c11 -mp1" )
    # The -g option is useful for the Intel VTune tool, but it should be
    # removed in production.  Comment this line out if not in use:
    # set ( CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} -g" )
#   set ( CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} -qopt-malloc-options=3" )
    # check all warnings and remarks (uncomment for development only):
#   set ( CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} -w3 -Wremarks -Werror " )
    if ( CMAKE_C_COMPILER_VERSION VERSION_LESS 19.0 )
        message ( FATAL_ERROR "icc version must be at least 19.0" )
    endif ( )
elseif ( "${CMAKE_C_COMPILER_ID}" STREQUAL "IntelLLVM" )
    # options for icx
elseif ( "${CMAKE_C_COMPILER_ID}" STREQUAL "Clang" )
    # options for clang
    set ( CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} -Wno-pointer-sign " )
    if ( CMAKE_C_COMPILER_VERSION VERSION_LESS 3.3 )
        message ( FATAL_ERROR "clang version must be at least 3.3" )
    endif ( )
elseif ( MSVC )
    # options for MicroSoft Visual Studio
    set ( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /O2 -wd\"4244\" -wd\"4146\" -wd\"4018\" -wd\"4996\" -wd\"4047\" -wd\"4554\"" )
elseif ( "${CMAKE_C_COMPILER_ID}" STREQUAL "PGI" )
    # options for PGI pgcc compiler.  The compiler has a bug, and the
    # -DPGI_COMPILER_BUG causes GraphBLAS to use a workaround.
    set ( CMAKE_C_FLAGS    "${CMAKE_C_FLAGS} -Mnoopenmp -noswitcherror -c11 -lm -DPGI_COMPILER_BUG" )
    set ( CMAKE_CXX_FLAGS  "${CMAKE_C_FLAGS} -Mnoopenmp -D__GCC_ATOMIC_TEST_AND_SET_TRUEVAL=1 -noswitcherror --c++11 -lm -DPGI_COMPILER_BUG" )
endif ( )

if ( ${CMAKE_BUILD_TYPE} STREQUAL "Debug" )
    set ( CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_DEBUG}" )
else ( )
    set ( CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_RELEASE}" )
endif ( )

