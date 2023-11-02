function grbmake
%GBMAKE compile the GraphBLAS library for statement coverage testing
%
% This function compiles ../Source to create the
% libgraphblas_tcov.so (or *.dylib) library, inserting code code for statement
% coverage testing.  It does not compile the mexFunctions.
%
% See also: grbcover, grbcover_edit

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (ispc)
    error ('The tests in Tcov are not ported to Windows') ;
end

% copy the GB_rename.h and GB_coverage.c files
copyfile ('../GraphBLAS/rename/GB_rename.h', 'tmp_include/GB_rename.h') ;
copyfile ('GB_coverage.c', 'tmp_source/GB_coverage.c') ;

% create the include files and place in tmp_include
hfiles = [ dir('../Include/*') ; ...
           dir('../Source/*.h') ; ...
           dir('../GraphBLAS/Config/*.h') ; ...
           dir('../Source/Template') ; ...
           dir('../Source/Factories') ; ...
           dir('../Source/Shared') ; ...
           dir('../JITpackage/*.h') ; ...
           dir('../Source/FactoryKernels/*.h') ; ] ;
count = grbcover_edit (hfiles, 0, 'tmp_include') ;
fprintf ('hfile count: %d\n', count) ;

% create the C files and place in tmp_source
cfiles = [ dir('../Source/*.c') ; ...
           dir('../Source/FactoryKernels/*.c') ; ...
           % use Tcov/PreJIT kernels ...
           dir('PreJIT/*.c') ; ...
           % not the PreJIT kernels in the primary source
           % dir('../PreJIT/*.c') ; ...
           % dir('../Config/GB_prejit.c') ; ...
           dir('../JITpackage/*.c')
           ] ;
count = grbcover_edit (cfiles, count, 'tmp_source') ;
fprintf ('cfile count: %d\n', count) ;

% save the count
fp = fopen ('tmp_cover/count', 'w') ;
fprintf (fp, '%d\n', count) ;
fclose (fp) ;

% revise this to match Source/Template/GB_coverage.h
GBCOVER_MAX = 31000 ;
assert (count < GBCOVER_MAX) ;

% compile the libgraphblas_tcov.so library

have_octave = (exist ('OCTAVE_VERSION', 'builtin') == 5) ;
if (have_octave)
    need_rename = false ;
else
    need_rename = ~verLessThan ('matlab', '9.10') ;
end

if (need_rename)
    fprintf ('Rename with -DGBMATLAB=1\n') ;
    system (sprintf ('make -j%d MATLAB="-DGBMATLAB=1"', feature ('numcores'))) ;
else
    system (sprintf ('make -j%d', feature ('numcores'))) ;
end

