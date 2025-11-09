function graphblas_install (cmake_options)
%GRAPHBLAS_INSTALL compile SuiteSparse:GraphBLAS for MATLAB or Octave
%
% Usage:
%   graphblas_install
%
% MATLAB 9.4 (R2018a) or Octave 7 later is required.  This function must
% be run while your current working directory is the same as the directory
% that contains graphblas_install.m.
%
% See also mex.
%
% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

% make sure we
here = pwd ;
my_fullpath = mfilename ('fullpath') ;
slash = strfind (my_fullpath, filesep) ;
if (~isempty (slash))
    slash = slash (end) ;
    my_folder = my_fullpath (1:slash-1) ;
    if (~isequal (my_folder, here))
        fprintf ('current working directory is: %s\n', here) ;
        fprintf ('graphblas_install is in     : %s\n', my_folder) ;
        fprintf ('do this before running graphblas_install:\n') ;
        fprintf ('   cd %s\n', my_folder) ;
        error ('graphblas_install must be run when in its containing folder') ;
    end
end

have_octave = (exist ('OCTAVE_VERSION', 'builtin') == 5) ;

if (have_octave)
    % Octave can use the normal libgraphblas.so
    need_rename = false ;
    if verLessThan ('octave', '7')
        error ('GrB:mex', 'Octave 7 or later is required') ;
    end
else
    if verLessThan ('matlab', '9.4')
        error ('GrB:mex', 'MATLAB 9.4 (R2018a) or later is required') ;
    end
    % MATLAB 9.10 (R2021a) and following include a built-in GraphBLAS library
    % that conflicts with this version, so rename this version.
    % Earlier versions of MATLAB can use this renamed version too, so
    % for simplicity, use libgraphblas_matlab.so for all MATLAB versions.
    need_rename = true ;
end

if (nargin < 1)
    cmake_options = '' ;
end

% by default, use OpenMP as found by cmake
openmp_library = '${OpenMP_C_LIBRARIES}' ;
if (ismac)
    % use the OpenMP library inside MATLAB
    % look for libomp.dylib for Apple Silicon Macs
    o = [matlabroot '/bin/maca64/libomp.dylib'] ;
    if (isfile (o))
        openmp_library = o ;
    else
        % look for libiomp5.dylib for Intel Macs
        o = [matlabroot '/sys/os/maci64/libiomp5.dylib'] ;
        if (isfile (o))
            openmp_library = o ;
        end
    end
end

% write the configuration file for cmake
f = fopen ('GraphBLAS_MATLAB_OpenMP.cmake', 'w') ;
fprintf (f, 'target_link_libraries ( graphblas_matlab PRIVATE %s )\n', ...
    openmp_library) ;
fclose (f) ;

% build the GraphBLAS library
threads = maxNumCompThreads * 2 ;

try

    % cd to the build directory
    if (need_rename)
        % building libgraphblas_matlab for MATLAB
        cd build
    else
        % building libgraphblas for Octave
        cd ../build
    end

    % cmd1: configure with cmake
    build_folder = pwd ;
    cmd1 = sprintf ('cmake %s ..', cmake_options) ;

    % build the GraphBLAS library
    cmd2 = sprintf ('cmake --build . --config Release -j%d', threads) ;

    % execute cmd1: configure with cmake
    clear mex
    fprintf ('\n================================\n%s\n', cmd1) ;
    [status, result] = system (cmd1, '-echo') ;
    if (status ~= 0)
        cd (here) ;
        error ('GrB:mex', 'GraphBLAS library not compiled') ;
    end

    % execute cmd2: build the GraphBLAS library
    fprintf ('\n================================\n%s\n', cmd2) ;
    [status, result] = system (cmd2, '-echo') ;
    cd (here) ;
    if (status ~= 0)
        error ('GrB:mex', 'GraphBLAS library not compiled') ;
    end

catch me
    me
    fprintf ('Building GraphBLAS with cmake failed.  Try this outside of MATLAB:\n') ;
    fprintf ('\n    cd %s\n    %s\n    %s\n', build_folder, cmd1, cmd2) ;
    cd (here) ;

    fprintf ('\nThen do this inside MATLAB:\n\n') ;
    fprintf ('    cd %s/@GrB/private\n    gbmake\n', here) ;
    return ;
end

% build the GraphBLAS MATLAB interface
cd '@GrB/private'
gbmake

cd (here) ;

