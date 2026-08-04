function gbmake (what)
%GBMAKE compile GrB/GhB interface for SuiteSparse:GraphBLAS
%
% Usage:
%   gbmake
%
% gbmake compiles the GrB interface for SuiteSparse:GraphBLAS.  The
% GraphBLAS library must already be compiled and installed.
% MATLAB 9.4 (R2018a) or Octave 10.0 later is required.
%
% You must run this command while in the GraphBLAS/GraphBLAS/private folder.
%
% For the Mac, the GraphBLAS library must be installed in /usr/local/lib/ as
% libgraphblas_matlab.dylib (or just libgraphblas.dylib for Octave).  It cannot
% be used where it is created in ../build, because of the default Mac security
% settings.  For Unix/Linux, the library is ../build/libgraphblas_matlab.so if
% found (or ../../build/libgraphblas.so for Octave), or in /usr/local/lib if
% not found there.
%
% See also mex, version, GrB.clear.
%
% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

addpath ('..') ;
have_octave = gb_octave ;

if (have_octave)
    % Octave can use the normal libgraphblas.so
    need_rename = 0 ;
    if verLessThan ('octave', '10.0')
        error ('GrB:mex', 'Octave 10.0 or later is required') ;
    end
    library_name = 'libgraphblas' ;
    app_name = 'Octave' ;
else
    if verLessThan ('matlab', '9.4')
        error ('GrB:mex', 'MATLAB 9.4 (R2018a) or later is required') ;
    end
    % MATLAB 9.10 (R2021a) and following include a built-in GraphBLAS library
    % that conflicts with this version, so rename this version.
    % Earlier versions of MATLAB can use this renamed version too, so
    % for simplicity, use libgraphblas_matlab.so for all MATLAB versions.
    need_rename = 1 ;
    library_name = 'libgraphblas_matlab' ;
    app_name = 'MATLAB' ;
end

fprintf ('Note: the %s dynamic library must already be\n', library_name) ;
fprintf ('compiled and installed prior to running this script.\n') ;

if (nargin < 1)
    what = '' ;
end

make_all = (isequal (what, 'all')) ;

% use -R2018a for the interleaved complex API
  flags = '-O -R2018a -DGBNCPUFEAT' ;
% the debug build or tracking build are required to enable the GrB.nmalloc
% checks in gbtest.m:
% flags = '-g -R2018a -DGBNCPUFEAT -DMALLOC_TRACKING' ;   % debug build
% flags = '-O -R2018a -DGBNCPUFEAT -DMALLOC_TRACKING' ;   % tracking build

if ispc
    % First do the following in GraphBLAS/build, in the Windows console:
    %
    %   cmake ..
    %   cmake --build . --config Release
    %
    % The above commands require MS Visual Studio.  The graphblas.lib is
    % compiled and placed in GraphBLAS/build/Release.  Then in the
    % Command Window do:
    %
    %   gbmake
    %
    if (need_rename)
        library_path = sprintf ('%s/../build/Release', pwd) ;
    else
        library_path = sprintf ('%s/../../build/Release', pwd) ;
    end
else
    % First do one the following in GraphBLAS (use JOBS=n for a parallel
    % build, which is faster):
    %
    %   make
    %   make JOBS=8
    %   sudo make install
    %
    % If you can't do "sudo make install" then add the GraphBLAS/build
    % folder to your LD_LIBRARY_PATH.  Then in this folder in the
    % Command Window do:
    %
    %   gbmake
    %
    here = pwd ;
    if (need_rename)
        cd ../build
    else
        cd ../../build
    end
    library_path = pwd ;
    cd (here) ;
end

if (have_octave)
    % Revise compiler flags for Octave.
    % Octave does not have the new MEX classdef object and as of version 7, the
    % mex command doesn't handle compiler options the same way.
    if (ismac)
%       the mexFunctions themselves do not need OpenMP, and they can be hard
%       to compile on the Mac
%       flags = [flags ' -std=c11 -Xclang -fopenmp -fPIC -Wno-pragmas' ] ;
        flags = [flags ' -std=c11 -fPIC -Wno-pragmas' ] ;
        rpath = ' ' ;
    else
%       flags = [flags ' -std=c11 -fopenmp -fPIC -Wno-pragmas' ] ;
        flags = [flags ' -std=c11 -fPIC -Wno-pragmas' ] ;
        rpath = sprintf (' ''-Wl,-rpath=%s'' ', library_path) ;
    end
    flags = [flags rpath] ;
else
    % Revise compiler flags for MATLAB.
    if (ismac)
        cflags = '' ;
        ldflags = '-fPIC' ;
        rpath = '-rpath ' ;
    elseif (isunix)
        cflags = '' ;
        ldflags = '-fPIC' ;
%       cflags = '-fopenmp' ;
%       ldflags = '-fopenmp -fPIC' ;
        rpath = '-rpath=' ;
    end
    if (ismac || isunix)
        rpath = sprintf (' -Wl,%s''''%s'''' ', rpath, library_path) ;
        flags = [ flags ' CFLAGS=''$CFLAGS ' cflags ' -Wno-pragmas'' '] ;
        flags = [ flags ' CXXFLAGS=''$CXXFLAGS ' cflags ' -Wno-pragmas'' '] ;
        flags = [ flags ' LDFLAGS=''$LDFLAGS ' ldflags rpath ' '' '] ;
    end
end

if ispc
    % Windows
    object_suffix = '.obj' ;
else
    % Linux, Mac
    object_suffix = '.o' ;
end

inc = '-Iutil -I../../Include -I../../Source ' ;
    inc = [inc '-I../../Source/include '] ;
    inc = [inc '-I../.. ' ] ;
    inc = [inc '-I../../Source/ij ' ] ;
    inc = [inc '-I../../Source/math ' ] ;
    inc = [inc '-I../../Source/cast ' ] ;
    inc = [inc '-I../../Source/binaryop ' ] ;
    inc = [inc '-I../../Source/transpose ' ] ;
    inc = [inc '-I../../Source/helper ' ] ;
    inc = [inc '-I../../Source/memory ' ] ;
    inc = [inc '-I../../Source/builtin ' ] ;
    inc = [inc '-I../../Source/builtin/include ' ] ;
    inc = [inc '-I../../Source/hyper ' ] ;

if (need_rename)
    % use the renamed library for MATLAB
    flags = [flags ' -DGBMATLAB=1 ' ] ;
    inc = [inc ' -I../rename ' ] ;
    libgraphblas = '-lgraphblas_matlab' ;
else
    % use the regular library for Octave
    libgraphblas = '-lgraphblas' ;
end

  silent = '-silent' ;      % completely silent
% silent = '-v' ;           % extremely verbose
% silent = '' ;

% determine if MATLAB/Octave support sparse single matrices
have_sparse_single = false ;
try
    A = single (speye (3)) ;
    have_sparse_single = issparse (A) && isequal (class (A), 'single') ;
catch
end
if (have_sparse_single)
    fprintf ('%s has sparse single matrices.\n', app_name) ;
end

% determine if the compiler supports C99 or MSVC complex types
cflag = ' -DGxB_HAVE_COMPLEX_C99=1' ;
if (ispc)
    try
        % try C99 complex types
        fprintf ('try C99 complex:\n') ;
        mexcmd = sprintf ('mex %s %s %s complex/check_mex_complex.c', ...
            silent, flags, cflag) ;
        % fprintf ('mexcmd: %s\n', mexcmd) ;
        eval (mexcmd) ;
    catch me
        % try MSVC complex types
        fprintf ('try MSVC complex:\n') ;
        try
            cflag = ' -DGxB_HAVE_COMPLEX_MSVC=1' ;
            mexcmd = sprintf ('mex %s %s %s complex/check_mex_complex.c', ...
                silent, flags, cflag) ;
            % fprintf ('mexcmd: %s\n', mexcmd) ;
            eval (mexcmd) ;
        catch me
            me
            error ('C99 or MSVC complex support required') ;
        end
    end
    flags = [flags cflag] ;
    check_mex_complex
end

Lflags = sprintf ('-L''%s''', library_path) ;

fprintf ('compiler flags: %s\n', flags) ;
fprintf ('compiler incs:  %s\n', inc) ;
fprintf ('linking flags:  %s\n', Lflags) ;
fprintf ('library:        %s\n', libgraphblas) ;

% Find the last modification time of any util/*.c or *.h file.
hfiles = [ dir('*.h') ; dir('util/*.c') ; dir('util/*.h') ; ] ;
htime = 0 ;
for k = 1:length (hfiles)
    t = datenum (hfiles (k).date) ; %#ok<*DATNM>
    htime = max (htime, t) ;
end

if (have_octave)
    flags = [ flags ' -DOCTAVE=1 '] ;
    fprintf ('\nBuilding GrB mexFunctions for Octave.\n') ;
    if (ismac)
        fprintf ('Ignore any ''ld:warning: duplicate -bunder_loader option'' warnings.\n\n') ;
    end
end

% compile the mexFunctions
mexfunctions = dir ('mexfunctions/*.c') ;
for k = 1:length (mexfunctions)

    % get the mexFunction filename and modification time
    mexfunc = mexfunctions (k).name ;
    mexfunction = [(mexfunctions (k).folder) filesep mexfunc] ;
    tc = datenum (mexfunctions(k).date) ;

    % get the compiled mexFunction modification time
    mexfuncname = mexfunc (1:end-2) ;
    mexfunction_compiled = [ '../' mexfuncname '.' mexext ] ;
    dobj = dir (mexfunction_compiled) ;
    if (isempty (dobj))
        % there is no compiled mexFunction; it must be compiled
        tobj = 0 ;
    else
        tobj = datenum (dobj.date) ;
    end

    % compile if it is newer than its object file, or any cfile is newer
    if (make_all || tc > tobj || htime > tobj)
        % compile the mexFunction
        if (have_octave)
            outfile = sprintf ('-o ../%s.mex', mexfuncname) ;
        else
            outfile = '-outdir ..' ;
        end
        mexcmd = sprintf ('mex %s %s %s %s %s ''%s'' %s', ...
            outfile, Lflags, silent, flags, inc, mexfunction, libgraphblas) ;
        % fprintf ('%s\n', mexcmd) ;
        fprintf (':') ;
        eval (mexcmd) ;

        if (have_octave && ismac)
            cmd = sprintf ('install_name_tool -add_rpath ''%s'' ../%s.mex', ...
                library_path, mexfuncname) ;
            % fprintf ('%s\n', cmd) ;
            system (cmd) ;
        end
    end
end

fprintf ('\n') ;

fprintf ('Compilation of the GrB/GhB interface to GraphBLAS is complete.\n') ;
fprintf ('Add the following commands to your startup.m file:\n\n') ;
here1 = cd ('..') ;
here2 = pwd ;
addpath (here2) ;
fprintf ('  addpath (''%s'') ;\n', here2) ;
cd ('..') ;
if (need_rename)
    cd ('GraphBLAS') ;
end
if ispc
    lib_path = sprintf ('%s/build/Release', here2) ;
    fprintf ('  addpath (''%s'') ;\n', lib_path) ;
    addpath (lib_path) ;
end
cd (here1) ;

fprintf ('\nFor a quick demo of GraphBLAS, type the following commands:\n\n') ;
fprintf ('  cd %s/demo\n', here2) ;
fprintf ('  gbdemo\n') ;

fprintf ('\nTo test GraphBLAS, type the following commands:\n\n') ;
fprintf ('  cd %s/test\n', here2) ;
fprintf ('  gbtest\n') ;

