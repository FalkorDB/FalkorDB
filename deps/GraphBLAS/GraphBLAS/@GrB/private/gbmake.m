function gbmake (what)
%GBMAKE compile @GrB interface for SuiteSparse:GraphBLAS
%
% Usage:
%   gbmake
%
% gbmake compiles the @GrB interface for SuiteSparse:GraphBLAS.  The
% GraphBLAS library must already be compiled and installed.
% MATLAB 9.4 (R2018a) or Octave 7 later is required.
%
% For the Mac, the GraphBLAS library must be installed in /usr/local/lib/ as
% libgraphblas_matlab.dylib (or just libgraphblas.dylib for Octave).  It cannot
% be used where it is created in ../build, because of the default Mac security
% settings.  For Unix/Linux, the library is ../build/libgraphblas_matlab.so if
% found (or libgraphblas.so for Octave), or in /usr/local/lib if not found
% there.
%
% See also mex, version, GrB.clear.
%
% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

have_octave = (exist ('OCTAVE_VERSION', 'builtin') == 5) ;

if (have_octave)
    % Octave can use the normal libgraphblas.so
    need_rename = 0 ;
    if verLessThan ('octave', '7')
        error ('GrB:mex', 'Octave 7 or later is required') ;
    end
    library_name = 'libgraphblas' ;
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
end

fprintf ('Note: the %s dynamic library must already be\n', library_name) ;
fprintf ('compiled and installed prior to running this script.\n') ;

if (nargin < 1)
    what = '' ;
end

make_all = (isequal (what, 'all')) ;

% use -R2018a for the new interleaved complex API
flags = '-O -R2018a -DGBNCPUFEAT' ;

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
        library_path = sprintf ('%s/../../build/Release', pwd) ;
    else
        library_path = sprintf ('%s/../../../build/Release', pwd) ;
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
        cd ../../build
    else
        cd ../../../build
    end
    library_path = pwd
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
        flags = [flags ' -std=c11 -fopenmp -fPIC -Wno-pragmas' ] ;
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
        cflags = '-fopenmp' ;
        ldflags = '-fopenmp -fPIC' ;
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

inc = '-Iutil -I../../../Include -I../../../Source ' ;
    inc = [inc '-I../../../Source/include '] ;
    inc = [inc '-I../../.. ' ] ;
    inc = [inc '-I../../../Source/ij ' ] ;
    inc = [inc '-I../../../Source/math ' ] ;
    inc = [inc '-I../../../Source/cast ' ] ;
    inc = [inc '-I../../../Source/binaryop ' ] ;
    inc = [inc '-I../../../Source/transpose ' ] ;
    inc = [inc '-I../../../Source/helper ' ] ;
    inc = [inc '-I../../../Source/builtin ' ] ;
    inc = [inc '-I../../../Source/hyper ' ] ;

if (need_rename)
    % use the renamed library for MATLAB
    flags = [flags ' -DGBMATLAB=1 ' ] ;
    inc = [inc ' -I../../rename ' ] ;
    libgraphblas = '-lgraphblas_matlab' ;
else
    % use the regular library for Octave
    libgraphblas = '-lgraphblas' ;
end

% determine if the compiler supports C99 or MSVC complex types
try
    % try C99 complex types
    cflag = ' -DGxB_HAVE_COMPLEX_C99=1' ;
    mexcmd = sprintf ('mex -silent %s %s complex/check_mex_complex.c', ...
        flags, cflag) ;
    eval (mexcmd) ;
catch
    % try MSVC complex types
    try
        cflag = ' -DGxB_HAVE_COMPLEX_MSVC=1' ;
        mexcmd = sprintf ('mex -silent %s %s complex/check_mex_complex.c', ...
            flags, cflag) ;
        eval (mexcmd) ;
    catch me
        me
        error ('C99 or MSVC complex support required') ;
    end
end
flags = [flags cflag] ;
check_mex_complex

Lflags = sprintf ('-L''%s''', library_path) ;

fprintf ('compiler flags: %s\n', flags) ;
fprintf ('compiler incs:  %s\n', inc) ;
fprintf ('linking flags:  %s\n', Lflags) ;
fprintf ('library:        %s\n', libgraphblas) ;

hfiles = [ dir('*.h') ; dir('util/*.h') ] ;

cfiles = dir ('util/*.c') ;

% Find the last modification time of any hfile.
% These are #include'd into source files.
htime = 0 ;
for k = 1:length (hfiles)
    t = datenum (hfiles (k).date) ; %#ok<*DATNM>
    htime = max (htime, t) ;
end

% compile any source files that need compiling
any_c_compiled = 0 ;
objlist = '' ;
for k = 1:length (cfiles)

    % get the full cfile filename and modification time
    cfile = [(cfiles (k).folder) filesep (cfiles (k).name)] ;
    tc = datenum (cfiles(k).date) ;

    % get the object file name
    ofile = cfiles(k).name ;
    objfile = [ ofile(1:end-2) object_suffix ] ;

    % get the object file modification time
    objlist = [ objlist ' ' objfile ] ;     %#ok
    dobj = dir (objfile) ;
    if (isempty (dobj))
        % there is no object file; the cfile must be compiled
        tobj = 0 ;
    else
        tobj = datenum (dobj.date) ;
    end

    % compile the cfile if it is newer than its object file, or any hfile
    if (make_all || tc > tobj || htime > tobj)
        % compile the cfile
        % fprintf ('%s\n', cfile) ;
        mexcmd = sprintf ('mex -c %s -silent %s ''%s''', flags, inc, cfile) ;
        % fprintf ('%s\n', mexcmd) ;
        fprintf ('.') ;
        eval (mexcmd) ;
        any_c_compiled = 1 ;
    end
end

% compile the mexFunctions

if (have_octave)
    fprintf ('\nBuilding GrB mexFunctions for Octave.\n') ;
    if (ismac)
        fprintf ('Ignore any ''ld:warning: duplicate -bunder_loader option'' warnings.\n\n') ;
    end
end

mexfunctions = dir ('mexfunctions/*.c') ;
for k = 1:length (mexfunctions)

    % get the mexFunction filename and modification time
    mexfunc = mexfunctions (k).name ;
    mexfunction = [(mexfunctions (k).folder) filesep mexfunc] ;
    tc = datenum (mexfunctions(k).date) ;

    % get the compiled mexFunction modification time
    mexfuncname = mexfunc (1:end-2) ;
    mexfunction_compiled = [ mexfuncname '.' mexext ] ;
    dobj = dir (mexfunction_compiled) ;
    if (isempty (dobj))
        % there is no compiled mexFunction; it must be compiled
        tobj = 0 ;
    else
        tobj = datenum (dobj.date) ;
    end

    % compile if it is newer than its object file, or if any cfile was compiled
    if (make_all || tc > tobj || any_c_compiled)
        % compile the mexFunction
        mexcmd = sprintf ('mex %s -silent %s %s ''%s'' %s %s', ...
            Lflags, flags, inc, mexfunction, objlist, libgraphblas) ;
        % fprintf ('%s\n', mexcmd) ;
        fprintf (':') ;
        eval (mexcmd) ;

        if (have_octave && ismac)
            cmd = sprintf ('install_name_tool -add_rpath ''%s'' %s.mex', ...
                library_path, mexfuncname) ;
            % fprintf ('%s\n', cmd) ;
            system (cmd) ;
        end
    end
end

fprintf ('\n') ;

fprintf ('Compilation of the @GrB interface to GraphBLAS is complete.\n') ;
fprintf ('Add the following commands to your startup.m file:\n\n') ;
here1 = cd ('../..') ;
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

