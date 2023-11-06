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
% libgraphblas_matlab.dylib.  It cannot be used where it is created in
% ../build, because of the default Mac security settings.  For Unix/Linux, the
% library used is ../build/libgraphblas_matlab.so if found, or in
% /usr/local/lib if not found there.
%
% See also mex, version, GrB.clear.
%
% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('Note: the libgraphblas_matlab dynamic library must already be\n') ;
fprintf ('compiled prior to running this script.\n') ;

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
    what = '' ;
end

make_all = (isequal (what, 'all')) ;

flags = '-O -R2018a -DGBNCPUFEAT' ;

% use -R2018a for the new interleaved complex API
if (have_octave)
    % Octave does not have the new MEX classdef object and as of version 7, the
    % mex command doesn't handle compiler options the same way.
    flags = [flags ' -std=c11 -fopenmp -fPIC -Wno-pragmas' ] ;
else
    try
        if (strncmp (computer, 'GLNX', 4))
            % remove -ansi from CFLAGS and replace it with -std=c11
            cc = mex.getCompilerConfigurations ('C', 'Selected') ;
            env = cc.Details.SetEnv ;
            c1 = strfind (env, 'CFLAGS=') ;
            q = strfind (env, '"') ;
            q = q (q > c1) ;
            if (~isempty (c1) && length (q) > 1)
                c2 = q (2) ;
                cflags = env (c1:c2) ;  % the CFLAGS="..." string
                ansi = strfind (cflags, '-ansi') ;
                if (~isempty (ansi))
                    cflags = [cflags(1:ansi-1) '-std=c11' cflags(ansi+5:end)] ;
                    flags = [flags ' ' cflags] ;
                    fprintf ('using -std=c11 instead of default -ansi\n') ;
                end
            end
        end
    catch
    end
    if (~ismac && isunix)
        flags = [ flags   ' CFLAGS="$CXXFLAGS -fopenmp -fPIC -Wno-pragmas" '] ;
        flags = [ flags ' CXXFLAGS="$CXXFLAGS -fopenmp -fPIC -Wno-pragmas" '] ;
        flags = [ flags  ' LDFLAGS="$LDFLAGS  -fopenmp -fPIC" '] ;
    end
end

if ispc
    % Windows
    object_suffix = '.obj' ;
else
    % Linux, Mac
    object_suffix = '.o' ;
end

inc = '-Iutil -I../../../Include -I../../../Source -I../../../Source/Shared -I../../../Source/Template -I../../../Source/Factories ' ;

if ispc
    % First do the following in GraphBLAS/build, in the Windows console:
    %
    %   cmake ..
    %   devenv graphblas.sln /build "release|x64" /project graphblas
    %
    % The above commands require MS Visual Studio.  The graphblas.lib is
    % compiled and placed in GraphBLAS/build/Release.  Then in the
    % Command Window do:
    %
    %   gbmake
    if (need_rename)
        library = sprintf ('%s/../../build/Release', pwd) ;
    else
        library = sprintf ('%s/../../../build/Release', pwd) ;
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
    if (need_rename)
        library = sprintf ('%s/../../build', pwd) ;
    else
        library = sprintf ('%s/../../../build', pwd) ;
    end
end

if (need_rename)
    flags = [flags ' -DGBMATLAB=1 ' ] ;
    inc = [inc ' -I../../rename ' ] ;
    libgraphblas = '-lgraphblas_matlab' ;
else
    libgraphblas = '-lgraphblas' ;
end

ldflags = sprintf ('-L''%s''', library) ;

hfiles = [ dir('*.h') ; dir('util/*.h') ] ;

cfiles = dir ('util/*.c') ;

% Find the last modification time of any hfile.
% These are #include'd into source files.
htime = 0 ;
for k = 1:length (hfiles)
    t = datenum (hfiles (k).date) ;
    htime = max (htime, t) ;
end

% compile any source files that need compiling
any_c_compiled = false ;
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
        fprintf ('.') ;
        mexcmd = sprintf ('mex -c %s -silent %s ''%s''', flags, inc, cfile) ;
        eval (mexcmd) ;
        any_c_compiled = true ;
    end
end

mexfunctions = dir ('mexfunctions/*.c') ;

% compile the mexFunctions
for k = 1:length (mexfunctions)

    % get the mexFunction filename and modification time
    mexfunc = mexfunctions (k).name ;
    mexfunction = [(mexfunctions (k).folder) filesep mexfunc] ;
    tc = datenum (mexfunctions(k).date) ;

    % get the compiled mexFunction modification time
    mexfunction_compiled = [ mexfunc(1:end-2) '.' mexext ] ;
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
            ldflags, flags, inc, mexfunction, objlist, libgraphblas) ;
        % fprintf ('%s\n', mexcmd) ;
        fprintf (':') ;
        eval (mexcmd) ;
    end
end

fprintf ('\n') ;

fprintf ('Compilation of the @GrB interface to GraphBLAS is complete.\n') ;
fprintf ('Add the following commands to your startup.m file:\n\n') ;
here1 = cd ('../..') ;
addpath (pwd) ;
fprintf ('  addpath (''%s'') ;\n', pwd) ;
cd ('..') ;
if (need_rename)
    cd ('GraphBLAS') ;
end
if ispc
    lib_path = sprintf ('%s/build/Release', pwd) ;
else
    lib_path = sprintf ('%s/build', pwd) ;
end
fprintf ('  addpath (''%s'') ;\n', lib_path) ;
addpath (lib_path) ;
cd (here1) ;

fprintf ('\nFor a quick demo of GraphBLAS, type the following commands:\n\n') ;
fprintf ('  cd ../../demo\n') ;
fprintf ('  gbdemo\n') ;

fprintf ('\nTo test GraphBLAS, type the following commands:\n\n') ;
fprintf ('  cd ../../test\n') ;
fprintf ('  gbtest\n') ;

