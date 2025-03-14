function testmake (what)
%TESTMAKE compiles the test interface to GraphBLAS
% and dynamically links it with the libraries in ../build/libgraphblas.
%
% This @GrB interface to GraphBLAS is meant for testing and development,
% not for general use.
%
% Usage:
%
%   testmake        % just make what has changed (does not check any changes
%                   % in -lgraphblas, use 'testmake all' if recompilation is
%                   needed
%   testmake all    % make everything from scratch
%
% See also graphblas_install.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

here = pwd ;
if (ispc)
    here = strrep (here, filesep, '/') ;
end
if (isempty (strfind (here, 'GraphBLAS/Test')))
    % this function should only be done in GraphBLAS/Test
    error ('testmake should be used in Test directory only') ;
end

fprintf ('\nCompiling GraphBLAS tests:\n') ;

have_octave = (exist ('OCTAVE_VERSION', 'builtin') == 5) ;
if (have_octave)
    need_rename = false ;
else
    need_rename = true ; % was: ~verLessThan ('matlab', '9.10') ;
end

try
    spok (sparse (1)) ;
catch
    here = pwd ;
    cd ('spok') ;
    spok_install ;
    cd (here) ;
end

if (nargin < 1)
    what = '' ;
end

make_all = (isequal (what, 'all')) ;

flags = '-g -R2018a -DGBNCPUFEAT' ;

cflags = '' ;

mexfunctions = dir ('GB_mex_*.c') ;
cfiles = [ dir('../Demo/Include/usercomplex.c') ; dir('GB_mx_*.c') ] ;

hfiles = [ dir('*.h') ; dir('Template/*.c') ; dir('../Demo/Include/usercomplex.h') ] ;
inc = '-ITemplate -I../Include -I../Source -I../lz4 -I../rmm_wrap' ;
inc = [inc ' -I../zstd -I../zstd/zstd_subset -I.'] ;
inc = [inc ' -I../Config '] ;
inc = [inc ' -I../Source/builtin '] ;
inc = [inc ' -I../Source/hyper '] ;

if (ismac)
    % Mac (do 'make install' for GraphBLAS first)
    if (need_rename)
        libraries = '-L/usr/local/lib -lgraphblas_matlab' ; % -lomp' ;
    else
        libraries = '-L/usr/local/lib -lgraphblas' ; % -lomp' ;
    end
elseif (ispc)
    % Windows
    if (need_rename)
        libraries = '-L../GraphBLAS/build/Release -L. -lgraphblas_matlab' ;
    else
        libraries = '-L../build/Release -L. -lgraphblas' ;
    end
    flags = [ flags ' CFLAGS="$CXXFLAGS -wd\"4244\" -wd\"4146\" -wd\"4217\" -wd\"4286\" -wd\"4018\" -wd\"4996\" -wd\"4047\" -wd\"4554\"" '] ;
else
    % Linux
    if (need_rename)
        libraries = '-L../GraphBLAS/build -L. -lgraphblas_matlab' ;
    else
        libraries = '-L../build -L. -lgraphblas' ;
    end
end

if ispc
    if (need_rename)
        library_path = sprintf ('%s/../GraphBLAS/build/Release', pwd) ;
    else
        library_path = sprintf ('%s/../build/Release', pwd) ;
    end
else
    if (need_rename)
        library_path = sprintf ('%s/../GraphBLAS/build', pwd) ;
    else
        library_path = sprintf ('%s/../build', pwd) ;
    end
end

% revise compiler flags for MATLAB
if (ismac)
    ldflags = '-fPIC' ;
    rpath = '-rpath ' ;
elseif (isunix)
    cflags = [cflags ' -fopenmp'] ;
    ldflags = '-fopenmp -fPIC' ;
    rpath = '-rpath=' ;
end
if (ismac || isunix)
    rpath = sprintf (' -Wl,%s''''%s'''' ', rpath, library_path) ;
    flags = [ flags ' CFLAGS=''$CFLAGS ' cflags ' -Wno-pragmas'' '] ;
    flags = [ flags ' CXXFLAGS=''$CXXFLAGS ' cflags ' -Wno-pragmas'' '] ;
    flags = [ flags ' LDFLAGS=''$LDFLAGS ' ldflags rpath ' '' '] ;
end

if (need_rename)
    fprintf ('Linking with -lgraphblas_matlab\n') ;
    flags = [flags ' -DGBMATLAB=1 ' ] ;
    inc = [inc ' -I../GraphBLAS/rename ' ] ;
    libgraphblas = '-lgraphblas_matlab' ;
else
    libgraphblas = '-lgraphblas' ;
end

Lflags = sprintf ('-L''%s''', library_path) ;

fprintf ('compiler flags: %s\n', flags) ;
fprintf ('compiler incs:  %s\n', inc) ;
fprintf ('linking flags:  %s\n', Lflags) ;
fprintf ('library:        %s\n', libgraphblas) ;

%-------------------------------------------------------------------------------

dryrun = false ;

% Find the last modification time of any hfile.
% These are #include'd into source files.
htime = 0 ;
for k = 1:length (hfiles)
    t = datenum (hfiles (k).date) ;
    htime = max (htime, t) ;
end

if (ispc)
    obj_extension = '.obj' ;
else
    obj_extension = '.o' ;
end

% compile any source files that need compiling
any_c_compiled = false ;
objlist = '' ;
for k = 1:length (cfiles)

    % get the full cfile filename and  modification time
    cfile = [(cfiles (k).folder) filesep (cfiles (k).name)] ;
    tc = datenum (cfiles(k).date) ;

    % get the object file name
    ofile = cfiles(k).name ;
    objfile = [ ofile(1:end-2) obj_extension ] ;

    % get the object file modification time
    ofiles {k} = objfile ;
    objlist = [ objlist ' ' objfile ] ;
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
        fprintf ('.') ;
        % fprintf ('%s\n', cfile) ;
        mexcmd = sprintf ('mex -c %s -silent %s %s', flags, inc, cfile) ;
        % fprintf ('\n%s\n', mexcmd) ;
        if (dryrun)
            fprintf ('%s\n', mexcmd) ;
        else
            % fprintf ('%s\n', mexcmd) ;
            eval (mexcmd) ;
        end
        any_c_compiled = true ;
    end
end

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
            Lflags, flags, inc, mexfunction, objlist, libraries) ;
        fprintf (':') ;
        % fprintf ('%s\n', mexfunction) ;
        if (dryrun)
            fprintf ('%s\n', mexcmd) ;
        else
            % fprintf ('%s\n', mexcmd) ;
            eval (mexcmd) ;
        end
    end
end

% compile GB_spones_mex
mex -g -R2018a GB_spones_mex.c

% load the library
GB_mex_init ;

