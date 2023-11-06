function grbcover (what)
%GBCOVER compile ../Test/* for statement coverage testing
%
% This function compiles just the mexFunctions in ../Test.
% It does not compile the GraphBLAS library itself.
%
% See also: grbcover_edit, grbmake

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (ispc)
    error ('The tests in Tcov are not ported to Windows') ;
end

% compile the mexFunctions

if (nargin < 1)
    what = '' ;
end

make_all = (isequal (what, 'all')) ;

% get a list of the GraphBLAS mexFunctions
mexfunctions = dir ('../Test/GB_mex_*.c') ;

% remove GB_mex_tricount and GB_mex_bfs from the list of mexFunctions
nmex = length (mexfunctions) ;
for k = nmex:-1:1
    if (isequal (mexfunctions (k).name, 'GB_mex_tricount.c')) || ...
       (isequal (mexfunctions (k).name, 'GB_mex_bfs.c'))
        mexfunctions (k) = [ ] ;
    end
end

% list of C files to compile
cfiles = [ dir('../Test/GB_mx_*.c') ; dir('../Demo/Include/usercomplex.c') ] ;

% list of *.h and template file dependencies
hfiles = [ dir('../Test/*.h') ; dir('../Test/Template/*.c') ] ;

% list of include directories
inc = '-Itmp_include -I../Test -I../Test/Template -I../lz4 -I../rmm_wrap' ;
inc = [inc ' -I../zstd -I../zstd/zstd_subset -I. -I../xxHash'] ;
inc = [inc ' -I../Source/JitKernels '] ;

have_octave = (exist ('OCTAVE_VERSION', 'builtin') == 5) ;
if (have_octave)
    need_rename = false ;
else
    need_rename = ~verLessThan ('matlab', '9.10') ;
end

addpath ../Test
addpath ../Test/spok

flags = '-g -DGBCOVER -R2018a -DGBNCPUFEAT' ;
if (need_rename)
    flags = [flags ' -DGBMATLAB=1 '] ;
end

fprintf ('\nCompiling GraphBLAS tests\nplease wait [') ;

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
                fprintf ('compiling with -std=c11 instead of default -ansi\n') ;
            end
        end
    end
end

libraries = sprintf ('-L%s -lgraphblas_tcov', pwd) ;

if (ismac)
    % Mac (do 'make install' for GraphBLAS first)
%   flags = [ flags   ' CFLAGS="$CXXFLAGS -Xpreprocessor -fopenmp" ' ] ;
%   flags = [ flags ' CXXFLAGS="$CXXFLAGS -Xpreprocessor -fopenmp" ' ] ;
%   flags = [ flags  ' LDFLAGS="$LDFLAGS  -fopenmp"' ] ;
else
    % Linux
    flags = [ flags   ' CFLAGS="$CXXFLAGS -fopenmp -fPIC -Wno-pragmas" '] ;
    flags = [ flags ' CXXFLAGS="$CXXFLAGS -fopenmp -fPIC -Wno-pragmas" '] ;
    flags = [ flags  ' LDFLAGS="$LDFLAGS  -fopenmp -fPIC" '] ;
end

dryrun = false ;

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

    % get the full cfile filename and  modification time
    cfile = [(cfiles (k).folder) filesep (cfiles (k).name)] ;
    tc = datenum (cfiles(k).date) ;

    % get the object file name
    ofile = cfiles(k).name ;
    objfile = [ ofile(1:end-2) '.o' ] ;

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
        if (dryrun)
            fprintf ('%s\n', mexcmd) ;
        else
            eval (mexcmd) ;
        end
        any_c_compiled = true ;
    end
end

if (ismac)
    objlist = [objlist ' libgraphblas_tcov.dylib '] ;
end
% objlist = [objlist ' ../cpu_features/build/libcpu_features.a'] ;

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
        mexcmd = sprintf ('mex -silent %s %s %s %s %s', ...
            flags, inc, mexfunction, objlist, libraries) ;
        fprintf ('.', mexfunction) ;
        % fprintf ('%s\n', mexfunction) ;
        if (dryrun)
            fprintf ('%s\n', mexcmd) ;
        else
            eval (mexcmd) ;
        end
    end
end

fprintf (']\n') ;

