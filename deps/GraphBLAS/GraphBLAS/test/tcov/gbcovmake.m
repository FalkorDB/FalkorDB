function gbcovmake
%GBCOVMAKE compile the interface for statement coverage testing
%
% See also: gbcover, gbcov_edit

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('Compiling @GrB interface for mexFunction statement coverage...\n') ;
warning ('off', 'MATLAB:MKDIR:DirectoryExists') ;
mkdir ('tmp/@GrB/') ;
mkdir ('tmp/@GrB/private') ;
mkdir ('tmp/@GrB/util') ;
mkdir ('tmp/cover') ;
warning ('on', 'MATLAB:MKDIR:DirectoryExists') ;

% copy all m-files into tmp/@GrB
mfiles = dir ('../../@GrB/*.m') ;
for k = 1:length (mfiles)
    copyfile ([(mfiles (k).folder) '/' (mfiles (k).name)], 'tmp/@GrB/') ;
end

% copy all private m-files into tmp/@GrB/private
mfiles = dir ('../../@GrB/private/*.m') ;
for k = 1:length (mfiles)
    copyfile ([(mfiles (k).folder) '/' (mfiles (k).name)], 'tmp/@GrB/private') ;
end

% copy the *.h files
copyfile ('../../@GrB/private/util/*.h', 'tmp/@GrB/util') ;

% copy and edit the mexfunction/*.c files
cfiles = dir ('../../@GrB/private/mexfunctions/*.c') ; 
count = gbcov_edit (cfiles, 0, 'tmp/@GrB/private') ;

% copy and edit the util/*.c files
ufiles = [ dir('../../@GrB/private/util/*.c') ; dir('*.c') ] ;
count = gbcov_edit (ufiles, count, 'tmp/@GrB/util') ;

% create the gbfinish.c file and place in tmp/@GrB/util
f = fopen ('tmp/@GrB/util/gbcovfinish.c', 'w') ;
fprintf (f, '#include "gb_interface.h"\n') ;
fprintf (f, 'int64_t gbcov [GBCOV_MAX] ;\n') ;
fprintf (f, 'int gbcov_max = %d ;\n', count) ;
fclose (f) ;

% compile the modified interface

% use -R2018a for the new interleaved complex API
flags = '-g -R2018a -DGBCOV' ;

if ispc
    library_path = sprintf ('%s/../../build/Release', pwd) ;
else
    library_path = sprintf ('%s/../../build', pwd) ;
end

here = pwd ;

% use renamed version for all MATLAB versions:
flags = [flags ' -DGBMATLAB=1 ' ] ;
inc = sprintf ('-I%s/../../rename ', here) ;
libraries = '-L../../../../../build -L. -L/usr/local/lib -lgraphblas_matlab' ;

% revise compiler flags for MATLAB
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

inc = [inc '-I. -I../util '] ;
    inc = [inc '-I../../../../../.. ' ] ;
    inc = [inc '-I../../../../../../Include '] ;
    inc = [inc '-I../../../../../../Source ' ] ;
    inc = [inc '-I../../../../../../Source/include '] ;
    inc = [inc '-I../../../../../../Source/ij ' ] ;
    inc = [inc '-I../../../../../../Source/math ' ] ;
    inc = [inc '-I../../../../../../Source/cast ' ] ;
    inc = [inc '-I../../../../../../Source/binaryop ' ] ;
    inc = [inc '-I../../../../../../Source/transpose ' ] ;
    inc = [inc '-I../../../../../../Source/helper ' ] ;
    inc = [inc '-I../../../../../../Source/builtin ' ] ;
    inc = [inc '-I../../../../../../Source/hyper ' ] ;

Lflags = sprintf ('-L''%s''', library_path) ;

fprintf ('compiler flags: %s\n', flags) ;
fprintf ('compiler incs:  %s\n', inc) ;
fprintf ('linking flags:  %s\n', Lflags) ;
fprintf ('libraries:      %s\n', libraries) ;

cd tmp/@GrB/private
try

    % compile util files
    cfiles = dir ('../util/*.c') ;

    objlist = '' ;
    for k = 1:length (cfiles)
        % get the full cfile filename
        cfile = [(cfiles (k).folder) '/' (cfiles (k).name)] ;
        % get the object file name
        ofile = cfiles(k).name ;
        objfile = [ ofile(1:end-2) '.o' ] ;
        objlist = [ objlist ' ' objfile ] ; %#ok<*AGROW>
        % compile the cfile
        mexcmd = sprintf ('mex -c %s -silent %s %s', flags, inc, cfile) ;
        fprintf ('.') ;
        % fprintf ('%s\n', cfile) ;
        % fprintf ('%s\n', mexcmd) ;
        eval (mexcmd) ;
    end

    mexfunctions = dir ('*.c') ;

    % compile the mexFunctions
    for k = 1:length (mexfunctions)

        % get the mexFunction filename and modification time
        mexfunc = mexfunctions (k).name ;
        mexfunction = [(mexfunctions (k).folder) '/' mexfunc] ;

        % compile the mexFunction
        mexcmd = sprintf ('mex %s -silent %s %s ''%s'' %s %s', ...
            Lflags, flags, inc, mexfunction, objlist, libraries) ;
        fprintf (':') ;
        % fprintf ('%s\n', mexfunction) ;
        % fprintf ('%s\n', mexcmd) ;
        eval (mexcmd) ;
    end
    fprintf ('\n') ;

catch me
    disp (me.message)
end
cd (here)

