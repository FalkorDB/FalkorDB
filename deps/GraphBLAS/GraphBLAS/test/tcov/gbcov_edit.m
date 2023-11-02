function count = gbcov_edit (infiles, count, outdir)
%GBCOV_EDIT create a version of GraphBLAS for statement coverage tests
%
% Usage:
% count = gbcov_edit (infiles, count)
%
% The infiles argument can either be a struct from the 'dir' command, or it can
% be a string with the name of a single file.  This function adds statement
% coverage counters to a set of input files.  For each of them, a modified file
% of the same name is placed in outdir, with statement coverage added.  The
% input files are modified in a simple way.  Each line that is all blank except
% for "{ " at the end of the line is converted to:
%
%   { gbcov [count]++ ;

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

% infiles can be a struct from dir, or a single string with one filename
if (~isstruct (infiles))
    infiles = dir (infiles) ;
end
nfiles = length (infiles) ;

for k = 1:nfiles

    if (infiles (k).bytes == 0)
        continue ;
    end

    infile  = [infiles(k).folder '/' infiles(k).name] ;
    outfile = [outdir '/' infiles(k).name] ;
    fprintf ('.') ;

    f_input  = fopen (infile,  'r') ;
    f_output = fopen (outfile, 'w') ;

    % get the first line
    cline = fgetl (f_input) ;
    len = length (cline) ;

    while (ischar (cline))

        if (isempty (cline))

            % empty line: as-is
            fprintf (f_output, '\n') ;

        elseif (len > 1 && all (cline (1:len-2) == ' ') ...
                && (cline (len-1) == '{') && (cline (len) == ' '))

            % left curly brackect and space at the end of the line
            % "{ " changes to "{   gbcov[n]++ ; "

            fprintf (f_output, '%s  gbcov[%d]++ ;\n', cline, count) ;
            count = count + 1 ;

        else

            % otherwise the line is copied as-is
            fprintf (f_output, '%s\n', cline) ;

        end

        % get the next line
        cline = fgetl (f_input) ;
        len = length (cline) ;

    end

    fclose (f_input) ;
    fclose (f_output) ;
end

fprintf ('\n') ;

