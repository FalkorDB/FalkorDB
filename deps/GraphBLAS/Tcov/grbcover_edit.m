function count = grbcover_edit (infiles, count, outdir)
%GBCOVER_EDIT create a version of GraphBLAS for statement coverage tests
%
% Usage:
% count = grbcover_edit (infiles, count)
%
% The infiles argument can either be a struct from the 'dir' command, or it can
% be a string with the name of a single file.  This function adds statement
% coverage counters to a set of input files.  For each of them, a modified file
% of the same name is placed in cover/, with statement coverage added.  The
% input files are modified in a simple way.  Each line that is all blank except
% for "{ " at the end of the line is converted to:
%
%   { GB_cov [count]++ ;
%
% In a switch statement, a counter is added to each case and to the default,
% but only if the colon has spaces on either side (" : ").
%
%       case stuff :  statement
%       default :     statement
%
% are converted to:
%
%       case stuff :  GB_cov[count]++ ; statement
%       default :     GB_cov[count]++ ; statement
%

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (ispc)
    error ('The tests in Tcov are not ported to Windows') ;
end

% infiles can be a struct from dir, or a single string with one filename
if (~isstruct (infiles))
    infiles = dir (infiles) ;
end
nfiles = length (infiles) ;
enabled = true ;

for k = 1:nfiles

    if (infiles (k).bytes == 0)
        continue ;
    end

    infile  = [infiles(k).folder '/' infiles(k).name] ;
    outfile = [outdir '/' infiles(k).name] ;
    fprintf ('.') ;
    if (mod (k, 40) == 0)
        fprintf ('\n') ;
    end

    f_input  = fopen (infile,  'r') ;
    f_output = fopen (outfile, 'w') ;

    % get the first line
    cline = fgetl (f_input) ;
    len = length (cline) ;
    indent = false ;

    while (ischar (cline))

        if (isempty (cline))

            % empty line: as-is
            fprintf (f_output, '\n') ;

        elseif (len > 1 && all (cline (1:len-2) == ' ') ...
                && (cline (len-1) == '{') && (cline (len) == ' '))

            % left curly brackect and space at the end of the line
            % "{ " changes to "{   GB_cov[n]++ ; "

            if (enabled)
                fprintf (f_output, '%s  GB_cov[%d]++ ;\n', cline, count) ;
                count = count + 1 ;
            else
                fprintf (f_output, '%s\n', cline) ;
            end

        elseif ((~isempty (strfind (cline, ' case ')) || ...
                 ~isempty (strfind (cline, ' default '))) && ...
                 ~isempty (strfind (cline, ' : ')))

            % a switch case statement, or "default : "
            % "case stuff : statement" => "case stuff : GB_cov[n]++ ; statement"

            if (enabled)
                colon = find (cline == ':', 1) ;
                fprintf (f_output, '%s : GB_cov[%d]++ ; %s\n', ...
                    cline (1:colon-1), count, cline (colon+1:end)) ;
                count = count + 1 ;
            else
                fprintf (f_output, '%s\n', cline) ;
            end

        else

            % otherwise the line is copied as-is
            fprintf (f_output, '%s\n', cline) ;

            % determine if the code is commented out
            if (isequal (cline, '#if 0') && enabled)
                % code coverage disabled until reaching "#endif"
                indent = false ;
                enabled = false ;
            elseif (isequal (cline, '    #if 0') && enabled)
                % code coverage disabled until reaching "    #endif"
                indent = true ;
                enabled = false ;
            elseif (isequal (cline, '#endif') && (~indent) && (~enabled))
                % code coverage enabled
                enabled = true ;
            elseif (isequal (cline, '    #endif') && (indent) && (~enabled))
                % code coverage enabled
                enabled = true ;
            end

        end

        % get the next line
        cline = fgetl (f_input) ;
        len = length (cline) ;

    end

    fclose (f_input) ;
    fclose (f_output) ;
end

fprintf ('\n') ;

