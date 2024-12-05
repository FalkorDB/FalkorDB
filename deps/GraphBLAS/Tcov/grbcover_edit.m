function count = grbcover_edit (infiles, count, outdir)
%GRBCOVER_EDIT create a version of GraphBLAS for statement coverage tests
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

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2024, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (ispc)
    error ('The tests in Tcov are not ported to Windows') ;
end

% infiles can be a struct from dir, or a single string with one filename
if (~isstruct (infiles))
    infiles = dir (infiles) ;
end
nfiles = length (infiles) ;

% determine which types are disabled:
types = { 'bool', 'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', ...
    'uint32', 'uint64', 'fp32', 'fp64', 'fc32', 'fc64' } ;
disabled = zeros (length (types), 1) ;
for k = 1:length(types)
    t = upper (types {k}) ;
    [status, result] = system (sprintf ( ...
        'grep "^   #define GxB_NO_%s" ../Source/GB_control.h', t)) ;
    disabled (k) = ~isempty (result) ;
    if (disabled (k))
        fprintf ('type disabled: %s\n', t) ;
    end
end

for k = 1:nfiles

    if (mod (k, 70) == 0)
        fprintf ('\n') ;
    end

    if (infiles (k).bytes == 0)
        continue ;
    end

    infile  = [infiles(k).folder '/' infiles(k).name] ;
    outfile = [outdir '/' infiles(k).name] ;

    enabled = true ;
    coverage = true ;

    if (contains (infile, 'FactoryKernel'))
        % this is a FactoryKernel; check if its type is disabled
        if (contains (infile, 'GB_sel__'))
            % select FactoryKernels are never disabled
            coverage = true ;
        elseif (contains (infile, 'GB_uop__identity'))
            % identity FactoryKernels are never disabled by type
            coverage = true ;
        else
            % all other FactoryKernels may be disabled
            found = zeros (length (types), 1) ;
            for kk = 1:length (types)
                t = types {kk} ;
                t = ['_' t] ;
                found (kk) = length (strfind (infile, t)) ;
            end
            if (sum (found .* disabled) > 0)
                % no coverage for this file
                coverage = false ;
            end
        end
    end

    if (coverage)
        fprintf ('.') ;
    else
        fprintf ('o') ;
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

        elseif (len >= 2 && isequal (cline (1:2), '//'))

            % comment line: as-is
            fprintf (f_output, '%s\n', cline) ;

        elseif (contains (cline, '#include "'))

            if (contains (cline, '/GB_'))
                % convert '#include "mxm/template/GB_AxB_whatever.h'
                % to just '#include "GB_AxB_whatever.h'
                quote = strfind (cline, '"') ;
                quote = quote (1) ;
                gb = strfind (cline, '/GB_') ;
                gb = gb (1) ;
                fprintf (f_output, '%s%s\n', ...
                    cline (1:quote), cline (gb+1:end)) ;
            else
                % no change to this line
                fprintf (f_output, '%s\n', cline) ;
            end

        elseif (len > 1 && all (cline (1:len-2) == ' ') ...
                && (cline (len-1) == '{') && (cline (len) == ' '))

            % left curly brackect and space at the end of the line
            % "{ " changes to "{   GB_cov[n]++ ; "

            if (coverage && enabled)
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

            if (coverage && enabled)
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
            if (~coverage)
                % do nothing
            elseif (isequal (cline, '#if 0') && enabled)
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

