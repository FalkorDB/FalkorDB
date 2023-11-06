function gbcovshow
%GBCOVSHOW report GraphBLAS statement coverage

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

% report the coverage summary

global gbcov_global

if (isempty (gbcov_global))
    error ('no coverage stats') ;
end

covered = sum (gbcov_global > 0) ;
not_covered = find (gbcov_global == 0) - 1 ;
n = length (gbcov_global) ;

fprintf ('test coverage: %d of %d (%0.4f%%), not covered %d\n', ...
    covered, n, 100 * (covered / n), length (not_covered)) ;

% create the coverage reports in tmp/cover

infiles = dir ('tmp/@GrB/*/*.c') ;

nfiles = length (infiles) ;

for k = 1:nfiles

    if (infiles (k).bytes == 0)
        continue ;
    end

    infile   = [ infiles(k).folder filesep infiles(k).name ] ;
    outfile  = [ 'tmp/cover/' infiles(k).name ] ;

    f_input  = fopen (infile,  'r') ;
    f_output = fopen (outfile, 'w') ;

    % get the first line
    cline = fgetl (f_input) ;

    while (ischar (cline))

        fprintf (f_output, '%s\n', cline) ;

        if (~isempty (strfind (cline, 'gbcov[')) && ...
            ~isempty (strfind (cline, '++'))) %#ok<*STREMP>
            % got one; get the count
            k1 = strfind (cline, '[') ;
            k2 = strfind (cline, ']') ;
            s = cline (k1+1:k2-1) ;
            i = str2num (s) + 1 ; %#ok<*ST2NM>
            c = gbcov_global (i) ;
            if (c == 0)
                fprintf (f_output, '// NOT COVERED:\n') ;
            else
                fprintf (f_output, '// covered: %d\n', c) ;
            end
        end

        cline = fgetl (f_input) ;
    end

    fclose (f_output) ;
    fclose (f_input) ;

end

