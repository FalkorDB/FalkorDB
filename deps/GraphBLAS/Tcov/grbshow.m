function grbshow
%GBSHOW create a test coverage report in tmp_cover/

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (ispc)
    error ('The tests in Tcov are not ported to Windows') ;
end

GB_mex_finalize ;
infiles = [ dir('tmp_source/*.*') ; dir('tmp_include/*.*') ] ;

nfiles = length (infiles) ;

load grbstat.mat

for k = 1:nfiles

    if (infiles (k).bytes == 0)
        continue ;
    end

    infile   = [ infiles(k).folder filesep infiles(k).name ] ;
    outfile  = [ 'tmp_cover/' infiles(k).name ] ;

    f_input  = fopen (infile,  'r') ;
    f_output = fopen (outfile, 'w') ;

    % get the first line
    cline = fgetl (f_input) ;

    while (ischar (cline))

        fprintf (f_output, '%s\n', cline) ;

        if (~isempty (strfind (cline, 'GB_cov[')) && ...
            ~isempty (strfind (cline, '++')))
            % got one; get the count
            k1 = strfind (cline, '[') ;
            k2 = strfind (cline, ']') ;
            s = cline (k1+1:k2-1) ;
            i = str2num (s) + 1 ;
            c = GraphBLAS_grbcov (i) ;
            if (c == 0)
                fprintf (f_output, '// NOT COVERED (%d):\n', i-1) ;
            else
                fprintf (f_output, '// covered (%d): %d\n', i-1, c) ;
            end
        end

        cline = fgetl (f_input) ;
    end

    fclose (f_output) ;
    fclose (f_input) ;

end


