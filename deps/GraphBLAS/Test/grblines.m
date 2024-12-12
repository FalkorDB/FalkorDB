function n = grblines
%GRBLINES total # of lines in the test coverage
n = 1 ;
if (~isempty (strfind (pwd, 'Tcov')))
    % load in the # of lines in the test coverage
    fp = fopen ('tmp_cover/count', 'r') ;
    n = textscan (fp, '%f') ;
    n = n {1} ;
    fclose (fp) ;
end

