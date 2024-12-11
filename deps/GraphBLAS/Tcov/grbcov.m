function grbcov
%GRBCOV compile, run, and evaluate test coverage

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2024, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

clear all
tstart = tic ;

system ('make purge') ;

fp = fopen ('log.txt', 'a') ;
fprintf (fp, '%s grbcov starting\n', datestr (now)) ;
fclose (fp) ;

!rmtmph
clear mex
grbmake ;
testcov ;
grbshow ;
ttotal = toc (tstart) ;

fp = fopen ('log.txt', 'a') ;
fprintf (fp, '%s grbcov ending\n', datestr (now)) ;
fclose (fp) ;

fprintf ('\nTotal time, incl compilation: %8.2f minutes\n', ttotal / 60) ;

