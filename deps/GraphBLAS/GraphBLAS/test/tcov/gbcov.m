function gbcov
%GBCOV run all GraphBLAS tests, with statement coverage

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

tstart = tic ;

global gbcov_global
gbcov_global = [ ] ;

try
    % clear the default GraphBLAS library
    GrB.finalize ;
catch
end

% compile the coverage-test version of the GrB/GhB mexFunctions
gbcovmake

addpath ('..') ;            % add the test folder to the path
try
    rmpath ('../..') ;      % remove the regular GrB, GhB classes
catch me
end

rmpath ('tmp') ;            % remove the modified GrB, GhB classes
which ('GrB')
which ('GhB')

addpath ('tmp') ;           % add back the modified GrB, GhB classes
which_GrB = which ('GrB') ;
which_GhB = which ('GhB') ;

% run the tests
gbtest ;

try
    % clear the test coverage version of the GrB library
    GrB.finalize ;
catch
end

addpath ('../..') ;         % add back the regular GrB class
rmpath ('tmp') ;            % remove the modified GrB class

% report the coverage
fprintf ('Revised GrB tested: %s\n', which_GrB) ;
fprintf ('Revised GhB tested: %s\n', which_GhB) ;
gbcovshow ;
fprintf ('Now with usual GrB: %s\n', which ('GrB')) ;
fprintf ('Now with usual GhB: %s\n', which ('GhB')) ;

try
    % reload the default GrB library
    GrB.init ;
catch
end

fprintf ('gbcov total time: %g sec\n', toc (tstart)) ;

