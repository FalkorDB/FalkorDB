function gbcov
%GBCOV run all GraphBLAS tests, with statement coverage

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

% compile the coverage-test version of the @GrB mexFunctions
global gbcov_global
gbcov_global = [ ] ;

try
    % clear the default GrB library
    GrB.finalize ;
catch
end

gbcovmake
addpath ('..') ;            % add the test folder to the path
try
    rmpath ('../..') ;      % remove the regular @GrB class
catch me
end

rmpath ('tmp') ;            % remove the modified @GrB class
which ('GrB')
assert (isempty (which ('GrB')))

addpath ('tmp') ;           % add back the modified @GrB class
s = which ('GrB') ;

% run the tests
gbtest ;

try
    % clear the test coverage version of the GrB library
    GrB.finalize ;
catch
end

addpath ('../..') ;         % add back the regular @GrB class
rmpath ('tmp') ;            % remove the modified @GrB class

% report the coverage
fprintf ('Revised @GrB tested: %s\n', s) ;
gbcovshow ;
fprintf ('Now with usual @GrB: %s\n', which ('GrB')) ;

try
    % reload the default GrB library
    GrB.init ;
catch
end
