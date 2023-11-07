function gbcov
%GBCOV run all GraphBLAS tests, with statement coverage

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

% compile the coverage-test version of the @GrB mexFunctions
global gbcov_global %#ok<*NUSED>

try
    % clear the default GrB library
    GrB.finalize ;
catch
end

gbcovmake
addpath ('..') ;            % add the test folder to the path
rmpath ('../..') ;          % remove the regular @GrB class
addpath ('tmp') ;           % add the modified @GrB class

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
gbcovshow ;

try
    % reload the default GrB library
    GrB.init ;
catch
end
