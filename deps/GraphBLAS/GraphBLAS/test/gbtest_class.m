%GBTEST_CLASS for testing GraphBLAS in MATLAB
%
% Use the MATLAB Test Browser to run this test in the MATLAB Desktop.  First,
% select this file to test by clicking on the "+" icon.  Next, select the "Open
% coverage settings" icon and add add the GrB and GhB folders and all gb*.m and
% gzb*.m files.  Run the test (the "|>" icon).  The code coverage is displayed
% when the test finishes.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

classdef gbtest_class < matlab.unittest.TestCase
    methods (Test)
        function gbtest (testCase)
            expected = true ;
            actual = gbtest ;
            testCase.verifyEqual (actual, expected) ;
        end
    end
end
