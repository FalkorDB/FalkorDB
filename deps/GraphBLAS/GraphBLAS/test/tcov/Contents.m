% GraphBLAS/GraphBLAS/test/tcov/Contents.m:
%
% The gbcov script compiles the interface with statement coverage
% enabled, and then runs the full test suite (../gbtest).  Next, it uses
% gbcovshow to create the statement coverage report in tmp/cover (one for
% each file).  To remove all temporary files, use 'make distclean' or
% remove the tmp/* files and folders manually.
%
%   gbcov      - run all GraphBLAS tests, with statement coverage
%
% Utilities:
%
%   gbcovmake  - compile the interface for statement coverage testing
%   gbcovshow  - report GraphBLAS statement coverage
%   gbcov_edit - create a version of GraphBLAS for statement coverage tests

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

