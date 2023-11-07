function testgen (threads,longtests)
%TESTALL run all GraphBLAS tests
%
% Usage:
% testall ;             % runs just the shorter tests (about 30 minutes)
%
% testall(threads) ;    % run with specific list of threads and chunk sizes
% testall([ ],1) ;      % run all longer tests, with default # of threads
%
% threads is a cell array. Each entry is 2-by-1, with the first value being
% the # of threads to use and the 2nd being the chunk size.  The default is
% {[4 1]} if empty or not present.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

GB_mex_init ;
testall_time = tic ;

if (nargin < 2)
    % run the shorter tests by default
    longtests = 0 ;
end

if (nargin < 1)
    threads = [ ] ;
end
if (isempty (threads))
    threads {1} = [4 1] ;
end
t = threads ;

% single thread
s {1} = [1 1] ;


% clear the statement coverage counts
clear global GraphBLAS_grbcov

% use built-in complex data types by default
GB_builtin_complex_set (true) ;

% many of the tests use spok in SuiteSparse, a copy of which is
% included here in GraphBLAS/Test/spok.
addpath ('../Test/spok') ;
try
    spok (sparse (1)) ;
catch
    here = pwd ;
    cd ../Test/spok ;
    spok_install ;
    cd (here) ;
end

logstat ;             % start the log.txt
hack = GB_mex_hack ;

% JIT and factory controls

% default
j404 = {4,0,4} ;    % JIT     on, off, on
f110 = {1,1,0} ;    % factory on, on , off

% just one run, both JIT and factory on
j4 = {4} ;          % JIT     on
f1 = {1} ;          % factory on

% run twice
j44 = {4,4} ;       % JIT     on, on
f10 = {1,0} ;       % factory on, off

% run twice
j40 = {4,0} ;       % JIT     on, off
f11 = {1,1} ;       % factory on, on


% default
j404 = {0} ;    % JIT     on, off, on
f110 = {0} ;    % factory on, on , off

% just one run, both JIT and factory on
j4 = {0} ;          % JIT     on
f1 = {0} ;          % factory on

% run twice
j44 = {0} ;       % JIT     on, on
f10 = {0} ;       % factory on, off

% run twice
j40 = {0} ;       % JIT     on, off
f11 = {0} ;       % factory on, on

j4040 = {0} ;    % JIT     on, off, on , off
f1100 = {0} ;    % factory on, on , off, off

% start with the Werk stack enabled
hack (2) = 0 ; GB_mex_hack (hack) ;

malloc_debugging = stat ;

%===============================================================================
% statement coverage test, with malloc debugging
%===============================================================================

logstat ('test254'    ,t, j404, f110) ; % mask types

hack (2) = 1 ; GB_mex_hack (hack) ; % disable the Werk stack

logstat ('test191'    ,t, j4  , f1  ) ; % test split

hack (2) = 0 ; GB_mex_hack (hack) ; % re-enable the Werk stack

% Turn off malloc debugging
if (malloc_debugging)
    debug_off
    fprintf ('[malloc debugging turned off]\n') ;
    fp = fopen ('log.txt', 'a') ;
    fprintf (fp, '[malloc debugging turned off]\n') ;
    fclose (fp) ;
end

logstat ('test142'    ,t, j4040, f1100) ; % test GrB_assign with accum
logstat ('test125'    ,t, j4  , f1  ) ; % test GrB_mxm: row and column scaling

%----

if (malloc_debugging)
    debug_on
    fprintf ('[malloc debugging turned back on]\n') ;
    fp = fopen ('log.txt', 'a') ;
    fprintf (fp, '[malloc debugging turned back on]\n') ;
    fclose (fp) ;
end

