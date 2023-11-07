function testall (threads,longtests)
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

j0 = {0} ;          % JIT off
f0 = {0} ;          % factory off

% run twice
j44 = {4,4} ;       % JIT     on, on
f10 = {1,0} ;       % factory on, off
f00 = {0,0} ;       % factory off, off

% run twice
j04 = {0,4} ;       % JIT     off, off
j40 = {4,0} ;       % JIT     on, off
f11 = {1,1} ;       % factory on, on

j4040 = {4,0,4,0} ;    % JIT     on, off, on , off
f1100 = {1,1,0,0} ;    % factory on, on , off, off

j040 = {0,4,0} ;    % JIT     off, on , off

j440 = {4,4,0} ;    % JIT     on, on , off
f100 = {1,0,0} ;    % factory on, off, off

% start with the Werk stack enabled
hack (2) = 0 ; GB_mex_hack (hack) ;

malloc_debugging = stat ;

%===============================================================================
% statement coverage test, with malloc debugging
%===============================================================================

%----------------------------------------
% tests with high rates (over 100/sec)
%----------------------------------------

logstat ('test250'    ,t, j44 , f10 ) ; % JIT tests, set/get, other tests
logstat ('test279'    ,t, j0  , f1  ) ; % blob get/set
logstat ('test278'    ,t, j0  , f1  ) ; % descriptor get/set
logstat ('test277'    ,t, j0  , f1  ) ; % context get/set
logstat ('test276'    ,t, j0  , f1  ) ; % semiring get/set
logstat ('test275'    ,t, j0  , f1  ) ; % monoid get/set
logstat ('test274'    ,t, j0  , f1  ) ; % index unary op get/set
logstat ('test273'    ,t, j0  , f1  ) ; % global get/set
logstat ('test272'    ,t, j0  , f1  ) ; % misc simple tests
logstat ('test271'    ,t, j0  , f1  ) ; % binary op get/set
logstat ('test270'    ,t, j0  , f1  ) ; % unary op get/set
logstat ('test269'    ,t, j0  , f1  ) ; % get/set for type, scalar, vec, mtx
logstat ('test268'    ,t, j4  , f1  ) ; % C<M>=Z sparse masker
jall = {4,3,2,1,4,2} ;
fall = {1,1,1,1,0,0} ;
logstat ('test145'    ,t, jall, fall) ; % dot4 for C += A'*B

hack (2) = 1 ; GB_mex_hack (hack) ; % disable the Werk stack

logstat ('test240'    ,t, j4  , f1  ) ; % test dot4, saxpy4, and saxpy5
logstat ('test240'    ,s, j4  , f1  ) ; % test dot4, saxpy4, and saxpy5 (1 task)
logstat ('test237'    ,t, j440, f100) ; % test GrB_mxm (saxpy4)
logstat ('test237'    ,s, j40 , f10 ) ; % test GrB_mxm (saxpy4) (1 task)

hack (2) = 0 ; GB_mex_hack (hack) ; % re-enable the Werk stack

logstat ('test267'    ,t, j40 , f00 ) ; % JIT error handling
logstat ('test265'    ,t, j4  , f0  ) ; % reduce to scalar with user types
logstat ('test264'    ,t, j4  , f0  ) ; % enumify / macrofy tests
logstat ('test263'    ,t, j4  , f0  ) ; % JIT tests
logstat ('test262'    ,t, j0  , f1  ) ; % GB_mask
logstat ('test261'    ,t, j4  , f0  ) ; % serialize/deserialize error handling
logstat ('test260'    ,t, j4  , f0  ) ; % demacrofy name
logstat ('test259'    ,t, j4  , f0  ) ; % plus_plus_fp32 semiring
logstat ('test258'    ,t, j4  , f0  ) ; % reduce-to-vector for UDT
logstat ('test257'    ,t, j4  , f0  ) ; % JIT error handling
logstat ('test255'    ,t, j4  , f1  ) ; % flip binop
logstat ('test254'    ,t, j440, f100) ; %% mask types
logstat ('test253'    ,t, j4  , f1  ) ; % basic JIT tests
logstat ('test252'    ,t, j4  , f1  ) ; % basic tests
%ogstat ('test251'    ,t, j404, f110) ; % dot4, dot2, with plus_pair
logstat ('test251'    ,t, j44 , f10 ) ; % dot4, dot2, with plus_pair
logstat ('test249'    ,t, j4  , f1  ) ; % GxB_Context object
logstat ('test247'    ,t, j4  , f1  ) ; % GrB_mxm: fine Hash method
logstat ('test246'    ,t, j4  , f1  ) ; % GrB_mxm parallelism (slice_balanced)

%ogstat ('test01'     ,t, j44 , f10 ) ; % error handling
logstat ('test01'     ,t, j4  , f1  ) ; % error handling
logstat ('test245'    ,t, j40 , f11 ) ; % test complex row/col scale
logstat ('test199'    ,t, j4  , f1  ) ; % test dot2 with hypersparse
logstat ('test83'     ,t, j4  , f1  ) ; % GrB_assign with C_replace and empty J
logstat ('test210'    ,t, j4  , f1  ) ; % iso assign25: C<M,struct>=A
logstat ('test165'    ,t, j4  , f1  ) ; % test C=A*B', A is diagonal, B bitmap
logstat ('test219'    ,s, j44 , f10 ) ; % test reduce to scalar (1 thread)
logstat ('test241'    ,t, j4  , f1  ) ; % test GrB_mxm, trigger the swap_rule
logstat ('test220'    ,t, j4  , f1  ) ; % test mask C<M>=Z, iso case
logstat ('test211'    ,t, j4  , f1  ) ; % test iso assign
logstat ('test202'    ,t, j40 , f11 ) ; % test iso add and emult
logstat ('test152'    ,t, j404, f110) ; % test binops C=A+B, all matrices dense
logstat ('test222'    ,t, j4  , f1  ) ; % test user selectop for iso matrices

hack (2) = 1 ; GB_mex_hack (hack) ; % disable the Werk stack

logstat ('test256'    ,t, j4  , f0  ) ; % JIT error handling
logstat ('test186'    ,t, j40 , f11 ) ; % saxpy, all formats  (slice_balanced)
logstat ('test186(0)' ,t, j4  , f1  ) ; % repeat with default slice_balanced
%ogstat ('test150'    ,t, j40 , f10 ) ; %% mxm zombies, typecasting (dot3,saxpy)
logstat ('test150'    ,t, j0  , f0  ) ; %% mxm zombies, typecasting (dot3,saxpy)

hack (2) = 0 ; GB_mex_hack (hack) ; % re-enable the Werk stack

logstat ('test239'    ,t, j44 , f10 ) ; % test GxB_eWiseUnion
logstat ('test235'    ,t, j4  , f1  ) ; % test GxB_eWiseUnion and GrB_eWiseAdd
logstat ('test226'    ,t, j4  , f1  ) ; % test kron with iso matrices
logstat ('test223'    ,t, j4  , f1  ) ; % test matrix multiply, C<!M>=A*B
logstat ('test204'    ,t, j4  , f1  ) ; % test iso diag
logstat ('test203'    ,t, j4  , f1  ) ; % test iso subref
logstat ('test183'    ,s, j4  , f1  ) ; % test eWiseMult with hypersparse mask
logstat ('test179'    ,t, j44 , f10 ) ; % test bitmap select
logstat ('test174'    ,t, j4  , f1  ) ; % test GrB_assign C<A>=A
logstat ('test155'    ,t, j4  , f1  ) ; % test GrB_*_setElement, removeElement
%ogstat ('test156'    ,t, j44 , f10 ) ; % test GrB_assign C=A with typecasting
logstat ('test136'    ,s, j4  , f1  ) ; % subassignment special cases
logstat ('test02'     ,t, j4  , f1  ) ; % matrix copy and dup tests
logstat ('test109'    ,t, j404, f110) ; % terminal monoid with user-defined type
logstat ('test04'     ,t, j4  , f1  ) ; % simple mask and transpose test
logstat ('test207'    ,t, j4  , f1  ) ; % test iso subref
logstat ('test221'    ,t, j4  , f1  ) ; % test C += A, C is bitmap and A is full
logstat ('test162'    ,t, j4  , f1  ) ; % test C<M>=A*B with very sparse M
%ogstat ('test159'    ,t, j40 , f10 ) ; %% test A*B
logstat ('test159'    ,t, j0  , f0  ) ; %% test A*B
logstat ('test09'     ,t, j4  , f1  ) ; % duplicate I,J test of GB_mex_subassign
logstat ('test132'    ,t, j4  , f1  ) ; % setElement
%ogstat ('test141'    ,t, j404, f110) ; % eWiseAdd with dense matrices
logstat ('test141'    ,t, j0  , f1  ) ; % eWiseAdd with dense matrices
logstat ('testc2(1,1)',t, j44 , f10 ) ; % complex tests (quick case, builtin)
logstat ('test214'    ,t, j4  , f1  ) ; % test C<M>=A'*B (tricount)
logstat ('test213'    ,t, j4  , f1  ) ; % test iso assign (method 05d)
logstat ('test206'    ,t, j44 , f10 ) ; % test iso select
logstat ('test212'    ,t, j44 , f10 ) ; % test iso mask all zero
logstat ('test128'    ,t, j4  , f1  ) ; % eWiseMult, eWiseAdd, eWiseUnion cases
logstat ('test82'     ,t, j4  , f1  ) ; % GrB_extract with index range (hyper)

%----------------------------------------
% tests with good rates (30 to 100/sec)
%----------------------------------------

logstat ('test229'    ,t, j40 , f11 ) ; % test setElement
logstat ('test144'    ,t, j4  , f1  ) ; % cumsum

%----------------------------------------
% tests with decent rates (20 to 30/sec)
%----------------------------------------

hack (2) = 1 ; GB_mex_hack (hack) ; % disable the Werk stack

%ogstat ('test14'     ,t, j404, f110) ; % GrB_reduce
logstat ('test14'     ,t, j44 , f10 ) ; % GrB_reduce
logstat ('test180'    ,s, j4  , f1  ) ; % test assign and subassign (1 thread)
%ogstat ('test180'    ,t, j4  , f1  ) ; % test assign and subassign (4 threads)
logstat ('test236'    ,t, j4  , f1  ) ; % test GxB_Matrix_sort, GxB_Vector_sort

hack (2) = 0 ; GB_mex_hack (hack) ; % re-enable the Werk stack

%----------------------------------------
% tests with decent rates (10 to 20/sec)
%----------------------------------------

logstat ('test232'    ,t, j4  , f1  ) ; % test assign with GrB_Scalar
logstat ('test228'    ,t, j4  , f1  ) ; % test serialize/deserialize

%----------------------------------------
% tests with low coverage/sec rates (1/sec to 10/sec)
%----------------------------------------

hack (2) = 1 ; GB_mex_hack (hack) ; % disable the Werk stack

logstat ('test154'    ,t, j40 , f11 ) ; % apply with binop and scalar binding
logstat ('test238'    ,t, j44 , f10 ) ; % test GrB_mxm (dot4 and dot2)
%ogstat ('test151b'   ,t, j404, f110) ; % test bshift operator
logstat ('test151b'   ,t, j4  , f1  ) ; % test bshift operator
logstat ('test184'    ,t, j4  , f1  ) ; % special cases: mxm, transpose, build
logstat ('test191'    ,t, j40 , f10 ) ; %% test split
logstat ('test188'    ,t, j40 , f11 ) ; % test concat

hack (2) = 0 ; GB_mex_hack (hack) ; % re-enable the Werk stack

logstat ('test224'    ,t, j4  , f1  ) ; % test unpack/pack
logstat ('test196'    ,t, j4  , f1  ) ; % test hypersparse concat
logstat ('test209'    ,t, j4  , f1  ) ; % test iso build
logstat ('test104'    ,t, j4  , f1  ) ; % export/import

%----------------------------------------
% tests with very low coverage/sec rates  (< 1/sec)
%----------------------------------------

logstat ('test189'    ,t, j4  , f1  ) ; % test large assign
logstat ('test194'    ,t, j4  , f1  ) ; % test GxB_Vector_diag
logstat ('test76'     ,s, j4  , f1  ) ; % GxB_resize (single threaded)
logstat ('test244'    ,t, j4  , f1  ) ; % test GxB_Matrix_reshape*

%===============================================================================
% tests with no malloc debugging
%===============================================================================

% Turn off malloc debugging
if (malloc_debugging)
    debug_off
    fprintf ('[malloc debugging turned off]\n') ;
    fp = fopen ('log.txt', 'a') ;
    fprintf (fp, '[malloc debugging turned off]\n') ;
    fclose (fp) ;
end

%----------------------------------------
% tests with good rates (30 to 100/sec)
%----------------------------------------

logstat ('test201'    ,t, j4  , f1  ) ; % test iso reduce to vector
logstat ('test225'    ,t, j4  , f1  ) ; % test mask operations (GB_masker)
logstat ('test176'    ,t, j4  , f1  ) ; % test GrB_assign, method 09, 11
logstat ('test208'    ,t, j4  , f1  ) ; % test iso apply, bind 1st and 2nd
logstat ('test216'    ,t, j4  , f1  ) ; % test C<A>=A, iso case
%ogstat ('test142'    ,t, j4040, f1100) ; %% test GrB_assign with accum
logstat ('test142'    ,t, j040, f100) ; %% test GrB_assign with accum
logstat ('test137'    ,s, j40 , f11 ) ; % GrB_eWiseMult, FIRST and SECOND
logstat ('test139'    ,s, j4  , f1  ) ; % merge sort, special cases
logstat ('test172'    ,t, j4  , f1  ) ; % test eWiseMult with M bitmap/full
logstat ('test148'    ,t, j4  , f1  ) ; % ewise with alias

%----------------------------------------
% tests with decent rates (20 to 30/sec)
%----------------------------------------

logstat ('test157'    ,t, j4  , f1  ) ; % test sparsity formats
logstat ('test182'    ,s, j4  , f1  ) ; % test for internal wait

%----------------------------------------
% tests with decent rates (10 to 20/sec)
%----------------------------------------

logstat ('test108'    ,t, j40 , f10 ) ; % boolean monoids
logstat ('test130'    ,t, j4  , f1  ) ; % GrB_apply, hypersparse cases
logstat ('test124'    ,t, j4  , f1  ) ; % GrB_extract, case 6
logstat ('test138'    ,s, j4  , f1  ) ; % assign, coarse-only tasks in IxJ slice
logstat ('test227'    ,t, j4  , f1  ) ; % test kron
logstat ('test125'    ,t, j4  , f1  ) ; % test GrB_mxm: row and column scaling

%----------------------------------------
% 1 to 10/sec
%----------------------------------------

logstat ('test234'    ,t, j40 , f11 ) ; % test GxB_eWiseUnion
logstat ('test242'    ,t, j4  , f1  ) ; % test GxB_Iterator for matrices
logstat ('test173'    ,t, j4  , f1  ) ; % test GrB_assign C<A>=A
logstat ('test200'    ,t, j4  , f1  ) ; % test iso full matrix multiply
logstat ('test197'    ,t, j4  , f1  ) ; % test large sparse split
logstat ('test84'     ,t, j4  , f1  ) ; % GrB_assign (row/col with C CSR/CSC)
%ogstat ('test19b'    ,t, j4  , f1  ) ; % GrB_assign, many pending operators
logstat ('test19b'    ,s, j4  , f1  ) ; % GrB_assign, many pending operators
logstat ('test133'    ,t, j4  , f1  ) ; % test mask operations (GB_masker)
logstat ('test80'     ,t, j4  , f1  ) ; % test GrB_mxm on all semirings
%ogstat ('test151'    ,t, j44 , f10 ) ; % test bitwise operators
%ogstat ('test23'     ,t, j40 , f11 ) ; % quick test of GB_*_build
logstat ('test23'     ,t, j0  , f1  ) ; % quick test of GB_*_build
logstat ('test135'    ,t, j4  , f1  ) ; % reduce to scalar
%ogstat ('test160'    ,s, j40 , f11 ) ; % test A*B, single threaded
logstat ('test160'    ,s, j0  , f1  ) ; % test A*B, single threaded
logstat ('test54'     ,t, j4  , f1  ) ; % assign and extract with begin:inc:end
logstat ('test129'    ,t, j4  , f1  ) ; % test GxB_select (tril, nonz, hyper)
logstat ('test69'     ,t, j4  , f1  ) ; % assign and subassign with alias
logstat ('test230'    ,t, j4  , f1  ) ; % test apply with idxunops
%ogstat ('test74'     ,t, j40 , f11 ) ; % test GrB_mxm on all semirings
logstat ('test74'     ,t, j0  , f1  ) ; % test GrB_mxm on all semirings
%ogstat ('test127'    ,t, j40 , f11 ) ; % test eWiseAdd, eWiseMult
logstat ('test127'    ,t, j0  , f1  ) ; % test eWiseAdd, eWiseMult
logstat ('test19'     ,t, j4  , f1  ) ; % GxB_subassign, many pending operators

%----------------------------------------
% < 1 per sec
%----------------------------------------

logstat ('test11'     ,t, j4  , f1  ) ; % exhaustive test of GrB_extractTuples
logstat ('test215'    ,t, j4  , f1  ) ; % test C<M>=A'*B (dot2, ANY_PAIR)
logstat ('test193'    ,t, j4  , f1  ) ; % test GxB_Matrix_diag
%ogstat ('test195'    ,t, j0  , f1  ) ; % all variants of saxpy3 slice_balanced
logstat ('test195'    ,t, j4  , f1  ) ; % all variants of saxpy3 slice_balanced
%ogstat ('test233'    ,t, j4  , f1  ) ; % bitmap saxpy C=A*B, A sparse, B bitmap
logstat ('test243'    ,t, j4  , f1  ) ; % test GxB_Vector_Iterator
%ogstat ('test29'     ,t, j40 , f11 ) ; % reduce with zombies
logstat ('test29'     ,t, j0  , f1  ) ; % reduce with zombies

%ogstat ('testc2(0,0)',t, j404, f110) ; % A'*B, A+B, A*B, user-defined complex
logstat ('testc2(0,0)',t, j04,  f10 ) ; % A'*B, A+B, A*B, user-defined complex
logstat ('testc4(0)'  ,t, j4  , f1  ) ; % extractElement, setElement, user type
logstat ('testc7(0)'  ,t, j4  , f1  ) ; % assign, builtin complex
logstat ('testcc(1)'  ,t, j4  , f1  ) ; % transpose, builtin complex

hack (2) = 1 ; GB_mex_hack (hack) ; % disable the Werk stack

logstat ('test187'    ,t, j4  , f1  ) ; % test dup/assign for all formats
logstat ('test192'    ,t, j4  , f1  ) ; % test C<C,struct>=scalar
logstat ('test181'    ,s, j4  , f1  ) ; % transpose with explicit zeros in mask
logstat ('test185'    ,s, j4  , f1  ) ; % test dot4, saxpy for all sparsity

hack (2) = 0 ; GB_mex_hack (hack) ; % re-enable the Werk stack

logstat ('test53'     ,t, j4  , f1  ) ; % quick test of GB_mex_Matrix_extract
logstat ('test17'     ,t, j4  , f1  ) ; % quick test of GrB_*_extractElement
%ogstat ('test231'    ,t, j0  , f1  ) ; % test GrB_select with idxunp
logstat ('test231'    ,t, j4  , f1  ) ; % test GrB_select with idxunp

%----------------------------------------
% longer tests (200 seconds to 600 seconds, or low rate of coverage)
%----------------------------------------

logstat ('test10'     ,t, j4  , f1  ) ; % GrB_apply
logstat ('test75b'    ,t, j4  , f1  ) ; % test GrB_mxm A'*B
logstat ('test21b'    ,t, j4  , f1  ) ; % quick test of GB_mex_assign
logstat ('testca(1)'  ,t, j4  , f1  ) ; % test complex mxm, mxv, and vxm
logstat ('test81'     ,t, j4  , f1  ) ; % extract with stride, range, backwards
logstat ('test18'     ,t, j4  , f1  ) ; % GrB_eWiseAdd and eWiseMult

if (malloc_debugging)
    debug_on
    fprintf ('[malloc debugging turned back on]\n') ;
    fp = fopen ('log.txt', 'a') ;
    fprintf (fp, '[malloc debugging turned back on]\n') ;
    fclose (fp) ;
end

t = toc (testall_time) ;
fprintf ('\ntestall: all tests passed, total time %0.4g minutes\n', t / 60) ;

