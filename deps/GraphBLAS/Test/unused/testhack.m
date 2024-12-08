function testall (threads, mdebug)
%TESTALL run all GraphBLAS tests
%
% Usage:
%
% testall ;             % runs just the shorter tests
% testall(threads) ;    % run with specific list of threads and chunk sizes
% testall(threads,1) ;  % runs with malloc debugging enabled
%
% threads is a cell array. Each entry is 2-by-1, with the first value being
% the # of threads to use and the 2nd being the chunk size.  The default is
% {[4 1]} if threads is empty or not present.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2024, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

GrB.init ;
GB_mex_init ;
testall_time = tic ;

if (nargin < 1)
    threads = [ ] ;
end
if (isempty (threads))
    threads {1} = [4 1] ;
end
t = threads ;

if (nargin < 2)
    mdebug = false ;
end

% single thread
s {1} = [1 1] ;

% clear the statement coverage counts
%% clear global GraphBLAS_grbcov

global GraphBLAS_debug GraphBLAS_grbcov GraphBLAS_grbcovs ...
    GraphBLAS_scripts GraphBLAS_times

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

% run once
j4 = {4} ;          % JIT     on
f1 = {1} ;          % factory on
j0 = {0} ;          % JIT     off
f0 = {0} ;          % factory off

% run twice
j44 = {4,4} ;       % JIT     on, on
j04 = {0,4} ;       % JIT     off, on
j40 = {4,0} ;       % JIT     on, off
f10 = {1,0} ;       % factory on, off
f00 = {0,0} ;       % factory off, off
f11 = {1,1} ;       % factory on, on
j42 = {4,2} ;       % JIT     on, pause

% 3 runs
j440 = {4,4,0} ;    % JIT     on, on , off
j404 = {4,0,4} ;    % JIT     on, off, on
f100 = {1,0,0} ;    % factory on, off, off
f110 = {1,1,0} ;    % factory on, on , off

% start with the Werk stack enabled
hack (2) = 0 ; GB_mex_hack (hack) ;

% save the current malloc debug status
debug_save = stat ;

%===============================================================================
% quick tests (< 1 sec)
%===============================================================================

% < 1 second: debug_off
set_malloc_debug (mdebug, 0)
logstat ('test109'    ,t, j404, f110) ; % terminal monoid with user-defined type
logstat ('test138'    ,s, j4  , f1  ) ; % assign, coarse-only tasks in IxJ slice
logstat ('test139'    ,s, j4  , f1  ) ; % merge sort, special cases
logstat ('test172'    ,t, j4  , f1  ) ; % test eWiseMult with M bitmap/full
logstat ('test155'    ,t, j4  , f1  ) ; % test GrB_*_setElement, removeElement
logstat ('test174'    ,t, j4  , f1  ) ; % test GrB_assign C<A>=A
logstat ('test203'    ,t, j4  , f1  ) ; % test iso subref
logstat ('test213'    ,t, j4  , f1  ) ; % test iso assign (method 05d)
logstat ('test216'    ,t, j4  , f1  ) ; % test C<A>=A, iso case
logstat ('test225'    ,t, j4  , f1  ) ; % test mask operations (GB_masker)
logstat ('test226'    ,t, j4  , f1  ) ; % test kron with iso matrices
logstat ('test235'    ,t, j4  , f1  ) ; % test GxB_eWiseUnion and GrB_eWiseAdd
logstat ('test252'    ,t, j4  , f1  ) ; % basic tests
logstat ('test253'    ,t, j4  , f1  ) ; % basic JIT tests
logstat ('test255'    ,t, j4  , f1  ) ; % flip binop
logstat ('test257'    ,t, j4  , f0  ) ; % JIT error handling
logstat ('test260'    ,t, j4  , f0  ) ; % demacrofy name
logstat ('test261'    ,t, j4  , f0  ) ; % serialize/deserialize error handling
logstat ('test262'    ,t, j0  , f1  ) ; % GB_mask
logstat ('test263'    ,t, j4  , f0  ) ; % JIT tests
logstat ('test264'    ,t, j4  , f0  ) ; % enumify / macrofy tests
logstat ('test265'    ,t, j4  , f0  ) ; % reduce to scalar with user types
logstat ('test267'    ,t, j4  , f0  ) ; % JIT error handling
logstat ('test269'    ,t, j0  , f1  ) ; % get/set for type, scalar, vec, mtx
logstat ('test271'    ,t, j0  , f1  ) ; % binary op get/set
logstat ('test272'    ,t, j0  , f1  ) ; % misc simple tests
logstat ('test273'    ,t, j0  , f1  ) ; % Global get/set
logstat ('test274'    ,t, j0  , f1  ) ; % index unary op get/set
logstat ('test276'    ,t, j0  , f1  ) ; % semiring get/set
logstat ('test277'    ,t, j0  , f1  ) ; % context get/set
logstat ('test279'    ,t, j0  , f1  ) ; % blob get/set
logstat ('test281'    ,t, j4  , f1  ) ; % test user-defined idx unop, no JIT
logstat ('test268'    ,t, j4  , f1  ) ; % C<M>=Z sparse masker
logstat ('test247'    ,t, j4  , f1  ) ; % GrB_mxm: fine Hash method
logstat ('test207'    ,t, j4  , f1  ) ; % test iso subref
logstat ('test211'    ,t, j4  , f1  ) ; % test iso assign
logstat ('test183'    ,s, j4  , f1  ) ; % test eWiseMult with hypersparse mask
logstat ('test212'    ,t, j44 , f10 ) ; % test iso mask all zero
logstat ('test219'    ,s, j44 , f10 ) ; % test reduce to scalar (1 thread)

% < 1 second: debug_on
set_malloc_debug (mdebug, 1)
logstat ('test09'     ,t, j4  , f1  ) ; % duplicate I,J test of GB_mex_subassign
logstat ('test108'    ,t, j40 , f10 ) ; % boolean monoids
logstat ('test137'    ,s, j40 , f11 ) ; % GrB_eWiseMult, FIRST and SECOND
logstat ('test124'    ,t, j4  , f1  ) ; % GrB_extract, case 6
logstat ('test133'    ,t, j4  , f1  ) ; % test mask operations (GB_masker)
logstat ('test176'    ,t, j4  , f1  ) ; % test GrB_assign, method 09, 11
logstat ('test197'    ,t, j4  , f1  ) ; % test large sparse split
logstat ('test201'    ,t, j4  , f1  ) ; % test iso reduce to vector and scalar
logstat ('test208'    ,t, j4  , f1  ) ; % test iso apply, bind 1st and 2nd
logstat ('test214'    ,t, j4  , f1  ) ; % test C<M>=A'*B (tricount)
logstat ('test223'    ,t, j4  , f1  ) ; % test matrix multiply, C<!M>=A*B
logstat ('test241'    ,t, j4  , f1  ) ; % test GrB_mxm, trigger the swap_rule
logstat ('test270'    ,t, j0  , f1  ) ; % unary op get/set
logstat ('test199'    ,t, j4  , f1  ) ; % test dot2 with hypersparse
logstat ('test210'    ,t, j4  , f1  ) ; % iso assign25: C<M,struct>=A
logstat ('test165'    ,t, j4  , f1  ) ; % test C=A*B', A is diagonal, B bitmap
logstat ('test221'    ,t, j4  , f1  ) ; % test C += A, C is bitmap and A is full
logstat ('test278'    ,t, j0  , f1  ) ; % descriptor get/set
logstat ('test162'    ,t, j4  , f1  ) ; % test C<M>=A*B with very sparse M
logstat ('test275'    ,t, j0  , f1  ) ; % monoid get/set
logstat ('test220'    ,t, j4  , f1  ) ; % test mask C<M>=Z, iso case
logstat ('test83'     ,t, j4  , f1  ) ; % GrB_assign with C_replace and empty J
logstat ('test04'     ,t, j4  , f1  ) ; % simple mask and transpose test
logstat ('test132'    ,t, j4  , f1  ) ; % setElement
logstat ('test82'     ,t, j4  , f1  ) ; % GrB_extract with index range (hyper)
logstat ('test202'    ,t, j40 , f11 ) ; % test iso add and emult
logstat ('test222'    ,t, j4  , f1  ) ; % test user selectop for iso matrices
logstat ('test204'    ,t, j4  , f1  ) ; % test iso diag
logstat ('test258'    ,t, j4  , f0  ) ; % reduce-to-vector for UDT
logstat ('test136'    ,s, j4  , f1  ) ; % subassignment special cases
logstat ('test128'    ,t, j4  , f1  ) ; % eWiseMult, eWiseAdd, eWiseUnion cases
logstat ('test144'    ,t, j4  , f1  ) ; % cumsum

%===============================================================================
% 1 to 10 seconds
%===============================================================================

% 1 to 10 seconds: debug_off
set_malloc_debug (mdebug, 0)
logstat ('testc2(0,0)',t, j0  , f1  ) ; % A'*B, A+B, A*B, user-defined complex
logstat ('test239'    ,t, j44 , f10 ) ; % test GxB_eWiseUnion
logstat ('test245'    ,t, j40 , f11 ) ; % test complex row/col scale
logstat ('test159'    ,t, j0  , f0  ) ; % test A*B
logstat ('test259'    ,t, j4  , f0  ) ; % plus_plus_fp32 semiring
logstat ('testc4(0)'  ,t, j4  , f1  ) ; % extractElement, setElement, user type
logstat ('test157'    ,t, j4  , f1  ) ; % test sparsity formats
logstat ('test182'    ,s, j4  , f1  ) ; % test for internal wait
logstat ('test195'    ,t, j4  , f1  ) ; % all variants of saxpy3 slice_balanced
logstat ('test173'    ,t, j4  , f1  ) ; % test GrB_assign C<A>=A
logstat ('test135'    ,t, j4  , f1  ) ; % reduce to scalar
logstat ('test84'     ,t, j4  , f1  ) ; % GrB_assign (row/col with C CSR/CSC)
logstat ('test215'    ,t, j4  , f1  ) ; % test C<M>=A'*B (dot2, ANY_PAIR)
logstat ('test80'     ,t, j4  , f1  ) ; % test GrB_mxm on all semirings
logstat ('test200'    ,t, j4  , f1  ) ; % test iso full matrix multiply
logstat ('test283'    ,t, j4  , f1  ) ; % test index binary op
logstat ('test254'    ,t, j44 , f10 ) ; % mask types
logstat ('test19'     ,t, j4  , f1  ) ; % GxB_subassign, many pending operators
logstat ('test142'    ,t, j0  , f1  ) ; % test GrB_assign with accum
logstat ('test142b'   ,t, j4  , f0  ) ; % test GrB_assign with accum
logstat ('test54'     ,t, j4  , f1  ) ; % assign and extract with begin:inc:end
logstat ('testcc(1)'  ,t, j4  , f1  ) ; % transpose, builtin complex
logstat ('testc2(1,1)',t, j44 , f10 ) ; % complex tests (quick case, builtin)
logstat ('test227'    ,t, j4  , f1  ) ; % test kron
logstat ('test19b'    ,s, j4  , f1  ) ; % GrB_assign, many pending operators
logstat ('test141'    ,t, j0  , f1  ) ; % eWiseAdd with dense matrices
logstat ('test179'    ,t, j44 , f10 ) ; % test bitmap select
logstat ('test232'    ,t, j4  , f1  ) ; % test assign with GrB_Scalar

% 1 to 10 seconds, no Werk, debug_off
hack (2) = 1 ; GB_mex_hack (hack) ; % disable the Werk stack
logstat ('test256'    ,t, j4  , f0  ) ; % JIT error handling
hack (2) = 0 ; GB_mex_hack (hack) ; % re-enable the Werk stack

% 1 to 10 seconds: debug_on
set_malloc_debug (mdebug, 1)
logstat ('test130'    ,t, j4  , f1  ) ; % GrB_apply, hypersparse cases
logstat ('test148'    ,t, j4  , f1  ) ; % ewise with alias
logstat ('test231'    ,t, j4  , f1  ) ; % test GrB_select with idxunp
logstat ('test129'    ,t, j4  , f1  ) ; % test GxB_select (tril, nonz, hyper)
logstat ('test69'     ,t, j4  , f1  ) ; % assign and subassign with alias
logstat ('test11'     ,t, j4  , f1  ) ; % exhaustive test of GrB_extractTuples
logstat ('test29'     ,t, j0  , f1  ) ; % reduce with zombies
logstat ('test282'    ,t, j42 , f11 ) ; % test argmax, index binary op
logstat ('test249'    ,t, j4  , f1  ) ; % GxB_Context object
logstat ('test196'    ,t, j4  , f1  ) ; % test hypersparse concat
logstat ('test250'    ,t, j44 , f10 ) ; % JIT tests, set/get, other tests
logstat ('test145'    ,t, j42 , f11 ) ; % dot4 for C += A'*B
logstat ('test229'    ,t, j40 , f11 ) ; % test setElement
logstat ('test209'    ,t, j4  , f1  ) ; % test iso build
logstat ('test224'    ,t, j4  , f1  ) ; % test unpack/pack

% 1 to 10 seconds, no Werk, debug_on
hack (2) = 1 ; GB_mex_hack (hack) ; % disable the Werk stack
logstat ('test150'    ,t, j0  , f0  ) ; % mxm zombies, typecasting (dot3,saxpy)
logstat ('test240'    ,t, j4  , f1  ) ; % test dot4, saxpy4, and saxpy5
logstat ('test240'    ,s, j4  , f1  ) ; % test dot4, saxpy4, and saxpy5 (1 task)
logstat ('test237'    ,t, j40 , f10 ) ; % test GrB_mxm (saxpy4)
logstat ('test237'    ,s, j40 , f10 ) ; % test GrB_mxm (saxpy4) (1 task)
logstat ('test184'    ,t, j4  , f1  ) ; % special cases: mxm, transpose, build
logstat ('test236'    ,t, j4  , f1  ) ; % test GxB_Matrix_sort, GxB_Vector_sort
hack (2) = 0 ; GB_mex_hack (hack) ; % re-enable the Werk stack

%===============================================================================
% 10 to 100 seconds
%===============================================================================

% 10 to 100 seconds: debug_off
set_malloc_debug (mdebug, 0)
logstat ('test18'     ,t, j4  , f1  ) ; % GrB_eWiseAdd and eWiseMult
logstat ('testc7(0)'  ,t, j4  , f1  ) ; % assign, builtin complex
logstat ('test193'    ,t, j4  , f1  ) ; % test GxB_Matrix_diag
logstat ('test127'    ,t, j0  , f1  ) ; % test eWiseAdd, eWiseMult
logstat ('test23'     ,t, j0  , f1  ) ; % quick test of GB_*_build
logstat ('test243'    ,t, j4  , f1  ) ; % test GxB_Vector_Iterator
logstat ('test53'     ,t, j4  , f1  ) ; % quick test of GB_mex_Matrix_extract
logstat ('test242'    ,t, j4  , f1  ) ; % test GxB_Iterator for matrices
logstat ('test17'     ,t, j4  , f1  ) ; % quick test of GrB_*_extractElement
logstat ('test246'    ,t, j4  , f1  ) ; % GrB_mxm parallelism (slice_balanced)
logstat ('test206'    ,t, j44 , f10 ) ; % test iso select
logstat ('test251'    ,t, j4  , f1  ) ; % dot4, dot2, with plus_pair
logstat ('test251b'   ,t, j4  , f0  ) ; % dot4, dot2, with plus_pair
logstat ('test152'    ,t, j44 , f10 ) ; % test binops C=A+B, all matrices dense
logstat ('test160'    ,s, j0  , f1  ) ; % test A*B, single threaded

% 10 to 100 seconds, no Werk, debug_off
hack (2) = 1 ; GB_mex_hack (hack) ; % disable the Werk stack
logstat ('test188b'   ,t, j0  , f1  ) ; % test concat
logstat ('test186'    ,t, j4  , f1  ) ; % saxpy, all formats (slice_balanced)
%ogstat ('test186'    ,t, j40 , f11 ) ; % saxpy, all formats (slice_balanced)
%ogstat ('test186(0)' ,t, j4  , f1  ) ; % repeat with default slice_balanced
logstat ('test192'    ,t, j4  , f1  ) ; % test C<C,struct>=scalar
logstat ('test187'    ,t, j4  , f1  ) ; % test dup/assign for all formats
logstat ('test181'    ,s, j4  , f1  ) ; % transpose with explicit zeros in mask
logstat ('test238'    ,t, j44 , f10 ) ; % test GrB_mxm (dot4 and dot2)
hack (2) = 0 ; GB_mex_hack (hack) ; % re-enable the Werk stack

% 10 to 100 seconds: debug_on
set_malloc_debug (mdebug, 1)
logstat ('test189'    ,t, j4  , f1  ) ; % test large assign
logstat ('test169'    ,t, j0  , f1  ) ; % C<M>=A+B with many formats
logstat ('test76'     ,s, j4  , f1  ) ; % GxB_resize (single threaded)
logstat ('test01'     ,t, j4  , f1  ) ; % error handling
logstat ('test228'    ,t, j4  , f1  ) ; % test serialize/deserialize
logstat ('test104'    ,t, j4  , f1  ) ; % export/import
logstat ('test244'    ,t, j4  , f1  ) ; % test GxB_Matrix_reshape*

% 10 to 100 seconds, no Werk, debug_on
hack (2) = 1 ; GB_mex_hack (hack) ; % disable the Werk stack
logstat ('test180'    ,s, j4  , f1  ) ; % test assign and subassign (1 thread)
logstat ('test188'    ,t, j4  , f1  ) ; % test concat
logstat ('test151b'   ,t, j4  , f1  ) ; % test bshift operator
logstat ('test191'    ,t, j40 , f10 ) ; % test split
logstat ('test14'     ,t, j4  , f1  ) ; % GrB_reduce
logstat ('test14b'    ,t, j4  , f0  ) ; % GrB_reduce
hack (2) = 0 ; GB_mex_hack (hack) ; % re-enable the Werk stack

%===============================================================================
% > 100 seconds
%===============================================================================

% > 100 seconds, debug_off
set_malloc_debug (mdebug, 0)
logstat ('test125'    ,t, j4  , f1  ) ; % test GrB_mxm: row and column scaling
logstat ('test280'    ,t, j4  , f1  ) ; % subassign method 26
logstat ('test10'     ,t, j4  , f1  ) ; % GrB_apply
logstat ('test75b'    ,t, j4  , f1  ) ; % test GrB_mxm A'*B
logstat ('test81'     ,t, j4  , f1  ) ; % extract with stride, range, backwards
logstat ('test230'    ,t, j4  , f1  ) ; % test apply with idxunops
logstat ('test21b'    ,t, j4  , f1  ) ; % quick test of GB_mex_assign
logstat ('test74'     ,t, j0  , f1  ) ; % test GrB_mxm on all semirings
logstat ('test234'    ,t, j4  , f1  ) ; % test GxB_eWiseUnion
logstat ('test234b'   ,t, j0  , f1  ) ; % test GxB_eWiseUnion

% > 100 seconds, no Werk, debug_off
hack (2) = 1 ; GB_mex_hack (hack) ; % disable the Werk stack
logstat ('test185'    ,s, j4  , f1  ) ; % test dot4, saxpy for all sparsity
hack (2) = 0 ; GB_mex_hack (hack) ; % re-enable the Werk stack

% > 100 seconds, debug_on
set_malloc_debug (mdebug, 1)
logstat ('testca(1)'  ,t, j4  , f1  ) ; % test complex mxm, mxv, and vxm
logstat ('test194'    ,t, j4  , f1  ) ; % test GxB_Vector_diag

% > 100 seconds, no Werk, debug_on
hack (2) = 1 ; GB_mex_hack (hack) ; % disable the Werk stack
logstat ('test154'    ,t, j4  , f1  ) ; % apply with binop and scalar binding
logstat ('test154b'   ,t, j0  , f1  ) ; % apply with binop and scalar binding
hack (2) = 0 ; GB_mex_hack (hack) ; % re-enable the Werk stack

%===============================================================================
% finalize
%===============================================================================

% restore the original malloc debug state
set_malloc_debug (mdebug, debug_save)
t = toc (testall_time) ;
fprintf ('\ntestall: all tests passed, total time %0.4g minutes\n', t / 60) ;

