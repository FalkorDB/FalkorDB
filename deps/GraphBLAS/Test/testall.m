function testall (threads, mdebug)
%TESTALL run all GraphBLAS tests
%
% Usage:
%
% testall ;             % runs just the shorter tests
% testall(threads) ;    % run with specific list of threads and chunk sizes
% testall(threads,1) ;  % runs with malloc debugging enabled
% testall([ ],1) ;      % default # threads, with malloc debugging enabled
%
% threads is a cell array. Each entry is 2-by-1, with the first value being
% the # of threads to use and the 2nd being the chunk size.  The default is
% {[4 1]} if threads is empty or not present.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

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

% all pji_controls
o = 0:7 ;

% clear the statement coverage counts
clear global GraphBLAS_grbcov

global GraphBLAS_debug GraphBLAS_grbcov

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

% start with the Werk stack enabled
hack (2) = 0 ; GB_mex_hack (hack) ;     % enable the Werk stack

% save the current malloc debug status
debug_save = stat ;

% run once
J4 = {4} ;          % JIT     on
F1 = {1} ;          % factory on
J0 = {0} ;          % JIT     off
F0 = {0} ;          % factory off

% run twice
J44 = {4,4} ;       % JIT     on, on
J40 = {4,0} ;       % JIT     on, off
F10 = {1,0} ;       % factory on, off
F11 = {1,1} ;       % factory on, on
J42 = {4,2} ;       % JIT     on, pause
F00 = {0,0} ;
J00 = {0,0} ;

% 3 runs
J404 = {4,0,4} ;    % JIT     on, off, on
F110 = {1,1,0} ;    % factory on, on , off
J440 = {4,4,0} ;
J400 = {4,0,0} ;
F100 = {4,4,0} ;

% 4 runs:
J4040 = {4,0,4,0} ;
F1100 = {1,1,0,0} ;

%{
J4 = {4,0} ;          % JIT     on
F1 = {1,0} ;          % factory on
J0 = {0,0} ;          % JIT     off
F0 = {0,0} ;          % factory off

% run twice
J44 = {4,4,0} ;       % JIT     on, on
J40 = {4,0,0} ;       % JIT     on, off
F10 = {1,0,0} ;       % factory on, off
F11 = {1,1,0} ;       % factory on, on
J42 = {4,2,0} ;       % JIT     on, pause
F00 = {0,0,0} ;
J00 = {0,0,0} ;

% 3 runs
J404 = {4,0,4,0} ;    % JIT     on, off, on
F110 = {1,1,0,0} ;    % factory on, on , off
J440 = {4,4,0,0} ;
J400 = {4,0,0,0} ;
F100 = {4,4,0,0} ;
%}

%===============================================================================
% quick tests (< 1 sec)
%===============================================================================

% < 1 second: debug_on
set_malloc_debug (mdebug, 1) ;
logstat ('test303'    ,t, J404 , F110 ) ; % C=A(I,J), method 6
logstat ('test300'    ,t, J0   , F0   ) ; % print function for a type
logstat ('test301'    ,t, J40  , F11  ) ; % assign method27, C<C,struct>+=A
logstat ('test302'    ,t, J0   , F0   ) ; % GPU controls
logstat ('test155'    ,t, J40  , F10  , [0 2 4]) ; % setElement, removeElement

% < 1 second: debug_off
set_malloc_debug (mdebug, 0) ;
logstat ('test299'    ,t, J0   , F0   ) ; % unload a vector, with wait
logstat ('test298'    ,t, J40  , F10  ) ; % assign 08n when A is full
logstat ('test297'    ,t, J4   , F0   ) ; % plus_one semiring
logstat ('test295'    ,t, J4   , F1   ) ; % get/set iso
logstat ('test294'    ,t, J0   , F1   ) ; % reduce with zombies
logstat ('test293'    ,t, J0   , F0   ) ; % msort/qsort: all int variants
logstat ('test291'    ,t, J0   , F0   , [0 1 2 4]) ; % GB_ix_realloc
logstat ('test290'    ,t, J0   , F0   ) ; % large symbolic bitmap_subref
logstat ('test287'    ,t, J0   , F0   , [0 4]) ; % misc tests
logstat ('test286'    ,t, J40  , F00  , [0 1 2 4]) ; % kron with index binop
logstat ('test78'     ,t, J40  , F00  , [0 4]) ; % subref
logstat ('test285'    ,t, J40  , F00  ) ; % GB_mex_assign (bitmap, 7_whole)
logstat ('test247'    ,t, J40  , F10  ) ; % GrB_mxm: fine Hash method
logstat ('test109'    ,t, J4040, F1100) ; % terminal monoid with user-defn type
logstat ('test138'    ,s, J40  , F10  ) ; % assign, coarse-only in IxJ slice
logstat ('test172'    ,t, J40  , F10  ) ; % eWiseMult with M bitmap/full
logstat ('test174'    ,t, J40  , F10  ) ; % GrB_assign C<A>=A
logstat ('test203'    ,t, J4   , F1   ) ; % iso subref
logstat ('test213'    ,t, J40  , F10  ) ; % iso assign (method 05d)
logstat ('test216'    ,t, J4   , F1   ) ; % C<A>=A, iso case
logstat ('test225'    ,t, J40  , F10  ) ; % mask operations (GB_masker)
logstat ('test226'    ,t, J40  , F10  ) ; % kron with iso matrices
logstat ('test235'    ,t, J40  , F10  ) ; % GxB_eWiseUnion, GrB_eWiseAdd
logstat ('test252'    ,t, J4   , F1   ) ; % basic tests
logstat ('test253'    ,t, J4   , F1   ) ; % basic JIT tests
logstat ('test255'    ,t, J4   , F1   ) ; % flip binop
logstat ('test257'    ,t, J40  , F00  ) ; % JIT error handling
logstat ('test260'    ,t, J4   , F0   ) ; % demacrofy name
logstat ('test261'    ,t, J4   , F0   ) ; % serialize/deserialize errors
logstat ('test262'    ,t, J0   , F1   ) ; % GB_mask
logstat ('test263'    ,t, J40  , F00  ) ; % JIT tests
logstat ('test264'    ,t, J4   , F0   ) ; % enumify / macrofy tests
logstat ('test265'    ,t, J40  , F00  ) ; % reduce to scalar with user types
logstat ('test267'    ,t, J4   , F0   ) ; % JIT error handling
logstat ('test269'    ,t, J0   , F1   ) ; % get/set for type, scalar, vec, mtx
logstat ('test271'    ,t, J0   , F1   ) ; % binary op get/set
logstat ('test272'    ,t, J0   , F1   ) ; % misc simple tests
logstat ('test273'    ,t, J0   , F1   ) ; % Global get/set
logstat ('test274'    ,t, J0   , F1   ) ; % index unary op get/set
logstat ('test276'    ,t, J0   , F1   ) ; % semiring get/set
logstat ('test277'    ,t, J0   , F1   ) ; % context get/set
logstat ('test279'    ,t, J0   , F1   ) ; % blob get/set
logstat ('test281'    ,t, J4   , F1   ) ; % user-defined idx unop, no JIT
logstat ('test268'    ,t, J40  , F10  ) ; % C<M>=Z sparse masker
logstat ('test207'    ,t, J4   , F1   , [0 1]) ; % iso subref
logstat ('test211'    ,t, J40  , F10  ) ; % iso assign
logstat ('test183'    ,s, J4   , F1   ) ; % eWiseMult w/hypersparse mask
logstat ('test212'    ,t, J40  , F10  ) ; % iso mask all zero
logstat ('test219'    ,s, J40  , F10  ) ; % reduce to scalar (1 thread)

% < 1 second: debug_on
set_malloc_debug (mdebug, 1) ;
logstat ('test296'    ,t, J4   , F1   ) ; % integer overflow in saxpy3 cumsum
logstat ('test289'    ,t, J0   , F0   ) ; % container tests
logstat ('test288'    ,t, J0   , F0   ) ; % load/unload tests
logstat ('test244'    ,t, J4   , F1   , [0 1]) ; % GxB_Matrix_reshape*
logstat ('test194'    ,t, J4   , F1   ) ; % GxB_Vector_diag
logstat ('test09'     ,t, J40  , F10  ) ; % duplicate I,J in GB_mex_subassign
logstat ('test108'    ,t, J40  , F10  ) ; % boolean monoids
logstat ('test137'    ,s, J400 , F110 ) ; % GrB_eWiseMult, FIRST and SECOND
logstat ('test124'    ,t, J4   , F1   ) ; % GrB_extract, case 6
logstat ('test133'    ,t, J40  , F10  ) ; % mask operations (GB_masker)
logstat ('test176'    ,t, J40  , F10  ) ; % GrB_assign, method 09, 11
logstat ('test197'    ,t, J40  , F10  ) ; % large sparse split
logstat ('test201'    ,t, J4   , F1   ) ; % iso reduce to vector, scalar
logstat ('test208'    ,t, J4   , F1   ) ; % iso apply, bind 1st and 2nd
logstat ('test214'    ,t, J40  , F10  , [0 1]) ; % C<M>=A'*B (tricount)
logstat ('test223'    ,t, J40  , F10  ) ; % matrix multiply, C<!M>=A*B
logstat ('test241'    ,t, J40  , F10  ) ; % GrB_mxm, trigger swap_rule
logstat ('test270'    ,t, J0   , F1   ) ; % unary op get/set
logstat ('test199'    ,t, J4   , F1   ) ; % dot2 with hypersparse
logstat ('test210'    ,t, J40  , F10  ) ; % iso assign25: C<M,struct>=A
logstat ('test165'    ,t, J4   , F1   ) ; % C=A*B', A diagonal, B bitmap
logstat ('test221'    ,t, J40  , F10  ) ; % C += A, C bitmap, A full
logstat ('test278'    ,t, J0   , F1   ) ; % descriptor get/set
logstat ('test162'    ,t, J40  , F10  ) ; % C<M>=A*B with very sparse M
logstat ('test275'    ,t, J0   , F1   ) ; % monoid get/set
logstat ('test220'    ,t, J4   , F1   ) ; % mask C<M>=Z, iso case, kron
logstat ('test83'     ,t, J40  , F10  ) ; % GrB_assign, C_replace and empty J
logstat ('test04'     ,t, J40  , F10  ) ; % simple mask and transpose test
logstat ('test132'    ,t, J4   , F1   ) ; % setElement
logstat ('test82'     ,t, J4   , F1   ) ; % GrB_extract, index range (hyper)
logstat ('test202'    ,t, J400 , F110 , [0 1 2]) ; % iso add and emult
logstat ('test222'    ,t, J4   , F1   ) ; % user selectop, iso matrices
logstat ('test204'    ,t, J4   , F1   ) ; % iso diag
logstat ('test258'    ,t, J40  , F00  , [0 1]) ; % reduce-to-vector for UDT
logstat ('test136'    ,s, J40  , F10  ) ; % subassignment special cases
logstat ('test128'    ,t, J40  , F10  ) ; % eWiseMult, eWiseAdd, eWiseUnion
logstat ('test144'    ,t, J4   , F1   ) ; % cumsum
logstat ('test81'     ,t, J4   , F1   ) ; % extract stride, range, backwards

%===============================================================================
% 1 to 10 seconds
%===============================================================================

% 1 to 10 seconds: debug_off
set_malloc_debug (mdebug, 0) ;
logstat ('testc2(0,0)',t, J0   , F1   , [0 1]) ; % user-defn complex
logstat ('test239'    ,t, J44  , F10  ) ; % GxB_eWiseUnion
logstat ('test245'    ,t, J40  , F11  ) ; % complex row/col scale
logstat ('test159'    ,t, J0   , F0   ) ; % A*B
logstat ('test259'    ,t, J40  , F00  ) ; % plus_plus_fp32 semiring
logstat ('testc4(0)'  ,t, J4   , F1   ) ; % extractElement, setElement, udt
logstat ('test157'    ,t, J4   , F1   ) ; % sparsity formats
logstat ('test182'    ,s, J40  , F10  ) ; % for internal wait
logstat ('test195'    ,t, J4   , F1   ) ; % saxpy3 slice_balanced
logstat ('test135'    ,t, J4   , F1   ) ; % reduce to scalar
logstat ('test215'    ,t, J4   , F1   ) ; % C<M>=A'*B (dot2, ANY_PAIR)
logstat ('test80'     ,t, J4   , F1   ) ; % GrB_mxm on all semirings
logstat ('test200'    ,t, J4   , F1   ) ; % iso full matrix multiply
logstat ('test283'    ,t, J4   , F1   , [0 1]) ; % index binary op
logstat ('test254'    ,t, J44  , F10  ) ; % mask types
logstat ('test54'     ,t, J40  , F10  ) ; % assign, extract with begin:inc:end
logstat ('testcc(1)'  ,t, J40  , F10  ) ; % transpose, builtin complex
logstat ('testc2(1,1)',t, J44  , F10  ) ; % complex tests (quick case, builtin)
logstat ('test141'    ,t, J0   , F1   ) ; % eWiseAdd with dense matrices
logstat ('test179'    ,t, J44  , F10  ) ; % bitmap select

% 1 to 10 seconds, no Werk, debug_off
hack (2) = 1 ; GB_mex_hack (hack) ;     % disable the Werk stack
logstat ('test188b'   ,t, J0   , F1   ) ; % concat
logstat ('test185'    ,s, J4   , F1   ) ; % dot4, saxpy for all sparsity
logstat ('test256'    ,t, J40  , F00  , [0 1]) ; % JIT error handling
logstat ('test238b'   ,t, J4   , F0   ) ; % GrB_mxm (dot4 and dot2)
logstat ('test238'    ,t, J4   , F1   ) ; % GrB_mxm (dot4 and dot2)
% Note that test186 can sometimes non-deterministically miss this block of code
% in GB_AxB_saxbit_A_sparse_B_bitmap_template.c, about line 352, so it is run
% 3 times:
%                      ...
%                      else if (cb == keep)
%                      {    <----- here
%                           // C(i,j) is already present
%                           #if !GB_IS_ANY_MONOID
%                           GB_MULT_A_ik_B_kj ;             // t = A(i,k)*B(k,j)
%                           GB_Z_ATOMIC_UPDATE_HX (i, t) ;    // C(i,j) += t
%                           #endif
%                       }
%                       GB_ATOMIC_WRITE
%                       Cb [pC] = cb ;                  // unlock the entry
%                      ...
logstat ('test186'    ,t, J4   , F1   ) ; % saxpy, all formats (slice_balanced)
logstat ('test186'    ,t, J4   , F1   ) ; % saxpy, all formats (slice_balanced)
logstat ('test186'    ,t, J4   , F1   ) ; % saxpy, all formats (slice_balanced)
hack (2) = 0 ; GB_mex_hack (hack) ;     % re-enable the Werk stack

% 1 to 10 seconds: debug_on
set_malloc_debug (mdebug, 1) ;
logstat ('testca(1)'  ,t, J40  , F10  ) ; % complex mxm, mxv, and vxm
logstat ('test148'    ,t, J40  , F10  ) ; % ewise with alias
logstat ('test231'    ,t, J4   , F1   ) ; % GrB_select with idxunp
logstat ('test129'    ,t, J4   , F1   ) ; % GxB_select (tril, nonz, hyper)
logstat ('test69'     ,t, J40  , F10  ) ; % assign and subassign with alias
logstat ('test29'     ,t, J00  , F10  , [0 1]) ; % reduce tests
logstat ('test282'    ,t, J4   , F1   ) ; % argmax, index binary op
logstat ('test249'    ,t, J40  , F10  ) ; % GxB_Context object
logstat ('test196'    ,t, J4   , F1   ) ; % hypersparse concat
logstat ('test250'    ,t, J44  , F10  ) ; % JIT tests, set/get, other tests
logstat ('test145'    ,t, J42  , F11  ) ; % dot4 for C += A'*B
logstat ('test229'    ,t, J4   , F1   ) ; % setElement
logstat ('test209'    ,t, J4   , F1   , [0 1]) ; % iso/non-iso build
logstat ('test224'    ,t, J4   , F1   ) ; % unpack/pack

% 1 to 10 seconds, no Werk, debug_on
hack (2) = 1 ; GB_mex_hack (hack) ;     % disable the Werk stack
logstat ('test191'    ,t, J40  , F10  ) ; % split
logstat ('test150'    ,t, J0   , F0   ) ; % mxm zombies, typecasting
logstat ('test240'    ,t, J40  , F10  ) ; % dot4, saxpy4, and saxpy5
logstat ('test237'    ,t, J40  , F10  ) ; % GrB_mxm (saxpy4)
logstat ('test237'    ,s, J40  , F10  ) ; % GrB_mxm (saxpy4) (1 task)
logstat ('test184'    ,t, J4   , F1   ) ; % mxm, transp, build
logstat ('test236'    ,t, J4   , F1   ) ; % GxB_*_sort
hack (2) = 0 ; GB_mex_hack (hack) ;     % re-enable the Werk stack

%===============================================================================
% 10 to 100 seconds
%===============================================================================

% 10 to 100 seconds: debug_off
set_malloc_debug (mdebug, 0) ;
logstat ('test84'     ,s, J40  , F10  , [0 2]) ; % GrB_assign (row/col)
logstat ('test84'     ,t, J40  , F10  , [0 2]) ; % GrB_assign (row/col)
logstat ('test173'    ,t, J40  , F10  ) ; % GrB_assign C<A>=A
logstat ('test230'    ,t, J40  , F10  ) ; % apply with idxunops
logstat ('test18'     ,t, J40  , F10  ) ; % GrB_eWiseAdd and eWiseMult
logstat ('testc7(0)'  ,t, J40  , F10  ) ; % assign, builtin complex
logstat ('test193'    ,t, J4   , F1   ) ; % GxB_Matrix_diag
logstat ('test127'    ,t, J0   , F1   ) ; % eWiseAdd, eWiseMult
logstat ('test23'     ,t, J0   , F1   ) ; % quick test of GB_*_build
logstat ('test243'    ,t, J4   , F1   ) ; % GxB_Vector_Iterator
logstat ('test53'     ,t, J40  , F10  ) ; % GB_mex_Matrix_extract
logstat ('test242'    ,t, J4   , F1   ) ; % GxB_Iterator for matrices
logstat ('test17'     ,t, J4   , F1   ) ; % quick test of GrB_*_extractElement
logstat ('test246'    ,t, J4   , F1   ) ; % GrB_mxm: fine Hash, parallelism
logstat ('test251b'   ,t, J4   , F0   ) ; % dot4, dot2, with plus_pair
logstat ('test251'    ,t, J4   , F1   ) ; % dot4, dot2, with plus_pair
logstat ('test152'    ,t, J44  , F10  ) ; % binops C=A+B, all dense
logstat ('test160'    ,s, J0   , F1   ) ; % A*B, single threaded
logstat ('test232'    ,t, J40  , F10  ) ; % assign with GrB_Scalar
logstat ('test142b'   ,t, J40  , F00  ) ; % GrB_assign with accum
logstat ('test142'    ,t, J4   , F1   ) ; % GrB_assign with accum
logstat ('test227'    ,t, J4   , F1   ) ; % kron
logstat ('test292'    ,t, J4   , F1   ) ; % build_Vector with large vector

% 10 to 100 seconds, no Werk, debug_off
hack (2) = 1 ; GB_mex_hack (hack) ;     % disable the Werk stack
logstat ('test192'    ,t, J4   , F1   ) ; % C<C,struct>=scalar
logstat ('test181'    ,s, J40  , F10  ) ; % transpose with 0's in mask
hack (2) = 0 ; GB_mex_hack (hack) ;     % re-enable the Werk stack

% 10 to 100 seconds: debug_on
set_malloc_debug (mdebug, 1) ;
logstat ('test130'    ,t, J40  , F10  ) ; % GrB_apply, hypersparse cases
logstat ('test206'    ,t, J44  , F10  ) ; % iso select and iso resize
logstat ('test02'     ,t, J4   , F1   ) ; % matrix copy and dup tests
logstat ('test11'     ,t, J4   , F1   ) ; % GrB_extractTuples
logstat ('test187'    ,t, J40  , F10  ) ; % dup/assign for all formats
logstat ('test169'    ,t, J0   , F1   ) ; % C<M>=A+B with many formats
logstat ('test76'     ,s, J4   , F1   ) ; % GxB_resize (single threaded)
logstat ('test01'     ,t, J40  , F10  ) ; % error handling
logstat ('test228'    ,t, J4   , F1   ) ; % serialize/deserialize
logstat ('test104'    ,t, J4   , F1   ) ; % export/import
logstat ('test284'    ,t, J40  , F11  ) ; % semirings w/ index binary ops

% 10 to 100 seconds, no Werk, debug_on
hack (2) = 1 ; GB_mex_hack (hack) ;     % disable the Werk stack
logstat ('test180'    ,s, J40  , F10  ) ; % assign and subassign (1 thread)
logstat ('test188'    ,t, J4   , F1   ) ; % concat
logstat ('test151b'   ,t, J40  , F10  ) ; % bshift operator
logstat ('test14b'    ,t, J4   , F0   ) ; % GrB_reduce
logstat ('test14'     ,t, J4   , F1   ) ; % GrB_reduce
hack (2) = 0 ; GB_mex_hack (hack) ;     % re-enable the Werk stack

%===============================================================================
% > 100 seconds
%===============================================================================

% > 100 seconds, debug_off
set_malloc_debug (mdebug, 0) ;
logstat ('test125'    ,t, J4   , F1   ) ; % GrB_mxm: row and column scaling
logstat ('test10'     ,t, J4   , F1   ) ; % GrB_apply
logstat ('test75b'    ,t, J4   , F1   ) ; % GrB_mxm A'*B
logstat ('test74'     ,t, J0   , F1   ) ; % GrB_mxm on all semirings
logstat ('test234'    ,t, J4   , F1   ) ; % GxB_eWiseUnion

% > 100 seconds, no Werk, debug_on
set_malloc_debug (mdebug, 1) ;
hack (2) = 1 ; GB_mex_hack (hack) ;     % disable the Werk stack
logstat ('test154b'   ,t, J0   , F1   ) ; % apply binop and scalar binding
logstat ('test154'    ,t, J4   , F1   ) ; % apply binop and scalar binding
hack (2) = 0 ; GB_mex_hack (hack) ;     % re-enable the Werk stack

% > 100 seconds, debug_on
logstat ('test21b'    ,t, J0   , F0   ) ; % GB_mex_assign
logstat ('test19b'    ,s, J40  , F10  ) ; % GrB_assign, many pending ops

% > 100 seconds, debug_off
set_malloc_debug (mdebug, 0) ;
logstat ('test19'     ,t, J40  , F10  ) ; % GxB_subassign, many pending ops
logstat ('test280(0)' ,t, J4   , F1   ) ; % subassign method 26

%===============================================================================
% finalize
%===============================================================================

% restore the original malloc debug state
set_malloc_debug (mdebug, debug_save) ;
t = toc (testall_time) ;
fprintf ('\ntestall: all tests passed, total time %0.4g minutes\n', t / 60) ;

