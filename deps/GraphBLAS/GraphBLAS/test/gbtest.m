function s = gbtest
%GBTEST test GraphBLAS MATLAB/Octave interface
% First compile the GraphBLAS library by following the instructions in the
% README.m. file in the top-level GraphBLAS folder.  Then run this test while
% in the GraphBLAS/GraphBLAS/test folder that contains this gbtest.m file.
%
% This test has been ported to Octave 10.2 and 11.1, as of GraphBLAS v10.4.  A
% few features differ between Octave and MATLAB, so those tests are skipped for
% Octave.  Octave passes all of the essential tests below.
%
% Example (if GraphBLAS is in your /home/me folder):
%
%   cd /home/me/GraphBLAS/GraphBLAS
%   graphblas_install
%   cd test
%   gbtest
%
% See also GrB, GhB.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

% gbtest3 requires ../demo/dnn_builtin.m and ../demo/dnn_builtin2gb.m.
demo_folder = fullfile (fileparts (mfilename ('fullpath')), '../demo') ;
addpath (demo_folder) ;
rng ('default') ;

% GrB.nmalloc is always zero unless gbmake compiles the GrB/GhB interface
% with -DMALLOC_TRACKING enabled; in that case, it records the # of malloc'd
% spaces that have yet to be freed.  See gbmake.m for details.  The gbcov.m
% script always compiles the interface with tracking enabled, for test coverage
% results.

have_octave = gb_octave ;

gbtest0         % test GrB.clear
gbtest0 (1)
gbtest0 (2)
assert (GrB.nmalloc == 0) ;

gbtest1         % test GrB
gbtest1 (1)
gbtest1 (2)
assert (GrB.nmalloc == 0) ;

gbtest2         % list all binary operators
gbtest2 (1)
gbtest2 (2)
assert (GrB.nmalloc == 0) ;

gbtest3         % test dnn
gbtest3 (1)
gbtest3 (2)
assert (GrB.nmalloc == 0) ;

gbtest4         % list all possible semirings
gbtest4 (1)
gbtest4 (2)
assert (GrB.nmalloc == 0) ;

gbtest5         % test GrB.descriptorinfo
gbtest5 (1)
gbtest5 (2)
assert (GrB.nmalloc == 0) ;

gbtest6         % test GrB.mxm
gbtest6 (1)
gbtest6 (2)
assert (GrB.nmalloc == 0) ;

gbtest7         % test GrB.build
gbtest7 (1)
gbtest7 (2)
assert (GrB.nmalloc == 0) ;

gbtest8         % test GrB.select
gbtest8 (1)
gbtest8 (2)
assert (GrB.nmalloc == 0) ;

gbtest9         % test eye and speye
gbtest9 (1)
gbtest9 (2)
assert (GrB.nmalloc == 0) ;

gbtest10        % test GrB.assign
gbtest10 (1)
gbtest10 (2)
assert (GrB.nmalloc == 0) ;

gbtest11        % test GrB, sparse
gbtest11 (1)
gbtest11 (2)
assert (GrB.nmalloc == 0) ;

gbtest12        % test GrB.eadd, GrB.emult, GrB.eunion
gbtest12 (1)
gbtest12 (2)
assert (GrB.nmalloc == 0) ;

gbtest13        % test find and GrB.extracttuples
gbtest13 (1)
gbtest13 (2)
assert (GrB.nmalloc == 0) ;

gbtest14        % test kron and GrB.kronecker
gbtest14 (1)
gbtest14 (2)
assert (GrB.nmalloc == 0) ;

gbtest15        % list all unary operators
gbtest15 (1)
gbtest15 (2)
assert (GrB.nmalloc == 0) ;

gbtest16        % test GrB.extract
gbtest16 (1)
gbtest16 (2)
assert (GrB.nmalloc == 0) ;

gbtest17        % test GrB.trans
gbtest17 (1)
gbtest17 (2)
assert (GrB.nmalloc == 0) ;

gbtest18        % test comparators (and, or, >, ...)
gbtest18 (1)
gbtest18 (2)
assert (GrB.nmalloc == 0) ;

gbtest19        % test mpower
gbtest19 (1)
gbtest19 (2)
assert (GrB.nmalloc == 0) ;

gbtest20        % test bandwidth, isdiag, ceil, floor, round, fix
gbtest20 (1)
gbtest20 (2)
assert (GrB.nmalloc == 0) ;

gbtest21        % test isfinite, isinf, isnan
gbtest21 (1)
gbtest21 (2)
assert (GrB.nmalloc == 0) ;

gbtest22        % test reduce to scalar
gbtest22 (1)
gbtest22 (2)
assert (GrB.nmalloc == 0) ;

gbtest23        % test min and max
gbtest23 (1)
gbtest23 (2)
assert (GrB.nmalloc == 0) ;

gbtest24        % test any, all
gbtest24 (1)
gbtest24 (2)
assert (GrB.nmalloc == 0) ;

gbtest25        % test diag, tril, triu
gbtest25 (1)
gbtest25 (2)
assert (GrB.nmalloc == 0) ;

gbtest26        % test typecasting
gbtest26 (1)
gbtest26 (2)
assert (GrB.nmalloc == 0) ;

gbtest27        % test conversion to full
gbtest27 (1)
gbtest27 (2)
assert (GrB.nmalloc == 0) ;

gbtest28        % test GrB.build
gbtest28 (1)
gbtest28 (2)
assert (GrB.nmalloc == 0) ;

gbtest29        % test subsref and subsasgn with logical indexing
gbtest29 (1)
gbtest29 (2)
assert (GrB.nmalloc == 0) ;

gbtest30        % test colon notation
gbtest30 (1)
gbtest30 (2)
assert (GrB.nmalloc == 0) ;

gbtest31        % test GrB and casting
gbtest31 (1)
gbtest31 (2)
assert (GrB.nmalloc == 0) ;

gbtest32        % test nonzeros
gbtest32 (1)
gbtest32 (2)
assert (GrB.nmalloc == 0) ;

gbtest33        % test spones, numel, nzmax, size, length, isempty, ...
gbtest33 (1)
gbtest33 (2)
assert (GrB.nmalloc == 0) ;

gbtest34        % test repmat
gbtest34 (1)
gbtest34 (2)
assert (GrB.nmalloc == 0) ;

gbtest35        % test reshape
gbtest35 (1)
gbtest35 (2)
assert (GrB.nmalloc == 0) ;

gbtest36        % test abs, sign
gbtest36 (1)
gbtest36 (2)
assert (GrB.nmalloc == 0) ;

gbtest37        % test istril, istriu, isbanded, isdiag, ishermitian, ...
gbtest37 (1)
gbtest37 (2)
assert (GrB.nmalloc == 0) ;

gbtest38        % test sqrt, eps, ceil, floor, round, fix, real, conj, ...
gbtest38 (1)
gbtest38 (2)
assert (GrB.nmalloc == 0) ;

gbtest39        % test amd, colamd, symamd, symrcm, dmperm, etree
gbtest39 (1)
gbtest39 (2)
assert (GrB.nmalloc == 0) ;

gbtest40        % test sum, prod, max, min, any, all, norm
gbtest40 (1)
gbtest40 (2)
assert (GrB.nmalloc == 0) ;

gbtest41        % test ones, zeros, false
gbtest41 (1)
gbtest41 (2)
assert (GrB.nmalloc == 0) ;

gbtest42        % test for nan
gbtest42 (1)
gbtest42 (2)
assert (GrB.nmalloc == 0) ;

gbtest43        % test error handling
gbtest43 (1)
gbtest43 (2)
assert (GrB.nmalloc == 0) ;

gbtest44        % test subsasgn, mtimes, plus, false, ...
gbtest44 (1)
gbtest44 (2)
assert (GrB.nmalloc == 0) ;

gbtest45        % test GrB.vreduce
gbtest45 (1)
gbtest45 (2)
assert (GrB.nmalloc == 0) ;

gbtest46        % test GrB.subassign and GrB.assign
gbtest46 (1)
gbtest46 (2)
assert (GrB.nmalloc == 0) ;

gbtest47        % test GrB.entries, GrB.nonz, numel
gbtest47 (1)
gbtest47 (2)
assert (GrB.nmalloc == 0) ;

gbtest48        % test GrB.apply
gbtest48 (1)
gbtest48 (2)
assert (GrB.nmalloc == 0) ;

gbtest49        % test GrB.prune
gbtest49 (1)
gbtest49 (2)
assert (GrB.nmalloc == 0) ;

gbtest50        % test GrB.ktruss and GrB.tricount
gbtest50 (1)
gbtest50 (2)
assert (GrB.nmalloc == 0) ;

gbtest51        % test GrB.tricount
gbtest51 (1)
gbtest51 (2)
assert (GrB.nmalloc == 0) ;

gbtest52        % test GrB.format
gbtest52 (1)
gbtest52 (2)
assert (GrB.nmalloc == 0) ;

gbtest53        % test GrB.monoidinfo
gbtest53 (1)
gbtest53 (2)
assert (GrB.nmalloc == 0) ;

gbtest54        % test GrB.compact
gbtest54 (1)
gbtest54 (2)
assert (GrB.nmalloc == 0) ;

gbtest55        % test disp
gbtest55 (1)
gbtest55 (2)
assert (GrB.nmalloc == 0) ;

gbtest56        % test GrB.empty
gbtest56 (1)
gbtest56 (2)
assert (GrB.nmalloc == 0) ;

gbtest57        % test fprintf and sprintf
gbtest57 (1)
gbtest57 (2)
assert (GrB.nmalloc == 0) ;

gbtest58        % test uplus
gbtest58 (1)
gbtest58 (2)
assert (GrB.nmalloc == 0) ;

gbtest59        % test end
gbtest59 (1)
gbtest59 (2)
assert (GrB.nmalloc == 0) ;

gbtest60        % test issigned
gbtest60 (1)
gbtest60 (2)
assert (GrB.nmalloc == 0) ;

gbtest62        % test ldivide, rdivide, mldivide, mrdivide
gbtest62 (1)
gbtest62 (2)
assert (GrB.nmalloc == 0) ;

gbtest65        % test GrB.mis
gbtest65 (1)
gbtest65 (2)
assert (GrB.nmalloc == 0) ;

if (~have_octave)
    % the Graph and DiGraph methods do not appear in octave
    gbtest61        % test GrB.laplacian
    gbtest61 (1)
    gbtest61 (2)
    assert (GrB.nmalloc == 0) ;

    gbtest63        % test GrB.incidence
    gbtest63 (1)
    gbtest63 (2)
    assert (GrB.nmalloc == 0) ;

    gbtest64        % test GrB.pagerank
    gbtest64 (1)
    gbtest64 (2)
    assert (GrB.nmalloc == 0) ;

    gbtest66        % test graph
    gbtest66 (1)
    gbtest66 (2)
    assert (GrB.nmalloc == 0) ;

    gbtest67        % test digraph
    gbtest67 (1)
    gbtest67 (2)
    assert (GrB.nmalloc == 0) ;
end

gbtest68        % test isequal
gbtest68 (1)
gbtest68 (2)
assert (GrB.nmalloc == 0) ;

gbtest69        % test flip
gbtest69 (1)
gbtest69 (2)
assert (GrB.nmalloc == 0) ;

gbtest70        % test GrB.random
gbtest70 (1)
assert (GrB.nmalloc == 0) ;

gbtest71        % test GrB.selectopinfo
gbtest71 (1)
gbtest71 (2)
assert (GrB.nmalloc == 0) ;

gbtest72        % test any-pair semiring
gbtest72 (1)
gbtest72 (2)
assert (GrB.nmalloc == 0) ;

gbtest73        % test GrB.normdiff
gbtest73 (1)
gbtest73 (2)
assert (GrB.nmalloc == 0) ;

if (~have_octave)
    % octave returns double, MATLAB returns integer.
    % This would be easy to fix but the tests are skipped for octave.
    gbtest74        % test bitwise operators
    gbtest74 (1)
    gbtest74 (2)
    assert (GrB.nmalloc == 0) ;

    gbtest75        % test bitshift
    gbtest75 (1)
    gbtest75 (2)
    assert (GrB.nmalloc == 0) ;
end

gbtest76        % test trig functions
gbtest76 (1)
gbtest76 (2)
assert (GrB.nmalloc == 0) ;

gbtest77        % test error handling
gbtest77 (1)
gbtest77 (2)
assert (GrB.nmalloc == 0) ;

if (~have_octave)
    % octave: bit index must be in proper range.
    % MATLAB: bit indices outside the size of the integer are ignored.
    % This would be easy to fix but the tests are skipped for octave.
    gbtest78        % test integer operations
    gbtest78 (1)
    gbtest78 (2)
    assert (GrB.nmalloc == 0) ;
end

gbtest79        % test power
gbtest79 (1)
gbtest79 (2)
assert (GrB.nmalloc == 0) ;

gbtest80        % test complex division and power
gbtest80 (1)
gbtest80 (2)
assert (GrB.nmalloc == 0) ;

gbtest81        % test complex operators
gbtest81 (1)
gbtest81 (2)
assert (GrB.nmalloc == 0) ;

gbtest82        % test complex A*B, A'*B, A*B', A'*B', A+B
gbtest82 (1)
gbtest82 (2)
assert (GrB.nmalloc == 0) ;

gbtest83        % test GrB.apply
gbtest83 (1)
gbtest83 (2)
gbtest83 (1, 0)
assert (GrB.nmalloc == 0) ;

gbtest84        % test GrB.assign
gbtest84 (1)
gbtest84 (2)
assert (GrB.nmalloc == 0) ;

gbtest85        % test GrB.subassign
gbtest85 (1)
gbtest85 (2)
assert (GrB.nmalloc == 0) ;

gbtest86        % test GrB.mxm
gbtest86 (1)
gbtest86 (2)
assert (GrB.nmalloc == 0) ;

gbtest87        % test GrB.eadd
gbtest87 (1)
gbtest87 (2)
gbtest87 (1, 0)
assert (GrB.nmalloc == 0) ;

gbtest88        % test GrB.emult
gbtest88 (1)
gbtest88 (2)
gbtest88 (1, 0)
assert (GrB.nmalloc == 0) ;

gbtest89        % test GrB.extract
gbtest89 (1)
gbtest89 (2)
gbtest89 (1, 0)
assert (GrB.nmalloc == 0) ;

gbtest90        % test GrB.reduce
gbtest90 (1)
gbtest90 (2)
gbtest90 (1, 0)
assert (GrB.nmalloc == 0) ;

gbtest91        % test GrB.trans
gbtest91 (1)
gbtest91 (2)
gbtest91 (1, 0)
assert (GrB.nmalloc == 0) ;

gbtest92        % test GrB.kronecker
gbtest92 (1)
gbtest92 (2)
gbtest92 (1, 0)
assert (GrB.nmalloc == 0) ;

gbtest93        % test GrB.select
gbtest93 (1)
gbtest93 (2)
gbtest93 (1, 0)
assert (GrB.nmalloc == 0) ;

gbtest94        % test GrB.vreduce
gbtest94 (1)
gbtest94 (2)
assert (GrB.nmalloc == 0) ;

gbtest95        % test indexing
gbtest95 (1)
gbtest95 (2)
assert (GrB.nmalloc == 0) ;

gbtest97        % test GrB.apply2
gbtest97 (1)
gbtest97 (2)
gbtest97 (1, 0)
assert (GrB.nmalloc == 0) ;

gbtest98        % test row/col degree for hypersparse matrices
gbtest98 (1)
gbtest98 (2)
assert (GrB.nmalloc == 0) ;

gbtest99        % test performance of C=A'*B and C=A'
gbtest99 (1)
gbtest99 (2)
assert (GrB.nmalloc == 0) ;

gbtest100       % test GrB.ver and GrB.version
gbtest100 (1)
gbtest100 (2)
assert (GrB.nmalloc == 0) ;

if (~have_octave)
    % this test fails in Octave 10.2 but works in Octave 11.1.
    gbtest101       % test loading of v3 GraphBLAS objects
    gbtest101 (1)
    gbtest101 (2)
    assert (GrB.nmalloc == 0) ;
end

gbtest102       % test horzcat, vertcat, cat, cell2mat
gbtest102 (1)
gbtest102 (2)
assert (GrB.nmalloc == 0) ;

gbtest103       % test iso matrices
gbtest103 (1)
gbtest103 (2)
assert (GrB.nmalloc == 0) ;

gbtest104       % test formats
gbtest104 (1)
gbtest104 (2)
assert (GrB.nmalloc == 0) ;

gbtest105       % test logical assignment with iso matrices
gbtest105 (1)
gbtest105 (2)
assert (GrB.nmalloc == 0) ;

gbtest106       % test build
gbtest106 (1)
gbtest106 (2)
assert (GrB.nmalloc == 0) ;

gbtest107       % test cell2mat error handling
gbtest107 (1)
gbtest107 (2)
assert (GrB.nmalloc == 0) ;

gbtest108       % test mat2cell
gbtest108 (1)
gbtest108 (2)
assert (GrB.nmalloc == 0) ;

gbtest109       % test num2cell
gbtest109 (1)
gbtest109 (2)
assert (GrB.nmalloc == 0) ;

gbtest110       % test argmax
gbtest110 (1)
gbtest110 (2)
assert (GrB.nmalloc == 0) ;

gbtest111       % test argmin
gbtest111 (1)
gbtest111 (2)
assert (GrB.nmalloc == 0) ;

if (~have_octave)
    % octave cannot save/load an object
    gbtest112       % test load and save
    gbtest112 (1)
    gbtest112 (2)
    assert (GrB.nmalloc == 0) ;
end

gbtest113       % test ones and eq
gbtest113 (1)
gbtest113 (2)
assert (GrB.nmalloc == 0) ;

gbtest114       % test kron with iso matrices
gbtest114 (1)
gbtest114 (2)
assert (GrB.nmalloc == 0) ;

gbtest115       % test serialize/deserialize
gbtest115 (1)
gbtest115 (2)
assert (GrB.nmalloc == 0) ;

gbtest116       % test GrB.binopinfo for index_unary operators
gbtest116 (1)
gbtest116 (2)
assert (GrB.nmalloc == 0) ;

gbtest117       % test idxunop in GrB.apply2
gbtest117 (1)
gbtest117 (2)
assert (GrB.nmalloc == 0) ;

gbtest118       % test GrB.argsort
gbtest118 (1)
gbtest118 (2)
assert (GrB.nmalloc == 0) ;

gbtest119       % test GrB.eunion
gbtest119 (1)
gbtest119 (2)
gbtest119 (1, 0)
assert (GrB.nmalloc == 0) ;

gbtest120       % test subsref
gbtest120 (1)
gbtest120 (2)
assert (GrB.nmalloc == 0) ;

gbtest121       % test times with scalars
gbtest121 (1)
gbtest121 (2)
assert (GrB.nmalloc == 0) ;

gbtest122       % test reshape
gbtest122 (1)
gbtest122 (2)
assert (GrB.nmalloc == 0) ;

gbtest123       % test reshape
gbtest123 (1)
gbtest123 (2)
assert (GrB.nmalloc == 0) ;

gbtest124       % test binops
gbtest124 (1)
gbtest124 (2)
assert (GrB.nmalloc == 0) ;

gbtest125       % test monoids
gbtest125 (1)
gbtest125 (2)
assert (GrB.nmalloc == 0) ;

gbtest126       % test selectops
gbtest126 (1)
gbtest126 (2)
assert (GrB.nmalloc == 0) ;

gbtest127       % test semirings
gbtest127 (1)
gbtest127 (2)
assert (GrB.nmalloc == 0) ;

gbtest128       % test unops
gbtest128 (1)
gbtest128 (2)
assert (GrB.nmalloc == 0) ;

gbtest129       % test jit
gbtest129 (1)
gbtest129 (2)
assert (GrB.nmalloc == 0) ;

gbtest130       % test argmin and argmax
gbtest130 (1)
gbtest130 (2)
assert (GrB.nmalloc == 0) ;

gbtest131       % misc error handling
gbtest131 (1)
gbtest131 (2)
assert (GrB.nmalloc == 0) ;

if (~have_octave)
    % octave cannot save/load an object
    gbtest132       % test load/save from prior versions of GraphBLAS
    gbtest132 (1)
    gbtest132 (2)
    assert (GrB.nmalloc == 0) ;
end

gbtest133       % simple inplace tests of GhB.apply
assert (GrB.nmalloc == 0) ;

gbtest134       % test inplace usage for GhB.apply
assert (GrB.nmalloc == 0) ;

gbtest135       % test inplace usage for GhB.apply2
assert (GrB.nmalloc == 0) ;

gbtest136       % test inplace usage for GhB.assign
assert (GrB.nmalloc == 0) ;

gbtest137       % test inplace usage for GhB.subassign
assert (GrB.nmalloc == 0) ;

gbtest138       % test inplace usage for GhB.eunion
assert (GrB.nmalloc == 0) ;

gbtest139       % test inplace usage for GhB.emult
assert (GrB.nmalloc == 0) ;

gbtest140       % test inplace usage for GhB.eadd
assert (GrB.nmalloc == 0) ;

gbtest141       % test inplace usage for GhB.kronecker
assert (GrB.nmalloc == 0) ;

gbtest142       % test inplace usage for GhB.mxm
assert (GrB.nmalloc == 0) ;

gbtest143       % test inplace usage for GhB.reduce
assert (GrB.nmalloc == 0) ;

gbtest144       % test inplace usage for GhB.vreduce
assert (GrB.nmalloc == 0) ;

gbtest145       % test inplace usage for GhB.trans
assert (GrB.nmalloc == 0) ;

gbtest146       % test inplace usage for GhB.select
assert (GrB.nmalloc == 0) ;

gbtest147       % test inplace usage for GhB.extract
assert (GrB.nmalloc == 0) ;

gbtest148       % test log, log2, log10, sqrt: complex to real
gbtest148 (1)
gbtest148 (2)
assert (GrB.nmalloc == 0) ;

gbtest149       % test [GrB,GhB].expand
gbtest149 (1)
gbtest149 (2)
assert (GrB.nmalloc == 0) ;

gbtest150       % test [GrB,GhB].wait
assert (GrB.nmalloc == 0) ;

gbtest151       % test error handling
assert (GrB.nmalloc == 0) ;

gbtest152       % test nvals
gbtest152 (1)
gbtest152 (2)
assert (GrB.nmalloc == 0) ;

gbtest153       % test GhB.apply2 (not inplace but with pending work)
assert (GrB.nmalloc == 0) ;

gbtest96        % test GrB.optype
gbtest96 (1)
gbtest96 (2)
assert (GrB.nmalloc == 0) ;

gbtest154       % test GrB.bytes
gbtest155       % test GhB.get and GhB.set

if (~have_octave)
    % the Graph and DiGraph methods do not appear in octave
    gbtest156       % test GhB.bfs, error handling
    gbtest157       % test GhB.bfs
    gbtest00        % test GrB.bfs and plot (graph (G))
    gbtest00 (1)
    gbtest00 (2)
    assert (GrB.nmalloc == 0) ;
end

% restore default # of threads
demo_nproc ;
assert (GrB.nmalloc == 0) ;

GrB.clear
assert (GrB.nmalloc == 0) ;

fprintf ('\ngbtest: all tests passed\n') ;

if (nargout > 0)
    s = true ;
end

