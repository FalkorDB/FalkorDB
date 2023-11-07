function gbtest
%GBTEST test GraphBLAS interface
% First compile the GraphBLAS library by typing 'make' in the top-level
% GraphBLAS folder, in your system shell.  That statement will use cmake to
% compile GraphBLAS.  Use 'make JOBS=40' to compile in parallel (replace '40'
% with the number of cores in your system).  Next, do the following:
%
% This test has been ported to Octave 7, as of SuiteSparse:GraphBLAS v5.1.  A
% few features differ between Octave and MATLAB, so those tests are skipped for
% Octave.  Octave passes all of the essential tests below.
%
% Example:
%
%   cd GraphBLAS/GraphBLAS
%   addpath (pwd) ;
%   savepath ;          % if this fails, edit your startup.m file
%   cd @GrB/private
%   gbmake ;            % compile the interface to GraphBLAS
%   cd ../../test
%   gbtest              % run this test
%
% See also GrB.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

% gbtest3 requires ../demo/dnn_builtin.m and ../demo/dnn_builtin2gb.m.
demo_folder = fullfile (fileparts (mfilename ('fullpath')), '../demo') ;
addpath (demo_folder) ;
rng ('default') ;

have_octave = gb_octave ;

gbtest0   % test GrB.clear
gbtest1   % test GrB
gbtest2   % list all binary operators
gbtest3   % test dnn
gbtest4   % list all possible semirings
gbtest5   % test GrB.descriptorinfo
gbtest6   % test GrB.mxm
gbtest7   % test GrB.build
gbtest8   % test GrB.select
gbtest9   % test eye and speye
gbtest10  % test GrB.assign
gbtest11  % test GrB, sparse
gbtest12  % test GrB.eadd, GrB.emult, GrB.eunion
gbtest13  % test find and GrB.extracttuples
gbtest14  % test kron and GrB.kronecker
gbtest15  % list all unary operators
gbtest16  % test GrB.extract
gbtest17  % test GrB.trans
gbtest18  % test comparators (and, or, >, ...)
gbtest19  % test mpower
gbtest20  % test bandwidth, isdiag, ceil, floor, round, fix
gbtest21  % test isfinite, isinf, isnan
gbtest22  % test reduce to scalar
gbtest23  % test min and max
gbtest24  % test any, all
gbtest25  % test diag, tril, triu
gbtest26  % test typecasting
gbtest27  % test conversion to full
gbtest28  % test GrB.build
gbtest29  % test subsref and subsasgn with logical indexing
gbtest30  % test colon notation
gbtest31  % test GrB and casting
gbtest32  % test nonzeros
gbtest33  % test spones, numel, nzmax, size, length, isempty, issparse, ...
gbtest34  % test repmat
gbtest35  % test reshape
gbtest36  % test abs, sign
gbtest37  % test istril, istriu, isbanded, isdiag, ishermitian, ...
gbtest38  % test sqrt, eps, ceil, floor, round, fix, real, conj, ...
gbtest39  % test amd, colamd, symamd, symrcm, dmperm, etree
gbtest40  % test sum, prod, max, min, any, all, norm
gbtest41  % test ones, zeros, false
gbtest42  % test for nan
gbtest43  % test error handling
gbtest44  % test subsasgn, mtimes, plus, false, ...
gbtest45  % test GrB.vreduce
gbtest46  % test GrB.subassign and GrB.assign
gbtest47  % test GrB.entries, GrB.nonz, numel
gbtest48  % test GrB.apply
gbtest49  % test GrB.prune
gbtest50  % test GrB.ktruss and GrB.tricount
gbtest51  % test GrB.tricount
gbtest52  % test GrB.format
gbtest53  % test GrB.monoidinfo
gbtest54  % test GrB.compact
gbtest55  % test disp
gbtest56  % test GrB.empty
gbtest57  % test fprintf and sprintf
gbtest58  % test uplus
gbtest59  % test end
gbtest60  % test issigned
gbtest62  % test ldivide, rdivide, mldivide, mrdivide
gbtest65  % test GrB.mis

if (~have_octave)
    % the Graph and DiGraph methods do not appear in octave
    gbtest61  % test GrB.laplacian
    gbtest63  % test GrB.incidence
    gbtest64  % test GrB.pagerank
    gbtest66  % test graph
    gbtest67  % test digraph
end

gbtest68  % test isequal
gbtest69  % test flip
gbtest70  % test GrB.random
gbtest71  % test GrB.selectopinfo
gbtest72  % test any-pair semiring
gbtest73  % test GrB.normdiff

if (~have_octave)
    % octave returns double, MATLAB returns integer.
    % This would be easy to fix but the tests are skipped for octave.
    gbtest74  % test bitwise operators
    gbtest75  % test bitshift
end

gbtest76  % test trig functions
gbtest77  % test error handling

if (~have_octave)
    % octave: bit index must be in proper range.
    % MATLAB: bit indices outside the size of the integer are ignored.
    % This would be easy to fix but the tests are skipped for octave.
    gbtest78  % test integer operations
end

gbtest79  % test power
gbtest80  % test complex division and power
gbtest81  % test complex operators
gbtest82  % test complex A*B, A'*B, A*B', A'*B', A+B
gbtest83  % test GrB.apply
gbtest84  % test GrB.assign
gbtest85  % test GrB.subassign
gbtest86  % test GrB.mxm
gbtest87  % test GrB.eadd
gbtest88  % test GrB.emult
gbtest89  % test GrB.extract
gbtest90  % test GrB.reduce
gbtest91  % test GrB.trans
gbtest92  % test GrB.kronecker
gbtest93  % test GrB.select
gbtest94  % test GrB.vreduce
gbtest95  % test indexing
gbtest97  % test GrB.apply2
gbtest98  % test row/col degree for hypersparse matrices
gbtest99  % test performance of C=A'*B and C=A'
gbtest100 % test GrB.ver and GrB.version
if (~have_octave)
    % octave cannot load the mat file from MATLAB with a v3 @GrB object
    gbtest101 % test loading of v3 GraphBLAS objects
end
gbtest102 % test horzcat, vertcat, cat, cell2mat
gbtest103 % test iso matrices
gbtest104 % test formats
gbtest105 % test logical assignment with iso matrices
gbtest106 % test build
gbtest107 % test cell2mat error handling
gbtest108 % test mat2cell
gbtest109 % test num2cell
gbtest110 % test argmax
gbtest111 % test argmin
gbtest112 % test load and save
gbtest113 % test ones and eq
gbtest114 % test kron with iso matrices
gbtest115 % test serialize/deserialize
gbtest116 % test GrB.binopinfo for index_unary operators
gbtest117 % test idxunop in GrB.apply2
gbtest118 % test GrB.argsort
gbtest119 % test GrB.eunion
gbtest120 % test subsref
gbtest121 % test times with scalars
gbtest122 % test reshape
gbtest96  % test GrB.optype

if (~have_octave)
    % the Graph and DiGraph methods do not appear in octave
    gbtest00  % test GrB.bfs and plot (graph (G))
end

% restore default # of threads
demo_nproc ;
GrB.clear

fprintf ('\ngbtest: all tests passed\n') ;

