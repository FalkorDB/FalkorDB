% SuiteSparse/GraphBLAS/GraphBLAS/test: testing GraphBLAS interface
%
%  gbtest    - test GraphBLAS interface
%
%  gbtest0   - test GrB.clear
%  gbtest1   - test GrB
%  gbtest2   - list all binary operators
%  gbtest3   - test dnn
%  gbtest4   - list all semirings
%  gbtest5   - test GrB.descriptorinfo
%  gbtest6   - test GrB.mxm
%  gbtest7   - test GrB.build
%  gbtest8   - test GrB.select
%  gbtest9   - test eye and speye
%  gbtest10  - test GrB.assign
%  gbtest11  - test GrB, sparse
%  gbtest12  - test GrB.eadd, GrB.emult, GrB.eunion
%  gbtest13  - test find and GrB.extracttuples
%  gbtest14  - test kron and GrB.kronecker
%  gbtest15  - list all unary operators
%  gbtest16  - test GrB.extract
%  gbtest17  - test GrB.trans
%  gbtest18  - test comparators (and, or, >, ...)
%  gbtest19  - test mpower
%  gbtest20  - test bandwidth, isdiag, ceil, floor, round, fix
%  gbtest21  - test isfinite, isinf, isnan
%  gbtest22  - test reduce to scalar
%  gbtest23  - test min and max
%  gbtest24  - test any, all
%  gbtest25  - test diag, tril, triu
%  gbtest26  - test typecasting
%  gbtest27  - test conversion to full
%  gbtest28  - test GrB.build
%  gbtest29  - test subsref and subsasgn with logical indexing
%  gbtest30  - test colon notation
%  gbtest31  - test GrB and casting
%  gbtest32  - test nonzeros
%  gbtest33  - test spones, numel, nzmax, size, length, is*, ...
%  gbtest34  - test repmat
%  gbtest35  - test reshape
%  gbtest36  - test abs, sign
%  gbtest37  - test istril, istriu, isbanded, isdiag, ishermitian, ...
%  gbtest38  - test sqrt, eps, ceil, floor, round, fix, real, conj, ...
%  gbtest39  - test amd, colamd, symamd, symrcm, dmperm, etree
%  gbtest40  - test sum, prod, max, min, any, all, norm
%  gbtest41  - test ones, zeros, false
%  gbtest42  - test for nan
%  gbtest43  - test error handling
%  gbtest44  - test subsasgn, mtimes, plus, false, ...
%  gbtest45  - test GrB.vreduce
%  gbtest46  - test GrB.subassign and GrB.assign
%  gbtest47  - test GrB.entries, GrB.nonz, numel
%  gbtest48  - test GrB.apply
%  gbtest49  - test GrB.prune
%  gbtest50  - test GrB.ktruss and GrB.tricount
%  gbtest51  - test GrB.tricount and concatenate
%  gbtest52  - test GrB.format
%  gbtest53  - test GrB.monoidinfo
%  gbtest54  - test GrB.compact
%  gbtest55  - test disp
%  gbtest56  - test GrB.empty
%  gbtest57  - test fprintf and sprintf
%  gbtest58  - test uplus
%  gbtest59  - test end
%  gbtest60  - test GrB.issigned
%  gbtest61  - test GrB.laplacian
%  gbtest62  - test ldivide, rdivide, mldivide, mrdivide
%  gbtest63  - test GrB.incidence
%  gbtest64  - test GrB.pagerank
%  gbtest65  - test GrB.mis
%  gbtest66  - test graph
%  gbtest67  - test digraph
%  gbtest68  - test isequal
%  gbtest69  - test flip
%  gbtest70  - test GrB.random
%  gbtest71  - test GrB.selectopinfo
%  gbtest72  - test any-pair semiring
%  gbtest73  - test GrB.normdiff
%  gbtest74  - test bitwise operators
%  gbtest75  - test bitshift
%  gbtest76  - test trig and other functions
%  gbtest77  - test error handling
%  gbtest78  - test integer operators
%  gbtest79  - test real power
%  gbtest80  - test complex division and power
%  gbtest81  - test complex operators
%  gbtest82  - test complex A*B, A'*B, A*B', A'*B', A+B
%  gbtest83  - test GrB.apply
%  gbtest84  - test GrB.assign
%  gbtest85  - test GrB.subassign
%  gbtest86  - test GrB.mxm
%  gbtest87  - test GrB.eadd
%  gbtest88  - test GrB.emult
%  gbtest89  - test GrB.extract
%  gbtest90  - test GrB.reduce
%  gbtest91  - test GrB.trans
%  gbtest92  - test GrB.kronecker
%  gbtest93  - test GrB.select
%  gbtest94  - test GrB.vreduce
%  gbtest95  - test indexing
%  gbtest96  - test GrB.optype
%  gbtest97  - test GrB.apply2
%  gbtest98  - test row/col degree for hypersparse matrices
%  gbtest99  - test performance of C=A'*B and C=A'
%  gbtest00  - test GrB.bfs and plot (graph (G))
%  gbtest100 - test GrB.ver and GrB.version
%  gbtest101 - test loading of v3 GraphBLAS objects
%  gbtest102 - test horzcat, vertcat, cat, cell2mat, mat2cell, num2cell
%  gbtest103 - test iso matrices
%  gbtest104 - test formats
%  gbtest105 - test logical assignment with iso matrices
%  gbtest106 - test build
%  gbtest107 - test cell2mat error handling
%  gbtest108 - test mat2cell
%  gbtest109 - test num2cell
%  gbtest110 - test argmax
%  gbtest111 - test argmin
%  gbtest112 - test load and save
%  gbtest113 - test ones and eq
%  gbtest114 - test kron with iso matrices
%  gbtest115 - test serialize/deserialize
%  gbtest116 - list all idxunop operators for GrB.apply2
%  gbtest117 - test idxunop in GrB.apply2
%  gbtest118 - test GrB.argsort
%  gbtest119 - test GrB.eunion
%  gbtest120 - test subsref
%  gbtest121 - test times with scalars
%
% Utilities and other tests:
%
%  gbtest_binops  - list of all binary operators
%  gbtest_types   - return a cell array of strings, listing all types
%  gbtest_eq      - tests if A and B are equal, after dropping zeros.
%  gbtest_perf1   - test A*x performance
%  gbtest_perf2   - test A'*x performance
%  gbtest_cast    - cast a built-in matrix to another type.
%  gbtest_complex - return list of complex operators
%  gbtest_err     - compare two matrices
%  gb_contains    - same as contains (text, pattern)
%  gb_octave      - return true if Octave is in use, false for MATLAB

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

