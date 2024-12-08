function test238(tasks)
%TEST238 test GrB_mxm (dot4 and dot2)

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2024, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin < 1)
    tasks = [ ] ;
end

if (isempty (tasks))
    tasks = {
    { 1, 1, 1, 2}, ... % (  5,   5)
    { 1, 1, 1, 3}, ... % (  2,   7)
    { 1, 1, 1, 4}, ... % (  2,   9)
    { 1, 1, 1, 5}, ... % (  1,  10)
    { 1, 1, 4, 1}, ... % (  1,  11)
    { 1, 1, 4, 2}, ... % (  2,  13)
    { 1, 1, 4, 4}, ... % (  1,  14)
    { 1, 1, 8, 1}, ... % (  1,  15)
    { 1, 1, 8, 2}, ... % (  2,  17)
    { 1, 1, 8, 4}, ... % (  2,  19)
    { 1, 4, 1, 2}, ... % (  2,  21)
    { 1, 4, 1, 4}, ... % (  1,  22)
    { 1, 4, 4, 2}, ... % (  2,  24)
    { 1, 4, 4, 4}, ... % (  1,  25)
    { 1, 4, 8, 2}, ... % (  2,  27)
    { 1, 4, 8, 4}, ... % (  2,  29)
    { 1, 8, 1, 1}, ... % (  1,  30)
    { 1, 8, 1, 2}, ... % (  2,  32)
    { 1, 8, 1, 4}, ... % (  2,  34)
    { 1, 8, 4, 2}, ... % (  2,  36)
    { 1, 8, 4, 4}, ... % (  2,  38)
    { 1, 8, 8, 1}, ... % (  1,  39)
    { 1, 8, 8, 2}, ... % (  2,  41)
    { 1, 8, 8, 4}, ... % (  2,  43)
    { 4, 1, 1, 2}, ... % (  1,  44)
    { 4, 1, 1, 3}, ... % (  1,  45)
    { 4, 1, 1, 4}, ... % (  1,  46)
    } ;
end

track_coverage = false ;
if (track_coverage)
    global GraphBLAS_grbcov
    track_coverage = ~isempty (GraphBLAS_grbcov) ;
    clast = sum (GraphBLAS_grbcov > 0) ;
    cfirst = clast ;
    orig = GraphBLAS_grbcov ;
end

rng ('default') ;

desc.inp0 = 'tran' ;

n = 8 ;

% create the test matrices
AA {1} = GB_spec_random (n, n, inf) ;
AA {2} = GB_spec_random (n, n, 0.3) ;
for k = [1 2 4] % [1 2 4 8 32]
    FF {k} = GB_spec_random (n, k, inf) ;
    BB {k,1} = GB_spec_random (n, k, inf) ;
    BB {k,2} = GB_spec_random (n, k, 0.3) ;
end

iso = 0 ;

for kk = 1:length(tasks)
    task = tasks {kk} ;
    k = task {1} ;
    A_sparsity = task {2} ;
    B_sparsity = task {3} ;
    trial = task {4} ;
% end

% for k = [1 2 4] % [1 2 4 8 32]
    C0 = sparse (n,k) ;
%   fprintf ('\n%2d', k) ;
%     for iso = 0 % [0 1]

        clear F
        if (iso)
            F.matrix = pi * ones (n,k) ;
        else
            F = FF {k} ;
%           F = GB_spec_random (n, k, inf) ;
        end
        F.sparsity = 8 ;    % full
        F.iso = iso ;

%       for A_sparsity = [1 2 4 8]
%           for B_sparsity = [1 2 4 8]

%               fprintf ('.') ;
                clear A B
                if (A_sparsity == 8)
                    A = AA {1} ;
%                   A = GB_spec_random (n, n, inf) ;
                else
                    A = AA {2} ;
%                   A = GB_spec_random (n, n, 0.3) ;
                end
                A.sparsity = A_sparsity ;

                if (B_sparsity == 8)
                    B = BB {k,1} ;
%                   B = GB_spec_random (n, k, inf) ;
                else
                    B = BB {k,2} ;
%                   B = GB_spec_random (n, k, 0.3) ;
                end
                B.sparsity = B_sparsity ;

%               for trial = 1:5

                    if (trial == 1)
                        % plus_times_double
                        accum.opname = 'plus' ;
                        accum.optype = 'double' ;
                        semiring.add = 'plus' ;
                        semiring.multiply = 'times' ;
                        semiring.class = 'double' ;
                        tol = 1e-12 ;
                        A.class = 'double' ;
                        B.class = 'double' ;
                        F.class = 'double' ;
                    elseif (trial == 2)
                        % max_firstj_int64
                        accum.opname = 'max' ;
                        accum.optype = 'int64' ;
                        semiring.add = 'max' ;
                        semiring.multiply = 'firstj' ;
                        semiring.class = 'int64' ;
                        tol = 0 ;
                        A.class = 'int64' ;
                        B.class = 'int64' ;
                        F.class = 'int64' ;
                    elseif (trial == 3)
                        % max_firstj1_int64
                        accum.opname = 'max' ;
                        accum.optype = 'int64' ;
                        semiring.multiply = 'firstj1' ;
                        semiring.add = 'max' ;
                        semiring.class = 'int64' ;
                        tol = 0 ;
                        A.class = 'int64' ;
                        B.class = 'int64' ;
                        F.class = 'int64' ;
                    elseif (trial == 4)
                        % min_firstj_int64
                        accum.opname = 'min' ;
                        accum.optype = 'int64' ;
                        semiring.add = 'min' ;
                        semiring.multiply = 'firstj' ;
                        semiring.class = 'int64' ;
                        tol = 0 ;
                        A.class = 'int64' ;
                        B.class = 'int64' ;
                        F.class = 'int64' ;
                    else
                        % min_firstj1_int64
                        accum.opname = 'min' ;
                        accum.optype = 'int64' ;
                        semiring.multiply = 'firstj1' ;
                        semiring.add = 'min' ;
                        semiring.class = 'int64' ;
                        tol = 0 ;
                        A.class = 'int64' ;
                        B.class = 'int64' ;
                        F.class = 'int64' ;
                    end

                    % C = F ; C += A'*B, using dot4
                    C1 = GB_mex_mxm  (F, [ ], accum, semiring, A, B, desc) ;
                    C2 = GB_spec_mxm (F, [ ], accum, semiring, A, B, desc) ;
                    GB_spec_compare (C1, C2, tol) ;

                    % C = A'*B, using dot2
                    C1 = GB_mex_mxm  (C0, [ ], [ ], semiring, A, B, desc) ;
                    C2 = GB_spec_mxm (C0, [ ], [ ], semiring, A, B, desc) ;
                    GB_spec_compare (C1, C2, tol) ;

                    if (track_coverage)
                        c = sum (GraphBLAS_grbcov > 0) ;
                        d = c - clast ;
                        if (d > 0)
                            fprintf ('    {%2d, %d, %d, %d},', ...
                                k, A_sparsity, B_sparsity, trial) ;
                            fprintf (' ... %% (%3d, %3d)\n', d, c-cfirst) ;
                        end
                        clast = c ;
                    else
                        fprintf ('.') ;
                    end

%               end
%           end
%       end
%   end
end

%{
fini = GraphBLAS_grbcov ;
o = find (orig > 0) ;
f = find (fini > 0) ;
% setdiff (f,o)'
% save t2 orig fini
tt = load ('t2.mat') ;
fgood = find (tt.fini > 0) ;
setdiff (fgood, f)' + 1
%}

fprintf ('\ntest238: all tests passed\n') ;

