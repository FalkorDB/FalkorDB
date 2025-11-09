function test191(tasks)
%TEST191 test split

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('test191 ----------- Tiles = split (A)\n') ;

if (nargin < 1)
    tasks = [ ] ;
end

if (isempty (tasks))
    tasks = {
    {1,        'logical', 1, 0}, ... % (  3,   3)
    {1,        'logical', 4, 0}, ... % (  2,   5)
    {1,          'int16', 1, 0}, ... % (  1,   6)
    {1,          'int16', 4, 0}, ... % (  1,   7)
    {1,          'int32', 1, 0}, ... % (  1,   8)
    {1,          'int32', 4, 0}, ... % (  1,   9)
    {1,          'int64', 4, 0}, ... % (  1,  10)
    {1, 'double complex', 1, 0}, ... % (  1,  11)
    {1, 'double complex', 4, 0}, ... % (  1,  12)
    {2,        'logical', 8, 0}, ... % (  3,  15)
    {2,          'int16', 8, 0}, ... % (  1,  16)
    {2,          'int32', 8, 0}, ... % (  1,  17)
    {2,          'int64', 8, 0}, ... % (  1,  18)
    {2, 'double complex', 8, 0}, ... % (  1,  19)
    } ;
end

track_coverage = false ;
if (track_coverage)
    global GraphBLAS_grbcov
    track_coverage = ~isempty (GraphBLAS_grbcov) ;
    clast = sum (GraphBLAS_grbcov > 0) ;
    cfirst = clast ;
end

[~, ~, ~, types, ~, ~] = GB_spec_opsall ;
types = types.all ;

rng ('default') ;

m = 100 ;
n = 110 ;
ms = [10 1 89] ;
ns = [1 4 50 45 10] ;

% densities = [1e-4 0.01 0.2 0.8 inf] ;
densities = [0.01 inf] ;

% create the test matrices
for kd = 1:length(densities)
    dd = densities (kd) ;
    AA {kd} = GB_spec_random (m, n, dd, 128) ;
end

for kk = 1:length(tasks)
    task = tasks {kk} ;
    kd = task {1} ;
    atype = task {2} ;
    sparsity_control = task {3} ;
    is_csc = task {4} ;

% for kd = 1:length(densities)
%   dd = densities (kd) ;
%   fprintf ('\nd = %g\n', dd) ;
    A = AA {kd} ;

%   for ka = 1:length (types)
%       atype = types {ka} ;
%       A = GB_spec_random (m, n, dd, 128, atype) ;
        A.class = atype ;

%       for sparsity_control = [1 2 4 8]
%           fprintf ('.') ;
            A.sparsity = sparsity_control ;

%           for is_csc = [0 1]

                A.is_csc = is_csc ;
                C2 = GB_spec_split (A, ms, ns) ;
                C1 = GB_mex_split  (A, ms, ns) ;
                for i = 1:length(ms)
                    for j = 1:length(ns)
                        GB_spec_compare (C1 {i,j}, C2 {i,j}) ;
                    end
                end

                if (nnz (A.matrix) > 0)
                    % also try the iso case
                    B = A ;
                    B.matrix = spones (A.matrix) * pi ;
                    B.iso = true ;
                    C2 = GB_spec_split (B, ms, ns) ;
                    C1 = GB_mex_split  (B, ms, ns) ;
                    for i = 1:length(ms)
                        for j = 1:length(ns)
                            GB_spec_compare (C1 {i,j}, C2 {i,j}) ;
                        end
                    end
                end

                if (track_coverage)
                    c = sum (GraphBLAS_grbcov > 0) ;
                    d = c - clast ;
                    if (d > 0)
                        aa = sprintf ('''%s''', atype) ;
                        fprintf ('{%d, %16s, %d, %d},', ...
                            kd, aa, sparsity_control, is_csc) ;
                        fprintf (' ... %% (%3d, %3d)\n', d, c-cfirst) ;
                    end
                    clast = c ;
                else
                    fprintf ('.') ;
                end

%           end
%       end
%   end
end

fprintf ('\n') ;
fprintf ('test191: all tests passed\n') ;

