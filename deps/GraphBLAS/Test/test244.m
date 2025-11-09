function test244(tasks)
%TEST244 test reshape

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin < 1)
    tasks = [ ] ;
end

if (isempty (tasks))
    tasks = {
    {        'double',  6,  6, 1, 1, 0, 0}, ... % ( 15,  15)
    {        'double',  6,  6, 1, 4, 0, 0}, ... % (  2,  17)
    } ;
end

track_coverage = false ;
if (track_coverage)
    global GraphBLAS_grbcov
    track_coverage = ~isempty (GraphBLAS_grbcov) ;
    clast = sum (GraphBLAS_grbcov > 0) ;
    cfirst = clast ;
end

rng ('default')

[~, ~, ~, types, ~, ~] = GB_spec_opsall ;
types = types.all ;

% create the test matrices
densities = [0.3 inf] ;
for m = 6 % [1 2 6] % 1:6
    for n = 6 % [1 2 6] % 1:6
        for kd = 1:2
            AA {m,n,kd} = GB_spec_random (m, n, densities (kd), 99) ;
        end
    end
end

for kk = 1:length(tasks)
    task = tasks {kk} ;
    type = task {1} ;
    m = task {2} ;
    m = task {3} ;
    kd = task {4} ;
    sparsity = task {5} ;
    is_csc = task {6} ;
    iso = task {7} ;
% end

% for k1 = 1:length(types)
%     type = types {k1} ;
%   fprintf ('\n%-14s ', type) ;

%     for m = 6 % [1 2 6] % 1:6
%         for n = 6 % [1 2 6] % 1:6
            mn = m*n ;
            f = factor (mn) ;

%           for d = [0.3 inf]
%           for kd = 1:2

                A = AA {m,n,kd} ; % GB_spec_random (m, n, d, 99, type) ;
%               fprintf ('.') ;
%               for sparsity = [1 2 4 8]
                    A.sparsity = sparsity ;
%                   for is_csc = [0 1]
                        A.is_csc = is_csc ;
%                       for iso = [false true]
                            A.iso = iso ;

                            for k = 1:length (f)
                                S = nchoosek (f, k) ;
                                for i = 1:size(S,1)
                                    m2 = prod (S (i,:)) ;
                                    n2 = mn / m2 ;
                                    % reshape by column
                                    C1 = A ;
                                    x = 1 ;
                                    if (iso)
                                        [i,j,x] = find (C1.matrix, 1,'first') ;
                                        C1.matrix (C1.pattern) = x ;
                                    end
                                    C1.matrix  = reshape (C1.matrix,  m2, n2) ;
                                    C1.pattern = reshape (C1.pattern, m2, n2) ;
                                    for inplace = [false true]
                                        C2 = GB_mex_reshape (A, m2, n2, ...
                                            true, inplace) ;
                                        GB_spec_compare (C1, C2, 0) ;
                                    end
                                    % reshape by row
                                    C1 = A ;
                                    if (iso)
                                        C1.matrix (C1.pattern) = x ;
                                    end
                                    C1.matrix  = reshape (C1.matrix', n2, m2)' ;
                                    C1.pattern = reshape (C1.pattern', n2, m2)';
                                    for inplace = [false true]
                                        C2 = GB_mex_reshape (A, m2, n2, ...
                                            false, inplace) ;
                                        GB_spec_compare (C1, C2, 0) ;
                                    end
                                end
                            end

    if (track_coverage)
        c = sum (GraphBLAS_grbcov > 0) ;
        d = c - clast ;
        if (d > 0)
            tt = sprintf ('''%s''', type) ;
            fprintf ('{%16s, %2d, %2d, %d, %d, %d, %d},', ...
                tt, m, n, kd, sparsity, is_csc, iso) ;
            fprintf (' ... %% (%3d, %3d)\n', d, c-cfirst) ;
        end
        clast = c ;
    else
        fprintf ('.') ;
    end

%                       end
%                   end
%               end
%           end
%       end
%   end
end

fprintf ('\ntest244: all tests passed\n') ;
