function gbdemo2 (bnz)
%GBDEMO2 Extreme performance differences: GraphBLAS vs built-in methods
%
% Usage:
%
%       gbdemo2             % uses a default bnz = 6000
%       gbdemo2 (20000)     % uses bnz = 20000
%
% Many of the GraphBLAS operations used in gbdemo are perhaps 3x to
% 50x faster than the corresponding MATLAB operations, depending on how
% many cores your computer has.  Here's an example where GraphBLAS is
% asymptotically far faster than MATLAB R2019a: a simple assignment
% for a large matrix C:
%
%       C(I,J) = A
%
% The matrix C is constructed via C = kron (B,B) where nnz (B) is
% roughly the bnz provided on input (with a default of bnz = 6000),
% so that C will have about bnz^2 entries, or 36 million by default.
% I and J are chosen randomly, and A is 5000-by-5000.
%
% When the problem becomes large, MATLAB will take a very long time.
% If you have enough memory, and want to see higher speedups in
% GraphBLAS, increase bnz (and be prepared to wait even longer).
% With the default bnz = 6000, this test takes about 4GB of RAM.
%
% On my Dell XPS 4-core laptop (Intel(R) Core(TM) i7-8565U, 16GB
% RAM), using MATLAB R2019a, when C becomes 9 million by 9 million,
% the computation C(I,J)=A for MATLAB matrices C, I, J, and A takes
% several minutes, whereas GraphBLAS takes less than a second, or
% about 500x faster than MATLAB.  On a desktop with an Intel(R)
% Xeon(R) CPU E5-2698 v4 @ 2.20GHz with 20 hardware cores, the
% speedup over MATLAB is even more dramatic (up to 2,660x has been
% observed).
%
% See also GrB.assign, subsasgn.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

% reset to the default number of threads
ncores = demo_nproc ;
GrB.clear ;
nthreads = GrB.threads ;
help gbdemo2
fprintf ('\n# of threads used in GraphBLAS: %d\n\n', nthreads) ;

if (nargin < 1)
    bnz = 6000 ;
end

k = 5000 ;
anz = 50000 ;
A = sprandn (k, k, anz / k^2) ;

for n = 1000:1000:6000

    % reset the random number generator for repeatable results
    rng ('default') ;

    tic
    B = sprandn (n, n, bnz / n^2) ;
    C = kron (B, B) ;
    cn = size (C,1) ;
    I = randperm (cn, k) ;
    J = randperm (cn, k) ;
    G = GrB (C) ;
    t_setup = toc ;

    fprintf ('\nC(I,J)=A where C is %g million -by- %g million\n', ...
        cn /1e6, cn /1e6) ;
    fprintf ('with %g million entries:\n\n', nnz (C) / 1e6) ;
    fprintf ('    A is %d-by-%d with %d entries\n', k, k, nnz (A)) ;
    fprintf ('    setup time:     %g sec\n', t_setup) ;

    % do the assignment in GraphBLAS
    tic
    G (I,J) = A ;
    gb_time = toc ;

    fprintf ('    GraphBLAS time: %g sec\n', gb_time) ;
    fprintf ('    Starting MATLAB ... please wait ... \n') ;

    % do the same assignment in pure MATLAB
    tic
    C (I,J) = A ;
    builtin_time = toc ;

    fprintf ('    MATLAB time:    %g sec\n', builtin_time) ;
    fprintf ('    Speedup of GraphBLAS over MATLAB: %g\n', ...
        builtin_time / gb_time) ;

    % check the result
    tic
    assert (isequal (C, double (G))) ;
    t_check = toc ;
    fprintf ('    check time:     %g sec\n', t_check) ;
    fprintf ('    all tests passed\n') ;

end


