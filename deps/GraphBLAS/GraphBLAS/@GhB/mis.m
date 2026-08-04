function iset = mis (A_arg, check)
%GHB.MIS variant of Luby's maximal independent set algorithm.
%
%   iset = GhB.mis (A) ;
%
% Given an n-by-n symmetric adjacency matrix A of an undirected graph, GhB.mis
% (A) finds a maximal set of independent nodes and returns it as a logical
% vector, iset, where iset(i) of true implies node i is a member of the set.
%
% The matrix A must not have any diagonal entries (self edges), and it must be
% symmetric.  These conditions are not checked by default, and results are
% undefined if they do not hold.  In particular, diagonal entries will cause
% the method to stall.  To check these conditions, use:
%
%   iset = GhB.mis (A, 'check') ;
%
% Reference: M Luby. 1985. A simple parallel algorithm for the maximal
% independent set problem. In Proceedings of the seventeenth annual ACM
% symposium on Theory of computing (STOC '85). ACM, New York, NY, USA, 1-10.
% DOI: https://doi.org/10.1145/22145.22146

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

[m, n] = size (A_arg) ;
if (m ~= n)
    error ('GrB:error', 'A must be square') ;
end

% convert A to logical
% A = GhB.apply ('1.logical', A_arg) ;
A = A_arg ;

if (nargin < 2)
    check = false ;
else
    if (isequal (check, 'check'))
        check = true ;
    else
        error ('GrB:error', 'unknown option') ;
    end
end

if (check)
    if (nnz (diag (A)) > 0)
        error ('GrB:error', 'A must not have any diagonal entries') ;
    end
    if (~issymmetric (A))
        error ('GrB:error', 'A must be symmetric') ;
    end
end

neighbor_max = GhB (n, 1) ;
new_neighbors = GhB (n, 1, 'logical') ;
candidates = GhB (n, 1, 'logical') ;

% Initialize independent set vector
iset = GhB (n, 1, 'logical') ;

% descriptor: C_replace
r_desc.out = 'replace' ;

% descriptor: C_replace + structural complement of mask
sr_desc.mask = 'complement' ;
sr_desc.out  = 'replace' ;

% compute the degree of each nodes
desc1 = r_desc  ;
desc2 = struct ;
if (GhB.isbyrow (A))
    % degrees = GhB.vreduce ('+.double',  A) ;
    degrees = GhB.entries (A, 'row', 'degree') ;
else
    degrees = GhB.entries (A, 'col', 'degree') ;
    desc1.in0 = 'transpose' ;
    desc2.in0 = 'transpose' ;
end

% Singletons require special treatment.  Since they have no neighbors, their
% prob is never greater than the max of their neighbors, so they never get
% selected and cause the method to stall.  To avoid this case they are removed
% from the candidate set at the begining, and added to the iset.

% candidates (degree != 0) = true
GhB.assign (candidates, degrees, true) ;

% add all singletons to iset
% iset (degree == 0) = 1
GhB.assign (iset, degrees, true, sr_desc) ;

% Iterate while there are candidates to check.
ncand = GhB.entries (candidates) ;
last_ncand = ncand ;

while (ncand > 0)

    % compute a random probability scaled by inverse of degree
    % FUTURE: this is slower than it should be; rand may not be parallel,
    prob = 0.0001 + rand (n,1) ./ (1 + 2 * degrees) ;
    prob = GhB.assign (prob, candidates, prob, r_desc) ;

    % compute the max probability of all neighbors
    GhB.mxm (neighbor_max, candidates, 'max.second.double', A, prob, desc1) ;

    % select node if its probability is > than all its active neighbors
    new_members = GhB.eadd (prob, '>', neighbor_max) ;

    % add new members to independent set.
    GhB.eadd (iset, iset, '|', new_members) ;

    % remove new members from set of candidates
    GhB.apply (candidates, new_members, 'identity', candidates, sr_desc) ;

    ncand = GhB.entries (candidates) ;
    if (ncand == 0)
        break ;                    % early exit condition
    end

    % Neighbors of new members can also be removed from candidates
    GhB.mxm (new_neighbors, candidates, '|.second.logical', A, new_members, ...
        desc2) ;
    GhB.apply (candidates, new_neighbors, 'identity', candidates, sr_desc) ;
    ncand = GhB.entries (candidates) ;

    % this will not occur, unless the input is corrupted somehow
    if (last_ncand == ncand)
        error ('GrB:error', 'method stalled; rerun with ''check'' option') ;
    end
    last_ncand = ncand ;
end

% drop explicit false values
iset = GhB.prune (iset) ;

