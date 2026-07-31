function result = gb_entries (ghb, A, varargin)
%GB_ENTRIES count or query the entries of a matrix.  Not user-callable.
% Implements GrB.entries (A, ...) and GrB.nonz (A, ...).

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (gb_is_grb (A))
    A = struct (A) ;
end

% get the string arguments
dim = 'all' ;           % 'all', 'row', or 'col'
kind = 'count' ;        % 'count', 'list', or 'degree'
for k = 1:nargin-2
    arg = varargin {k} ;
    switch arg
        case { 'all', 'row', 'col' }
            dim = arg ;
        case { 'count', 'list', 'degree' }
            kind = arg ;
        otherwise
            error ('GrB:error', 'unknown option') ;
    end
end

if (isequal (dim, 'all'))

    switch kind
        case 'count'
            % number of entries in A
            % e = GrB.entries (A)
            result = gzb_nvals (A) ;
        case 'list'
            % list of values of unique entries
            % X = GrB.entries (A, 'list')
            gbmex_wait (A) ;
            result = unique (gbmex_extractvalues (A)) ;
        otherwise
            error ('GrB:error', '''all'' and ''degree'' cannot be combined') ;
    end

else

    % get the row or column degree
    result = gzb_degree (ghb, A, dim) ;    % dim is 'row' or 'col'

    switch kind
        case 'count'
            % number of non-empty rows/cols
            % e = GrB.entries (A, 'row')
            % e = GrB.entries (A, 'col')
            result = gzb_nvals (gzb_select (ghb, result, 'nonzero')) ;
        case 'list'
            % list of non-empty rows/cols
            % I = GrB.entries (A, 'row', 'list')
            % J = GrB.entries (A, 'col', 'list')
            desc.base = 'one-based int' ;
            S = gzb_select (1, result, 'nonzero') ;
            gbmex_wait (S) ;
            % return result as a builtin MATLAB/Octave vector
            result = gbmex_extracttuples (1, S, desc) ;
        % case 'degree'
            % degree of all rows/cols
            % d = GrB.entries (A, 'row', 'degree')
            % d = GrB.entries (A, 'col', 'degree')
            % result is returned as a GraphBLAS struct, already computed above
    end
end

