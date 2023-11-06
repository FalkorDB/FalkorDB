function test06 (A,B,fulltests,method_list)
%TEST06 test GrB_mxm on all semirings
%
% Usage: test06(A)
%        test06(A,B)
%        test06(A,B,fulltests)
%
% with no input, a small 10-by-10 matrix is used.  If A is a scalar, it is a
% matrix id number from the SuiteSparse collection otherwise A is the sparse
% matrix to use in the test

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('test06: GrB_mxm on all semirings\n') ;

[binops, ~, add_ops, types, ~, ~] = GB_spec_opsall ;
% mult_ops = binops.positional ;
mult_ops = binops.all ;
types = types.all ;

if (nargin < 3)
    fprintf ('\n-------------- GrB_mxm on all semirings\n') ;
    fulltests = 1 ;
end

if (nargin < 2)
    B = [ ] ;
end

if (nargin < 4)
    method_list = 0:3 ;
end

rng ('default') ;

if (isscalar (A))
    Prob = ssget (A)
    A = Prob.A ;
    clear Prob
    A (1,2) = 1 ;
    if (isempty (B))
        B = 2*A ;

%       n = size (A,2) ;
%       p = randperm (n) ;
%       B = sparse (1:n, p, ones (n,1)) ;
%       p = randperm (n) ;
%       B = B + sparse (1:n, p, ones (n,1)) ;
%       p = randperm (n) ;
%       B = B + sparse (1:n, p, ones (n,1)) ;
%       e = ones (n,1) ;
%       % B = spdiags([e -2*e e], -1:1, n, n) ;
%       % B = spdiags([-2*e e], 0:1, n, n) ;
%       % B = speye(n);
%       T = A ; A = B ; B = T ;

    end
end

assert (issparse (A)) ;

[m n] = size (A) ;
Cin = sparse (m, n) ;
assert (m == n) ;

rng ('default') ;
[i, j, ~] = find (A) ;
nz = nnz (A) ;
p = randperm (nz, floor (nz/2)) ;

Mask = sparse (i (p), j (p), ones (length (p),1), m, n) + ...
       spones (sprandn (m, n, 1/n)) ;


tic
C = A*B ;
tm1 = toc ;

% Mask = spones (A) ;
% Mask = sparse (i (p), j (p), ones (length (p),1), m, n) ;

tic
C = A'*B ;
tm2 = toc ;

tic
C = A*B' ;
tm3 = toc ;

tic
C = A'*B' ;
tm5 = toc ;

if (n > 500)
    fprintf ('built-in time: %g %g %g %g\n', tm1, tm2, tm3, tm5) ;
    fprintf ('with mask:\n') ;
end

tic
C = (A*B) .* Mask ;
tmm1 = toc ;

tic
C = (A'*B) .* Mask ;
tmm2 = toc ;

tic
C = (A*B') .* Mask ;
tmm3 = toc ;

tic
C = (A'*B') .* Mask ;
tmm5 = toc ;

if (n > 500)
    fprintf ('built-in time: %g %g %g %g\n', tmm1, tmm2, tmm3, tmm5) ;
end

dnn = struct ;
dtn = struct ( 'inp0', 'tran' ) ;
dnt = struct ( 'inp1', 'tran' ) ;
dtt = struct ( 'inp0', 'tran', 'inp1', 'tran' ) ;

n_semirings = 0 ;

if (fulltests)
    k1_list = 1:length (mult_ops) ;
    k2_list = 1:length (add_ops) ;
    k3_list = 1:length (types) ;
else
    % just use plus-times-double semiring
    k1_list = 4 ;
    k2_list = 3 ;
    k3_list = 11 ;
end

n = size (A,1) ;

for k1 = k1_list % 1:length(mult_ops)
    mulop = mult_ops {k1} ;
    if (n <= 500)
        fprintf ('\n%s', mulop) ;
    end

    for k2 = k2_list % 1:length(add_ops)
        addop = add_ops {k2} ;

        for k3 = k3_list % 1:length (types)
            semiring_type = types {k3} ;
            if (n <= 500)
               fprintf ('.') ;
            end

            semiring.multiply = mulop ;
            semiring.add = addop ;
            semiring.class = semiring_type ;

            % create the semiring.  some are not valid because the or,and,xor,eq
            % monoids can only be used when z is boolean for z=mult(x,y).
            try
                [mult_op add_op id] = GB_spec_semiring (semiring) ;
                [mult_opname mult_optype ztype xtype ytype] = ...
                    GB_spec_operator (mult_op) ;
                [ add_opname  add_optype] = GB_spec_operator (add_op) ;
                identity = GB_spec_identity (semiring.add, add_optype) ;
            catch me
                if (~isempty (strfind (me.message, 'gotcha')))
                    semiring
                end
                continue
            end

            n_semirings = n_semirings + 1 ;

            for method = method_list % 0:3

                if (n > 500)
                    fprintf ('%3d ', n_semirings) ;
                    fprintf ('[%6s %6s %8s] : ', mulop, addop, semiring_type) ;
                end

                if (method == 1)
                    algo = 'hash' ;
                    if (n > 500)
                        fprintf ('hash ') ;
                    end
                elseif (method == 2)
                    algo = 'gustavson' ;
                    if (n > 500)
                        fprintf ('g/s  ') ;
                    end
                elseif (method == 3)
                    algo = 'dot' ;
                    if (n > 500)
                        fprintf ('dot  ') ;
                    end
                else
                    algo = 'default' ;
                    if (n > 500)
                        fprintf ('auto ') ;
                    end
                end
                if (isequal (algo, 'dot'))
                    ok = (n < 1000) ;
                else
                    ok = true ;
                end

                dnn.axb = algo ;
                dnt.axb = algo ;
                dtn.axb = algo ;
                dtt.axb = algo ;

                t1 = nan ;
                t2 = nan ;
                t3 = nan ;
                t4 = nan ;

                % C = A*B, no mask
                % tic
                if (ok)
                tic
                C1 = GB_mex_mxm  (Cin, [ ], [ ], semiring, A, B, dnn) ;
                t1 = toc ;
                if (n < 200)
                C2 = GB_spec_mxm (Cin, [ ], [ ], semiring, A, B, dnn);
                GB_spec_compare (C1, C2, id) ;
                end
                end

                % C = A'*B, no mask
                if (ok)
                tic
                C1 = GB_mex_mxm  (Cin, [ ], [ ], semiring, A, B, dtn);
                t2 = toc ;
                if (n < 200)
                C2 = GB_spec_mxm (Cin, [ ], [ ], semiring, A, B, dtn);
                GB_spec_compare (C1, C2, id) ;
                end
                end

                % C = A*B', no mask
                if (ok)
                tic
                C1 = GB_mex_mxm  (Cin, [ ], [ ], semiring, A, B, dnt);
                t3 = toc ;
                if (n < 200)
                C2 = GB_spec_mxm (Cin, [ ], [ ], semiring, A, B, dnt);
                GB_spec_compare (C1, C2, id) ;
                end
                end

                % C = A'*B', no mask
                if (ok)
                tic
                C1 = GB_mex_mxm  (Cin, [ ], [ ], semiring, A, B, dtt);
                t4 = toc ;
                if (n < 200)
                C2 = GB_spec_mxm (Cin, [ ], [ ], semiring, A, B, dtt);
                GB_spec_compare (C1, C2, id) ;
                end
                end

                if (n > 500)
                    fprintf (...
                    'speedups %10.4f(%s) %10.4f(%s) %10.4f(%s) %10.4f(%s) ', ...
                    tm1/t1, tm2/t2, tm3/t3, tm5/t4 ) ;
                end

                % C = A*B, with mask
                tic
                C1 = GB_mex_mxm  (Cin, Mask, [ ], semiring, A, B, dnn);
                t1 = toc ;
                if (n < 200)
                C2 = GB_spec_mxm (Cin, Mask, [ ], semiring, A, B, dnn);
                GB_spec_compare (C1, C2, id) ;
                end

                % C = A'*B, with mask
                tic
                C1 = GB_mex_mxm  (Cin, Mask, [ ], semiring, A, B, dtn);
                t2 = toc ;
                if (n < 200)
                C2 = GB_spec_mxm (Cin, Mask, [ ], semiring, A, B, dtn);
                GB_spec_compare (C1, C2, id) ;
                end

                % C = A*B', with mask
                tic
                C1 = GB_mex_mxm  (Cin, Mask, [ ], semiring, A, B, dnt);
                t3 = toc ;
                if (n < 200)
                C2 = GB_spec_mxm (Cin, Mask, [ ], semiring, A, B, dnt);
                GB_spec_compare (C1, C2, id) ;
                end

                % C = A'*B', with mask
                tic
                C1 = GB_mex_mxm  (Cin, Mask, [ ], semiring, A, B, dtt);
                t4 = toc ;
                if (n < 200)
                C2 = GB_spec_mxm (Cin, Mask, [ ], semiring, A, B, dtt);
                GB_spec_compare (C1, C2, id) ;
                end

                if (n > 500)
                    fprintf (...
                    'speedups %10.4f(%s) %10.4f(%s) %10.4f(%s) %10.4f(%s) ', ...
                    tmm1/t1, tmm2/t2, tmm3/t3, tmm5/t4) ;
                    fprintf ('\n') ;
                end

            end
        end
    end
end

% n_semirings

if (fulltests)
    fprintf ('\ntest06: all tests passed\n') ;
end

