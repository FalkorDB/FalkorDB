function C = GB_spec_kron_idx (A, B, atrans, btrans)
% C = kron(A,B), using the mykronidx operator

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin < 3)
    atrans = [ ] ;
end
if (nargin < 4)
    btrans = [ ] ;
end
if (isempty (atrans))
    atrans = 0 ;
end
if (isempty (btrans))
    btrans = 0 ;
end

if (atrans)
    A = A' ;
end

if (btrans)
    B = B' ;
end

[ia,ja,x] = find (A) ;
[ib,jb,x] = find (B) ;

anz = length (ia) ;
bnz = length (ib) ;
cnz = anz * bnz ;

ic = zeros (cnz,1) ;
jc = zeros (cnz,1) ;
xc = zeros (cnz,1) ;
[ma, na] = size (A) ;
[mb, nb] = size (B) ;
nc = na * nb ;
mc = ma * mb ;

kc = 0 ;
for ka = 1:anz
    for kb = 1:bnz
        kc = kc + 1 ;
        ic (kc) = (ia (ka) - 1) * mb + ib (kb) ;
        jc (kc) = (ja (ka) - 1) * nb + jb (kb) ;
        xc (kc) = (ia (ka) * 1000000 + ...
                   ja (ka) *   10000 + ...
                   ib (kb) *     100 + ...
                   jb (kb)) ;
    end
end

C = sparse (ic, jc, xc, mc, nc) ;

