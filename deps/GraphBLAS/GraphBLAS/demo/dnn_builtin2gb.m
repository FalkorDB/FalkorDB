function [W, bias, Y0] = dnn_builtin2gb (ghb, W, bias, Y0)
%DNN_BUILTIN2GB convert sparse deep neural network from built-in to GraphBLAS
%
% Usage:
%
%   [W, bias, Y0] = dnn_builtin2gb (ghb, W, bias, Y0) ;
%
% This method converts a sparse deep neural network problem in the 2019 MIT
% GraphChallenge so that it can be used in GhB.dnn.
%
% If ghb = 0, GrB matrices are created.  Otherwise GhB matrices are created.
%
% See also GrB.dnn, dnn_builtin.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fmt = 'by row' ;
prec = 'single' ;

d = struct ('format', fmt) ;
n = size (Y0, 2) ;

if (ghb)
    Y0 = GhB (Y0, prec, fmt) ;
    for k=1:length(W)
        W {k} = GhB (W {k}, prec, fmt) ;
        bias {k} = GhB.build (1:n, 1:n, bias {k}, n, n, '+', prec, d) ;
    end
else
    Y0 = GrB (Y0, prec, fmt) ;
    for k=1:length(W)
        W {k} = GrB (W {k}, prec, fmt) ;
        bias {k} = GrB.build (1:n, 1:n, bias {k}, n, n, '+', prec, d) ;
    end
end

