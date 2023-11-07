function [tM tG] = summary (tm, tg, ktrials)

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

tM = median (tm (1:ktrials)) ;
tG = median (tg (1:ktrials)) ;
