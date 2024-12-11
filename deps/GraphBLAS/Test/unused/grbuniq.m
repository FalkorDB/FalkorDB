
% load grbstat
ntests = length (GraphBLAS_grbcovs) ;
n = length (GraphBLAS_grbcovs {1}) ;
G = cell2mat (GraphBLAS_grbcovs') ;
T = cell2mat (GraphBLAS_times') ;

Mine = zeros (ntests, n) ;
Mine (1, :) = G (1, :) ;

for k = 2:ntests
    % find the statements covered by this test
    Mine (k,:) = G (k,:) - G (k-1,:) ;
end

% Mine (k,i) = 1 if the kth test covers statement i
Mine = full (spones (Mine)) ;

S = sum (Mine) ;
U = (S == 1) ;
keeper = [ ] ;

for k = 1:ntests
    my_uniq = (Mine (k,:) > 0) & U ;
    n_uniq = sum (my_uniq) ;
    t = T (k) ;
    fprintf ('%3d %-20s : %5d %9.2f', k, GraphBLAS_scripts {k}, n_uniq, t) ;
    fprintf ('   %10.2f ', t/n_uniq) ;
    if (n_uniq == 0)
        fprintf ('<<<<<<<<<<<') ;
    else
        keeper = [keeper k] ;
    end
    fprintf ('\n') ;
end

fprintf ('\n====================================\n') ;

[ignore I] = sort (T, 'descend') ;
for kk = 1:ntests
    k = I (kk) ;
    my_uniq = (Mine (k,:) > 0) & U ;
    n_uniq = sum (my_uniq) ;
    t = T (k) ;
    fprintf ('%3d %3d %-20s : %5d %9.2f', kk, k, GraphBLAS_scripts {k}, n_uniq, t) ;
    fprintf ('   %10.2f ', t/n_uniq) ;
    if (n_uniq == 0)
        fprintf ('<<<<<<<<<<<') ;
        Mine (k,:) = 0 ;
        S = sum (Mine) ;
        U = (S == 1) ;
    else
        keeper = [keeper k] ;
    end
    fprintf ('\n') ;
end



%{
fprintf ('\n====================================\n') ;

Mine2 = Mine (keeper, :) ;
S2 = sum (Mine2) ;
U2 = (S2 == 1) ;
ntests2 = length (keeper) ;
T2 = T (keeper) ;
Scr = GraphBLAS_scripts (keeper) ;

for k = 1:ntests2
    my_uniq = (Mine2 (k,:) > 0) & U2 ;
    n_uniq = sum (my_uniq) ;
    t = T2 (k) ;
    fprintf ('%3d %-20s : %5d %6.2f', k, Scr {k}, n_uniq, t) ;
    fprintf ('   %10.2f ', t/n_uniq) ;
    if (n_uniq == 0)
        fprintf ('<<<<<<<<<<<') ;
    else
        keeper = [keeper k] ;
    end
    fprintf ('\n') ;
end

sum (T) /60
sum (T2) /60
(sum (T) - sum (T2)) / 60
%}
