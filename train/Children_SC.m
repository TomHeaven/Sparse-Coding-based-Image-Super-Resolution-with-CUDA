function Children_SC(D)

% normalize the dictionary
norm_D = sqrt(sum(D.^2, 1)); 
D = D./repmat(norm_D, size(D, 1), 1);

[m,n] = size(D);
base = eye(m);  % base
Children_sparse_coe = zeros(n,m);  % sparse coefficients of base
for i = 1:m
    fprintf('%d\n', i);
    Children_sparse_coe(:,i) = lbreg_fixedstep(D,base(:,i),0.1,[]);
    %base_sparse_coe(:,i) = mexLasso(base(:,i),Dictionary,param);
end
disp([num2str(m),' Children Sparse Coefficients have been obtained']);
save('Children_sparse_coe.mat','Children_sparse_coe');
