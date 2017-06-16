function x = Data_Separation(D,y,Children_sparse_coe)


for k = 1:block_size^2
    temp_sum_coe = temp_sum_coe + img_crop(k) * base_sparse_coe(:,k);
    recon_block(:,temp_index) = recon_block(:,temp_index) + img_crop(k) * Dictionary * base_sparse_coe(:,k); 
end
