| dataset_name | n | d | k | m | q | repetitions | ari_mean | ari_std | ari_ci95_low | ari_ci95_high | delta_q95_mean | delta_q95_std | delta_q95_ci95_low | delta_q95_ci95_high | r_phi_mean | r_phi_std | r_phi_ci95_low | r_phi_ci95_high | ground_truth_ari_full | ground_truth_ari_projected |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| arcene | 100 | 10000 | 5 | 16 | 1 | 5 | 0.9373 | 0.0249 | 0.9064 | 0.9683 | 0.6706 | 0.0560 | 0.6011 | 0.7401 | 1.0046 | 0.0039 | 0.9997 | 1.0095 | 0.0284 | 0.0255 |
| arcene | 100 | 10000 | 10 | 16 | 1 | 5 | 0.8260 | 0.0873 | 0.7177 | 0.9344 | 0.6706 | 0.0560 | 0.6011 | 0.7401 | 1.0079 | 0.0051 | 1.0016 | 1.0142 | 0.0283 | 0.0287 |
| fashion_mnist | 5000 | 784 | 5 | 64 | 1 | 5 | 0.8338 | 0.0128 | 0.8179 | 0.8497 | 0.3146 | 0.0092 | 0.3032 | 0.3260 | 1.0010 | 0.0004 | 1.0005 | 1.0015 | 0.2840 | 0.2759 |
| fashion_mnist | 5000 | 784 | 10 | 64 | 1 | 5 | 0.8231 | 0.0151 | 0.8044 | 0.8419 | 0.3146 | 0.0092 | 0.3032 | 0.3260 | 1.0012 | 0.0003 | 1.0008 | 1.0016 | 0.3410 | 0.3275 |
| gaussian_mixture | 2000 | 256 | 5 | 16 | 1 | 5 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 0.5634 | 0.0572 | 0.4924 | 0.6344 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| gaussian_mixture | 2000 | 256 | 10 | 256 | 1 | 5 | 0.8549 | 0.0329 | 0.8141 | 0.8957 | 0.1529 | 0.0180 | 0.1306 | 0.1752 | 1.0000 | 0.0001 | 0.9999 | 1.0001 | 0.7157 | 0.6968 |
| gaussian_mixture | 2000 | 512 | 5 | 16 | 1 | 5 | 0.8824 | 0.0727 | 0.7921 | 0.9728 | 0.6559 | 0.0682 | 0.5712 | 0.7406 | 1.0180 | 0.0187 | 0.9948 | 1.0412 | 0.6953 | 0.6607 |
| gaussian_mixture | 2000 | 512 | 10 | 16 | 1 | 5 | 0.9231 | 0.0321 | 0.8833 | 0.9630 | 0.6559 | 0.0682 | 0.5712 | 0.7406 | 1.0001 | 0.0001 | 0.9999 | 1.0002 | 0.9462 | 0.9235 |
| mnist | 5000 | 784 | 5 | 256 | 1 | 5 | 0.8311 | 0.0091 | 0.8199 | 0.8424 | 0.1669 | 0.0062 | 0.1592 | 0.1746 | 1.0006 | 0.0001 | 1.0004 | 1.0007 | 0.3007 | 0.2943 |
| mnist | 5000 | 784 | 10 | 256 | 1 | 5 | 0.8372 | 0.0146 | 0.8191 | 0.8553 | 0.1669 | 0.0062 | 0.1592 | 0.1746 | 1.0008 | 0.0001 | 1.0006 | 1.0009 | 0.3582 | 0.3496 |

Means, sample standard deviations, and two-sided 95% Student-t confidence intervals are computed over paired seeds for each configuration. ARI vs full measures whether SparseJL preserves the full-space clustering. Ground-truth ARI measures clustering quality where labels exist; it is not available for the DBSCAN-margin generator. This pilot uses five seeds and is preliminary.
