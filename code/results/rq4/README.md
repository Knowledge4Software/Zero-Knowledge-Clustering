# RQ4 — stability certificates

`runs/native/` holds 265 synthetic centroid and DBSCAN certificate rows. `runs/real/` holds 100 real-data centroid certificate rows on MNIST, Fashion-MNIST, Gisette, Arcene, and 20 Newsgroups. `analysis/` contains coverage, near-band, phase-diagram, and real-data summaries.

The assignment guarantee is conditional on the realized embedding event. That global event failed in all 100 real-data rows at the tested epsilon, so report pointwise conditional checks and unconditional changes separately. The approximately 99.95% DBSCAN pair coverage is dominated by far pairs; 250 of 499,500 pairs per synthetic row are in the difficult near-radius band.
