# Targeted literature check, 25 October 2026

This is an initial targeted review, not an exhaustive novelty search or an
independent verification of all the cited papers' proofs.

* Kane and Nelson, *Sparser Johnson-Lindenstrauss Transforms* (journal version,
  2014): sparse random dimensionality reduction and its embedding-time advantage
  are established. Adding a sparse map alone is not a new mathematical
  contribution. [Primary paper](https://arxiv.org/abs/1012.1577)
* Balabanov and Nouy, *Randomized linear algebra for model reduction. Part II*
  (2021): random sketches, residual-based reduced approximation and a posteriori
  sketch-quality verification already coexist in the literature. Our proposed
  hint/aggregate experiment must demonstrate a distinct authenticated protocol
  and useful measured tradeoff, rather than claim sketch-based certification
  itself as new. [Primary paper](https://arxiv.org/abs/1910.14378)
* Hardt and Woodruff, *How Robust are Linear Sketches to Adaptive Inputs?*
  (2013): correlated, adaptively chosen queries can defeat general linear
  norm sketches. A fixed-family JL probability statement does not automatically
  cover unrestricted future center updates. Our tested deterministic envelopes
  and untrusted hints avoid relying on such an extrapolation for acceptance.
  [Primary paper](https://arxiv.org/abs/1211.1056)
* Dey and Cherkaoui, *Deterministic Johnson-Lindenstrauss Projections from Pisot
  beta-Transformations for Zero-Knowledge Private Routing* (August2026 preprint):
  this explicitly combines ZK, centroid comparisons and dimension reduction.
  Its application and construction differ from the reusable committed-catalog
  cost-change protocol. Still, it is direct evidence that a broad novelty claim
  of combining JL with ZK/centroids would be inadequate. This entry does not
  endorse every theorem or engineering claim in that preprint.
  [Primary preprint](https://arxiv.org/abs/2608.13078)

The plausible research direction would be a precise protocol where an untrusted
JL proposal reduces authenticated exact work while a separate deterministic
verifier retains soundness for future centers, with all preprocessing, extra
storage and fallback costs included. The current small tests do not establish
a sufficiently large benefit to support that contribution claim.
