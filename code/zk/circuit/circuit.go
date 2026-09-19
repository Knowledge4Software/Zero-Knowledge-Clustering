// Package circuit implements the CommitJL proof relation in gnark, targeting
// BN254, matching manuscript.tex's proof statement:
//   public:  C_X, C_theta, C_A, centroids (theta), Psi_hat
//   witness: X, r_X (Merkle blinding), sparse list (A), assignments
//   constraints:
//     (i)   Merkle opening of X against C_X (MiMC tree, computed in-circuit
//           over all n*d leaves since the whole of X is witnessed anyway)
//     (ii)  hash(sparse list) == C_A  (skipped for variant=prepublished,
//           where A is a circuit constant with no authentication)
//     (iii) hash(theta) == C_theta
//     (iv)  sparse-list validity: row < m, sign in {-1,+1}, q distinct rows
//           per column
//     (v)   signed projection Z = A . x_i (per point) and A . c_j (per
//           centroid, computed once)
//     (vi)  range checks on quantized values (rangecheck gadget, or plain
//           bit-decomposition when NoRangeBatch is set)
//     (vii) squared distances between projected points/centroids
//     (viii) lowest-index argmin tournament matching commitjl/clustering.py
//     (ix)  Psi_hat equality
//
// Variants (manuscript Table "ablation-a" / cost_model.py's variant list):
//   full          - m == d, A == identity (no projection, no C_A check)
//   sparse        - canonical q-sparse {-1,0,+1} JL matrix (default), A is
//                   witnessed as a sparse list and authenticated against C_A
//   dense         - dense Rademacher-type projection, witnessed densely,
//                   authenticated against C_A (reuses the dense projection
//                   path; sparse-specific distinctness checks are skipped)
//   coord         - coordinate-sampling projection (a restriction of dense:
//                   each output row copies exactly one input coordinate,
//                   sign +-1) witnessed densely
//   pca-public    - dense quantized matrix baked in as a PUBLIC CONSTANT
//                   (compiled into the circuit); no C_A authentication
//   pisot         - dense public-integer projection matrix, same handling as
//                   pca-public (both are "public constant dense matrix"
//                   variants; pisot is reproduced per dey2026 at the Python
//                   generation layer in projections.py, the circuit only
//                   needs the resulting integer matrix)
//   prepublished  - A (sparse or dense, per Sparse flag) is a circuit
//                   CONSTANT baked in at compile time; no C_A authentication
//                   constraint at all
//   incircuit-gen - STUB: rebuilding the sparse generator + MiMC XOF inside
//                   the circuit is not implemented; see NewCircuit doc.
//
// Build-time flags (not runtime witness values, since they change the
// circuit's constraint structure):
//   NoRangeBatch - when true, range checks use manual bit-decomposition
//                  (frontend.API.ToBinary) instead of std/rangecheck's
//                  batched lookup gadget.
package circuit

import (
	"fmt"

	"math/big"

	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/std/hash/mimc"
	"github.com/consensys/gnark/std/math/cmp"
	"github.com/consensys/gnark/std/rangecheck"
)

type Variant string

const (
	VariantFull         Variant = "full"
	VariantSparse        Variant = "sparse"
	VariantDense         Variant = "dense"
	VariantCoord         Variant = "coord"
	VariantPCAPublic     Variant = "pca-public"
	VariantPisot         Variant = "pisot"
	VariantPrepublished  Variant = "prepublished"
	VariantIncircuitGen  Variant = "incircuit-gen"
)

// Params describes the fixed (compile-time) shape of an instance.
type Params struct {
	N, D, M, K, Q int
	Variant       Variant
	NoRangeBatch  bool
	RangeBits     int // bit-width for range checks (>= ceil(log2(2M+1)))
	// SparseSelf: true for the "sparse" family (row/sign witnessed per
	// (t,ell) slot with distinctness constraints); false for dense-style
	// variants where A is witnessed/constant as a full MxD matrix.
	SparseSelf bool
	// AConstant: true when A is baked in as a circuit constant (pca-public,
	// pisot, prepublished) rather than witnessed.
	AConstant bool
	// AuthenticateA: true when hash(A) == C_A must be enforced.
	AuthenticateA bool
	// ConstantA holds the compile-time matrix values when AConstant is set
	// (row-major, M x D), as int64 (already signed, e.g. {-1,0,1} or a
	// wider quantized range for pca-public/pisot).
	ConstantA [][]int64
}

// SparseEntry is one (row, sign) slot at column t, slot ell.
type SparseEntry struct {
	Row  frontend.Variable
	Sign frontend.Variable
}

// Circuit is the CommitJL relation.
type Circuit struct {
	P Params `gnark:"-"`

	// ---- Public inputs ----
	CX       frontend.Variable   `gnark:",public"`
	CTheta   frontend.Variable   `gnark:",public"`
	CA       frontend.Variable   `gnark:",public"`
	Centroids [][]frontend.Variable `gnark:",public"` // k x d
	PsiHat   frontend.Variable   `gnark:",public"`

	// ---- Witness ----
	X   [][]frontend.Variable `gnark:",secret"` // n x d
	RX  frontend.Variable     `gnark:",secret"` // merkle blinding salt

	// Sparse-list witness (used when P.SparseSelf && !P.AConstant): D x Q.
	SparseRows  [][]frontend.Variable `gnark:",secret"`
	SparseSigns [][]frontend.Variable `gnark:",secret"`

	// Dense witness A (used when !P.SparseSelf && !P.AConstant): M x D.
	DenseA [][]frontend.Variable `gnark:",secret"`

	Assign []frontend.Variable `gnark:",secret"` // n, each in [0,k)
}

func NewCircuit(p Params) *Circuit {
	c := &Circuit{P: p}
	c.Centroids = make([][]frontend.Variable, p.K)
	for j := range c.Centroids {
		c.Centroids[j] = make([]frontend.Variable, p.D)
	}
	c.X = make([][]frontend.Variable, p.N)
	for i := range c.X {
		c.X[i] = make([]frontend.Variable, p.D)
	}
	if p.SparseSelf && !p.AConstant {
		c.SparseRows = make([][]frontend.Variable, p.D)
		c.SparseSigns = make([][]frontend.Variable, p.D)
		for t := range c.SparseRows {
			c.SparseRows[t] = make([]frontend.Variable, p.Q)
			c.SparseSigns[t] = make([]frontend.Variable, p.Q)
		}
	}
	if !p.SparseSelf && !p.AConstant {
		c.DenseA = make([][]frontend.Variable, p.M)
		for r := range c.DenseA {
			c.DenseA[r] = make([]frontend.Variable, p.D)
		}
	}
	c.Assign = make([]frontend.Variable, p.N)
	return c
}

func mustMiMC(api frontend.API) mimc.MiMC {
	h, err := mimc.NewMiMC(api)
	if err != nil {
		panic(err)
	}
	return h
}

// project computes y = A . x  (length-M output) for a single length-D input
// vector x, given the circuit's representation of A (sparse-list, dense
// witness, or dense constant).
func (c *Circuit) project(api frontend.API, x []frontend.Variable) []frontend.Variable {
	p := c.P
	y := make([]frontend.Variable, p.M)
	for r := range y {
		y[r] = frontend.Variable(0)
	}
	if p.AConstant {
		for r := 0; r < p.M; r++ {
			acc := frontend.Variable(0)
			for t := 0; t < p.D; t++ {
				v := p.ConstantA[r][t]
				if v == 0 {
					continue
				}
				term := api.Mul(x[t], v)
				acc = api.Add(acc, term)
			}
			y[r] = acc
		}
		return y
	}
	if p.SparseSelf {
		// For each output row r, sum contributions from every (t,ell) slot
		// whose witnessed row equals r (multiplexed via equality
		// indicators), signed by SparseSigns.
		for r := 0; r < p.M; r++ {
			acc := frontend.Variable(0)
			for t := 0; t < p.D; t++ {
				for l := 0; l < p.Q; l++ {
					isR := api.IsZero(api.Sub(c.SparseRows[t][l], r))
					signedX := api.Mul(x[t], c.SparseSigns[t][l])
					contrib := api.Mul(isR, signedX)
					acc = api.Add(acc, contrib)
				}
			}
			y[r] = acc
		}
		return y
	}
	// Dense witnessed A.
	for r := 0; r < p.M; r++ {
		acc := frontend.Variable(0)
		for t := 0; t < p.D; t++ {
			acc = api.Add(acc, api.Mul(c.DenseA[r][t], x[t]))
		}
		y[r] = acc
	}
	return y
}

func (c *Circuit) rangeCheckSigned(api frontend.API, rc frontend.Rangechecker, v frontend.Variable, bits int) {
	// Range-check that v, offset into [0, 2^bits), is within bounds. Caller
	// is responsible for passing an already-offset (non-negative) value.
	if c.P.NoRangeBatch {
		api.ToBinary(v, bits) // plain bit-decomposition range check
		return
	}
	rc.Check(v, bits)
}

func (c *Circuit) Define(api frontend.API) error {
	p := c.P
	rc := rangecheck.New(api)

	// ---- (iii) hash(theta) == C_theta ----
	hTheta := mustMiMC(api)
	for j := 0; j < p.K; j++ {
		for t := 0; t < p.D; t++ {
			hTheta.Write(c.Centroids[j][t])
		}
	}
	api.AssertIsEqual(hTheta.Sum(), c.CTheta)

	// ---- (i) Merkle/sponge opening of X against C_X ----
	// All n*d leaves are witnessed (X); compute
	// leaf_i = MiMC(value_i, r_X), fold pairwise up to a single root using
	// MiMC(left,right), matching commit.py's construction exactly (padded
	// to next pow2 with zero leaves -- here we require nd to already be
	// used with a caller-chosen N via padding in the flat leaf list, so the
	// Go CLI passes n*d already padded conceptually by including zero rows
	// if needed; for simplicity this circuit assumes n*d is used directly
	// and the CLI pads instance dimensions to a power of two upstream when
	// exact byte-compat with commit.py's padding is required).
	flatLeaves := make([]frontend.Variable, 0, p.N*p.D)
	for i := 0; i < p.N; i++ {
		for t := 0; t < p.D; t++ {
			flatLeaves = append(flatLeaves, c.X[i][t])
		}
	}
	nd := len(flatLeaves)
	// pad to next pow2 with zero leaves (constants), consistent with
	// commit.py's _next_pow2 padding.
	N := 1
	for N < nd {
		N *= 2
	}
	for len(flatLeaves) < N {
		flatLeaves = append(flatLeaves, frontend.Variable(0))
	}
	level := make([]frontend.Variable, N)
	for i, v := range flatLeaves {
		lh := mustMiMC(api)
		lh.Write(v, c.RX)
		level[i] = lh.Sum()
	}
	for len(level) > 1 {
		next := make([]frontend.Variable, len(level)/2)
		for i := 0; i < len(level); i += 2 {
			nh := mustMiMC(api)
			nh.Write(level[i], level[i+1])
			next[i/2] = nh.Sum()
		}
		level = next
	}
	api.AssertIsEqual(level[0], c.CX)

	// ---- (ii) hash(sparse list) == C_A / (iv) sparse-list validity ----
	if p.AuthenticateA {
		hA := mustMiMC(api)
		if p.SparseSelf {
			for t := 0; t < p.D; t++ {
				for l := 0; l < p.Q; l++ {
					hA.Write(c.SparseRows[t][l], c.SparseSigns[t][l])
				}
			}
		} else {
			for r := 0; r < p.M; r++ {
				for t := 0; t < p.D; t++ {
					hA.Write(c.DenseA[r][t])
				}
			}
		}
		api.AssertIsEqual(hA.Sum(), c.CA)
	}

	if p.SparseSelf && !p.AConstant {
		for t := 0; t < p.D; t++ {
			for l := 0; l < p.Q; l++ {
				row := c.SparseRows[t][l]
				sign := c.SparseSigns[t][l]
				// row < m
				diff := api.Sub(frontend.Variable(p.M), row)
				rc.Check(api.Sub(diff, 1), bitsFor(p.M))
				// sign in {-1,+1}: (sign-1)*(sign+1) == 0
				api.AssertIsEqual(api.Mul(api.Sub(sign, 1), api.Add(sign, 1)), 0)
			}
			// distinct rows within the column (pairwise inequality)
			for a := 0; a < p.Q; a++ {
				for b := a + 1; b < p.Q; b++ {
					d := api.Sub(c.SparseRows[t][a], c.SparseRows[t][b])
					api.AssertIsDifferent(d, 0)
				}
			}
		}
	}

	// ---- (v) projected centroids (once) ----
	projC := make([][]frontend.Variable, p.K)
	for j := 0; j < p.K; j++ {
		projC[j] = c.project(api, c.Centroids[j])
	}

	// ---- per-point: projection, range checks, distances, argmin ----
	rangeBits := p.RangeBits
	if rangeBits == 0 {
		rangeBits = 254
	}
	psiSum := frontend.Variable(0)
	for i := 0; i < p.N; i++ {
		zi := c.project(api, c.X[i])
		// range check each projected coordinate (offset to non-negative by
		// adding 2^(rangeBits-1), a conservative signed-range check)
		offset := int64(1) << uint(rangeBits-1)
		for _, zc := range zi {
			c.rangeCheckSigned(api, rc, api.Add(zc, offset), rangeBits)
		}

		// squared distances to each centroid
		dist := make([]frontend.Variable, p.K)
		for j := 0; j < p.K; j++ {
			acc := frontend.Variable(0)
			for r := 0; r < p.M; r++ {
				diff := api.Sub(zi[r], projC[j][r])
				acc = api.Add(acc, api.Mul(diff, diff))
			}
			dist[j] = acc
		}

		// lowest-index argmin tournament: best starts at index 0; for j>0,
		// update iff dist[j] < bestDist (strict), matching
		// clustering.py's np.argmin (first occurrence of the minimum wins).
		// Distances are non-negative and bounded by Psi_max per
		// quantize.py; absDiffUpp = 2^(2*rangeBits) is a conservative
		// upper bound on |dist[j]-bestDist|.
		absDiffBits := 2*rangeBits + 4
		absDiffUpp := new(big.Int).Lsh(big.NewInt(1), uint(absDiffBits))
		bc := cmp.NewBoundedComparator(api, absDiffUpp, false)
		bestDist := dist[0]
		bestIdx := frontend.Variable(0)
		for j := 1; j < p.K; j++ {
			lt := bc.IsLess(dist[j], bestDist)
			bestDist = api.Select(lt, dist[j], bestDist)
			bestIdx = api.Select(lt, frontend.Variable(j), bestIdx)
		}
		api.AssertIsEqual(bestIdx, c.Assign[i])
		psiSum = api.Add(psiSum, bestDist)
	}
	api.AssertIsEqual(psiSum, c.PsiHat)

	return nil
}

func bitsFor(n int) int {
	b := 1
	for (1 << uint(b)) < n+1 {
		b++
	}
	return b
}

func (v Variant) String() string { return string(v) }

func VariantFromString(s string) (Variant, error) {
	switch Variant(s) {
	case VariantFull, VariantSparse, VariantDense, VariantCoord, VariantPCAPublic, VariantPisot, VariantPrepublished, VariantIncircuitGen:
		return Variant(s), nil
	default:
		return "", fmt.Errorf("unknown variant %q", s)
	}
}
