package circuit

// Stage-2 (per-query) relation over an already committed compressed matrix Z.
//
// Stage 1 (not part of this relation, proved once) binds Z = f(X) to the
// committed data. Every later query proves its answer from Z alone:
//
//	public:  CZA (root of the assignment block), CZR (root of the remaining
//	         columns, cost queries only), CTheta and centres (or centres
//	         compiled in as constants), PsiHat (cost queries)
//	witness: Z (n x W), salt RZ, assignments
//	constraints:
//	  (i)   Merkle opening of each needed block of Z (MiMC, same leaf/node
//	        construction as the Stage-0 data commitment in circuit.go)
//	  (ii)  signed range checks on the opened coordinates and public centres
//	  (iii) first-index nearest centre on the first AssignDims columns
//	  (iv)  cost queries: sum over points of the squared distance to the
//	        assigned centre over ALL W columns equals PsiHat; with EnergyCol
//	        the cost is the head distance plus the committed exact residual
//	        energy in column AssignDims (the sketch tail is not opened)
//
// Methods map onto the same relation: full (Z = X, AssignDims = W = d),
// PCA / Sparse JL (W = m, AssignDims = m) and Protected JL (W = head + tail,
// AssignDims = head rank, since the tail cannot change the nearest centre).

import (
	"fmt"
	"math/big"

	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/std/math/cmp"
	"github.com/consensys/gnark/std/rangecheck"
)

type QueryParams struct {
	N, W, K      int
	AssignDims   int  // leading columns that decide the nearest centre
	CoordBits    int  // every coordinate satisfies |v| <= 2^CoordBits
	Cost         bool // false: assignment query; true: assignment + cost
	ConstCenters bool // centres compiled into the relation as constants
	// EnergyCol: column AssignDims holds the exact residual energy
	// ||(I-UU^T)(x-c_0)||^2 >= 0 (bounded by 2^EnergyBits); cost queries use
	// head distance + energy and open only the head and that column.
	EnergyCol    bool
	EnergyBits   int
	Centers      [][]int64 // K x W, used when ConstCenters
	NoRangeBatch bool
}

type QueryCircuit struct {
	P QueryParams `gnark:"-"`

	CZA     frontend.Variable     `gnark:",public"`
	CZR     frontend.Variable     `gnark:",public"`
	CTheta  frontend.Variable     `gnark:",public"`
	Centers [][]frontend.Variable `gnark:",public"` // K x W unless ConstCenters
	PsiHat  frontend.Variable     `gnark:",public"`

	Z      [][]frontend.Variable `gnark:",secret"`
	RZ     frontend.Variable     `gnark:",secret"`
	Assign []frontend.Variable   `gnark:",secret"`
}

func NewQueryCircuit(p QueryParams) *QueryCircuit {
	c := &QueryCircuit{P: p}
	if !p.ConstCenters {
		c.Centers = make([][]frontend.Variable, p.K)
		for j := range c.Centers {
			c.Centers[j] = make([]frontend.Variable, p.W)
		}
	}
	c.Z = make([][]frontend.Variable, p.N)
	for i := range c.Z {
		c.Z[i] = make([]frontend.Variable, p.W)
	}
	c.Assign = make([]frontend.Variable, p.N)
	return c
}

func (p QueryParams) Validate() error {
	if p.N < 1 || p.W < 1 || p.K < 1 || p.AssignDims < 1 || p.AssignDims > p.W {
		return fmt.Errorf("invalid query dimensions")
	}
	if p.CoordBits < 1 || p.CoordBits > 62 {
		return fmt.Errorf("coordinate bits must be in [1,62]")
	}
	if p.EnergyCol && (p.AssignDims >= p.W || p.EnergyBits < 1 || p.EnergyBits > 120) {
		return fmt.Errorf("energy column requires AssignDims < W and EnergyBits in [1,120]")
	}
	if p.ConstCenters {
		if len(p.Centers) != p.K {
			return fmt.Errorf("constant centres must have K rows")
		}
		bound := int64(1) << uint(p.CoordBits)
		for _, row := range p.Centers {
			if len(row) != p.W {
				return fmt.Errorf("constant centres must have W columns")
			}
			for _, v := range row {
				if v > bound || v < -bound {
					return fmt.Errorf("constant centre outside coordinate bound")
				}
			}
		}
	}
	return nil
}

// merkleRoot hashes MiMC(v, salt) leaves and folds them pairwise; an odd
// node is carried up unchanged, so the cost is exact in the number of leaves
// (no power-of-two padding).
func merkleRoot(api frontend.API, values []frontend.Variable, salt frontend.Variable) frontend.Variable {
	level := make([]frontend.Variable, len(values))
	for i, v := range values {
		h := mustMiMC(api)
		h.Write(v, salt)
		level[i] = h.Sum()
	}
	for len(level) > 1 {
		next := make([]frontend.Variable, 0, (len(level)+1)/2)
		for i := 0; i+1 < len(level); i += 2 {
			h := mustMiMC(api)
			h.Write(level[i], level[i+1])
			next = append(next, h.Sum())
		}
		if len(level)%2 == 1 {
			next = append(next, level[len(level)-1])
		}
		level = next
	}
	return level[0]
}

func (c *QueryCircuit) Define(api frontend.API) error {
	p := c.P
	if err := p.Validate(); err != nil {
		return err
	}
	rc := rangecheck.New(api)
	bound := new(big.Int).Lsh(big.NewInt(1), uint(p.CoordBits))
	needed := p.AssignDims
	if p.Cost && !p.EnergyCol {
		needed = p.W
	}
	energyBound := new(big.Int)
	if p.EnergyCol {
		energyBound.Lsh(big.NewInt(1), uint(p.EnergyBits))
	}
	for i := 0; i < p.N; i++ {
		for r := 0; r < needed; r++ {
			rangeCheckSigned(api, rc, c.Z[i][r], bound, p.NoRangeBatch)
		}
		if p.Cost && p.EnergyCol {
			rangeCheckUnsigned(api, rc, c.Z[i][p.AssignDims], energyBound, p.NoRangeBatch)
		}
	}

	// Centres: public inputs bound by CTheta, or compile-time constants.
	center := func(j, r int) frontend.Variable {
		if p.ConstCenters {
			return p.Centers[j][r]
		}
		return c.Centers[j][r]
	}
	if !p.ConstCenters {
		h := mustMiMC(api)
		for j := 0; j < p.K; j++ {
			for r := 0; r < p.W; r++ {
				rangeCheckSigned(api, rc, c.Centers[j][r], bound, p.NoRangeBatch)
				h.Write(c.Centers[j][r])
			}
		}
		api.AssertIsEqual(h.Sum(), c.CTheta)
	} else {
		api.AssertIsEqual(c.CTheta, 0) // centres are part of the compiled relation
	}
	if !p.Cost || (p.AssignDims == p.W && !p.EnergyCol) {
		api.AssertIsEqual(c.CZR, 0) // no remaining block is opened
	}
	if !p.Cost {
		api.AssertIsEqual(c.PsiHat, 0)
	}

	// (i) open the needed blocks of the committed Z.
	block := func(from, to int) []frontend.Variable {
		out := make([]frontend.Variable, 0, p.N*(to-from))
		for i := 0; i < p.N; i++ {
			out = append(out, c.Z[i][from:to]...)
		}
		return out
	}
	api.AssertIsEqual(merkleRoot(api, block(0, p.AssignDims), c.RZ), c.CZA)
	if p.Cost && p.EnergyCol {
		api.AssertIsEqual(merkleRoot(api, block(p.AssignDims, p.AssignDims+1), c.RZ), c.CZR)
	} else if p.Cost && p.AssignDims < p.W {
		api.AssertIsEqual(merkleRoot(api, block(p.AssignDims, p.W), c.RZ), c.CZR)
	}

	// (iii) nearest centre on the assignment block via affine scores
	// score_j = shift - ||c_j||^2 + 2<z, c_j>, first maximum wins.
	a := int64(p.AssignDims)
	squared := new(big.Int).Mul(bound, bound)
	scoreBound := new(big.Int).Mul(squared, big.NewInt(4*a)) // scores lie in [0, 4 A B^2]
	shift := new(big.Int).Mul(squared, big.NewInt(3*a))
	tailDims := 0
	if p.Cost && !p.EnergyCol {
		tailDims = p.W - p.AssignDims
	}
	distBound := new(big.Int).Mul(squared, big.NewInt(4*int64(p.AssignDims+tailDims)))
	distBound.Add(distBound, energyBound)
	totalBound := new(big.Int).Mul(distBound, big.NewInt(int64(p.N)))
	if new(big.Int).Mul(totalBound, big.NewInt(2)).Cmp(api.Compiler().Field()) >= 0 {
		return fmt.Errorf("aggregate distance bound is unsafe for the field")
	}
	norm := func(j, from, to int) frontend.Variable {
		acc := frontend.Variable(0)
		for r := from; r < to; r++ {
			acc = api.Add(acc, api.Mul(center(j, r), center(j, r)))
		}
		return acc
	}
	headNorm := make([]frontend.Variable, p.K)
	tailNorm := make([]frontend.Variable, p.K)
	for j := 0; j < p.K; j++ {
		headNorm[j] = norm(j, 0, p.AssignDims)
		tailNorm[j] = norm(j, p.AssignDims, p.AssignDims+tailDims)
	}
	bc := cmp.NewBoundedComparator(api, scoreBound, false)
	psi := frontend.Variable(0)
	for i := 0; i < p.N; i++ {
		z := c.Z[i]
		var best, bestTailDot, bestTailNorm frontend.Variable
		idx := frontend.Variable(0)
		for j := 0; j < p.K; j++ {
			score := api.Sub(shift, headNorm[j])
			for r := 0; r < p.AssignDims; r++ {
				score = api.Add(score, api.Mul(2, z[r], center(j, r)))
			}
			rangeCheckUnsigned(api, rc, score, scoreBound, p.NoRangeBatch)
			tailDot := frontend.Variable(0)
			for r := p.AssignDims; r < p.AssignDims+tailDims; r++ {
				tailDot = api.Add(tailDot, api.Mul(z[r], center(j, r)))
			}
			if j == 0 {
				best, bestTailDot, bestTailNorm = score, tailDot, tailNorm[0]
				continue
			}
			greater := bc.IsLess(best, score)
			best = api.Select(greater, score, best)
			idx = api.Select(greater, j, idx)
			if tailDims > 0 {
				bestTailDot = api.Select(greater, tailDot, bestTailDot)
				bestTailNorm = api.Select(greater, tailNorm[j], bestTailNorm)
			}
		}
		api.AssertIsEqual(idx, c.Assign[i])
		if !p.Cost {
			continue
		}
		// (iv) ||z - c_a||^2 = (||z_A||^2 + shift - best) + (||z_T||^2 - 2<z_T,c_aT> + ||c_aT||^2)
		zNorm := frontend.Variable(0)
		for r := 0; r < p.AssignDims+tailDims; r++ {
			zNorm = api.Add(zNorm, api.Mul(z[r], z[r]))
		}
		distance := api.Sub(api.Add(zNorm, shift), best)
		if tailDims > 0 {
			distance = api.Add(api.Sub(distance, api.Mul(2, bestTailDot)), bestTailNorm)
		}
		if p.EnergyCol {
			distance = api.Add(distance, z[p.AssignDims])
		}
		rangeCheckUnsigned(api, rc, distance, distBound, p.NoRangeBatch)
		psi = api.Add(psi, distance)
	}
	if p.Cost {
		rangeCheckUnsigned(api, rc, psi, totalBound, p.NoRangeBatch)
		api.AssertIsEqual(psi, c.PsiHat)
	}
	return nil
}
