package circuit

// Stronger assignment certificates from within-center distance differences.
// Keep the existing absolute cost interval and approximate computation.
import (
	"fmt"
	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/std/math/cmp"
	"github.com/consensys/gnark/std/rangecheck"
	"math/big"
)

type BoundRelativeReuse struct {
	Absolute      BoundCertifiedReuseQuery
	RelativeCount frontend.Variable     `gnark:",public"`
	Coeff         [][]frontend.Variable `gnark:",public"`
	Constant      []frontend.Variable   `gnark:",public"`
	PairRadius    [][]frontend.Variable `gnark:",public"`
}

// Derived once from approved centers. Fields are private so callers cannot
// substitute unverified coefficients or bounds into the prepared builders.
type RelativeReuseParameters struct {
	modelHash           string
	heads               [][]*big.Int
	offs                []*big.Int
	z, pairs            *PairedParameters
	gram, absoluteLimit *big.Int
}

func NewRelativeReuseParameters(m BoundModel, centers [][]int64) (*RelativeReuseParameters, error) {
	if err := m.Validate(); err != nil {
		return nil, err
	}
	if err := validateNewCenters(m, centers); err != nil {
		return nil, err
	}
	heads, offs, err := ReuseCenterStats(m, centers)
	if err != nil {
		return nil, err
	}
	z, pairs, err := relativeParameters(m, centers)
	if err != nil {
		return nil, err
	}
	_, _, limit := certificateBounds(m)
	return &RelativeReuseParameters{modelHash: m.Hash().String(), heads: heads, offs: offs, z: z, pairs: pairs, gram: m.GramDefect(), absoluteLimit: limit}, nil
}

func NewBoundRelativeReuse(m BoundModel, q, k int) *BoundRelativeReuse {
	c := &BoundRelativeReuse{Absolute: *NewBoundCertifiedReuseQuery(m, q, k), Coeff: make([][]frontend.Variable, k), Constant: make([]frontend.Variable, k), PairRadius: make([][]frontend.Variable, k)}
	for j := 0; j < k; j++ {
		c.Coeff[j] = make([]frontend.Variable, m.Width())
		c.PairRadius[j] = make([]frontend.Variable, k)
	}
	return c
}
func (c *BoundRelativeReuse) Define(api frontend.API) error {
	if err := c.Absolute.Define(api); err != nil {
		return err
	}
	q := &c.Absolute.Query
	m := q.M
	radius, co, constant, limit := pairedLimits(m)
	if mul(limit, bi(int64(32*(q.Q+q.KNew)))).Cmp(api.Compiler().Field()) >= 0 {
		return fmt.Errorf("unsafe relative-certificate field bound")
	}
	rc := rangecheck.New(api)
	bc := cmp.NewBoundedComparator(api, limit, false)
	t2 := square(bi(m.BasisScale))
	for j := range c.Coeff {
		for _, v := range c.Coeff[j] {
			rangeCheckSigned(api, rc, v, co, false)
		}
		rangeCheckSigned(api, rc, c.Constant[j], constant, false)
		for _, v := range c.PairRadius[j] {
			rangeCheckUnsigned(api, rc, v, mul(bi(2), radius), false)
		}
	}
	count := frontend.Variable(0)
	for i, h := range q.H {
		scores := make([]frontend.Variable, q.KNew)
		selected := frontend.Variable(0)
		winner := make([]frontend.Variable, q.KNew)
		for j := range scores {
			score := c.Constant[j]
			for a, v := range h {
				score = api.Add(score, api.Mul(c.Coeff[j][a], v))
			}
			scores[j] = score
			winner[j] = api.IsZero(api.Sub(q.Assign[i], j))
			selected = api.Add(selected, api.Mul(winner[j], score))
		}
		valid := frontend.Variable(1)
		for j := range scores {
			r := frontend.Variable(0)
			for w, mask := range winner {
				r = api.Add(r, api.Mul(mask, c.PairRadius[j][w]))
			}
			error := api.Mul(2, t2, c.Absolute.RowRadii[i], r)
			strict := bc.IsLess(error, api.Sub(scores[j], selected))
			valid = api.Mul(valid, api.Select(winner[j], 1, strict))
		}
		count = api.Add(count, valid)
	}
	api.AssertIsEqual(c.RelativeCount, count)
	return nil
}
func relativeParameters(m BoundModel, centers [][]int64) (*PairedParameters, *PairedParameters, error) {
	anchor := [][]int64{m.Centers[0], m.Centers[0]}
	z, e := PairedChangeParameters(m, anchor, centers)
	if e != nil {
		return nil, nil, e
	}
	p, e := PairedChangeParameters(m, centers, centers)
	return z, p, e
}
func fillRelativeParameters(c *BoundRelativeReuse, params *RelativeReuseParameters) {
	z, p := params.z, params.pairs
	k := len(params.heads)
	for j := 0; j < k; j++ {
		for a, v := range z.Coeff[2*j] {
			c.Coeff[j][a] = v
		}
		c.Constant[j] = z.Constant[2*j]
		for w := 0; w < k; w++ {
			c.PairRadius[j][w] = p.Radius[j*k+w]
		}
	}
}
func FillBoundRelativeReuse(p *BoundPrepared, centers [][]int64, indices []int) (*BoundRelativeReuse, error) {
	params, err := NewRelativeReuseParameters(p.M, centers)
	if err != nil {
		return nil, err
	}
	return FillBoundRelativeReusePrepared(p, params, indices)
}

func FillBoundRelativeReusePrepared(p *BoundPrepared, params *RelativeReuseParameters, indices []int) (*BoundRelativeReuse, error) {
	if params == nil || params.modelHash != p.M.Hash().String() {
		return nil, fmt.Errorf("relative parameter model mismatch")
	}
	base, ref, err := fillBoundCertifiedReuseWithGram(p, params.heads, params.offs, indices, params.gram)
	if err != nil {
		return nil, err
	}
	k := len(params.heads)
	c := NewBoundRelativeReuse(p.M, len(indices), k)
	c.Absolute = *base
	fillRelativeParameters(c, params)
	count := 0
	t2 := square(bi(p.M.BasisScale))
	for i, idx := range indices {
		z := make([]*big.Int, k)
		for j := range z {
			z[j] = new(big.Int).Set(c.Constant[j].(*big.Int))
			for a, v := range p.H[idx] {
				z[j].Add(z[j], mul(c.Coeff[j][a].(*big.Int), v))
			}
		}
		w := ref.Labels[i]
		valid := true
		for j := range z {
			if j == w {
				continue
			}
			error := mul(mul(mul(bi(2), t2), ref.Radii[i]), c.PairRadius[j][w].(*big.Int))
			if new(big.Int).Sub(z[j], z[w]).Cmp(error) <= 0 {
				valid = false
			}
		}
		if valid {
			count++
		}
	}
	c.RelativeCount = count
	return c, nil
}
func BoundRelativeReusePublic(m BoundModel, s BoundStatement, centers [][]int64, indices []int, psi, margin string, absoluteCount, relativeCount int, lower, upper string) (*BoundRelativeReuse, error) {
	params, err := NewRelativeReuseParameters(m, centers)
	if err != nil {
		return nil, err
	}
	return BoundRelativeReusePublicPrepared(m, s, params, indices, psi, margin, absoluteCount, relativeCount, lower, upper)
}

func BoundRelativeReusePublicPrepared(m BoundModel, s BoundStatement, params *RelativeReuseParameters, indices []int, psi, margin string, absoluteCount, relativeCount int, lower, upper string) (*BoundRelativeReuse, error) {
	if params == nil || params.modelHash != m.Hash().String() {
		return nil, fmt.Errorf("relative parameter model mismatch")
	}
	if relativeCount < 0 || relativeCount > len(indices) {
		return nil, fmt.Errorf("invalid relative certified count")
	}
	query, err := boundReusePublicFromStats(m, s, params.heads, params.offs, indices, psi, margin)
	if err != nil {
		return nil, err
	}
	base, err := certifiedReusePublicFromQuery(query, absoluteCount, lower, upper, params.absoluteLimit)
	if err != nil {
		return nil, err
	}
	c := NewBoundRelativeReuse(m, len(indices), len(params.heads))
	c.Absolute = *base
	c.RelativeCount = relativeCount
	fillRelativeParameters(c, params)
	return c, nil
}
