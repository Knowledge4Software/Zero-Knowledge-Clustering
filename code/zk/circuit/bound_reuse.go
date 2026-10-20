package circuit

// Reusable query: new public centres against an unchanged Stage-1 catalog.
//
// The stored rows (H, E) and the context/model hash are those of the deployed
// model; Stage 1 is not repeated. A new centre c' enters only through public
// statistics computed by the verifier from the published c':
//   CHead_j = H(c'_j) (same integer head as rows) and COff_j = E(c'_j).
// Each row's claimed distance is ||H - CHead_j||^2 + E + COff_j, i.e. the
// head/energy estimate with the residual cross term set to zero. It is exact
// when c'_j lies in the deployed basis span (COff_j = 0 then) and approximate
// otherwise. One circuit/setup serves every centre set with the same K'.
// MarginSum is a soft output: sum over rows of (second-best - best) distance.

import (
	"fmt"
	"math/big"

	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/std/math/cmp"
	"github.com/consensys/gnark/std/rangecheck"
)

type BoundReuseQuery struct {
	M                                       BoundModel            `gnark:"-"`
	Q, KNew                                 int                   `gnark:"-"`
	CX, CH, CE, CT, Context, Psi, MarginSum frontend.Variable     `gnark:",public"`
	Indices                                 []frontend.Variable   `gnark:",public"`
	CHead                                   [][]frontend.Variable `gnark:",public"`
	COff                                    []frontend.Variable   `gnark:",public"`
	H, PH                                   [][]frontend.Variable
	E, Assign                               []frontend.Variable
	Salt                                    frontend.Variable
}

func (m BoundModel) reusable() error {
	if !m.HasEnergy() || m.Method == "affine" || m.Method == "cached" || m.UsesTailQuery() {
		return fmt.Errorf("reuse queries need a basis head with energy")
	}
	return nil
}

// ReuseBounds: |score - shift| < shift, distances and margins are bounded.
func reuseBounds(m BoundModel) (shift, score, distance *big.Int) {
	l := m.Limits()
	w := bi(int64(m.Width()))
	h2 := square(l.H)
	shift = add(mul(mul(bi(6), w), h2), mul(bi(2), l.E))
	score = mul(bi(2), shift)
	distance = add(mul(mul(bi(4), w), h2), mul(bi(2), l.E))
	return
}

func NewBoundReuseQuery(m BoundModel, q, knew int) *BoundReuseQuery {
	c := &BoundReuseQuery{M: m, Q: q, KNew: knew, Indices: make([]frontend.Variable, q),
		CHead: make([][]frontend.Variable, knew), COff: make([]frontend.Variable, knew),
		H: make([][]frontend.Variable, q), PH: make([][]frontend.Variable, q),
		E: make([]frontend.Variable, q), Assign: make([]frontend.Variable, q)}
	for j := range c.CHead {
		c.CHead[j] = make([]frontend.Variable, m.Width())
	}
	for i := 0; i < q; i++ {
		c.H[i] = make([]frontend.Variable, m.Width())
		c.PH[i] = make([]frontend.Variable, m.Depth())
	}
	return c
}

func (c *BoundReuseQuery) Define(api frontend.API) error {
	return c.define(api, nil)
}

func (c *BoundReuseQuery) define(api frontend.API, certificate *BoundCertifiedReuseQuery) error {
	m := c.M
	if err := m.Validate(); err != nil {
		return err
	}
	if err := m.reusable(); err != nil {
		return err
	}
	if c.Q < 1 || c.Q > m.N || c.KNew < 2 || c.KNew > 128 {
		return fmt.Errorf("invalid reuse query shape")
	}
	l := m.Limits()
	shift, scoreBound, distBound := reuseBounds(m)
	if mul(scoreBound, bi(int64(c.Q)*16)).Cmp(api.Compiler().Field()) >= 0 {
		return fmt.Errorf("unsafe aggregate bound")
	}
	rc := rangecheck.New(api)
	boundBindContext(api, m, c.CX, c.CH, c.CE, c.CT, c.Context)
	// Public centre statistics: bounded like row statistics; norms once per centre.
	norms := make([]frontend.Variable, c.KNew)
	for j := range c.CHead {
		norm := frontend.Variable(0)
		for _, v := range c.CHead[j] {
			rangeCheckSigned(api, rc, v, l.H, false)
			norm = api.Add(norm, api.Mul(v, v))
		}
		rangeCheckUnsigned(api, rc, c.COff[j], l.E, false)
		norms[j] = api.Add(norm, c.COff[j])
	}
	var cs *certificateCircuitState
	if certificate != nil {
		var err error
		cs, err = newCertificateCircuitState(api, rc, certificate)
		if err != nil {
			return err
		}
	}
	bc := cmp.NewBoundedComparator(api, scoreBound, false)
	ic := cmp.NewBoundedComparator(api, bi(int64(m.N)), false)
	sum, margins := frontend.Variable(0), frontend.Variable(0)
	for i := 0; i < c.Q; i++ {
		if i > 0 {
			ic.AssertIsLess(c.Indices[i-1], c.Indices[i])
		}
		for _, v := range c.H[i] {
			rangeCheckSigned(api, rc, v, l.H, false)
		}
		rangeCheckUnsigned(api, rc, c.E[i], l.E, false)
		values := append(append([]frontend.Variable{}, c.H[i]...), c.E[i])
		boundOpen(api, m, boundH, c.Indices[i], c.Salt, values, c.PH[i], c.CH)
		var best, second frontend.Variable
		scores := make([]frontend.Variable, c.KNew)
		idx := frontend.Variable(0)
		for j := 0; j < c.KNew; j++ {
			score := frontend.Variable(shift)
			for t, v := range c.H[i] {
				score = api.Add(score, api.Mul(2, api.Mul(c.CHead[j][t], v)))
			}
			score = api.Sub(score, norms[j])
			scores[j] = score
			rangeCheckUnsigned(api, rc, score, scoreBound, false)
			if j == 0 {
				best, second = score, 0
				continue
			}
			above := bc.IsLess(best, score)
			if j == 1 {
				second = api.Select(above, best, score)
			} else {
				aboveSecond := bc.IsLess(second, score)
				second = api.Select(above, best, api.Select(aboveSecond, score, second))
			}
			best = api.Select(above, score, best)
			idx = api.Select(above, j, idx)
		}
		api.AssertIsEqual(c.Assign[i], idx)
		norm := frontend.Variable(0)
		for _, v := range c.H[i] {
			norm = api.Add(norm, api.Mul(v, v))
		}
		distance := api.Add(api.Add(norm, c.E[i]), api.Sub(shift, best))
		rangeCheckUnsigned(api, rc, distance, distBound, false)
		margin := api.Sub(best, second)
		rangeCheckUnsigned(api, rc, margin, scoreBound, false)
		sum = api.Add(sum, distance)
		margins = api.Add(margins, margin)
		if cs != nil {
			cs.row(api, rc, i, norm, best, scores)
		}
	}
	api.AssertIsEqual(c.Psi, sum)
	api.AssertIsEqual(c.MarginSum, margins)
	if cs != nil {
		api.AssertIsEqual(certificate.CertifiedCount, cs.count)
		api.AssertIsEqual(certificate.CostLower, cs.lower)
		api.AssertIsEqual(certificate.CostUpper, cs.upper)
	}
	return nil
}

// ReuseCenterStats computes the public statistics of new centres (quantized
// with the deployed input scale) in the deployed model's integer units.
func ReuseCenterStats(m BoundModel, centers [][]int64) (heads [][]*big.Int, offs []*big.Int, err error) {
	if err = m.reusable(); err != nil {
		return
	}
	limit := int64(1) << uint(m.InputBits)
	for _, row := range centers {
		if len(row) != m.D {
			return nil, nil, fmt.Errorf("new centre width mismatch")
		}
		for _, v := range row {
			if v < -limit || v > limit {
				return nil, nil, fmt.Errorf("new centre outside input range")
			}
		}
		h, e, _ := BoundStats(m, row)
		heads, offs = append(heads, h), append(offs, e)
	}
	return
}

// ReuseAnswer is the native reference for BoundReuseQuery (first-index ties).
func (p *BoundPrepared) ReuseAnswer(heads [][]*big.Int, offs []*big.Int, indices []int) ([]int, *big.Int, *big.Int, error) {
	labels := make([]int, len(indices))
	sum, margins := bi(0), bi(0)
	for a, i := range indices {
		if i < 0 || i >= p.M.N || (a > 0 && i <= indices[a-1]) {
			return nil, nil, nil, fmt.Errorf("query indices must be strictly increasing and in range")
		}
		var best, second *big.Int
		for j, hc := range heads {
			d := new(big.Int).Set(offs[j])
			for t, v := range p.H[i] {
				d.Add(d, square(new(big.Int).Sub(v, hc[t])))
			}
			if j == 0 || d.Cmp(best) < 0 {
				if j > 0 {
					second = best
				}
				best, labels[a] = d, j
			} else if second == nil || d.Cmp(second) < 0 {
				second = d
			}
		}
		sum.Add(sum, add(best, p.E[i]))
		margins.Add(margins, new(big.Int).Sub(second, best))
	}
	return labels, sum, margins, nil
}

func FillBoundReuseQuery(p *BoundPrepared, heads [][]*big.Int, offs []*big.Int, indices []int) (*BoundReuseQuery, error) {
	labels, psi, margins, err := p.ReuseAnswer(heads, offs, indices)
	if err != nil {
		return nil, err
	}
	c := NewBoundReuseQuery(p.M, len(indices), len(heads))
	c.CX, c.CH, c.CE, c.CT, c.Context, c.Psi, c.MarginSum = p.CX, p.CH, p.CE, p.CT, p.Context, psi, margins
	c.Salt = p.SaltZ
	for j, h := range heads {
		for t, v := range h {
			c.CHead[j][t] = v
		}
		c.COff[j] = offs[j]
	}
	for a, i := range indices {
		c.Indices[a] = i
		c.Assign[a] = labels[a]
		c.E[a] = p.E[i]
		for t, v := range p.H[i] {
			c.H[a][t] = v
		}
		for t, v := range p.TH.Path(i) {
			c.PH[a][t] = v
		}
	}
	return c, nil
}
