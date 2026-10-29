package circuit

// Integer-valid accuracy certificates for the rounded basis. Let B be the
// integer basis, T its scale and g=max row-sum |BB^t-T^2 I|. For every row x
// and center c, D is the existing head/energy estimate, Q=T^4||x-c||^2, and
// h2=||H(x)-H(c)||^2. Then
//   T^2 |D-Q| <= 2 T^2 ceil(sqrt(E(x))) ceil(sqrt(E(c))) + g h2.
// This follows by expanding B^t(BB^t-T^2 I)B, not by assuming that the rounded
// basis is orthonormal. Certificates concern the original QUANTIZED records.

import (
	"fmt"
	"math/big"

	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/std/math/cmp"
)

type BoundCertifiedReuseQuery struct {
	Query                                BoundReuseQuery
	CertifiedCount, CostLower, CostUpper frontend.Variable   `gnark:",public"`
	CenterRadii                          []frontend.Variable `gnark:",public"`
	RowRadii                             []frontend.Variable
}

func NewBoundCertifiedReuseQuery(m BoundModel, q, knew int) *BoundCertifiedReuseQuery {
	return &BoundCertifiedReuseQuery{Query: *NewBoundReuseQuery(m, q, knew),
		CenterRadii: make([]frontend.Variable, knew), RowRadii: make([]frontend.Variable, q)}
}

func ceilSqrt(v *big.Int) *big.Int {
	u := new(big.Int).Sqrt(v)
	if square(u).Cmp(v) < 0 {
		u.Add(u, bi(1))
	}
	return u
}

func (m BoundModel) GramDefect() *big.Int {
	if matrix, ok := boundedGram64(m); ok {
		maximum := int64(0)
		for _, row := range matrix {
			sum := int64(0)
			for _, v := range row {
				if v < 0 {
					v = -v
				}
				sum += v
			}
			if sum > maximum {
				maximum = sum
			}
		}
		return bi(maximum)
	}
	g := bi(0)
	t2 := square(bi(m.BasisScale))
	for i, row := range m.Basis {
		sum := bi(0)
		for j, other := range m.Basis {
			dot := bi(0)
			for t, v := range row {
				dot.Add(dot, mul(bi(v), bi(other[t])))
			}
			if i == j {
				dot.Sub(dot, t2)
			}
			sum.Add(sum, new(big.Int).Abs(dot))
		}
		if sum.Cmp(g) > 0 {
			g.Set(sum)
		}
	}
	return g
}

func certificateBounds(m BoundModel) (radius, delta, limit *big.Int) {
	l := m.Limits()
	t2 := square(bi(m.BasisScale))
	g := m.GramDefect()
	radius = ceilSqrt(l.E)
	delta = add(mul(mul(bi(2), square(radius)), t2), mul(g, mul(bi(int64(4*m.Width())), square(l.H))))
	_, score, distance := reuseBounds(m)
	limit = mul(bi(4), add(add(mul(add(score, distance), t2), mul(bi(2), delta)), bi(1)))
	return
}

type certificateCircuitState struct {
	cert                       *BoundCertifiedReuseQuery
	comparison                 *cmp.BoundedComparator
	g, t2, radiusMax, deltaMax *big.Int
	count, lower, upper        frontend.Variable
}

func newCertificateCircuitState(api frontend.API, rc frontend.Rangechecker, c *BoundCertifiedReuseQuery) (*certificateCircuitState, error) {
	m := c.Query.M
	radius, delta, limit := certificateBounds(m)
	if mul(limit, bi(int64(16*(c.Query.Q+c.Query.KNew)))).Cmp(api.Compiler().Field()) >= 0 {
		return nil, fmt.Errorf("unsafe certificate field bound")
	}
	s := &certificateCircuitState{cert: c, comparison: cmp.NewBoundedComparator(api, limit, false),
		g: m.GramDefect(), t2: square(bi(m.BasisScale)), radiusMax: radius, deltaMax: delta, count: 0, lower: 0, upper: 0}
	rootCmp := cmp.NewBoundedComparator(api, add(square(radius), bi(1)), false)
	for j, u := range c.CenterRadii {
		rangeCheckUnsigned(api, rc, u, radius, false)
		rootCmp.AssertIsLess(c.Query.COff[j], api.Add(api.Mul(u, u), 1))
	}
	return s, nil
}

func (s *certificateCircuitState) row(api frontend.API, rc frontend.Rangechecker, i int, norm, best frontend.Variable, scores []frontend.Variable) {
	c := s.cert
	q := &c.Query
	u := c.RowRadii[i]
	rangeCheckUnsigned(api, rc, u, s.radiusMax, false)
	cmp.NewBoundedComparator(api, add(square(s.radiusMax), bi(1)), false).AssertIsLess(q.E[i], api.Add(api.Mul(u, u), 1))
	deltas := make([]frontend.Variable, q.KNew)
	distances := make([]frontend.Variable, q.KNew)
	winnerDelta := frontend.Variable(0)
	shift, _, _ := reuseBounds(q.M)
	for j, score := range scores {
		d := api.Add(norm, q.E[i], api.Sub(shift, score))
		h2 := api.Sub(d, q.E[i], q.COff[j])
		deltas[j] = api.Add(api.Mul(2, s.t2, u, c.CenterRadii[j]), api.Mul(s.g, h2))
		distances[j] = api.Mul(s.t2, d)
		winnerDelta = api.Add(winnerDelta, api.Mul(api.IsZero(api.Sub(q.Assign[i], j)), deltas[j]))
	}
	valid := frontend.Variable(1)
	var lo, hi frontend.Variable
	for j := range scores {
		gap := api.Mul(s.t2, api.Sub(best, scores[j]))
		strict := s.comparison.IsLess(api.Add(winnerDelta, deltas[j]), gap)
		winner := api.IsZero(api.Sub(q.Assign[i], j))
		valid = api.Mul(valid, api.Select(winner, 1, strict))
		l := api.Sub(api.Add(distances[j], s.deltaMax), deltas[j])
		h := api.Add(distances[j], deltas[j])
		if j == 0 {
			lo, hi = l, h
		} else {
			lo = api.Select(s.comparison.IsLess(l, lo), l, lo)
			hi = api.Select(s.comparison.IsLess(h, hi), h, hi)
		}
	}
	lo = api.Select(s.comparison.IsLess(lo, s.deltaMax), 0, api.Sub(lo, s.deltaMax))
	s.count = api.Add(s.count, valid)
	s.lower = api.Add(s.lower, lo)
	s.upper = api.Add(s.upper, hi)
}

func (c *BoundCertifiedReuseQuery) Define(api frontend.API) error { return c.Query.define(api, c) }

// NativeCertificate includes private flags only for numerical diagnostics.
// Public statements reveal a count and an interval, not these per-row flags.
type NativeCertificate struct {
	Labels                    []int
	Flags                     []bool
	Count                     int
	Psi, Margin, Lower, Upper *big.Int
	Radii                     []*big.Int
}

func (p *BoundPrepared) ReuseCertificate(heads [][]*big.Int, offs []*big.Int, indices []int) (*NativeCertificate, error) {
	return p.reuseCertificateWithGram(heads, offs, indices, nil)
}

// A cached Gram defect is only supplied by model-pinned parameter builders.
func (p *BoundPrepared) reuseCertificateWithGram(heads [][]*big.Int, offs []*big.Int, indices []int, gram *big.Int) (*NativeCertificate, error) {
	if err := p.M.Validate(); err != nil {
		return nil, err
	}
	if err := p.M.reusable(); err != nil {
		return nil, err
	}
	if err := validatePublicIndices(p.M, indices); err != nil {
		return nil, err
	}
	if len(heads) < 2 || len(heads) != len(offs) {
		return nil, fmt.Errorf("invalid certificate center statistics")
	}
	labels, psi, margin, err := p.ReuseAnswer(heads, offs, indices)
	if err != nil {
		return nil, err
	}
	r := &NativeCertificate{Labels: labels, Flags: make([]bool, len(indices)), Psi: psi, Margin: margin, Lower: bi(0), Upper: bi(0), Radii: make([]*big.Int, len(indices))}
	g := gram
	if g == nil {
		g = p.M.GramDefect()
	}
	t2 := square(bi(p.M.BasisScale))
	cr := make([]*big.Int, len(offs))
	for j, e := range offs {
		cr[j] = ceilSqrt(e)
	}
	for a, i := range indices {
		u := ceilSqrt(p.E[i])
		r.Radii[a] = u
		ds := make([]*big.Int, len(heads))
		errors := make([]*big.Int, len(heads))
		var lo, hi *big.Int
		for j, hc := range heads {
			h2 := bi(0)
			for t, v := range p.H[i] {
				h2.Add(h2, square(new(big.Int).Sub(v, hc[t])))
			}
			ds[j] = mul(add(add(h2, p.E[i]), offs[j]), t2)
			errors[j] = add(mul(mul(mul(bi(2), u), cr[j]), t2), mul(g, h2))
			l := new(big.Int).Sub(ds[j], errors[j])
			h := add(ds[j], errors[j])
			if j == 0 || l.Cmp(lo) < 0 {
				lo = l
			}
			if j == 0 || h.Cmp(hi) < 0 {
				hi = h
			}
		}
		if lo.Sign() < 0 {
			lo = bi(0)
		}
		r.Lower.Add(r.Lower, lo)
		r.Upper.Add(r.Upper, hi)
		winner := labels[a]
		valid := true
		for j := range heads {
			if j != winner && new(big.Int).Sub(ds[j], ds[winner]).Cmp(add(errors[j], errors[winner])) <= 0 {
				valid = false
			}
		}
		r.Flags[a] = valid
		if valid {
			r.Count++
		}
	}
	return r, nil
}

func FillBoundCertifiedReuseQuery(p *BoundPrepared, heads [][]*big.Int, offs []*big.Int, indices []int) (*BoundCertifiedReuseQuery, *NativeCertificate, error) {
	return fillBoundCertifiedReuseWithGram(p, heads, offs, indices, nil)
}

func fillBoundCertifiedReuseWithGram(p *BoundPrepared, heads [][]*big.Int, offs []*big.Int, indices []int, gram *big.Int) (*BoundCertifiedReuseQuery, *NativeCertificate, error) {
	r, err := p.reuseCertificateWithGram(heads, offs, indices, gram)
	if err != nil {
		return nil, nil, err
	}
	query, err := FillBoundReuseQuery(p, heads, offs, indices)
	if err != nil {
		return nil, nil, err
	}
	c := NewBoundCertifiedReuseQuery(p.M, len(indices), len(heads))
	c.Query = *query
	c.CertifiedCount, c.CostLower, c.CostUpper = r.Count, r.Lower, r.Upper
	for j, e := range offs {
		c.CenterRadii[j] = ceilSqrt(e)
	}
	for a, u := range r.Radii {
		c.RowRadii[a] = u
	}
	return c, r, nil
}

func BoundCertifiedReusePublic(m BoundModel, s BoundStatement, centers [][]int64, indices []int, psi, margin string, count int, lower, upper string) (*BoundCertifiedReuseQuery, error) {
	q, err := BoundReusePublic(m, s, centers, indices, psi, margin)
	if err != nil {
		return nil, err
	}
	_, _, limit := certificateBounds(m)
	return certifiedReusePublicFromQuery(q, count, lower, upper, limit)
}

func certifiedReusePublicFromQuery(q *BoundReuseQuery, count int, lower, upper string, limit *big.Int) (*BoundCertifiedReuseQuery, error) {
	m, indices := q.M, q.Indices
	if count < 0 || count > len(indices) {
		return nil, fmt.Errorf("invalid certified count")
	}
	lo, err := canonicalPublicInt(lower)
	if err != nil {
		return nil, err
	}
	hi, err := canonicalPublicInt(upper)
	if err != nil {
		return nil, err
	}
	if lo.Cmp(hi) > 0 || hi.Cmp(mul(limit, bi(int64(len(indices))))) > 0 {
		return nil, fmt.Errorf("invalid certified cost interval")
	}
	c := NewBoundCertifiedReuseQuery(m, len(indices), len(q.CHead))
	c.Query = *q
	c.CertifiedCount, c.CostLower, c.CostUpper = count, lo, hi
	for j, v := range q.COff {
		c.CenterRadii[j] = ceilSqrt(v.(*big.Int))
	}
	return c, nil
}
