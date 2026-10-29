package circuit

// Paired cost-change bounds cancel the shared row residual and evaluate the
// rounded-basis Gram correction exactly. Public coefficients are independently
// reconstructed from approved old/new centers by the standalone verifier.
import (
	"fmt"
	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/std/math/cmp"
	"github.com/consensys/gnark/std/rangecheck"
	"math/big"
)

type PairedParameters struct {
	Coeff            [][]*big.Int
	Constant, Radius []*big.Int
	Old, New         int
	modelHash        string
	limit            *big.Int
}
type NativePairedChange struct {
	Lower, Upper *big.Int
	Radii        []*big.Int
}

func pairedResidual(m BoundModel, x []int64, head []*big.Int) []*big.Int {
	r := make([]*big.Int, m.D)
	t := bi(m.BasisScale)
	t2 := square(t)
	for j := range r {
		r[j] = mul(t2, bi(x[j]-m.Centers[0][j]))
		for a := range m.Basis {
			r[j].Sub(r[j], mul(bi(m.Basis[a][j]), new(big.Int).Quo(head[a], t)))
		}
	}
	return r
}

func PairedChangeParameters(m BoundModel, old, next [][]int64) (*PairedParameters, error) {
	if err := m.Validate(); err != nil {
		return nil, err
	}
	if err := m.reusable(); err != nil {
		return nil, err
	}
	if len(old) < 2 || len(next) < 2 || len(old) > 128 || len(next) > 128 {
		return nil, fmt.Errorf("invalid paired center counts")
	}
	ho, eo, err := ReuseCenterStats(m, old)
	if err != nil {
		return nil, err
	}
	hn, en, err := ReuseCenterStats(m, next)
	if err != nil {
		return nil, err
	}
	rank := m.Width()
	t2 := square(bi(m.BasisScale))
	gram := pairedGramMatrix(m)
	quadratic := func(h []*big.Int) *big.Int {
		v := bi(0)
		for a := range h {
			for j := range h {
				v.Add(v, mul(mul(h[a], gram[a][j]), h[j]))
			}
		}
		return v
	}
	norm := func(h []*big.Int) *big.Int {
		v := bi(0)
		for _, x := range h {
			v.Add(v, square(x))
		}
		return v
	}
	ro := make([][]*big.Int, len(old))
	rn := make([][]*big.Int, len(next))
	for a := range old {
		ro[a] = pairedResidual(m, old[a], ho[a])
	}
	for a := range next {
		rn[a] = pairedResidual(m, next[a], hn[a])
	}
	p := &PairedParameters{Old: len(old), New: len(next)}
	for a := range next {
		for j := range old {
			dh := make([]*big.Int, rank)
			for z := range dh {
				dh[z] = new(big.Int).Sub(hn[a][z], ho[j][z])
			}
			co := make([]*big.Int, rank)
			for z := range co {
				v := bi(0)
				for y := range dh {
					v.Add(v, mul(gram[z][y], dh[y]))
				}
				v.Sub(v, mul(t2, dh[z]))
				co[z] = mul(bi(2), v)
			}
			v := new(big.Int).Sub(norm(hn[a]), norm(ho[j]))
			v.Add(v, en[a])
			v.Sub(v, eo[j])
			v.Mul(v, t2)
			v.Sub(v, new(big.Int).Sub(quadratic(hn[a]), quadratic(ho[j])))
			e := bi(0)
			for d := range rn[a] {
				e.Add(e, square(new(big.Int).Sub(rn[a][d], ro[j][d])))
			}
			p.Coeff = append(p.Coeff, co)
			p.Constant = append(p.Constant, v)
			p.Radius = append(p.Radius, ceilSqrt(e))
		}
	}
	p.modelHash = m.Hash().String()
	_, _, _, p.limit = pairedLimits(m)
	return p, nil
}

func pairedLimits(m BoundModel) (radius, coeff, constant, limit *big.Int) {
	l := m.Limits()
	t2 := square(bi(m.BasisScale))
	g := m.GramDefect()
	rank := bi(int64(m.Width()))
	radius = ceilSqrt(l.E)
	coeff = mul(mul(bi(4), l.H), add(g, t2))
	constant = add(mul(t2, add(mul(mul(bi(2), rank), square(l.H)), mul(bi(2), l.E))), mul(mul(mul(bi(2), g), rank), square(l.H)))
	limit = add(add(mul(mul(rank, l.H), coeff), constant), add(mul(mul(bi(4), t2), square(radius)), bi(1)))
	return
}

type BoundPairedChange struct {
	M                                     BoundModel            `gnark:"-"`
	Q, KOld, KNew                         int                   `gnark:"-"`
	CX, CH, CE, CT, Context, Lower, Upper frontend.Variable     `gnark:",public"`
	Indices                               []frontend.Variable   `gnark:",public"`
	Coeff                                 [][]frontend.Variable `gnark:",public"`
	Constant, DeltaRadius                 []frontend.Variable   `gnark:",public"`
	H, PH                                 [][]frontend.Variable
	E, RowRadius                          []frontend.Variable
	Salt                                  frontend.Variable
}

func NewBoundPairedChange(m BoundModel, q, ko, kn int) *BoundPairedChange {
	c := &BoundPairedChange{M: m, Q: q, KOld: ko, KNew: kn, Indices: make([]frontend.Variable, q), Coeff: make([][]frontend.Variable, ko*kn), Constant: make([]frontend.Variable, ko*kn), DeltaRadius: make([]frontend.Variable, ko*kn), H: make([][]frontend.Variable, q), PH: make([][]frontend.Variable, q), E: make([]frontend.Variable, q), RowRadius: make([]frontend.Variable, q)}
	for p := range c.Coeff {
		c.Coeff[p] = make([]frontend.Variable, m.Width())
	}
	for i := range c.H {
		c.H[i] = make([]frontend.Variable, m.Width())
		c.PH[i] = make([]frontend.Variable, m.Depth())
	}
	return c
}

func (c *BoundPairedChange) Define(api frontend.API) error {
	m := c.M
	if err := m.Validate(); err != nil {
		return err
	}
	if err := m.reusable(); err != nil {
		return err
	}
	if c.Q < 1 || c.Q > m.N || c.KOld < 2 || c.KOld > 128 || c.KNew < 2 || c.KNew > 128 {
		return fmt.Errorf("invalid paired circuit shape")
	}
	radius, coeff, constant, limit := pairedLimits(m)
	if mul(limit, bi(int64(32*(c.Q+c.KOld+c.KNew)))).Cmp(api.Compiler().Field()) >= 0 {
		return fmt.Errorf("unsafe paired field bound")
	}
	rc := rangecheck.New(api)
	bc := cmp.NewBoundedComparator(api, mul(bi(4), limit), false)
	ic := cmp.NewBoundedComparator(api, bi(int64(m.N)), false)
	l := m.Limits()
	t2 := square(bi(m.BasisScale))
	boundBindContext(api, m, c.CX, c.CH, c.CE, c.CT, c.Context)
	for p := range c.Coeff {
		for _, v := range c.Coeff[p] {
			rangeCheckSigned(api, rc, v, coeff, false)
		}
		rangeCheckSigned(api, rc, c.Constant[p], constant, false)
		rangeCheckUnsigned(api, rc, c.DeltaRadius[p], mul(bi(2), radius), false)
	}
	lower, upper := frontend.Variable(0), frontend.Variable(0)
	for i := 0; i < c.Q; i++ {
		if i > 0 {
			ic.AssertIsLess(c.Indices[i-1], c.Indices[i])
		}
		for _, v := range c.H[i] {
			rangeCheckSigned(api, rc, v, l.H, false)
		}
		rangeCheckUnsigned(api, rc, c.E[i], l.E, false)
		rangeCheckUnsigned(api, rc, c.RowRadius[i], radius, false)
		cmp.NewBoundedComparator(api, add(square(radius), bi(1)), false).AssertIsLess(c.E[i], api.Add(api.Mul(c.RowRadius[i], c.RowRadius[i]), 1))
		values := append(append([]frontend.Variable{}, c.H[i]...), c.E[i])
		boundOpen(api, m, boundH, c.Indices[i], c.Salt, values, c.PH[i], c.CH)
		lows := make([]frontend.Variable, c.KOld*c.KNew)
		highs := make([]frontend.Variable, len(lows))
		for p := range lows {
			score := c.Constant[p]
			for z, v := range c.H[i] {
				score = api.Add(score, api.Mul(v, c.Coeff[p][z]))
			}
			err := api.Mul(2, t2, c.RowRadius[i], c.DeltaRadius[p])
			lows[p] = api.Add(limit, api.Sub(score, err))
			highs[p] = api.Add(limit, score, err)
		}
		var rowLo, rowHi frontend.Variable
		for a := 0; a < c.KNew; a++ {
			v := lows[a*c.KOld]
			for b := 1; b < c.KOld; b++ {
				x := lows[a*c.KOld+b]
				v = api.Select(bc.IsLess(v, x), x, v)
			}
			if a == 0 {
				rowLo = v
			} else {
				rowLo = api.Select(bc.IsLess(v, rowLo), v, rowLo)
			}
		}
		for b := 0; b < c.KOld; b++ {
			v := highs[b]
			for a := 1; a < c.KNew; a++ {
				x := highs[a*c.KOld+b]
				v = api.Select(bc.IsLess(x, v), x, v)
			}
			if b == 0 {
				rowHi = v
			} else {
				rowHi = api.Select(bc.IsLess(rowHi, v), v, rowHi)
			}
		}
		lower = api.Add(lower, api.Sub(rowLo, limit))
		upper = api.Add(upper, api.Sub(rowHi, limit))
	}
	api.AssertIsEqual(c.Lower, lower)
	api.AssertIsEqual(c.Upper, upper)
	return nil
}

func pairedPublicParameters(c *BoundPairedChange, p *PairedParameters) {
	for a := range p.Coeff {
		for z, v := range p.Coeff[a] {
			c.Coeff[a][z] = v
		}
		c.Constant[a] = p.Constant[a]
		c.DeltaRadius[a] = p.Radius[a]
	}
}

func FillBoundPairedChange(p *BoundPrepared, old, next [][]int64, indices []int) (*BoundPairedChange, *NativePairedChange, error) {
	if err := validatePublicIndices(p.M, indices); err != nil {
		return nil, nil, err
	}
	params, err := PairedChangeParameters(p.M, old, next)
	if err != nil {
		return nil, nil, err
	}
	return FillBoundPairedChangePrepared(p, params, indices)
}

// Parameters must come from PairedChangeParameters for the approved centers.
func FillBoundPairedChangePrepared(p *BoundPrepared, params *PairedParameters, indices []int) (*BoundPairedChange, *NativePairedChange, error) {
	if params == nil || params.modelHash != p.M.Hash().String() {
		return nil, nil, fmt.Errorf("paired parameter model mismatch")
	}
	if err := validatePublicIndices(p.M, indices); err != nil {
		return nil, nil, err
	}
	c := NewBoundPairedChange(p.M, len(indices), params.Old, params.New)
	pairedPublicParameters(c, params)
	c.CX, c.CH, c.CE, c.CT, c.Context, c.Salt = p.CX, p.CH, p.CE, p.CT, p.Context, p.SaltZ
	result := &NativePairedChange{Lower: bi(0), Upper: bi(0), Radii: make([]*big.Int, len(indices))}
	t2 := square(bi(p.M.BasisScale))
	for a, i := range indices {
		c.Indices[a] = i
		c.E[a] = p.E[i]
		c.RowRadius[a] = ceilSqrt(p.E[i])
		result.Radii[a] = c.RowRadius[a].(*big.Int)
		for z, v := range p.H[i] {
			c.H[a][z] = v
		}
		for z, v := range p.TH.Path(i) {
			c.PH[a][z] = v
		}
		low := make([]*big.Int, len(params.Coeff))
		high := make([]*big.Int, len(low))
		for j, co := range params.Coeff {
			score := new(big.Int).Set(params.Constant[j])
			for z, v := range co {
				score.Add(score, mul(v, p.H[i][z]))
			}
			e := mul(mul(mul(bi(2), t2), result.Radii[a]), params.Radius[j])
			low[j] = new(big.Int).Sub(score, e)
			high[j] = add(score, e)
		}
		var lo, hi *big.Int
		for n := 0; n < params.New; n++ {
			var v *big.Int
			for o := 0; o < params.Old; o++ {
				x := low[n*params.Old+o]
				if v == nil || x.Cmp(v) > 0 {
					v = x
				}
			}
			if lo == nil || v.Cmp(lo) < 0 {
				lo = v
			}
		}
		for o := 0; o < params.Old; o++ {
			var v *big.Int
			for n := 0; n < params.New; n++ {
				x := high[n*params.Old+o]
				if v == nil || x.Cmp(v) < 0 {
					v = x
				}
			}
			if hi == nil || v.Cmp(hi) > 0 {
				hi = v
			}
		}
		if lo.Cmp(hi) > 0 {
			return nil, nil, fmt.Errorf("inconsistent paired interval")
		}
		result.Lower.Add(result.Lower, lo)
		result.Upper.Add(result.Upper, hi)
	}
	c.Lower, c.Upper = result.Lower, result.Upper
	return c, result, nil
}

func BoundPairedChangePublic(m BoundModel, s BoundStatement, old, next [][]int64, indices []int, lower, upper string) (*BoundPairedChange, error) {
	if err := validatePublicIndices(m, indices); err != nil {
		return nil, err
	}
	p, err := PairedChangeParameters(m, old, next)
	if err != nil {
		return nil, err
	}
	return BoundPairedChangePublicPrepared(m, s, p, indices, lower, upper)
}

// The verifier derives parameters once from its approved old/new center sets.
func BoundPairedChangePublicPrepared(m BoundModel, s BoundStatement, p *PairedParameters, indices []int, lower, upper string) (*BoundPairedChange, error) {
	if p == nil || p.modelHash != m.Hash().String() {
		return nil, fmt.Errorf("paired parameter model mismatch")
	}
	if err := validatePublicIndices(m, indices); err != nil {
		return nil, err
	}
	parse := func(text string) (*big.Int, error) {
		v, ok := new(big.Int).SetString(text, 10)
		if !ok || v.String() != text {
			return nil, fmt.Errorf("noncanonical signed cost change")
		}
		return v, nil
	}
	lo, err := parse(lower)
	if err != nil {
		return nil, err
	}
	hi, err := parse(upper)
	if err != nil {
		return nil, err
	}
	maximum := mul(p.limit, bi(int64(len(indices))))
	if lo.Cmp(hi) > 0 || new(big.Int).Abs(lo).Cmp(maximum) > 0 || new(big.Int).Abs(hi).Cmp(maximum) > 0 {
		return nil, fmt.Errorf("paired interval outside approved range")
	}
	c := NewBoundPairedChange(m, len(indices), p.Old, p.New)
	pairedPublicParameters(c, p)
	c.CX, c.CH, c.CE, c.CT, c.Context, err = s.Values(m)
	if err != nil {
		return nil, err
	}
	for a, i := range indices {
		c.Indices[a] = i
	}
	c.Lower, c.Upper = lo, hi
	return c, nil
}
