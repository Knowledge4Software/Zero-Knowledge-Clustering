package circuit

import (
	"fmt"
	"math/big"

	"github.com/consensys/gnark/frontend"
	poseidon "github.com/consensys/gnark/std/hash/poseidon2"
	"github.com/consensys/gnark/std/math/cmp"
	"github.com/consensys/gnark/std/rangecheck"
)

func boundHash(api frontend.API, v ...frontend.Variable) frontend.Variable {
	h, err := poseidon.New(api)
	if err != nil {
		panic(err)
	}
	h.Write(v...)
	return h.Sum()
}
func boundLeafCircuit(api frontend.API, m BoundModel, domain int, index, salt frontend.Variable, values []frontend.Variable) frontend.Variable {
	var tag frontend.Variable = m.Hash()
	if domain == boundX {
		tag = 0
	}
	v := []frontend.Variable{domain, m.N, m.D, tag, index, salt}
	return boundHash(api, append(v, values...)...)
}
func boundOpen(api frontend.API, m BoundModel, domain int, index, salt frontend.Variable, values, path []frontend.Variable, root frontend.Variable) {
	var bits []frontend.Variable
	if m.Depth() == 0 {
		api.AssertIsEqual(index, 0)
	} else {
		bits = api.ToBinary(index, m.Depth())
	}
	current := boundLeafCircuit(api, m, domain, index, salt, values)
	for level, sibling := range path {
		left := api.Select(bits[level], sibling, current)
		right := api.Select(bits[level], current, sibling)
		current = boundHash(api, boundNode, level, left, right)
	}
	api.AssertIsEqual(current, root)
}
func boundBindContext(api frontend.API, m BoundModel, cx, ch, ce, ct, context frontend.Variable) {
	api.AssertIsEqual(boundHash(api, boundContext, m.Hash(), cx, ch, ce, ct), context)
	api.AssertIsEqual(ce, 0) // head and energy share one row commitment
	if m.TailWidth == 0 {
		api.AssertIsEqual(ct, 0)
	}
	if m.Method == "full" {
		api.AssertIsEqual(ch, cx)
	}
}

// BoundPreprocess proves one indexed row from the original commitment and its
// deterministically derived statistics under globally shared roots. A verifier
// must require coverage of every catalog index before authorizing queries.
type BoundPreprocess struct {
	M                       BoundModel        `gnark:"-"`
	CX, CH, CE, CT, Context frontend.Variable `gnark:",public"`
	Index                   frontend.Variable `gnark:",public"`
	X, H, T, Dot            []frontend.Variable
	E, SaltX, SaltZ         frontend.Variable
	PX, PH, PE, PT          []frontend.Variable
}

func NewBoundPreprocess(m BoundModel) *BoundPreprocess {
	c := &BoundPreprocess{M: m, X: make([]frontend.Variable, m.D), H: make([]frontend.Variable, m.Width()), T: make([]frontend.Variable, m.TailWidth), PX: make([]frontend.Variable, m.Depth()), PH: make([]frontend.Variable, m.Depth()), Dot: make([]frontend.Variable, len(m.Basis))}
	if m.TailWidth > 0 {
		c.PT = make([]frontend.Variable, m.Depth())
	}
	return c
}
func (c *BoundPreprocess) Define(api frontend.API) error {
	return c.defineWithHeadSuffix(api, nil)
}

// A separately keyed compact group profile binds its private group in the
// same head/energy leaf. Legacy profiles retain exactly their original leaf.
func (c *BoundPreprocess) defineWithHeadSuffix(api frontend.API, suffix []frontend.Variable) error {
	m := c.M
	if err := m.Validate(); err != nil {
		return err
	}
	if m.Method == "full" {
		return fmt.Errorf("full uses the original root directly; no preprocessing proof")
	}
	l := m.Limits()
	rc := rangecheck.New(api)
	boundBindContext(api, m, c.CX, c.CH, c.CE, c.CT, c.Context)
	w := make([]frontend.Variable, m.D)
	for t := range c.X {
		rangeCheckSigned(api, rc, c.X[t], l.X, false)
		w[t] = api.Sub(c.X[t], m.Centers[0][t])
	}
	boundOpen(api, m, boundX, c.Index, c.SaltX, c.X, c.PX, c.CX)
	if m.Method == "cached" {
		norm := frontend.Variable(0)
		for _, v := range w {
			norm = api.Add(norm, api.Mul(v, v))
		}
		shift := new(big.Int).Rsh(new(big.Int).Set(l.Score), 1)
		best, idx := frontend.Variable(shift), frontend.Variable(0)
		bc := cmp.NewBoundedComparator(api, l.Score, false)
		for j := 1; j < m.K; j++ {
			score := frontend.Variable(shift)
			for t, v := range w {
				delta := m.Centers[j][t] - m.Centers[0][t]
				score = api.Add(score, api.Sub(api.Mul(2*delta, v), square(bi(delta))))
			}
			rangeCheckUnsigned(api, rc, score, l.Score, false)
			greater := bc.IsLess(best, score)
			best = api.Select(greater, score, best)
			idx = api.Select(greater, j, idx)
		}
		api.AssertIsEqual(c.H[0], idx)
		api.AssertIsEqual(c.E, api.Add(norm, api.Sub(shift, best)))
	} else if m.Method == "affine" {
		norm := frontend.Variable(0)
		for _, v := range w {
			norm = api.Add(norm, api.Mul(v, v))
		}
		api.AssertIsEqual(c.E, norm)
		for j := range c.H {
			score := frontend.Variable(0)
			for t, v := range w {
				dj := m.Centers[j+1][t] - m.Centers[0][t]
				score = api.Add(score, api.Sub(api.Mul(2*dj, v), square(bi(dj))))
			}
			api.AssertIsEqual(c.H[j], score)
		}
	} else {
		dot := c.Dot
		for j, row := range m.Basis {
			value := frontend.Variable(0)
			for t, b := range row {
				value = api.Add(value, api.Mul(b, w[t]))
			}
			api.AssertIsEqual(dot[j], value)
			api.AssertIsEqual(c.H[j], api.Mul(m.BasisScale, dot[j]))
		}
		if m.HasEnergy() || m.UsesTailQuery() {
			energy := frontend.Variable(0)
			tail := make([]frontend.Variable, m.TailWidth)
			for j := range tail {
				tail[j] = 0
			}
			for t := range w {
				residual := api.Mul(m.BasisScale*m.BasisScale, w[t])
				for j, row := range m.Basis {
					residual = api.Sub(residual, api.Mul(row[t], dot[j]))
				}
				if m.HasEnergy() {
					energy = api.Add(energy, api.Mul(residual, residual))
				}
				if m.TailWidth > 0 {
					tail[m.TailRows[t]] = api.Add(tail[m.TailRows[t]], api.Mul(m.TailSigns[t], residual))
				}
			}
			api.AssertIsEqual(c.E, energy)
			for j := range tail {
				api.AssertIsEqual(c.T[j], tail[j])
			}
		} else {
			api.AssertIsEqual(c.E, 0)
		}
	}
	for _, v := range c.H {
		rangeCheckSigned(api, rc, v, l.H, false)
	}
	values := append([]frontend.Variable{}, c.H...)
	if m.HasEnergy() {
		rangeCheckUnsigned(api, rc, c.E, l.E, false)
		values = append(values, c.E)
	}
	values = append(values, suffix...)
	boundOpen(api, m, boundH, c.Index, c.SaltZ, values, c.PH, c.CH)
	if m.TailWidth > 0 {
		for _, v := range c.T {
			rangeCheckSigned(api, rc, v, l.T, false)
		}
		boundOpen(api, m, boundT, c.Index, c.SaltZ, c.T, c.PT, c.CT)
	}
	return nil
}
func FillBoundPreprocess(p *BoundPrepared, index int) *BoundPreprocess {
	c := NewBoundPreprocess(p.M)
	c.CX, c.CH, c.CE, c.CT, c.Context = p.CX, p.CH, p.CE, p.CT, p.Context
	c.Index = index
	c.SaltX, c.SaltZ = p.SaltX, p.SaltZ
	c.E = p.E[index]
	for t, v := range p.X[index] {
		c.X[t] = v
	}
	for t, v := range p.H[index] {
		c.H[t] = v
	}
	for t, v := range p.T[index] {
		c.T[t] = v
	}
	for t := range c.Dot {
		c.Dot[t] = new(big.Int).Quo(p.H[index][t], bi(p.M.BasisScale))
	}
	for t, v := range p.TX.Path(index) {
		c.PX[t] = v
	}
	for t, v := range p.TH.Path(index) {
		c.PH[t] = v
	}
	if p.TE != nil {
		for t, v := range p.TE.Path(index) {
			c.PE[t] = v
		}
	}
	if p.TT != nil {
		for t, v := range p.TT.Path(index) {
			c.PT[t] = v
		}
	}
	return c
}

// BoundQuery authenticates a changing, strictly ordered subset under the same
// catalog roots used by every Stage-1 row proof. It does not rebuild a root
// over the selected rows. This is an actual membership-path relation.
type BoundQuery struct {
	M                            BoundModel          `gnark:"-"`
	Q                            int                 `gnark:"-"`
	CX, CH, CE, CT, Context, Psi frontend.Variable   `gnark:",public"`
	Indices                      []frontend.Variable `gnark:",public"`
	H, PH, PE                    [][]frontend.Variable
	Tail, PT                     [][]frontend.Variable
	E, Assign                    []frontend.Variable
	Salt                         frontend.Variable
}

func NewBoundQuery(m BoundModel, q int) *BoundQuery {
	c := &BoundQuery{M: m, Q: q, Indices: make([]frontend.Variable, q), H: make([][]frontend.Variable, q), PH: make([][]frontend.Variable, q), PE: make([][]frontend.Variable, q), E: make([]frontend.Variable, q), Assign: make([]frontend.Variable, q)}
	for i := 0; i < q; i++ {
		c.H[i] = make([]frontend.Variable, m.Width())
		c.PH[i] = make([]frontend.Variable, m.Depth())
	}
	if m.UsesTailQuery() {
		c.Tail = make([][]frontend.Variable, q)
		c.PT = make([][]frontend.Variable, q)
		for i := 0; i < q; i++ {
			c.Tail[i] = make([]frontend.Variable, m.TailWidth)
			c.PT[i] = make([]frontend.Variable, m.Depth())
		}
	}
	return c
}
func (c *BoundQuery) Define(api frontend.API) error {
	m := c.M
	if err := m.Validate(); err != nil {
		return err
	}
	if c.Q < 1 || c.Q > m.N {
		return fmt.Errorf("invalid query size")
	}
	l := m.Limits()
	if mul(l.Distance, bi(int64(c.Q)*2)).Cmp(api.Compiler().Field()) >= 0 {
		return fmt.Errorf("unsafe aggregate bound")
	}
	rc := rangecheck.New(api)
	boundBindContext(api, m, c.CX, c.CH, c.CE, c.CT, c.Context)
	hcenters := make([][]*big.Int, m.K)
	tcenters := make([][]*big.Int, m.K)
	for j, row := range m.Centers {
		hcenters[j], _, tcenters[j] = BoundStats(m, row)
	}
	sb := new(big.Int).Set(l.Score)
	if sb.Sign() == 0 {
		sb.SetInt64(1)
	}
	bc := cmp.NewBoundedComparator(api, sb, false)
	ic := cmp.NewBoundedComparator(api, bi(int64(m.N)), false)
	sum := frontend.Variable(0)
	for i := 0; i < c.Q; i++ {
		if i > 0 {
			ic.AssertIsLess(c.Indices[i-1], c.Indices[i])
		}
		for _, v := range c.H[i] {
			rangeCheckSigned(api, rc, v, l.H, false)
		}
		domain := boundH
		if m.Method == "full" {
			domain = boundX
		}
		values := append([]frontend.Variable{}, c.H[i]...)
		if m.HasEnergy() {
			rangeCheckUnsigned(api, rc, c.E[i], l.E, false)
			values = append(values, c.E[i])
		} else {
			api.AssertIsEqual(c.E[i], 0)
		}
		boundOpen(api, m, domain, c.Indices[i], c.Salt, values, c.PH[i], c.CH)
		if m.Method == "cached" {
			rangeCheckUnsigned(api, rc, c.H[i][0], bi(int64(m.K-1)), false)
			api.AssertIsEqual(c.Assign[i], c.H[i][0])
			sum = api.Add(sum, c.E[i])
			continue
		}
		if m.UsesTailQuery() {
			for _, v := range c.Tail[i] {
				rangeCheckSigned(api, rc, v, l.T, false)
			}
			boundOpen(api, m, boundT, c.Indices[i], c.Salt, c.Tail[i], c.PT[i], c.CT)
		}
		var best frontend.Variable
		idx := frontend.Variable(0)
		shift := new(big.Int).Rsh(new(big.Int).Set(l.Score), 1)
		for j := 0; j < m.K; j++ {
			var score frontend.Variable
			if m.Method == "affine" {
				score = shift
				if j > 0 {
					score = api.Add(c.H[i][j-1], shift)
				}
			} else {
				score = shift
				for t, v := range c.H[i] {
					score = api.Add(score, api.Sub(api.Mul(mul(bi(2), hcenters[j][t]), v), square(hcenters[j][t])))
				}
				if m.UsesTailQuery() {
					for t, v := range c.Tail[i] {
						score = api.Add(score, api.Sub(api.Mul(mul(bi(2), tcenters[j][t]), v), square(tcenters[j][t])))
					}
				}
			}
			rangeCheckUnsigned(api, rc, score, l.Score, false)
			if j == 0 {
				best = score
				continue
			}
			greater := bc.IsLess(best, score)
			best = api.Select(greater, score, best)
			idx = api.Select(greater, j, idx)
		}
		api.AssertIsEqual(c.Assign[i], idx)
		var distance frontend.Variable
		if m.Method == "affine" {
			distance = api.Add(c.E[i], api.Sub(shift, best))
		} else {
			norm := frontend.Variable(0)
			for _, v := range c.H[i] {
				norm = api.Add(norm, api.Mul(v, v))
			}
			if m.UsesTailQuery() {
				for _, v := range c.Tail[i] {
					norm = api.Add(norm, api.Mul(v, v))
				}
			}
			distance = api.Add(norm, api.Sub(shift, best))
			if m.HasEnergy() {
				distance = api.Add(distance, c.E[i])
			}
		}
		rangeCheckUnsigned(api, rc, distance, l.Distance, false)
		sum = api.Add(sum, distance)
	}
	rangeCheckUnsigned(api, rc, sum, mul(l.Distance, bi(int64(c.Q))), false)
	api.AssertIsEqual(c.Psi, sum)
	return nil
}
func FillBoundQuery(p *BoundPrepared, indices []int) (*BoundQuery, error) {
	labels, psi, err := p.Answer(indices)
	if err != nil {
		return nil, err
	}
	c := NewBoundQuery(p.M, len(indices))
	c.CX, c.CH, c.CE, c.CT, c.Context, c.Psi = p.CX, p.CH, p.CE, p.CT, p.Context, psi
	c.Salt = p.SaltZ
	if p.M.Method == "full" {
		c.Salt = p.SaltX
	}
	for i, index := range indices {
		c.Indices[i] = index
		c.Assign[i] = labels[i]
		c.E[i] = p.E[index]
		for t, v := range p.H[index] {
			c.H[i][t] = v
		}
		for t, v := range p.TH.Path(index) {
			c.PH[i][t] = v
		}
		if p.M.UsesTailQuery() {
			for t, v := range p.T[index] {
				c.Tail[i][t] = v
			}
			for t, v := range p.TT.Path(index) {
				c.PT[i][t] = v
			}
		}
		if p.TE != nil {
			for t, v := range p.TE.Path(index) {
				c.PE[i][t] = v
			}
		}
	}
	return c, nil
}
