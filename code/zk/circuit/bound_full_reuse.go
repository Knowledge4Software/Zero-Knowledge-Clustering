package circuit

// Exact original-space baseline with public, changing centers. The original
// root, quantizer, dimensions and query size are fixed; no Stage 1 is required.
import (
	"fmt"
	"math/big"

	"github.com/consensys/gnark-crypto/ecc/bn254/fr"
	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/std/math/cmp"
	"github.com/consensys/gnark/std/rangecheck"
)

type BoundFullReuseQuery struct {
	M                            BoundModel            `gnark:"-"`
	Q, KNew                      int                   `gnark:"-"`
	CX, CH, CE, CT, Context, Psi frontend.Variable     `gnark:",public"`
	Indices                      []frontend.Variable   `gnark:",public"`
	Centers                      [][]frontend.Variable `gnark:",public"`
	X, Paths                     [][]frontend.Variable
	Assign                       []frontend.Variable
	Salt                         frontend.Variable
}

func NewBoundFullReuseQuery(m BoundModel, q, knew int) *BoundFullReuseQuery {
	c := &BoundFullReuseQuery{M: m, Q: q, KNew: knew, Indices: make([]frontend.Variable, q),
		Centers: make([][]frontend.Variable, knew), X: make([][]frontend.Variable, q),
		Paths: make([][]frontend.Variable, q), Assign: make([]frontend.Variable, q)}
	for j := range c.Centers {
		c.Centers[j] = make([]frontend.Variable, m.D)
	}
	for i := range c.X {
		c.X[i] = make([]frontend.Variable, m.D)
		c.Paths[i] = make([]frontend.Variable, m.Depth())
	}
	return c
}

func validateNewCenters(m BoundModel, centers [][]int64) error {
	if len(centers) < 2 || len(centers) > 128 {
		return fmt.Errorf("invalid public center count")
	}
	limit := int64(1) << uint(m.InputBits)
	for _, row := range centers {
		if len(row) != m.D {
			return fmt.Errorf("new center width mismatch")
		}
		for _, v := range row {
			if v < -limit || v > limit {
				return fmt.Errorf("new center outside approved input range")
			}
		}
	}
	return nil
}

func (c *BoundFullReuseQuery) Define(api frontend.API) error {
	m := c.M
	if err := m.Validate(); err != nil {
		return err
	}
	if m.Method != "full" || c.Q < 1 || c.Q > m.N || c.KNew < 2 || c.KNew > 128 {
		return fmt.Errorf("invalid dynamic Full query shape")
	}
	l := m.Limits()
	// Every public center and private row coordinate is in [-X,X].
	distanceBound := mul(bi(int64(4*m.D)), square(l.X))
	if mul(distanceBound, bi(int64(c.Q)*16)).Cmp(api.Compiler().Field()) >= 0 {
		return fmt.Errorf("unsafe Full aggregate bound")
	}
	rc := rangecheck.New(api)
	boundBindContext(api, m, c.CX, c.CH, c.CE, c.CT, c.Context)
	for _, center := range c.Centers {
		for _, v := range center {
			rangeCheckSigned(api, rc, v, l.X, false)
		}
	}
	bc := cmp.NewBoundedComparator(api, add(distanceBound, bi(1)), false)
	ic := cmp.NewBoundedComparator(api, bi(int64(m.N)), false)
	sum := frontend.Variable(0)
	for i := 0; i < c.Q; i++ {
		if i > 0 {
			ic.AssertIsLess(c.Indices[i-1], c.Indices[i])
		}
		for _, v := range c.X[i] {
			rangeCheckSigned(api, rc, v, l.X, false)
		}
		boundOpen(api, m, boundX, c.Indices[i], c.Salt, c.X[i], c.Paths[i], c.CX)
		var best frontend.Variable
		winner := frontend.Variable(0)
		for j, center := range c.Centers {
			d := frontend.Variable(0)
			for t, v := range c.X[i] {
				delta := api.Sub(v, center[t])
				d = api.Add(d, api.Mul(delta, delta))
			}
			rangeCheckUnsigned(api, rc, d, distanceBound, false)
			if j == 0 {
				best = d
				continue
			}
			less := bc.IsLess(d, best)
			best = api.Select(less, d, best)
			winner = api.Select(less, j, winner)
		}
		api.AssertIsEqual(c.Assign[i], winner)
		sum = api.Add(sum, best)
	}
	api.AssertIsEqual(c.Psi, sum)
	return nil
}

func FullReuseAnswer(p *BoundPrepared, centers [][]int64, indices []int) ([]int, *big.Int, error) {
	if p.M.Method != "full" {
		return nil, nil, fmt.Errorf("dynamic Full needs original-row model")
	}
	if err := validateNewCenters(p.M, centers); err != nil {
		return nil, nil, err
	}
	if err := validatePublicIndices(p.M, indices); err != nil {
		return nil, nil, err
	}
	labels := make([]int, len(indices))
	sum := bi(0)
	for a, i := range indices {
		var best *big.Int
		for j, center := range centers {
			d := bi(0)
			for t, v := range p.X[i] {
				d.Add(d, square(bi(v-center[t])))
			}
			if j == 0 || d.Cmp(best) < 0 {
				best = d
				labels[a] = j
			}
		}
		sum.Add(sum, best)
	}
	return labels, sum, nil
}

func FillBoundFullReuseQuery(p *BoundPrepared, centers [][]int64, indices []int) (*BoundFullReuseQuery, error) {
	labels, psi, err := FullReuseAnswer(p, centers, indices)
	if err != nil {
		return nil, err
	}
	c := NewBoundFullReuseQuery(p.M, len(indices), len(centers))
	c.CX, c.CH, c.CE, c.CT, c.Context, c.Psi = p.CX, p.CH, p.CE, p.CT, p.Context, psi
	c.Salt = p.SaltX
	for j, row := range centers {
		for t, v := range row {
			c.Centers[j][t] = v
		}
	}
	for a, i := range indices {
		c.Indices[a] = i
		c.Assign[a] = labels[a]
		for t, v := range p.X[i] {
			c.X[a][t] = v
		}
		for t, v := range p.TX.Path(i) {
			c.Paths[a][t] = v
		}
	}
	return c, nil
}

func validatePublicIndices(m BoundModel, indices []int) error {
	if len(indices) < 1 || len(indices) > m.N {
		return fmt.Errorf("invalid query size")
	}
	for a, i := range indices {
		if i < 0 || i >= m.N || (a > 0 && i <= indices[a-1]) {
			return fmt.Errorf("query indices must be strictly increasing and bounded")
		}
	}
	return nil
}

func canonicalPublicInt(text string) (*big.Int, error) {
	v, ok := new(big.Int).SetString(text, 10)
	if !ok || v.Sign() < 0 || v.Cmp(fr.Modulus()) >= 0 || v.String() != text {
		return nil, fmt.Errorf("invalid canonical nonnegative public integer")
	}
	return v, nil
}

func BoundFullReusePublic(m BoundModel, s BoundStatement, centers [][]int64, indices []int, psi string) (*BoundFullReuseQuery, error) {
	if err := m.Validate(); err != nil {
		return nil, err
	}
	if m.Method != "full" {
		return nil, fmt.Errorf("not a Full model")
	}
	if err := validateNewCenters(m, centers); err != nil {
		return nil, err
	}
	if err := validatePublicIndices(m, indices); err != nil {
		return nil, err
	}
	cx, ch, ce, ct, ctx, err := s.Values(m)
	if err != nil {
		return nil, err
	}
	v, err := canonicalPublicInt(psi)
	if err != nil {
		return nil, err
	}
	max := mul(bi(int64(len(indices)*4*m.D)), square(m.Limits().X))
	if v.Cmp(max) > 0 {
		return nil, fmt.Errorf("Full cost exceeds integer bound")
	}
	c := NewBoundFullReuseQuery(m, len(indices), len(centers))
	c.CX, c.CH, c.CE, c.CT, c.Context, c.Psi = cx, ch, ce, ct, ctx, v
	for j, row := range centers {
		for t, v := range row {
			c.Centers[j][t] = v
		}
	}
	for i, index := range indices {
		c.Indices[i] = index
	}
	return c, nil
}
