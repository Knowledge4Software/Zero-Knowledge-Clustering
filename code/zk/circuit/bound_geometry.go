package circuit

// Geometry queries authenticate two rows under the existing global roots and
// prove distance IN THE STORED EMBEDDING. Original-space fidelity is a separate
// empirical/geometric condition, not a consequence of this arithmetic proof.
import (
	"fmt"
	"math/big"

	"github.com/consensys/gnark-crypto/ecc/bn254/fr"
	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/std/math/cmp"
	"github.com/consensys/gnark/std/rangecheck"
)

type BoundGeometry struct {
	M                                 BoundModel           `gnark:"-"`
	CX, CH, CE, CT, Context, Distance frontend.Variable    `gnark:",public"`
	Indices                           [2]frontend.Variable `gnark:",public"`
	H, Tail, PH, PT                   [2][]frontend.Variable
	E                                 [2]frontend.Variable
	Salt                              frontend.Variable
}

func NewBoundGeometry(m BoundModel) *BoundGeometry {
	c := &BoundGeometry{M: m}
	for i := 0; i < 2; i++ {
		c.H[i] = make([]frontend.Variable, m.Width())
		c.Tail[i] = make([]frontend.Variable, m.TailWidth)
		c.PH[i] = make([]frontend.Variable, m.Depth())
		if m.TailWidth > 0 {
			c.PT[i] = make([]frontend.Variable, m.Depth())
		}
	}
	return c
}
func boundGeometryLimit(m BoundModel) *big.Int {
	l := m.Limits()
	return add(mul(bi(int64(4*m.Width())), square(l.H)), mul(bi(int64(4*m.TailWidth)), square(l.T)))
}
func (c *BoundGeometry) Define(api frontend.API) error {
	m := c.M
	if err := m.Validate(); err != nil {
		return err
	}
	if m.Method == "affine" || m.Method == "cached" || m.N < 2 {
		return fmt.Errorf("geometry requires a Euclidean representation and two rows")
	}
	limit := boundGeometryLimit(m)
	if mul(limit, bi(4)).Cmp(api.Compiler().Field()) >= 0 {
		return fmt.Errorf("unsafe geometry bound")
	}
	l := m.Limits()
	rc := rangecheck.New(api)
	boundBindContext(api, m, c.CX, c.CH, c.CE, c.CT, c.Context)
	cmp.NewBoundedComparator(api, bi(int64(m.N)), false).AssertIsLess(c.Indices[0], c.Indices[1])
	for i := 0; i < 2; i++ {
		for _, v := range c.H[i] {
			rangeCheckSigned(api, rc, v, l.H, false)
		}
		values := append([]frontend.Variable{}, c.H[i]...)
		if m.HasEnergy() {
			rangeCheckUnsigned(api, rc, c.E[i], l.E, false)
			values = append(values, c.E[i]) // authenticate the joint head/energy leaf; E is not a Euclidean coordinate
		} else {
			api.AssertIsEqual(c.E[i], 0)
		}
		domain := boundH
		if m.Method == "full" {
			domain = boundX
		}
		boundOpen(api, m, domain, c.Indices[i], c.Salt, values, c.PH[i], c.CH)
		if m.TailWidth > 0 {
			for _, v := range c.Tail[i] {
				rangeCheckSigned(api, rc, v, l.T, false)
			}
			boundOpen(api, m, boundT, c.Indices[i], c.Salt, c.Tail[i], c.PT[i], c.CT)
		}
	}
	distance := frontend.Variable(0)
	for j := range c.H[0] {
		v := api.Sub(c.H[0][j], c.H[1][j])
		distance = api.Add(distance, api.Mul(v, v))
	}
	for j := range c.Tail[0] {
		v := api.Sub(c.Tail[0][j], c.Tail[1][j])
		distance = api.Add(distance, api.Mul(v, v))
	}
	rangeCheckUnsigned(api, rc, distance, limit, false)
	api.AssertIsEqual(c.Distance, distance)
	return nil
}
func (p *BoundPrepared) GeometryDistance(indices []int) (*big.Int, error) {
	if p.M.Method == "affine" || p.M.Method == "cached" || len(indices) != 2 || indices[0] < 0 || indices[0] >= indices[1] || indices[1] >= p.M.N {
		return nil, fmt.Errorf("invalid geometry method/pair")
	}
	d := bi(0)
	a, b := indices[0], indices[1]
	for j, v := range p.H[a] {
		d.Add(d, square(new(big.Int).Sub(v, p.H[b][j])))
	}
	for j, v := range p.T[a] {
		d.Add(d, square(new(big.Int).Sub(v, p.T[b][j])))
	}
	return d, nil
}
func FillBoundGeometry(p *BoundPrepared, indices []int) (*BoundGeometry, error) {
	d, err := p.GeometryDistance(indices)
	if err != nil {
		return nil, err
	}
	c := NewBoundGeometry(p.M)
	c.CX, c.CH, c.CE, c.CT, c.Context, c.Distance = p.CX, p.CH, p.CE, p.CT, p.Context, d
	c.Salt = p.SaltZ
	if p.M.Method == "full" {
		c.Salt = p.SaltX
	}
	for i, index := range indices {
		c.Indices[i] = index
		c.E[i] = p.E[index]
		for j, v := range p.H[index] {
			c.H[i][j] = v
		}
		for j, v := range p.T[index] {
			c.Tail[i][j] = v
		}
		for j, v := range p.TH.Path(index) {
			c.PH[i][j] = v
		}
		if p.M.TailWidth > 0 {
			for j, v := range p.TT.Path(index) {
				c.PT[i][j] = v
			}
		}
	}
	return c, nil
}
func BoundGeometryPublic(m BoundModel, s BoundStatement, indices []int, distance string) (*BoundGeometry, error) {
	if err := m.Validate(); err != nil {
		return nil, err
	}
	if m.Method == "affine" || m.Method == "cached" || len(indices) != 2 || indices[0] < 0 || indices[0] >= indices[1] || indices[1] >= m.N {
		return nil, fmt.Errorf("invalid geometry pair")
	}
	v, ok := new(big.Int).SetString(distance, 10)
	if !ok || v.Sign() < 0 || v.Cmp(fr.Modulus()) >= 0 || v.String() != distance {
		return nil, fmt.Errorf("invalid canonical geometry distance")
	}
	cx, ch, ce, ct, ctx, err := s.Values(m)
	if err != nil {
		return nil, err
	}
	c := NewBoundGeometry(m)
	c.CX, c.CH, c.CE, c.CT, c.Context, c.Distance = cx, ch, ce, ct, ctx, v
	c.Indices[0], c.Indices[1] = indices[0], indices[1]
	return c, nil
}
