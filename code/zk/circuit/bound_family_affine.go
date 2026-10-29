package circuit

// Exact known-family baseline: the complete approved center bank is baked into
// the affine model before preprocessing. Query positions select an arbitrary
// ordered subset of that bank. Unknown future centers require new preparation.
import (
	"fmt"
	"math/big"
	"reflect"

	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/std/math/cmp"
	"github.com/consensys/gnark/std/rangecheck"
)

type BoundFamilyAffineQuery struct {
	M                            BoundModel          `gnark:"-"`
	Q, KNew                      int                 `gnark:"-"`
	CX, CH, CE, CT, Context, Psi frontend.Variable   `gnark:",public"`
	Indices, Positions           []frontend.Variable `gnark:",public"`
	H, PH                        [][]frontend.Variable
	E, Assign                    []frontend.Variable
	Salt                         frontend.Variable
}

func NewBoundFamilyAffineQuery(m BoundModel, q, k int) *BoundFamilyAffineQuery {
	c := &BoundFamilyAffineQuery{M: m, Q: q, KNew: k, Indices: make([]frontend.Variable, q), Positions: make([]frontend.Variable, k), H: make([][]frontend.Variable, q), PH: make([][]frontend.Variable, q), E: make([]frontend.Variable, q), Assign: make([]frontend.Variable, q)}
	for i := range c.H {
		c.H[i] = make([]frontend.Variable, m.Width())
		c.PH[i] = make([]frontend.Variable, m.Depth())
	}
	return c
}

func (c *BoundFamilyAffineQuery) Define(api frontend.API) error {
	m := c.M
	if err := m.Validate(); err != nil {
		return err
	}
	if m.Method != "affine" || c.Q < 1 || c.Q > m.N || c.KNew < 2 || c.KNew > 128 {
		return fmt.Errorf("invalid family affine query shape")
	}
	l := m.Limits()
	if mul(l.Distance, bi(int64(c.Q)*16)).Cmp(api.Compiler().Field()) >= 0 {
		return fmt.Errorf("unsafe family affine aggregate")
	}
	rc := rangecheck.New(api)
	boundBindContext(api, m, c.CX, c.CH, c.CE, c.CT, c.Context)
	selector := make([][]frontend.Variable, c.KNew)
	for j, pos := range c.Positions {
		rangeCheckUnsigned(api, rc, pos, bi(int64(m.K-1)), false)
		selector[j] = make([]frontend.Variable, m.K-1)
		for b := 1; b < m.K; b++ {
			selector[j][b-1] = api.IsZero(api.Sub(pos, b))
		}
	}
	bc := cmp.NewBoundedComparator(api, l.Score, false)
	ic := cmp.NewBoundedComparator(api, bi(int64(m.N)), false)
	sum := frontend.Variable(0)
	for i := 0; i < c.Q; i++ {
		if i > 0 {
			ic.AssertIsLess(c.Indices[i-1], c.Indices[i])
		}
		for _, v := range c.H[i] {
			rangeCheckSigned(api, rc, v, l.H, false)
		}
		rangeCheckUnsigned(api, rc, c.E[i], l.E, false)
		boundOpen(api, m, boundH, c.Indices[i], c.Salt, append(append([]frontend.Variable{}, c.H[i]...), c.E[i]), c.PH[i], c.CH)
		var best frontend.Variable
		winner := frontend.Variable(0)
		for j := 0; j < c.KNew; j++ {
			score := frontend.Variable(l.H)
			for b, v := range c.H[i] {
				score = api.Add(score, api.Mul(selector[j][b], v))
			}
			rangeCheckUnsigned(api, rc, score, l.Score, false)
			if j == 0 {
				best = score
				continue
			}
			better := bc.IsLess(best, score)
			best = api.Select(better, score, best)
			winner = api.Select(better, j, winner)
		}
		api.AssertIsEqual(c.Assign[i], winner)
		distance := api.Add(c.E[i], l.H, api.Neg(best))
		rangeCheckUnsigned(api, rc, distance, l.Distance, false)
		sum = api.Add(sum, distance)
	}
	api.AssertIsEqual(c.Psi, sum)
	return nil
}

func FamilyAffinePositions(m BoundModel, centers [][]int64) ([]int, error) {
	if err := m.Validate(); err != nil {
		return nil, err
	}
	if m.Method != "affine" {
		return nil, fmt.Errorf("family baseline requires affine bank model")
	}
	if err := validateNewCenters(m, centers); err != nil {
		return nil, err
	}
	positions := make([]int, len(centers))
	for j, center := range centers {
		found := -1
		for b, approved := range m.Centers {
			if reflect.DeepEqual(center, approved) {
				found = b
				break
			}
		}
		if found < 0 {
			return nil, fmt.Errorf("center outside preprocessed family")
		}
		positions[j] = found
	}
	return positions, nil
}

func FillBoundFamilyAffineQuery(p *BoundPrepared, centers [][]int64, indices []int) (*BoundFamilyAffineQuery, error) {
	positions, err := FamilyAffinePositions(p.M, centers)
	if err != nil {
		return nil, err
	}
	if err = validatePublicIndices(p.M, indices); err != nil {
		return nil, err
	}
	c := NewBoundFamilyAffineQuery(p.M, len(indices), len(centers))
	c.CX, c.CH, c.CE, c.CT, c.Context = p.CX, p.CH, p.CE, p.CT, p.Context
	c.Salt = p.SaltZ
	for j, v := range positions {
		c.Positions[j] = v
	}
	sum := bi(0)
	for a, i := range indices {
		var best *big.Int
		winner := 0
		for j, pos := range positions {
			score := bi(0)
			if pos > 0 {
				score = p.H[i][pos-1]
			}
			if best == nil || score.Cmp(best) > 0 {
				best = new(big.Int).Set(score)
				winner = j
			}
		}
		sum.Add(sum, new(big.Int).Sub(p.E[i], best))
		c.Indices[a], c.Assign[a], c.E[a] = i, winner, p.E[i]
		for b, v := range p.H[i] {
			c.H[a][b] = v
		}
		for b, v := range p.TH.Path(i) {
			c.PH[a][b] = v
		}
	}
	c.Psi = sum
	return c, nil
}

func BoundFamilyAffinePublic(m BoundModel, s BoundStatement, centers [][]int64, indices []int, psi string) (*BoundFamilyAffineQuery, error) {
	positions, err := FamilyAffinePositions(m, centers)
	if err != nil {
		return nil, err
	}
	if err = validatePublicIndices(m, indices); err != nil {
		return nil, err
	}
	cx, ch, ce, ct, context, err := s.Values(m)
	if err != nil {
		return nil, err
	}
	value, err := canonicalPublicInt(psi)
	if err != nil {
		return nil, err
	}
	if value.Cmp(mul(m.Limits().Distance, bi(int64(len(indices))))) > 0 {
		return nil, fmt.Errorf("family cost outside integer bound")
	}
	c := NewBoundFamilyAffineQuery(m, len(indices), len(centers))
	c.CX, c.CH, c.CE, c.CT, c.Context, c.Psi = cx, ch, ce, ct, context, value
	for j, v := range positions {
		c.Positions[j] = v
	}
	for a, i := range indices {
		c.Indices[a] = i
	}
	return c, nil
}
