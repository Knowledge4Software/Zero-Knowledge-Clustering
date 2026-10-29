package circuit

// Projection-free authenticated group-moment baseline. Method=affine is used
// only as the existing bounded, no-basis model schema; no affine/head/energy
// statistics are computed, committed, or proved by this profile.
import (
	"fmt"
	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/std/math/cmp"
	"github.com/consensys/gnark/std/rangecheck"
	"math/big"
)

func ValidateMomentsCenters(m BoundModel, centers [][]int64) error {
	if len(centers) != m.K {
		return fmt.Errorf("moments center count")
	}
	limit := int64(1) << uint(m.InputBits)
	for _, row := range centers {
		if len(row) != m.D {
			return fmt.Errorf("moments center width")
		}
		for _, v := range row {
			if v < -limit || v > limit {
				return fmt.Errorf("moments center range")
			}
		}
	}
	return nil
}

func PrepareMomentsOnly(m BoundModel, x [][]int64, sx, sz *big.Int) (*BoundPrepared, error) {
	if err := m.Validate(); err != nil {
		return nil, err
	}
	if m.Method != "affine" || len(m.Basis) != 0 || len(x) != m.N {
		return nil, fmt.Errorf("invalid moments-only catalog")
	}
	p := &BoundPrepared{M: m, X: x, SaltX: sx, SaltZ: sz, CH: bi(0), CE: bi(0), CT: bi(0)}
	leaves := make([]*big.Int, m.N)
	maximum := int64(1) << uint(m.InputBits)
	for i, row := range x {
		if len(row) != m.D {
			return nil, fmt.Errorf("row width")
		}
		values := make([]*big.Int, m.D)
		for d, v := range row {
			if v < -maximum || v > maximum {
				return nil, fmt.Errorf("row range")
			}
			values[d] = bi(v)
		}
		leaves[i] = boundLeaf(m, boundX, i, sx, values)
	}
	p.TX = NewBoundTree(leaves)
	p.CX = p.TX.Root()
	p.Context = boundNativeHash(bi(boundContext), m.Hash(), p.CX, p.CH, p.CE, p.CT)
	return p, nil
}

type BoundMomentsRow struct {
	M                                                          BoundModel          `gnark:"-"`
	CX, GroupRoot, StatsRoot, Challenge, Index, Previous, Next frontend.Variable   `gnark:",public"`
	Masked                                                     []frontend.Variable `gnark:",public"`
	X, PX, PG, PreviousMask, NextMask                          []frontend.Variable
	Group, SaltX, SaltZ                                        frontend.Variable
}

func NewBoundMomentsRow(m BoundModel) *BoundMomentsRow {
	return &BoundMomentsRow{M: m, X: make([]frontend.Variable, m.D), PX: make([]frontend.Variable, m.Depth()), PG: make([]frontend.Variable, m.Depth()), Masked: make([]frontend.Variable, m.K), PreviousMask: make([]frontend.Variable, m.K), NextMask: make([]frontend.Variable, m.K)}
}
func (c *BoundMomentsRow) Define(api frontend.API) error {
	m := c.M
	if err := m.Validate(); err != nil {
		return err
	}
	if m.Method != "affine" || len(m.Basis) != 0 {
		return fmt.Errorf("moments-only model required")
	}
	rc := rangecheck.New(api)
	xmax := m.Limits().X
	for _, v := range c.X {
		rangeCheckSigned(api, rc, v, xmax, false)
	}
	boundOpen(api, m, boundX, c.Index, c.SaltX, c.X, c.PX, c.CX)
	rangeCheckUnsigned(api, rc, c.Group, bi(int64(m.K-1)), false)
	boundOpen(api, m, boundFingerprintGroup, c.Index, c.SaltZ, []frontend.Variable{c.Group}, c.PG, c.GroupRoot)
	limit := mul(bi(int64(6*m.D)), square(xmax))
	if mul(limit, bi(4)).Cmp(api.Compiler().Field()) >= 0 {
		return fmt.Errorf("unsafe old-score range")
	}
	selected := frontend.Variable(0)
	scores := make([]frontend.Variable, m.K)
	for g, center := range m.Centers {
		score := frontend.Variable(0)
		for d, v := range center {
			score = api.Add(score, square(bi(v)), api.Mul(-2*v, c.X[d]))
		}
		scores[g] = score
		selected = api.Add(selected, api.Mul(api.IsZero(api.Sub(c.Group, g)), score))
	}
	bc := cmp.NewBoundedComparator(api, limit, false)
	for _, score := range scores {
		api.AssertIsEqual(bc.IsLess(score, selected), 0)
	}
	api.AssertIsDifferent(c.Challenge, 0)
	api.AssertIsEqual(c.Previous, fingerprintMaskCircuit(api, m, c.CX, 0, c.GroupRoot, c.StatsRoot, c.Challenge, c.Index, c.PreviousMask))
	api.AssertIsEqual(c.Next, fingerprintMaskCircuit(api, m, c.CX, 0, c.GroupRoot, c.StatsRoot, c.Challenge, api.Add(c.Index, 1), c.NextMask))
	poly := frontend.Variable(0)
	for d := m.D - 1; d >= 0; d-- {
		poly = api.Add(api.Mul(poly, c.Challenge), c.X[d])
	}
	poly = api.Add(1, api.Mul(poly, c.Challenge))
	for g := 0; g < m.K; g++ {
		api.AssertIsEqual(c.Masked[g], api.Add(api.Mul(api.IsZero(api.Sub(c.Group, g)), poly), c.NextMask[g], api.Neg(c.PreviousMask[g])))
	}
	return nil
}
func FillBoundMomentsRow(p *BoundPrepared, g *GroupMoments, f *GroupFingerprintWitness, i int) *BoundMomentsRow {
	c := NewBoundMomentsRow(p.M)
	c.CX, c.GroupRoot, c.StatsRoot, c.Challenge, c.Index = p.CX, f.Commitment.Membership.Root(), f.Commitment.StatsRoot, f.Challenge, i
	c.Previous, c.Next, c.Group, c.SaltX, c.SaltZ = f.MaskHash[i], f.MaskHash[i+1], g.Groups[i], p.SaltX, f.Commitment.Salt
	for d, v := range p.X[i] {
		c.X[d] = v
	}
	for a, v := range p.TX.Path(i) {
		c.PX[a] = v
	}
	for a, v := range f.Commitment.Membership.Path(i) {
		c.PG[a] = v
	}
	for j := range c.Masked {
		c.Masked[j], c.PreviousMask[j], c.NextMask[j] = f.Masked[i][j], f.Masks[i][j], f.Masks[i+1][j]
	}
	return c
}
func MomentsRowPublic(m BoundModel, s BoundStatement, h GroupFingerprintHeader, row GroupFingerprintPublicRow) (*BoundMomentsRow, error) {
	cx, ch, _, _, _, err := s.Values(m)
	if err != nil {
		return nil, err
	}
	if m.Method != "affine" || ch.Sign() != 0 || row.Index < 0 || row.Index >= m.N || len(row.Masked) != m.K {
		return nil, fmt.Errorf("invalid moments row statement")
	}
	if err = fingerprintCanonical(h.GroupRoot, h.StatsRoot, h.Challenge, row.Previous, row.Next); err != nil {
		return nil, err
	}
	if h.Challenge.Sign() == 0 {
		return nil, fmt.Errorf("zero challenge")
	}
	if err = fingerprintCanonical(row.Masked...); err != nil {
		return nil, err
	}
	c := NewBoundMomentsRow(m)
	c.CX, c.GroupRoot, c.StatsRoot, c.Challenge, c.Index = cx, h.GroupRoot, h.StatsRoot, h.Challenge, row.Index
	c.Previous, c.Next = row.Previous, row.Next
	for g, v := range row.Masked {
		c.Masked[g] = v
	}
	return c, nil
}

// The relation proves an upper bound obtained by assigning each old group
// wholly to one new center. Native construction chooses the best target per
// group; proving its optimality is unnecessary for the sound improvement sign.
type BoundMomentsUpper struct {
	Link    BoundGroupFingerprintLink
	Centers [][]frontend.Variable `gnark:",public"`
	Upper   frontend.Variable     `gnark:",public"`
	Targets []frontend.Variable
}

func NewBoundMomentsUpper(m BoundModel) *BoundMomentsUpper {
	c := &BoundMomentsUpper{Link: *NewBoundGroupFingerprintLink(m), Centers: make([][]frontend.Variable, m.K), Targets: make([]frontend.Variable, m.K)}
	for j := range c.Centers {
		c.Centers[j] = make([]frontend.Variable, m.D)
	}
	return c
}
func (c *BoundMomentsUpper) Define(api frontend.API) error {
	m := c.Link.M
	if m.Method != "affine" || len(m.Basis) != 0 {
		return fmt.Errorf("moments-only schema required")
	}
	if err := c.Link.Define(api); err != nil {
		return err
	}
	api.AssertIsEqual(c.Link.CH, 0)
	rc := rangecheck.New(api)
	limit := m.Limits().X
	upper := frontend.Variable(0)
	for _, center := range c.Centers {
		for _, v := range center {
			rangeCheckSigned(api, rc, v, limit, false)
		}
	}
	for g := 0; g < m.K; g++ {
		rangeCheckUnsigned(api, rc, c.Targets[g], bi(int64(m.K-1)), false)
		norm := frontend.Variable(0)
		for d := 0; d < m.D; d++ {
			value := frontend.Variable(0)
			for j := 0; j < m.K; j++ {
				value = api.Add(value, api.Mul(api.IsZero(api.Sub(c.Targets[g], j)), c.Centers[j][d]))
			}
			norm = api.Add(norm, api.Sub(api.Mul(value, value), square(bi(m.Centers[g][d]))))
			upper = api.Add(upper, api.Mul(2, c.Link.Sums[g][d], api.Sub(m.Centers[g][d], value)))
		}
		upper = api.Add(upper, api.Mul(c.Link.Counts[g], norm))
	}
	rangeCheckSigned(api, rc, c.Upper, mul(bi(int64(4*m.N*m.D)), square(limit)), false)
	api.AssertIsEqual(c.Upper, upper)
	return nil
}
func FillBoundMomentsUpper(p *BoundPrepared, g *GroupMoments, f *GroupFingerprintWitness, centers [][]int64) (*BoundMomentsUpper, error) {
	if err := ValidateMomentsCenters(p.M, centers); err != nil {
		return nil, err
	}
	link, err := FillBoundGroupFingerprintLink(p, g, f)
	if err != nil {
		return nil, err
	}
	c := NewBoundMomentsUpper(p.M)
	c.Link = *link
	total := bi(0)
	for j, row := range centers {
		for d, v := range row {
			c.Centers[j][d] = v
		}
	}
	for group := 0; group < p.M.K; group++ {
		var best *big.Int
		winner := 0
		for j, row := range centers {
			a := bi(0)
			norm := bi(0)
			for d, v := range row {
				a.Add(a, mul(bi(2*(p.M.Centers[group][d]-v)), g.Sums[group][d]))
				norm.Add(norm, new(big.Int).Sub(square(bi(v)), square(bi(p.M.Centers[group][d]))))
			}
			a.Add(a, mul(g.Counts[group], norm))
			if best == nil || a.Cmp(best) < 0 {
				best = a
				winner = j
			}
		}
		c.Targets[group] = winner
		total.Add(total, best)
	}
	c.Upper = total
	return c, nil
}
func MomentsUpperPublic(m BoundModel, s BoundStatement, h GroupFingerprintHeader, sums []*big.Int, first, last *big.Int, centers [][]int64, upper *big.Int) (*BoundMomentsUpper, error) {
	link, err := GroupFingerprintLinkPublic(m, s, h, sums, first, last)
	if err != nil {
		return nil, err
	}
	if m.Method != "affine" || s.CH != "0" {
		return nil, fmt.Errorf("moments-only roots required")
	}
	if err = ValidateMomentsCenters(m, centers); err != nil {
		return nil, err
	}
	if upper == nil || new(big.Int).Abs(upper).Cmp(mul(bi(int64(4*m.N*m.D)), square(m.Limits().X))) > 0 {
		return nil, fmt.Errorf("upper out of range")
	}
	c := NewBoundMomentsUpper(m)
	c.Link = *link
	c.Upper = upper
	for j, row := range centers {
		for d, v := range row {
			c.Centers[j][d] = v
		}
	}
	return c, nil
}
