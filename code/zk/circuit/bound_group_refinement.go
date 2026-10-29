package circuit

// Optional exact corrections for a small, explicitly indexed subset. The
// complete compact preprocessing and ordinary batch proofs remain mandatory.
// Corrections use CANONICAL minimal row radii. Ordinary proofs may use larger
// valid radii: monotonicity makes the resulting corrected total conservative.
// This relation alone is not a full-catalog decision certificate.
import (
	"fmt"
	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/std/math/cmp"
	"github.com/consensys/gnark/std/rangecheck"
	"math/big"
)

type BoundGroupRefinement struct {
	M                                           BoundModel            `gnark:"-"`
	Q                                           int                   `gnark:"-"`
	CX, CH, CE, CT, Context, GroupRoot          frontend.Variable     `gnark:",public"`
	Indices                                     []frontend.Variable   `gnark:",public"`
	Centers                                     [][]frontend.Variable `gnark:",public"`
	NewCoeff, NewRadius                         [][]frontend.Variable `gnark:",public"`
	NewConstant                                 []frontend.Variable   `gnark:",public"`
	CorrectionLower, CorrectionUpper            frontend.Variable     `gnark:",public"`
	X, PX, H, PH                                [][]frontend.Variable
	E, Radius, Group                            []frontend.Variable
	SaltX, Salt, CanonicalLower, CanonicalUpper frontend.Variable
}

func NewBoundGroupRefinement(m BoundModel, q int) *BoundGroupRefinement {
	b := NewBoundGroupSwitchBatchCompact(m, q)
	c := &BoundGroupRefinement{M: m, Q: q, Indices: b.Indices, NewCoeff: b.NewCoeff,
		NewRadius: b.NewRadius, NewConstant: b.NewConstant, H: b.H, PH: b.PH,
		E: b.E, Radius: b.Radius, Group: b.Group,
		X: make([][]frontend.Variable, q), PX: make([][]frontend.Variable, q),
		Centers: make([][]frontend.Variable, m.K)}
	for i := 0; i < q; i++ {
		c.X[i] = make([]frontend.Variable, m.D)
		c.PX[i] = make([]frontend.Variable, m.Depth())
	}
	for j := range c.Centers {
		c.Centers[j] = make([]frontend.Variable, m.D)
	}
	return c
}

func (c *BoundGroupRefinement) switchRelation() *BoundGroupSwitchBatch {
	return &BoundGroupSwitchBatch{M: c.M, Q: c.Q, ExactOld: true, CompactGroup: true,
		CX: c.CX, CH: c.CH, CE: c.CE, CT: c.CT, Context: c.Context, GroupRoot: c.GroupRoot,
		Indices: c.Indices, NewCoeff: c.NewCoeff, NewRadius: c.NewRadius, NewConstant: c.NewConstant,
		H: c.H, PH: c.PH, E: c.E, Radius: c.Radius, Group: c.Group, Salt: c.Salt, GroupSalt: c.Salt,
		Lower: c.CanonicalLower, Upper: c.CanonicalUpper}
}

func (c *BoundGroupRefinement) Define(api frontend.API) error {
	if c.Q < 1 || c.Q > 16 || c.Q > c.M.N {
		return fmt.Errorf("invalid bounded refinement capacity")
	}
	b := c.switchRelation()
	if err := b.Define(api); err != nil {
		return err
	}
	m := c.M
	rc := rangecheck.New(api)
	limits := m.Limits()
	radius, _, _, _ := pairedLimits(m)
	radiusCmp := cmp.NewBoundedComparator(api, add(square(radius), bi(1)), false)
	scoreBound := mul(bi(int64(3*m.D)), square(limits.X))
	scoreCmp := cmp.NewBoundedComparator(api, mul(scoreBound, bi(2)), false)
	for _, center := range c.Centers {
		for _, v := range center {
			rangeCheckSigned(api, rc, v, limits.X, false)
		}
	}
	exact := frontend.Variable(0)
	for i := 0; i < c.Q; i++ {
		for _, v := range c.X[i] {
			rangeCheckSigned(api, rc, v, limits.X, false)
		}
		boundOpen(api, m, boundX, c.Indices[i], c.SaltX, c.X[i], c.PX[i], c.CX)
		// E <= r^2 is already checked by the switching relation. Enforce the
		// other half of r=ceil(sqrt(E)), including the E=r=0 corner case.
		zero := api.IsZero(c.Radius[i])
		api.AssertIsEqual(zero, api.IsZero(c.E[i]))
		previousSquare := api.Mul(api.Sub(c.Radius[i], 1), api.Sub(c.Radius[i], 1), api.Sub(1, zero))
		radiusCmp.AssertIsLess(previousSquare, api.Add(c.E[i], zero))
		selected := frontend.Variable(0)
		minimum := frontend.Variable(0)
		for j, center := range c.Centers {
			score := frontend.Variable(scoreBound)
			for d, v := range center {
				score = api.Add(score, api.Mul(v, v), api.Neg(api.Mul(2, c.X[i][d], v)))
			}
			selected = api.Add(selected, api.Mul(api.IsZero(api.Sub(c.Group[i], j)), score))
			if j == 0 {
				minimum = score
			} else {
				minimum = api.Select(scoreCmp.IsLess(score, minimum), score, minimum)
			}
		}
		exact = api.Add(exact, api.Sub(minimum, selected))
	}
	t6 := new(big.Int).Exp(bi(m.BasisScale), bi(6), nil)
	api.AssertIsEqual(c.CorrectionLower, api.Sub(api.Mul(exact, t6), c.CanonicalLower))
	api.AssertIsEqual(c.CorrectionUpper, api.Sub(api.Mul(exact, t6), c.CanonicalUpper))
	maximum := mul(groupSwitchLimit(m), bi(int64(4*c.Q)))
	rangeCheckUnsigned(api, rc, c.CorrectionLower, maximum, false)
	rangeCheckSigned(api, rc, c.CorrectionUpper, maximum, false)
	// Tightening must move the lower bound upwards and upper downwards.
	rangeCheckUnsigned(api, rc, api.Neg(c.CorrectionUpper), maximum, false)
	return nil
}

func FillBoundGroupRefinement(p *BoundPrepared, g *GroupMoments, commit *GroupFingerprintCommitment,
	centers [][]int64, indices []int) (*BoundGroupRefinement, error) {
	params, err := NewGroupSwitchParameters(p.M, centers)
	if err != nil {
		return nil, err
	}
	b, err := FillBoundGroupSwitchBatchCompact(p, g, commit, params, indices)
	if err != nil {
		return nil, err
	}
	c := NewBoundGroupRefinement(p.M, len(indices))
	c.CX, c.CH, c.CE, c.CT, c.Context, c.GroupRoot = b.CX, b.CH, b.CE, b.CT, b.Context, b.GroupRoot
	c.Indices, c.NewCoeff, c.NewRadius, c.NewConstant = b.Indices, b.NewCoeff, b.NewRadius, b.NewConstant
	c.H, c.PH, c.E, c.Radius, c.Group, c.Salt, c.SaltX = b.H, b.PH, b.E, b.Radius, b.Group, p.SaltZ, p.SaltX
	c.CanonicalLower, c.CanonicalUpper = b.Lower, b.Upper
	exact := bi(0)
	for i, index := range indices {
		for d, v := range p.X[index] {
			c.X[i][d] = v
		}
		for d, v := range p.TX.Path(index) {
			c.PX[i][d] = v
		}
		var minimum, selected *big.Int
		for j, center := range centers {
			distance := bi(0)
			for d, v := range center {
				distance.Add(distance, square(new(big.Int).Sub(bi(p.X[index][d]), bi(v))))
			}
			if minimum == nil || distance.Cmp(minimum) < 0 {
				minimum = new(big.Int).Set(distance)
			}
			if j == g.Groups[index] {
				selected = new(big.Int).Set(distance)
			}
		}
		exact.Add(exact, new(big.Int).Sub(minimum, selected))
	}
	for j, row := range centers {
		for d, v := range row {
			c.Centers[j][d] = v
		}
	}
	exact.Mul(exact, new(big.Int).Exp(bi(p.M.BasisScale), bi(6), nil))
	c.CorrectionLower = new(big.Int).Sub(exact, b.Lower.(*big.Int))
	c.CorrectionUpper = new(big.Int).Sub(exact, b.Upper.(*big.Int))
	return c, nil
}

func GroupRefinementPublic(m BoundModel, s BoundStatement, groupRoot *big.Int,
	centers [][]int64, indices []int, lower, upper *big.Int) (*BoundGroupRefinement, error) {
	params, err := NewGroupSwitchParameters(m, centers)
	if err != nil {
		return nil, err
	}
	b, err := GroupSwitchPublic(m, s, groupRoot, params, indices, bi(0), bi(0))
	if err != nil {
		return nil, err
	}
	maximum := mul(groupSwitchLimit(m), bi(int64(4*len(indices))))
	if len(indices) > 16 || lower == nil || upper == nil || lower.Sign() < 0 || upper.Sign() > 0 ||
		lower.Cmp(maximum) > 0 || new(big.Int).Abs(upper).Cmp(maximum) > 0 {
		return nil, fmt.Errorf("invalid refinement correction")
	}
	c := NewBoundGroupRefinement(m, len(indices))
	c.CX, c.CH, c.CE, c.CT, c.Context, c.GroupRoot = b.CX, b.CH, b.CE, b.CT, b.Context, b.GroupRoot
	c.Indices, c.NewCoeff, c.NewRadius, c.NewConstant = b.Indices, b.NewCoeff, b.NewRadius, b.NewConstant
	c.CorrectionLower, c.CorrectionUpper = lower, upper
	for j, row := range centers {
		for d, v := range row {
			c.Centers[j][d] = v
		}
	}
	return c, nil
}
