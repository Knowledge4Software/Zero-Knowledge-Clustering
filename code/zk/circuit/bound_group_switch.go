package circuit

// Experimental cost-change certificate: an exact assigned-group aggregate,
// plus authenticated intervals for switching away from each fixed group.
// These query relations require complete fingerprint preprocessing with an
// independently pinned post-commit auditor challenge. They do not replace it.
import (
	"fmt"
	"math/big"

	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/std/math/cmp"
	"github.com/consensys/gnark/std/rangecheck"
)

type GroupSwitchParameters struct {
	modelHash                      string
	oldZ, oldPairs, newZ, newPairs *PairedParameters
}

func ValidateGroupSwitchCenters(m BoundModel, centers [][]int64) error {
	if err := m.Validate(); err != nil {
		return err
	}
	if err := m.reusable(); err != nil {
		return err
	}
	if !m.HasEnergy() || m.TailWidth != 0 || len(centers) != m.K {
		return fmt.Errorf("fixed group/center correspondence and head energy required")
	}
	return validateNewCenters(m, centers)
}

func NewGroupSwitchParameters(m BoundModel, centers [][]int64) (*GroupSwitchParameters, error) {
	if err := ValidateGroupSwitchCenters(m, centers); err != nil {
		return nil, err
	}
	a, b, err := relativeParameters(m, m.Centers)
	if err != nil {
		return nil, err
	}
	c, d, err := relativeParameters(m, centers)
	if err != nil {
		return nil, err
	}
	return &GroupSwitchParameters{m.Hash().String(), a, b, c, d}, nil
}

func groupSwitchLimit(m BoundModel) *big.Int {
	_, _, _, limit := pairedLimits(m)
	return mul(limit, bi(16))
}

type BoundGroupSwitchBatch struct {
	ExactOld                                         bool                  `gnark:"-"`
	CompactGroup                                     bool                  `gnark:"-"`
	M                                                BoundModel            `gnark:"-"`
	Q                                                int                   `gnark:"-"`
	CX, CH, CE, CT, Context, GroupRoot, Lower, Upper frontend.Variable     `gnark:",public"`
	Indices                                          []frontend.Variable   `gnark:",public"`
	NewCoeff, NewRadius                              [][]frontend.Variable `gnark:",public"`
	NewConstant                                      []frontend.Variable   `gnark:",public"`
	H, PH, GroupPath                                 [][]frontend.Variable
	E, Radius, Group                                 []frontend.Variable
	Salt, GroupSalt                                  frontend.Variable
}

func NewBoundGroupSwitchBatch(m BoundModel, q int) *BoundGroupSwitchBatch {
	c := &BoundGroupSwitchBatch{M: m, Q: q, Indices: make([]frontend.Variable, q), NewCoeff: make([][]frontend.Variable, m.K), NewRadius: make([][]frontend.Variable, m.K), NewConstant: make([]frontend.Variable, m.K), H: make([][]frontend.Variable, q), PH: make([][]frontend.Variable, q), GroupPath: make([][]frontend.Variable, q), E: make([]frontend.Variable, q), Radius: make([]frontend.Variable, q), Group: make([]frontend.Variable, q)}
	for j := 0; j < m.K; j++ {
		c.NewCoeff[j] = make([]frontend.Variable, m.Width())
		c.NewRadius[j] = make([]frontend.Variable, m.K)
	}
	for i := 0; i < q; i++ {
		c.H[i] = make([]frontend.Variable, m.Width())
		c.PH[i] = make([]frontend.Variable, m.Depth())
		c.GroupPath[i] = make([]frontend.Variable, m.Depth())
	}
	return c
}

func NewBoundGroupSwitchBatchExact(m BoundModel, q int) *BoundGroupSwitchBatch {
	c := NewBoundGroupSwitchBatch(m, q)
	c.ExactOld = true
	return c
}

func NewBoundGroupSwitchBatchCompact(m BoundModel, q int) *BoundGroupSwitchBatch {
	c := NewBoundGroupSwitchBatchExact(m, q)
	c.CompactGroup = true
	c.GroupPath = nil
	return c
}

func (c *BoundGroupSwitchBatch) Define(api frontend.API) error {
	m := c.M
	if err := m.Validate(); err != nil {
		return err
	}
	if !m.HasEnergy() || m.TailWidth != 0 || c.Q < 1 || c.Q > m.N {
		return fmt.Errorf("invalid group switching shape")
	}
	radius, co, constant, _ := pairedLimits(m)
	limit := groupSwitchLimit(m)
	if mul(limit, bi(int64(64*(m.N+c.Q+m.K)))).Cmp(api.Compiler().Field()) >= 0 {
		return fmt.Errorf("unsafe group switching field bound")
	}
	oldZ, oldPairs, err := relativeParameters(m, m.Centers)
	if err != nil {
		return err
	}
	rc := rangecheck.New(api)
	bc := cmp.NewBoundedComparator(api, mul(limit, bi(2)), false)
	ic := cmp.NewBoundedComparator(api, bi(int64(m.N)), false)
	limits := m.Limits()
	boundBindContext(api, m, c.CX, c.CH, c.CE, c.CT, c.Context)
	for j := 0; j < m.K; j++ {
		for _, v := range c.NewCoeff[j] {
			rangeCheckSigned(api, rc, v, co, false)
		}
		rangeCheckSigned(api, rc, c.NewConstant[j], constant, false)
		for _, v := range c.NewRadius[j] {
			rangeCheckUnsigned(api, rc, v, mul(radius, bi(2)), false)
		}
	}
	t2 := square(bi(m.BasisScale))
	lower, upper := frontend.Variable(0), frontend.Variable(0)
	for i := 0; i < c.Q; i++ {
		if i > 0 {
			ic.AssertIsLess(c.Indices[i-1], c.Indices[i])
		}
		for _, v := range c.H[i] {
			rangeCheckSigned(api, rc, v, limits.H, false)
		}
		rangeCheckUnsigned(api, rc, c.E[i], limits.E, false)
		values := append(append([]frontend.Variable{}, c.H[i]...), c.E[i])
		if c.CompactGroup {
			values = append(values, c.Group[i])
		}
		boundOpen(api, m, boundH, c.Indices[i], c.Salt, values, c.PH[i], c.CH)
		rangeCheckUnsigned(api, rc, c.Group[i], bi(int64(m.K-1)), false)
		if c.CompactGroup {
			api.AssertIsEqual(c.GroupRoot, c.CH)
			api.AssertIsEqual(c.GroupSalt, c.Salt)
		} else {
			boundOpen(api, m, boundFingerprintGroup, c.Indices[i], c.GroupSalt, []frontend.Variable{c.Group[i]}, c.GroupPath[i], c.GroupRoot)
		}
		rangeCheckUnsigned(api, rc, c.Radius[i], radius, false)
		cmp.NewBoundedComparator(api, add(square(radius), bi(1)), false).AssertIsLess(c.E[i], api.Add(api.Mul(c.Radius[i], c.Radius[i]), 1))
		members := make([]frontend.Variable, m.K)
		for j := range members {
			members[j] = api.IsZero(api.Sub(c.Group[i], j))
		}
		minsLo, minsHi := make([]frontend.Variable, 2), make([]frontend.Variable, 2)
		for version := 0; version < 2; version++ {
			if c.ExactOld && version == 0 {
				// Scores below are shifted by limit; old b is exactly zero.
				minsLo[0], minsHi[0] = limit, limit
				continue
			}
			scores := make([]frontend.Variable, m.K)
			selected := frontend.Variable(0)
			for j := 0; j < m.K; j++ {
				var score frontend.Variable = c.NewConstant[j]
				if version == 0 {
					score = oldZ.Constant[2*j]
				}
				for a, v := range c.H[i] {
					var coefficient frontend.Variable = c.NewCoeff[j][a]
					if version == 0 {
						coefficient = oldZ.Coeff[2*j][a]
					}
					score = api.Add(score, api.Mul(coefficient, v))
				}
				scores[j] = score
				selected = api.Add(selected, api.Mul(members[j], score))
			}
			for j := 0; j < m.K; j++ {
				r := frontend.Variable(0)
				for g, member := range members {
					var v frontend.Variable = c.NewRadius[j][g]
					if version == 0 {
						v = oldPairs.Radius[j*m.K+g]
					}
					r = api.Add(r, api.Mul(member, v))
				}
				error := api.Mul(2, t2, c.Radius[i], r)
				gap := api.Sub(scores[j], selected)
				lo, hi := api.Add(limit, gap, api.Neg(error)), api.Add(limit, gap, error)
				if j == 0 {
					minsLo[version], minsHi[version] = lo, hi
				} else {
					minsLo[version] = api.Select(bc.IsLess(lo, minsLo[version]), lo, minsLo[version])
					minsHi[version] = api.Select(bc.IsLess(hi, minsHi[version]), hi, minsHi[version])
				}
			}
		}
		lower = api.Add(lower, api.Sub(minsLo[1], minsHi[0]))
		upper = api.Add(upper, api.Sub(minsHi[1], minsLo[0]))
	}
	maximum := mul(limit, bi(int64(2*c.Q)))
	rangeCheckSigned(api, rc, c.Lower, maximum, false)
	rangeCheckSigned(api, rc, c.Upper, maximum, false)
	api.AssertIsEqual(c.Lower, lower)
	api.AssertIsEqual(c.Upper, upper)
	return nil
}

func groupSwitchNative(p *BoundPrepared, params *GroupSwitchParameters, index, group int, exactOld bool) (*big.Int, *big.Int) {
	rowRadius := ceilSqrt(p.E[index])
	t2 := square(bi(p.M.BasisScale))
	lo, hi := make([]*big.Int, 2), make([]*big.Int, 2)
	for version := 0; version < 2; version++ {
		if exactOld && version == 0 {
			lo[0], hi[0] = bi(0), bi(0)
			continue
		}
		z, pairs := params.newZ, params.newPairs
		if version == 0 {
			z, pairs = params.oldZ, params.oldPairs
		}
		scores := make([]*big.Int, p.M.K)
		for j := range scores {
			scores[j] = new(big.Int).Set(z.Constant[2*j])
			for a, v := range p.H[index] {
				scores[j].Add(scores[j], mul(z.Coeff[2*j][a], v))
			}
		}
		for j := range scores {
			gap := new(big.Int).Sub(scores[j], scores[group])
			error := mul(mul(mul(bi(2), t2), rowRadius), pairs.Radius[j*p.M.K+group])
			a, b := new(big.Int).Sub(gap, error), add(gap, error)
			if j == 0 || a.Cmp(lo[version]) < 0 {
				lo[version] = a
			}
			if j == 0 || b.Cmp(hi[version]) < 0 {
				hi[version] = b
			}
		}
	}
	return new(big.Int).Sub(lo[1], hi[0]), new(big.Int).Sub(hi[1], lo[0])
}

func groupSwitchPublicParams(c *BoundGroupSwitchBatch, p *GroupSwitchParameters) {
	for j := 0; j < c.M.K; j++ {
		c.NewConstant[j] = p.newZ.Constant[2*j]
		for a, v := range p.newZ.Coeff[2*j] {
			c.NewCoeff[j][a] = v
		}
		for g := 0; g < c.M.K; g++ {
			c.NewRadius[j][g] = p.newPairs.Radius[j*c.M.K+g]
		}
	}
}

func FillBoundGroupSwitchBatch(p *BoundPrepared, g *GroupMoments, commit *GroupFingerprintCommitment, params *GroupSwitchParameters, indices []int) (*BoundGroupSwitchBatch, error) {
	return FillBoundGroupSwitchBatchMode(p, g, commit, params, indices, false)
}

func FillBoundGroupSwitchBatchMode(p *BoundPrepared, g *GroupMoments, commit *GroupFingerprintCommitment, params *GroupSwitchParameters, indices []int, exactOld bool) (*BoundGroupSwitchBatch, error) {
	if params == nil || params.modelHash != p.M.Hash().String() {
		return nil, fmt.Errorf("group switch model mismatch")
	}
	if err := validatePublicIndices(p.M, indices); err != nil {
		return nil, err
	}
	c := NewBoundGroupSwitchBatch(p.M, len(indices))
	c.ExactOld = exactOld
	c.CX, c.CH, c.CE, c.CT, c.Context, c.GroupRoot = p.CX, p.CH, p.CE, p.CT, p.Context, commit.Membership.Root()
	c.Salt, c.GroupSalt = p.SaltZ, commit.Salt
	groupSwitchPublicParams(c, params)
	lower, upper := bi(0), bi(0)
	for a, i := range indices {
		c.Indices[a], c.E[a], c.Radius[a], c.Group[a] = i, p.E[i], ceilSqrt(p.E[i]), g.Groups[i]
		for d, v := range p.H[i] {
			c.H[a][d] = v
		}
		for d, v := range p.TH.Path(i) {
			c.PH[a][d] = v
		}
		for d, v := range commit.Membership.Path(i) {
			c.GroupPath[a][d] = v
		}
		lo, hi := groupSwitchNative(p, params, i, g.Groups[i], exactOld)
		lower.Add(lower, lo)
		upper.Add(upper, hi)
	}
	c.Lower, c.Upper = lower, upper
	return c, nil
}

func GroupSwitchPublic(m BoundModel, s BoundStatement, groupRoot *big.Int, params *GroupSwitchParameters, indices []int, lower, upper *big.Int) (*BoundGroupSwitchBatch, error) {
	if params == nil || params.modelHash != m.Hash().String() {
		return nil, fmt.Errorf("group switch model mismatch")
	}
	if err := validatePublicIndices(m, indices); err != nil {
		return nil, err
	}
	if err := fingerprintCanonical(groupRoot); err != nil {
		return nil, err
	}
	maximum := mul(groupSwitchLimit(m), bi(int64(2*len(indices))))
	if lower == nil || upper == nil || lower.Cmp(upper) > 0 || new(big.Int).Abs(lower).Cmp(maximum) > 0 || new(big.Int).Abs(upper).Cmp(maximum) > 0 {
		return nil, fmt.Errorf("invalid switching correction bounds")
	}
	c := NewBoundGroupSwitchBatch(m, len(indices))
	cx, ch, ce, ct, context, err := s.Values(m)
	if err != nil {
		return nil, err
	}
	c.CX, c.CH, c.CE, c.CT, c.Context, c.GroupRoot, c.Lower, c.Upper = cx, ch, ce, ct, context, groupRoot, lower, upper
	groupSwitchPublicParams(c, params)
	for a, i := range indices {
		c.Indices[a] = i
	}
	return c, nil
}

type BoundGroupAssignedChange struct {
	M                                    BoundModel            `gnark:"-"`
	CX, CH, GroupRoot, StatsRoot, Change frontend.Variable     `gnark:",public"`
	Centers                              [][]frontend.Variable `gnark:",public"`
	Counts                               []frontend.Variable
	Sums                                 [][]frontend.Variable
	Salt                                 frontend.Variable
}

func NewBoundGroupAssignedChange(m BoundModel) *BoundGroupAssignedChange {
	c := &BoundGroupAssignedChange{M: m, Centers: make([][]frontend.Variable, m.K), Counts: make([]frontend.Variable, m.K), Sums: make([][]frontend.Variable, m.K)}
	for j := 0; j < m.K; j++ {
		c.Centers[j] = make([]frontend.Variable, m.D)
		c.Sums[j] = make([]frontend.Variable, m.D)
	}
	return c
}

func (c *BoundGroupAssignedChange) Define(api frontend.API) error {
	m := c.M
	if err := m.Validate(); err != nil {
		return err
	}
	if !m.HasEnergy() || m.TailWidth != 0 {
		return fmt.Errorf("group assigned change requires head/energy")
	}
	rc := rangecheck.New(api)
	values := []frontend.Variable{boundFingerprintStats, m.Hash(), c.CX, c.CH, c.GroupRoot, c.Salt}
	count, change := frontend.Variable(0), frontend.Variable(0)
	for j := 0; j < m.K; j++ {
		rangeCheckUnsigned(api, rc, c.Counts[j], bi(int64(m.N)), false)
		count = api.Add(count, c.Counts[j])
		values = append(values, c.Counts[j])
		norm := frontend.Variable(0)
		for d, sum := range c.Sums[j] {
			rangeCheckSigned(api, rc, sum, mul(bi(int64(m.N)), m.Limits().X), false)
			values = append(values, sum)
			v := c.Centers[j][d]
			rangeCheckSigned(api, rc, v, m.Limits().X, false)
			norm = api.Add(norm, api.Sub(api.Mul(v, v), square(bi(m.Centers[j][d]))))
			change = api.Add(change, api.Mul(2, sum, api.Sub(m.Centers[j][d], v)))
		}
		change = api.Add(change, api.Mul(c.Counts[j], norm))
	}
	api.AssertIsEqual(count, m.N)
	api.AssertIsEqual(c.StatsRoot, boundHash(api, values...))
	maximum := mul(bi(int64(4*m.N*m.D)), square(m.Limits().X))
	rangeCheckSigned(api, rc, c.Change, maximum, false)
	api.AssertIsEqual(c.Change, change)
	return nil
}

func FillBoundGroupAssignedChange(p *BoundPrepared, g *GroupMoments, commit *GroupFingerprintCommitment, centers [][]int64) (*BoundGroupAssignedChange, error) {
	if len(centers) != p.M.K {
		return nil, fmt.Errorf("fixed group center count required")
	}
	if err := validateNewCenters(p.M, centers); err != nil {
		return nil, err
	}
	c := NewBoundGroupAssignedChange(p.M)
	c.CX, c.CH, c.GroupRoot, c.StatsRoot, c.Salt = p.CX, p.CH, commit.Membership.Root(), commit.StatsRoot, commit.Salt
	total := bi(0)
	for j := 0; j < p.M.K; j++ {
		c.Counts[j] = g.Counts[j]
		norm := bi(0)
		for d, v := range centers[j] {
			c.Centers[j][d] = v
			c.Sums[j][d] = g.Sums[j][d]
			norm.Add(norm, new(big.Int).Sub(square(bi(v)), square(bi(p.M.Centers[j][d]))))
			total.Add(total, mul(mul(bi(2), g.Sums[j][d]), bi(p.M.Centers[j][d]-v)))
		}
		total.Add(total, mul(g.Counts[j], norm))
	}
	c.Change = total
	return c, nil
}

func GroupAssignedChangePublic(m BoundModel, s BoundStatement, header GroupFingerprintHeader, centers [][]int64, change *big.Int) (*BoundGroupAssignedChange, error) {
	if err := m.Validate(); err != nil {
		return nil, err
	}
	if len(centers) != m.K {
		return nil, fmt.Errorf("fixed group center count required")
	}
	if err := validateNewCenters(m, centers); err != nil {
		return nil, err
	}
	if err := fingerprintCanonical(header.GroupRoot, header.StatsRoot); err != nil {
		return nil, err
	}
	maximum := mul(bi(int64(4*m.N*m.D)), square(m.Limits().X))
	if change == nil || new(big.Int).Abs(change).Cmp(maximum) > 0 {
		return nil, fmt.Errorf("invalid assigned change")
	}
	cx, ch, _, _, _, err := s.Values(m)
	if err != nil {
		return nil, err
	}
	c := NewBoundGroupAssignedChange(m)
	c.CX, c.CH, c.GroupRoot, c.StatsRoot, c.Change = cx, ch, header.GroupRoot, header.StatsRoot, change
	for j := range centers {
		for d, v := range centers[j] {
			c.Centers[j][d] = v
		}
	}
	return c, nil
}
