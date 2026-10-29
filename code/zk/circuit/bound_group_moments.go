package circuit

// Fixed-full-catalog group-moment audit. This prototype deliberately proves
// complete preprocessing in one circuit; it makes no production scaling claim.
import (
	"fmt"
	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/std/math/cmp"
	"github.com/consensys/gnark/std/rangecheck"
	"math/big"
)

const boundGroupDomain = 109

type GroupMoments struct {
	Groups           []int
	Sums             [][]*big.Int
	Counts           []*big.Int
	Salt, Commitment *big.Int
}

func PrepareGroupMoments(p *BoundPrepared, salt *big.Int) (*GroupMoments, error) {
	indices := make([]int, p.M.N)
	for i := range indices {
		indices[i] = i
	}
	h, e, err := ReuseCenterStats(p.M, p.M.Centers)
	if err != nil {
		return nil, err
	}
	labels, _, _, err := p.ReuseAnswer(h, e, indices)
	if err != nil {
		return nil, err
	}
	return PrepareGroupMomentsForLabels(p, salt, labels)
}

// The exact-old profile computes labels during fixed-catalog preprocessing.
// Its row proof must independently enforce the corresponding old argmin.
func PrepareExactOldGroupMoments(p *BoundPrepared, salt *big.Int) (*GroupMoments, error) {
	labels := make([]int, p.M.N)
	for i, row := range p.X {
		var best *big.Int
		for j, center := range p.M.Centers {
			distance := bi(0)
			for d, value := range row {
				distance.Add(distance, square(bi(value-center[d])))
			}
			if best == nil || distance.Cmp(best) < 0 {
				best = distance
				labels[i] = j
			}
		}
	}
	return PrepareGroupMomentsForLabels(p, salt, labels)
}

func PrepareGroupMomentsForLabels(p *BoundPrepared, salt *big.Int, labels []int) (*GroupMoments, error) {
	if len(labels) != p.M.N {
		return nil, fmt.Errorf("fixed group labels do not cover catalog")
	}
	for _, label := range labels {
		if label < 0 || label >= p.M.K {
			return nil, fmt.Errorf("invalid fixed group")
		}
	}
	g := &GroupMoments{Groups: labels, Sums: make([][]*big.Int, p.M.K), Counts: make([]*big.Int, p.M.K), Salt: salt}
	for j := range g.Sums {
		g.Counts[j] = bi(0)
		g.Sums[j] = make([]*big.Int, p.M.D)
		for d := range g.Sums[j] {
			g.Sums[j][d] = bi(0)
		}
	}
	for i, row := range p.X {
		j := labels[i]
		g.Counts[j].Add(g.Counts[j], bi(1))
		for d, v := range row {
			g.Sums[j][d].Add(g.Sums[j][d], bi(v))
		}
	}
	values := []*big.Int{bi(boundGroupDomain), p.M.Hash(), p.CX, p.CH, salt}
	for _, v := range labels {
		values = append(values, bi(int64(v)))
	}
	for j := range g.Sums {
		values = append(values, g.Counts[j])
		values = append(values, g.Sums[j]...)
	}
	g.Commitment = boundNativeHash(values...)
	return g, nil
}

type GroupCatalog struct {
	Groups, Counts []frontend.Variable
	Sums           [][]frontend.Variable
	Salt           frontend.Variable
}

func newGroupCatalog(m BoundModel) GroupCatalog {
	g := GroupCatalog{Groups: make([]frontend.Variable, m.N), Counts: make([]frontend.Variable, m.K), Sums: make([][]frontend.Variable, m.K)}
	for j := range g.Sums {
		g.Sums[j] = make([]frontend.Variable, m.D)
	}
	return g
}
func (g *GroupCatalog) constrain(api frontend.API, m BoundModel, cx, ch, commit frontend.Variable) {
	rc := rangecheck.New(api)
	values := []frontend.Variable{boundGroupDomain, m.Hash(), cx, ch, g.Salt}
	for _, v := range g.Groups {
		rangeCheckUnsigned(api, rc, v, bi(int64(m.K-1)), false)
		values = append(values, v)
	}
	total := frontend.Variable(0)
	for j := range g.Sums {
		rangeCheckUnsigned(api, rc, g.Counts[j], bi(int64(m.N)), false)
		total = api.Add(total, g.Counts[j])
		values = append(values, g.Counts[j])
		for _, v := range g.Sums[j] {
			rangeCheckSigned(api, rc, v, mul(bi(int64(m.N)), m.Limits().X), false)
			values = append(values, v)
		}
	}
	api.AssertIsEqual(total, m.N)
	api.AssertIsEqual(boundHash(api, values...), commit)
}
func fillGroupCatalog(c *GroupCatalog, g *GroupMoments) {
	c.Salt = g.Salt
	for i, v := range g.Groups {
		c.Groups[i] = v
	}
	for j := range g.Sums {
		c.Counts[j] = g.Counts[j]
		for d, v := range g.Sums[j] {
			c.Sums[j][d] = v
		}
	}
}

type BoundGroupPreprocess struct {
	M                                  BoundModel        `gnark:"-"`
	CX, CH, CE, CT, Context, GroupRoot frontend.Variable `gnark:",public"`
	Rows                               []groupBindingRow
	Catalog                            GroupCatalog
}

// Embedding BoundPreprocess directly would add its row public fields to the
// schema. This wrapper keeps row witnesses private and uses fixed indices.
type groupBindingRow struct {
	X, H, Dot, PX, PH []frontend.Variable
	E, SaltX, SaltZ   frontend.Variable
}

func NewBoundGroupPreprocess(m BoundModel) *BoundGroupPreprocess {
	c := &BoundGroupPreprocess{M: m, Rows: make([]groupBindingRow, m.N), Catalog: newGroupCatalog(m)}
	for i := range c.Rows {
		c.Rows[i] = groupBindingRow{X: make([]frontend.Variable, m.D), H: make([]frontend.Variable, m.Width()), Dot: make([]frontend.Variable, len(m.Basis)), PX: make([]frontend.Variable, m.Depth()), PH: make([]frontend.Variable, m.Depth())}
	}
	return c
}
func (c *BoundGroupPreprocess) Define(api frontend.API) error {
	m := c.M
	if err := m.Validate(); err != nil {
		return err
	}
	if err := m.reusable(); err != nil {
		return err
	}
	if m.TailWidth != 0 {
		return fmt.Errorf("group moment prototype requires head/energy without a tail")
	}
	// Fixed positions supply exact complete coverage; no prover-chosen row list.
	for i := range c.Rows {
		r := &c.Rows[i]
		row := &BoundPreprocess{M: m, CX: c.CX, CH: c.CH, CE: c.CE, CT: c.CT, Context: c.Context, Index: i, X: r.X, H: r.H, Dot: r.Dot, PX: r.PX, PH: r.PH, E: r.E, SaltX: r.SaltX, SaltZ: r.SaltZ}
		if err := row.Define(api); err != nil {
			return err
		}
	}
	c.Catalog.constrain(api, m, c.CX, c.CH, c.GroupRoot)
	for j := 0; j < m.K; j++ {
		count := frontend.Variable(0)
		sums := make([]frontend.Variable, m.D)
		for d := range sums {
			sums[d] = 0
		}
		for i := range c.Rows {
			member := api.IsZero(api.Sub(c.Catalog.Groups[i], j))
			count = api.Add(count, member)
			for d, v := range c.Rows[i].X {
				sums[d] = api.Add(sums[d], api.Mul(member, v))
			}
		}
		api.AssertIsEqual(c.Catalog.Counts[j], count)
		for d, v := range sums {
			api.AssertIsEqual(c.Catalog.Sums[j][d], v)
		}
	}
	return nil
}
func FillBoundGroupPreprocess(p *BoundPrepared, g *GroupMoments) *BoundGroupPreprocess {
	c := NewBoundGroupPreprocess(p.M)
	c.CX, c.CH, c.CE, c.CT, c.Context, c.GroupRoot = p.CX, p.CH, p.CE, p.CT, p.Context, g.Commitment
	for i := range c.Rows {
		r := FillBoundPreprocess(p, i)
		c.Rows[i] = groupBindingRow{X: r.X, H: r.H, Dot: r.Dot, PX: r.PX, PH: r.PH, E: r.E, SaltX: r.SaltX, SaltZ: r.SaltZ}
	}
	fillGroupCatalog(&c.Catalog, g)
	return c
}

type BoundGroupQuery struct {
	M                                          BoundModel            `gnark:"-"`
	Capacity                                   int                   `gnark:"-"`
	CX, CH, CE, CT, Context, GroupRoot, Change frontend.Variable     `gnark:",public"`
	Centers, CHead                             [][]frontend.Variable `gnark:",public"`
	COff, CenterRadius, Indices, Active        []frontend.Variable   `gnark:",public"`
	Catalog                                    GroupCatalog
	H, PH, X, PX                               [][]frontend.Variable
	E, RowRadius                               []frontend.Variable
	SaltX, SaltZ                               frontend.Variable
}

func NewBoundGroupQuery(m BoundModel, capacity int) *BoundGroupQuery {
	c := &BoundGroupQuery{M: m, Capacity: capacity, Catalog: newGroupCatalog(m), Centers: make([][]frontend.Variable, m.K), CHead: make([][]frontend.Variable, m.K), COff: make([]frontend.Variable, m.K), CenterRadius: make([]frontend.Variable, m.K), Indices: make([]frontend.Variable, capacity), Active: make([]frontend.Variable, capacity), H: make([][]frontend.Variable, m.N), PH: make([][]frontend.Variable, m.N), E: make([]frontend.Variable, m.N), RowRadius: make([]frontend.Variable, m.N), X: make([][]frontend.Variable, capacity), PX: make([][]frontend.Variable, capacity)}
	for j := range c.Centers {
		c.Centers[j] = make([]frontend.Variable, m.D)
		c.CHead[j] = make([]frontend.Variable, m.Width())
	}
	for i := range c.H {
		c.H[i] = make([]frontend.Variable, m.Width())
		c.PH[i] = make([]frontend.Variable, m.Depth())
	}
	for i := range c.X {
		c.X[i] = make([]frontend.Variable, m.D)
		c.PX[i] = make([]frontend.Variable, m.Depth())
	}
	return c
}

func (c *BoundGroupQuery) Define(api frontend.API) error {
	m := c.M
	if err := m.Validate(); err != nil {
		return err
	}
	if err := m.reusable(); err != nil {
		return err
	}
	if m.TailWidth != 0 {
		return fmt.Errorf("group moment prototype requires head/energy without a tail")
	}
	if c.Capacity < 0 || c.Capacity > m.N {
		return fmt.Errorf("invalid correction budget")
	}
	l := m.Limits()
	radius, _, limit := certificateBounds(m)
	distanceMax := mul(bi(int64(4*m.D)), square(l.X))
	if mul(add(limit, distanceMax), bi(int64(32*m.N))).Cmp(api.Compiler().Field()) >= 0 {
		return fmt.Errorf("unsafe group aggregate bound")
	}
	rc := rangecheck.New(api)
	boundBindContext(api, m, c.CX, c.CH, c.CE, c.CT, c.Context)
	c.Catalog.constrain(api, m, c.CX, c.CH, c.GroupRoot)
	rootCmp := cmp.NewBoundedComparator(api, add(square(radius), bi(1)), false)
	bc := cmp.NewBoundedComparator(api, limit, false)
	dc := cmp.NewBoundedComparator(api, add(distanceMax, bi(1)), false)
	ic := cmp.NewBoundedComparator(api, bi(int64(m.N)), false)
	oldH, oldE, err := ReuseCenterStats(m, m.Centers)
	if err != nil {
		return err
	}
	t2 := square(bi(m.BasisScale))
	gram := m.GramDefect()
	for j := range c.Centers {
		for _, v := range c.Centers[j] {
			rangeCheckSigned(api, rc, v, l.X, false)
		}
		for _, v := range c.CHead[j] {
			rangeCheckSigned(api, rc, v, l.H, false)
		}
		rangeCheckUnsigned(api, rc, c.COff[j], l.E, false)
		rangeCheckUnsigned(api, rc, c.CenterRadius[j], radius, false)
		rootCmp.AssertIsLess(c.COff[j], api.Add(api.Mul(c.CenterRadius[j], c.CenterRadius[j]), 1))
	}
	for a := range c.Indices {
		api.AssertIsBoolean(c.Active[a])
		rangeCheckUnsigned(api, rc, c.Indices[a], bi(int64(m.N-1)), false)
		api.AssertIsEqual(api.Mul(api.Sub(1, c.Active[a]), c.Indices[a]), 0)
		if a > 0 {
			api.AssertIsEqual(api.Mul(c.Active[a], api.Sub(1, c.Active[a-1])), 0)
			api.AssertIsEqual(api.Mul(c.Active[a], api.Sub(1, ic.IsLess(c.Indices[a-1], c.Indices[a]))), 0)
		}
	}
	for i := range c.H {
		for _, v := range c.H[i] {
			rangeCheckSigned(api, rc, v, l.H, false)
		}
		rangeCheckUnsigned(api, rc, c.E[i], l.E, false)
		boundOpen(api, m, boundH, i, c.SaltZ, append(append([]frontend.Variable{}, c.H[i]...), c.E[i]), c.PH[i], c.CH)
		rangeCheckUnsigned(api, rc, c.RowRadius[i], radius, false)
		rootCmp.AssertIsLess(c.E[i], api.Add(api.Mul(c.RowRadius[i], c.RowRadius[i]), 1))
		covered := frontend.Variable(0)
		for a, idx := range c.Indices {
			covered = api.Add(covered, api.Mul(c.Active[a], api.IsZero(api.Sub(idx, i))))
		}
		api.AssertIsBoolean(covered)
		stable := api.Sub(1, covered)
		for version := 0; version < 2; version++ {
			distances := make([]frontend.Variable, m.K)
			errors := make([]frontend.Variable, m.K)
			chosenD, chosenE := frontend.Variable(0), frontend.Variable(0)
			for j := 0; j < m.K; j++ {
				h2 := frontend.Variable(0)
				for d, h := range c.H[i] {
					var center frontend.Variable = c.CHead[j][d]
					if version == 0 {
						center = oldH[j][d]
					}
					diff := api.Sub(h, center)
					h2 = api.Add(h2, api.Mul(diff, diff))
				}
				var energy frontend.Variable = c.COff[j]
				var cr frontend.Variable = c.CenterRadius[j]
				if version == 0 {
					energy = oldE[j]
					cr = ceilSqrt(oldE[j])
				}
				distances[j] = api.Mul(t2, api.Add(h2, c.E[i], energy))
				errors[j] = api.Add(api.Mul(2, t2, c.RowRadius[i], cr), api.Mul(gram, h2))
				member := api.IsZero(api.Sub(c.Catalog.Groups[i], j))
				chosenD = api.Add(chosenD, api.Mul(member, distances[j]))
				chosenE = api.Add(chosenE, api.Mul(member, errors[j]))
			}
			for j := 0; j < m.K; j++ {
				other := api.Sub(1, api.IsZero(api.Sub(c.Catalog.Groups[i], j)))
				strict := bc.IsLess(api.Add(chosenE, errors[j]), api.Sub(distances[j], chosenD))
				api.AssertIsEqual(api.Mul(stable, other, api.Sub(1, strict)), 0)
			}
		}
	}
	norms := make([]frontend.Variable, m.K)
	total := frontend.Variable(0)
	for j := 0; j < m.K; j++ {
		norms[j] = 0
		dot := frontend.Variable(0)
		for d, v := range c.Centers[j] {
			norms[j] = api.Add(norms[j], api.Sub(api.Mul(v, v), square(bi(m.Centers[j][d]))))
			dot = api.Add(dot, api.Mul(2, c.Catalog.Sums[j][d], api.Sub(m.Centers[j][d], v)))
		}
		total = api.Add(total, dot, api.Mul(c.Catalog.Counts[j], norms[j]))
	}
	for a := range c.Indices {
		for _, v := range c.X[a] {
			rangeCheckSigned(api, rc, v, l.X, false)
		}
		boundOpen(api, m, boundX, c.Indices[a], c.SaltX, c.X[a], c.PX[a], c.CX)
		group := frontend.Variable(0)
		for i, g := range c.Catalog.Groups {
			group = api.Add(group, api.Mul(api.IsZero(api.Sub(c.Indices[a], i)), g))
		}
		var oldMin, newMin frontend.Variable
		assigned := frontend.Variable(0)
		for j := 0; j < m.K; j++ {
			od, nd := frontend.Variable(0), frontend.Variable(0)
			for d, x := range c.X[a] {
				u, v := api.Sub(x, m.Centers[j][d]), api.Sub(x, c.Centers[j][d])
				od = api.Add(od, api.Mul(u, u))
				nd = api.Add(nd, api.Mul(v, v))
			}
			rangeCheckUnsigned(api, rc, od, distanceMax, false)
			rangeCheckUnsigned(api, rc, nd, distanceMax, false)
			if j == 0 {
				oldMin, newMin = od, nd
			} else {
				oldMin = api.Select(dc.IsLess(od, oldMin), od, oldMin)
				newMin = api.Select(dc.IsLess(nd, newMin), nd, newMin)
			}
			assigned = api.Add(assigned, api.Mul(api.IsZero(api.Sub(group, j)), api.Sub(nd, od)))
		}
		total = api.Add(total, api.Mul(c.Active[a], api.Sub(newMin, oldMin, assigned)))
	}
	rangeCheckSigned(api, rc, c.Change, mul(bi(int64(m.N)), distanceMax), false)
	api.AssertIsEqual(c.Change, total)
	return nil
}

func GroupCorrectionIndices(p *BoundPrepared, g *GroupMoments, centers [][]int64) ([]int, *big.Int, error) {
	indices := make([]int, p.M.N)
	for i := range indices {
		indices[i] = i
	}
	oh, oe, err := ReuseCenterStats(p.M, p.M.Centers)
	if err != nil {
		return nil, nil, err
	}
	nh, ne, err := ReuseCenterStats(p.M, centers)
	if err != nil {
		return nil, nil, err
	}
	a, err := p.ReuseCertificate(oh, oe, indices)
	if err != nil {
		return nil, nil, err
	}
	b, err := p.ReuseCertificate(nh, ne, indices)
	if err != nil {
		return nil, nil, err
	}
	result := []int{}
	delta := bi(0)
	for i := range indices {
		if !(a.Flags[i] && b.Flags[i] && a.Labels[i] == g.Groups[i] && b.Labels[i] == g.Groups[i]) {
			result = append(result, i)
		}
		var before, after *big.Int
		for j := 0; j < p.M.K; j++ {
			u, v := bi(0), bi(0)
			for d, x := range p.X[i] {
				u.Add(u, square(bi(x-p.M.Centers[j][d])))
				v.Add(v, square(bi(x-centers[j][d])))
			}
			if before == nil || u.Cmp(before) < 0 {
				before = u
			}
			if after == nil || v.Cmp(after) < 0 {
				after = v
			}
		}
		delta.Add(delta, new(big.Int).Sub(after, before))
	}
	return result, delta, nil
}
func FillBoundGroupQuery(p *BoundPrepared, g *GroupMoments, centers [][]int64, capacity int) (*BoundGroupQuery, error) {
	if err := validateNewCenters(p.M, centers); err != nil {
		return nil, err
	}
	ids, delta, err := GroupCorrectionIndices(p, g, centers)
	if err != nil {
		return nil, err
	}
	if len(ids) > capacity {
		return nil, fmt.Errorf("correction budget exceeded")
	}
	c := NewBoundGroupQuery(p.M, capacity)
	c.CX, c.CH, c.CE, c.CT, c.Context, c.GroupRoot, c.Change = p.CX, p.CH, p.CE, p.CT, p.Context, g.Commitment, delta
	c.SaltX, c.SaltZ = p.SaltX, p.SaltZ
	fillGroupCatalog(&c.Catalog, g)
	heads, energy, err := ReuseCenterStats(p.M, centers)
	if err != nil {
		return nil, err
	}
	for j, row := range centers {
		for d, v := range row {
			c.Centers[j][d] = v
		}
		for d, v := range heads[j] {
			c.CHead[j][d] = v
		}
		c.COff[j] = energy[j]
		c.CenterRadius[j] = ceilSqrt(energy[j])
	}
	for i := range c.H {
		for d, v := range p.H[i] {
			c.H[i][d] = v
		}
		for d, v := range p.TH.Path(i) {
			c.PH[i][d] = v
		}
		c.E[i] = p.E[i]
		c.RowRadius[i] = ceilSqrt(p.E[i])
	}
	for a := 0; a < capacity; a++ {
		idx := 0
		c.Active[a] = 0
		if a < len(ids) {
			idx = ids[a]
			c.Active[a] = 1
		}
		c.Indices[a] = idx
		for d, v := range p.X[idx] {
			c.X[a][d] = v
		}
		for d, v := range p.TX.Path(idx) {
			c.PX[a][d] = v
		}
	}
	return c, nil
}

// Public-only builders derive center statistics from independently approved
// full centers. Reading a prover's serialized public witness is not enough.
func GroupPreprocessPublic(m BoundModel, s BoundStatement, root string) (*BoundGroupPreprocess, error) {
	cx, ch, ce, ct, ctx, err := s.Values(m)
	if err != nil {
		return nil, err
	}
	gr, err := canonicalPublicInt(root)
	if err != nil {
		return nil, err
	}
	c := NewBoundGroupPreprocess(m)
	c.CX, c.CH, c.CE, c.CT, c.Context, c.GroupRoot = cx, ch, ce, ct, ctx, gr
	return c, nil
}
func GroupQueryPublic(m BoundModel, s BoundStatement, root string, centers [][]int64, ids []int, active []bool, change string) (*BoundGroupQuery, error) {
	base, err := GroupPreprocessPublic(m, s, root)
	if err != nil {
		return nil, err
	}
	if len(ids) != len(active) || len(ids) > m.N {
		return nil, fmt.Errorf("invalid correction shape")
	}
	if err := validateNewCenters(m, centers); err != nil {
		return nil, err
	}
	if len(centers) != m.K {
		return nil, fmt.Errorf("center count differs")
	}
	delta, ok := new(big.Int).SetString(change, 10)
	if !ok || delta.String() != change {
		return nil, fmt.Errorf("invalid signed delta")
	}
	c := NewBoundGroupQuery(m, len(ids))
	c.CX, c.CH, c.CE, c.CT, c.Context, c.GroupRoot, c.Change = base.CX, base.CH, base.CE, base.CT, base.Context, base.GroupRoot, delta
	h, e, err := ReuseCenterStats(m, centers)
	if err != nil {
		return nil, err
	}
	for j, row := range centers {
		for d, v := range row {
			c.Centers[j][d] = v
		}
		for d, v := range h[j] {
			c.CHead[j][d] = v
		}
		c.COff[j] = e[j]
		c.CenterRadius[j] = ceilSqrt(e[j])
	}
	previous := -1
	ended := false
	for i, idx := range ids {
		if idx < 0 || idx >= m.N {
			return nil, fmt.Errorf("invalid correction index")
		}
		c.Indices[i] = idx
		c.Active[i] = 0
		if active[i] {
			if ended || idx <= previous {
				return nil, fmt.Errorf("invalid correction ordering")
			}
			previous = idx
			c.Active[i] = 1
		} else {
			ended = true
			if idx != 0 {
				return nil, fmt.Errorf("invalid padding")
			}
		}
	}
	return c, nil
}
