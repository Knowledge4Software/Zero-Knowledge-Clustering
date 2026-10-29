package circuit

// Experimental preprocessing bridge for private group moments. The auditor
// must choose an unpredictable nonzero field challenge AFTER the input,
// membership and moment commitments have been fixed, then pin that challenge
// independently. A prover-selected challenge is not an authenticated protocol.
// A false fixed moment vector passes its degree-D polynomial test with
// probability at most D/(p-1), per group, in addition to the underlying proof
// and commitment assumptions. This file is not a complete audit service.
import (
	"crypto/rand"
	"fmt"
	"math/big"

	"github.com/consensys/gnark-crypto/ecc/bn254/fr"
	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/std/math/cmp"
	"github.com/consensys/gnark/std/rangecheck"
)

const (
	boundFingerprintGroup = 130
	boundFingerprintStats = 131
	boundFingerprintMask  = 132
)

type GroupFingerprintCommitment struct {
	Membership      *BoundTree
	StatsRoot, Salt *big.Int
}

func fingerprintStatsRoot(m BoundModel, p *BoundPrepared, g *GroupMoments, membership, salt *big.Int) *big.Int {
	values := []*big.Int{bi(boundFingerprintStats), m.Hash(), p.CX, p.CH, membership, salt}
	for j := range g.Counts {
		values = append(values, g.Counts[j])
		values = append(values, g.Sums[j]...)
	}
	return boundNativeHash(values...)
}

func PrepareGroupFingerprintCommitment(p *BoundPrepared, g *GroupMoments, salt *big.Int) (*GroupFingerprintCommitment, error) {
	if err := p.M.Validate(); err != nil {
		return nil, err
	}
	if !p.M.HasEnergy() || p.M.TailWidth != 0 || len(g.Groups) != p.M.N || len(g.Counts) != p.M.K {
		return nil, fmt.Errorf("invalid fingerprint group catalog")
	}
	leaves := make([]*big.Int, p.M.N)
	for i, group := range g.Groups {
		if group < 0 || group >= p.M.K {
			return nil, fmt.Errorf("invalid group label")
		}
		leaves[i] = boundLeaf(p.M, boundFingerprintGroup, i, salt, []*big.Int{bi(int64(group))})
	}
	c := &GroupFingerprintCommitment{Membership: NewBoundTree(leaves), Salt: new(big.Int).Set(salt)}
	c.StatsRoot = fingerprintStatsRoot(p.M, p, g, c.Membership.Root(), salt)
	return c, nil
}

type GroupFingerprintWitness struct {
	Commitment *GroupFingerprintCommitment
	Challenge  *big.Int
	Masks      [][]*big.Int
	MaskHash   []*big.Int
	Masked     [][]*big.Int
}

func fingerprintMaskHash(m BoundModel, cx, ch, cg, cs, challenge *big.Int, index int, masks []*big.Int) *big.Int {
	values := []*big.Int{bi(boundFingerprintMask), m.Hash(), cx, ch, cg, cs, challenge, bi(int64(index))}
	return boundNativeHash(append(values, masks...)...)
}

// challenge must be supplied by the independently pinned auditor stage, after
// c's commitments. Fresh private masks hide the public row contributions.
func FillGroupFingerprintWitness(p *BoundPrepared, g *GroupMoments, c *GroupFingerprintCommitment, challenge *big.Int) (*GroupFingerprintWitness, error) {
	if challenge == nil || challenge.Sign() <= 0 || challenge.Cmp(fr.Modulus()) >= 0 {
		return nil, fmt.Errorf("invalid nonzero auditor challenge")
	}
	if c == nil || c.StatsRoot.Cmp(fingerprintStatsRoot(p.M, p, g, c.Membership.Root(), c.Salt)) != 0 {
		return nil, fmt.Errorf("fingerprint commitment mismatch")
	}
	f := &GroupFingerprintWitness{Commitment: c, Challenge: new(big.Int).Set(challenge), Masks: make([][]*big.Int, p.M.N+1), MaskHash: make([]*big.Int, p.M.N+1), Masked: make([][]*big.Int, p.M.N)}
	for i := range f.Masks {
		f.Masks[i] = make([]*big.Int, p.M.K)
		for j := range f.Masks[i] {
			v, err := rand.Int(rand.Reader, fr.Modulus())
			if err != nil {
				return nil, err
			}
			f.Masks[i][j] = v
		}
		f.MaskHash[i] = fingerprintMaskHash(p.M, p.CX, p.CH, c.Membership.Root(), c.StatsRoot, challenge, i, f.Masks[i])
	}
	for i, row := range p.X {
		poly := bi(0)
		for d := len(row) - 1; d >= 0; d-- {
			poly.Add(mul(poly, challenge), bi(row[d]))
			poly.Mod(poly, fr.Modulus())
		}
		poly.Add(mul(poly, challenge), bi(1))
		poly.Mod(poly, fr.Modulus())
		f.Masked[i] = make([]*big.Int, p.M.K)
		for j := range f.Masked[i] {
			v := new(big.Int).Sub(f.Masks[i+1][j], f.Masks[i][j])
			if g.Groups[i] == j {
				v.Add(v, poly)
			}
			f.Masked[i][j] = v.Mod(v, fr.Modulus())
		}
	}
	return f, nil
}

func fingerprintMaskCircuit(api frontend.API, m BoundModel, cx, ch, cg, cs, challenge, index frontend.Variable, masks []frontend.Variable) frontend.Variable {
	v := []frontend.Variable{boundFingerprintMask, m.Hash(), cx, ch, cg, cs, challenge, index}
	return boundHash(api, append(v, masks...)...)
}

type BoundGroupFingerprintRow struct {
	ExactOld                                                        bool `gnark:"-"`
	CompactGroup                                                    bool `gnark:"-"`
	Original                                                        BoundPreprocess
	GroupRoot, StatsRoot, Challenge, PreviousMaskHash, NextMaskHash frontend.Variable   `gnark:",public"`
	Masked                                                          []frontend.Variable `gnark:",public"`
	Group, GroupSalt                                                frontend.Variable
	GroupPath, PreviousMask, NextMask                               []frontend.Variable
}

func NewBoundGroupFingerprintRow(m BoundModel) *BoundGroupFingerprintRow {
	return &BoundGroupFingerprintRow{Original: *NewBoundPreprocess(m), Masked: make([]frontend.Variable, m.K), GroupPath: make([]frontend.Variable, m.Depth()), PreviousMask: make([]frontend.Variable, m.K), NextMask: make([]frontend.Variable, m.K)}
}

func NewBoundGroupFingerprintRowExact(m BoundModel) *BoundGroupFingerprintRow {
	c := NewBoundGroupFingerprintRow(m)
	c.ExactOld = true
	return c
}

func NewBoundGroupFingerprintRowCompact(m BoundModel) *BoundGroupFingerprintRow {
	c := NewBoundGroupFingerprintRowExact(m)
	c.CompactGroup = true
	c.GroupPath = nil
	return c
}

func (c *BoundGroupFingerprintRow) Define(api frontend.API) error {
	var err error
	if c.CompactGroup {
		err = c.Original.defineWithHeadSuffix(api, []frontend.Variable{c.Group})
	} else {
		err = c.Original.Define(api)
	}
	if err != nil {
		return err
	}
	m := c.Original.M
	if !m.HasEnergy() || m.TailWidth != 0 {
		return fmt.Errorf("fingerprint requires head/energy without tail")
	}
	api.AssertIsDifferent(c.Challenge, 0)
	rangeCheckUnsigned(api, rangecheck.New(api), c.Group, bi(int64(m.K-1)), false)
	if c.CompactGroup {
		api.AssertIsEqual(c.GroupRoot, c.Original.CH)
		api.AssertIsEqual(c.GroupSalt, c.Original.SaltZ)
	} else {
		boundOpen(api, m, boundFingerprintGroup, c.Original.Index, c.GroupSalt, []frontend.Variable{c.Group}, c.GroupPath, c.GroupRoot)
	}
	if c.ExactOld {
		// The common ||x||^2 term cancels. With static public old centers,
		// these scores are linear combinations of the authenticated original X.
		limit := mul(bi(int64(3*m.D)), square(m.Limits().X))
		if mul(limit, bi(8)).Cmp(api.Compiler().Field()) >= 0 {
			return fmt.Errorf("unsafe exact old score bound")
		}
		scores := make([]frontend.Variable, m.K)
		selected := frontend.Variable(0)
		for j, center := range m.Centers {
			score := frontend.Variable(0)
			for d, value := range center {
				score = api.Add(score, square(bi(value)), api.Mul(-2*value, c.Original.X[d]))
			}
			scores[j] = score
			selected = api.Add(selected, api.Mul(api.IsZero(api.Sub(c.Group, j)), score))
		}
		bc := cmp.NewBoundedComparator(api, mul(limit, bi(2)), false)
		for _, score := range scores {
			// Ties are valid nearest groups; deterministic labels are not needed
			// for the exact old cost or the zero switching advantage.
			api.AssertIsEqual(bc.IsLess(score, selected), 0)
		}
	}
	api.AssertIsEqual(c.PreviousMaskHash, fingerprintMaskCircuit(api, m, c.Original.CX, c.Original.CH, c.GroupRoot, c.StatsRoot, c.Challenge, c.Original.Index, c.PreviousMask))
	api.AssertIsEqual(c.NextMaskHash, fingerprintMaskCircuit(api, m, c.Original.CX, c.Original.CH, c.GroupRoot, c.StatsRoot, c.Challenge, api.Add(c.Original.Index, 1), c.NextMask))
	poly := frontend.Variable(0)
	for d := m.D - 1; d >= 0; d-- {
		poly = api.Add(api.Mul(poly, c.Challenge), c.Original.X[d])
	}
	poly = api.Add(1, api.Mul(poly, c.Challenge))
	for j := 0; j < m.K; j++ {
		member := api.IsZero(api.Sub(c.Group, j))
		api.AssertIsEqual(c.Masked[j], api.Add(api.Mul(member, poly), c.NextMask[j], api.Neg(c.PreviousMask[j])))
	}
	return nil
}

func FillBoundGroupFingerprintRow(p *BoundPrepared, g *GroupMoments, f *GroupFingerprintWitness, index int) *BoundGroupFingerprintRow {
	c := NewBoundGroupFingerprintRow(p.M)
	c.Original = *FillBoundPreprocess(p, index)
	c.GroupRoot, c.StatsRoot, c.Challenge = f.Commitment.Membership.Root(), f.Commitment.StatsRoot, f.Challenge
	c.PreviousMaskHash, c.NextMaskHash = f.MaskHash[index], f.MaskHash[index+1]
	c.Group, c.GroupSalt = g.Groups[index], f.Commitment.Salt
	for j, v := range f.Commitment.Membership.Path(index) {
		c.GroupPath[j] = v
	}
	for j := 0; j < p.M.K; j++ {
		c.Masked[j] = f.Masked[index][j]
		c.PreviousMask[j] = f.Masks[index][j]
		c.NextMask[j] = f.Masks[index+1][j]
	}
	return c
}

type GroupFingerprintPublicRow struct {
	Index          int
	Previous, Next *big.Int
	Masked         []*big.Int
}

// These roots and the post-commit auditor challenge must be pinned independently
// of prover metadata. Public builders never read original rows or group sums.
type GroupFingerprintHeader struct {
	GroupRoot, StatsRoot, Challenge *big.Int
}

func fingerprintCanonical(values ...*big.Int) error {
	for _, v := range values {
		if v == nil || v.Sign() < 0 || v.Cmp(fr.Modulus()) >= 0 {
			return fmt.Errorf("noncanonical fingerprint field")
		}
	}
	return nil
}

func GroupFingerprintRowPublic(m BoundModel, s BoundStatement, header GroupFingerprintHeader, row GroupFingerprintPublicRow) (*BoundGroupFingerprintRow, error) {
	if err := m.Validate(); err != nil {
		return nil, err
	}
	if err := fingerprintCanonical(header.GroupRoot, header.StatsRoot, header.Challenge, row.Previous, row.Next); err != nil {
		return nil, err
	}
	if header.Challenge.Sign() == 0 || len(row.Masked) != m.K {
		return nil, fmt.Errorf("invalid fingerprint public shape/challenge")
	}
	if err := fingerprintCanonical(row.Masked...); err != nil {
		return nil, err
	}
	original, err := BoundPreprocessPublic(m, s, row.Index)
	if err != nil {
		return nil, err
	}
	c := NewBoundGroupFingerprintRow(m)
	c.Original = *original
	c.GroupRoot, c.StatsRoot, c.Challenge = header.GroupRoot, header.StatsRoot, header.Challenge
	c.PreviousMaskHash, c.NextMaskHash = row.Previous, row.Next
	for j, v := range row.Masked {
		c.Masked[j] = v
	}
	return c, nil
}

// sums/first/last must be reconstructed from the complete, verified row chain.
func GroupFingerprintLinkPublic(m BoundModel, s BoundStatement, header GroupFingerprintHeader, sums []*big.Int, first, last *big.Int) (*BoundGroupFingerprintLink, error) {
	if err := m.Validate(); err != nil {
		return nil, err
	}
	if err := fingerprintCanonical(header.GroupRoot, header.StatsRoot, header.Challenge, first, last); err != nil {
		return nil, err
	}
	if header.Challenge.Sign() == 0 || len(sums) != m.K {
		return nil, fmt.Errorf("invalid fingerprint link shape/challenge")
	}
	if err := fingerprintCanonical(sums...); err != nil {
		return nil, err
	}
	cx, ch, _, _, _, err := s.Values(m)
	if err != nil {
		return nil, err
	}
	c := NewBoundGroupFingerprintLink(m)
	c.CX, c.CH = cx, ch
	c.GroupRoot, c.StatsRoot, c.Challenge = header.GroupRoot, header.StatsRoot, header.Challenge
	c.FirstMaskHash, c.LastMaskHash = first, last
	for j, v := range sums {
		c.MaskedSum[j] = v
	}
	return c, nil
}

// Coverage/mask-chain checks are in addition to verifying EACH row proof with
// the same independently approved model/roots/challenge. This function alone
// does not authenticate rows or commitments.
func FoldGroupFingerprintRows(m BoundModel, rows []GroupFingerprintPublicRow) ([]*big.Int, error) {
	if len(rows) != m.N {
		return nil, fmt.Errorf("incomplete fingerprint row coverage")
	}
	sums := make([]*big.Int, m.K)
	for j := range sums {
		sums[j] = bi(0)
	}
	for i, row := range rows {
		if row.Index != i || row.Previous == nil || row.Next == nil || len(row.Masked) != m.K {
			return nil, fmt.Errorf("invalid fingerprint row identity")
		}
		if row.Previous.Sign() < 0 || row.Next.Sign() < 0 || row.Previous.Cmp(fr.Modulus()) >= 0 || row.Next.Cmp(fr.Modulus()) >= 0 {
			return nil, fmt.Errorf("noncanonical mask commitment")
		}
		if i > 0 && rows[i-1].Next.Cmp(row.Previous) != 0 {
			return nil, fmt.Errorf("broken fingerprint mask chain")
		}
		for j, v := range row.Masked {
			if v == nil || v.Sign() < 0 || v.Cmp(fr.Modulus()) >= 0 {
				return nil, fmt.Errorf("noncanonical fingerprint contribution")
			}
			sums[j].Add(sums[j], v)
			sums[j].Mod(sums[j], fr.Modulus())
		}
	}
	return sums, nil
}

type BoundGroupFingerprintLink struct {
	M                                                                    BoundModel          `gnark:"-"`
	CX, CH, GroupRoot, StatsRoot, Challenge, FirstMaskHash, LastMaskHash frontend.Variable   `gnark:",public"`
	MaskedSum                                                            []frontend.Variable `gnark:",public"`
	Counts                                                               []frontend.Variable
	Sums                                                                 [][]frontend.Variable
	GroupSalt                                                            frontend.Variable
	FirstMask, LastMask                                                  []frontend.Variable
}

func NewBoundGroupFingerprintLink(m BoundModel) *BoundGroupFingerprintLink {
	c := &BoundGroupFingerprintLink{M: m, MaskedSum: make([]frontend.Variable, m.K), Counts: make([]frontend.Variable, m.K), Sums: make([][]frontend.Variable, m.K), FirstMask: make([]frontend.Variable, m.K), LastMask: make([]frontend.Variable, m.K)}
	for j := range c.Sums {
		c.Sums[j] = make([]frontend.Variable, m.D)
	}
	return c
}

func (c *BoundGroupFingerprintLink) Define(api frontend.API) error {
	m := c.M
	if err := m.Validate(); err != nil {
		return err
	}
	if !m.HasEnergy() || m.TailWidth != 0 {
		return fmt.Errorf("fingerprint requires head/energy without tail")
	}
	api.AssertIsDifferent(c.Challenge, 0)
	rc := rangecheck.New(api)
	// Limits scans the entire rounded basis. It is constant throughout this
	// definition; recomputing it for each of K*D sum coordinates was expensive.
	sumBound := mul(bi(int64(m.N)), m.Limits().X)
	values := []frontend.Variable{boundFingerprintStats, m.Hash(), c.CX, c.CH, c.GroupRoot, c.GroupSalt}
	total := frontend.Variable(0)
	for j := 0; j < m.K; j++ {
		rangeCheckUnsigned(api, rc, c.Counts[j], bi(int64(m.N)), false)
		total = api.Add(total, c.Counts[j])
		values = append(values, c.Counts[j])
		for _, v := range c.Sums[j] {
			rangeCheckSigned(api, rc, v, sumBound, false)
			values = append(values, v)
		}
		poly := frontend.Variable(0)
		for d := m.D - 1; d >= 0; d-- {
			poly = api.Add(api.Mul(poly, c.Challenge), c.Sums[j][d])
		}
		poly = api.Add(c.Counts[j], api.Mul(poly, c.Challenge))
		api.AssertIsEqual(c.MaskedSum[j], api.Add(poly, c.LastMask[j], api.Neg(c.FirstMask[j])))
	}
	api.AssertIsEqual(total, m.N)
	api.AssertIsEqual(c.StatsRoot, boundHash(api, values...))
	api.AssertIsEqual(c.FirstMaskHash, fingerprintMaskCircuit(api, m, c.CX, c.CH, c.GroupRoot, c.StatsRoot, c.Challenge, 0, c.FirstMask))
	api.AssertIsEqual(c.LastMaskHash, fingerprintMaskCircuit(api, m, c.CX, c.CH, c.GroupRoot, c.StatsRoot, c.Challenge, m.N, c.LastMask))
	return nil
}

func FillBoundGroupFingerprintLink(p *BoundPrepared, g *GroupMoments, f *GroupFingerprintWitness) (*BoundGroupFingerprintLink, error) {
	rows := make([]GroupFingerprintPublicRow, p.M.N)
	for i := range rows {
		rows[i] = GroupFingerprintPublicRow{i, f.MaskHash[i], f.MaskHash[i+1], f.Masked[i]}
	}
	sums, err := FoldGroupFingerprintRows(p.M, rows)
	if err != nil {
		return nil, err
	}
	c := NewBoundGroupFingerprintLink(p.M)
	c.CX, c.CH = p.CX, p.CH
	c.GroupRoot, c.StatsRoot, c.Challenge = f.Commitment.Membership.Root(), f.Commitment.StatsRoot, f.Challenge
	c.FirstMaskHash, c.LastMaskHash = f.MaskHash[0], f.MaskHash[p.M.N]
	c.GroupSalt = f.Commitment.Salt
	for j := 0; j < p.M.K; j++ {
		c.MaskedSum[j] = sums[j]
		c.Counts[j] = g.Counts[j]
		c.FirstMask[j], c.LastMask[j] = f.Masks[0][j], f.Masks[p.M.N][j]
		for d, v := range g.Sums[j] {
			c.Sums[j][d] = v
		}
	}
	return c, nil
}
