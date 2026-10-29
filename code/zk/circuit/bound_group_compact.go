package circuit

// Compact profile: authenticate group membership in H/E leaves, and combine
// the fingerprint moment-link check with each assigned-change proof. It uses
// separate versioned receipts/keys and never reinterprets legacy commitments.
import (
	"fmt"
	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/std/rangecheck"
	"math/big"
)

func PrepareCompactGroup(p *BoundPrepared, g *GroupMoments, salt *big.Int) (*GroupFingerprintCommitment, error) {
	if len(g.Groups) != p.M.N {
		return nil, fmt.Errorf("compact groups do not cover catalog")
	}
	leaves := make([]*big.Int, p.M.N)
	for i := range leaves {
		values := append(append([]*big.Int{}, p.H[i]...), p.E[i], bi(int64(g.Groups[i])))
		leaves[i] = boundLeaf(p.M, boundH, i, p.SaltZ, values)
	}
	p.TH = NewBoundTree(leaves)
	p.CH = p.TH.Root()
	p.Context = boundNativeHash(bi(boundContext), p.M.Hash(), p.CX, p.CH, p.CE, p.CT)
	c := &GroupFingerprintCommitment{Membership: p.TH, Salt: new(big.Int).Set(salt)}
	c.StatsRoot = fingerprintStatsRoot(p.M, p, g, c.Membership.Root(), salt)
	return c, nil
}

func FillBoundGroupFingerprintRowCompact(p *BoundPrepared, g *GroupMoments, f *GroupFingerprintWitness, i int) *BoundGroupFingerprintRow {
	c := FillBoundGroupFingerprintRow(p, g, f, i)
	c.ExactOld, c.CompactGroup, c.GroupPath = true, true, nil
	return c
}

func FillBoundGroupSwitchBatchCompact(p *BoundPrepared, g *GroupMoments, commit *GroupFingerprintCommitment, params *GroupSwitchParameters, indices []int) (*BoundGroupSwitchBatch, error) {
	c, err := FillBoundGroupSwitchBatchMode(p, g, commit, params, indices, true)
	if err != nil {
		return nil, err
	}
	c.CompactGroup, c.GroupPath = true, nil
	return c, nil
}

type BoundGroupLinkedChange struct {
	Link    BoundGroupFingerprintLink
	Centers [][]frontend.Variable `gnark:",public"`
	Change  frontend.Variable     `gnark:",public"`
}

func NewBoundGroupLinkedChange(m BoundModel) *BoundGroupLinkedChange {
	c := &BoundGroupLinkedChange{Link: *NewBoundGroupFingerprintLink(m), Centers: make([][]frontend.Variable, m.K)}
	for j := range c.Centers {
		c.Centers[j] = make([]frontend.Variable, m.D)
	}
	return c
}

func (c *BoundGroupLinkedChange) Define(api frontend.API) error {
	if err := c.Link.Define(api); err != nil {
		return err
	}
	m := c.Link.M
	api.AssertIsEqual(c.Link.GroupRoot, c.Link.CH)
	rc := rangecheck.New(api)
	inputBound := m.Limits().X
	change := frontend.Variable(0)
	for j := 0; j < m.K; j++ {
		norm := frontend.Variable(0)
		for d, value := range c.Centers[j] {
			rangeCheckSigned(api, rc, value, inputBound, false)
			norm = api.Add(norm, api.Sub(api.Mul(value, value), square(bi(m.Centers[j][d]))))
			change = api.Add(change, api.Mul(2, c.Link.Sums[j][d], api.Sub(m.Centers[j][d], value)))
		}
		change = api.Add(change, api.Mul(c.Link.Counts[j], norm))
	}
	maximum := mul(bi(int64(4*m.N*m.D)), square(inputBound))
	rangeCheckSigned(api, rc, c.Change, maximum, false)
	api.AssertIsEqual(c.Change, change)
	return nil
}

func FillBoundGroupLinkedChange(p *BoundPrepared, g *GroupMoments, f *GroupFingerprintWitness, centers [][]int64) (*BoundGroupLinkedChange, error) {
	link, err := FillBoundGroupFingerprintLink(p, g, f)
	if err != nil {
		return nil, err
	}
	moment, err := FillBoundGroupAssignedChange(p, g, f.Commitment, centers)
	if err != nil {
		return nil, err
	}
	c := NewBoundGroupLinkedChange(p.M)
	c.Link = *link
	c.Centers = moment.Centers
	c.Change = moment.Change
	return c, nil
}

func GroupLinkedChangePublic(m BoundModel, statement BoundStatement, header GroupFingerprintHeader, sums []*big.Int, first, last *big.Int, centers [][]int64, change *big.Int) (*BoundGroupLinkedChange, error) {
	link, err := GroupFingerprintLinkPublic(m, statement, header, sums, first, last)
	if err != nil {
		return nil, err
	}
	moment, err := GroupAssignedChangePublic(m, statement, header, centers, change)
	if err != nil {
		return nil, err
	}
	if header.GroupRoot.String() != statement.CH {
		return nil, fmt.Errorf("compact group root must equal head root")
	}
	c := NewBoundGroupLinkedChange(m)
	c.Link = *link
	c.Centers = moment.Centers
	c.Change = change
	return c, nil
}
