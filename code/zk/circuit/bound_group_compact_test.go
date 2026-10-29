package circuit

import (
	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark/test"
	"math/big"
	"testing"
)

func compactFixture(t *testing.T) (*BoundPrepared, *GroupMoments, *GroupFingerprintWitness, [][]int64) {
	p, _, centers := groupFixture(t)
	g, err := PrepareExactOldGroupMoments(p, p.SaltZ)
	if err != nil {
		t.Fatal(err)
	}
	commit, err := PrepareCompactGroup(p, g, p.SaltZ)
	if err != nil {
		t.Fatal(err)
	}
	f, err := FillGroupFingerprintWitness(p, g, commit, bi(37))
	if err != nil {
		t.Fatal(err)
	}
	return p, g, f, centers
}

func TestCompactFingerprintAndLinkedChange(t *testing.T) {
	p, g, f, centers := compactFixture(t)
	for i := 0; i < p.M.N; i++ {
		if err := test.IsSolved(NewBoundGroupFingerprintRowCompact(p.M), FillBoundGroupFingerprintRowCompact(p, g, f, i), ecc.BN254.ScalarField()); err != nil {
			t.Fatal(err)
		}
	}
	w, err := FillBoundGroupLinkedChange(p, g, f, centers)
	if err != nil {
		t.Fatal(err)
	}
	if err := test.IsSolved(NewBoundGroupLinkedChange(p.M), w, ecc.BN254.ScalarField()); err != nil {
		t.Fatal(err)
	}
	w.Change = add(w.Change.(*big.Int), bi(1))
	if test.IsSolved(NewBoundGroupLinkedChange(p.M), w, ecc.BN254.ScalarField()) == nil {
		t.Fatal("false linked change accepted")
	}
	w, err = FillBoundGroupLinkedChange(p, g, f, centers)
	if err != nil {
		t.Fatal(err)
	}
	w.Link.MaskedSum[0] = add(w.Link.MaskedSum[0].(*big.Int), bi(1))
	if test.IsSolved(NewBoundGroupLinkedChange(p.M), w, ecc.BN254.ScalarField()) == nil {
		t.Fatal("false moment fingerprint accepted")
	}
}

func TestCompactSwitchContainsFullAndBindsTiedGroup(t *testing.T) {
	p, g, f, centers := compactFixture(t)
	params, err := NewGroupSwitchParameters(p.M, centers)
	if err != nil {
		t.Fatal(err)
	}
	w, err := FillBoundGroupSwitchBatchCompact(p, g, f.Commitment, params, []int{0, 1, 2, 3})
	if err != nil {
		t.Fatal(err)
	}
	if err = test.IsSolved(NewBoundGroupSwitchBatchCompact(p.M, 4), w, ecc.BN254.ScalarField()); err != nil {
		t.Fatal(err)
	}
	moment, err := FillBoundGroupAssignedChange(p, g, f.Commitment, centers)
	if err != nil {
		t.Fatal(err)
	}
	t6 := new(big.Int).Exp(bi(p.M.BasisScale), bi(6), nil)
	base := mul(moment.Change.(*big.Int), t6)
	truth := mul(new(big.Int).Sub(exactPairedCost(p.X, centers), exactPairedCost(p.X, p.M.Centers)), t6)
	if add(base, w.Lower.(*big.Int)).Cmp(truth) > 0 || add(base, w.Upper.(*big.Int)).Cmp(truth) < 0 {
		t.Fatal("compact interval excludes Full")
	}
	// Row1 is tied between the old centers. An alternative label is a valid
	// argmin, but must still fail because the committed H/E/group leaf pins it.
	g.Groups[1] = 1 - g.Groups[1]
	w, err = FillBoundGroupSwitchBatchCompact(p, g, f.Commitment, params, []int{1})
	if err != nil {
		t.Fatal(err)
	}
	if test.IsSolved(NewBoundGroupSwitchBatchCompact(p.M, 1), w, ecc.BN254.ScalarField()) == nil {
		t.Fatal("changed tied membership accepted")
	}
}

// This is a disclosure test, not a claim that zero knowledge hides public A.
// If both coordinate probes are authorized, their exact aggregates recover
// the private-witness group count and coordinate sum algebraically.
func TestPublicAssignedChangeCoordinateProbeLeakage(t *testing.T) {
	p, g, f, _ := compactFixture(t)
	probe := func(delta int64) *big.Int {
		centers := make([][]int64, p.M.K)
		for j, row := range p.M.Centers {
			centers[j] = append([]int64{}, row...)
		}
		centers[0][0] += delta
		w, err := FillBoundGroupLinkedChange(p, g, f, centers)
		if err != nil {
			t.Fatal(err)
		}
		return w.Change.(*big.Int)
	}
	plus, minus := probe(1), probe(-1)
	n := new(big.Int).Quo(new(big.Int).Add(plus, minus), bi(2))
	coordinateSum := new(big.Int).Sub(mul(n, bi(p.M.Centers[0][0])), new(big.Int).Quo(new(big.Int).Sub(plus, minus), bi(4)))
	if n.Cmp(g.Counts[0]) != 0 || coordinateSum.Cmp(g.Sums[0][0]) != 0 {
		t.Fatal("public aggregate leakage calculation disagrees with protocol output")
	}
}
