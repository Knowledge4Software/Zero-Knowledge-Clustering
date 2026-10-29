package circuit

import (
	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark/test"
	"math/big"
	"math/rand"
	"testing"
)

func TestGroupSwitchRoundedBasisContainsTrueChange(t *testing.T) {
	rng := rand.New(rand.NewSource(241024))
	for trial := 0; trial < 8; trial++ {
		m := BoundModel{Version: BoundVersion, Method: "learned_energy", N: 4, D: 4, K: 2, InputBits: 4, BasisScale: 8, InputScale: "1", Centers: [][]int64{{0, 0, 0, 0}, {8, 0, 0, 0}}, Basis: [][]int64{{7, 1, 0, 0}, {0, 7, 1, 0}}}
		rows := make([][]int64, 4)
		for i := range rows {
			rows[i] = make([]int64, 4)
			for d := range rows[i] {
				rows[i][d] = int64(rng.Intn(13) - 6)
			}
		}
		centers := [][]int64{{1, 0, 1, 0}, {7, 1, 0, 1}}
		p, err := PrepareBound(m, rows, bi(7), bi(9))
		if err != nil {
			t.Fatal(err)
		}
		g, err := PrepareGroupMoments(p, bi(11))
		if err != nil {
			t.Fatal(err)
		}
		commit, err := PrepareGroupFingerprintCommitment(p, g, bi(43))
		if err != nil {
			t.Fatal(err)
		}
		params, err := NewGroupSwitchParameters(m, centers)
		if err != nil {
			t.Fatal(err)
		}
		batch, err := FillBoundGroupSwitchBatch(p, g, commit, params, []int{0, 1, 2, 3})
		if err != nil {
			t.Fatal(err)
		}
		if err = test.IsSolved(NewBoundGroupSwitchBatch(m, 4), batch, ecc.BN254.ScalarField()); err != nil {
			t.Fatal(err)
		}
		moment, err := FillBoundGroupAssignedChange(p, g, commit, centers)
		if err != nil {
			t.Fatal(err)
		}
		if err = test.IsSolved(NewBoundGroupAssignedChange(m), moment, ecc.BN254.ScalarField()); err != nil {
			t.Fatal(err)
		}
		t6 := new(big.Int).Exp(bi(m.BasisScale), bi(6), nil)
		truth := mul(new(big.Int).Sub(exactPairedCost(rows, centers), exactPairedCost(rows, m.Centers)), t6)
		base := mul(moment.Change.(*big.Int), t6)
		lo, hi := add(base, batch.Lower.(*big.Int)), add(base, batch.Upper.(*big.Int))
		if lo.Cmp(truth) > 0 || hi.Cmp(truth) < 0 {
			t.Fatal("cost-change certificate excludes exact Full")
		}
		batch.Lower = add(batch.Lower.(*big.Int), bi(1))
		if test.IsSolved(NewBoundGroupSwitchBatch(m, 4), batch, ecc.BN254.ScalarField()) == nil {
			t.Fatal("false correction bound accepted")
		}
	}
}

func TestGroupSwitchWorksWithNonNearestFixedGroups(t *testing.T) {
	m := BoundModel{Version: BoundVersion, Method: "learned_energy", N: 1, D: 1, K: 2, InputBits: 4, BasisScale: 8, InputScale: "1", Centers: [][]int64{{0}, {10}}, Basis: [][]int64{{8}}}
	p, err := PrepareBound(m, [][]int64{{0}}, bi(7), bi(9))
	if err != nil {
		t.Fatal(err)
	}
	g := &GroupMoments{Groups: []int{1}, Counts: []*big.Int{bi(0), bi(1)}, Sums: [][]*big.Int{{bi(0)}, {bi(0)}}}
	commit, err := PrepareGroupFingerprintCommitment(p, g, bi(43))
	if err != nil {
		t.Fatal(err)
	}
	centers := [][]int64{{9}, {1}}
	params, err := NewGroupSwitchParameters(m, centers)
	if err != nil {
		t.Fatal(err)
	}
	moment, err := FillBoundGroupAssignedChange(p, g, commit, centers)
	if err != nil {
		t.Fatal(err)
	}
	batch, err := FillBoundGroupSwitchBatch(p, g, commit, params, []int{0})
	if err != nil {
		t.Fatal(err)
	}
	if err = test.IsSolved(NewBoundGroupSwitchBatch(m, 1), batch, ecc.BN254.ScalarField()); err != nil {
		t.Fatal(err)
	}
	if moment.Change.(*big.Int).Cmp(bi(-99)) != 0 {
		t.Fatal("counterexample assigned delta changed")
	}
	t6 := new(big.Int).Exp(bi(8), bi(6), nil)
	base := mul(moment.Change.(*big.Int), t6)
	if add(base, batch.Lower.(*big.Int)).Cmp(t6) != 0 || add(base, batch.Upper.(*big.Int)).Cmp(t6) != 0 {
		t.Fatal("incorrect sign when fixed group differs from old nearest center")
	}
}
