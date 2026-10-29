package circuit

import (
	"math/big"
	"math/rand"
	"testing"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark/test"
)

func TestExactOldFingerprintRejectsAuthenticatedWrongGroupAndAllowsTies(t *testing.T) {
	m := BoundModel{Version: BoundVersion, Method: "learned_energy", N: 4, D: 2, K: 2, InputBits: 4, BasisScale: 8, InputScale: "1", Centers: [][]int64{{0, 0}, {8, 0}}, Basis: [][]int64{{8, 0}}}
	p, err := PrepareBound(m, [][]int64{{0, 1}, {7, 1}, {4, 0}, {-1, -1}}, bi(7), bi(9))
	if err != nil {
		t.Fatal(err)
	}
	for _, labels := range [][]int{{0, 1, 0, 0}, {0, 1, 1, 0}, {1, 1, 0, 0}} {
		g, err := PrepareGroupMomentsForLabels(p, bi(11), labels)
		if err != nil {
			t.Fatal(err)
		}
		commit, err := PrepareGroupFingerprintCommitment(p, g, bi(43))
		if err != nil {
			t.Fatal(err)
		}
		f, err := FillGroupFingerprintWitness(p, g, commit, bi(37))
		if err != nil {
			t.Fatal(err)
		}
		for i := 0; i < m.N; i++ {
			w := FillBoundGroupFingerprintRow(p, g, f, i)
			w.ExactOld = true
			err := test.IsSolved(NewBoundGroupFingerprintRowExact(m), w, ecc.BN254.ScalarField())
			wrong := i == 0 && labels[0] == 1
			if (err != nil) != wrong {
				t.Fatalf("row %d labels %v: wrong=%v err=%v", i, labels, wrong, err)
			}
		}
	}
}

func TestExactOldGroupSwitchContainsFullWithRoundedBasis(t *testing.T) {
	rng := rand.New(rand.NewSource(241025))
	for trial := 0; trial < 8; trial++ {
		m := BoundModel{Version: BoundVersion, Method: "learned_energy", N: 4, D: 4, K: 2, InputBits: 4, BasisScale: 8, InputScale: "1", Centers: [][]int64{{0, 0, 0, 0}, {8, 0, 0, 0}}, Basis: [][]int64{{7, 1, 0, 0}, {0, 7, 1, 0}}}
		rows := make([][]int64, m.N)
		for i := range rows {
			rows[i] = make([]int64, m.D)
			for d := range rows[i] {
				rows[i][d] = int64(rng.Intn(13) - 6)
			}
		}
		centers := [][]int64{{1, 0, 1, 0}, {7, 1, 0, 1}}
		p, err := PrepareBound(m, rows, bi(7), bi(9))
		if err != nil {
			t.Fatal(err)
		}
		g, err := PrepareExactOldGroupMoments(p, bi(11))
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
		batch, err := FillBoundGroupSwitchBatchMode(p, g, commit, params, []int{0, 1, 2, 3}, true)
		if err != nil {
			t.Fatal(err)
		}
		if err = test.IsSolved(NewBoundGroupSwitchBatchExact(m, 4), batch, ecc.BN254.ScalarField()); err != nil {
			t.Fatal(err)
		}
		moment, err := FillBoundGroupAssignedChange(p, g, commit, centers)
		if err != nil {
			t.Fatal(err)
		}
		t6 := new(big.Int).Exp(bi(m.BasisScale), bi(6), nil)
		truth := mul(new(big.Int).Sub(exactPairedCost(rows, centers), exactPairedCost(rows, m.Centers)), t6)
		base := mul(moment.Change.(*big.Int), t6)
		lo, hi := add(base, batch.Lower.(*big.Int)), add(base, batch.Upper.(*big.Int))
		if lo.Cmp(truth) > 0 || hi.Cmp(truth) < 0 || hi.Cmp(base) > 0 {
			t.Fatal("exact-old interval excludes Full or exceeds A")
		}
		batch.Lower = add(batch.Lower.(*big.Int), bi(1))
		if test.IsSolved(NewBoundGroupSwitchBatchExact(m, 4), batch, ecc.BN254.ScalarField()) == nil {
			t.Fatal("forged correction accepted")
		}
	}
}
