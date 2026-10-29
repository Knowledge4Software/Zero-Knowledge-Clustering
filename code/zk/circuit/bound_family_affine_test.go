package circuit

import (
	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark/test"
	"math/big"
	"testing"
)

func TestFamilyAffineExactSubsetAndUnknownCenters(t *testing.T) {
	m := BoundModel{Version: BoundVersion, Method: "affine", N: 4, D: 3, K: 4, InputBits: 4, BasisScale: 8, InputScale: "1", Centers: [][]int64{{0, 0, 0}, {8, 0, 0}, {0, 8, 0}, {8, 8, 0}}}
	rows := [][]int64{{0, 0, 1}, {4, 0, 1}, {7, 7, 2}, {1, 7, 2}}
	p, err := PrepareBound(m, rows, bi(11), bi(13))
	if err != nil {
		t.Fatal(err)
	}
	for _, centers := range [][][]int64{{m.Centers[0], m.Centers[1]}, {m.Centers[3], m.Centers[2], m.Centers[3]}, {m.Centers[1], m.Centers[0]}} {
		w, err := FillBoundFamilyAffineQuery(p, centers, []int{0, 1, 2, 3})
		if err != nil {
			t.Fatal(err)
		}
		truth := exactPairedCost(rows, centers)
		if w.Psi.(*big.Int).Cmp(truth) != 0 {
			t.Fatal("family cost differs from Full")
		}
		if err = test.IsSolved(NewBoundFamilyAffineQuery(m, 4, len(centers)), w, ecc.BN254.ScalarField()); err != nil {
			t.Fatal(err)
		}
		w.Psi = add(truth, bi(1))
		if test.IsSolved(NewBoundFamilyAffineQuery(m, 4, len(centers)), w, ecc.BN254.ScalarField()) == nil {
			t.Fatal("false exact cost accepted")
		}
	}
	if _, err = FillBoundFamilyAffineQuery(p, [][]int64{{1, 1, 0}, m.Centers[0]}, []int{0}); err == nil {
		t.Fatal("unknown future center accepted")
	}
	w, err := FillBoundFamilyAffineQuery(p, [][]int64{m.Centers[0], m.Centers[1]}, []int{0, 1})
	if err != nil {
		t.Fatal(err)
	}
	w.Positions[0] = m.K
	if test.IsSolved(NewBoundFamilyAffineQuery(m, 2, 2), w, ecc.BN254.ScalarField()) == nil {
		t.Fatal("out-of-bank selector accepted")
	}
}
