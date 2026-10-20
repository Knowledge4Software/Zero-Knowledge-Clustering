package circuit

import (
	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark/test"
	"math/big"
	"testing"
)

func TestFullReuseExactAcrossPublicCenterUpdates(t *testing.T) {
	m := BoundModel{Version: BoundVersion, Method: "full", N: 4, D: 3, K: 2, InputBits: 4, BasisScale: 8, InputScale: "1", Centers: [][]int64{{0, 0, 0}, {2, 0, 0}}}
	x := [][]int64{{0, 1, 0}, {3, 0, 0}, {-2, 2, 1}, {4, 2, 3}}
	p, err := PrepareBound(m, x, bi(11), bi(22))
	if err != nil {
		t.Fatal(err)
	}
	sets := [][][]int64{{{0, 0, 0}, {2, 0, 0}}, {{-1, 2, 0}, {4, 1, 3}}, {{16, -16, 16}, {-16, 16, -16}}}
	indices := []int{0, 1, 2, 3}
	shape := NewBoundFullReuseQuery(m, 4, 2)
	for _, centers := range sets {
		w, err := FillBoundFullReuseQuery(p, centers, indices)
		if err != nil {
			t.Fatal(err)
		}
		if err = test.IsSolved(shape, w, ecc.BN254.ScalarField()); err != nil {
			t.Fatal(err)
		}
		fm := m
		fm.Centers = centers
		fp, err := PrepareBound(fm, x, bi(11), bi(22))
		if err != nil {
			t.Fatal(err)
		}
		labels, cost, err := fp.Answer(indices)
		if err != nil {
			t.Fatal(err)
		}
		if cost.Cmp(w.Psi.(*big.Int)) != 0 {
			t.Fatal("dynamic Full differs from exact reference")
		}
		for i, label := range labels {
			if label != w.Assign[i].(int) {
				t.Fatal("different exact assignment")
			}
		}
	}
	for _, alter := range []string{"cost", "assignment", "center", "root", "index"} {
		w, _ := FillBoundFullReuseQuery(p, sets[1], indices)
		switch alter {
		case "cost":
			w.Psi = add(w.Psi.(*big.Int), bi(1))
		case "assignment":
			w.Assign[0] = 1 - w.Assign[0].(int)
		case "center":
			w.Centers[0][0] = 17
		case "root":
			w.CX = bi(1)
		case "index":
			w.Indices[1] = 0
		}
		if test.IsSolved(shape, w, ecc.BN254.ScalarField()) == nil {
			t.Fatalf("accepted altered %s", alter)
		}
	}
	if _, err = BoundFullReusePublic(m, p.Statement(), sets[0], []int{0, 0}, "0"); err == nil {
		t.Fatal("accepted duplicate public indices")
	}
}
