package circuit

import (
	"math/big"
	"testing"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark/test"
)

func reuseFixture(t *testing.T) *BoundPrepared {
	m := BoundModel{Version: BoundVersion, Method: "learned_energy", N: 4, D: 4, K: 2, InputBits: 4, BasisScale: 8,
		InputScale: "1", Centers: [][]int64{{0, 0, 0, 0}, {2, 0, 0, 0}}, Basis: [][]int64{{8, 0, 0, 0}, {0, 8, 0, 0}}}
	p, err := PrepareBound(m, [][]int64{{1, 1, 0, 0}, {3, 2, 0, 1}, {-2, 1, 1, 0}, {2, -1, 0, 2}}, bi(123), bi(456))
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < m.N; i++ { // learned basis rank 2 > K-1 still binds through Stage 1
		if err := test.IsSolved(NewBoundPreprocess(m), FillBoundPreprocess(p, i), ecc.BN254.ScalarField()); err != nil {
			t.Fatal(err)
		}
	}
	return p
}

func TestBoundReuseInSpanCentresMatchFull(t *testing.T) {
	p := reuseFixture(t)
	newC := [][]int64{{1, 0, 0, 0}, {0, 2, 0, 0}, {-1, -1, 0, 0}} // inside the deployed basis span
	heads, offs, err := ReuseCenterStats(p.M, newC)
	if err != nil {
		t.Fatal(err)
	}
	full := BoundModel{Version: BoundVersion, Method: "full", N: 4, D: 4, K: 3, InputBits: 4, BasisScale: 8, InputScale: "1",
		Centers: nil}
	full.Centers = newC
	fp, err := PrepareBound(full, p.X, bi(1), bi(2))
	if err != nil {
		t.Fatal(err)
	}
	indices := []int{0, 1, 2, 3}
	labels, psi, _, err := p.ReuseAnswer(heads, offs, indices)
	if err != nil {
		t.Fatal(err)
	}
	want, phi, _ := fp.Answer(indices)
	if psi.Cmp(mul(phi, bi(4096))) != 0 { // T^4 = 8^4 units
		t.Fatalf("in-span reuse cost %v differs from Full %v", psi, phi)
	}
	for i := range labels {
		if labels[i] != want[i] {
			t.Fatal("in-span reuse assignment differs from Full")
		}
	}
}

func TestBoundReuseCircuitAcceptsHonestAndRejectsAltered(t *testing.T) {
	p := reuseFixture(t)
	newC := [][]int64{{1, 0, 0, 0}, {0, 2, 0, 0}, {0, 0, 3, 1}} // last centre leaves the span
	heads, offs, err := ReuseCenterStats(p.M, newC)
	if err != nil {
		t.Fatal(err)
	}
	if offs[2].Sign() == 0 {
		t.Fatal("expected a residual offset for the out-of-span centre")
	}
	indices := []int{0, 2, 3}
	shape := func() *BoundReuseQuery { return NewBoundReuseQuery(p.M, 3, 3) }
	w, err := FillBoundReuseQuery(p, heads, offs, indices)
	if err != nil {
		t.Fatal(err)
	}
	if err = test.IsSolved(shape(), w, ecc.BN254.ScalarField()); err != nil {
		t.Fatal(err)
	}
	for _, alter := range []string{"psi", "margin", "assign", "head", "offset"} {
		w, _ := FillBoundReuseQuery(p, heads, offs, indices)
		switch alter {
		case "psi":
			w.Psi = add(w.Psi.(*big.Int), bi(1))
		case "margin":
			w.MarginSum = add(w.MarginSum.(*big.Int), bi(1))
		case "assign":
			w.Assign[0] = (w.Assign[0].(int) + 1) % 3
		case "head":
			w.H[0][0] = add(w.H[0][0].(*big.Int), bi(1))
		case "offset":
			w.COff[1] = add(w.COff[1].(*big.Int), bi(1))
		}
		if test.IsSolved(shape(), w, ecc.BN254.ScalarField()) == nil {
			t.Fatalf("accepted altered %s", alter)
		}
	}
	m := p.M
	m.Method = "affine"
	m.Basis = nil
	if _, _, err := ReuseCenterStats(m, newC); err == nil {
		t.Fatal("affine scores presented as a reusable head")
	}
}
