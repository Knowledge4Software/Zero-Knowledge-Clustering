package circuit

import (
	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark/test"
	"math/big"
	"testing"
)

func TestBoundLinearTailUsesAndAuthenticatesResidual(t *testing.T) {
	p, err := boundFixture("protected_linear")
	if err != nil {
		t.Fatal(err)
	}
	// Use non-axis, rounded directions so centers have nonzero residual tails.
	m := p.M
	m.Basis = [][]int64{{6, 3, 0, 0}}
	p, err = PrepareBound(m, p.X, p.SaltX, p.SaltZ)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < m.N; i++ {
		if p.E[i].Sign() != 0 {
			t.Fatal("no-energy ablation retained energy")
		}
		if err = test.IsSolved(NewBoundPreprocess(m), FillBoundPreprocess(p, i), ecc.BN254.ScalarField()); err != nil {
			t.Fatal(err)
		}
	}
	indices := []int{0, 1, 2, 3}
	labels, cost, err := p.Answer(indices)
	if err != nil {
		t.Fatal(err)
	}
	expected := bi(0)
	for a, i := range indices {
		var best *big.Int
		winner := 0
		for j, row := range m.Centers {
			h, _, tail := BoundStats(m, row)
			d := bi(0)
			for t, v := range p.H[i] {
				d.Add(d, square(new(big.Int).Sub(v, h[t])))
			}
			for t, v := range p.T[i] {
				d.Add(d, square(new(big.Int).Sub(v, tail[t])))
			}
			if best == nil || d.Cmp(best) < 0 {
				best = d
				winner = j
			}
		}
		if labels[a] != winner {
			t.Fatal("wrong nearest embedded center")
		}
		expected.Add(expected, best)
	}
	if expected.Cmp(cost) != 0 {
		t.Fatal("wrong embedded cost")
	}
	w, err := FillBoundQuery(p, indices)
	if err != nil {
		t.Fatal(err)
	}
	if err = test.IsSolved(NewBoundQuery(m, 4), w, ecc.BN254.ScalarField()); err != nil {
		t.Fatal(err)
	}
	w.Tail[0][0] = add(w.Tail[0][0].(*big.Int), bi(1))
	if test.IsSolved(NewBoundQuery(m, 4), w, ecc.BN254.ScalarField()) == nil {
		t.Fatal("accepted altered tail opening")
	}
	w, _ = FillBoundQuery(p, indices)
	w.Psi = add(cost, bi(1))
	if test.IsSolved(NewBoundQuery(m, 4), w, ecc.BN254.ScalarField()) == nil {
		t.Fatal("accepted altered cost")
	}
	g, err := FillBoundGeometry(p, []int{0, 2})
	if err != nil {
		t.Fatal(err)
	}
	if err = test.IsSolved(NewBoundGeometry(m), g, ecc.BN254.ScalarField()); err != nil {
		t.Fatal(err)
	}
}
