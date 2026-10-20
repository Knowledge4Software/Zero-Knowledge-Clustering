package circuit

import (
	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark/test"
	"testing"
)

func TestBoundCachedAnswersMatchFullAndRejectFalseStatistics(t *testing.T) {
	p, err := boundFixture("cached")
	if err != nil {
		t.Fatal(err)
	}
	full, err := boundFixture("full")
	if err != nil {
		t.Fatal(err)
	}
	indices := []int{0, 1, 2, 3}
	labels, cost, _ := p.Answer(indices)
	want, phi, _ := full.Answer(indices)
	if cost.Cmp(phi) != 0 {
		t.Fatal("cached cost differs from Full")
	}
	for i := range labels {
		if labels[i] != want[i] {
			t.Fatal("cached assignment differs from Full")
		}
	}
	for i := 0; i < p.M.N; i++ {
		if err = test.IsSolved(NewBoundPreprocess(p.M), FillBoundPreprocess(p, i), ecc.BN254.ScalarField()); err != nil {
			t.Fatal(err)
		}
	}
	w, err := FillBoundQuery(p, indices)
	if err != nil {
		t.Fatal(err)
	}
	if err = test.IsSolved(NewBoundQuery(p.M, 4), w, ecc.BN254.ScalarField()); err != nil {
		t.Fatal(err)
	}
	w.Assign[0] = 1 - labels[0]
	if test.IsSolved(NewBoundQuery(p.M, 4), w, ecc.BN254.ScalarField()) == nil {
		t.Fatal("accepted wrong assignment")
	}
	w, _ = FillBoundQuery(p, indices)
	w.Psi = add(cost, bi(1))
	if test.IsSolved(NewBoundQuery(p.M, 4), w, ecc.BN254.ScalarField()) == nil {
		t.Fatal("accepted wrong cost")
	}
	for _, block := range []string{"winner", "cost"} {
		w := FillBoundPreprocess(p, 0)
		if block == "winner" {
			w.H[0] = bi(int64(1 - labels[0]))
		} else {
			w.E = add(p.E[0], bi(1))
		}
		if test.IsSolved(NewBoundPreprocess(p.M), w, ecc.BN254.ScalarField()) == nil {
			t.Fatal("accepted altered cached statistic")
		}
	}
	m := p.M
	m.Centers[1] = append([]int64{}, m.Centers[0]...)
	tied, err := PrepareBound(m, p.X, p.SaltX, p.SaltZ)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < m.N; i++ {
		if tied.H[i][0].Sign() != 0 {
			t.Fatal("tie did not choose first index")
		}
		if err = test.IsSolved(NewBoundPreprocess(m), FillBoundPreprocess(tied, i), ecc.BN254.ScalarField()); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := FillBoundGeometry(p, []int{0, 2}); err == nil {
		t.Fatal("cached answers presented as an embedding")
	}
}
