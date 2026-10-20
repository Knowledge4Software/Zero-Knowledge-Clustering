package circuit

import (
	"math/big"
	"math/rand"
	"testing"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark/test"
)

func TestCertifiedReuseRoundingTermAndTrueCost(t *testing.T) {
	m := BoundModel{Version: BoundVersion, Method: "learned_energy", N: 2, D: 1, K: 2, InputBits: 4, BasisScale: 8, InputScale: "1", Centers: [][]int64{{0}, {2}}, Basis: [][]int64{{7}}}
	p, err := PrepareBound(m, [][]int64{{1}, {3}}, bi(2), bi(3))
	if err != nil {
		t.Fatal(err)
	}
	centers := [][]int64{{0}, {2}}
	heads, offs, err := ReuseCenterStats(m, centers)
	if err != nil {
		t.Fatal(err)
	}
	w, r, err := FillBoundCertifiedReuseQuery(p, heads, offs, []int{0, 1})
	if err != nil {
		t.Fatal(err)
	}
	if m.GramDefect().Cmp(bi(15)) != 0 {
		t.Fatal("wrong rounded Gram defect")
	}
	// For x=1,c=0 the center residual is zero, yet D=3361 != T^4=4096.
	// A cross-term-only certificate would therefore be unsound.
	if square(p.H[0][0]).Cmp(bi(3136)) != 0 || p.E[0].Cmp(bi(225)) != 0 {
		t.Fatal("invalid rounding fixture")
	}
	truth := mul(bi(2), bi(262144)) // T^6 * two exact squared distances of one
	if r.Lower.Cmp(truth) > 0 || r.Upper.Cmp(truth) < 0 {
		t.Fatal("interval missed original quantized cost")
	}
	if err = test.IsSolved(NewBoundCertifiedReuseQuery(m, 2, 2), w, ecc.BN254.ScalarField()); err != nil {
		t.Fatal(err)
	}
}

func TestCertifiedReuseHonestAndTamperedStatements(t *testing.T) {
	p := reuseFixture(t)
	centers := [][]int64{{1, 0, 0, 0}, {0, 2, 0, 0}, {0, 0, 3, 1}}
	heads, offs, err := ReuseCenterStats(p.M, centers)
	if err != nil {
		t.Fatal(err)
	}
	indices := []int{0, 2, 3}
	shape := NewBoundCertifiedReuseQuery(p.M, 3, 3)
	w, _, err := FillBoundCertifiedReuseQuery(p, heads, offs, indices)
	if err != nil {
		t.Fatal(err)
	}
	if err = test.IsSolved(shape, w, ecc.BN254.ScalarField()); err != nil {
		t.Fatal(err)
	}
	for _, alter := range []string{"count", "lower", "upper", "row_radius", "center_radius", "cost", "assignment"} {
		w, _, _ := FillBoundCertifiedReuseQuery(p, heads, offs, indices)
		switch alter {
		case "count":
			w.CertifiedCount = w.CertifiedCount.(int) + 1
		case "lower":
			w.CostLower = add(w.CostLower.(*big.Int), bi(1))
		case "upper":
			w.CostUpper = add(w.CostUpper.(*big.Int), bi(1))
		case "row_radius":
			w.RowRadii[1] = 0
		case "center_radius":
			w.CenterRadii[2] = 0
		case "cost":
			w.Query.Psi = add(w.Query.Psi.(*big.Int), bi(1))
		case "assignment":
			w.Query.Assign[0] = (w.Query.Assign[0].(int) + 1) % 3
		}
		if test.IsSolved(shape, w, ecc.BN254.ScalarField()) == nil {
			t.Fatalf("accepted altered certificate %s", alter)
		}
	}
}

func TestCertifiedReuseNeverCertifiesWrongOriginalAssignment(t *testing.T) {
	rng := rand.New(rand.NewSource(491))
	for repetition := 0; repetition < 24; repetition++ {
		m := BoundModel{Version: BoundVersion, Method: "learned_energy", N: 4, D: 4, K: 2, InputBits: 4, BasisScale: 8, InputScale: "1", Centers: [][]int64{{0, 0, 0, 0}, {2, 0, 0, 0}}, Basis: [][]int64{{7, 1, 0, 0}, {0, 7, 1, 0}}}
		x := make([][]int64, 4)
		for i := range x {
			x[i] = make([]int64, 4)
			for j := range x[i] {
				x[i][j] = int64(rng.Intn(17) - 8)
			}
		}
		centers := make([][]int64, 3)
		for j := range centers {
			centers[j] = make([]int64, 4)
			for k := range centers[j] {
				centers[j][k] = int64(rng.Intn(17) - 8)
			}
		}
		p, err := PrepareBound(m, x, bi(2), bi(3))
		if err != nil {
			t.Fatal(err)
		}
		heads, offs, _ := ReuseCenterStats(m, centers)
		r, err := p.ReuseCertificate(heads, offs, []int{0, 1, 2, 3})
		if err != nil {
			t.Fatal(err)
		}
		fm := m
		fm.Method = "full"
		fm.Basis = nil
		fp, err := PrepareBound(fm, x, bi(2), bi(3))
		if err != nil {
			t.Fatal(err)
		}
		labels, phi, err := FullReuseAnswer(fp, centers, []int{0, 1, 2, 3})
		if err != nil {
			t.Fatal(err)
		}
		truth := mul(phi, new(big.Int).Exp(bi(m.BasisScale), bi(6), nil))
		if r.Lower.Cmp(truth) > 0 || r.Upper.Cmp(truth) < 0 {
			t.Fatal("cost certificate missed original-space optimum")
		}
		for i, flag := range r.Flags {
			if flag && r.Labels[i] != labels[i] {
				t.Fatal("certified an incorrect original-space assignment")
			}
		}
	}
}
