package circuit

import (
	"bytes"
	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/test"
	"math/big"
	"math/rand"
	"testing"
)

func TestRelativeReuseCircuitAndOriginalAssignments(t *testing.T) {
	rng := rand.New(rand.NewSource(10424))
	for trial := 0; trial < 6; trial++ {
		m := BoundModel{Version: BoundVersion, Method: "learned_energy", N: 4, D: 4, K: 2, InputBits: 4, BasisScale: 8, InputScale: "1", Centers: [][]int64{{0, 0, 0, 0}, {8, 0, 0, 0}}, Basis: [][]int64{{7, 1, 0, 0}, {0, 7, 1, 0}}}
		rows := make([][]int64, 4)
		for i := range rows {
			rows[i] = make([]int64, 4)
			for d := range rows[i] {
				rows[i][d] = int64(rng.Intn(13) - 6)
			}
		}
		centers := [][]int64{{0, 1, 0, 0}, {6, 1, 0, 0}, {-6, 0, 0, 0}}
		p, err := PrepareBound(m, rows, bi(13), bi(17))
		if err != nil {
			t.Fatal(err)
		}
		w, err := FillBoundRelativeReuse(p, centers, []int{0, 1, 2, 3})
		if err != nil {
			t.Fatal(err)
		}
		if err = test.IsSolved(NewBoundRelativeReuse(m, 4, 3), w, ecc.BN254.ScalarField()); err != nil {
			t.Fatal(err)
		}
		for i := range rows {
			single, err := FillBoundRelativeReuse(p, centers, []int{i})
			if err != nil {
				t.Fatal(err)
			}
			if single.RelativeCount.(int) == 1 {
				best, idx := int64(1<<62), 0
				for j, c := range centers {
					dist := int64(0)
					for d, x := range rows[i] {
						v := x - c[d]
						dist += v * v
					}
					if dist < best {
						best, idx = dist, j
					}
				}
				if single.Absolute.Query.Assign[0].(int) != idx {
					t.Fatal("false relative assignment certificate")
				}
			}
		}
	}
}

func TestRelativePreparedParametersPreservePublicWitness(t *testing.T) {
	p := reuseFixture(t)
	centers := [][]int64{{0, 1, 0, 0}, {2, 1, 0, 0}}
	params, err := NewRelativeReuseParameters(p.M, centers)
	if err != nil {
		t.Fatal(err)
	}
	for _, indices := range [][]int{{0, 1}, {2, 3}, {0, 1, 2, 3}} {
		w, err := FillBoundRelativeReusePrepared(p, params, indices)
		if err != nil {
			t.Fatal(err)
		}
		base := &w.Absolute
		public, err := BoundRelativeReusePublicPrepared(p.M, p.Statement(), params, indices,
			base.Query.Psi.(*big.Int).String(), base.Query.MarginSum.(*big.Int).String(),
			base.CertifiedCount.(int), w.RelativeCount.(int),
			base.CostLower.(*big.Int).String(), base.CostUpper.(*big.Int).String())
		if err != nil {
			t.Fatal(err)
		}
		full, err := frontend.NewWitness(w, ecc.BN254.ScalarField())
		if err != nil {
			t.Fatal(err)
		}
		expected, err := full.Public()
		if err != nil {
			t.Fatal(err)
		}
		actual, err := frontend.NewWitness(public, ecc.BN254.ScalarField(), frontend.PublicOnly())
		if err != nil {
			t.Fatal(err)
		}
		var a, b bytes.Buffer
		if _, err = expected.WriteTo(&a); err != nil {
			t.Fatal(err)
		}
		if _, err = actual.WriteTo(&b); err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(a.Bytes(), b.Bytes()) {
			t.Fatal("prepared public witness differs from prover statement")
		}
	}
	changed := p.M
	changed.InputScale = "2"
	if _, err := BoundRelativeReusePublicPrepared(changed, p.Statement(), params, []int{0}, "0", "0", 0, 0, "0", "0"); err == nil {
		t.Fatal("accepted cached parameters from a different model")
	}
	if _, err := FillBoundRelativeReusePrepared(p, nil, []int{0}); err == nil {
		t.Fatal("accepted missing parameters")
	}
}
func TestRelativeReuseFalseCountAndEnergyRadiusRejected(t *testing.T) {
	p := reuseFixture(t)
	centers := [][]int64{{0, 1, 0, 0}, {2, 1, 0, 0}}
	w, err := FillBoundRelativeReuse(p, centers, []int{0, 1, 2, 3})
	if err != nil {
		t.Fatal(err)
	}
	shape := NewBoundRelativeReuse(p.M, 4, 2)
	w.RelativeCount = w.RelativeCount.(int) + 1
	if err = test.IsSolved(shape, w, ecc.BN254.ScalarField()); err == nil {
		t.Fatal("false relative coverage accepted")
	}
	w, err = FillBoundRelativeReuse(p, centers, []int{0, 1, 2, 3})
	if err != nil {
		t.Fatal(err)
	}
	// Fixture row0 lies exactly in the basis span; row1 has nonzero residual.
	w.Absolute.RowRadii[1] = 0
	if err = test.IsSolved(shape, w, ecc.BN254.ScalarField()); err == nil {
		t.Fatal("unbounded residual accepted")
	}
}
