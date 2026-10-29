package circuit

import (
	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark/test"
	"math/big"
	"math/rand"
	"testing"
)

func exactPairedCost(x, cs [][]int64) *big.Int {
	total := bi(0)
	for _, row := range x {
		var best *big.Int
		for _, c := range cs {
			v := bi(0)
			for j, z := range row {
				v.Add(v, square(bi(z-c[j])))
			}
			if best == nil || v.Cmp(best) < 0 {
				best = v
			}
		}
		total.Add(total, best)
	}
	return total
}

func TestPairedChangeBoundAndCircuitWithRoundedBasis(t *testing.T) {
	rng := rand.New(rand.NewSource(80921))
	for repetition := 0; repetition < 12; repetition++ {
		m := BoundModel{Version: BoundVersion, Method: "learned_energy", N: 4, D: 4, K: 2, InputBits: 4, BasisScale: 8, InputScale: "1", Centers: [][]int64{{0, 0, 0, 0}, {2, 0, 0, 0}}, Basis: [][]int64{{7, 1, 0, 0}, {0, 7, 1, 0}}}
		rows := make([][]int64, 4)
		for i := range rows {
			rows[i] = make([]int64, 4)
			for j := range rows[i] {
				rows[i][j] = int64(rng.Intn(17) - 8)
			}
		}
		centers := make([][]int64, 3)
		for i := range centers {
			centers[i] = make([]int64, 4)
			for j := range centers[i] {
				centers[i][j] = int64(rng.Intn(17) - 8)
			}
		}
		p, err := PrepareBound(m, rows, bi(2), bi(3))
		if err != nil {
			t.Fatal(err)
		}
		w, r, err := FillBoundPairedChange(p, m.Centers, centers, []int{0, 1, 2, 3})
		if err != nil {
			t.Fatal(err)
		}
		truth := mul(new(big.Int).Sub(exactPairedCost(rows, centers), exactPairedCost(rows, m.Centers)), new(big.Int).Exp(bi(8), bi(6), nil))
		if r.Lower.Cmp(truth) > 0 || r.Upper.Cmp(truth) < 0 {
			t.Fatal("paired interval missed exact change")
		}
		if err = test.IsSolved(NewBoundPairedChange(m, 4, 2, 3), w, ecc.BN254.ScalarField()); err != nil {
			t.Fatal(err)
		}
	}
}

func TestPairedChangeTamperingAndIdenticalCenters(t *testing.T) {
	p := reuseFixture(t)
	old := p.M.Centers
	next := [][]int64{{1, 0, 0, 0}, {0, 2, 0, 0}, {0, 0, 3, 1}}
	for _, kind := range []string{"lower", "upper", "coefficient", "constant", "delta_radius", "row_radius", "head", "index", "root"} {
		w, _, err := FillBoundPairedChange(p, old, next, []int{0, 2, 3})
		if err != nil {
			t.Fatal(err)
		}
		switch kind {
		case "lower":
			w.Lower = add(w.Lower.(*big.Int), bi(1))
		case "upper":
			w.Upper = add(w.Upper.(*big.Int), bi(1))
		case "coefficient":
			w.Coeff[0][0] = new(big.Int).Lsh(bi(1), 200) // public host rejects changed coefficients; circuit also rejects unsafe range
		case "constant":
			w.Constant[0] = new(big.Int).Lsh(bi(1), 200)
		case "delta_radius":
			w.DeltaRadius[0] = new(big.Int).Lsh(bi(1), 200)
		case "row_radius":
			w.RowRadius[1] = 0
		case "head":
			w.H[0][0] = add(w.H[0][0].(*big.Int), bi(1))
		case "index":
			w.Indices[1] = w.Indices[0]
		case "root":
			w.CH = add(p.CH, bi(1))
		}
		if test.IsSolved(NewBoundPairedChange(p.M, 3, len(old), len(next)), w, ecc.BN254.ScalarField()) == nil {
			t.Fatalf("accepted tampered %s", kind)
		}
	}
	w, r, err := FillBoundPairedChange(p, old, old, []int{0, 1, 2, 3})
	if err != nil {
		t.Fatal(err)
	}
	if r.Lower.Sign() != 0 || r.Upper.Sign() != 0 {
		t.Fatal("identical objectives must give zero change")
	}
	if err = test.IsSolved(NewBoundPairedChange(p.M, 4, len(old), len(old)), w, ecc.BN254.ScalarField()); err != nil {
		t.Fatal(err)
	}
	if _, err = BoundPairedChangePublic(p.M, p.Statement(), old, next, []int{0, 1}, "-01", "2"); err == nil {
		t.Fatal("accepted noncanonical signed integer")
	}
}
