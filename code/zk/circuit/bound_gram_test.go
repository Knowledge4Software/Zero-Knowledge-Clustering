package circuit

import (
	"math"
	"math/big"
	"math/rand"
	"testing"
)

func gramOracle(m BoundModel) ([][]*big.Int, *big.Int) {
	g := make([][]*big.Int, len(m.Basis))
	maximum := bi(0)
	for i, row := range m.Basis {
		g[i] = make([]*big.Int, len(m.Basis))
		sum := bi(0)
		for j, other := range m.Basis {
			v := bi(0)
			for d, a := range row {
				v.Add(v, new(big.Int).Mul(big.NewInt(a), big.NewInt(other[d])))
			}
			if i == j {
				v.Sub(v, new(big.Int).Mul(big.NewInt(m.BasisScale), big.NewInt(m.BasisScale)))
			}
			g[i][j] = v
			sum.Add(sum, new(big.Int).Abs(v))
		}
		if sum.Cmp(maximum) > 0 {
			maximum.Set(sum)
		}
	}
	return g, maximum
}

func TestExactNativeGramMatchesBigIntegerOracle(t *testing.T) {
	rng := rand.New(rand.NewSource(20261025))
	for _, scale := range []int64{64, 1024, 16384, 1 << 20} {
		m := BoundModel{D: 17, BasisScale: scale, Basis: make([][]int64, 9)}
		for i := range m.Basis {
			m.Basis[i] = make([]int64, m.D)
			for d := range m.Basis[i] {
				m.Basis[i][d] = rng.Int63n(2*scale+1) - scale
			}
		}
		want, defect := gramOracle(m)
		got := pairedGramMatrix(m)
		if m.GramDefect().Cmp(defect) != 0 {
			t.Fatal("native defect differs from integer oracle")
		}
		for i := range want {
			for j := range want[i] {
				if got[i][j].Cmp(want[i][j]) != 0 {
					t.Fatal("native Gram entry differs")
				}
			}
		}
	}
	// Prove the worst allowed absolute-row-sum bound fits signed int64.
	worst := new(big.Int).Mul(big.NewInt(128*10001), new(big.Int).Lsh(big.NewInt(1), 40))
	if worst.Cmp(big.NewInt(math.MaxInt64)) >= 0 {
		t.Fatal("int64 proof no longer holds")
	}
}

func TestUnboundedNativeGramFallsBackWithoutOverflow(t *testing.T) {
	m := BoundModel{D: 2, BasisScale: 1 << 20, Basis: [][]int64{{math.MaxInt64, math.MinInt64 + 1}, {1, -1}}}
	if _, ok := boundedGram64(m); ok {
		t.Fatal("unbounded basis entered int64 path")
	}
	want, defect := gramOracle(m)
	got := pairedGramMatrix(m)
	if m.GramDefect().Cmp(defect) != 0 {
		t.Fatal("unbounded defect overflow")
	}
	for i := range want {
		for j := range want[i] {
			if got[i][j].Cmp(want[i][j]) != 0 {
				t.Fatal("unbounded Gram overflow")
			}
		}
	}
}
