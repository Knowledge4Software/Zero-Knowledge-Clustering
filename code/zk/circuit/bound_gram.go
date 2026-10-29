package circuit

import "math/big"

// For VALID bounded models, every Gram entry has magnitude <=(D+1)*T^2,
// and every absolute row sum <=128*10001*2^40 <2^63. The native int64 path
// is exact, not floating point. Unvalidated/unbounded callers keep big.Int.
func boundedGram64(m BoundModel) ([][]int64, bool) {
	if m.D < 1 || m.D > 10000 || m.BasisScale < 1 || m.BasisScale > 1<<20 || len(m.Basis) > 128 {
		return nil, false
	}
	for _, row := range m.Basis {
		if len(row) != m.D {
			return nil, false
		}
		for _, v := range row {
			if v < -m.BasisScale || v > m.BasisScale {
				return nil, false
			}
		}
	}
	g := make([][]int64, len(m.Basis))
	t2 := m.BasisScale * m.BasisScale
	for i, row := range m.Basis {
		g[i] = make([]int64, len(m.Basis))
		for j, other := range m.Basis {
			dot := int64(0)
			for d, v := range row {
				dot += v * other[d]
			}
			if i == j {
				dot -= t2
			}
			g[i][j] = dot
		}
	}
	return g, true
}

func pairedGramMatrix(m BoundModel) [][]*big.Int {
	if values, ok := boundedGram64(m); ok {
		g := make([][]*big.Int, len(values))
		for i, row := range values {
			g[i] = make([]*big.Int, len(row))
			for j, v := range row {
				g[i][j] = bi(v)
			}
		}
		return g
	}
	g := make([][]*big.Int, len(m.Basis))
	t2 := square(bi(m.BasisScale))
	for i, row := range m.Basis {
		g[i] = make([]*big.Int, len(m.Basis))
		for j, other := range m.Basis {
			value := bi(0)
			for d, v := range row {
				value.Add(value, mul(bi(v), bi(other[d])))
			}
			if i == j {
				value.Sub(value, t2)
			}
			g[i][j] = value
		}
	}
	return g
}
