package circuit

import (
	"fmt"
	"math/big"

	"github.com/consensys/gnark-crypto/ecc/bn254/fr"
	nativemimc "github.com/consensys/gnark-crypto/ecc/bn254/fr/mimc"
)

// QueryAnswer is the native (out-of-circuit) result for a query instance.
type QueryAnswer struct {
	Assign   []int
	Psi      *big.Int
	CZA, CZR *big.Int
	CTheta   *big.Int
}

func feBytes(v *big.Int) []byte {
	var e fr.Element
	e.SetBigInt(v)
	b := e.Bytes()
	return b[:]
}

func nativeHash(values ...*big.Int) *big.Int {
	h := nativemimc.NewMiMC()
	for _, v := range values {
		h.Write(feBytes(v))
	}
	return new(big.Int).SetBytes(h.Sum(nil))
}

func nativeMerkleRoot(values []int64, salt *big.Int) *big.Int {
	level := make([]*big.Int, len(values))
	for i, v := range values {
		level[i] = nativeHash(big.NewInt(v), salt)
	}
	for len(level) > 1 {
		next := make([]*big.Int, 0, (len(level)+1)/2)
		for i := 0; i+1 < len(level); i += 2 {
			next = append(next, nativeHash(level[i], level[i+1]))
		}
		if len(level)%2 == 1 {
			next = append(next, level[len(level)-1])
		}
		level = next
	}
	return level[0]
}

// SolveQuery computes the relation's public outputs exactly: first-index
// nearest centre on the assignment block, the full-width cost and the roots.
func SolveQuery(p QueryParams, z [][]int64, centers [][]int64, salt *big.Int) (QueryAnswer, error) {
	if len(z) != p.N || len(centers) != p.K {
		return QueryAnswer{}, fmt.Errorf("instance shape mismatch")
	}
	ans := QueryAnswer{Assign: make([]int, p.N), Psi: big.NewInt(0), CZR: big.NewInt(0), CTheta: big.NewInt(0)}
	sq := func(a, b []int64, from, to int) *big.Int {
		s := big.NewInt(0)
		for r := from; r < to; r++ {
			d := big.NewInt(a[r] - b[r])
			s.Add(s, d.Mul(d, d))
		}
		return s
	}
	for i, row := range z {
		if len(row) != p.W {
			return QueryAnswer{}, fmt.Errorf("row width mismatch")
		}
		best := 0
		bestD := sq(row, centers[0], 0, p.AssignDims)
		for j := 1; j < p.K; j++ {
			if d := sq(row, centers[j], 0, p.AssignDims); d.Cmp(bestD) < 0 {
				best, bestD = j, d
			}
		}
		ans.Assign[i] = best
		if p.Cost && p.EnergyCol {
			ans.Psi.Add(ans.Psi, sq(row, centers[best], 0, p.AssignDims))
			ans.Psi.Add(ans.Psi, big.NewInt(row[p.AssignDims]))
		} else if p.Cost {
			ans.Psi.Add(ans.Psi, sq(row, centers[best], 0, p.W))
		}
	}
	flat := func(from, to int) []int64 {
		out := make([]int64, 0, p.N*(to-from))
		for _, row := range z {
			out = append(out, row[from:to]...)
		}
		return out
	}
	ans.CZA = nativeMerkleRoot(flat(0, p.AssignDims), salt)
	if p.Cost && p.EnergyCol {
		ans.CZR = nativeMerkleRoot(flat(p.AssignDims, p.AssignDims+1), salt)
	} else if p.Cost && p.AssignDims < p.W {
		ans.CZR = nativeMerkleRoot(flat(p.AssignDims, p.W), salt)
	}
	if !p.ConstCenters {
		values := make([]*big.Int, 0, p.K*p.W)
		for _, row := range centers {
			for _, v := range row {
				values = append(values, big.NewInt(v))
			}
		}
		ans.CTheta = nativeHash(values...)
	}
	return ans, nil
}

// FillQueryWitness assigns a QueryCircuit from integer data and its answer.
func FillQueryWitness(p QueryParams, z [][]int64, centers [][]int64, salt *big.Int, ans QueryAnswer) *QueryCircuit {
	w := NewQueryCircuit(p)
	w.CZA, w.CZR, w.CTheta, w.RZ = ans.CZA, ans.CZR, ans.CTheta, salt
	w.PsiHat = big.NewInt(0)
	if p.Cost {
		w.PsiHat = ans.Psi
	}
	for i := range z {
		for r := range z[i] {
			w.Z[i][r] = z[i][r]
		}
		w.Assign[i] = ans.Assign[i]
	}
	if !p.ConstCenters {
		for j := range centers {
			for r := range centers[j] {
				w.Centers[j][r] = centers[j][r]
			}
		}
	}
	return w
}
