package circuit

import (
	"math/big"
	"math/rand"
	"testing"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark/test"
)

func randomQuery(rng *rand.Rand, n, w, k, bits int) ([][]int64, [][]int64) {
	bound := int64(1) << uint(bits)
	draw := func(rows int) [][]int64 {
		out := make([][]int64, rows)
		for i := range out {
			out[i] = make([]int64, w)
			for r := range out[i] {
				out[i][r] = rng.Int63n(2*bound+1) - bound
			}
		}
		return out
	}
	return draw(n), draw(k)
}

func TestQueryRelationAllModes(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	salt := big.NewInt(12345)
	for _, cost := range []bool{false, true} {
		for _, constant := range []bool{false, true} {
			for _, assignDims := range []int{2, 5} {
				z, centers := randomQuery(rng, 6, 5, 3, 6)
				p := QueryParams{N: 6, W: 5, K: 3, AssignDims: assignDims, CoordBits: 6, Cost: cost, ConstCenters: constant}
				if constant {
					p.Centers = centers
				}
				ans, err := SolveQuery(p, z, centers, salt)
				if err != nil {
					t.Fatal(err)
				}
				assert := test.NewAssert(t)
				good := FillQueryWitness(p, z, centers, salt, ans)
				assert.SolvingSucceeded(NewQueryCircuit(p), good, test.WithCurves(ecc.BN254))

				// A wrong assignment, a wrong cost or a tampered Z must be rejected.
				bad := FillQueryWitness(p, z, centers, salt, ans)
				bad.Assign[0] = (ans.Assign[0] + 1) % p.K
				assert.SolvingFailed(NewQueryCircuit(p), bad, test.WithCurves(ecc.BN254))
				tampered := FillQueryWitness(p, z, centers, salt, ans)
				tampered.Z[1][0] = z[1][0] + 1
				assert.SolvingFailed(NewQueryCircuit(p), tampered, test.WithCurves(ecc.BN254))
				if cost {
					wrong := FillQueryWitness(p, z, centers, salt, ans)
					wrong.PsiHat = new(big.Int).Add(ans.Psi, big.NewInt(1))
					assert.SolvingFailed(NewQueryCircuit(p), wrong, test.WithCurves(ecc.BN254))
				}
			}
		}
	}
}

func TestQueryEnergyColumn(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	salt := big.NewInt(777)
	for _, constant := range []bool{false, true} {
		z, centers := randomQuery(rng, 6, 6, 3, 6)
		for i := range z {
			z[i][2] = rng.Int63n(1 << 10) // column 2 = residual energy, non-negative
		}
		for j := range centers {
			centers[j][2] = 0
		}
		p := QueryParams{N: 6, W: 6, K: 3, AssignDims: 2, CoordBits: 6, Cost: true,
			ConstCenters: constant, EnergyCol: true, EnergyBits: 10}
		if constant {
			p.Centers = centers
		}
		ans, err := SolveQuery(p, z, centers, salt)
		if err != nil {
			t.Fatal(err)
		}
		assert := test.NewAssert(t)
		assert.SolvingSucceeded(NewQueryCircuit(p), FillQueryWitness(p, z, centers, salt, ans), test.WithCurves(ecc.BN254))
		wrong := FillQueryWitness(p, z, centers, salt, ans)
		wrong.PsiHat = new(big.Int).Sub(ans.Psi, big.NewInt(1))
		assert.SolvingFailed(NewQueryCircuit(p), wrong, test.WithCurves(ecc.BN254))
		negative := FillQueryWitness(p, z, centers, salt, ans)
		negative.Z[0][2] = -1 // a negative energy must be rejected
		assert.SolvingFailed(NewQueryCircuit(p), negative, test.WithCurves(ecc.BN254))
		hidden := FillQueryWitness(p, z, centers, salt, ans)
		hidden.Z[1][2] = z[1][2] - 1 // altering the committed energy breaks the opening
		hidden.PsiHat = new(big.Int).Sub(ans.Psi, big.NewInt(1))
		assert.SolvingFailed(NewQueryCircuit(p), hidden, test.WithCurves(ecc.BN254))
	}
}
