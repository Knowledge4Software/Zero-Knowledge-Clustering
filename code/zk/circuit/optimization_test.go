package circuit

import (
	"math/big"
	"testing"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark/constraint"
	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/frontend/cs/r1cs"
	"github.com/consensys/gnark/frontend/cs/scs"
)

// Independently construct integer projections, assignments, costs and all
// commitments. Mutated row/sign/input cases recompute their commitments too.
func optimizationWitness(p Params, mutation string) *Circuit {
	w := NewCircuit(p)
	x := [][]int64{{4096, 4096, 4096, 4096}, {-4096, -4096, -4096, -4096}}
	centers := [][]int64{{0, 0, 0, 0}, {0, 0, 0, 0}} // exact tie
	if mutation == "honest maximum distance" {
		for j := range centers {
			for t := range centers[j] {
				centers[j][t] = -4096
			}
		}
	}
	if mutation == "honest mixed signs" {
		x = [][]int64{{-3, 21, -7, 4}, {50, -12, 9, -30}}
		centers = [][]int64{{1, -2, 3, -4}, {-5, 6, -7, 8}}
	}
	if mutation == "input range" {
		x[0][0] = 4097
	}
	if mutation == "center range" {
		centers[0][0] = -4097
	}
	a := make([][]int64, p.M)
	for r := range a {
		a[r] = make([]int64, p.D)
	}
	var matrixHash []*big.Int
	for t := 0; t < p.D; t++ {
		for l := 0; l < p.Q; l++ {
			row, sign := int64(l), int64(1)
			if mutation == "honest mixed signs" {
				row = int64((t + l) % p.M)
				if (t+l)%2 == 1 {
					sign = -1
				}
			}
			if mutation == "duplicate rows" && t == 0 && l == 1 {
				row = 0
			}
			if mutation == "negative row" && t == 0 && l == 0 {
				row = -1
			}
			if mutation == "large row" && t == 0 && l == 0 {
				row = int64(p.M)
			}
			if mutation == "invalid sign" && t == 0 && l == 0 {
				sign = 2
			}
			w.SparseRows[t][l], w.SparseSigns[t][l] = row, sign
			matrixHash = append(matrixHash, big.NewInt(row), big.NewInt(sign))
			if row >= 0 && row < int64(p.M) {
				a[row][t] += sign
			}
		}
	}
	project := func(v []int64) []int64 {
		z := make([]int64, p.M)
		for r := range z {
			for t := range v {
				z[r] += a[r][t] * v[t]
			}
		}
		if len(p.ProtectedBasis) == 0 {
			return z
		}
		head := make([]int64, len(p.ProtectedBasis))
		out := make([]int64, len(head)+p.M)
		for l, row := range p.ProtectedBasis {
			for t, value := range row {
				head[l] += value * v[t]
			}
			out[l] = p.BasisScale * head[l]
		}
		for r := range z {
			out[len(head)+r] = p.BasisScale * p.BasisScale * z[r]
			for l, row := range p.ProtectedBasis {
				var ab int64
				for t, value := range row {
					ab += a[r][t] * value
				}
				out[len(head)+r] -= ab * head[l]
			}
		}
		return out
	}
	var centerHash []*big.Int
	for j := range centers {
		for t, value := range centers[j] {
			w.Centroids[j][t] = value
			centerHash = append(centerHash, big.NewInt(value))
		}
	}
	w.CTheta, w.CA, w.RX = relationHash(centerHash...), relationHash(matrixHash...), 7
	var leaves []*big.Int
	var cost int64
	for i := range x {
		for t, value := range x[i] {
			w.X[i][t] = value
			leaves = append(leaves, relationHash(big.NewInt(value), big.NewInt(7)))
		}
		z := project(x[i])
		best, bestIdx := int64(1<<62), 0
		for j := range centers {
			zc := project(centers[j])
			var dist int64
			for r := range z {
				diff := z[r] - zc[r]
				weight := int64(1)
				if r < len(p.ProtectedBasis) {
					weight = int64(p.Q)
				}
				dist += weight * diff * diff
			}
			if dist < best {
				best, bestIdx = dist, j
			}
		}
		w.Assign[i] = bestIdx
		cost += best
	}
	count := 1
	for count < len(leaves) {
		count *= 2
	}
	for len(leaves) < count {
		leaves = append(leaves, relationHash(big.NewInt(0), big.NewInt(7)))
	}
	for len(leaves) > 1 {
		next := make([]*big.Int, len(leaves)/2)
		for i := range next {
			next[i] = relationHash(leaves[2*i], leaves[2*i+1])
		}
		leaves = next
	}
	w.CX, w.PsiHat = leaves[0], cost
	switch mutation {
	case "tie":
		w.Assign[0] = 1
	case "wrong objective":
		w.PsiHat = cost + 1
	case "wrong matrix hash":
		w.CA = 0
	case "wrong input hash":
		w.CX = 0
	case "wrong center hash":
		w.CTheta = 0
	}
	return w
}

func TestOptimizationPreservesRelation(t *testing.T) {
	for _, backend := range []string{"groth16", "plonk"} {
		for _, legacy := range []bool{false, true} {
			for _, noBatch := range []bool{false, true} {
				p := Params{N: 2, D: 4, M: 2, K: 2, Q: 2, Variant: VariantSparse,
					SparseSelf: true, AuthenticateA: true, RangeBits: 48,
					LegacyProjection: legacy, LegacyBounds: legacy, NoRangeBatch: noBatch}
				var cs constraint.ConstraintSystem
				var err error
				if backend == "plonk" {
					cs, err = frontend.Compile(ecc.BN254.ScalarField(), scs.NewBuilder, NewCircuit(p))
				} else {
					cs, err = frontend.Compile(ecc.BN254.ScalarField(), r1cs.NewBuilder, NewCircuit(p))
				}
				if err != nil {
					t.Fatal(err)
				}
				for _, mutation := range []string{"honest", "honest maximum distance", "honest mixed signs", "input range", "center range", "duplicate rows", "negative row", "large row", "invalid sign", "tie", "wrong objective", "wrong matrix hash", "wrong input hash", "wrong center hash"} {
					witness, err := frontend.NewWitness(optimizationWitness(p, mutation), ecc.BN254.ScalarField())
					if err != nil {
						t.Fatal(err)
					}
					err = cs.IsSolved(witness)
					valid := mutation == "honest" || mutation == "honest maximum distance" || mutation == "honest mixed signs"
					if (err == nil) != valid {
						t.Fatalf("%s legacy=%v noBatch=%v mutation=%s: %v", backend, legacy, noBatch, mutation, err)
					}
				}
			}
		}
	}
}

func TestDerivedProjectionBounds(t *testing.T) {
	for _, tc := range []struct {
		name     string
		p        Params
		expected int64
	}{
		{"sparse independent of q", Params{D: 256, SparseSelf: true, Q: 4}, 1048576},
		{"dense signs", Params{D: 256, Variant: VariantDense}, 1048576},
		{"coordinate", Params{D: 256, Variant: VariantCoord}, 4096},
		{"all-zero constant matrix", Params{AConstant: true, ConstantA: [][]int64{{0, 0}}}, 1},
		{"constant negative weights", Params{AConstant: true, ConstantA: [][]int64{{-2, 3}, {1, 0}}}, 20480},
		{"restrictive configured bound", Params{D: 256, SparseSelf: true, RangeBits: 12}, 2047},
		{"legacy bound", Params{D: 256, SparseSelf: true, RangeBits: 32, LegacyBounds: true}, 2147483647},
	} {
		t.Run(tc.name, func(t *testing.T) {
			bound, err := NewCircuit(tc.p).projectionBound()
			if err != nil || bound.Cmp(big.NewInt(tc.expected)) != 0 {
				t.Fatalf("bound=%v err=%v", bound, err)
			}
		})
	}
	// Abs(MinInt64) must not overflow int64 before big.Int conversion.
	c := &Circuit{P: Params{AConstant: true, ConstantA: [][]int64{{-1 << 63}}, RangeBits: 100}}
	bound, err := c.projectionBound()
	expected := new(big.Int).Lsh(big.NewInt(4096), 63)
	if err != nil || bound.Cmp(expected) != 0 {
		t.Fatalf("bound=%v err=%v", bound, err)
	}
}
