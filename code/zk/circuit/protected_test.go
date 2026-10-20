package circuit

import (
	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark/constraint"
	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/frontend/cs/r1cs"
	"github.com/consensys/gnark/frontend/cs/scs"
	"testing"
)

func TestExperimentalProjectionAndScores(t *testing.T) {
	for _, backend := range []string{"r1cs", "scs"} {
		for _, score := range []bool{false, true} {
			for _, specialized := range []bool{false, true} {
				for _, protected := range []bool{false, true} {
					p := Params{N: 2, D: 4, M: 2, K: 2, Q: 2, Variant: VariantSparse, SparseSelf: true, AuthenticateA: true, RangeBits: 48, ScoreDistances: score}
					if specialized {
						p.FixedSparseRows = [][]int{{0, 1}, {0, 1}, {0, 1}, {0, 1}}
					}
					if protected {
						p.ProtectedBasis = [][]int64{{8, 0, 0, 0}}
						p.BasisScale = 8
					}
					var cs constraint.ConstraintSystem
					var err error
					if backend == "r1cs" {
						cs, err = frontend.Compile(ecc.BN254.ScalarField(), r1cs.NewBuilder, NewCircuit(p))
					} else {
						cs, err = frontend.Compile(ecc.BN254.ScalarField(), scs.NewBuilder, NewCircuit(p))
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
						valid := mutation == "honest" || mutation == "honest maximum distance" || (mutation == "honest mixed signs" && !specialized)
						if (err == nil) != valid {
							t.Fatalf("backend=%s score=%v specialized=%v protected=%v mutation=%s err=%v", backend, score, specialized, protected, mutation, err)
						}
					}
				}
			}
		}
	}
}

func TestScoreFullAndConstant(t *testing.T) {
	for _, backend := range []string{"r1cs", "scs"} {
		p := Params{N: 2, D: 4, M: 4, K: 2, Variant: VariantFull, AConstant: true, ScoreDistances: true, RangeBits: 48,
			ConstantA: [][]int64{{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}}}
		// Use the authenticated sparse fixture for X/centroids, then override
		// its objective with independently computed original-space distance.
		fixture := Params{N: 2, D: 4, M: 2, K: 2, Q: 2, Variant: VariantSparse, SparseSelf: true, AuthenticateA: true}
		w := optimizationWitness(fixture, "honest")
		full := NewCircuit(p)
		full.CX = w.CX
		full.CTheta = w.CTheta
		full.CA = 0
		full.Centroids = w.Centroids
		full.X = w.X
		full.RX = w.RX
		full.Assign = w.Assign
		full.PsiHat = int64(2 * 4 * 4096 * 4096)
		var cs constraint.ConstraintSystem
		var err error
		if backend == "r1cs" {
			cs, err = frontend.Compile(ecc.BN254.ScalarField(), r1cs.NewBuilder, NewCircuit(p))
		} else {
			cs, err = frontend.Compile(ecc.BN254.ScalarField(), scs.NewBuilder, NewCircuit(p))
		}
		if err != nil {
			t.Fatal(err)
		}
		witness, err := frontend.NewWitness(full, ecc.BN254.ScalarField())
		if err != nil {
			t.Fatal(err)
		}
		if err = cs.IsSolved(witness); err != nil {
			t.Fatal(err)
		}
	}
}
