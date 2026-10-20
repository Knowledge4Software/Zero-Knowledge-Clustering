package circuit

import (
	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark/constraint"
	"github.com/consensys/gnark/constraint/solver"
	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/frontend/cs/r1cs"
	"github.com/consensys/gnark/frontend/cs/scs"
	"math/big"
	"testing"
)

func TestLookupSparseRelation(t *testing.T) {
	for _, backend := range []string{"r1cs", "scs"} {
		for _, protected := range []bool{false, true} {
			p := Params{N: 2, D: 4, M: 2, K: 2, Q: 2, Variant: VariantSparse, SparseSelf: true, AuthenticateA: true, RangeBits: 48, ScoreDistances: true, LookupSparse: true}
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
				w, err := frontend.NewWitness(optimizationWitness(p, mutation), ecc.BN254.ScalarField())
				if err != nil {
					t.Fatal(err)
				}
				err = cs.IsSolved(w)
				valid := mutation == "honest" || mutation == "honest maximum distance" || mutation == "honest mixed signs"
				if (err == nil) != valid {
					t.Fatalf("backend=%s protected=%v mutation=%s err=%v", backend, protected, mutation, err)
				}
			}
		}
	}
}

// Isolate the permutation/ordering checks from hashing and argmin. A forged
// accumulator path is paired with its forged public output, so an output
// equality cannot alone reject a dropped contribution.
type lookupOnlyCircuit struct {
	X      [4]frontend.Variable
	Target [2]frontend.Variable `gnark:",public"`
}

func (c *lookupOnlyCircuit) Define(api frontend.API) error {
	p := Params{D: 4, M: 2, Q: 2}
	inner := &Circuit{P: p, SparseRows: make([][]frontend.Variable, 4), SparseSigns: make([][]frontend.Variable, 4)}
	for t := 0; t < 4; t++ {
		inner.SparseRows[t] = []frontend.Variable{0, 1}
		inner.SparseSigns[t] = []frontend.Variable{1, 1}
	}
	z := inner.lookupProject(api, c.X[:])
	for r := range z {
		api.AssertIsEqual(z[r], c.Target[r])
	}
	return nil
}
func TestLookupRejectsForgedPaths(t *testing.T) {
	cs, err := frontend.Compile(ecc.BN254.ScalarField(), r1cs.NewBuilder, &lookupOnlyCircuit{})
	if err != nil {
		t.Fatal(err)
	}
	for _, attack := range []string{"honest", "fork", "cycle", "wrong accumulator", "wrong terminal"} {
		assignment := &lookupOnlyCircuit{X: [4]frontend.Variable{1, 2, 3, 4}, Target: [2]frontend.Variable{10, 10}}
		if attack == "fork" {
			assignment.Target = [2]frontend.Variable{9, 10}
		}
		if attack == "wrong terminal" {
			assignment.Target = [2]frontend.Variable{0, 10}
		}
		w, err := frontend.NewWitness(assignment, ecc.BN254.ScalarField())
		if err != nil {
			t.Fatal(err)
		}
		hint := func(field *big.Int, in, out []*big.Int) error {
			if err := sparseRouteHint(field, in, out); err != nil {
				return err
			}
			switch attack {
			case "fork":
				// Node 2 reuses bucket-zero's root, dropping the first update.
				out[8+2].SetInt64(0)
				for i := 0; i < 8; i++ {
					previous := int(out[8+i].Int64())
					before := new(big.Int)
					if previous >= 2 {
						before.Set(out[previous-2])
					}
					increment := new(big.Int).Mul(in[3+i/2], in[3+4+2*i+1])
					out[i].Add(before, increment)
					out[i].Mod(out[i], field)
				}
			case "cycle":
				out[8].SetInt64(2)
			case "wrong accumulator":
				out[0].Add(out[0], big.NewInt(1))
			case "wrong terminal":
				out[16].SetInt64(0)
			}
			return nil
		}
		err = cs.IsSolved(w, solver.OverrideHint(solver.GetHintID(sparseRouteHint), hint))
		if (err == nil) != (attack == "honest") {
			t.Fatalf("attack=%s err=%v", attack, err)
		}
	}
}
