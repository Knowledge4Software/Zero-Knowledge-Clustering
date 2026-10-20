package circuit

import (
	"math/big"
	"testing"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/frontend/cs/r1cs"
	"github.com/consensys/gnark/test"
)

// These are trust-boundary tests, not claims that the Stage-2 relation is
// unsound. A separate Stage-1 proof must authenticate the derivation of roots.
func TestQueryAuditEnergyNeedsPreprocessingBinding(t *testing.T) {
	p := QueryParams{N: 1, W: 4, K: 2, AssignDims: 1, CoordBits: 6,
		Cost: true, ConstCenters: true, EnergyCol: true, EnergyBits: 10,
		Centers: [][]int64{{0, 0, 0, 0}, {3, 0, 0, 0}}}
	salt := big.NewInt(789)
	z := [][]int64{{1, 100, 7, -2}}
	a, err := SolveQuery(p, z, p.Centers, salt)
	if err != nil { t.Fatal(err) }
	assert := test.NewAssert(t)
	assert.SolvingSucceeded(NewQueryCircuit(p), FillQueryWitness(p, z, p.Centers, salt, a), test.WithCurves(ecc.BN254))
	// Re-sealing false energy creates a different root, not a hash collision.
	// Stage 2 accepts this different compressed input as designed. Stage 1 must
	// reject it if it is asserted to represent the same original committed X.
	fake := [][]int64{{1, 0, 7, -2}}
	b, err := SolveQuery(p, fake, p.Centers, salt)
	if err != nil { t.Fatal(err) }
	if a.CZA.Cmp(b.CZA) != 0 || a.CZR.Cmp(b.CZR) == 0 || new(big.Int).Sub(a.Psi, b.Psi).Int64() != 100 {
		t.Fatal("unexpected commitment/cost change")
	}
	assert.SolvingSucceeded(NewQueryCircuit(p), FillQueryWitness(p, fake, p.Centers, salt, b), test.WithCurves(ecc.BN254))
}

func TestQueryAuditTailDoesNotAffectEnergyQueries(t *testing.T) {
	p := QueryParams{N: 1, W: 4, K: 2, AssignDims: 1, CoordBits: 6,
		Cost: true, ConstCenters: true, EnergyCol: true, EnergyBits: 10,
		Centers: [][]int64{{0, 0, 0, 0}, {3, 0, 0, 0}}}
	salt := big.NewInt(789)
	z := [][]int64{{1, 100, 7, -2}}
	a, err := SolveQuery(p, z, p.Centers, salt)
	if err != nil { t.Fatal(err) }
	short := p
	short.W = 2
	short.Centers = [][]int64{{0, 0}, {3, 0}}
	zShort := [][]int64{{1, 100}}
	b, err := SolveQuery(short, zShort, short.Centers, salt)
	if err != nil { t.Fatal(err) }
	if a.Psi.Cmp(b.Psi) != 0 || a.CZA.Cmp(b.CZA) != 0 || a.CZR.Cmp(b.CZR) != 0 || a.Assign[0] != b.Assign[0] {
		t.Fatal("removing the unused tail changed the public answer")
	}
	for _, cost := range []bool{false, true} {
		p.Cost, short.Cost = cost, cost
		wide, err := frontend.Compile(ecc.BN254.ScalarField(), r1cs.NewBuilder, NewQueryCircuit(p))
		if err != nil { t.Fatal(err) }
		narrow, err := frontend.Compile(ecc.BN254.ScalarField(), r1cs.NewBuilder, NewQueryCircuit(short))
		if err != nil { t.Fatal(err) }
		if wide.GetNbConstraints() != narrow.GetNbConstraints() {
			t.Fatalf("unused tail changed constraints: %d vs %d", wide.GetNbConstraints(), narrow.GetNbConstraints())
		}
	}
}
