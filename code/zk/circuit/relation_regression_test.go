package circuit

import (
	"math/big"
	"testing"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark-crypto/ecc/bn254/fr"
	cryptomimc "github.com/consensys/gnark-crypto/ecc/bn254/fr/mimc"
	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/frontend/cs/r1cs"
)

func relationHash(values ...*big.Int) *big.Int {
	h := cryptomimc.NewFieldHasher()
	for _, value := range values {
		var e fr.Element
		e.SetBigInt(value)
		h.WriteElement(e)
	}
	result := h.SumElement()
	return result.BigInt(new(big.Int))
}

// Recompute commitments even for malicious inputs so failures cannot be
// explained merely by a stale hash. Run the actual compiled R1CS solver.
func TestFullRelationRejectsMalformedWitnesses(t *testing.T) {
	for _, noBatch := range []bool{false, true} {
		p := Params{N: 1, D: 1, M: 1, K: 1, Q: 1, Variant: VariantSparse,
			SparseSelf: true, AuthenticateA: true, RangeBits: 32, NoRangeBatch: noBatch}
		cs, err := frontend.Compile(ecc.BN254.ScalarField(), r1cs.NewBuilder, NewCircuit(p))
		if err != nil {
			t.Fatal(err)
		}
		cases := []struct {
			name                               string
			x, center, row, sign, cost, assign int64
			valid                              bool
		}{
			{"positive boundary", 4096, 0, 0, 1, 4096 * 4096, 0, true},
			{"negative boundary", -4096, 0, 0, 1, 4096 * 4096, 0, true},
			{"input too large", 4097, 0, 0, 1, 4097 * 4097, 0, false},
			{"input too negative", -4097, 0, 0, 1, 4097 * 4097, 0, false},
			{"centroid too large", 0, 4097, 0, 1, 4097 * 4097, 0, false},
			{"negative row", 3, 0, -1, 1, 0, 0, false},
			{"row equals m", 3, 0, 1, 1, 0, 0, false},
			{"invalid sign", 3, 0, 0, 2, 36, 0, false},
			{"wrong cost", 3, 0, 0, 1, 8, 0, false},
			{"negative assignment", 3, 0, 0, 1, 9, -1, false},
		}
		for _, tc := range cases {
			t.Run(tc.name, func(t *testing.T) {
				w := NewCircuit(p)
				w.X[0][0], w.Centroids[0][0], w.RX = tc.x, tc.center, 7
				w.SparseRows[0][0], w.SparseSigns[0][0] = tc.row, tc.sign
				w.Assign[0], w.PsiHat = tc.assign, tc.cost
				w.CX = relationHash(big.NewInt(tc.x), big.NewInt(7))
				w.CTheta = relationHash(big.NewInt(tc.center))
				w.CA = relationHash(big.NewInt(tc.row), big.NewInt(tc.sign))
				witness, err := frontend.NewWitness(w, ecc.BN254.ScalarField())
				if err != nil {
					t.Fatal(err)
				}
				err = cs.IsSolved(witness)
				if (err == nil) != tc.valid {
					t.Fatalf("valid=%v, solver error=%v", tc.valid, err)
				}
			})
		}
	}
}

func TestDenseFieldWrapCannotMasqueradeAsSmallProjection(t *testing.T) {
	p := Params{N: 1, D: 1, M: 1, K: 1, Q: 1, Variant: VariantDense, AuthenticateA: true, RangeBits: 32}
	cs, err := frontend.Compile(ecc.BN254.ScalarField(), r1cs.NewBuilder, NewCircuit(p))
	if err != nil {
		t.Fatal(err)
	}
	// 2 * ((p+1)/2) == 1 mod p, but is NOT the integer projection 1.
	coefficient := new(big.Int).Add(ecc.BN254.ScalarField(), big.NewInt(1))
	coefficient.Div(coefficient, big.NewInt(2))
	for _, v := range []*big.Int{big.NewInt(1), coefficient} {
		w := NewCircuit(p)
		w.X[0][0], w.Centroids[0][0], w.RX = 2, 0, 7
		w.DenseA[0][0], w.Assign[0] = v, 0
		w.PsiHat = 4
		if v.Cmp(coefficient) == 0 {
			w.PsiHat = 1
		}
		w.CX = relationHash(big.NewInt(2), big.NewInt(7))
		w.CTheta = relationHash(big.NewInt(0))
		w.CA = relationHash(v)
		witness, err := frontend.NewWitness(w, ecc.BN254.ScalarField())
		if err != nil {
			t.Fatal(err)
		}
		err = cs.IsSolved(witness)
		if (err == nil) != (v.Int64() == 1) {
			t.Fatalf("coefficient=%v, error=%v", v, err)
		}
	}
}
