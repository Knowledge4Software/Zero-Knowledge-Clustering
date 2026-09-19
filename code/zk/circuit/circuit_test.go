package circuit

import (
	"testing"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/frontend/cs/r1cs"
	"github.com/consensys/gnark/test"
)

func TestSparseTiny(t *testing.T) {
	assert := test.NewAssert(t)
	p := Params{
		N: 2, D: 3, M: 2, K: 2, Q: 1,
		Variant: VariantSparse, SparseSelf: true, AuthenticateA: true,
		RangeBits: 32,
	}
	c := NewCircuit(p)
	_, err := frontend.Compile(ecc.BN254.ScalarField(), r1cs.NewBuilder, c)
	assert.NoError(err)
}
