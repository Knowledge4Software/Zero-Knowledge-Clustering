package circuit

import (
	"math/big"
	"testing"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/frontend/cs/r1cs"
	"github.com/consensys/gnark/std/rangecheck"
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

type signedBoundCircuit struct {
	Value frontend.Variable
}

func (c *signedBoundCircuit) Define(api frontend.API) error {
	rangeCheckSigned(api, rangecheck.New(api), c.Value, big.NewInt(InputCoordinateBound), false)
	return nil
}

func TestInputCoordinateRangeRejectsOverflow(t *testing.T) {
	assert := test.NewAssert(t)
	circuit := &signedBoundCircuit{}
	assert.SolvingSucceeded(circuit, &signedBoundCircuit{Value: -InputCoordinateBound})
	assert.SolvingSucceeded(circuit, &signedBoundCircuit{Value: InputCoordinateBound})
	assert.SolvingFailed(circuit, &signedBoundCircuit{Value: InputCoordinateBound + 1})
	assert.SolvingFailed(circuit, &signedBoundCircuit{Value: -InputCoordinateBound - 1})
}

type sparseRowRangeCircuit struct {
	Row frontend.Variable
}

func (c *sparseRowRangeCircuit) Define(api frontend.API) error {
	rangeCheckUnsigned(api, rangecheck.New(api), c.Row, big.NewInt(2), false)
	return nil
}

func TestSparseRowRejectsNegativeAndOutOfRangeRepresentations(t *testing.T) {
	assert := test.NewAssert(t)
	circuit := &sparseRowRangeCircuit{}
	assert.SolvingSucceeded(circuit, &sparseRowRangeCircuit{Row: 0})
	assert.SolvingSucceeded(circuit, &sparseRowRangeCircuit{Row: 2})
	assert.SolvingFailed(circuit, &sparseRowRangeCircuit{Row: -1})
	assert.SolvingFailed(circuit, &sparseRowRangeCircuit{Row: 3})
}
