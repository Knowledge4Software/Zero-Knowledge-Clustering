package circuit

// Generic authenticated sparse scatter without an M*D coefficient matrix.
// Accumulator nodes form disjoint per-bucket paths. Lookup authenticates
// predecessor row/value; ordering prevents cycles; a committed-challenge
// permutation check prevents forks and dropped contributions. Experimental:
// external security review is required before production deployment.
import (
	"fmt"
	"github.com/consensys/gnark/constraint/solver"
	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/std/lookup/logderivlookup"
	"github.com/consensys/gnark/std/math/cmp"
	"github.com/consensys/gnark/std/multicommit"
	"github.com/consensys/gnark/std/rangecheck"
	"math/big"
)

func init() { solver.RegisterHint(sparseRouteHint) }

func sparseRouteHint(field *big.Int, inputs, outputs []*big.Int) error {
	if len(inputs) < 3 || !inputs[0].IsInt64() || !inputs[1].IsInt64() || !inputs[2].IsInt64() {
		return fmt.Errorf("invalid sparse routing header")
	}
	m, d, q := int(inputs[0].Int64()), int(inputs[1].Int64()), int(inputs[2].Int64())
	slots := d * q
	if m < 1 || d < 1 || q < 1 || len(inputs) != 3+d+2*slots || len(outputs) != 2*slots+m {
		return fmt.Errorf("invalid sparse routing shape")
	}
	tails := make([]int, m)
	sums := make([]big.Int, m)
	for r := range tails {
		tails[r] = r
	}
	for i := 0; i < slots; i++ {
		row := inputs[3+d+2*i]
		if !row.IsInt64() || row.Sign() < 0 || row.Int64() >= int64(m) {
			return fmt.Errorf("sparse routing row out of bounds")
		}
		r := int(row.Int64())
		outputs[slots+i].SetInt64(int64(tails[r]))
		contribution := new(big.Int).Mul(inputs[3+i/q], inputs[3+d+2*i+1])
		sums[r].Add(&sums[r], contribution)
		sums[r].Mod(&sums[r], field)
		outputs[i].Set(&sums[r])
		tails[r] = m + i
	}
	for r, tail := range tails {
		outputs[2*slots+r].SetInt64(int64(tail))
	}
	return nil
}

func (c *Circuit) lookupProject(api frontend.API, x []frontend.Variable) []frontend.Variable {
	p := c.P
	slots := p.D * p.Q
	nodes := p.M + slots
	inputs := []frontend.Variable{p.M, p.D, p.Q}
	inputs = append(inputs, x...)
	for t := 0; t < p.D; t++ {
		for l := 0; l < p.Q; l++ {
			inputs = append(inputs, c.SparseRows[t][l], c.SparseSigns[t][l])
		}
	}
	hints, err := api.NewHint(sparseRouteHint, 2*slots+p.M, inputs...)
	if err != nil {
		panic(err)
	}
	rowTable := logderivlookup.New(api)
	accumulatorTable := logderivlookup.New(api)
	for r := 0; r < p.M; r++ {
		rowTable.Insert(r)
		accumulatorTable.Insert(0)
	}
	for i := 0; i < slots; i++ {
		rowTable.Insert(c.SparseRows[i/p.Q][i%p.Q])
		accumulatorTable.Insert(hints[i])
	}
	rc := rangecheck.New(api)
	comparator := cmp.NewBoundedComparator(api, big.NewInt(int64(nodes)), false)
	consumed := make([]frontend.Variable, 0, nodes)
	for i := 0; i < slots; i++ {
		previous := hints[slots+i]
		rangeCheckUnsigned(api, rc, previous, big.NewInt(int64(nodes-1)), p.NoRangeBatch)
		api.AssertIsEqual(comparator.IsLess(previous, p.M+i), 1)
		row := rowTable.Lookup(previous)[0]
		before := accumulatorTable.Lookup(previous)[0]
		api.AssertIsEqual(row, c.SparseRows[i/p.Q][i%p.Q])
		api.AssertIsEqual(hints[i], api.Add(before, api.Mul(x[i/p.Q], c.SparseSigns[i/p.Q][i%p.Q])))
		consumed = append(consumed, previous)
	}
	out := make([]frontend.Variable, p.M)
	for r := 0; r < p.M; r++ {
		last := hints[2*slots+r]
		rangeCheckUnsigned(api, rc, last, big.NewInt(int64(nodes-1)), p.NoRangeBatch)
		api.AssertIsEqual(rowTable.Lookup(last)[0], r)
		out[r] = accumulatorTable.Lookup(last)[0]
		consumed = append(consumed, last)
	}
	// Equal characteristic polynomials of consumed indices and [0,nodes)
	// establish a permutation. Failure <= nodes/|Fr| for uniform challenge,
	// plus composed lookup/commitment soundness. Challenge follows binding.
	// Register during the deferred pass, alongside the lookup arguments.
	// Registering immediately would close multicommit before later tables.
	api.Compiler().Defer(func(api frontend.API) error {
		multicommit.WithCommitment(api, func(api frontend.API, challenge frontend.Variable) error {
			left, right := frontend.Variable(1), frontend.Variable(1)
			for i, index := range consumed {
				left = api.Mul(left, api.Sub(challenge, index))
				right = api.Mul(right, api.Sub(challenge, i))
			}
			api.AssertIsEqual(left, right)
			return nil
		}, consumed...)
		return nil
	})
	return out
}
