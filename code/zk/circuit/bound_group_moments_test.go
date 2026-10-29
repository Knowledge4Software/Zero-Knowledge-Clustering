package circuit

import (
	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/test"
	"math/big"
	"testing"
)

func groupFixture(t *testing.T) (*BoundPrepared, *GroupMoments, [][]int64) {
	m := BoundModel{Version: BoundVersion, Method: "learned_energy", N: 4, D: 4, K: 2, InputBits: 4, BasisScale: 8, InputScale: "1", Centers: [][]int64{{0, 0, 0, 0}, {8, 0, 0, 0}}, Basis: [][]int64{{8, 0, 0, 0}}}
	rows := [][]int64{{0, 0, 0, 0}, {4, 1, 0, 0}, {8, 0, 0, 0}, {7, 1, 0, 0}}
	p, err := PrepareBound(m, rows, bi(7), bi(9))
	if err != nil {
		t.Fatal(err)
	}
	g, err := PrepareGroupMoments(p, bi(11))
	if err != nil {
		t.Fatal(err)
	}
	return p, g, [][]int64{{0, 0, 0, 0}, {6, 0, 0, 0}}
}
func TestGroupBindingAndCorrections(t *testing.T) {
	p, g, centers := groupFixture(t)
	before := FillBoundGroupPreprocess(p, g)
	if err := test.IsSolved(NewBoundGroupPreprocess(p.M), before, ecc.BN254.ScalarField()); err != nil {
		t.Fatal(err)
	}
	ids, delta, err := GroupCorrectionIndices(p, g, centers)
	if err != nil {
		t.Fatal(err)
	}
	if len(ids) == 0 {
		t.Fatal("fixture must exercise exact corrections")
	}
	w, err := FillBoundGroupQuery(p, g, centers, len(ids))
	if err != nil {
		t.Fatal(err)
	}
	if err := test.IsSolved(NewBoundGroupQuery(p.M, len(ids)), w, ecc.BN254.ScalarField()); err != nil {
		t.Fatal(err)
	}
	truth := new(big.Int).Sub(exactPairedCost(p.X, centers), exactPairedCost(p.X, p.M.Centers))
	if truth.Cmp(delta) != 0 {
		t.Fatal("wrong native delta")
	}
}
func TestGroupTamperingAndCoverageRejected(t *testing.T) {
	p, g, centers := groupFixture(t)
	binding := NewBoundGroupPreprocess(p.M)
	bad := FillBoundGroupPreprocess(p, g)
	bad.Catalog.Sums[0][0] = bi(123)
	if err := test.IsSolved(binding, bad, ecc.BN254.ScalarField()); err == nil {
		t.Fatal("tampered sum accepted")
	}
	bad = FillBoundGroupPreprocess(p, g)
	bad.Catalog.Groups[0] = 1
	if err := test.IsSolved(binding, bad, ecc.BN254.ScalarField()); err == nil {
		t.Fatal("changed membership accepted")
	}
	bad = FillBoundGroupPreprocess(p, g)
	bad.Rows[1] = bad.Rows[0]
	if err := test.IsSolved(binding, bad, ecc.BN254.ScalarField()); err == nil {
		t.Fatal("duplicated row accepted")
	}
	ids, _, _ := GroupCorrectionIndices(p, g, centers)
	shape := NewBoundGroupQuery(p.M, len(ids))
	fresh := func() *BoundGroupQuery {
		w, e := FillBoundGroupQuery(p, g, centers, len(ids))
		if e != nil {
			t.Fatal(e)
		}
		return w
	}
	w := fresh()
	w.Change = 12345
	if err := test.IsSolved(shape, w, ecc.BN254.ScalarField()); err == nil {
		t.Fatal("false output accepted")
	}
	w = fresh()
	for i := range w.Active {
		w.Active[i] = 0
		w.Indices[i] = 0
		for d, v := range p.X[0] {
			w.X[i][d] = v
		}
		for d, v := range p.TX.Path(0) {
			w.PX[i][d] = v
		}
	}
	if err := test.IsSolved(shape, w, ecc.BN254.ScalarField()); err == nil {
		t.Fatal("omitted corrections accepted")
	}
	w = fresh()
	w.X[0][0] = 15
	if err := test.IsSolved(shape, w, ecc.BN254.ScalarField()); err == nil {
		t.Fatal("unauthenticated exact row accepted")
	}
	if len(ids) > 1 {
		w = fresh()
		w.Indices[1] = w.Indices[0]
		if err := test.IsSolved(shape, w, ecc.BN254.ScalarField()); err == nil {
			t.Fatal("duplicate correction accepted")
		}
	}
}

// Changing both the hash and sums must still fail original-data binding.
func TestSelfConsistentFalseGroupCatalogRejected(t *testing.T) {
	p, g, _ := groupFixture(t)
	g.Sums[0][0] = add(g.Sums[0][0], bi(1))
	values := []*big.Int{bi(boundGroupDomain), p.M.Hash(), p.CX, p.CH, g.Salt}
	for _, v := range g.Groups {
		values = append(values, bi(int64(v)))
	}
	for j := range g.Sums {
		values = append(values, g.Counts[j])
		values = append(values, g.Sums[j]...)
	}
	g.Commitment = boundNativeHash(values...)
	if err := test.IsSolved(NewBoundGroupPreprocess(p.M), FillBoundGroupPreprocess(p, g), ecc.BN254.ScalarField()); err == nil {
		t.Fatal("self-consistent false moment catalog accepted")
	}
}

func TestDuplicateActiveCorrectionRejected(t *testing.T) {
	p, g, centers := groupFixture(t)
	w, err := FillBoundGroupQuery(p, g, centers, 2)
	if err != nil {
		t.Fatal(err)
	}
	w.Active[1] = 1
	w.Indices[1] = w.Indices[0]
	w.X[1] = append([]frontend.Variable{}, w.X[0]...)
	w.PX[1] = append([]frontend.Variable{}, w.PX[0]...)
	if err := test.IsSolved(NewBoundGroupQuery(p.M, 2), w, ecc.BN254.ScalarField()); err == nil {
		t.Fatal("duplicate active corrections accepted")
	}
}
