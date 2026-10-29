package circuit

import (
	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark/test"
	"math/big"
	"testing"
)

func TestGroupRefinementAuthenticatedTightening(t *testing.T) {
	p, g, f, centers := compactFixture(t)
	w, err := FillBoundGroupRefinement(p, g, f.Commitment, centers, []int{0, 2})
	if err != nil {
		t.Fatal(err)
	}
	if err = test.IsSolved(NewBoundGroupRefinement(p.M, 2), w, ecc.BN254.ScalarField()); err != nil {
		t.Fatal(err)
	}
	params, err := NewGroupSwitchParameters(p.M, centers)
	if err != nil {
		t.Fatal(err)
	}
	b, err := FillBoundGroupSwitchBatchCompact(p, g, f.Commitment, params, []int{0, 1, 2, 3})
	if err != nil {
		t.Fatal(err)
	}
	a, err := FillBoundGroupAssignedChange(p, g, f.Commitment, centers)
	if err != nil {
		t.Fatal(err)
	}
	t6 := new(big.Int).Exp(bi(p.M.BasisScale), bi(6), nil)
	truth := mul(new(big.Int).Sub(exactPairedCost(p.X, centers), exactPairedCost(p.X, p.M.Centers)), t6)
	base := mul(a.Change.(*big.Int), t6)
	lower := add(add(base, b.Lower.(*big.Int)), w.CorrectionLower.(*big.Int))
	upper := add(add(base, b.Upper.(*big.Int)), w.CorrectionUpper.(*big.Int))
	if lower.Cmp(truth) > 0 || upper.Cmp(truth) < 0 {
		t.Fatal("refinement excludes exact Full")
	}
	if w.CorrectionLower.(*big.Int).Sign() < 0 || w.CorrectionUpper.(*big.Int).Sign() > 0 {
		t.Fatal("not a tightening")
	}
	if _, err = GroupRefinementPublic(p.M, p.Statement(), f.Commitment.Membership.Root(), centers, []int{0, 2}, w.CorrectionLower.(*big.Int), w.CorrectionUpper.(*big.Int)); err != nil {
		t.Fatal(err)
	}
	w.CorrectionLower = add(w.CorrectionLower.(*big.Int), bi(1))
	if test.IsSolved(NewBoundGroupRefinement(p.M, 2), w, ecc.BN254.ScalarField()) == nil {
		t.Fatal("forged correction accepted")
	}
}

func TestGroupRefinementRejectsSubstitutedRowsAndNoncanonicalRadius(t *testing.T) {
	p, g, f, centers := compactFixture(t)
	fresh := func() *BoundGroupRefinement {
		w, e := FillBoundGroupRefinement(p, g, f.Commitment, centers, []int{0})
		if e != nil {
			t.Fatal(e)
		}
		return w
	}
	w := fresh()
	w.X[0][0] = p.X[0][0] + 1
	if test.IsSolved(NewBoundGroupRefinement(p.M, 1), w, ecc.BN254.ScalarField()) == nil {
		t.Fatal("substituted original row accepted")
	}
	w = fresh()
	w.Radius[0] = add(w.Radius[0].(*big.Int), bi(1))
	if test.IsSolved(NewBoundGroupRefinement(p.M, 1), w, ecc.BN254.ScalarField()) == nil {
		t.Fatal("nonminimal refinement radius accepted")
	}
	w = fresh()
	w.Group[0] = (g.Groups[0] + 1) % p.M.K
	if test.IsSolved(NewBoundGroupRefinement(p.M, 1), w, ecc.BN254.ScalarField()) == nil {
		t.Fatal("changed authenticated group accepted")
	}
	w = fresh()
	w.Indices[0] = 1
	if test.IsSolved(NewBoundGroupRefinement(p.M, 1), w, ecc.BN254.ScalarField()) == nil {
		t.Fatal("wrong index accepted")
	}
	if _, err := GroupRefinementPublic(p.M, p.Statement(), f.Commitment.Membership.Root(), centers, []int{0}, bi(-1), bi(0)); err == nil {
		t.Fatal("negative lower tightening accepted")
	}
}
