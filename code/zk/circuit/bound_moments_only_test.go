package circuit

import (
	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark/test"
	"math/big"
	"testing"
)

func momentsFixture(t *testing.T) (*BoundPrepared, *GroupMoments, *GroupFingerprintWitness, [][]int64) {
	t.Helper()
	base, _, newCenters := groupFixture(t)
	m := base.M
	m.Method = "affine"
	m.Basis = nil
	p, err := PrepareMomentsOnly(m, base.X, bi(11), bi(13))
	if err != nil {
		t.Fatal(err)
	}
	g, err := PrepareExactOldGroupMoments(p, p.SaltZ)
	if err != nil {
		t.Fatal(err)
	}
	commit, err := PrepareGroupFingerprintCommitment(p, g, p.SaltZ)
	if err != nil {
		t.Fatal(err)
	}
	f, err := FillGroupFingerprintWitness(p, g, commit, bi(37))
	if err != nil {
		t.Fatal(err)
	}
	return p, g, f, newCenters
}
func TestMomentsOnlyAuthenticatedRowsAndBestTargetUpper(t *testing.T) {
	p, g, f, centers := momentsFixture(t)
	if len(p.H) != 0 || len(p.E) != 0 || p.TH != nil || p.CH.Sign() != 0 {
		t.Fatal("baseline computed compact statistics")
	}
	for i := range p.X {
		if err := test.IsSolved(NewBoundMomentsRow(p.M), FillBoundMomentsRow(p, g, f, i), ecc.BN254.ScalarField()); err != nil {
			t.Fatal(err)
		}
	}
	w, err := FillBoundMomentsUpper(p, g, f, centers)
	if err != nil {
		t.Fatal(err)
	}
	if err = test.IsSolved(NewBoundMomentsUpper(p.M), w, ecc.BN254.ScalarField()); err != nil {
		t.Fatal(err)
	}
	truth := new(big.Int).Sub(exactPairedCost(p.X, centers), exactPairedCost(p.X, p.M.Centers))
	if truth.Cmp(w.Upper.(*big.Int)) > 0 {
		t.Fatal("invalid upper")
	}
	w.Upper = add(w.Upper.(*big.Int), bi(-1))
	if test.IsSolved(NewBoundMomentsUpper(p.M), w, ecc.BN254.ScalarField()) == nil {
		t.Fatal("forged upper accepted")
	}
	w, err = FillBoundMomentsUpper(p, g, f, centers)
	if err != nil {
		t.Fatal(err)
	}
	w.Targets[0] = p.M.K
	if test.IsSolved(NewBoundMomentsUpper(p.M), w, ecc.BN254.ScalarField()) == nil {
		t.Fatal("out-of-range target accepted")
	}
}
func TestMomentsOnlyRejectsAuthenticatedNonNearestGroup(t *testing.T) {
	p, g, _, _ := momentsFixture(t)
	labels := append([]int{}, g.Groups...)
	labels[0] = (labels[0] + 1) % p.M.K
	bad, err := PrepareGroupMomentsForLabels(p, p.SaltZ, labels)
	if err != nil {
		t.Fatal(err)
	}
	commit, err := PrepareGroupFingerprintCommitment(p, bad, p.SaltZ)
	if err != nil {
		t.Fatal(err)
	}
	f, err := FillGroupFingerprintWitness(p, bad, commit, bi(37))
	if err != nil {
		t.Fatal(err)
	}
	if test.IsSolved(NewBoundMomentsRow(p.M), FillBoundMomentsRow(p, bad, f, 0), ecc.BN254.ScalarField()) == nil {
		t.Fatal("authenticated nonnearest old group accepted")
	}
}
