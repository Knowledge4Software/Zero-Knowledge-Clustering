package circuit

import (
	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark/test"
	"testing"
)

func fingerprintFixture(t *testing.T) (*BoundPrepared, *GroupMoments, *GroupFingerprintWitness) {
	p, g, _ := groupFixture(t)
	c, err := PrepareGroupFingerprintCommitment(p, g, bi(43))
	if err != nil {
		t.Fatal(err)
	}
	f, err := FillGroupFingerprintWitness(p, g, c, bi(37))
	if err != nil {
		t.Fatal(err)
	}
	return p, g, f
}

func TestGroupFingerprintHonestRowsAndLink(t *testing.T) {
	p, g, f := fingerprintFixture(t)
	for i := 0; i < p.M.N; i++ {
		if err := test.IsSolved(NewBoundGroupFingerprintRow(p.M), FillBoundGroupFingerprintRow(p, g, f, i), ecc.BN254.ScalarField()); err != nil {
			t.Fatal(err)
		}
	}
	link, err := FillBoundGroupFingerprintLink(p, g, f)
	if err != nil {
		t.Fatal(err)
	}
	if err = test.IsSolved(NewBoundGroupFingerprintLink(p.M), link, ecc.BN254.ScalarField()); err != nil {
		t.Fatal(err)
	}
}

func TestGroupFingerprintRejectsSelfConsistentFalseMoments(t *testing.T) {
	p, g, _ := fingerprintFixture(t)
	g.Sums[0][0] = add(g.Sums[0][0], bi(1))
	commit, err := PrepareGroupFingerprintCommitment(p, g, bi(43))
	if err != nil {
		t.Fatal(err)
	}
	f, err := FillGroupFingerprintWitness(p, g, commit, bi(37))
	if err != nil {
		t.Fatal(err)
	}
	link, err := FillBoundGroupFingerprintLink(p, g, f)
	if err != nil {
		t.Fatal(err)
	}
	if err = test.IsSolved(NewBoundGroupFingerprintLink(p.M), link, ecc.BN254.ScalarField()); err == nil {
		t.Fatal("self-consistent false moment commitment accepted")
	}
}

func TestGroupFingerprintTamperingAndCoverage(t *testing.T) {
	p, g, f := fingerprintFixture(t)
	row := FillBoundGroupFingerprintRow(p, g, f, 0)
	row.Masked[0] = add(f.Masked[0][0], bi(1))
	if test.IsSolved(NewBoundGroupFingerprintRow(p.M), row, ecc.BN254.ScalarField()) == nil {
		t.Fatal("false public masked contribution accepted")
	}
	row = FillBoundGroupFingerprintRow(p, g, f, 0)
	row.Group = 1 - g.Groups[0]
	if test.IsSolved(NewBoundGroupFingerprintRow(p.M), row, ecc.BN254.ScalarField()) == nil {
		t.Fatal("changed private membership accepted")
	}
	row = FillBoundGroupFingerprintRow(p, g, f, 0)
	row.Challenge = 0
	if test.IsSolved(NewBoundGroupFingerprintRow(p.M), row, ecc.BN254.ScalarField()) == nil {
		t.Fatal("zero auditor challenge accepted")
	}
	rows := make([]GroupFingerprintPublicRow, p.M.N)
	for i := range rows {
		rows[i] = GroupFingerprintPublicRow{i, f.MaskHash[i], f.MaskHash[i+1], f.Masked[i]}
	}
	if _, err := FoldGroupFingerprintRows(p.M, rows[:p.M.N-1]); err == nil {
		t.Fatal("omitted row accepted")
	}
	rows[1].Index = 0
	if _, err := FoldGroupFingerprintRows(p.M, rows); err == nil {
		t.Fatal("duplicate row accepted")
	}
	rows[1].Index = 1
	rows[1].Previous = bi(12345)
	if _, err := FoldGroupFingerprintRows(p.M, rows); err == nil {
		t.Fatal("broken mask chain accepted")
	}
}

// A known challenge allows a false polynomial with that challenge as a root.
// This deliberately passing forged link demonstrates why the separate auditor
// commit-then-challenge step is a REQUIRED security condition, not metadata.
func TestGroupFingerprintRequiresCommitBeforeChallenge(t *testing.T) {
	p, g, _ := fingerprintFixture(t)
	g.Sums[0][0] = add(g.Sums[0][0], bi(37))
	g.Sums[0][1] = add(g.Sums[0][1], bi(-1))
	commit, err := PrepareGroupFingerprintCommitment(p, g, bi(43))
	if err != nil {
		t.Fatal(err)
	}
	f, err := FillGroupFingerprintWitness(p, g, commit, bi(37))
	if err != nil {
		t.Fatal(err)
	}
	link, err := FillBoundGroupFingerprintLink(p, g, f)
	if err != nil {
		t.Fatal(err)
	}
	if err = test.IsSolved(NewBoundGroupFingerprintLink(p.M), link, ecc.BN254.ScalarField()); err != nil {
		t.Fatal("known-challenge collision fixture no longer demonstrates the required timing assumption", err)
	}
}
