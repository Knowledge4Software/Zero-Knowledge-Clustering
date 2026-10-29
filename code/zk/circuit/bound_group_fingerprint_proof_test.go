package circuit

import (
	"bytes"
	"math/big"
	"testing"
	"time"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark/backend/groth16"
	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/frontend/cs/r1cs"
)

func TestGroupFingerprintGroth16PublicReplay(t *testing.T) {
	p, g, f := fingerprintFixture(t)
	header := GroupFingerprintHeader{f.Commitment.Membership.Root(), f.Commitment.StatsRoot, f.Challenge}
	started := time.Now()
	rowCS, err := frontend.Compile(ecc.BN254.ScalarField(), r1cs.NewBuilder, NewBoundGroupFingerprintRow(p.M))
	if err != nil {
		t.Fatal(err)
	}
	rowPK, rowVK, err := groth16.Setup(rowCS)
	if err != nil {
		t.Fatal(err)
	}
	rows := make([]GroupFingerprintPublicRow, p.M.N)
	for i := range rows {
		rows[i] = GroupFingerprintPublicRow{i, f.MaskHash[i], f.MaskHash[i+1], f.Masked[i]}
		full, err := frontend.NewWitness(FillBoundGroupFingerprintRow(p, g, f, i), ecc.BN254.ScalarField())
		if err != nil {
			t.Fatal(err)
		}
		proof, err := groth16.Prove(rowCS, rowPK, full)
		if err != nil {
			t.Fatal(err)
		}
		var encoded bytes.Buffer
		if _, err = proof.WriteTo(&encoded); err != nil {
			t.Fatal(err)
		}
		replayed := groth16.NewProof(ecc.BN254)
		if _, err = replayed.ReadFrom(&encoded); err != nil {
			t.Fatal(err)
		}
		public, err := GroupFingerprintRowPublic(p.M, p.Statement(), header, rows[i])
		if err != nil {
			t.Fatal(err)
		}
		witness, err := frontend.NewWitness(public, ecc.BN254.ScalarField(), frontend.PublicOnly())
		if err != nil {
			t.Fatal(err)
		}
		if err = groth16.Verify(replayed, rowVK, witness); err != nil {
			t.Fatal(err)
		}
		if i == 0 {
			public.Masked[0] = add(f.Masked[0][0], bi(1))
			bad, err := frontend.NewWitness(public, ecc.BN254.ScalarField(), frontend.PublicOnly())
			if err != nil {
				t.Fatal(err)
			}
			if groth16.Verify(replayed, rowVK, bad) == nil {
				t.Fatal("tampered public contribution verified")
			}
		}
	}
	sums, err := FoldGroupFingerprintRows(p.M, rows)
	if err != nil {
		t.Fatal(err)
	}
	linkCS, err := frontend.Compile(ecc.BN254.ScalarField(), r1cs.NewBuilder, NewBoundGroupFingerprintLink(p.M))
	if err != nil {
		t.Fatal(err)
	}
	linkPK, linkVK, err := groth16.Setup(linkCS)
	if err != nil {
		t.Fatal(err)
	}
	link, err := FillBoundGroupFingerprintLink(p, g, f)
	if err != nil {
		t.Fatal(err)
	}
	full, err := frontend.NewWitness(link, ecc.BN254.ScalarField())
	if err != nil {
		t.Fatal(err)
	}
	proof, err := groth16.Prove(linkCS, linkPK, full)
	if err != nil {
		t.Fatal(err)
	}
	public, err := GroupFingerprintLinkPublic(p.M, p.Statement(), header, sums, rows[0].Previous, rows[p.M.N-1].Next)
	if err != nil {
		t.Fatal(err)
	}
	witness, err := frontend.NewWitness(public, ecc.BN254.ScalarField(), frontend.PublicOnly())
	if err != nil {
		t.Fatal(err)
	}
	if err = groth16.Verify(proof, linkVK, witness); err != nil {
		t.Fatal(err)
	}
	centers := [][]int64{{0, 0, 0, 0}, {6, 0, 0, 0}}
	moment, err := FillBoundGroupAssignedChange(p, g, f.Commitment, centers)
	if err != nil {
		t.Fatal(err)
	}
	momentCS, err := frontend.Compile(ecc.BN254.ScalarField(), r1cs.NewBuilder, NewBoundGroupAssignedChange(p.M))
	if err != nil {
		t.Fatal(err)
	}
	momentPK, momentVK, err := groth16.Setup(momentCS)
	if err != nil {
		t.Fatal(err)
	}
	full, err = frontend.NewWitness(moment, ecc.BN254.ScalarField())
	if err != nil {
		t.Fatal(err)
	}
	proof, err = groth16.Prove(momentCS, momentPK, full)
	if err != nil {
		t.Fatal(err)
	}
	momentPublic, err := GroupAssignedChangePublic(p.M, p.Statement(), header, centers, moment.Change.(*big.Int))
	if err != nil {
		t.Fatal(err)
	}
	witness, err = frontend.NewWitness(momentPublic, ecc.BN254.ScalarField(), frontend.PublicOnly())
	if err != nil {
		t.Fatal(err)
	}
	if err = groth16.Verify(proof, momentVK, witness); err != nil {
		t.Fatal(err)
	}
	params, err := NewGroupSwitchParameters(p.M, centers)
	if err != nil {
		t.Fatal(err)
	}
	batch, err := FillBoundGroupSwitchBatch(p, g, f.Commitment, params, []int{0, 1, 2, 3})
	if err != nil {
		t.Fatal(err)
	}
	batchCS, err := frontend.Compile(ecc.BN254.ScalarField(), r1cs.NewBuilder, NewBoundGroupSwitchBatch(p.M, 4))
	if err != nil {
		t.Fatal(err)
	}
	batchPK, batchVK, err := groth16.Setup(batchCS)
	if err != nil {
		t.Fatal(err)
	}
	full, err = frontend.NewWitness(batch, ecc.BN254.ScalarField())
	if err != nil {
		t.Fatal(err)
	}
	proof, err = groth16.Prove(batchCS, batchPK, full)
	if err != nil {
		t.Fatal(err)
	}
	batchPublic, err := GroupSwitchPublic(p.M, p.Statement(), header.GroupRoot, params, []int{0, 1, 2, 3}, batch.Lower.(*big.Int), batch.Upper.(*big.Int))
	if err != nil {
		t.Fatal(err)
	}
	witness, err = frontend.NewWitness(batchPublic, ecc.BN254.ScalarField(), frontend.PublicOnly())
	if err != nil {
		t.Fatal(err)
	}
	if err = groth16.Verify(proof, batchVK, witness); err != nil {
		t.Fatal(err)
	}
	t6 := new(big.Int).Exp(bi(p.M.BasisScale), bi(6), nil)
	base := mul(moment.Change.(*big.Int), t6)
	truth := mul(new(big.Int).Sub(exactPairedCost(p.X, centers), exactPairedCost(p.X, p.M.Centers)), t6)
	if add(base, batch.Lower.(*big.Int)).Cmp(truth) > 0 || add(base, batch.Upper.(*big.Int)).Cmp(truth) < 0 {
		t.Fatal("verified output interval excludes Full")
	}
	t.Logf("fresh_public_proofs=%d row_constraints=%d link_constraints=%d moment_constraints=%d batch_constraints=%d elapsed_seconds=%.3f; small synthetic audit smoke, not a deployment speed benchmark", p.M.N+3, rowCS.GetNbConstraints(), linkCS.GetNbConstraints(), momentCS.GetNbConstraints(), batchCS.GetNbConstraints(), time.Since(started).Seconds())
}
