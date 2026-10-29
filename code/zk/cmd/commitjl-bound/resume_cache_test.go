package main

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark/backend/groth16"
	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/frontend/cs/r1cs"
)

type cacheSquare struct {
	X frontend.Variable
	Y frontend.Variable `gnark:",public"`
}

func (c *cacheSquare) Define(api frontend.API) error {
	api.AssertIsEqual(api.Mul(c.X, c.X), c.Y)
	return nil
}

func TestSetupCachePinsAndCorruption(t *testing.T) {
	previous := folder
	folder = t.TempDir()
	defer func() { folder = previous }()
	t.Setenv("COMMITJL_RESUME_PROOFS", "1")
	if err := save("trusted.json", Trust{ModelHash: "cache-test", Keys: map[string]string{}}); err != nil {
		t.Fatal(err)
	}
	cs, err := frontend.Compile(ecc.BN254.ScalarField(), r1cs.NewBuilder, &cacheSquare{})
	if err != nil {
		t.Fatal(err)
	}
	pk, vk, err := groth16.Setup(cs)
	if err != nil {
		t.Fatal(err)
	}
	if _, err = saveBinary("keys/square.vk", vk); err != nil {
		t.Fatal(err)
	}
	keySHA, err := digest(filepath.Join(folder, "keys/square.vk"))
	if err != nil {
		t.Fatal(err)
	}
	if err = save("trusted.json", Trust{ModelHash: "cache-test", Keys: map[string]string{"square": keySHA}}); err != nil {
		t.Fatal(err)
	}
	original := Result{Constraints: cs.GetNbConstraints()}
	if err = saveSetupCache("square", cs, pk, &original); err != nil {
		t.Fatal(err)
	}
	var measured Result
	loadedCS, loadedPK, loadedVK, hit, err := loadSetupCache("square", &measured)
	if err != nil || !hit || !measured.ResumedSetup {
		t.Fatalf("cache failed: %v", err)
	}
	var before, after bytes.Buffer
	if _, err = pk.WriteTo(&before); err != nil {
		t.Fatal(err)
	}
	if _, err = loadedPK.WriteTo(&after); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(before.Bytes(), after.Bytes()) {
		t.Fatal("buffered cache altered key")
	}
	if original.ProvingKeyBytes != int64(before.Len()) || measured.ProvingKeyBytes != original.ProvingKeyBytes {
		t.Fatal("serialized key accounting differs from the canonical bytes")
	}
	w, err := frontend.NewWitness(&cacheSquare{X: 7, Y: 49}, ecc.BN254.ScalarField())
	if err != nil {
		t.Fatal(err)
	}
	proof, err := groth16.Prove(loadedCS, loadedPK, w)
	if err != nil {
		t.Fatal(err)
	}
	pub, err := w.Public()
	if err != nil {
		t.Fatal(err)
	}
	if err = groth16.Verify(proof, loadedVK, pub); err != nil {
		t.Fatal(err)
	}
	if err = os.WriteFile(filepath.Join(folder, ".resume/square.pk"), []byte("corrupt"), 0600); err != nil {
		t.Fatal(err)
	}
	if _, _, _, _, err = loadSetupCache("square", &measured); err == nil {
		t.Fatal("corrupt key cache accepted")
	}
}
