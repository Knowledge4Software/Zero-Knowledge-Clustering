package main

// Opt-in local caches for long laptop experiments. Public exports never copy
// this directory. Every reuse checks the binary/model/key and public witness,
// then freshly verifies the saved proof. Default benchmark behavior is unchanged.
import (
	"bufio"
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark/backend/groth16"
	"github.com/consensys/gnark/backend/witness"
	"github.com/consensys/gnark/constraint"
	"os"
	"path/filepath"
)

func resumeEnabled() bool { return os.Getenv("COMMITJL_RESUME_PROOFS") == "1" }

type setupCache struct {
	ModelHash, BinarySHA, CCSSHA, PKSHA, VKSHA string
	Measured                                   Result
}

func loadSetupCache(key string, res *Result) (constraint.ConstraintSystem, groth16.ProvingKey, groth16.VerifyingKey, bool, error) {
	path := filepath.Join(folder, ".resume", key+".json")
	if _, err := os.Stat(path); os.IsNotExist(err) {
		return nil, nil, nil, false, nil
	}
	var record setupCache
	if err := load(path, &record); err != nil {
		return nil, nil, nil, false, err
	}
	var trust Trust
	if err := load(filepath.Join(folder, "trusted.json"), &trust); err != nil {
		return nil, nil, nil, false, err
	}
	exe, err := os.Executable()
	if err != nil {
		return nil, nil, nil, false, err
	}
	sha, err := digest(exe)
	if err != nil {
		return nil, nil, nil, false, err
	}
	if record.BinarySHA != sha || record.ModelHash != trust.ModelHash || record.VKSHA != trust.Keys[key] {
		return nil, nil, nil, false, fmt.Errorf("resume binary/model/key pin mismatch")
	}
	paths := []string{filepath.Join(folder, ".resume", key+".r1cs"), filepath.Join(folder, ".resume", key+".pk"), filepath.Join(folder, "keys", key+".vk")}
	for i, expected := range []string{record.CCSSHA, record.PKSHA, record.VKSHA} {
		s, e := digest(paths[i])
		if e != nil || s != expected {
			return nil, nil, nil, false, fmt.Errorf("resume file hash differs: %s", paths[i])
		}
	}
	cs := groth16.NewCS(ecc.BN254)
	pk := groth16.NewProvingKey(ecc.BN254)
	vk := groth16.NewVerifyingKey(ecc.BN254)
	f, e := os.Open(paths[0])
	if e != nil {
		return nil, nil, nil, false, e
	}
	_, e = cs.ReadFrom(bufio.NewReaderSize(f, binaryIOBufferSize))
	f.Close()
	if e != nil {
		return nil, nil, nil, false, e
	}
	f, e = os.Open(paths[1])
	if e != nil {
		return nil, nil, nil, false, e
	}
	_, e = pk.ReadFrom(bufio.NewReaderSize(f, binaryIOBufferSize))
	f.Close()
	if e != nil {
		return nil, nil, nil, false, e
	}
	f, e = os.Open(paths[2])
	if e != nil {
		return nil, nil, nil, false, e
	}
	_, e = vk.ReadFrom(bufio.NewReaderSize(f, binaryIOBufferSize))
	f.Close()
	if e != nil {
		return nil, nil, nil, false, e
	}
	res.Compile, res.Setup, res.Constraints = record.Measured.Compile, record.Measured.Setup, record.Measured.Constraints
	res.ProvingKeyBytes, res.VerifyingKeyBytes = record.Measured.ProvingKeyBytes, record.Measured.VerifyingKeyBytes
	res.ResumedSetup = true
	return cs, pk, vk, true, nil
}
func saveSetupCache(key string, cs constraint.ConstraintSystem, pk groth16.ProvingKey, res *Result) error {
	var trust Trust
	if err := load(filepath.Join(folder, "trusted.json"), &trust); err != nil {
		return err
	}
	exe, err := os.Executable()
	if err != nil {
		return err
	}
	bsha, err := digest(exe)
	if err != nil {
		return err
	}
	if _, err = saveBinary(".resume/"+key+".r1cs", cs); err != nil {
		return err
	}
	if res.ProvingKeyBytes, err = saveBinary(".resume/"+key+".pk", pk); err != nil {
		return err
	}
	csha, err := digest(filepath.Join(folder, ".resume", key+".r1cs"))
	if err != nil {
		return err
	}
	psha, err := digest(filepath.Join(folder, ".resume", key+".pk"))
	if err != nil {
		return err
	}
	return save(".resume/"+key+".json", setupCache{trust.ModelHash, bsha, csha, psha, trust.Keys[key], *res})
}

type sampleCache struct {
	PublicSHA, KeySHA string
	Measured          Sample
}

func sampleIdentity(pub witness.Witness, vk groth16.VerifyingKey) (string, string, error) {
	var a, b bytes.Buffer
	if _, e := pub.WriteTo(&a); e != nil {
		return "", "", e
	}
	if _, e := vk.WriteTo(&b); e != nil {
		return "", "", e
	}
	x, y := sha256.Sum256(a.Bytes()), sha256.Sum256(b.Bytes())
	return hex.EncodeToString(x[:]), hex.EncodeToString(y[:]), nil
}
func reuseSample(file string, pub witness.Witness, vk groth16.VerifyingKey) (Sample, bool, error) {
	var meta sampleCache
	path := filepath.Join(folder, file+".sample-cache.json")
	if _, e := os.Stat(path); os.IsNotExist(e) {
		return Sample{}, false, nil
	}
	if e := load(path, &meta); e != nil {
		return Sample{}, false, e
	}
	p, k, e := sampleIdentity(pub, vk)
	if e != nil {
		return Sample{}, false, e
	}
	if p != meta.PublicSHA || k != meta.KeySHA {
		return Sample{}, false, fmt.Errorf("cached witness/key differs")
	}
	proof := groth16.NewProof(ecc.BN254)
	f, e := os.Open(filepath.Join(folder, file))
	if e != nil {
		return Sample{}, false, e
	}
	_, e = proof.ReadFrom(bufio.NewReaderSize(f, binaryIOBufferSize))
	f.Close()
	if e != nil {
		return Sample{}, false, e
	}
	if e = groth16.Verify(proof, vk, pub); e != nil {
		return Sample{}, false, e
	}
	return meta.Measured, true, nil
}
func saveSampleCache(file string, s Sample, pub witness.Witness, vk groth16.VerifyingKey) error {
	p, k, e := sampleIdentity(pub, vk)
	if e != nil {
		return e
	}
	return save(file+".sample-cache.json", sampleCache{p, k, s})
}
