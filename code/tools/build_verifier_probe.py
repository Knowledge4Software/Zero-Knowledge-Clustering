"""Build a Linux prover instrumented only to export verifier artifacts.

The original prover sources/binary and circuit relation are unchanged. Generated
sources live in ignored .cache. Run from any directory; requires local Go cache.
"""
from pathlib import Path
import os
import subprocess

CODE=Path(__file__).resolve().parents[1]
OUT=CODE/'.cache/verifier_probe'
OUT.mkdir(parents=True,exist_ok=True)
text=(CODE/'zk/cmd/commitjl-prove/main.go').read_text()
assert text.count('res.ProofBytes = buf.Len()')==2
text=text.replace('"encoding/json"','"encoding/json"\n "io"\n "path/filepath"\n backendwitness "github.com/consensys/gnark/backend/witness"')
text=text.replace('res.ProofBytes = buf.Len()', '''res.ProofBytes = buf.Len()
            if rep == warmup+repetitions-1 {
                if err := exportVerifierArtifacts(inst.Backend, proof, vk, publicWitness); err != nil {
                    res.Error = "export verifier artifacts: " + err.Error(); return res
                }
            }''')
text=text.replace('logger.Disable() // keep stdout reserved for our JSON result','''logger.Disable() // keep stdout reserved for our JSON result
    if directory := os.Getenv("COMMITJL_VERIFY_ONLY"); directory != "" {
        if err := verifyOnly(directory); err != nil { fmt.Fprintln(os.Stderr, err); os.Exit(2) }
        return
    }''')
text+=r'''

func exportVerifierArtifacts(backend string, proof, vk, public io.WriterTo) error {
    directory := os.Getenv("COMMITJL_VERIFIER_EXPORT_DIR")
    if directory == "" { return nil }
    if err := os.MkdirAll(directory, 0700); err != nil { return err }
    for name, value := range map[string]io.WriterTo{"proof.bin":proof,"vk.bin":vk,"public.bin":public} {
        f, err := os.Create(filepath.Join(directory,name)); if err != nil { return err }
        _, err = value.WriteTo(f); closeErr := f.Close()
        if err != nil { return err }; if closeErr != nil { return closeErr }
    }
    return os.WriteFile(filepath.Join(directory,"backend.txt"), []byte(backend),0600)
}

func readVerifierFile(directory, name string, value io.ReaderFrom) error {
    f, err := os.Open(filepath.Join(directory,name)); if err != nil { return err }
    defer f.Close(); _, err = value.ReadFrom(f); return err
}

func verifyOnly(directory string) error {
    start := time.Now()
    backend, err := os.ReadFile(filepath.Join(directory,"backend.txt")); if err != nil { return err }
    public, err := backendwitness.New(ecc.BN254.ScalarField()); if err != nil { return err }
    if err = readVerifierFile(directory,"public.bin",public); err != nil { return err }
    samples := make([]float64,0,10)
    var loadSeconds float64
    if string(backend)=="plonk" {
        proof := backendplonk.NewProof(ecc.BN254); vk := backendplonk.NewVerifyingKey(ecc.BN254)
        if err = readVerifierFile(directory,"proof.bin",proof); err != nil { return err }
        if err = readVerifierFile(directory,"vk.bin",vk); err != nil { return err }
        loadSeconds = time.Since(start).Seconds()
        for i:=0; i<11; i++ { t:=time.Now(); if err=backendplonk.Verify(proof,vk,public); err!=nil{return err}; if i>0{samples=append(samples,time.Since(t).Seconds())} }
    } else if string(backend)=="groth16" {
        proof := backendgroth16.NewProof(ecc.BN254); vk := backendgroth16.NewVerifyingKey(ecc.BN254)
        if err = readVerifierFile(directory,"proof.bin",proof); err != nil { return err }
        if err = readVerifierFile(directory,"vk.bin",vk); err != nil { return err }
        loadSeconds = time.Since(start).Seconds()
        for i:=0; i<11; i++ { t:=time.Now(); if err=backendgroth16.Verify(proof,vk,public); err!=nil{return err}; if i>0{samples=append(samples,time.Since(t).Seconds())} }
    } else { return fmt.Errorf("unsupported backend: %s",backend) }
    value:=map[string]interface{}{"backend":string(backend),"verify_ok":true,"successful_verifications":10,
        "warmup":1,"verify_seconds_samples":samples,"deserialize_seconds":loadSeconds,
        "total_seconds":time.Since(start).Seconds(),"peak_process_rss_bytes":getMaxRSS()*1024,
        "memory_scope":"isolated verifier process including runtime and artifact loading; no compile/setup/prove"}
    return json.NewEncoder(os.Stdout).Encode(value)
}
'''
(OUT/'main.go').write_text(text)
(OUT/'rss_unix.go').write_text((CODE/'zk/cmd/commitjl-prove/rss_unix.go').read_text())
env={**os.environ,'GOOS':'linux','GOARCH':'amd64','CGO_ENABLED':'0'}
subprocess.run(['go','build','-o',str(OUT/'commitjl-verifier-probe'),str(OUT/'main.go'),str(OUT/'rss_unix.go')],cwd=CODE/'zk',env=env,check=True)
print(OUT/'commitjl-verifier-probe')
