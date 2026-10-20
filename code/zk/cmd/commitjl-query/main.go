// commitjl-query benchmarks the Stage-2 (per-query) relation over a committed
// compressed matrix: compile, optional Groth16 setup/prove/verify.
//
//	commitjl-query -instance inst.json [-out result.json]
//
// The instance holds integer Z (n x W) and centres (k x W); the CLI computes
// assignments, cost and commitment roots natively so the witness always
// matches the relation.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"math/big"
	"os"
	"runtime"
	"time"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark/backend/groth16"
	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/frontend/cs/r1cs"
	"github.com/consensys/gnark/logger"

	circpkg "commitjl/zk/circuit"
)

type Instance struct {
	Label        string    `json:"label"`
	N            int       `json:"n"`
	W            int       `json:"w"`
	K            int       `json:"k"`
	AssignDims   int       `json:"assign_dims"`
	CoordBits    int       `json:"coord_bits"`
	Cost         bool      `json:"cost"`
	ConstCenters bool      `json:"const_centers"`
	EnergyCol    bool      `json:"energy_col"`
	EnergyBits   int       `json:"energy_bits"`
	Z            [][]int64 `json:"z"`
	Centers      [][]int64 `json:"centers"`
	Salt         string    `json:"salt"`
	CompileOnly  bool      `json:"compile_only"`
	Repetitions  int       `json:"n_repetitions"`
	Warmup       int       `json:"n_warmup"`
}

type Result struct {
	Label          string    `json:"label"`
	NumConstraints int       `json:"num_constraints"`
	CompileSeconds float64   `json:"compile_seconds"`
	SetupSeconds   float64   `json:"setup_seconds"`
	WitnessSeconds float64   `json:"witness_seconds"`
	ProveSamples   []float64 `json:"prove_seconds_samples"`
	VerifySamples  []float64 `json:"verify_seconds_samples"`
	ProofBytes     int       `json:"proof_bytes"`
	VerifyOK       bool      `json:"verify_ok"`
	Assign         []int     `json:"assign"`
	Psi            string    `json:"psi"`
	PeakRSSBytesOS int64     `json:"peak_rss_bytes_os_maxrss"`
	Error          string    `json:"error,omitempty"`
}

type counter struct{ n int }

func (c *counter) Write(p []byte) (int, error) { c.n += len(p); return len(p), nil }

func run(inst *Instance) Result {
	res := Result{Label: inst.Label}
	p := circpkg.QueryParams{N: inst.N, W: inst.W, K: inst.K, AssignDims: inst.AssignDims,
		CoordBits: inst.CoordBits, Cost: inst.Cost, ConstCenters: inst.ConstCenters,
		EnergyCol: inst.EnergyCol, EnergyBits: inst.EnergyBits}
	if inst.ConstCenters {
		p.Centers = inst.Centers
	}
	if err := p.Validate(); err != nil {
		res.Error = err.Error()
		return res
	}
	salt, ok := new(big.Int).SetString(inst.Salt, 10)
	if !ok {
		res.Error = "invalid salt"
		return res
	}
	ans, err := circpkg.SolveQuery(p, inst.Z, inst.Centers, salt)
	if err != nil {
		res.Error = err.Error()
		return res
	}
	res.Assign, res.Psi = ans.Assign, ans.Psi.String()

	t0 := time.Now()
	ccs, err := frontend.Compile(ecc.BN254.ScalarField(), r1cs.NewBuilder, circpkg.NewQueryCircuit(p))
	res.CompileSeconds = time.Since(t0).Seconds()
	if err != nil {
		res.Error = "compile: " + err.Error()
		return res
	}
	res.NumConstraints = ccs.GetNbConstraints()
	if inst.CompileOnly {
		res.PeakRSSBytesOS = getMaxRSS()
		return res
	}
	t0 = time.Now()
	full, err := frontend.NewWitness(circpkg.FillQueryWitness(p, inst.Z, inst.Centers, salt, ans), ecc.BN254.ScalarField())
	if err != nil {
		res.Error = "witness: " + err.Error()
		return res
	}
	public, err := full.Public()
	if err != nil {
		res.Error = "public witness: " + err.Error()
		return res
	}
	res.WitnessSeconds = time.Since(t0).Seconds()
	t0 = time.Now()
	pk, vk, err := groth16.Setup(ccs)
	res.SetupSeconds = time.Since(t0).Seconds()
	if err != nil {
		res.Error = "setup: " + err.Error()
		return res
	}
	reps := inst.Repetitions
	if reps < 1 {
		reps = 1
	}
	for rep := 0; rep < inst.Warmup+reps; rep++ {
		t0 = time.Now()
		proof, err := groth16.Prove(ccs, pk, full)
		proveTime := time.Since(t0).Seconds()
		if err != nil {
			res.Error = "prove: " + err.Error()
			return res
		}
		t0 = time.Now()
		if err := groth16.Verify(proof, vk, public); err != nil {
			res.Error = "verify: " + err.Error()
			return res
		}
		verifyTime := time.Since(t0).Seconds()
		if rep >= inst.Warmup {
			res.ProveSamples = append(res.ProveSamples, proveTime)
			res.VerifySamples = append(res.VerifySamples, verifyTime)
			buf := &counter{}
			_, _ = proof.WriteTo(buf)
			res.ProofBytes = buf.n
		}
		runtime.GC()
	}
	res.VerifyOK = len(res.ProveSamples) == reps
	res.PeakRSSBytesOS = getMaxRSS()
	return res
}

func main() {
	logger.Disable()
	in := flag.String("instance", "", "instance JSON path")
	out := flag.String("out", "", "result JSON path (default stdout)")
	flag.Parse()
	data, err := os.ReadFile(*in)
	if err != nil {
		fmt.Fprintln(os.Stderr, "read instance:", err)
		os.Exit(1)
	}
	var inst Instance
	if err := json.Unmarshal(data, &inst); err != nil {
		fmt.Fprintln(os.Stderr, "parse instance:", err)
		os.Exit(1)
	}
	res := run(&inst)
	encoded, _ := json.MarshalIndent(res, "", "  ")
	if *out == "" {
		fmt.Println(string(encoded))
	} else if err := os.WriteFile(*out, encoded, 0644); err != nil {
		fmt.Fprintln(os.Stderr, "write result:", err)
		os.Exit(1)
	}
	if res.Error != "" {
		os.Exit(2)
	}
}
