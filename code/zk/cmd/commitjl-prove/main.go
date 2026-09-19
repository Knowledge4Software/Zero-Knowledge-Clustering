// commitjl-prove: reads an "instance" JSON (written by Python's
// commitjl/zk_bridge.py) describing a CommitJL circuit instance, compiles
// the circuit for the requested variant, runs setup, builds the witness,
// proves, verifies, and emits a JSON result with timings/sizes/RSS.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"math/big"
	"os"
	"runtime"
	"syscall"
	"time"

	"github.com/consensys/gnark-crypto/ecc"
	backendgroth16 "github.com/consensys/gnark/backend/groth16"
	backendplonk "github.com/consensys/gnark/backend/plonk"
	"github.com/consensys/gnark/constraint"
	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/frontend/cs/r1cs"
	"github.com/consensys/gnark/frontend/cs/scs"
	"github.com/consensys/gnark/logger"
	"github.com/consensys/gnark/test/unsafekzg"

	circpkg "commitjl/zk/circuit"
)

type Instance struct {
	N             int    `json:"n"`
	D             int    `json:"d"`
	M             int    `json:"m"`
	K             int    `json:"k"`
	Q             int    `json:"q"`
	Variant       string `json:"variant"`
	Backend       string `json:"backend"` // "groth16" | "plonk"
	NoRangeBatch  bool   `json:"no_range_batch"`
	RangeBits     int    `json:"range_bits"`

	CX     string     `json:"c_x"`
	CTheta string     `json:"c_theta"`
	CA     string     `json:"c_a"`
	PsiHat string     `json:"psi_hat"`
	Centroids [][]string `json:"centroids"` // k x d

	X  [][]string `json:"x"` // n x d
	RX string     `json:"r_x"`

	SparseRows  [][]string `json:"sparse_rows"`  // d x q
	SparseSigns [][]string `json:"sparse_signs"` // d x q
	DenseA      [][]string `json:"dense_a"`      // m x d
	ConstantA   [][]string `json:"constant_a"`   // m x d (for AConstant variants)

	Assign []int `json:"assign"`
}

type Result struct {
	Variant           string  `json:"variant"`
	Backend           string  `json:"backend"`
	NumConstraints    int     `json:"num_constraints"`
	CompileSeconds    float64 `json:"compile_seconds"`
	SetupSeconds      float64 `json:"setup_seconds"`
	WitnessSeconds    float64 `json:"witness_seconds"`
	ProveSeconds      float64 `json:"prove_seconds"`
	VerifySeconds     float64 `json:"verify_seconds"`
	ProofBytes        int     `json:"proof_bytes"`
	VerifyOK          bool    `json:"verify_ok"`
	PeakRSSBytesGoMem uint64  `json:"peak_rss_bytes_go_memstats"` // HeapSys as a Go-side proxy
	PeakRSSBytesOS    int64   `json:"peak_rss_bytes_os_maxrss"`   // syscall.Getrusage MaxRSS
	Error             string  `json:"error,omitempty"`
}

func str2fe(s string) (*big.Int, error) {
	bi, ok := new(big.Int).SetString(s, 10)
	if !ok {
		return nil, fmt.Errorf("invalid field element string: %q", s)
	}
	return bi, nil
}

func buildParams(inst *Instance) (circpkg.Params, error) {
	variant, err := circpkg.VariantFromString(inst.Variant)
	if err != nil {
		return circpkg.Params{}, err
	}
	p := circpkg.Params{
		N: inst.N, D: inst.D, M: inst.M, K: inst.K, Q: inst.Q,
		Variant:      variant,
		NoRangeBatch: inst.NoRangeBatch,
		RangeBits:    inst.RangeBits,
	}
	switch variant {
	case circpkg.VariantSparse, circpkg.VariantIncircuitGen:
		p.SparseSelf = true
		p.AuthenticateA = true
	case circpkg.VariantDense, circpkg.VariantCoord:
		p.SparseSelf = false
		p.AuthenticateA = true
	case circpkg.VariantPCAPublic, circpkg.VariantPisot:
		p.SparseSelf = false
		p.AConstant = true
		p.AuthenticateA = false
		p.ConstantA = strMatrixToInt64(inst.ConstantA)
	case circpkg.VariantPrepublished:
		p.SparseSelf = inst.SparseRows != nil
		p.AConstant = true
		p.AuthenticateA = false
		if p.SparseSelf {
			// bake sparse list into a dense constant matrix for simplicity
			p.ConstantA = sparseToConstantDense(inst, p.M, p.D)
		} else {
			p.ConstantA = strMatrixToInt64(inst.ConstantA)
		}
	case circpkg.VariantFull:
		p.SparseSelf = false
		p.AConstant = true
		p.AuthenticateA = false
		p.ConstantA = identityMatrix(p.M, p.D)
	default:
		return circpkg.Params{}, fmt.Errorf("variant %s not fully implemented in this bridge", variant)
	}
	if p.RangeBits == 0 {
		p.RangeBits = 48
	}
	return p, nil
}

func identityMatrix(m, d int) [][]int64 {
	out := make([][]int64, m)
	for i := range out {
		out[i] = make([]int64, d)
		if i < d {
			out[i][i] = 1
		}
	}
	return out
}

func strMatrixToInt64(s [][]string) [][]int64 {
	out := make([][]int64, len(s))
	for i, row := range s {
		out[i] = make([]int64, len(row))
		for j, v := range row {
			bi, _ := new(big.Int).SetString(v, 10)
			out[i][j] = bi.Int64()
		}
	}
	return out
}

func sparseToConstantDense(inst *Instance, m, d int) [][]int64 {
	out := make([][]int64, m)
	for i := range out {
		out[i] = make([]int64, d)
	}
	for t, row := range inst.SparseRows {
		for l, rs := range row {
			r := new(big.Int)
			r.SetString(rs, 10)
			sgn := new(big.Int)
			sgn.SetString(inst.SparseSigns[t][l], 10)
			out[r.Int64()][t] = sgn.Int64()
		}
	}
	return out
}

func fillWitness(c *circpkg.Circuit, inst *Instance, p circpkg.Params) error {
	set := func(dst *frontend.Variable, s string) error {
		bi, err := str2fe(s)
		if err != nil {
			return err
		}
		*dst = frontend.Variable(bi)
		return nil
	}
	if err := set(&c.CX, inst.CX); err != nil {
		return err
	}
	if err := set(&c.CTheta, inst.CTheta); err != nil {
		return err
	}
	if err := set(&c.CA, inst.CA); err != nil {
		return err
	}
	if err := set(&c.PsiHat, inst.PsiHat); err != nil {
		return err
	}
	if err := set(&c.RX, inst.RX); err != nil {
		return err
	}
	for j := 0; j < p.K; j++ {
		for t := 0; t < p.D; t++ {
			if err := set(&c.Centroids[j][t], inst.Centroids[j][t]); err != nil {
				return err
			}
		}
	}
	for i := 0; i < p.N; i++ {
		for t := 0; t < p.D; t++ {
			if err := set(&c.X[i][t], inst.X[i][t]); err != nil {
				return err
			}
		}
	}
	if p.SparseSelf && !p.AConstant {
		for t := 0; t < p.D; t++ {
			for l := 0; l < p.Q; l++ {
				if err := set(&c.SparseRows[t][l], inst.SparseRows[t][l]); err != nil {
					return err
				}
				if err := set(&c.SparseSigns[t][l], inst.SparseSigns[t][l]); err != nil {
					return err
				}
			}
		}
	}
	if !p.SparseSelf && !p.AConstant {
		for r := 0; r < p.M; r++ {
			for t := 0; t < p.D; t++ {
				if err := set(&c.DenseA[r][t], inst.DenseA[r][t]); err != nil {
					return err
				}
			}
		}
	}
	for i := 0; i < p.N; i++ {
		c.Assign[i] = frontend.Variable(inst.Assign[i])
	}
	return nil
}

func getMaxRSS() int64 {
	var ru syscall.Rusage
	if err := syscall.Getrusage(syscall.RUSAGE_SELF, &ru); err != nil {
		return -1
	}
	// on darwin, Maxrss is in bytes; on linux, kilobytes. Report raw value;
	// Python side (psutil) is the primary cross-checked source of truth.
	return int64(ru.Maxrss)
}

func run(inst *Instance) Result {
	res := Result{Variant: inst.Variant, Backend: inst.Backend}

	p, err := buildParams(inst)
	if err != nil {
		res.Error = err.Error()
		return res
	}
	c := circpkg.NewCircuit(p)

	t0 := time.Now()
	var ccs constraint.ConstraintSystem
	if inst.Backend == "plonk" {
		ccs, err = frontend.Compile(ecc.BN254.ScalarField(), scs.NewBuilder, c)
	} else {
		ccs, err = frontend.Compile(ecc.BN254.ScalarField(), r1cs.NewBuilder, c)
	}
	res.CompileSeconds = time.Since(t0).Seconds()
	if err != nil {
		res.Error = "compile: " + err.Error()
		return res
	}
	res.NumConstraints = ccs.GetNbConstraints()

	wc := circpkg.NewCircuit(p)
	if err := fillWitness(wc, inst, p); err != nil {
		res.Error = "witness-fill: " + err.Error()
		return res
	}

	t0 = time.Now()
	fullWitness, err := frontend.NewWitness(wc, ecc.BN254.ScalarField())
	if err != nil {
		res.Error = "witness: " + err.Error()
		return res
	}
	publicWitness, err := fullWitness.Public()
	if err != nil {
		res.Error = "public-witness: " + err.Error()
		return res
	}
	res.WitnessSeconds = time.Since(t0).Seconds()

	switch inst.Backend {
	case "plonk":
		t0 = time.Now()
		srs, srsLagrange, err := unsafekzg.NewSRS(ccs)
		if err != nil {
			res.Error = "srs: " + err.Error()
			return res
		}
		pk, vk, err := backendplonk.Setup(ccs, srs, srsLagrange)
		res.SetupSeconds = time.Since(t0).Seconds()
		if err != nil {
			res.Error = "setup: " + err.Error()
			return res
		}
		t0 = time.Now()
		proof, err := backendplonk.Prove(ccs, pk, fullWitness)
		res.ProveSeconds = time.Since(t0).Seconds()
		if err != nil {
			res.Error = "prove: " + err.Error()
			return res
		}
		buf := new(bytesBuffer)
		_, _ = proof.WriteTo(buf)
		res.ProofBytes = buf.Len()
		t0 = time.Now()
		verr := backendplonk.Verify(proof, vk, publicWitness)
		res.VerifySeconds = time.Since(t0).Seconds()
		res.VerifyOK = verr == nil
		if verr != nil {
			res.Error = "verify: " + verr.Error()
		}
	default:
		t0 = time.Now()
		pk, vk, err := backendgroth16.Setup(ccs)
		res.SetupSeconds = time.Since(t0).Seconds()
		if err != nil {
			res.Error = "setup: " + err.Error()
			return res
		}
		t0 = time.Now()
		proof, err := backendgroth16.Prove(ccs, pk, fullWitness)
		res.ProveSeconds = time.Since(t0).Seconds()
		if err != nil {
			res.Error = "prove: " + err.Error()
			return res
		}
		buf := new(bytesBuffer)
		_, _ = proof.WriteTo(buf)
		res.ProofBytes = buf.Len()
		t0 = time.Now()
		verr := backendgroth16.Verify(proof, vk, publicWitness)
		res.VerifySeconds = time.Since(t0).Seconds()
		res.VerifyOK = verr == nil
		if verr != nil {
			res.Error = "verify: " + verr.Error()
		}
	}

	var ms runtime.MemStats
	runtime.ReadMemStats(&ms)
	res.PeakRSSBytesGoMem = ms.HeapSys
	res.PeakRSSBytesOS = getMaxRSS()
	return res
}

// bytesBuffer is a minimal io.Writer that just counts bytes, avoiding the
// bytes package import name clash with frontend field types.
type bytesBuffer struct{ n int }

func (b *bytesBuffer) Write(p []byte) (int, error) {
	b.n += len(p)
	return len(p), nil
}
func (b *bytesBuffer) Len() int { return b.n }

func main() {
	logger.Disable() // keep stdout reserved for our JSON result
	inFile := flag.String("instance", "", "path to instance JSON (default: stdin)")
	outFile := flag.String("out", "", "path to write result JSON (default: stdout)")
	flag.Parse()

	var data []byte
	var err error
	if *inFile == "" {
		data, err = readAll(os.Stdin)
	} else {
		data, err = os.ReadFile(*inFile)
	}
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

	out, err := json.MarshalIndent(res, "", "  ")
	if err != nil {
		fmt.Fprintln(os.Stderr, "marshal result:", err)
		os.Exit(1)
	}
	if *outFile == "" {
		fmt.Println(string(out))
	} else {
		if err := os.WriteFile(*outFile, out, 0644); err != nil {
			fmt.Fprintln(os.Stderr, "write result:", err)
			os.Exit(1)
		}
	}
	if res.Error != "" {
		os.Exit(2)
	}
}

func readAll(f *os.File) ([]byte, error) {
	buf := make([]byte, 0, 4096)
	tmp := make([]byte, 4096)
	for {
		n, err := f.Read(tmp)
		if n > 0 {
			buf = append(buf, tmp[:n]...)
		}
		if err != nil {
			if err.Error() == "EOF" {
				break
			}
			return buf, nil
		}
	}
	return buf, nil
}
