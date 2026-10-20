// commitjl-prove: reads an "instance" JSON (written by Python's
// commitjl/zk_bridge.py) describing a CommitJL circuit instance, compiles
// the circuit for the requested variant, runs setup once, builds the witness,
// proves and verifies one or more times, and emits timings/sizes/RSS as JSON.
package main

import (
	"crypto/sha256"
	"encoding/json"
	"flag"
	"fmt"
	"math/big"
	"os"
	"runtime"
	"time"

	"github.com/consensys/gnark-crypto/ecc"
	backendgroth16 "github.com/consensys/gnark/backend/groth16"
	backendplonk "github.com/consensys/gnark/backend/plonk"
	"github.com/consensys/gnark/constraint"
	"github.com/consensys/gnark/constraint/solver"
	"github.com/consensys/gnark/frontend"
	frontendcs "github.com/consensys/gnark/frontend/cs"
	"github.com/consensys/gnark/frontend/cs/r1cs"
	"github.com/consensys/gnark/frontend/cs/scs"
	"github.com/consensys/gnark/logger"
	"github.com/consensys/gnark/test/unsafekzg"

	circpkg "commitjl/zk/circuit"
)

type Instance struct {
	N                int        `json:"n"`
	D                int        `json:"d"`
	M                int        `json:"m"`
	K                int        `json:"k"`
	Q                int        `json:"q"`
	Variant          string     `json:"variant"`
	Backend          string     `json:"backend"` // "groth16" | "plonk"
	NoRangeBatch     bool       `json:"no_range_batch"`
	RangeBits        int        `json:"range_bits"`
	LegacyProjection bool       `json:"legacy_projection"`
	LegacyBounds     bool       `json:"legacy_bounds"`
	ScoreDistances   bool       `json:"score_distances"`
	LookupSparse     bool       `json:"lookup_sparse"`
	SpecializeSparse bool       `json:"specialize_sparse"`
	ProtectedBasis   [][]string `json:"protected_basis"`
	BasisScale       int64      `json:"basis_scale"`
	SolveOnly        bool       `json:"solve_only"`
	DiagnosticPrefix string     `json:"diagnostic_prefix"`

	CX        string     `json:"c_x"`
	CTheta    string     `json:"c_theta"`
	CA        string     `json:"c_a"`
	PsiHat    string     `json:"psi_hat"`
	Centroids [][]string `json:"centroids"` // k x d

	X  [][]string `json:"x"` // n x d
	RX string     `json:"r_x"`

	SparseRows  [][]string `json:"sparse_rows"`  // d x q
	SparseSigns [][]string `json:"sparse_signs"` // d x q
	DenseA      [][]string `json:"dense_a"`      // m x d
	ConstantA   [][]string `json:"constant_a"`   // m x d (for AConstant variants)

	Assign      []int `json:"assign"`
	Repetitions int   `json:"n_repetitions"`
	Warmup      int   `json:"n_warmup"`
}

type Result struct {
	Variant           string    `json:"variant"`
	Backend           string    `json:"backend"`
	NumConstraints    int       `json:"num_constraints"`
	CompileSeconds    float64   `json:"compile_seconds"`
	SetupSeconds      float64   `json:"setup_seconds"`
	WitnessSeconds    float64   `json:"witness_seconds"`
	ProveSeconds      float64   `json:"prove_seconds"`
	VerifySeconds     float64   `json:"verify_seconds"`
	ProofBytes        int       `json:"proof_bytes"`
	VerifyOK          bool      `json:"verify_ok"`
	Repetitions       int       `json:"n_repetitions"`
	SuccessfulProofs  int       `json:"successful_proofs"`
	ProveSamples      []float64 `json:"prove_seconds_samples"`
	VerifySamples     []float64 `json:"verify_seconds_samples"`
	PeakRSSBytesGoMem uint64    `json:"peak_rss_bytes_go_memstats"` // HeapSys as a Go-side proxy
	PeakRSSBytesOS    int64     `json:"peak_rss_bytes_os_maxrss"`   // zero where the OS has no portable MaxRSS API
	Error             string    `json:"error,omitempty"`
	CircuitRevision   string    `json:"circuit_revision"`
	LegacyProjection  bool      `json:"legacy_projection"`
	LegacyBounds      bool      `json:"legacy_bounds"`
	ScoreDistances    bool      `json:"score_distances"`
	LookupSparse      bool      `json:"lookup_sparse"`
	SpecializeSparse  bool      `json:"specialize_sparse"`
	ProtectedRank     int       `json:"protected_rank"`
	RoutingPattern    [][]int   `json:"routing_pattern,omitempty"`
	SolveOnly         bool      `json:"solve_only"`
	SolveOK           bool      `json:"solve_ok"`
	SolveSeconds      float64   `json:"solve_seconds"`
	DiagnosticPrefix  string    `json:"diagnostic_prefix"`
}

func str2fe(s string) (*big.Int, error) {
	bi, ok := new(big.Int).SetString(s, 10)
	if !ok {
		return nil, fmt.Errorf("invalid field element string: %q", s)
	}
	return bi, nil
}

func buildParams(inst *Instance) (circpkg.Params, error) {
	if inst.DiagnosticPrefix != "" && !inst.SolveOnly {
		return circpkg.Params{}, fmt.Errorf("diagnostic prefixes require solve_only; proofs are forbidden")
	}
	if inst.N < 1 || inst.D < 1 || inst.M < 1 || inst.K < 1 {
		return circpkg.Params{}, fmt.Errorf("dimensions must be positive")
	}
	if inst.Backend != "groth16" && inst.Backend != "plonk" {
		return circpkg.Params{}, fmt.Errorf("unsupported backend %q", inst.Backend)
	}
	if inst.Repetitions < 0 || inst.Warmup < 0 {
		return circpkg.Params{}, fmt.Errorf("invalid repetition or warmup count")
	}
	variant, err := circpkg.VariantFromString(inst.Variant)
	if err != nil {
		return circpkg.Params{}, err
	}
	p := circpkg.Params{
		N: inst.N, D: inst.D, M: inst.M, K: inst.K, Q: inst.Q,
		Variant:          variant,
		NoRangeBatch:     inst.NoRangeBatch,
		RangeBits:        inst.RangeBits,
		LegacyProjection: inst.LegacyProjection,
		LegacyBounds:     inst.LegacyBounds,
		ScoreDistances:   inst.ScoreDistances,
		LookupSparse:     inst.LookupSparse,
		BasisScale:       inst.BasisScale,
		DiagnosticPrefix: inst.DiagnosticPrefix,
	}
	switch variant {
	case circpkg.VariantSparse:
		p.SparseSelf = true
		p.AuthenticateA = true
	case circpkg.VariantDense, circpkg.VariantCoord:
		p.SparseSelf = false
		p.AuthenticateA = true
	case circpkg.VariantPCAPublic, circpkg.VariantPisot:
		p.SparseSelf = false
		p.AConstant = true
		p.AuthenticateA = false
		p.ConstantA, err = strMatrixToInt64(inst.ConstantA)
	case circpkg.VariantPrepublished:
		p.SparseSelf = false
		p.AConstant = true
		p.AuthenticateA = false
		p.ConstantA, err = strMatrixToInt64(inst.ConstantA)
	case circpkg.VariantFull:
		p.SparseSelf = false
		p.AConstant = true
		p.AuthenticateA = false
		p.ConstantA = identityMatrix(p.M, p.D)
	default:
		return circpkg.Params{}, fmt.Errorf("variant %s not fully implemented in this bridge", variant)
	}
	if err != nil {
		return circpkg.Params{}, err
	}
	if err := validateWitnessShape(inst, p); err != nil {
		return circpkg.Params{}, err
	}
	if len(inst.ProtectedBasis) > 0 {
		p.ProtectedBasis, err = strMatrixToInt64(inst.ProtectedBasis)
		if err != nil {
			return circpkg.Params{}, err
		}
	}
	if inst.SpecializeSparse {
		if !p.SparseSelf || p.AConstant {
			return circpkg.Params{}, fmt.Errorf("specialization requires witnessed sparse")
		}
		p.FixedSparseRows = make([][]int, p.D)
		for t, rows := range inst.SparseRows {
			p.FixedSparseRows[t] = make([]int, p.Q)
			for l, value := range rows {
				bi, ok := new(big.Int).SetString(value, 10)
				if !ok || !bi.IsInt64() || bi.Sign() < 0 || bi.Cmp(big.NewInt(int64(p.M))) >= 0 {
					return circpkg.Params{}, fmt.Errorf("invalid specialized row")
				}
				p.FixedSparseRows[t][l] = int(bi.Int64())
			}
		}
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

func strMatrixToInt64(s [][]string) ([][]int64, error) {
	out := make([][]int64, len(s))
	for i, row := range s {
		out[i] = make([]int64, len(row))
		for j, v := range row {
			bi, ok := new(big.Int).SetString(v, 10)
			if !ok || !bi.IsInt64() {
				return nil, fmt.Errorf("constant matrix element must be a signed int64: %q", v)
			}
			out[i][j] = bi.Int64()
		}
	}
	return out, nil
}

func validateWitnessShape(inst *Instance, p circpkg.Params) error {
	check := func(name string, matrix [][]string, rows, columns int) error {
		if len(matrix) != rows {
			return fmt.Errorf("%s requires %d rows", name, rows)
		}
		for _, row := range matrix {
			if len(row) != columns {
				return fmt.Errorf("%s requires %d columns", name, columns)
			}
		}
		return nil
	}
	if err := check("x", inst.X, p.N, p.D); err != nil {
		return err
	}
	if err := check("centroids", inst.Centroids, p.K, p.D); err != nil {
		return err
	}
	if len(inst.Assign) != p.N {
		return fmt.Errorf("assign requires N entries")
	}
	if p.SparseSelf && !p.AConstant {
		if p.Q < 1 || p.Q > p.M {
			return fmt.Errorf("Q must be in [1,M]")
		}
		if err := check("sparse_rows", inst.SparseRows, p.D, p.Q); err != nil {
			return err
		}
		if err := check("sparse_signs", inst.SparseSigns, p.D, p.Q); err != nil {
			return err
		}
	}
	if !p.SparseSelf && !p.AConstant {
		return check("dense_a", inst.DenseA, p.M, p.D)
	}
	return nil
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

func run(inst *Instance) Result {
	res := Result{Variant: inst.Variant, Backend: inst.Backend,
		CircuitRevision:  "shared-routing-derived-bounds-v1",
		LegacyProjection: inst.LegacyProjection, LegacyBounds: inst.LegacyBounds}
	res.ScoreDistances = inst.ScoreDistances
	res.LookupSparse = inst.LookupSparse
	res.SpecializeSparse = inst.SpecializeSparse
	res.ProtectedRank = len(inst.ProtectedBasis)
	res.DiagnosticPrefix = inst.DiagnosticPrefix
	if inst.ScoreDistances || inst.SpecializeSparse || inst.LookupSparse || len(inst.ProtectedBasis) > 0 {
		res.CircuitRevision = "experimental-protected-sparse-scores-v2"
	}

	p, err := buildParams(inst)
	if err != nil {
		res.Error = err.Error()
		return res
	}
	res.RoutingPattern = p.FixedSparseRows
	c := circpkg.NewCircuit(p)

	t0 := time.Now()
	var ccs constraint.ConstraintSystem
	var compileOptions []frontend.CompileOption
	if inst.DiagnosticPrefix != "" {
		compileOptions = append(compileOptions, frontend.IgnoreUnconstrainedInputs())
	}
	if inst.Backend == "plonk" {
		ccs, err = frontend.Compile(ecc.BN254.ScalarField(), scs.NewBuilder, c, compileOptions...)
	} else {
		ccs, err = frontend.Compile(ecc.BN254.ScalarField(), r1cs.NewBuilder, c, compileOptions...)
	}
	res.CompileSeconds = time.Since(t0).Seconds()
	if err != nil {
		res.Error = "compile: " + err.Error()
		return res
	}
	res.NumConstraints = ccs.GetNbConstraints()
	repetitions := inst.Repetitions
	if repetitions < 1 {
		repetitions = 1
	}
	warmup := inst.Warmup
	if warmup < 0 {
		warmup = 0
	}
	res.Repetitions = repetitions

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
	if inst.SolveOnly {
		res.SolveOnly = true
		t0 = time.Now()
		// Solver-only profiling needs a diagnostic commitment challenge. The
		// real backends replace this hint with a cryptographic commitment.
		// This override is never passed to Prove and is not a security test.
		hint := func(mod *big.Int, inputs []*big.Int, outputs []*big.Int) error {
			h := sha256.New()
			for _, value := range inputs {
				h.Write(value.FillBytes(make([]byte, 32)))
			}
			outputs[0].SetBytes(h.Sum(nil))
			outputs[0].Mod(outputs[0], mod)
			if outputs[0].Sign() == 0 {
				outputs[0].SetInt64(1)
			}
			return nil
		}
		err = ccs.IsSolved(fullWitness, solver.OverrideHint(solver.GetHintID(frontendcs.Bsb22CommitmentComputePlaceholder), hint))
		res.SolveSeconds = time.Since(t0).Seconds()
		res.SolveOK = err == nil
		if err != nil {
			res.Error = "solve: " + err.Error()
		}
		return res // no proof or cryptographic verification was performed
	}

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
		for rep := 0; rep < warmup+repetitions; rep++ {
			t0 = time.Now()
			proof, proveErr := backendplonk.Prove(ccs, pk, fullWitness)
			proveTime := time.Since(t0).Seconds()
			if proveErr != nil {
				res.Error = "prove: " + proveErr.Error()
				return res
			}
			buf := new(bytesBuffer)
			_, _ = proof.WriteTo(buf)
			res.ProofBytes = buf.Len()
			t0 = time.Now()
			verr := backendplonk.Verify(proof, vk, publicWitness)
			verifyTime := time.Since(t0).Seconds()
			if verr != nil {
				res.Error = "verify: " + verr.Error()
				return res
			}
			if rep >= warmup {
				res.ProveSeconds += proveTime
				res.VerifySeconds += verifyTime
				res.ProveSamples = append(res.ProveSamples, proveTime)
				res.VerifySamples = append(res.VerifySamples, verifyTime)
				res.SuccessfulProofs++
			}
			runtime.GC() // release per-proof temporaries before the next repetition
		}
	default:
		t0 = time.Now()
		pk, vk, err := backendgroth16.Setup(ccs)
		res.SetupSeconds = time.Since(t0).Seconds()
		if err != nil {
			res.Error = "setup: " + err.Error()
			return res
		}
		for rep := 0; rep < warmup+repetitions; rep++ {
			t0 = time.Now()
			proof, proveErr := backendgroth16.Prove(ccs, pk, fullWitness)
			proveTime := time.Since(t0).Seconds()
			if proveErr != nil {
				res.Error = "prove: " + proveErr.Error()
				return res
			}
			buf := new(bytesBuffer)
			_, _ = proof.WriteTo(buf)
			res.ProofBytes = buf.Len()
			t0 = time.Now()
			verr := backendgroth16.Verify(proof, vk, publicWitness)
			verifyTime := time.Since(t0).Seconds()
			if verr != nil {
				res.Error = "verify: " + verr.Error()
				return res
			}
			if rep >= warmup {
				res.ProveSeconds += proveTime
				res.VerifySeconds += verifyTime
				res.ProveSamples = append(res.ProveSamples, proveTime)
				res.VerifySamples = append(res.VerifySamples, verifyTime)
				res.SuccessfulProofs++
			}
			runtime.GC() // release per-proof temporaries before the next repetition
		}
	}
	res.ProveSeconds /= float64(repetitions)
	res.VerifySeconds /= float64(repetitions)
	res.VerifyOK = res.SuccessfulProofs == repetitions

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
