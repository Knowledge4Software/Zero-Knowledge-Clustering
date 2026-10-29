// Backend sensitivity control for the EXISTING exact public-center relation.
// The test SRS is explicitly unsafe: this measures execution, not a deployment
// security claim. Heavy generation is remote-only; old binaries are untouched.
package main

import (
	"bufio"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math/big"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	c "commitjl/zk/circuit"
	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark/backend/plonk"
	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/frontend/cs/scs"
	"github.com/consensys/gnark/test/unsafekzg"
)

type input struct {
	Model   c.BoundModel `json:"model"`
	X       [][]int64    `json:"x"`
	SaltX   string       `json:"salt_x"`
	SaltZ   string       `json:"salt_z"`
	Centers [][][]int64  `json:"reuse_centers"`
}
type task struct {
	Centers [][][]int64 `json:"centers"`
}
type pin struct {
	Model     c.BoundModel     `json:"model"`
	Statement c.BoundStatement `json:"statement"`
	KeySHA    string           `json:"verifying_key_sha256"`
	Batch     int              `json:"batch"`
	UnsafeSRS bool             `json:"benchmark_only_unsafe_srs"`
}
type sample struct {
	Set            int     `json:"set"`
	Indices        []int   `json:"indices"`
	Cost           string  `json:"cost"`
	File           string  `json:"file"`
	SHA            string  `json:"proof_sha256"`
	WitnessSeconds float64 `json:"witness_seconds"`
	ProveSeconds   float64 `json:"prove_seconds"`
	VerifySeconds  float64 `json:"verify_seconds"`
	Bytes          int64   `json:"proof_bytes"`
}
type result struct {
	Complete       bool     `json:"complete"`
	Backend        string   `json:"backend"`
	UnsafeSRS      bool     `json:"benchmark_only_unsafe_srs"`
	Constraints    int      `json:"constraints"`
	CompileSeconds float64  `json:"compile_seconds"`
	PrepareSeconds float64  `json:"native_prepare_seconds"`
	SetupSeconds   float64  `json:"srs_and_setup_seconds"`
	SRSSeconds     float64  `json:"srs_generation_or_loading_seconds"`
	CircuitSetup   float64  `json:"circuit_setup_seconds"`
	SRSCache       string   `json:"benchmark_srs_cache,omitempty"`
	WallSeconds    float64  `json:"wall_seconds"`
	Samples        []sample `json:"samples"`
}

func read(path string, v any) error {
	b, e := os.ReadFile(path)
	if e != nil {
		return e
	}
	return json.Unmarshal(b, v)
}
func save(path string, v any) error {
	b, e := json.MarshalIndent(v, "", "  ")
	if e != nil {
		return e
	}
	return os.WriteFile(path, b, 0600)
}
func hash(path string) (string, error) {
	f, e := os.Open(path)
	if e != nil {
		return "", e
	}
	defer f.Close()
	h := sha256.New()
	if _, e = io.Copy(h, f); e != nil {
		return "", e
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}
func writeObject(path string, v io.WriterTo) (int64, error) {
	f, e := os.OpenFile(path, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
	if e != nil {
		return 0, e
	}
	buffer := bufio.NewWriterSize(f, 256<<10)
	n, e := v.WriteTo(buffer)
	if e == nil {
		e = buffer.Flush()
	}
	if e == nil {
		e = f.Sync()
	}
	closeError := f.Close()
	if e == nil {
		e = closeError
	}
	return n, e
}
func readObject(path string, v io.ReaderFrom) error {
	f, e := os.Open(path)
	if e != nil {
		return e
	}
	defer f.Close()
	_, e = v.ReadFrom(bufio.NewReaderSize(f, 256<<10))
	return e
}
func salt(text string) (*big.Int, error) {
	n, ok := new(big.Int).SetString(text, 10)
	if !ok || n.Sign() < 0 {
		return nil, fmt.Errorf("invalid owner salt")
	}
	return n, nil
}
func remoteOnly() error {
	host, e := os.Hostname()
	if e != nil {
		return e
	}
	if os.Getenv("COMMITJL_EXECUTION_TARGET") == "laptop" && runtime.GOOS == "windows" {
		var policy struct {
			Heavy  bool   `json:"heavy_runs_allowed"`
			Proofs bool   `json:"prover_generation_allowed"`
			Local  bool   `json:"local_campaign_authorized"`
			Host   string `json:"local_allowed_hostname"`
		}
		if e := read(os.Getenv("COMMITJL_EXECUTION_POLICY"), &policy); e != nil {
			return e
		}
		if policy.Heavy && policy.Proofs && policy.Local && strings.EqualFold(host, policy.Host) {
			return nil
		}
		return fmt.Errorf("local proof generation is not authorized by the current policy")
	}
	if runtime.GOOS != "linux" || os.Getenv("COMMITJL_EXECUTION_TARGET") != "mobin" || (strings.ToLower(host) != "mobin" && strings.ToLower(host) != "cloudfpsir") {
		return fmt.Errorf("heavy proof generation is held on the laptop; this control requires authorized Mobin Linux execution")
	}
	return nil
}
func validateTask(m c.BoundModel, t task, q int) error {
	if e := m.Validate(); e != nil {
		return e
	}
	if m.Method != "full" || q < 1 || m.N%q != 0 || len(t.Centers) == 0 {
		return fmt.Errorf("require exact Full, complete batches and candidate sets")
	}
	limit := int64(1) << uint(m.InputBits)
	for _, set := range t.Centers {
		if len(set) != m.K {
			return fmt.Errorf("center count differs from circuit shape")
		}
		for _, row := range set {
			if len(row) != m.D {
				return fmt.Errorf("center dimension mismatch")
			}
			for _, v := range row {
				if v < -limit || v > limit {
					return fmt.Errorf("center out of approved range")
				}
			}
		}
	}
	return nil
}
func verify(out, trust, approved string) error {
	if trust == "" || approved == "" {
		return fmt.Errorf("verification requires independently retained trust and complete public task")
	}
	var p pin
	var t task
	var r result
	if e := read(trust, &p); e != nil {
		return e
	}
	if e := read(approved, &t); e != nil {
		return e
	}
	if e := read(filepath.Join(out, "result.json"), &r); e != nil {
		return e
	}
	if !r.Complete || r.Backend != "gnark-plonk-bn254" || !p.UnsafeSRS || !r.UnsafeSRS {
		return fmt.Errorf("unsupported/incomplete benchmark control")
	}
	if e := validateTask(p.Model, t, p.Batch); e != nil {
		return e
	}
	h, e := hash(filepath.Join(out, "vk.bin"))
	if e != nil {
		return e
	}
	if h != p.KeySHA {
		return fmt.Errorf("unapproved verification key")
	}
	vk := plonk.NewVerifyingKey(ecc.BN254)
	if e = readObject(filepath.Join(out, "vk.bin"), vk); e != nil {
		return e
	}
	perSet := p.Model.N / p.Batch
	if len(r.Samples) != len(t.Centers)*perSet {
		return fmt.Errorf("incomplete query coverage")
	}
	started := time.Now()
	for pos, s := range r.Samples {
		set, offset := pos/perSet, (pos%perSet)*p.Batch
		if s.Set != set || len(s.Indices) != p.Batch || s.File != fmt.Sprintf("s%03d_b%06d.proof", set, offset/p.Batch) {
			return fmt.Errorf("reordered/substituted proof")
		}
		for j, index := range s.Indices {
			if index != offset+j {
				return fmt.Errorf("missing/duplicated catalog index")
			}
		}
		h, e = hash(filepath.Join(out, s.File))
		if e != nil {
			return e
		}
		if h != s.SHA {
			return fmt.Errorf("changed proof bytes")
		}
		pub, e := c.BoundFullReusePublic(p.Model, p.Statement, t.Centers[set], s.Indices, s.Cost)
		if e != nil {
			return e
		}
		w, e := frontend.NewWitness(pub, ecc.BN254.ScalarField(), frontend.PublicOnly())
		if e != nil {
			return e
		}
		proof := plonk.NewProof(ecc.BN254)
		if e = readObject(filepath.Join(out, s.File), proof); e != nil {
			return e
		}
		if e = plonk.Verify(proof, vk, w); e != nil {
			return e
		}
	}
	return save(filepath.Join(out, "public_verify.json"), map[string]any{"complete": true, "verified": len(r.Samples), "wall_seconds": time.Since(started).Seconds(), "benchmark_only_unsafe_srs": true, "private_rows_loaded": false})
}
func generate(source, out string, q, guard int, inspect, allowUnsafe bool, srsCache string, srsOnly bool) error {
	if !inspect {
		if e := remoteOnly(); e != nil {
			return e
		}
		if !allowUnsafe {
			return fmt.Errorf("benchmark SRS must be explicitly acknowledged")
		}
	}
	var in input
	if e := read(source, &in); e != nil {
		return e
	}
	if e := validateTask(in.Model, task{in.Centers}, q); e != nil {
		return e
	}
	// Refuse reuse of unfinished runs: a serial external ledger must charge
	// failures and choose a new attempt directory. No uncharged cache reuse.
	if e := os.Mkdir(out, 0700); e != nil {
		return e
	}
	started := time.Now()
	r := result{Backend: "gnark-plonk-bn254", UnsafeSRS: true, SRSCache: srsCache}
	shape := c.NewBoundFullReuseQuery(in.Model, q, in.Model.K)
	step := time.Now()
	ccs, e := frontend.Compile(ecc.BN254.ScalarField(), scs.NewBuilder, shape)
	if e != nil {
		return e
	}
	r.CompileSeconds = time.Since(step).Seconds()
	r.Constraints = ccs.GetNbConstraints()
	if inspect {
		return save(filepath.Join(out, "compile_only.json"), map[string]any{"complete": true, "constraints": r.Constraints, "compile_seconds": r.CompileSeconds, "proofs_generated": false, "within_guard": r.Constraints <= guard})
	}
	if r.Constraints > guard {
		return fmt.Errorf("constraint guard: %d > %d", r.Constraints, guard)
	}
	sx, e := salt(in.SaltX)
	if e != nil {
		return e
	}
	sz, e := salt(in.SaltZ)
	if e != nil {
		return e
	}
	step = time.Now()
	prepared, e := c.PrepareBound(in.Model, in.X, sx, sz)
	if e != nil {
		return e
	}
	r.PrepareSeconds = time.Since(step).Seconds()
	step = time.Now()
	var options []unsafekzg.Option
	if srsCache != "" {
		options = append(options, unsafekzg.WithCacheDir(srsCache))
	}
	srs, lagrange, e := unsafekzg.NewSRS(ccs, options...)
	if e != nil {
		return e
	}
	r.SRSSeconds = time.Since(step).Seconds()
	if srsOnly {
		return save(filepath.Join(out, "srs_preparation.json"), map[string]any{"complete": true, "proofs_generated": 0, "constraints": r.Constraints, "srs_generation_or_loading_seconds": r.SRSSeconds, "benchmark_only_unsafe_srs": true, "srs_cache": srsCache})
	}
	step = time.Now()
	pk, vk, e := plonk.Setup(ccs, srs, lagrange)
	if e != nil {
		return e
	}
	r.CircuitSetup = time.Since(step).Seconds()
	r.SetupSeconds = r.SRSSeconds + r.CircuitSetup
	if _, e = writeObject(filepath.Join(out, "vk.bin"), vk); e != nil {
		return e
	}
	keyHash, e := hash(filepath.Join(out, "vk.bin"))
	if e != nil {
		return e
	}
	if e = save(filepath.Join(out, "owner_pin.json"), pin{in.Model, prepared.Statement(), keyHash, q, true}); e != nil {
		return e
	}
	if e = save(filepath.Join(out, "approved_task.json"), task{in.Centers}); e != nil {
		return e
	}
	for set, centers := range in.Centers {
		for start := 0; start < in.Model.N; start += q {
			indices := make([]int, q)
			for j := range indices {
				indices[j] = start + j
			}
			s := sample{Set: set, Indices: indices, File: fmt.Sprintf("s%03d_b%06d.proof", set, start/q)}
			step = time.Now()
			assignment, e := c.FillBoundFullReuseQuery(prepared, centers, indices)
			if e != nil {
				return e
			}
			s.Cost = assignment.Psi.(*big.Int).String()
			w, e := frontend.NewWitness(assignment, ecc.BN254.ScalarField())
			if e != nil {
				return e
			}
			pub, e := w.Public()
			if e != nil {
				return e
			}
			s.WitnessSeconds = time.Since(step).Seconds()
			step = time.Now()
			proof, e := plonk.Prove(ccs, pk, w)
			if e != nil {
				return e
			}
			s.ProveSeconds = time.Since(step).Seconds()
			step = time.Now()
			if e = plonk.Verify(proof, vk, pub); e != nil {
				return e
			}
			s.VerifySeconds = time.Since(step).Seconds()
			s.Bytes, e = writeObject(filepath.Join(out, s.File), proof)
			if e != nil {
				return e
			}
			s.SHA, e = hash(filepath.Join(out, s.File))
			if e != nil {
				return e
			}
			r.Samples = append(r.Samples, s)
			r.WallSeconds = time.Since(started).Seconds()
			if e = save(filepath.Join(out, "result.json"), r); e != nil {
				return e
			}
		}
	}
	r.Complete = true
	r.WallSeconds = time.Since(started).Seconds()
	return save(filepath.Join(out, "result.json"), r)
}
func main() {
	phase := flag.String("phase", "inspect", "inspect, run or verify")
	source := flag.String("input", "", "owner's Full input")
	out := flag.String("out", "", "new output directory")
	trust := flag.String("trust", "", "independent owner pin")
	approved := flag.String("public-task", "", "independently approved full center arrays")
	q := flag.Int("q", 8, "complete row batch")
	guard := flag.Int("max-constraints", 1600000, "constraint guard")
	unsafe := flag.Bool("allow-benchmark-srs", false, "acknowledge test-only unsafe KZG SRS")
	srsCache := flag.String("srs-cache", "", "private test-SRS cache, never a production ceremony")
	srsOnly := flag.Bool("prepare-srs-only", false, "prepare/load the test SRS without circuit keys or proofs")
	flag.Parse()
	var e error
	if *out == "" {
		e = fmt.Errorf("output directory required")
	} else {
		switch *phase {
		case "inspect", "run":
			e = generate(*source, *out, *q, *guard, *phase == "inspect", *unsafe, *srsCache, *srsOnly)
		case "verify":
			e = verify(*out, *trust, *approved)
		default:
			e = fmt.Errorf("unknown phase")
		}
	}
	if e != nil {
		fmt.Fprintln(os.Stderr, e)
		os.Exit(1)
	}
}
