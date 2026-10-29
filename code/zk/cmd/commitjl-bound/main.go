// commitjl-bound benchmarks and independently verifies complete row-binding /
// authenticated-subset proof bundles. Trusted keys/model/input root are pinned
// in a separate manifest produced by the benchmark's trusted setup/data owner.
package main

import (
	"bufio"
	"crypto/sha256"
	"encoding/gob"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math/big"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"time"

	c "commitjl/zk/circuit"
	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark/backend/groth16"
	"github.com/consensys/gnark/constraint"
	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/frontend/cs/r1cs"
	"github.com/consensys/gnark/logger"
)

type Input struct {
	Model   c.BoundModel `json:"model"`
	X       [][]int64    `json:"x"`
	SaltX   string       `json:"salt_x"`
	SaltZ   string       `json:"salt_z"`
	Queries [][]int      `json:"queries"`
	// Reuse phase: new public centre sets (same input quantizer), all with equal K'.
	ReuseCenters   [][][]int64 `json:"reuse_centers,omitempty"`
	ComparisonBase [][]int64   `json:"comparison_base,omitempty"`
}
type Trust struct {
	Model        c.BoundModel      `json:"model"`
	ModelHash    string            `json:"model_hash"`
	OriginalRoot string            `json:"original_root"`
	Keys         map[string]string `json:"verifying_key_sha256"`
}
type Sample struct {
	ID                 int       `json:"id"`
	Indices            []int     `json:"indices,omitempty"`
	Assign             []int     `json:"assign,omitempty"`
	Psi                string    `json:"psi,omitempty"`
	Prove              []float64 `json:"prove_seconds"`
	Verify             []float64 `json:"verify_seconds"`
	Witness            float64   `json:"witness_seconds"`
	NativeAnswer       float64   `json:"native_answer_seconds"`
	ProofBytes         int64     `json:"proof_bytes"`
	PublicWitnessBytes int64     `json:"public_witness_bytes"`
	File               string    `json:"file"`
	Set                int       `json:"centre_set,omitempty"`
	Margin             string    `json:"margin_sum,omitempty"`
	RelativeCount      int       `json:"relative_certified_count,omitempty"`
	CertifiedCount     int       `json:"certified_count,omitempty"`
	CostLower          string    `json:"cost_lower_scaled,omitempty"`
	CostUpper          string    `json:"cost_upper_scaled,omitempty"`
	CertifiedFlags     []bool    `json:"certified_flags,omitempty"`
	ChangeLower        string    `json:"change_lower_scaled,omitempty"`
	ChangeUpper        string    `json:"change_upper_scaled,omitempty"`
}
type Result struct {
	ResumedSetup      bool             `json:"resumed_setup,omitempty"`
	Phase             string           `json:"phase"`
	Status            string           `json:"status"`
	Complete          bool             `json:"complete"`
	Constraints       int              `json:"constraints"`
	ProvingKeyBytes   int64            `json:"proving_key_bytes"`
	VerifyingKeyBytes int64            `json:"verifying_key_bytes"`
	Compile           float64          `json:"compile_seconds"`
	Setup             float64          `json:"setup_seconds"`
	Load              float64          `json:"load_seconds"`
	Native            float64          `json:"native_prepare_seconds"`
	Warmup            float64          `json:"warmup_seconds"`
	Statement         c.BoundStatement `json:"statement"`
	Samples           []Sample         `json:"samples"`
	Verified          int              `json:"verified"`
	Wall              float64          `json:"wall_seconds"`
	Error             string           `json:"error,omitempty"`
}

var folder string

const binaryIOBufferSize = 256 << 10

func save(name string, v any) error {
	path := filepath.Join(folder, name)
	if err := os.MkdirAll(filepath.Dir(path), 0700); err != nil {
		return err
	}
	b, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return err
	}
	f, err := os.OpenFile(path+".tmp", os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0600)
	if err != nil {
		return err
	}
	_, err = f.Write(b)
	if err == nil {
		err = f.Sync()
	}
	closeErr := f.Close()
	if err != nil {
		return err
	}
	if closeErr != nil {
		return closeErr
	}
	return os.Rename(path+".tmp", path)
}
func load(path string, v any) error {
	b, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	return json.Unmarshal(b, v)
}
func saveBinary(name string, v io.WriterTo) (int64, error) {
	path := filepath.Join(folder, name)
	if err := os.MkdirAll(filepath.Dir(path), 0700); err != nil {
		return 0, err
	}
	f, err := os.Create(path)
	if err != nil {
		return 0, err
	}
	buffer := bufio.NewWriterSize(f, binaryIOBufferSize)
	n, err := v.WriteTo(buffer)
	if err == nil {
		err = buffer.Flush()
	}
	if err == nil {
		err = f.Sync()
	}
	e := f.Close()
	if err == nil {
		err = e
	}
	return n, err
}
func digest(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	if _, err = io.Copy(h, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}
func prepared() (*c.BoundPrepared, error) {
	f, err := os.Open(filepath.Join(folder, "prepared.gob"))
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var p c.BoundPrepared
	err = gob.NewDecoder(f).Decode(&p)
	return &p, err
}
func parseInt(text string) (*big.Int, error) {
	v, ok := new(big.Int).SetString(text, 10)
	if !ok {
		return nil, fmt.Errorf("invalid integer")
	}
	return v, nil
}

func prepare(input string) (res Result, err error) {
	start := time.Now()
	res.Phase = "prepare"
	res.Status = "failed"
	defer func() {
		res.Wall = time.Since(start).Seconds()
		if err != nil {
			res.Error = err.Error()
		}
	}()
	var in Input
	if err = load(input, &in); err != nil {
		return
	}
	sx, e := parseInt(in.SaltX)
	if e != nil {
		err = e
		return
	}
	sz, e := parseInt(in.SaltZ)
	if e != nil {
		err = e
		return
	}
	if _, e = os.Stat(filepath.Join(folder, "prepare.json")); e == nil {
		err = fmt.Errorf("choose a new bundle directory; refusing to mix preparations")
		return
	}
	t := time.Now()
	p, e := c.PrepareBound(in.Model, in.X, sx, sz)
	res.Native = time.Since(t).Seconds()
	if e != nil {
		err = e
		return
	}
	f, e := os.Create(filepath.Join(folder, "prepared.gob"))
	if e != nil {
		err = e
		return
	}
	err = gob.NewEncoder(f).Encode(p)
	if err == nil {
		err = f.Sync()
	}
	f.Close()
	if err != nil {
		return
	}
	trust := Trust{Model: in.Model, ModelHash: in.Model.Hash().String(), OriginalRoot: p.CX.String(), Keys: map[string]string{}}
	if err = save("trusted.json", trust); err != nil {
		return
	}
	res.Statement = p.Statement()
	rows := make([]map[string]any, len(p.H))
	for i, h := range p.H {
		hs := make([]string, len(h))
		ts := make([]string, len(p.T[i]))
		for j, v := range h {
			hs[j] = v.String()
		}
		for j, v := range p.T[i] {
			ts[j] = v.String()
		}
		rows[i] = map[string]any{"index": i, "head": hs, "energy": p.E[i].String(), "tail": ts}
	}
	if err = save("statistics.json", rows); err != nil {
		return
	}
	res.Complete = true
	res.Status = "complete"
	return
}

func setup(shape frontend.Circuit, key string, limit int, res *Result) (constraint.ConstraintSystem, groth16.ProvingKey, groth16.VerifyingKey, error) {
	if resumeEnabled() {
		cs, pk, vk, hit, e := loadSetupCache(key, res)
		if e != nil {
			return nil, nil, nil, e
		}
		if hit {
			if res.Constraints > limit {
				return nil, nil, nil, fmt.Errorf("cached shape exceeds guard")
			}
			return cs, pk, vk, nil
		}
	}
	t := time.Now()
	ccs, err := frontend.Compile(ecc.BN254.ScalarField(), r1cs.NewBuilder, shape)
	res.Compile = time.Since(t).Seconds()
	if err != nil {
		return nil, nil, nil, err
	}
	res.Constraints = ccs.GetNbConstraints()
	if res.Constraints > limit {
		res.Status = "excluded_constraint_budget"
		return nil, nil, nil, nil
	}
	t = time.Now()
	pk, vk, err := groth16.Setup(ccs)
	res.Setup = time.Since(t).Seconds()
	if err != nil {
		return nil, nil, nil, err
	}
	if res.VerifyingKeyBytes, err = saveBinary("keys/"+key+".vk", vk); err != nil {
		return nil, nil, nil, err
	}
	// A resumable setup writes the same canonical key below. Count those
	// actual bytes instead of serializing a large elliptic-curve key twice.
	if !resumeEnabled() {
		if res.ProvingKeyBytes, err = pk.WriteTo(io.Discard); err != nil {
			return nil, nil, nil, err
		}
	}
	var trust Trust
	if err = load(filepath.Join(folder, "trusted.json"), &trust); err != nil {
		return nil, nil, nil, err
	}
	sha, e := digest(filepath.Join(folder, "keys", key+".vk"))
	if e != nil {
		return nil, nil, nil, e
	}
	trust.Keys[key] = sha
	if err = save("trusted.json", trust); err != nil {
		return nil, nil, nil, err
	}
	if resumeEnabled() {
		if e := saveSetupCache(key, ccs, pk, res); e != nil {
			return nil, nil, nil, e
		}
	}
	return ccs, pk, vk, nil
}

func prove(ccs constraint.ConstraintSystem, pk groth16.ProvingKey, vk groth16.VerifyingKey, w frontend.Circuit, file string, reps, warmup int) (sample Sample, err error) {
	t := time.Now()
	full, e := frontend.NewWitness(w, ecc.BN254.ScalarField())
	if e != nil {
		err = e
		return
	}
	pub, e := full.Public()
	if e != nil {
		err = e
		return
	}
	sample.Witness = time.Since(t).Seconds()
	if sample.PublicWitnessBytes, err = pub.WriteTo(io.Discard); err != nil {
		return
	}
	sample.File = file
	if resumeEnabled() {
		cached, hit, e := reuseSample(file, pub, vk)
		if e != nil {
			return sample, e
		}
		if hit {
			if len(cached.Prove) != reps {
				return sample, fmt.Errorf("cached repetitions differ")
			}
			return cached, nil
		}
	}
	for i := 0; i < reps+warmup; i++ {
		t = time.Now()
		proof, e := groth16.Prove(ccs, pk, full)
		pt := time.Since(t).Seconds()
		if e != nil {
			err = e
			return
		}
		t = time.Now()
		e = groth16.Verify(proof, vk, pub)
		vt := time.Since(t).Seconds()
		if e != nil {
			err = e
			return
		}
		if i >= warmup {
			sample.Prove = append(sample.Prove, pt)
			sample.Verify = append(sample.Verify, vt)
			sample.ProofBytes, e = saveBinary(file, proof)
			if e != nil {
				err = e
				return
			}
		}
		runtime.GC()
	}
	if resumeEnabled() {
		err = saveSampleCache(file, sample, pub, vk)
	}
	return
}

func execute(phase, input string, q, reps, warmup, limit int) (res Result, err error) {
	start := time.Now()
	res.Phase = phase
	res.Status = "failed"
	defer func() {
		res.Wall = time.Since(start).Seconds()
		if err != nil {
			res.Error = err.Error()
		}
	}()
	t := time.Now()
	p, e := prepared()
	res.Load = time.Since(t).Seconds()
	if e != nil {
		err = e
		return
	}
	if err = p.M.Validate(); err != nil {
		return
	}
	res.Statement = p.Statement()
	var trust Trust
	if err = load(filepath.Join(folder, "trusted.json"), &trust); err != nil {
		return
	}
	if trust.ModelHash != p.M.Hash().String() || trust.OriginalRoot != p.CX.String() {
		err = fmt.Errorf("prepared input differs from pinned data/model")
		return
	}
	key := "stage1"
	var shape frontend.Circuit
	if phase == "stage1" {
		if p.M.Method == "full" {
			res.Complete = true
			res.Status = "not_required_full"
			return
		}
		shape = c.NewBoundPreprocess(p.M)
	} else {
		key = fmt.Sprintf("query_%d", q)
		shape = c.NewBoundQuery(p.M, q)
		if q < 1 || q > p.M.N {
			err = fmt.Errorf("invalid query size")
			return
		}
		// Guard very large Full shapes before compiler allocation. This estimate
		// is a resource exclusion, never a measured constraint result.
		if p.M.Method == "full" && 200*q*(p.M.D+6+4*p.M.Depth()) > limit*12/10 {
			res.Status = "excluded_preflight_budget"
			return
		}
		if p.M.Method != "full" {
			var previous Result
			if err = load(filepath.Join(folder, "stage1.json"), &previous); err != nil {
				return
			}
			if !previous.Complete || previous.Verified != p.M.N {
				err = fmt.Errorf("complete preprocessing coverage required")
				return
			}
		}
	}
	ccs, pk, vk, e := setup(shape, key, limit, &res)
	if e != nil {
		err = e
		return
	}
	if ccs == nil {
		return
	}
	if phase == "stage1" {
		for index := 0; index < p.M.N; index++ {
			extra := 0
			if index == 0 {
				extra = warmup
			}
			file := fmt.Sprintf("stage1/row_%06d.proof", index)
			sample, e := prove(ccs, pk, vk, c.FillBoundPreprocess(p, index), file, 1, extra)
			if e != nil {
				err = e
				return
			}
			sample.ID = index
			res.Samples = append(res.Samples, sample)
			res.Verified++
			if err = save("stage1_progress.json", res); err != nil {
				return
			}
		}
	} else {
		var in Input
		if err = load(input, &in); err != nil {
			return
		}
		queries := [][]int{}
		for _, indices := range in.Queries {
			if len(indices) == q {
				queries = append(queries, indices)
			}
		}
		if len(queries) == 0 {
			err = fmt.Errorf("no queries of requested size")
			return
		}
		for id, indices := range queries {
			nativeStart := time.Now()
			w, e := c.FillBoundQuery(p, indices)
			if e != nil {
				err = e
				return
			}
			labels := make([]int, len(indices))
			for i, v := range w.Assign {
				labels[i] = v.(int)
			}
			psi := w.Psi.(*big.Int)
			nativeSeconds := time.Since(nativeStart).Seconds()
			file := fmt.Sprintf("queries/q%d_%04d.proof", q, id)
			sample, e := prove(ccs, pk, vk, w, file, reps, warmup)
			if e != nil {
				err = e
				return
			}
			sample.ID = id
			sample.NativeAnswer = nativeSeconds
			sample.Indices = indices
			sample.Assign = labels
			sample.Psi = psi.String()
			res.Samples = append(res.Samples, sample)
			res.Verified++
			if err = save(fmt.Sprintf("query_%d_progress.json", q), res); err != nil {
				return
			}
		}
	}
	res.Complete = true
	res.Status = "complete"
	return
}

func readKey(trust Trust, key string) (groth16.VerifyingKey, error) {
	name := filepath.Join(folder, "keys", key+".vk")
	sha, err := digest(name)
	if err != nil {
		return nil, err
	}
	if expected, ok := trust.Keys[key]; !ok || expected != sha {
		return nil, fmt.Errorf("verification key does not match trusted setup pin")
	}
	f, err := os.Open(name)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	vk := groth16.NewVerifyingKey(ecc.BN254)
	_, err = vk.ReadFrom(bufio.NewReaderSize(f, binaryIOBufferSize))
	return vk, err
}
func verifyFile(vk groth16.VerifyingKey, file string, w frontend.Circuit) error {
	f, err := os.Open(filepath.Join(folder, file))
	if err != nil {
		return err
	}
	defer f.Close()
	proof := groth16.NewProof(ecc.BN254)
	if _, err = proof.ReadFrom(bufio.NewReaderSize(f, binaryIOBufferSize)); err != nil {
		return err
	}
	public, err := frontend.NewWitness(w, ecc.BN254.ScalarField(), frontend.PublicOnly())
	if err != nil {
		return err
	}
	return groth16.Verify(proof, vk, public)
}

// The trust path must come from the data owner/trusted setup, not the prover.
// Verification reads no private X, head, energy, tail, or authentication path.
func verify(trustPath, taskPath string) (res Result, err error) {
	start := time.Now()
	res.Phase = "independent_verify"
	res.Status = "failed"
	defer func() {
		res.Wall = time.Since(start).Seconds()
		if err != nil {
			res.Error = err.Error()
		}
	}()
	var trust Trust
	if err = load(trustPath, &trust); err != nil {
		return
	}
	if err = trust.Model.Validate(); err != nil {
		return
	}
	m := trust.Model
	if m.Hash().String() != trust.ModelHash {
		err = fmt.Errorf("trusted model digest mismatch")
		return
	}
	var prep Result
	if err = load(filepath.Join(folder, "prepare.json"), &prep); err != nil {
		return
	}
	s := prep.Statement
	if _, _, _, _, _, err = s.Values(m); err != nil {
		return
	}
	if s.CX != trust.OriginalRoot {
		err = fmt.Errorf("original root differs from owner pin")
		return
	}
	res.Statement = s
	if m.Method != "full" {
		var phase Result
		if err = load(filepath.Join(folder, "stage1.json"), &phase); err != nil {
			return
		}
		if !phase.Complete || phase.Verified != m.N || phase.Statement != s || len(phase.Samples) != m.N {
			err = fmt.Errorf("missing preprocessing coverage or inconsistent roots")
			return
		}
		vk, e := readKey(trust, "stage1")
		if e != nil {
			err = e
			return
		}
		for i, sample := range phase.Samples {
			if sample.ID != i || sample.File != fmt.Sprintf("stage1/row_%06d.proof", i) {
				err = fmt.Errorf("duplicate/missing preprocessing index")
				return
			}
			public, e := c.BoundPreprocessPublic(m, s, i)
			if e != nil {
				err = e
				return
			}
			if err = verifyFile(vk, sample.File, public); err != nil {
				return
			}
			res.Verified++
		}
	}
	keys := []string{}
	for key := range trust.Keys {
		if key != "stage1" {
			keys = append(keys, key)
		}
	}
	sort.Strings(keys)
	if len(keys) == 0 {
		err = fmt.Errorf("no authenticated query proofs in bundle")
		return
	}
	if err = verifyChangingTaskCoverage(keys, taskPath); err != nil {
		return
	}
	for _, key := range keys {
		if strings.HasPrefix(key, "family_affine_") || strings.HasPrefix(key, "relative_reuse_") || strings.HasPrefix(key, "reuse_") || strings.HasPrefix(key, "full_reuse_") || strings.HasPrefix(key, "paired_change_") {
			if err = verifyChangingPhase(trust, m, s, key, taskPath, &res); err != nil {
				return
			}
			continue
		}
		var phase Result
		if err = load(filepath.Join(folder, key+".json"), &phase); err != nil {
			return
		}
		if !phase.Complete || phase.Statement != s || phase.Verified != len(phase.Samples) || len(phase.Samples) == 0 {
			err = fmt.Errorf("incomplete query proof phase")
			return
		}
		vk, e := readKey(trust, key)
		if e != nil {
			err = e
			return
		}
		for id, sample := range phase.Samples {
			if key == "geometry" {
				if sample.ID != id || sample.File != fmt.Sprintf("geometry/pair_%04d.proof", id) {
					err = fmt.Errorf("geometry proof identity mismatch")
					return
				}
				public, e := c.BoundGeometryPublic(m, s, sample.Indices, sample.Psi)
				if e != nil {
					err = e
					return
				}
				if err = verifyFile(vk, sample.File, public); err != nil {
					return
				}
				res.Verified++
				continue
			}
			expected := fmt.Sprintf("query_%d", len(sample.Indices))
			file := fmt.Sprintf("queries/q%d_%04d.proof", len(sample.Indices), id)
			if expected != key || sample.ID != id || sample.File != file {
				err = fmt.Errorf("query shape or proof identity mismatch")
				return
			}
			public, e := c.BoundQueryPublic(m, s, sample.Indices, sample.Psi)
			if e != nil {
				err = e
				return
			}
			if err = verifyFile(vk, sample.File, public); err != nil {
				return
			}
			res.Verified++
		}
	}
	res.Status = "complete"
	res.Complete = true
	return
}

func main() {
	logger.Disable()
	phase := flag.String("phase", "prepare", "prepare, stage1, query, reuse, full_reuse, geometry, verify")
	input := flag.String("instance", "", "original input JSON; not needed for verification")
	out := flag.String("out", "", "new bundle directory")
	trust := flag.String("trust", "", "pinned owner/setup manifest for independent verification")
	task := flag.String("public-task", "", "independently approved public centers and query subsets for replay")
	refineBase := flag.String("refinement-base", "", "complete compact base bundle, public-only for verification")
	refineTrust := flag.String("refinement-trust", "", "independently pinned refinement setup manifest")
	refineRequest := flag.String("refinement-request", "", "width-ordered bounded row requests")
	q := flag.Int("q", 1, "query batch size")
	reps := flag.Int("reps", 3, "measured query proofs")
	warmup := flag.Int("warmup", 1, "warmup proofs")
	limit := flag.Int("max-constraints", 1300000, "compiler constraint resource budget")
	flag.Parse()
	folder = *out
	if folder == "" {
		fmt.Fprintln(os.Stderr, "out directory required")
		os.Exit(1)
	}
	if *phase == "prepare" {
		if _, err := os.Stat(filepath.Join(folder, "prepare.json")); err == nil {
			fmt.Fprintln(os.Stderr, "existing preparation preserved; choose a new output directory")
			os.Exit(2)
		}
	}
	if err := os.MkdirAll(folder, 0700); err != nil {
		panic(err)
	}
	var result Result
	var err error
	name := *phase + ".json"
	switch *phase {
	case "prepare":
		result, err = prepare(*input)
	case "group_compact_prepare":
		result, err = prepareCompactGroup(*input)
	case "refinement_setup", "refinement_prove", "refinement_verify":
		result, err = executeRefinement(*phase, *trust, *task, *refineBase, *refineTrust, *refineRequest, *limit)
	case "moments_prepare", "moments_keys", "moments_rows", "moments_query", "moments_verify":
		result, err = executeMoments(*phase, *input, *trust, *task, *q, *limit)
	case "group_commit", "group_commit_exact", "group_keys", "group_rows", "group_link", "group_query", "group_verify":
		result, err = executeGroup(*phase, *trust, *task, *q, *limit)
	case "reuse":
		if *reps < 1 || *warmup < 0 {
			err = fmt.Errorf("invalid repeat count")
		} else {
			result, err = executeReuse(*input, *q, *reps, *warmup, *limit)
		}
		name = fmt.Sprintf("reuse_%d.json", *q)
	case "full_reuse":
		if *reps < 1 || *warmup < 0 {
			err = fmt.Errorf("invalid repeat count")
		} else {
			result, err = executeFullReuse(*input, *q, *reps, *warmup, *limit)
		}
		name = fmt.Sprintf("full_reuse_%d.json", *q)
	case "family_affine":
		if *reps < 1 || *warmup < 0 {
			err = fmt.Errorf("invalid repeat count")
		} else {
			result, err = executeFamilyAffine(*input, *q, *reps, *warmup, *limit)
		}
		name = fmt.Sprintf("family_affine_%d.json", *q)
	case "relative_reuse":
		if *reps < 1 || *warmup < 0 {
			err = fmt.Errorf("invalid repeat count")
		} else {
			result, err = executeRelativeReuse(*input, *q, *reps, *warmup, *limit)
		}
		name = fmt.Sprintf("relative_reuse_%d.json", *q)
	case "reuse_certified":
		if *reps < 1 || *warmup < 0 {
			err = fmt.Errorf("invalid repeat count")
		} else {
			result, err = executeCertifiedReuse(*input, *q, *reps, *warmup, *limit)
		}
		name = fmt.Sprintf("reuse_certified_%d.json", *q)
	case "paired_change":
		if *reps < 1 || *warmup < 0 {
			err = fmt.Errorf("invalid repeat count")
		} else {
			result, err = executePairedChange(*input, *q, *reps, *warmup, *limit)
		}
		name = fmt.Sprintf("paired_change_%d.json", *q)
	case "stage1", "query", "geometry":
		if *reps < 1 || *warmup < 0 {
			err = fmt.Errorf("invalid repeat count")
		} else {
			if *phase == "geometry" {
				result, err = executeGeometry(*input, *reps, *warmup, *limit)
			} else {
				result, err = execute(*phase, *input, *q, *reps, *warmup, *limit)
			}
		}
		if *phase == "query" {
			name = fmt.Sprintf("query_%d.json", *q)
		}
	case "verify":
		if *trust == "" {
			err = fmt.Errorf("explicit trusted manifest required")
		} else {
			result, err = verify(*trust, *task)
		}
	default:
		err = fmt.Errorf("unknown phase")
	}
	if err != nil {
		result.Error = err.Error()
	}
	if e := save(name, result); e != nil {
		fmt.Fprintln(os.Stderr, e)
		os.Exit(1)
	}
	b, _ := json.Marshal(map[string]any{"phase": result.Phase, "status": result.Status, "constraints": result.Constraints, "verified": result.Verified, "wall_seconds": result.Wall, "error": result.Error})
	fmt.Println(string(b))
	if err != nil {
		os.Exit(2)
	}
	if !result.Complete {
		os.Exit(3)
	}
}
