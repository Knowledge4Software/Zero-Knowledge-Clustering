package main

// Explicit, read-only microbenchmarks of frozen real-workload keys. These do
// not resume a campaign, change its receipts, or count as deployment timings.
import (
	"bufio"
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"testing"
	"time"

	c "commitjl/zk/circuit"
	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark/backend/groth16"
	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/frontend/cs/r1cs"
)

func TestFrozenCircuitEquivalence(t *testing.T) {
	source := os.Getenv("COMMITJL_BENCH_SOURCE")
	if source == "" {
		t.Skip("explicit frozen-workload source required")
	}
	previousCPU := runtime.GOMAXPROCS(2)
	defer runtime.GOMAXPROCS(previousCPU)
	previous := folder
	folder = filepath.Join(source, "bundle")
	defer func() { folder = previous }()
	p, err := prepared()
	if err != nil {
		t.Fatal(err)
	}
	var task PublicTask
	if err = load(filepath.Join(source, "approved.json"), &task); err != nil {
		t.Fatal(err)
	}
	q := len(task.Queries[0])
	shapes := groupProfileShapes(p.M, q, compactGroupVersion)
	results := []map[string]any{}
	for i, key := range groupProfileKeys(q, compactGroupVersion) {
		var before setupCache
		if err = load(filepath.Join(folder, ".resume", key+".json"), &before); err != nil {
			t.Fatal(err)
		}
		started := time.Now()
		cs, e := frontend.Compile(ecc.BN254.ScalarField(), r1cs.NewBuilder, shapes[i])
		seconds := time.Since(started).Seconds()
		if e != nil {
			t.Fatal(e)
		}
		h := sha256.New()
		if _, e = cs.WriteTo(h); e != nil {
			t.Fatal(e)
		}
		after := hex.EncodeToString(h.Sum(nil))
		if after != before.CCSSHA {
			t.Fatalf("%s: compiled constraint-system bytes changed: %s != %s", key, after, before.CCSSHA)
		}
		result := map[string]any{"key": key, "old_compile_seconds": before.Measured.Compile, "new_compile_seconds": seconds, "constraints": cs.GetNbConstraints(), "r1cs_sha256": after, "byte_identical": true}
		results = append(results, result)
		t.Log(result)
		shapes[i] = nil
		runtime.GC()
	}
	out := os.Getenv("COMMITJL_BENCH_OUTPUT")
	if out == "" {
		t.Fatal("separate output directory required")
	}
	if err = os.Mkdir(out, 0700); err != nil {
		t.Fatal(err)
	}
	data, err := json.MarshalIndent(results, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	if err = os.WriteFile(filepath.Join(out, "equivalence.json"), data, 0600); err != nil {
		t.Fatal(err)
	}
}

func TestFrozenProofPerformance(t *testing.T) {
	source := os.Getenv("COMMITJL_BENCH_SOURCE")
	if source == "" {
		t.Skip("set COMMITJL_BENCH_SOURCE and COMMITJL_BENCH_OUTPUT for explicit short benchmark")
	}
	out := os.Getenv("COMMITJL_BENCH_OUTPUT")
	if out == "" {
		t.Fatal("separate output directory is required")
	}
	if err := os.Mkdir(out, 0700); err != nil {
		t.Fatal(err) // Never overwrite a prior measurement or campaign.
	}
	oldFolder := folder
	folder = filepath.Join(source, "bundle")
	defer func() { folder = oldFolder }()
	p, err := prepared()
	if err != nil {
		t.Fatal(err)
	}
	var in Input
	if err = load(filepath.Join(source, "input.json"), &in); err != nil {
		t.Fatal(err)
	}
	family := os.Getenv("COMMITJL_BENCH_FAMILY")
	var assignment frontend.Circuit
	key := ""
	if family == "full" {
		key = fmt.Sprintf("full_reuse_8_k%d", p.M.K)
		assignment, err = c.FillBoundFullReuseQuery(p, in.ReuseCenters[1], in.Queries[0])
	} else {
		v, e := groupReadPrivate()
		if e != nil {
			t.Fatal(e)
		}
		var task PublicTask
		if e = load(filepath.Join(source, "approved.json"), &task); e != nil {
			t.Fatal(e)
		}
		switch family {
		case "row":
			key = "group_row_compact"
			assignment = c.FillBoundGroupFingerprintRowCompact(p, v.Moments, v.Witness, 0)
		case "linked":
			key = "group_linked_change"
			assignment, err = c.FillBoundGroupLinkedChange(p, v.Moments, v.Witness, task.Centers[0])
		case "switch":
			key = fmt.Sprintf("group_switch_compact_%d", len(task.Queries[0]))
			params, e := c.NewGroupSwitchParameters(p.M, task.Centers[0])
			if e != nil {
				t.Fatal(e)
			}
			assignment, err = c.FillBoundGroupSwitchBatchCompact(p, v.Moments, v.Commitment, params, task.Queries[0])
		default:
			t.Fatal("unknown benchmark family")
		}
	}
	if err != nil {
		t.Fatal(err)
	}
	var record setupCache
	var trust Trust
	if err = load(filepath.Join(folder, ".resume", key+".json"), &record); err != nil {
		t.Fatal(err)
	}
	if err = load(filepath.Join(folder, "trusted.json"), &trust); err != nil {
		t.Fatal(err)
	}
	if record.ModelHash != p.M.Hash().String() || record.ModelHash != trust.ModelHash || record.VKSHA != trust.Keys[key] {
		t.Fatal("source model/key pin mismatch")
	}
	cs := groth16.NewCS(ecc.BN254)
	pk := groth16.NewProvingKey(ecc.BN254)
	vk := groth16.NewVerifyingKey(ecc.BN254)
	paths := []string{filepath.Join(folder, ".resume", key+".r1cs"), filepath.Join(folder, ".resume", key+".pk"), filepath.Join(folder, "keys", key+".vk")}
	var keyReadSeconds float64
	for i, reader := range []io.ReaderFrom{cs, pk, vk} {
		actual, e := digest(paths[i])
		if e != nil || actual != []string{record.CCSSHA, record.PKSHA, record.VKSHA}[i] {
			t.Fatalf("source key file changed: %s", paths[i])
		}
		f, e := os.Open(paths[i])
		if e != nil {
			t.Fatal(e)
		}
		started := time.Now()
		_, e = reader.ReadFrom(f)
		if i == 1 {
			keyReadSeconds = time.Since(started).Seconds()
		}
		f.Close()
		if e != nil {
			t.Fatal(e)
		}
	}
	full, err := frontend.NewWitness(assignment, ecc.BN254.ScalarField())
	if err != nil {
		t.Fatal(err)
	}
	pub, err := full.Public()
	if err != nil {
		t.Fatal(err)
	}
	var publicBytes bytes.Buffer
	if _, err = pub.WriteTo(&publicBytes); err != nil {
		t.Fatal(err)
	}
	if err = os.WriteFile(filepath.Join(out, "public.witness"), publicBytes.Bytes(), 0600); err != nil {
		t.Fatal(err)
	}
	bufferedCache := os.Getenv("COMMITJL_BENCH_BUFFERED_KEY_CACHE") == "1"
	if os.Getenv("COMMITJL_BENCH_KEY_CACHE") == "1" || bufferedCache {
		encoding := "uncompressed"
		if bufferedCache {
			encoding = "compressed-buffered"
		}
		path := filepath.Join(out, "local_"+encoding+".pk")
		f, e := os.Create(path)
		if e != nil {
			t.Fatal(e)
		}
		started := time.Now()
		var n int64
		if bufferedCache {
			buffer := bufio.NewWriterSize(f, binaryIOBufferSize)
			n, e = pk.WriteTo(buffer)
			if e == nil {
				e = buffer.Flush()
			}
		} else {
			n, e = pk.WriteRawTo(f)
		}
		if e == nil {
			e = f.Sync()
		}
		f.Close()
		if e != nil {
			t.Fatal(e)
		}
		writeSeconds := time.Since(started).Seconds()
		pk = nil
		runtime.GC()
		f, e = os.Open(path)
		if e != nil {
			t.Fatal(e)
		}
		pk = groth16.NewProvingKey(ecc.BN254)
		started = time.Now()
		if bufferedCache {
			_, e = pk.ReadFrom(bufio.NewReaderSize(f, binaryIOBufferSize))
		} else {
			_, e = pk.ReadFrom(f)
		}
		readSeconds := time.Since(started).Seconds()
		f.Close()
		if e != nil {
			t.Fatal(e)
		}
		canonical := sha256.New()
		if _, e = pk.WriteTo(canonical); e != nil {
			t.Fatal(e)
		}
		if hex.EncodeToString(canonical.Sum(nil)) != record.PKSHA {
			t.Fatal("cache changed proving key")
		}
		if bufferedCache {
			sha, e := digest(path)
			if e != nil || sha != record.PKSHA {
				t.Fatal("buffered file bytes differ from original compressed key")
			}
		}
		proof, e := groth16.Prove(cs, pk, full)
		if e != nil {
			t.Fatal(e)
		}
		if e = groth16.Verify(proof, vk, pub); e != nil {
			t.Fatal(e)
		}
		info, e := os.Stat(paths[1])
		if e != nil {
			t.Fatal(e)
		}
		result := map[string]any{"key": key, "encoding": encoding, "compressed_unbuffered_read_seconds": keyReadSeconds, "optimized_read_seconds": readSeconds, "optimized_write_seconds": writeSeconds, "compressed_bytes": info.Size(), "optimized_bytes": n, "canonical_key_sha256": record.PKSHA, "canonical_key_identical": true, "fresh_proof_verified": true, "gomaxprocs": runtime.GOMAXPROCS(0), "scope": "Checked key deserialization microbenchmark; not a deployment timing."}
		data, e := json.MarshalIndent(result, "", "  ")
		if e != nil {
			t.Fatal(e)
		}
		if e = os.WriteFile(filepath.Join(out, "key_cache.json"), data, 0600); e != nil {
			t.Fatal(e)
		}
		t.Log(result)
		return
	}
	previous := runtime.GOMAXPROCS(2)
	defer runtime.GOMAXPROCS(previous)
	results := []map[string]any{}
	// Reverse the order on pass two to reduce startup and thermal bias.
	for pass, workers := range [][]int{{2, 4, 8}, {8, 4, 2}} {
		for _, n := range workers {
			runtime.GOMAXPROCS(n)
			runtime.GC()
			for rep := 0; rep < 2; rep++ {
				started := time.Now()
				proof, e := groth16.Prove(cs, pk, full)
				proving := time.Since(started).Seconds()
				if e != nil {
					t.Fatal(e)
				}
				var encoded bytes.Buffer
				if _, e = proof.WriteTo(&encoded); e != nil {
					t.Fatal(e)
				}
				// Fresh serialized replay, not just the in-memory proof object.
				read := groth16.NewProof(ecc.BN254)
				if _, e = read.ReadFrom(bytes.NewReader(encoded.Bytes())); e != nil {
					t.Fatal(e)
				}
				if e = groth16.Verify(read, vk, pub); e != nil {
					t.Fatal(e)
				}
				if e = os.WriteFile(filepath.Join(out, fmt.Sprintf("p%d_cpu%d_r%d.proof", pass, n, rep)), encoded.Bytes(), 0600); e != nil {
					t.Fatal(e)
				}
				startedGC := time.Now()
				runtime.GC() // Same per-proof GC policy as the production binary.
				result := map[string]any{"pass": pass, "workers": n, "rep": rep, "prove_seconds": proving, "gc_seconds": time.Since(startedGC).Seconds(), "wall_seconds": time.Since(started).Seconds(), "verified": true}
				results = append(results, result)
				t.Log(result)
				data, e := json.MarshalIndent(map[string]any{"family": family, "source": source, "key": key, "constraints": cs.GetNbConstraints(), "source_key_receipt": record, "samples": results, "scope": "Fresh proof microbenchmark; existing setup loaded and excluded. Not a full deployment comparison."}, "", "  ")
				if e != nil {
					t.Fatal(e)
				}
				if e = os.WriteFile(filepath.Join(out, "result.json"), data, 0600); e != nil {
					t.Fatal(e)
				}
			}
		}
	}
}
