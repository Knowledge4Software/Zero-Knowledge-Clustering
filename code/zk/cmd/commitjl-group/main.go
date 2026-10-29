// commitjl-group is a bounded full-catalog group-moment proof benchmark.
// The verify mode reads approved model/root/key pins and tasks, never row data.
package main

import (
	c "commitjl/zk/circuit"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark/backend/groth16"
	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/frontend/cs/r1cs"
	"github.com/consensys/gnark/logger"
	"io"
	"math/big"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"runtime/debug"
	"time"
)

type Input struct {
	Model                   c.BoundModel `json:"model"`
	X                       [][]int64    `json:"x"`
	Sets                    [][][]int64  `json:"sets"`
	SaltX, SaltZ, SaltGroup string
}
type Pins struct {
	Model       c.BoundModel
	Group, Full c.BoundStatement
	GroupRoot   string
	Capacity    int
	KeySHA      map[string]string
}
type Sample struct {
	Set                     int
	Change, Cost, Proof     string
	Indices                 []int
	Active                  []bool
	Witness, Prove, Verify  float64
	ProofBytes, PublicBytes int64
}
type Phase struct {
	Name             string
	Constraints      int
	Compile, Setup   float64
	PKBytes, VKBytes int64
	Samples          []Sample
}
type Summary struct {
	Complete                             bool
	Native, WorkflowWall, FreshReplay    float64
	GroupNative, FullNative              float64
	Phases                               []Phase
	GroupTotal, FullTotal, FullOverGroup float64
	FreshVerified                        int
	TamperRejected                       bool
	Scope                                string
}
type Replay struct {
	Complete bool
	Verified int
	Seconds  float64
	Scope    string
}

var folder string
var resume bool

func check(err error) {
	if err != nil {
		panic(err)
	}
}
func load(name string, out any) { b, e := os.ReadFile(name); check(e); check(json.Unmarshal(b, out)) }
func save(name string, value any) {
	b, e := json.MarshalIndent(value, "", "  ")
	check(e)
	check(os.WriteFile(filepath.Join(folder, name), b, 0600))
}
func digest(name string) string {
	b, e := os.ReadFile(name)
	check(e)
	v := sha256.Sum256(b)
	return hex.EncodeToString(v[:])
}
func number(v string) *big.Int {
	x, ok := new(big.Int).SetString(v, 10)
	if !ok {
		panic("invalid number")
	}
	return x
}
func store(name string, w io.WriterTo) int64 {
	f, e := os.Create(filepath.Join(folder, name))
	check(e)
	defer f.Close()
	n, e := w.WriteTo(f)
	check(e)
	return n
}
func fullModel(m c.BoundModel) c.BoundModel {
	m.Method = "full"
	m.Basis = nil
	m.TailWidth = 0
	m.TailRows = nil
	m.TailSigns = nil
	return m
}
func indices(n int) []int {
	r := make([]int, n)
	for i := range r {
		r[i] = i
	}
	return r
}

func measure(name string, shape frontend.Circuit, witnesses []frontend.Circuit, sets []int, limit int, pins *Pins, report *Summary) {
	if resume {
		if _, err := os.Stat(filepath.Join(folder, name+".json")); err == nil {
			var saved Phase
			load(filepath.Join(folder, name+".json"), &saved)
			if len(saved.Samples) == len(witnesses) && pins.KeySHA[name] != "" {
				if digest(filepath.Join(folder, name+".vk")) != pins.KeySHA[name] {
					panic("saved key differs from pin")
				}
				vk := groth16.NewVerifyingKey(ecc.BN254)
				f, e := os.Open(filepath.Join(folder, name+".vk"))
				check(e)
				_, e = vk.ReadFrom(f)
				check(e)
				check(f.Close())
				for i, w := range witnesses {
					if saved.Samples[i].Set != sets[i] || saved.Samples[i].Proof != fmt.Sprintf("%s_%02d.proof", name, i) {
						panic("saved task coverage differs")
					}
					all, e := frontend.NewWitness(w, ecc.BN254.ScalarField())
					check(e)
					pub, e := all.Public()
					check(e)
					proof := groth16.NewProof(ecc.BN254)
					f, e = os.Open(filepath.Join(folder, saved.Samples[i].Proof))
					check(e)
					_, e = proof.ReadFrom(f)
					check(e)
					check(f.Close())
					check(groth16.Verify(proof, vk, pub))
				}
				report.Phases = append(report.Phases, saved)
				save("summary_progress.json", report)
				fmt.Printf("reused and verified %s: %d proofs\n", name, len(saved.Samples))
				return
			}
		}
	}
	phase := Phase{Name: name}
	t := time.Now()
	cs, e := frontend.Compile(ecc.BN254.ScalarField(), r1cs.NewBuilder, shape)
	check(e)
	phase.Compile = time.Since(t).Seconds()
	phase.Constraints = cs.GetNbConstraints()
	fmt.Printf("compiled %s constraints=%d\n", name, phase.Constraints)
	if phase.Constraints > limit {
		save("excluded_"+name+".json", phase)
		panic("constraint guard exceeded; no setup started")
	}
	t = time.Now()
	pk, vk, e := groth16.Setup(cs)
	check(e)
	phase.Setup = time.Since(t).Seconds()
	phase.VKBytes = store(name+".vk", vk)
	phase.PKBytes, e = pk.WriteTo(io.Discard)
	check(e)
	pins.KeySHA[name] = digest(filepath.Join(folder, name+".vk"))
	save("pins.json", pins)
	for i, w := range witnesses {
		t = time.Now()
		all, e := frontend.NewWitness(w, ecc.BN254.ScalarField())
		check(e)
		pub, e := all.Public()
		check(e)
		sample := Sample{Set: sets[i], Witness: time.Since(t).Seconds(), Proof: fmt.Sprintf("%s_%02d.proof", name, i)}
		switch v := w.(type) {
		case *c.BoundGroupQuery:
			sample.Change = v.Change.(*big.Int).String()
			for a := range v.Indices {
				sample.Indices = append(sample.Indices, v.Indices[a].(int))
				sample.Active = append(sample.Active, v.Active[a].(int) == 1)
			}
		case *c.BoundFullReuseQuery:
			sample.Cost = v.Psi.(*big.Int).String()
		}
		t = time.Now()
		proof, e := groth16.Prove(cs, pk, all)
		check(e)
		sample.Prove = time.Since(t).Seconds()
		t = time.Now()
		check(groth16.Verify(proof, vk, pub))
		sample.Verify = time.Since(t).Seconds()
		sample.ProofBytes = store(sample.Proof, proof)
		sample.PublicBytes, e = pub.WriteTo(io.Discard)
		check(e)
		phase.Samples = append(phase.Samples, sample)
		save(name+".json", phase)
		fmt.Printf("proved %s set=%d prove=%.3fs verify=%.3fs\n", name, sample.Set, sample.Prove, sample.Verify)
	}
	report.Phases = append(report.Phases, phase)
	save("summary_progress.json", report)
	// Release proving keys/constraint memory before compiling the next shape.
	cs = nil
	pk = nil
	vk = nil
	runtime.GC()
	debug.FreeOSMemory()
}

func replay(pins Pins, tasks [][][]int64, negative bool) Replay {
	start := time.Now()
	r := Replay{Scope: "Fresh public-only Groth16 verification; model/root/task/key pins required independently approved in deployment."}
	phases := []string{"group_prepare", "group_query", "full"}
	var queries, full []Sample
	for _, name := range phases {
		if digest(filepath.Join(folder, name+".vk")) != pins.KeySHA[name] {
			panic("key hash mismatch")
		}
		vk := groth16.NewVerifyingKey(ecc.BN254)
		f, e := os.Open(filepath.Join(folder, name+".vk"))
		check(e)
		_, e = vk.ReadFrom(f)
		check(e)
		check(f.Close())
		var phase Phase
		load(filepath.Join(folder, name+".json"), &phase)
		expected := len(tasks)
		if name == "group_prepare" {
			expected = 1
		}
		if name == "group_query" {
			expected = len(tasks) - 1
		}
		if len(phase.Samples) != expected {
			panic("incomplete phase coverage")
		}
		for i, s := range phase.Samples {
			var public frontend.Circuit
			switch name {
			case "group_prepare":
				if s.Set != -1 {
					panic("invalid preprocess task")
				}
				public, e = c.GroupPreprocessPublic(pins.Model, pins.Group, pins.GroupRoot)
			case "group_query":
				if s.Set != i+1 || len(s.Indices) != pins.Capacity {
					panic("query task omitted/duplicated")
				}
				public, e = c.GroupQueryPublic(pins.Model, pins.Group, pins.GroupRoot, tasks[s.Set], s.Indices, s.Active, s.Change)
				queries = append(queries, s)
			case "full":
				if s.Set != i {
					panic("Full task omitted/duplicated")
				}
				public, e = c.BoundFullReusePublic(fullModel(pins.Model), pins.Full, tasks[s.Set], indices(pins.Model.N), s.Cost)
				full = append(full, s)
			}
			check(e)
			witness, e := frontend.NewWitness(public, ecc.BN254.ScalarField(), frontend.PublicOnly())
			check(e)
			if s.Proof != fmt.Sprintf("%s_%02d.proof", name, i) {
				panic("unexpected proof filename")
			}
			proof := groth16.NewProof(ecc.BN254)
			f, e = os.Open(filepath.Join(folder, s.Proof))
			check(e)
			_, e = proof.ReadFrom(f)
			check(e)
			check(f.Close())
			check(groth16.Verify(proof, vk, witness))
			r.Verified++
			if negative && name == "group_query" && i == 0 {
				changed := new(big.Int).Add(number(s.Change), big.NewInt(1))
				bad, e := c.GroupQueryPublic(pins.Model, pins.Group, pins.GroupRoot, tasks[s.Set], s.Indices, s.Active, changed.String())
				check(e)
				bw, e := frontend.NewWitness(bad, ecc.BN254.ScalarField(), frontend.PublicOnly())
				check(e)
				if groth16.Verify(proof, vk, bw) == nil {
					panic("false delta accepted")
				}
				changedCenters := make([][]int64, len(tasks[s.Set]))
				for j, row := range tasks[s.Set] {
					changedCenters[j] = append([]int64{}, row...)
				}
				if changedCenters[0][0] > 0 {
					changedCenters[0][0]--
				} else {
					changedCenters[0][0]++
				}
				bad, e = c.GroupQueryPublic(pins.Model, pins.Group, pins.GroupRoot, changedCenters, s.Indices, s.Active, s.Change)
				check(e)
				bw, e = frontend.NewWitness(bad, ecc.BN254.ScalarField(), frontend.PublicOnly())
				check(e)
				if groth16.Verify(proof, vk, bw) == nil {
					panic("changed approved centers accepted")
				}
			}
		}
	}
	for _, s := range queries {
		delta := new(big.Int).Sub(number(full[s.Set].Cost), number(full[0].Cost))
		if delta.Cmp(number(s.Change)) != 0 {
			panic("group/Full audited outputs disagree")
		}
	}
	r.Complete = true
	r.Seconds = time.Since(start).Seconds()
	return r
}
func main() {
	input := flag.String("input", "", "private benchmark owner input")
	out := flag.String("out", "", "output folder")
	verify := flag.Bool("verify", false, "public-only replay mode")
	resumeFlag := flag.Bool("resume", false, "reuse complete saved phases after checking proofs against reconstructed witnesses")
	limit := flag.Int("max-constraints", 1600000, "setup guard")
	flag.Parse()
	folder = *out
	resume = *resumeFlag
	runtime.GOMAXPROCS(2)
	debug.SetMemoryLimit(1200 << 20)
	logger.Disable()
	if *verify {
		var p Pins
		var tasks [][][]int64
		load(filepath.Join(folder, "pins.json"), &p)
		load(filepath.Join(folder, "tasks.json"), &tasks)
		r := replay(p, tasks, true)
		save("fresh_replay.json", r)
		fmt.Printf("public-only replay: %d proofs PASS\n", r.Verified)
		return
	}
	if folder == "" || *input == "" {
		panic("input and output required")
	}
	if _, e := os.Stat(folder); !os.IsNotExist(e) && !resume {
		panic("output must not exist")
	}
	check(os.MkdirAll(folder, 0700))
	start := time.Now()
	var in Input
	load(*input, &in)
	if len(in.Sets) < 2 {
		panic("baseline and update tasks required")
	}
	report := Summary{Scope: "Single-repetition bounded full-catalog prototype; includes complete authenticated group preparation and exact Full baseline+updates. Fixed correction capacity; no production-scale speed claim."}
	t := time.Now()
	p, e := c.PrepareBound(in.Model, in.X, number(in.SaltX), number(in.SaltZ))
	check(e)
	g, e := c.PrepareGroupMoments(p, number(in.SaltGroup))
	check(e)
	report.GroupNative = time.Since(t).Seconds()
	t = time.Now()
	fp, e := c.PrepareBound(fullModel(in.Model), in.X, number(in.SaltX), number(in.SaltZ))
	check(e)
	if p.CX.Cmp(fp.CX) != 0 {
		panic("original roots differ")
	}
	report.FullNative = time.Since(t).Seconds()
	t = time.Now()
	capacity := 0
	for _, centers := range in.Sets[1:] {
		ids, _, e := c.GroupCorrectionIndices(p, g, centers)
		check(e)
		if len(ids) > capacity {
			capacity = len(ids)
		}
	}
	report.GroupNative += time.Since(t).Seconds()
	pins := Pins{Model: in.Model, Group: p.Statement(), Full: fp.Statement(), GroupRoot: g.Commitment.String(), Capacity: capacity, KeySHA: map[string]string{}}
	if resume {
		var old Pins
		var tasks [][][]int64
		load(filepath.Join(folder, "pins.json"), &old)
		load(filepath.Join(folder, "tasks.json"), &tasks)
		if old.Model.Hash().Cmp(pins.Model.Hash()) != 0 || old.Group != pins.Group || old.Full != pins.Full || old.GroupRoot != pins.GroupRoot || old.Capacity != pins.Capacity || !reflect.DeepEqual(tasks, in.Sets) {
			panic("resume inputs differ from saved pins/tasks")
		}
		pins.KeySHA = old.KeySHA
		report.Scope += " Resumed from independently rechecked completed phases; totals reconstruct successful components, and WorkflowWall measures this invocation only."
	}
	report.Native = report.GroupNative + report.FullNative
	save("pins.json", pins)
	save("tasks.json", in.Sets)
	t = time.Now()
	prep := c.FillBoundGroupPreprocess(p, g)
	report.GroupNative += time.Since(t).Seconds()
	measure("group_prepare", c.NewBoundGroupPreprocess(p.M), []frontend.Circuit{prep}, []int{-1}, *limit, &pins, &report)
	groupW, fullW := []frontend.Circuit{}, []frontend.Circuit{}
	groupSets, fullSets := []int{}, []int{}
	for i, centers := range in.Sets {
		t = time.Now()
		fw, e := c.FillBoundFullReuseQuery(fp, centers, indices(p.M.N))
		check(e)
		report.FullNative += time.Since(t).Seconds()
		fullW = append(fullW, fw)
		fullSets = append(fullSets, i)
		if i > 0 {
			t = time.Now()
			w, e := c.FillBoundGroupQuery(p, g, centers, capacity)
			check(e)
			report.GroupNative += time.Since(t).Seconds()
			groupW = append(groupW, w)
			groupSets = append(groupSets, i)
		}
	}
	measure("group_query", c.NewBoundGroupQuery(p.M, capacity), groupW, groupSets, *limit, &pins, &report)
	measure("full", c.NewBoundFullReuseQuery(fp.M, fp.M.N, fp.M.K), fullW, fullSets, *limit, &pins, &report)
	replayed := replay(pins, in.Sets, true)
	report.FreshReplay = replayed.Seconds
	report.FreshVerified = replayed.Verified
	report.TamperRejected = true
	report.GroupTotal = report.GroupNative
	report.FullTotal = report.FullNative
	for _, phase := range report.Phases {
		total := phase.Compile + phase.Setup
		for _, s := range phase.Samples {
			total += s.Witness + s.Prove + s.Verify
		}
		if phase.Name == "full" {
			report.FullTotal += total
		} else {
			report.GroupTotal += total
		}
	}
	report.FullOverGroup = report.FullTotal / report.GroupTotal
	report.WorkflowWall = time.Since(start).Seconds()
	report.Complete = true
	save("summary.json", report)
	save("fresh_replay.json", replayed)
	fmt.Printf("COMPLETE group=%.2fs Full=%.2fs ratio=%.3f verified=%d\n", report.GroupTotal, report.FullTotal, report.FullOverGroup, report.FreshVerified)
}
