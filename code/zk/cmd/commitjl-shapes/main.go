// Compile-only resource checks. Constraint counts are not proving timings.
package main

import (
	c "commitjl/zk/circuit"
	"encoding/json"
	"flag"
	"fmt"
	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/frontend/cs/r1cs"
	"github.com/consensys/gnark/logger"
	"os"
	"runtime"
	"runtime/debug"
	"time"
)

type Record struct {
	Rank, Constraints int
	N, D, K           int
	Phase             string
	CompileSeconds    float64
}

func main() {
	input := flag.String("input", "", "JSON array of approved models")
	output := flag.String("out", "", "JSON output")
	tradeoff := flag.Bool("tradeoff", false, "compile stage1/relative q16 or Full q8 for the laptop tradeoff")
	fingerprint := flag.Bool("fingerprint", false, "compile experimental group-moment fingerprint row and link shapes")
	flag.Parse()
	runtime.GOMAXPROCS(2)
	debug.SetMemoryLimit(800 << 20)
	logger.Disable()
	data, err := os.ReadFile(*input)
	if err != nil {
		panic(err)
	}
	var models []c.BoundModel
	if err = json.Unmarshal(data, &models); err != nil {
		panic(err)
	}
	records := []Record{}
	for _, m := range models {
		if *fingerprint && m.Method == "full" {
			continue
		}
		phases := []string{"stage1", "paired_query_q4"}
		if *tradeoff {
			phases = []string{"stage1", "relative_query_q16"}
			if m.Method == "full" {
				phases = []string{"full_query_q8"}
			}
		}
		if *fingerprint {
			phases = []string{"stage1", "fingerprint_row", "fingerprint_link", "group_assigned_change", "group_switch_q16"}
		}
		for _, phase := range phases {
			var shape frontend.Circuit = c.NewBoundPreprocess(m)
			switch phase {
			case "paired_query_q4":
				shape = c.NewBoundPairedChange(m, 4, m.K, m.K)
			case "relative_query_q16":
				shape = c.NewBoundRelativeReuse(m, 16, m.K)
			case "full_query_q8":
				shape = c.NewBoundFullReuseQuery(m, 8, m.K)
			case "fingerprint_row":
				shape = c.NewBoundGroupFingerprintRow(m)
			case "fingerprint_link":
				shape = c.NewBoundGroupFingerprintLink(m)
			case "group_assigned_change":
				shape = c.NewBoundGroupAssignedChange(m)
			case "group_switch_q16":
				shape = c.NewBoundGroupSwitchBatch(m, 16)
			}
			start := time.Now()
			cs, e := frontend.Compile(ecc.BN254.ScalarField(), r1cs.NewBuilder, shape)
			if e != nil {
				panic(e)
			}
			row := Record{Rank: m.Width(), Constraints: cs.GetNbConstraints(), N: m.N, D: m.D, K: m.K, Phase: phase, CompileSeconds: time.Since(start).Seconds()}
			records = append(records, row)
			fmt.Printf("rank=%d phase=%s constraints=%d\n", row.Rank, row.Phase, row.Constraints)
			cs = nil
			runtime.GC()
			debug.FreeOSMemory()
		}
	}
	b, err := json.MarshalIndent(struct {
		Scope   string
		Records []Record
	}{"Compile only; no setup/proof speed or full-pipeline timing claim.", records}, "", "  ")
	if err != nil {
		panic(err)
	}
	if err = os.WriteFile(*output, b, 0600); err != nil {
		panic(err)
	}
}
