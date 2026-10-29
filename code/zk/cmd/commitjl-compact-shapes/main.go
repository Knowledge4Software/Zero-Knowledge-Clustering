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
	"math"
	"os"
	"runtime"
	"runtime/debug"
	"time"
)

func main() {
	input := flag.String("input", "", "frozen input")
	out := flag.String("out", "", "report")
	flag.Parse()
	logger.Disable()
	runtime.GOMAXPROCS(2)
	debug.SetMemoryLimit(800 << 20)
	data, err := os.ReadFile(*input)
	if err != nil {
		panic(err)
	}
	var in struct {
		Model c.BoundModel `json:"model"`
	}
	if err = json.Unmarshal(data, &in); err != nil {
		panic(err)
	}
	original := in.Model
	records := []map[string]any{}
	for _, bits := range []int{8, 10, 14} {
		m := original
		m.BasisScale = int64(1) << bits
		m.Basis = make([][]int64, len(original.Basis))
		for i, row := range original.Basis {
			m.Basis[i] = make([]int64, len(row))
			for j, v := range row {
				m.Basis[i][j] = int64(math.RoundToEven(float64(v) * float64(m.BasisScale) / float64(original.BasisScale)))
			}
		}
		for _, q := range []int{0, 8, 16, 32, 64} {
			var shape frontend.Circuit
			name := "switch"
			if q == 0 {
				shape = c.NewBoundGroupLinkedChange(m)
				name = "linked_change"
			} else {
				shape = c.NewBoundGroupSwitchBatchCompact(m, q)
			}
			start := time.Now()
			cs, err := frontend.Compile(ecc.BN254.ScalarField(), r1cs.NewBuilder, shape)
			if err != nil {
				panic(err)
			}
			r := map[string]any{"bits": bits, "q": q, "phase": name, "constraints": cs.GetNbConstraints(), "compile_seconds": time.Since(start).Seconds()}
			records = append(records, r)
			fmt.Println(r)
			cs = nil
			runtime.GC()
			debug.FreeOSMemory()
			payload, _ := json.MarshalIndent(map[string]any{"complete": false, "scope": "compile only, no measured speed", "records": records}, "", "  ")
			if err = os.WriteFile(*out, payload, 0600); err != nil {
				panic(err)
			}
		}
	}
	payload, _ := json.MarshalIndent(map[string]any{"complete": true, "scope": "compile only, no measured speed", "records": records}, "", "  ")
	if err = os.WriteFile(*out, payload, 0600); err != nil {
		panic(err)
	}
}
