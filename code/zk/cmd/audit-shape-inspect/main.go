// Compile-only resource preflight. Generates no setup, keys or proofs.
package main

import (
	c "commitjl/zk/circuit"
	"encoding/json"
	"flag"
	"fmt"
	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/frontend/cs/r1cs"
	"os"
	"time"
)

func main() {
	source := flag.String("input", "", "frozen predictor input")
	auxiliary := flag.String("auxiliary", "", "frozen certificate basis")
	output := flag.String("out", "", "new receipt")
	guard := flag.Int("max-constraints", 1600000, "same per-shape campaign guard")
	flag.Parse()
	fail := func(e error) { fmt.Fprintln(os.Stderr, e); os.Exit(1) }
	var in struct {
		Model c.BoundModel `json:"model"`
	}
	b, e := os.ReadFile(*source)
	if e != nil {
		fail(e)
	}
	if e = json.Unmarshal(b, &in); e != nil {
		fail(e)
	}
	var aux struct {
		Basis [][]int64 `json:"certificate_basis"`
	}
	b, e = os.ReadFile(*auxiliary)
	if e != nil {
		fail(e)
	}
	if e = json.Unmarshal(b, &aux); e != nil {
		fail(e)
	}
	m := in.Model
	m.Basis = aux.Basis
	if e = m.Validate(); e != nil {
		fail(e)
	}
	full := m
	full.Method = "full"
	full.Basis = nil
	names := []string{"reusable_exact_full_8", "compact_preprocess_row", "linked_assigned_change", "compact_switch_16", "exact_refinement_1"}
	shapes := []frontend.Circuit{c.NewBoundFullReuseQuery(full, 8, m.K), c.NewBoundGroupFingerprintRowCompact(m), c.NewBoundGroupLinkedChange(m), c.NewBoundGroupSwitchBatchCompact(m, 16), c.NewBoundGroupRefinement(m, 1)}
	rows := []map[string]any{}
	for i, shape := range shapes {
		tick := time.Now()
		cs, e := frontend.Compile(ecc.BN254.ScalarField(), r1cs.NewBuilder, shape)
		if e != nil {
			fail(e)
		}
		rows = append(rows, map[string]any{"family": names[i], "constraints": cs.GetNbConstraints(), "compile_seconds": time.Since(tick).Seconds(), "within_guard": cs.GetNbConstraints() <= *guard})
	}
	receipt := map[string]any{"complete": true, "proofs_generated": 0, "guard": *guard, "n": m.N, "k": m.K, "shapes": rows}
	b, e = json.MarshalIndent(receipt, "", "  ")
	if e != nil {
		fail(e)
	}
	f, e := os.OpenFile(*output, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
	if e != nil {
		fail(e)
	}
	if _, e = f.Write(b); e != nil {
		fail(e)
	}
	if e = f.Close(); e != nil {
		fail(e)
	}
}
