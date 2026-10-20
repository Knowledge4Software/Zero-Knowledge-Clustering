package main

import (
	"fmt"
	"math/big"
	"path/filepath"
	"time"

	c "commitjl/zk/circuit"
)

// executeReuse proves fixed-size subset queries for several new public centre
// sets against one completed Stage-1 catalog. One compile/setup serves all sets.
func executeReuse(input string, q, reps, warmup, limit int) (res Result, err error) {
	start := time.Now()
	res.Phase = "reuse"
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
	var previous Result
	if err = load(filepath.Join(folder, "stage1.json"), &previous); err != nil {
		return
	}
	if !previous.Complete || previous.Verified != p.M.N {
		err = fmt.Errorf("complete preprocessing coverage required")
		return
	}
	var in Input
	if err = load(input, &in); err != nil {
		return
	}
	if len(in.ReuseCenters) == 0 {
		err = fmt.Errorf("no reuse centre sets")
		return
	}
	knew := len(in.ReuseCenters[0])
	type stats struct {
		heads [][]*big.Int
		offs  []*big.Int
	}
	sets := make([]stats, len(in.ReuseCenters))
	for s, centres := range in.ReuseCenters {
		if len(centres) != knew {
			err = fmt.Errorf("all reuse centre sets must have the same size")
			return
		}
		sets[s].heads, sets[s].offs, err = c.ReuseCenterStats(p.M, centres)
		if err != nil {
			return
		}
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
	ccs, pk, vk, e := setup(c.NewBoundReuseQuery(p.M, q, knew), fmt.Sprintf("reuse_%d_k%d", q, knew), limit, &res)
	if e != nil {
		err = e
		return
	}
	if ccs == nil {
		return
	}
	for s, st := range sets {
		for id, indices := range queries {
			nativeStart := time.Now()
			w, e := c.FillBoundReuseQuery(p, st.heads, st.offs, indices)
			if e != nil {
				err = e
				return
			}
			nativeSeconds := time.Since(nativeStart).Seconds()
			labels := make([]int, len(indices))
			for i, v := range w.Assign {
				labels[i] = v.(int)
			}
			extra := 0
			if s == 0 && id == 0 {
				extra = warmup
			}
			file := fmt.Sprintf("reuse/s%03d_q%d_%04d.proof", s, q, id)
			sample, e := prove(ccs, pk, vk, w, file, reps, extra)
			if e != nil {
				err = e
				return
			}
			sample.ID, sample.Set, sample.NativeAnswer = id, s, nativeSeconds
			sample.Indices, sample.Assign = indices, labels
			sample.Psi, sample.Margin = w.Psi.(*big.Int).String(), w.MarginSum.(*big.Int).String()
			res.Samples = append(res.Samples, sample)
			res.Verified++
			if err = save(fmt.Sprintf("reuse_%d_progress.json", q), res); err != nil {
				return
			}
		}
	}
	res.Complete = true
	res.Status = "complete"
	return
}
