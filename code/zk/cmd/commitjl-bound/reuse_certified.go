package main

import (
	c "commitjl/zk/circuit"
	"fmt"
	"math/big"
	"path/filepath"
	"time"
)

func executeCertifiedReuse(input string, q, reps, warmup, limit int) (res Result, err error) {
	start := time.Now()
	res.Phase = "reuse_certified"
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
		err = fmt.Errorf("prepared input differs from trusted pin")
		return
	}
	var previous Result
	if err = load(filepath.Join(folder, "stage1.json"), &previous); err != nil {
		return
	}
	if !previous.Complete || previous.Verified != p.M.N || previous.Statement != res.Statement {
		err = fmt.Errorf("complete consistent Stage 1 coverage required")
		return
	}
	var in Input
	if err = load(input, &in); err != nil {
		return
	}
	if len(in.ReuseCenters) == 0 {
		err = fmt.Errorf("no public center sets")
		return
	}
	knew := len(in.ReuseCenters[0])
	queries := [][]int{}
	for _, indices := range in.Queries {
		if len(indices) == q {
			queries = append(queries, indices)
		}
	}
	if len(queries) == 0 {
		err = fmt.Errorf("no matching query batches")
		return
	}
	for _, centers := range in.ReuseCenters {
		if len(centers) != knew {
			err = fmt.Errorf("changing center count")
			return
		}
		if _, _, e = c.ReuseCenterStats(p.M, centers); e != nil {
			err = e
			return
		}
	}
	ccs, pk, vk, e := setup(c.NewBoundCertifiedReuseQuery(p.M, q, knew), fmt.Sprintf("reuse_certified_%d_k%d", q, knew), limit, &res)
	if e != nil {
		err = e
		return
	}
	if ccs == nil {
		return
	}
	for s, centers := range in.ReuseCenters {
		heads, offs, e := c.ReuseCenterStats(p.M, centers)
		if e != nil {
			err = e
			return
		}
		for id, indices := range queries {
			t = time.Now()
			w, native, e := c.FillBoundCertifiedReuseQuery(p, heads, offs, indices)
			if e != nil {
				err = e
				return
			}
			nativeSeconds := time.Since(t).Seconds()
			extra := 0
			if s == 0 && id == 0 {
				extra = warmup
			}
			file := fmt.Sprintf("reuse_certified/s%03d_q%d_%04d.proof", s, q, id)
			sample, e := prove(ccs, pk, vk, w, file, reps, extra)
			if e != nil {
				err = e
				return
			}
			sample.ID, sample.Set, sample.NativeAnswer = id, s, nativeSeconds
			sample.Indices, sample.Assign = indices, native.Labels
			sample.Psi, sample.Margin = w.Query.Psi.(*big.Int).String(), w.Query.MarginSum.(*big.Int).String()
			sample.CertifiedCount = native.Count
			sample.CertifiedFlags = native.Flags
			sample.CostLower, sample.CostUpper = native.Lower.String(), native.Upper.String()
			res.Samples = append(res.Samples, sample)
			res.Verified++
			if err = save(fmt.Sprintf("reuse_certified_%d_progress.json", q), res); err != nil {
				return
			}
		}
	}
	res.Status = "complete"
	res.Complete = true
	return
}
