package main

import (
	c "commitjl/zk/circuit"
	"fmt"
	"math/big"
	"path/filepath"
	"time"
)

func executePairedChange(input string, q, reps, warmup, limit int) (res Result, err error) {
	start := time.Now()
	res.Phase = "paired_change"
	res.Status = "failed"
	defer func() {
		res.Wall = time.Since(start).Seconds()
		if err != nil {
			res.Error = err.Error()
		}
	}()
	p, e := prepared()
	if e != nil {
		err = e
		return
	}
	res.Statement = p.Statement()
	var trust Trust
	if err = load(filepath.Join(folder, "trusted.json"), &trust); err != nil {
		return
	}
	if trust.ModelHash != p.M.Hash().String() || trust.OriginalRoot != p.CX.String() {
		err = fmt.Errorf("prepared data differs from owner pin")
		return
	}
	var prior Result
	if err = load(filepath.Join(folder, "stage1.json"), &prior); err != nil {
		return
	}
	if !prior.Complete || prior.Verified != p.M.N || prior.Statement != res.Statement {
		err = fmt.Errorf("paired change requires complete Stage1 coverage")
		return
	}
	var in Input
	if err = load(input, &in); err != nil {
		return
	}
	if len(in.ComparisonBase) != p.M.K || len(in.ReuseCenters) == 0 {
		err = fmt.Errorf("missing approved comparison base or updates")
		return
	}
	knew := len(in.ReuseCenters[0])
	queries := [][]int{}
	for _, v := range in.Queries {
		if len(v) == q {
			queries = append(queries, v)
		}
	}
	if len(queries) == 0 {
		err = fmt.Errorf("no matching paired query batch")
		return
	}
	parameters := make([]*c.PairedParameters, len(in.ReuseCenters))
	publicStart := time.Now()
	for i, v := range in.ReuseCenters {
		if len(v) != knew {
			err = fmt.Errorf("changing paired center count")
			return
		}
		if parameters[i], e = c.PairedChangeParameters(p.M, in.ComparisonBase, v); e != nil {
			err = e
			return
		}
	}
	res.Native = time.Since(publicStart).Seconds()
	ccs, pk, vk, e := setup(c.NewBoundPairedChange(p.M, q, p.M.K, knew), fmt.Sprintf("paired_change_%d_k%d", q, knew), limit, &res)
	if e != nil {
		err = e
		return
	}
	if ccs == nil {
		return
	}
	for s := range in.ReuseCenters {
		for id, indices := range queries {
			t := time.Now()
			w, _, e := c.FillBoundPairedChangePrepared(p, parameters[s], indices)
			if e != nil {
				err = e
				return
			}
			nativeSeconds := time.Since(t).Seconds()
			extra := 0
			if s == 0 && id == 0 {
				extra = warmup
			}
			file := fmt.Sprintf("paired_change/s%03d_q%d_%04d.proof", s, q, id)
			sample, e := prove(ccs, pk, vk, w, file, reps, extra)
			if e != nil {
				err = e
				return
			}
			sample.ID, sample.Set, sample.Indices, sample.NativeAnswer = id, s, indices, nativeSeconds
			sample.ChangeLower, sample.ChangeUpper = w.Lower.(*big.Int).String(), w.Upper.(*big.Int).String()
			res.Samples = append(res.Samples, sample)
			res.Verified++
			if err = save(fmt.Sprintf("paired_change_%d_progress.json", q), res); err != nil {
				return
			}
		}
	}
	res.Status = "complete"
	res.Complete = true
	return
}
