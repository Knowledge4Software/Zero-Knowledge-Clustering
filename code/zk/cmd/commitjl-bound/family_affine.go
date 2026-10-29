package main

import (
	c "commitjl/zk/circuit"
	"fmt"
	"math/big"
	"path/filepath"
	"time"
)

func executeFamilyAffine(input string, q, reps, warmup, limit int) (res Result, err error) {
	start := time.Now()
	res.Phase = "family_affine"
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
	if p.M.Method != "affine" {
		err = fmt.Errorf("family baseline requires affine bank model")
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
		err = fmt.Errorf("complete consistent affine preprocessing required")
		return
	}
	var in Input
	if err = load(input, &in); err != nil {
		return
	}
	if len(in.ReuseCenters) == 0 {
		err = fmt.Errorf("no approved public center sets")
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
			err = fmt.Errorf("changing center count within one query shape")
			return
		}
		if _, e = c.FamilyAffinePositions(p.M, centers); e != nil {
			err = e
			return
		}
	}
	ccs, pk, vk, e := setup(c.NewBoundFamilyAffineQuery(p.M, q, knew), fmt.Sprintf("family_affine_%d_k%d", q, knew), limit, &res)
	if e != nil {
		err = e
		return
	}
	if ccs == nil {
		return
	}
	for s, centers := range in.ReuseCenters {
		for id, indices := range queries {
			t = time.Now()
			w, e := c.FillBoundFamilyAffineQuery(p, centers, indices)
			if e != nil {
				err = e
				return
			}
			native := time.Since(t).Seconds()
			extra := 0
			if s == 0 && id == 0 {
				extra = warmup
			}
			file := fmt.Sprintf("family_affine/s%03d_q%d_%04d.proof", s, q, id)
			sample, e := prove(ccs, pk, vk, w, file, reps, extra)
			if e != nil {
				err = e
				return
			}
			sample.ID, sample.Set, sample.NativeAnswer = id, s, native
			sample.Indices = indices
			sample.Assign = make([]int, len(indices))
			for i, v := range w.Assign {
				sample.Assign[i] = v.(int)
			}
			sample.Psi = w.Psi.(*big.Int).String()
			res.Samples = append(res.Samples, sample)
			res.Verified++
			if err = save(fmt.Sprintf("family_affine_%d_progress.json", q), res); err != nil {
				return
			}
		}
	}
	res.Status = "complete"
	res.Complete = true
	return
}
