package main

import (
	c "commitjl/zk/circuit"
	"fmt"
	"path/filepath"
	"time"
)

// Use a NEW/copied bundle: setup pins the geometry key alongside the original
// keys. The running older campaign and its immutable binary are never changed.
func executeGeometry(input string, reps, warmup, limit int) (res Result, err error) {
	start := time.Now()
	res.Phase = "geometry"
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
		err = fmt.Errorf("prepared input differs from owner pin")
		return
	}
	if p.M.Method != "full" {
		var prior Result
		if err = load(filepath.Join(folder, "stage1.json"), &prior); err != nil {
			return
		}
		if !prior.Complete || prior.Verified != p.M.N {
			err = fmt.Errorf("complete Stage1 coverage required")
			return
		}
	}
	ccs, pk, vk, e := setup(c.NewBoundGeometry(p.M), "geometry", limit, &res)
	if e != nil {
		err = e
		return
	}
	if ccs == nil {
		return
	}
	var in Input
	if err = load(input, &in); err != nil {
		return
	}
	for _, indices := range in.Queries {
		if len(indices) != 2 {
			continue
		}
		t = time.Now()
		w, e := c.FillBoundGeometry(p, indices)
		if e != nil {
			err = e
			return
		}
		native := time.Since(t).Seconds()
		id := len(res.Samples)
		file := fmt.Sprintf("geometry/pair_%04d.proof", id)
		sample, e := prove(ccs, pk, vk, w, file, reps, warmup)
		if e != nil {
			err = e
			return
		}
		sample.ID = id
		sample.Indices = indices
		sample.Psi = w.Distance.(interface{ String() string }).String()
		sample.NativeAnswer = native
		res.Samples = append(res.Samples, sample)
		res.Verified++
		if err = save("geometry_progress.json", res); err != nil {
			return
		}
	}
	if len(res.Samples) == 0 {
		err = fmt.Errorf("no geometry pairs supplied")
		return
	}
	res.Status = "complete"
	res.Complete = true
	return
}
