package main

import (
	"fmt"
	"path/filepath"
	"reflect"
	"strings"

	c "commitjl/zk/circuit"
	"github.com/consensys/gnark/frontend"
)

// The expected task is supplied independently of prover phase metadata. It
// contains only intended public centers and approved ordered query subsets.
type PublicTask struct {
	Centers [][][]int64 `json:"reuse_centers"`
	Queries [][]int     `json:"queries"`
}

// Every requested batch must have a pinned query key for every changing-center
// relation being verified. Individual phase verification filters by batch size;
// without this aggregate check, an unmatched request could silently disappear.
func validateChangingTaskCoverage(keys []string, task PublicTask) error {
	kinds := map[string]bool{}
	available := map[string]bool{}
	for _, key := range keys {
		available[key] = true
		switch {
		case strings.HasPrefix(key, "reuse_certified_"):
			kinds["reuse_certified"] = true
		case strings.HasPrefix(key, "full_reuse_"):
			kinds["full_reuse"] = true
		case strings.HasPrefix(key, "reuse_"):
			kinds["reuse"] = true
		}
	}
	if len(kinds) == 0 {
		return nil
	}
	if len(task.Centers) == 0 || len(task.Queries) == 0 {
		return fmt.Errorf("empty approved changing-center task")
	}
	for kind := range kinds {
		for _, indices := range task.Queries {
			key := fmt.Sprintf("%s_%d_k%d", kind, len(indices), len(task.Centers[0]))
			if !available[key] {
				return fmt.Errorf("approved batch lacks pinned query key: %s", key)
			}
		}
	}
	return nil
}

func verifyChangingTaskCoverage(keys []string, taskPath string) error {
	changing := false
	for _, key := range keys {
		changing = changing || strings.HasPrefix(key, "reuse_") || strings.HasPrefix(key, "full_reuse_")
	}
	if !changing {
		return nil
	}
	if taskPath == "" {
		return fmt.Errorf("changing-center verification requires independently approved -public-task")
	}
	var task PublicTask
	if err := load(taskPath, &task); err != nil {
		return err
	}
	return validateChangingTaskCoverage(keys, task)
}

func verifyChangingPhase(trust Trust, m c.BoundModel, s c.BoundStatement, key, taskPath string, res *Result) error {
	if taskPath == "" {
		return fmt.Errorf("changing-center verification requires independently approved -public-task")
	}
	var task PublicTask
	if err := load(taskPath, &task); err != nil {
		return err
	}
	if len(task.Centers) == 0 {
		return fmt.Errorf("empty approved task")
	}
	kind := "reuse"
	if strings.HasPrefix(key, "reuse_certified_") {
		kind = "reuse_certified"
	}
	if strings.HasPrefix(key, "full_reuse_") {
		kind = "full_reuse"
	}
	var q, knew int
	if _, err := fmt.Sscanf(key, kind+"_%d_k%d", &q, &knew); err != nil || key != fmt.Sprintf("%s_%d_k%d", kind, q, knew) {
		return fmt.Errorf("invalid reusable key name")
	}
	if q < 1 || q > m.N || knew < 2 || knew > 128 {
		return fmt.Errorf("invalid reusable query shape")
	}
	queries := [][]int{}
	for _, indices := range task.Queries {
		if len(indices) == q {
			queries = append(queries, indices)
		}
	}
	if len(queries) == 0 {
		return fmt.Errorf("no approved query batches")
	}
	for _, centers := range task.Centers {
		if len(centers) != knew {
			return fmt.Errorf("approved center count differs from key")
		}
	}
	var phase Result
	if err := load(filepath.Join(folder, fmt.Sprintf("%s_%d.json", kind, q)), &phase); err != nil {
		return err
	}
	if phase.Phase != kind || !phase.Complete || phase.Statement != s || phase.Verified != len(phase.Samples) || len(phase.Samples) != len(task.Centers)*len(queries) {
		return fmt.Errorf("incomplete reusable phase or unexpected task coverage")
	}
	vk, err := readKey(trust, key)
	if err != nil {
		return err
	}
	for ordinal, sample := range phase.Samples {
		set, id := ordinal/len(queries), ordinal%len(queries)
		file := fmt.Sprintf("%s/s%03d_q%d_%04d.proof", kind, set, q, id)
		if sample.Set != set || sample.ID != id || sample.File != file || !reflect.DeepEqual(sample.Indices, queries[id]) {
			return fmt.Errorf("reuse proof identity, center set or indices differ from approved task")
		}
		var public frontend.Circuit
		if kind == "full_reuse" {
			public, err = c.BoundFullReusePublic(m, s, task.Centers[set], sample.Indices, sample.Psi)
		} else if kind == "reuse_certified" {
			public, err = c.BoundCertifiedReusePublic(m, s, task.Centers[set], sample.Indices, sample.Psi, sample.Margin, sample.CertifiedCount, sample.CostLower, sample.CostUpper)
		} else {
			public, err = c.BoundReusePublic(m, s, task.Centers[set], sample.Indices, sample.Psi, sample.Margin)
		}
		if err != nil {
			return err
		}
		if err = verifyFile(vk, sample.File, public); err != nil {
			return err
		}
		res.Verified++
	}
	return nil
}
