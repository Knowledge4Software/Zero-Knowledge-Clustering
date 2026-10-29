package main

import (
	c "commitjl/zk/circuit"
	"fmt"
	"math/big"
	"path/filepath"
	"reflect"
	"runtime"
	"time"
)

const refinementKey = "group_refinement_1"
const refinementBudget = 16

type RefinementRequest struct {
	Sets [][]int `json:"sets"`
}
type RefinementRecord struct {
	Set                    int      `json:"set"`
	Samples                []Sample `json:"samples"`
	Lower, Upper, Decision string
}

func canonicalRefinement(value string) (*big.Int, error) {
	v, e := parseInt(value)
	if e != nil || v.String() != value {
		return nil, fmt.Errorf("noncanonical refinement integer")
	}
	return v, nil
}

func refinementTrust(path string, pin GroupOwnerPin) (Trust, error) {
	var trust Trust
	if path == "" {
		return trust, fmt.Errorf("independent refinement key pin required")
	}
	if e := load(path, &trust); e != nil {
		return trust, e
	}
	if !reflect.DeepEqual(trust.Model, pin.Owner.Model) || trust.ModelHash != pin.Owner.ModelHash || trust.OriginalRoot != pin.Owner.OriginalRoot || len(trust.Keys) != 1 || trust.Keys[refinementKey] == "" {
		return trust, fmt.Errorf("refinement key/model/root pin mismatch")
	}
	return trust, nil
}

func executeRefinement(phase, pinPath, taskPath, base, keyPin, requestPath string, limit int) (res Result, err error) {
	started := time.Now()
	res.Phase = phase
	res.Status = "failed"
	defer func() {
		res.Wall = time.Since(started).Seconds()
		if err != nil {
			res.Error = err.Error()
		}
	}()
	currentFolder := folder
	if base == "" {
		return res, fmt.Errorf("complete independently pinned base required")
	}
	folder = base
	pin, e := groupReadPin(pinPath)
	folder = currentFolder
	if e != nil {
		return res, e
	}
	if pin.Version != compactGroupVersion {
		return res, fmt.Errorf("refinement requires complete compact profile")
	}
	res.Statement = pin.Statement
	var task PublicTask
	if e = load(taskPath, &task); e != nil {
		return res, e
	}
	if e = groupTask(pin.Owner.Model, pin.Q, task); e != nil {
		return res, e
	}
	m := pin.Owner.Model
	if phase == "refinement_setup" {
		trust := Trust{Model: m, ModelHash: m.Hash().String(), OriginalRoot: pin.Owner.OriginalRoot, Keys: map[string]string{}}
		var existing Trust
		if e = load(filepath.Join(folder, "trusted.json"), &existing); e == nil {
			if existing.ModelHash != trust.ModelHash || existing.OriginalRoot != trust.OriginalRoot {
				return res, fmt.Errorf("existing refinement setup belongs to a different base")
			}
		} else {
			if e = save("trusted.json", trust); e != nil {
				return res, e
			}
		}
		cs, _, _, e := setup(c.NewBoundGroupRefinement(m, 1), refinementKey, limit, &res)
		if e != nil {
			return res, e
		}
		if cs == nil {
			return res, fmt.Errorf("refinement shape exceeds guard")
		}
		res.Status = "complete"
		res.Complete = true
		return res, nil
	}
	trust, e := refinementTrust(keyPin, pin)
	if e != nil {
		return res, e
	}
	current := folder
	var records []GroupQueryRecord
	if base == "" {
		return res, fmt.Errorf("complete independently pinned base required")
	}
	if phase == "refinement_verify" {
		// No private catalog or cached native answer is loaded in this branch.
		// Reverify the entire base: a correction proof alone is insufficient.
		folder = base
		r, e := verifyCompactGroup(pin, taskPath)
		folder = current
		if e != nil || !r.Complete {
			return res, fmt.Errorf("base public replay failed: %v", e)
		}
		res.Verified = r.Verified
		if e = load(filepath.Join(base, "group_query_records.json"), &records); e != nil {
			return res, e
		}
		vk, e := readKey(trust, refinementKey)
		if e != nil {
			return res, e
		}
		var refined []RefinementRecord
		if e = load(filepath.Join(folder, "refinement_records.json"), &refined); e != nil {
			return res, e
		}
		if len(refined) != len(task.Centers) {
			return res, fmt.Errorf("refinement update coverage mismatch")
		}
		decisions := []map[string]any{}
		for set, r := range refined {
			if r.Set != set || len(r.Samples) > refinementBudget {
				return res, fmt.Errorf("invalid refinement set or budget")
			}
			lower, e := canonicalRefinement(records[set].Lower)
			if e != nil {
				return res, e
			}
			upper, e := canonicalRefinement(records[set].Upper)
			if e != nil {
				return res, e
			}
			seen := map[int]bool{}
			for id, s := range r.Samples {
				file := fmt.Sprintf("refinement/s%03d_r%02d.proof", set, id)
				if s.Set != set || s.ID != id || s.File != file || len(s.Indices) != 1 || seen[s.Indices[0]] {
					return res, fmt.Errorf("duplicate or substituted refinement row")
				}
				seen[s.Indices[0]] = true
				lo, e := canonicalRefinement(s.ChangeLower)
				if e != nil {
					return res, e
				}
				hi, e := canonicalRefinement(s.ChangeUpper)
				if e != nil {
					return res, e
				}
				pub, e := c.GroupRefinementPublic(m, pin.Statement, pin.Header.GroupRoot, task.Centers[set], s.Indices, lo, hi)
				if e != nil {
					return res, e
				}
				if e = verifyFile(vk, file, pub); e != nil {
					return res, e
				}
				lower.Add(lower, lo)
				upper.Add(upper, hi)
				res.Verified++
				res.Samples = append(res.Samples, s)
			}
			if lower.Cmp(upper) > 0 || r.Lower != lower.String() || r.Upper != upper.String() || r.Decision != groupDecision(lower, upper) {
				return res, fmt.Errorf("invalid corrected aggregate")
			}
			decisions = append(decisions, map[string]any{"set": set, "lower": lower.String(), "upper": upper.String(), "decision": r.Decision, "refined_rows": len(r.Samples), "denominator": new(big.Int).Exp(big.NewInt(m.BasisScale), big.NewInt(6), nil).String()})
		}
		if e = save("refinement_verified_decisions.json", decisions); e != nil {
			return res, e
		}
		res.Status = "complete"
		res.Complete = true
		return res, nil
	}
	var requests RefinementRequest
	if e = load(requestPath, &requests); e != nil {
		return res, e
	}
	if len(requests.Sets) != len(task.Centers) {
		return res, fmt.Errorf("refinement request coverage mismatch")
	}
	folder = base
	p, e := prepared()
	var g *groupPrivate
	if e == nil {
		g, e = groupReadPrivate()
	}
	folder = current
	if e != nil {
		return res, e
	}
	if p.Statement() != pin.Statement || p.M.Hash().String() != m.Hash().String() || g.Commitment.Membership.Root().Cmp(pin.Header.GroupRoot) != 0 {
		return res, fmt.Errorf("private base does not match owner pin")
	}
	if e = load(filepath.Join(base, "group_query_records.json"), &records); e != nil {
		return res, e
	}
	if len(records) != len(task.Centers) {
		return res, fmt.Errorf("base update coverage mismatch")
	}
	cs, pk, vk, e := setup(c.NewBoundGroupRefinement(m, 1), refinementKey, limit, &res)
	if e != nil {
		return res, e
	}
	if cs == nil {
		return res, fmt.Errorf("refinement shape exceeds guard")
	}
	if _, e = readKey(trust, refinementKey); e != nil {
		return res, e
	}
	refined := make([]RefinementRecord, len(task.Centers))
	for set, order := range requests.Sets {
		if len(order) > refinementBudget {
			return res, fmt.Errorf("refinement budget exceeded")
		}
		lower, e := canonicalRefinement(records[set].Lower)
		if e != nil {
			return res, e
		}
		upper, e := canonicalRefinement(records[set].Upper)
		if e != nil {
			return res, e
		}
		refined[set] = RefinementRecord{Set: set, Samples: []Sample{}}
		seen := map[int]bool{}
		for _, index := range order {
			if index < 0 || index >= m.N || seen[index] {
				return res, fmt.Errorf("invalid refinement index")
			}
			seen[index] = true
		}
		for _, index := range order {
			if groupDecision(lower, upper) != "inconclusive" {
				break
			}
			nativeStart := time.Now()
			w, e := c.FillBoundGroupRefinement(p, g.Moments, g.Commitment, task.Centers[set], []int{index})
			if e != nil {
				return res, e
			}
			native := time.Since(nativeStart).Seconds()
			id := len(refined[set].Samples)
			s, e := prove(cs, pk, vk, w, fmt.Sprintf("refinement/s%03d_r%02d.proof", set, id), 1, 0)
			if e != nil {
				return res, e
			}
			s.Set, s.ID, s.Indices, s.NativeAnswer = set, id, []int{index}, native
			s.ChangeLower = w.CorrectionLower.(*big.Int).String()
			s.ChangeUpper = w.CorrectionUpper.(*big.Int).String()
			refined[set].Samples = append(refined[set].Samples, s)
			res.Samples = append(res.Samples, s)
			res.Verified++
			lower.Add(lower, w.CorrectionLower.(*big.Int))
			upper.Add(upper, w.CorrectionUpper.(*big.Int))
		}
		refined[set].Lower, refined[set].Upper = lower.String(), upper.String()
		refined[set].Decision = groupDecision(lower, upper)
		if e = save("refinement_records_progress.json", refined); e != nil {
			return res, e
		}
	}
	if e = save("refinement_records.json", refined); e != nil {
		return res, e
	}
	runtime.KeepAlive(p)
	res.Status = "complete"
	res.Complete = true
	return res, nil
}
