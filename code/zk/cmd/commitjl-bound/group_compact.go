package main

import (
	c "commitjl/zk/circuit"
	"encoding/gob"
	"fmt"
	"math/big"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"time"
)

const compactGroupVersion = "group-compact-v3"

func prepareCompactGroup(input string) (res Result, err error) {
	started := time.Now()
	defer func() {
		res.Phase = "group_compact_prepare"
		res.Wall = time.Since(started).Seconds()
		if err != nil {
			res.Status = "failed"
			res.Complete = false
			res.Error = err.Error()
		}
	}()
	if _, e := os.Stat(filepath.Join(folder, "group_commit_receipt.json")); e == nil {
		return res, fmt.Errorf("immutable compact commit already exists")
	}
	res, err = prepare(input)
	if err != nil {
		return
	}
	p, e := prepared()
	if e != nil {
		return res, e
	}
	g, e := c.PrepareExactOldGroupMoments(p, p.SaltZ)
	if e != nil {
		return res, e
	}
	commit, e := c.PrepareCompactGroup(p, g, p.SaltZ)
	if e != nil {
		return res, e
	}
	f, e := os.OpenFile(filepath.Join(folder, "prepared.gob.tmp"), os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0600)
	if e != nil {
		return res, e
	}
	e = gob.NewEncoder(f).Encode(p)
	if e == nil {
		e = f.Sync()
	}
	ce := f.Close()
	if e != nil {
		return res, e
	}
	if ce != nil {
		return res, ce
	}
	if e = os.Rename(filepath.Join(folder, "prepared.gob.tmp"), filepath.Join(folder, "prepared.gob")); e != nil {
		return res, e
	}
	if e = groupGob(groupPrivate{Moments: g, Commitment: commit}); e != nil {
		return res, e
	}
	if e = save("group_commit_receipt.json", GroupCommitReceipt{compactGroupVersion, p.Statement(), commit.Membership.Root(), commit.StatsRoot}); e != nil {
		return res, e
	}
	res.Statement = p.Statement()
	return
}

func executeCompactQueries(pin GroupOwnerPin, taskPath string, p *c.BoundPrepared, v *groupPrivate, limit int) (res Result, err error) {
	started := time.Now()
	res.Phase = "group_query"
	res.Statement = p.Statement()
	res.Status = "failed"
	defer func() {
		res.Wall = time.Since(started).Seconds()
		if err != nil {
			res.Error = err.Error()
		}
	}()
	var task PublicTask
	if err = load(taskPath, &task); err != nil {
		return
	}
	if err = groupTask(p.M, pin.Q, task); err != nil {
		return
	}
	records := make([]GroupQueryRecord, len(task.Centers))
	cs, pk, vk, e := setup(c.NewBoundGroupLinkedChange(p.M), "group_linked_change", limit, &res)
	if e != nil {
		return res, e
	}
	if cs == nil {
		return res, fmt.Errorf("linked shape exceeds guard")
	}
	if e = groupCheckKeys(pin); e != nil {
		return res, e
	}
	for set, centers := range task.Centers {
		t := time.Now()
		w, e := c.FillBoundGroupLinkedChange(p, v.Moments, v.Witness, centers)
		if e != nil {
			return res, e
		}
		native := time.Since(t).Seconds()
		s, e := prove(cs, pk, vk, w, fmt.Sprintf("group_query/s%03d_moment.proof", set), 1, 0)
		if e != nil {
			return res, e
		}
		s.Set = set
		s.ID = set
		s.NativeAnswer = native
		records[set] = GroupQueryRecord{Set: set, Assigned: s, Change: w.Change.(*big.Int).String(), Mode: "switch"}
		res.Samples = append(res.Samples, s)
		res.Verified++
		if e = save("group_query_records_progress.json", records); e != nil {
			return res, e
		}
	}
	cs = nil
	pk = nil
	vk = nil
	runtime.GC()
	cs, pk, vk, e = setup(c.NewBoundGroupSwitchBatchCompact(p.M, pin.Q), fmt.Sprintf("group_switch_compact_%d", pin.Q), limit, &res)
	if e != nil {
		return res, e
	}
	if cs == nil {
		return res, fmt.Errorf("compact switch exceeds guard")
	}
	if e = groupCheckKeys(pin); e != nil {
		return res, e
	}
	t6 := new(big.Int).Exp(big.NewInt(p.M.BasisScale), big.NewInt(6), nil)
	for set, centers := range task.Centers {
		params, e := c.NewGroupSwitchParameters(p.M, centers)
		if e != nil {
			return res, e
		}
		a, _ := new(big.Int).SetString(records[set].Change, 10)
		lo := new(big.Int).Mul(a, t6)
		hi := new(big.Int).Set(lo)
		for id, indices := range task.Queries {
			t := time.Now()
			w, e := c.FillBoundGroupSwitchBatchCompact(p, v.Moments, v.Commitment, params, indices)
			if e != nil {
				return res, e
			}
			native := time.Since(t).Seconds()
			s, e := prove(cs, pk, vk, w, fmt.Sprintf("group_query/s%03d_b%04d.proof", set, id), 1, 0)
			if e != nil {
				return res, e
			}
			s.Set = set
			s.ID = id
			s.Indices = indices
			s.NativeAnswer = native
			s.ChangeLower = w.Lower.(*big.Int).String()
			s.ChangeUpper = w.Upper.(*big.Int).String()
			records[set].Batches = append(records[set].Batches, s)
			res.Samples = append(res.Samples, s)
			res.Verified++
			lo.Add(lo, w.Lower.(*big.Int))
			hi.Add(hi, w.Upper.(*big.Int))
			if e = save("group_query_records_progress.json", records); e != nil {
				return res, e
			}
		}
		records[set].Lower = lo.String()
		records[set].Upper = hi.String()
		records[set].Decision = groupDecision(lo, hi)
	}
	if err = save("group_query_records.json", records); err != nil {
		return
	}
	res.Status = "complete"
	res.Complete = true
	return
}

func verifyCompactGroup(pin GroupOwnerPin, taskPath string) (res Result, err error) {
	started := time.Now()
	res.Phase = "group_verify"
	res.Status = "failed"
	res.Statement = pin.Statement
	defer func() {
		res.Wall = time.Since(started).Seconds()
		if err != nil {
			res.Error = err.Error()
		}
	}()
	if err = groupCheckKeys(pin); err != nil {
		return
	}
	var task PublicTask
	if err = load(taskPath, &task); err != nil {
		return
	}
	if err = groupTask(pin.Owner.Model, pin.Q, task); err != nil {
		return
	}
	var rows []c.GroupFingerprintPublicRow
	if err = load(filepath.Join(folder, "group_public_rows.json"), &rows); err != nil {
		return
	}
	sums, e := c.FoldGroupFingerprintRows(pin.Owner.Model, rows)
	if e != nil {
		return res, e
	}
	phases := map[string]Result{}
	for _, phase := range []string{"group_rows", "group_query"} {
		var data Result
		if e = load(filepath.Join(folder, phase+".json"), &data); e != nil {
			return res, e
		}
		if !data.Complete || data.Status != "complete" || data.Statement != pin.Statement || data.Verified != len(data.Samples) {
			return res, fmt.Errorf("incomplete compact phase%s", phase)
		}
		phases[phase] = data
	}
	if len(phases["group_rows"].Samples) != pin.Owner.Model.N {
		return res, fmt.Errorf("compact row proof coverage mismatch")
	}
	vk, e := readKey(pin.Owner, "group_row_compact")
	if e != nil {
		return res, e
	}
	for i, row := range rows {
		s := phases["group_rows"].Samples[i]
		file := fmt.Sprintf("group_rows/row_%06d.proof", i)
		if s.ID != i || s.File != file {
			return res, fmt.Errorf("compact row identity mismatch")
		}
		pub, e := c.GroupFingerprintRowPublic(pin.Owner.Model, pin.Statement, pin.Header, row)
		if e != nil {
			return res, e
		}
		pub.ExactOld = true
		pub.CompactGroup = true
		pub.GroupPath = nil
		if e = verifyFile(vk, file, pub); e != nil {
			return res, e
		}
		res.Verified++
	}
	var records []GroupQueryRecord
	if e = load(filepath.Join(folder, "group_query_records.json"), &records); e != nil {
		return res, e
	}
	if len(records) != len(task.Centers) {
		return res, fmt.Errorf("compact update coverage mismatch")
	}
	momentVK, e := readKey(pin.Owner, "group_linked_change")
	if e != nil {
		return res, e
	}
	switchVK, e := readKey(pin.Owner, fmt.Sprintf("group_switch_compact_%d", pin.Q))
	if e != nil {
		return res, e
	}
	verified := []Sample{}
	for set, r := range records {
		if r.Set != set || r.Mode != "switch" || len(r.Batches) != len(task.Queries) {
			return res, fmt.Errorf("compact update mode or coverage mismatch")
		}
		a, e := parseInt(r.Change)
		if e != nil || a.String() != r.Change {
			return res, fmt.Errorf("noncanonical compact aggregate")
		}
		file := fmt.Sprintf("group_query/s%03d_moment.proof", set)
		if r.Assigned.Set != set || r.Assigned.ID != set || r.Assigned.File != file {
			return res, fmt.Errorf("linked moment identity mismatch")
		}
		pub, e := c.GroupLinkedChangePublic(pin.Owner.Model, pin.Statement, pin.Header, sums, rows[0].Previous, rows[len(rows)-1].Next, task.Centers[set], a)
		if e != nil {
			return res, e
		}
		if e = verifyFile(momentVK, file, pub); e != nil {
			return res, e
		}
		res.Verified++
		verified = append(verified, r.Assigned)
	}
	t6 := new(big.Int).Exp(big.NewInt(pin.Owner.Model.BasisScale), big.NewInt(6), nil)
	decisions := []map[string]any{}
	for set, r := range records {
		a, _ := parseInt(r.Change)
		lo := new(big.Int).Mul(a, t6)
		hi := new(big.Int).Set(lo)
		params, e := c.NewGroupSwitchParameters(pin.Owner.Model, task.Centers[set])
		if e != nil {
			return res, e
		}
		for id, s := range r.Batches {
			file := fmt.Sprintf("group_query/s%03d_b%04d.proof", set, id)
			if s.Set != set || s.ID != id || s.File != file || !reflect.DeepEqual(s.Indices, task.Queries[id]) {
				return res, fmt.Errorf("compact batch identity mismatch")
			}
			lower, e := parseInt(s.ChangeLower)
			if e != nil || lower.String() != s.ChangeLower {
				return res, fmt.Errorf("noncanonical compact lower")
			}
			upper, e := parseInt(s.ChangeUpper)
			if e != nil || upper.String() != s.ChangeUpper {
				return res, fmt.Errorf("noncanonical compact upper")
			}
			pub, e := c.GroupSwitchPublic(pin.Owner.Model, pin.Statement, pin.Header.GroupRoot, params, s.Indices, lower, upper)
			if e != nil {
				return res, e
			}
			pub.ExactOld = true
			pub.CompactGroup = true
			pub.GroupPath = nil
			if e = verifyFile(switchVK, file, pub); e != nil {
				return res, e
			}
			res.Verified++
			verified = append(verified, s)
			lo.Add(lo, lower)
			hi.Add(hi, upper)
		}
		if r.Lower != lo.String() || r.Upper != hi.String() || r.Decision != groupDecision(lo, hi) {
			return res, fmt.Errorf("compact aggregate decision mismatch")
		}
		decisions = append(decisions, map[string]any{"set": set, "lower": lo.String(), "upper": hi.String(), "denominator": t6.String(), "decision": groupDecision(lo, hi), "mode": "switch"})
	}
	if !reflect.DeepEqual(verified, phases["group_query"].Samples) {
		return res, fmt.Errorf("compact sample summary differs from replay")
	}
	if err = save("group_verified_decisions.json", decisions); err != nil {
		return
	}
	res.Status = "complete"
	res.Complete = true
	return
}
