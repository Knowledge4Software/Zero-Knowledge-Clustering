package main

import (
	c "commitjl/zk/circuit"
	"encoding/gob"
	"fmt"
	"github.com/consensys/gnark/frontend"
	"math/big"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"time"
)

const momentsOnlyVersion = "group-moments-only-v1"

func momentsTask(m c.BoundModel, q int, task PublicTask) error {
	if q < 1 || m.N%q != 0 || len(task.Centers) == 0 || len(task.Queries) != m.N/q {
		return fmt.Errorf("moments task coverage")
	}
	if !reflect.DeepEqual(task.ComparisonBase, m.Centers) {
		return fmt.Errorf("moments comparison base")
	}
	for batch, indices := range task.Queries {
		if len(indices) != q {
			return fmt.Errorf("moments batch width")
		}
		for a, i := range indices {
			if i != batch*q+a {
				return fmt.Errorf("moments full ordered coverage")
			}
		}
	}
	for _, centers := range task.Centers {
		if err := c.ValidateMomentsCenters(m, centers); err != nil {
			return err
		}
	}
	return nil
}

var momentsKeys = []string{"moments_row", "moments_upper"}

func executeMoments(phase, input, pinPath, taskPath string, q, limit int) (res Result, err error) {
	started := time.Now()
	res.Phase = phase
	res.Status = "failed"
	defer func() {
		res.Wall = time.Since(started).Seconds()
		if err != nil {
			res.Error = err.Error()
		}
	}()
	if !resumeEnabled() {
		return res, fmt.Errorf("moments baseline requires resumable cost accounting")
	}
	if phase == "moments_prepare" {
		if _, e := os.Stat(filepath.Join(folder, "group_commit_receipt.json")); e == nil {
			return res, fmt.Errorf("immutable moments preparation exists")
		}
		var in Input
		if err = load(input, &in); err != nil {
			return
		}
		sx, e := parseInt(in.SaltX)
		if e != nil {
			return res, e
		}
		sz, e := parseInt(in.SaltZ)
		if e != nil {
			return res, e
		}
		p, e := c.PrepareMomentsOnly(in.Model, in.X, sx, sz)
		if e != nil {
			return res, e
		}
		g, e := c.PrepareExactOldGroupMoments(p, sz)
		if e != nil {
			return res, e
		}
		commit, e := c.PrepareGroupFingerprintCommitment(p, g, sz)
		if e != nil {
			return res, e
		}
		f, e := os.Create(filepath.Join(folder, "prepared.gob"))
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
		if err = groupGob(groupPrivate{Moments: g, Commitment: commit}); err != nil {
			return
		}
		if err = save("trusted.json", Trust{Model: in.Model, ModelHash: in.Model.Hash().String(), OriginalRoot: p.CX.String(), Keys: map[string]string{}}); err != nil {
			return
		}
		if err = save("group_commit_receipt.json", GroupCommitReceipt{momentsOnlyVersion, p.Statement(), commit.Membership.Root(), commit.StatsRoot}); err != nil {
			return
		}
		res.Statement = p.Statement()
	} else if phase == "moments_keys" {
		p, e := prepared()
		if e != nil {
			return res, e
		}
		shapes := []frontend.Circuit{c.NewBoundMomentsRow(p.M), c.NewBoundMomentsUpper(p.M)}
		details := map[string]Result{}
		for i, key := range momentsKeys {
			item := Result{Phase: key}
			cs, _, _, e := setup(shapes[i], key, limit, &item)
			if e != nil {
				return res, e
			}
			if cs == nil {
				return res, fmt.Errorf("moments key exceeds guard")
			}
			item.Complete = true
			item.Status = "complete"
			details[key] = item
			res.Compile += item.Compile
			res.Setup += item.Setup
			shapes[i] = nil
			cs = nil
			runtime.GC()
		}
		if err = save("moments_key_details.json", details); err != nil {
			return
		}
		res.Statement = p.Statement()
	} else {
		pin, e := momentsPin(pinPath)
		if e != nil {
			return res, e
		}
		res.Statement = pin.Statement
		if phase == "moments_verify" {
			return verifyMoments(pin, taskPath)
		}
		p, v, e := groupLoadBound(pin)
		if e != nil {
			return res, e
		}
		if phase == "moments_rows" {
			cs, pk, vk, e := setup(c.NewBoundMomentsRow(p.M), momentsKeys[0], limit, &res)
			if e != nil {
				return res, e
			}
			if cs == nil {
				return res, fmt.Errorf("moments row guard")
			}
			if err = save("group_public_rows.json", groupRows(p, v)); err != nil {
				return
			}
			for i := 0; i < p.M.N; i++ {
				w := c.FillBoundMomentsRow(p, v.Moments, v.Witness, i)
				s, e := prove(cs, pk, vk, w, fmt.Sprintf("moments_rows/row_%06d.proof", i), 1, 0)
				if e != nil {
					return res, e
				}
				s.ID = i
				res.Samples = append(res.Samples, s)
				res.Verified++
				if err = save("moments_rows_progress.json", res); err != nil {
					return
				}
			}
		} else if phase == "moments_query" {
			var task PublicTask
			if err = load(taskPath, &task); err != nil {
				return
			}
			if err = momentsTask(p.M, q, task); err != nil {
				return
			}
			cs, pk, vk, e := setup(c.NewBoundMomentsUpper(p.M), momentsKeys[1], limit, &res)
			if e != nil {
				return res, e
			}
			if cs == nil {
				return res, fmt.Errorf("moments upper guard")
			}
			records := []GroupQueryRecord{}
			for set, centers := range task.Centers {
				t := time.Now()
				w, e := c.FillBoundMomentsUpper(p, v.Moments, v.Witness, centers)
				if e != nil {
					return res, e
				}
				native := time.Since(t).Seconds()
				s, e := prove(cs, pk, vk, w, fmt.Sprintf("moments_query/s%03d.proof", set), 1, 0)
				if e != nil {
					return res, e
				}
				s.ID, s.Set, s.NativeAnswer = set, set, native
				upper := w.Upper.(*big.Int)
				lower := momentsLower(p.M)
				decision := "inconclusive"
				if upper.Sign() < 0 {
					decision = "improvement"
				}
				if reflect.DeepEqual(centers, p.M.Centers) {
					lower, upper = big.NewInt(0), big.NewInt(0)
					decision = "equal"
				}
				records = append(records, GroupQueryRecord{Set: set, Assigned: s, Change: w.Upper.(*big.Int).String(), Mode: "moments-upper", Lower: lower.String(), Upper: upper.String(), Decision: decision})
				res.Samples = append(res.Samples, s)
				res.Verified++
				if err = save("moments_query_progress.json", res); err != nil {
					return
				}
				if err = save("moments_records_progress.json", records); err != nil {
					return
				}
			}
			if err = save("moments_query_records.json", records); err != nil {
				return
			}
		} else {
			return res, fmt.Errorf("unknown moments phase")
		}
	}
	res.Complete = true
	res.Status = "complete"
	return
}

func momentsLower(m c.BoundModel) *big.Int {
	x := new(big.Int).Lsh(big.NewInt(1), uint(m.InputBits))
	return new(big.Int).Neg(new(big.Int).Mul(big.NewInt(int64(4*m.N*m.D)), new(big.Int).Mul(x, x)))
}

func momentsPin(path string) (pin GroupOwnerPin, err error) {
	if path == "" {
		return pin, fmt.Errorf("independent moments owner pin required")
	}
	if err = load(path, &pin); err != nil {
		return
	}
	m := pin.Owner.Model
	if err = m.Validate(); err != nil {
		return
	}
	if pin.Version != momentsOnlyVersion || m.Method != "affine" || len(m.Basis) != 0 || pin.Statement.CH != "0" || pin.Owner.ModelHash != m.Hash().String() || pin.Owner.OriginalRoot != pin.Statement.CX || pin.Q < 1 || m.N%pin.Q != 0 {
		return pin, fmt.Errorf("moments pin/model mismatch")
	}
	var receipt GroupCommitReceipt
	if err = load(filepath.Join(folder, "group_commit_receipt.json"), &receipt); err != nil {
		return
	}
	hash, e := digest(filepath.Join(folder, "group_commit_receipt.json"))
	if e != nil {
		return pin, e
	}
	if receipt.Version != pin.Version || hash != pin.CommitSHA || receipt.Statement != pin.Statement || receipt.GroupRoot == nil || receipt.StatsRoot == nil || pin.Header.GroupRoot == nil || pin.Header.StatsRoot == nil || receipt.GroupRoot.Cmp(pin.Header.GroupRoot) != 0 || receipt.StatsRoot.Cmp(pin.Header.StatsRoot) != 0 {
		return pin, fmt.Errorf("moments commitment pin mismatch")
	}
	if len(pin.Owner.Keys) != 2 {
		return pin, fmt.Errorf("moments key coverage")
	}
	for _, key := range momentsKeys {
		if _, e = readKey(pin.Owner, key); e != nil {
			return pin, e
		}
	}
	row := c.GroupFingerprintPublicRow{Index: 0, Previous: big.NewInt(0), Next: big.NewInt(0), Masked: make([]*big.Int, m.K)}
	for j := range row.Masked {
		row.Masked[j] = big.NewInt(0)
	}
	_, err = c.MomentsRowPublic(m, pin.Statement, pin.Header, row)
	return
}

func verifyMoments(pin GroupOwnerPin, taskPath string) (res Result, err error) {
	started := time.Now()
	res.Phase = "moments_verify"
	res.Status = "failed"
	res.Statement = pin.Statement
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
	m := pin.Owner.Model
	if err = momentsTask(m, pin.Q, task); err != nil {
		return
	}
	var rows []c.GroupFingerprintPublicRow
	if err = load(filepath.Join(folder, "group_public_rows.json"), &rows); err != nil {
		return
	}
	sums, e := c.FoldGroupFingerprintRows(m, rows)
	if e != nil {
		return res, e
	}
	var rp, qp Result
	for name, p := range map[string]*Result{"moments_rows": &rp, "moments_query": &qp} {
		if err = load(filepath.Join(folder, name+".json"), p); err != nil {
			return
		}
		if !p.Complete || p.Status != "complete" || p.Statement != pin.Statement || p.Verified != len(p.Samples) {
			return res, fmt.Errorf("incomplete moments phase")
		}
	}
	if len(rp.Samples) != m.N || len(qp.Samples) != len(task.Centers) {
		return res, fmt.Errorf("moments proof coverage")
	}
	rowVK, e := readKey(pin.Owner, momentsKeys[0])
	if e != nil {
		return res, e
	}
	for i, row := range rows {
		s := rp.Samples[i]
		file := fmt.Sprintf("moments_rows/row_%06d.proof", i)
		if s.ID != i || s.File != file {
			return res, fmt.Errorf("moments row order")
		}
		pub, e := c.MomentsRowPublic(m, pin.Statement, pin.Header, row)
		if e != nil {
			return res, e
		}
		if e = verifyFile(rowVK, file, pub); e != nil {
			return res, e
		}
		res.Verified++
	}
	var records []GroupQueryRecord
	if err = load(filepath.Join(folder, "moments_query_records.json"), &records); err != nil {
		return
	}
	if len(records) != len(task.Centers) {
		return res, fmt.Errorf("moments record coverage")
	}
	vk, e := readKey(pin.Owner, momentsKeys[1])
	if e != nil {
		return res, e
	}
	decisions := []map[string]any{}
	for set, r := range records {
		file := fmt.Sprintf("moments_query/s%03d.proof", set)
		if r.Set != set || r.Mode != "moments-upper" || len(r.Batches) != 0 || r.Assigned.File != file || r.Assigned.ID != set || r.Assigned.Set != set || !reflect.DeepEqual(r.Assigned, qp.Samples[set]) {
			return res, fmt.Errorf("moments record identity")
		}
		upper, e := parseInt(r.Change)
		if e != nil || upper.String() != r.Change {
			return res, fmt.Errorf("noncanonical upper")
		}
		pub, e := c.MomentsUpperPublic(m, pin.Statement, pin.Header, sums, rows[0].Previous, rows[len(rows)-1].Next, task.Centers[set], upper)
		if e != nil {
			return res, e
		}
		if e = verifyFile(vk, file, pub); e != nil {
			return res, e
		}
		res.Verified++
		lower := momentsLower(m)
		decision := "inconclusive"
		if upper.Sign() < 0 {
			decision = "improvement"
		}
		if reflect.DeepEqual(task.Centers[set], m.Centers) {
			lower, upper = big.NewInt(0), big.NewInt(0)
			decision = "equal"
		}
		if r.Lower != lower.String() || r.Upper != upper.String() || r.Decision != decision {
			return res, fmt.Errorf("moments summary differs from replay")
		}
		decisions = append(decisions, map[string]any{"set": set, "lower": lower.String(), "upper": upper.String(), "denominator": "1", "decision": decision, "mode": "moments-upper"})
	}
	if err = save("moments_verified_decisions.json", decisions); err != nil {
		return
	}
	res.Complete = true
	res.Status = "complete"
	return
}
