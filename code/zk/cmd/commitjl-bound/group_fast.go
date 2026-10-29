package main

// Experimental interactive, full-catalog group audit. The independently supplied
// owner pin includes a challenge issued AFTER the commitment receipt. The local
// benchmark simulates those roles in separate processes; it is not a production
// authenticated challenge service.
import (
	"encoding/gob"
	"fmt"
	"math/big"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"time"

	c "commitjl/zk/circuit"
	"github.com/consensys/gnark/frontend"
)

type GroupCommitReceipt struct {
	Version              string           `json:"version"`
	Statement            c.BoundStatement `json:"statement"`
	GroupRoot, StatsRoot *big.Int
}
type GroupOwnerPin struct {
	Version   string                   `json:"version"`
	Owner     Trust                    `json:"owner"`
	Statement c.BoundStatement         `json:"statement"`
	Header    c.GroupFingerprintHeader `json:"header"`
	CommitSHA string                   `json:"commit_sha256"`
	Q         int                      `json:"q"`
}
type groupPrivate struct {
	Moments    *c.GroupMoments
	Commitment *c.GroupFingerprintCommitment
	Witness    *c.GroupFingerprintWitness
}
type GroupQueryRecord struct {
	Set                    int      `json:"set"`
	Assigned               Sample   `json:"assigned"`
	Change                 string   `json:"assigned_change"`
	Batches                []Sample `json:"batches"`
	Mode                   string   `json:"mode,omitempty"`
	Lower, Upper, Decision string
}

func groupGob(value any) error {
	path := filepath.Join(folder, "group_private.gob")
	f, e := os.OpenFile(path+".tmp", os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0600)
	if e != nil {
		return e
	}
	e = gob.NewEncoder(f).Encode(value)
	if e == nil {
		e = f.Sync()
	}
	ce := f.Close()
	if e != nil {
		return e
	}
	if ce != nil {
		return ce
	}
	return os.Rename(path+".tmp", path)
}
func groupReadPrivate() (*groupPrivate, error) {
	f, e := os.Open(filepath.Join(folder, "group_private.gob"))
	if e != nil {
		return nil, e
	}
	defer f.Close()
	var v groupPrivate
	e = gob.NewDecoder(f).Decode(&v)
	return &v, e
}
func groupKeys(q int, exact bool) []string {
	if exact {
		return []string{"group_row_exact", "group_link", "group_moment", fmt.Sprintf("group_switch_exact_%d", q)}
	}
	return []string{"group_row", "group_link", "group_moment", fmt.Sprintf("group_switch_%d", q)}
}
func groupShapes(m c.BoundModel, q int, exact bool) []frontend.Circuit {
	if exact {
		return []frontend.Circuit{c.NewBoundGroupFingerprintRowExact(m), c.NewBoundGroupFingerprintLink(m), c.NewBoundGroupAssignedChange(m), c.NewBoundGroupSwitchBatchExact(m, q)}
	}
	return []frontend.Circuit{c.NewBoundGroupFingerprintRow(m), c.NewBoundGroupFingerprintLink(m), c.NewBoundGroupAssignedChange(m), c.NewBoundGroupSwitchBatch(m, q)}
}

func groupProfileKeys(q int, version string) []string {
	if version == compactGroupVersion {
		return []string{"group_row_compact", "group_linked_change", fmt.Sprintf("group_switch_compact_%d", q)}
	}
	return groupKeys(q, version == "group-exact-old-v2")
}

func groupProfileShapes(m c.BoundModel, q int, version string) []frontend.Circuit {
	if version == compactGroupVersion {
		return []frontend.Circuit{c.NewBoundGroupFingerprintRowCompact(m), c.NewBoundGroupLinkedChange(m), c.NewBoundGroupSwitchBatchCompact(m, q)}
	}
	return groupShapes(m, q, version == "group-exact-old-v2")
}
func groupReadPin(path string) (pin GroupOwnerPin, err error) {
	if path == "" {
		return pin, fmt.Errorf("independent post-commit owner/challenge pin required")
	}
	if err = load(path, &pin); err != nil {
		return
	}
	m := pin.Owner.Model
	if err = m.Validate(); err != nil {
		return
	}
	if (pin.Version != "group-fingerprint-v1" && pin.Version != "group-exact-old-v2" && pin.Version != compactGroupVersion) || pin.Owner.ModelHash != m.Hash().String() || pin.Owner.OriginalRoot != pin.Statement.CX || pin.Q < 1 || pin.Q > m.N || m.N%pin.Q != 0 {
		return pin, fmt.Errorf("invalid group owner pin")
	}
	var receipt GroupCommitReceipt
	if err = load(filepath.Join(folder, "group_commit_receipt.json"), &receipt); err != nil {
		return
	}
	hash, e := digest(filepath.Join(folder, "group_commit_receipt.json"))
	if e != nil {
		return pin, e
	}
	if hash != pin.CommitSHA || receipt.Version != pin.Version || receipt.Statement != pin.Statement || receipt.GroupRoot == nil || receipt.StatsRoot == nil || pin.Header.GroupRoot == nil || pin.Header.StatsRoot == nil || receipt.GroupRoot.Cmp(pin.Header.GroupRoot) != 0 || receipt.StatsRoot.Cmp(pin.Header.StatsRoot) != 0 {
		return pin, fmt.Errorf("commit receipt differs from independently pinned commitment")
	}
	if pin.Version == compactGroupVersion && pin.Header.GroupRoot.String() != pin.Statement.CH {
		return pin, fmt.Errorf("compact group/head roots differ")
	}
	// Public constructors check canonical roots and the nonzero challenge.
	row := c.GroupFingerprintPublicRow{Index: 0, Previous: big.NewInt(0), Next: big.NewInt(0), Masked: make([]*big.Int, m.K)}
	for j := range row.Masked {
		row.Masked[j] = big.NewInt(0)
	}
	_, err = c.GroupFingerprintRowPublic(m, pin.Statement, pin.Header, row)
	return
}
func groupCheckKeys(pin GroupOwnerPin) error {
	keys := groupProfileKeys(pin.Q, pin.Version)
	if len(pin.Owner.Keys) != len(keys) {
		return fmt.Errorf("unexpected independently pinned group keys")
	}
	for _, key := range keys {
		if _, err := readKey(pin.Owner, key); err != nil {
			return err
		}
	}
	return nil
}
func groupRows(p *c.BoundPrepared, v *groupPrivate) []c.GroupFingerprintPublicRow {
	rows := make([]c.GroupFingerprintPublicRow, p.M.N)
	for i := range rows {
		rows[i] = c.GroupFingerprintPublicRow{Index: i, Previous: v.Witness.MaskHash[i], Next: v.Witness.MaskHash[i+1], Masked: v.Witness.Masked[i]}
	}
	return rows
}
func groupLoadBound(pin GroupOwnerPin) (*c.BoundPrepared, *groupPrivate, error) {
	p, e := prepared()
	if e != nil {
		return nil, nil, e
	}
	if p.Statement() != pin.Statement || p.M.Hash().String() != pin.Owner.ModelHash {
		return nil, nil, fmt.Errorf("private prepared state differs from pinned model/roots")
	}
	v, e := groupReadPrivate()
	if e != nil {
		return nil, nil, e
	}
	if v.Commitment == nil || v.Commitment.Membership.Root().Cmp(pin.Header.GroupRoot) != 0 || v.Commitment.StatsRoot.Cmp(pin.Header.StatsRoot) != 0 {
		return nil, nil, fmt.Errorf("private group commitments differ from pin")
	}
	if v.Witness == nil {
		v.Witness, e = c.FillGroupFingerprintWitness(p, v.Moments, v.Commitment, pin.Header.Challenge)
		if e != nil {
			return nil, nil, e
		}
		if e = groupGob(v); e != nil {
			return nil, nil, e
		}
	} else if v.Witness.Challenge.Cmp(pin.Header.Challenge) != 0 {
		return nil, nil, fmt.Errorf("cannot change challenge or discard old proof work")
	}
	return p, v, nil
}

func executeGroup(phase, pinPath, taskPath string, q, limit int) (res Result, err error) {
	started := time.Now()
	res.Phase = phase
	res.Status = "failed"
	defer func() {
		res.Wall = time.Since(started).Seconds()
		if err != nil {
			res.Error = err.Error()
		}
	}()
	if phase == "group_verify" {
		return verifyGroup(pinPath, taskPath)
	}
	if !resumeEnabled() {
		return res, fmt.Errorf("group prototype requires COMMITJL_RESUME_PROOFS=1 to preserve setup/proof costs")
	}
	if phase == "group_commit" || phase == "group_commit_exact" {
		if _, e := os.Stat(filepath.Join(folder, "group_commit_receipt.json")); e == nil {
			return res, fmt.Errorf("commit receipt is immutable; choose a new run directory")
		}
		p, e := prepared()
		if e != nil {
			return res, e
		}
		// Fixed private approximate old-label groups; no exact-old-argmin shortcut.
		g, e := c.PrepareGroupMoments(p, p.SaltZ)
		version := "group-fingerprint-v1"
		if phase == "group_commit_exact" {
			g, e = c.PrepareExactOldGroupMoments(p, p.SaltZ)
			version = "group-exact-old-v2"
		}
		if e != nil {
			return res, e
		}
		commit, e := c.PrepareGroupFingerprintCommitment(p, g, p.SaltZ)
		if e != nil {
			return res, e
		}
		if e = groupGob(groupPrivate{Moments: g, Commitment: commit}); e != nil {
			return res, e
		}
		if e = save("group_commit_receipt.json", GroupCommitReceipt{version, p.Statement(), commit.Membership.Root(), commit.StatsRoot}); e != nil {
			return res, e
		}
		res.Statement = p.Statement()
		res.Status = "complete"
		res.Complete = true
		return res, nil
	}
	if phase == "group_keys" {
		p, e := prepared()
		if e != nil {
			return res, e
		}
		res.Statement = p.Statement()
		if q < 1 || q > p.M.N || p.M.N%q != 0 {
			return res, fmt.Errorf("full catalog must partition into q-sized batches")
		}
		var receipt GroupCommitReceipt
		if e = load(filepath.Join(folder, "group_commit_receipt.json"), &receipt); e != nil {
			return res, e
		}
		shapes := groupProfileShapes(p.M, q, receipt.Version)
		details := map[string]Result{}
		for i, key := range groupProfileKeys(q, receipt.Version) {
			item := Result{Phase: key, Statement: p.Statement()}
			cs, _, _, e := setup(shapes[i], key, limit, &item)
			if e != nil {
				return res, e
			}
			if cs == nil {
				return res, fmt.Errorf("group shape exceeds guard: %s", key)
			}
			item.Status = "complete"
			item.Complete = true
			details[key] = item
			res.Compile += item.Compile
			res.Setup += item.Setup
			if item.Constraints > res.Constraints {
				res.Constraints = item.Constraints
			}
			shapes[i] = nil
			cs = nil
			runtime.GC()
		}
		if e = save("group_key_details.json", details); e != nil {
			return res, e
		}
		res.Status = "complete"
		res.Complete = true
		return res, nil
	}
	pin, e := groupReadPin(pinPath)
	if e != nil {
		return res, e
	}
	if e = groupCheckKeys(pin); e != nil {
		return res, e
	}
	p, v, e := groupLoadBound(pin)
	if e != nil {
		return res, e
	}
	res.Statement = p.Statement()
	var shape frontend.Circuit
	exact := pin.Version == "group-exact-old-v2" || pin.Version == compactGroupVersion
	key := ""
	if phase == "group_rows" {
		key = "group_row"
		shape = c.NewBoundGroupFingerprintRow(p.M)
		if exact {
			key = "group_row_exact"
			shape = c.NewBoundGroupFingerprintRowExact(p.M)
		}
		if pin.Version == compactGroupVersion {
			key = "group_row_compact"
			shape = c.NewBoundGroupFingerprintRowCompact(p.M)
		}
	} else if phase == "group_link" {
		key = "group_link"
		shape = c.NewBoundGroupFingerprintLink(p.M)
	} else if phase == "group_query" {
		if pin.Version == compactGroupVersion {
			return executeCompactQueries(pin, taskPath, p, v, limit)
		}
		return executeGroupQueries(pin, taskPath, p, v, limit)
	} else {
		return res, fmt.Errorf("unknown group phase")
	}
	cs, pk, vk, e := setup(shape, key, limit, &res)
	if e != nil {
		return res, e
	}
	if cs == nil {
		return res, fmt.Errorf("group shape exceeds guard")
	}
	if e = groupCheckKeys(pin); e != nil {
		return res, e
	}
	if phase == "group_rows" {
		if e = save("group_public_rows.json", groupRows(p, v)); e != nil {
			return res, e
		}
		for i := 0; i < p.M.N; i++ {
			w := c.FillBoundGroupFingerprintRow(p, v.Moments, v.Witness, i)
			w.ExactOld = exact
			if pin.Version == compactGroupVersion {
				w = c.FillBoundGroupFingerprintRowCompact(p, v.Moments, v.Witness, i)
			}
			s, e := prove(cs, pk, vk, w, fmt.Sprintf("group_rows/row_%06d.proof", i), 1, 0)
			if e != nil {
				return res, e
			}
			s.ID = i
			res.Samples = append(res.Samples, s)
			res.Verified++
			if e = save("group_rows_progress.json", res); e != nil {
				return res, e
			}
		}
	} else {
		w, e := c.FillBoundGroupFingerprintLink(p, v.Moments, v.Witness)
		if e != nil {
			return res, e
		}
		s, e := prove(cs, pk, vk, w, "group_link/link.proof", 1, 0)
		if e != nil {
			return res, e
		}
		res.Samples = []Sample{s}
		res.Verified = 1
	}
	res.Status = "complete"
	res.Complete = true
	return res, nil
}

func executeGroupQueries(pin GroupOwnerPin, taskPath string, p *c.BoundPrepared, v *groupPrivate, limit int) (res Result, err error) {
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
	if err = groupTask(pin.Owner.Model, pin.Q, task); err != nil {
		return
	}
	records := make([]GroupQueryRecord, len(task.Centers))
	// All aggregate proofs use one setup; retain only that key in memory.
	cs, pk, vk, e := setup(c.NewBoundGroupAssignedChange(p.M), "group_moment", limit, &res)
	if e != nil {
		return res, e
	}
	if cs == nil {
		return res, fmt.Errorf("group moment exceeds guard")
	}
	if e = groupCheckKeys(pin); e != nil {
		return res, e
	}
	for set, centers := range task.Centers {
		t := time.Now()
		w, e := c.FillBoundGroupAssignedChange(p, v.Moments, v.Commitment, centers)
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
		records[set] = GroupQueryRecord{Set: set, Assigned: s, Change: w.Change.(*big.Int).String()}
		if mode, lo, hi, fast := groupExactFastInterval(pin.Version, p.M, centers, w.Change.(*big.Int)); fast {
			records[set].Mode = mode
			records[set].Lower, records[set].Upper = lo.String(), hi.String()
			records[set].Decision = groupDecision(lo, hi)
		} else if pin.Version == "group-exact-old-v2" {
			records[set].Mode = "switch"
		}
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
	needsSwitch := false
	for _, record := range records {
		if record.Mode == "" || record.Mode == "switch" {
			needsSwitch = true
		}
	}
	if !needsSwitch {
		if err = save("group_query_records.json", records); err != nil {
			return
		}
		res.Status, res.Complete = "complete", true
		return
	}
	exact := pin.Version == "group-exact-old-v2"
	switchKey := fmt.Sprintf("group_switch_%d", pin.Q)
	switchShape := c.NewBoundGroupSwitchBatch(p.M, pin.Q)
	if exact {
		switchKey = fmt.Sprintf("group_switch_exact_%d", pin.Q)
		switchShape.ExactOld = true
	}
	cs, pk, vk, e = setup(switchShape, switchKey, limit, &res)
	if e != nil {
		return res, e
	}
	if cs == nil {
		return res, fmt.Errorf("group switch exceeds guard")
	}
	if e = groupCheckKeys(pin); e != nil {
		return res, e
	}
	t6 := new(big.Int).Exp(big.NewInt(p.M.BasisScale), big.NewInt(6), nil)
	for set, centers := range task.Centers {
		if records[set].Mode != "" && records[set].Mode != "switch" {
			continue
		}
		params, e := c.NewGroupSwitchParameters(p.M, centers)
		if e != nil {
			return res, e
		}
		a, _ := new(big.Int).SetString(records[set].Change, 10)
		lo := new(big.Int).Mul(a, t6)
		hi := new(big.Int).Set(lo)
		for id, indices := range task.Queries {
			t := time.Now()
			w, e := c.FillBoundGroupSwitchBatchMode(p, v.Moments, v.Commitment, params, indices, exact)
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

// This shortcut is sound only after ALL preprocessing rows prove their exact
// old argmin under the independently pinned exact-profile verification key.
// For each row b_old=0 and -4*D*X^2 <= b_new <= 0, hence Delta <= A.
// The conservative interval deliberately does not claim a tight cost estimate.
func groupExactFastInterval(version string, m c.BoundModel, centers [][]int64, a *big.Int) (string, *big.Int, *big.Int, bool) {
	if version != "group-exact-old-v2" {
		return "", nil, nil, false
	}
	if reflect.DeepEqual(centers, m.Centers) && a.Sign() == 0 {
		return "identity", big.NewInt(0), big.NewInt(0), true
	}
	if a.Sign() >= 0 {
		return "", nil, nil, false
	}
	t6 := new(big.Int).Exp(big.NewInt(m.BasisScale), big.NewInt(6), nil)
	hi := new(big.Int).Mul(a, t6)
	width := new(big.Int).Mul(m.Limits().X, m.Limits().X)
	width.Mul(width, big.NewInt(4)).Mul(width, big.NewInt(int64(m.N))).Mul(width, big.NewInt(int64(m.D))).Mul(width, t6)
	return "assigned_upper_bound", new(big.Int).Sub(hi, width), hi, true
}
func groupDecision(lo, hi *big.Int) string {
	if lo.Sign() > 0 {
		return "worsening"
	}
	if hi.Sign() < 0 {
		return "improvement"
	}
	if lo.Sign() == 0 && hi.Sign() == 0 {
		return "equal"
	}
	return "inconclusive"
}
func groupTask(m c.BoundModel, q int, task PublicTask) error {
	if len(task.Centers) == 0 || len(task.Queries) != m.N/q {
		return fmt.Errorf("empty updates or incomplete fixed-catalog query coverage")
	}
	if len(task.ComparisonBase) > 0 && !reflect.DeepEqual(task.ComparisonBase, m.Centers) {
		return fmt.Errorf("comparison base differs from pinned old centers")
	}
	for b, indices := range task.Queries {
		if len(indices) != q {
			return fmt.Errorf("query shape differs from pinned batch")
		}
		for a, i := range indices {
			if i != b*q+a {
				return fmt.Errorf("group audit requires complete ordered coverage exactly once")
			}
		}
	}
	for _, centers := range task.Centers {
		if len(centers) != m.K {
			return fmt.Errorf("group center correspondence changed")
		}
		if e := c.ValidateGroupSwitchCenters(m, centers); e != nil {
			return e
		}
	}
	return nil
}

func verifyGroup(pinPath, taskPath string) (res Result, err error) {
	started := time.Now()
	res.Phase = "group_verify"
	res.Status = "failed"
	defer func() {
		res.Wall = time.Since(started).Seconds()
		if err != nil {
			res.Error = err.Error()
		}
	}()
	pin, e := groupReadPin(pinPath)
	if e != nil {
		return res, e
	}
	if pin.Version == compactGroupVersion {
		return verifyCompactGroup(pin, taskPath)
	}
	res.Statement = pin.Statement
	if e = groupCheckKeys(pin); e != nil {
		return res, e
	}
	var task PublicTask
	if e = load(taskPath, &task); e != nil {
		return res, e
	}
	if e = groupTask(pin.Owner.Model, pin.Q, task); e != nil {
		return res, e
	}
	var rows []c.GroupFingerprintPublicRow
	if e = load(filepath.Join(folder, "group_public_rows.json"), &rows); e != nil {
		return res, e
	}
	sums, e := c.FoldGroupFingerprintRows(pin.Owner.Model, rows)
	if e != nil {
		return res, e
	}
	phases := map[string]Result{}
	for _, phase := range []string{"group_rows", "group_link", "group_query"} {
		var data Result
		if e = load(filepath.Join(folder, phase+".json"), &data); e != nil {
			return res, e
		}
		if !data.Complete || data.Status != "complete" || data.Statement != pin.Statement || data.Verified != len(data.Samples) {
			return res, fmt.Errorf("incomplete or mismatched %s phase", phase)
		}
		phases[phase] = data
	}
	if len(phases["group_rows"].Samples) != pin.Owner.Model.N || len(phases["group_link"].Samples) != 1 {
		return res, fmt.Errorf("group preprocessing proof coverage mismatch")
	}
	exact := pin.Version == "group-exact-old-v2"
	rowKey := "group_row"
	if exact {
		rowKey = "group_row_exact"
	}
	vk, e := readKey(pin.Owner, rowKey)
	if e != nil {
		return res, e
	}
	for i, row := range rows {
		sample := phases["group_rows"].Samples[i]
		file := fmt.Sprintf("group_rows/row_%06d.proof", i)
		if sample.ID != i || sample.File != file {
			return res, fmt.Errorf("fingerprint proof identity mismatch")
		}
		pub, e := c.GroupFingerprintRowPublic(pin.Owner.Model, pin.Statement, pin.Header, row)
		if e != nil {
			return res, e
		}
		pub.ExactOld = exact
		if e = verifyFile(vk, file, pub); e != nil {
			return res, e
		}
		res.Verified++
	}
	link := phases["group_link"].Samples[0]
	if link.ID != 0 || link.File != "group_link/link.proof" {
		return res, fmt.Errorf("link proof identity mismatch")
	}
	pub, e := c.GroupFingerprintLinkPublic(pin.Owner.Model, pin.Statement, pin.Header, sums, rows[0].Previous, rows[len(rows)-1].Next)
	if e != nil {
		return res, e
	}
	vk, e = readKey(pin.Owner, "group_link")
	if e != nil {
		return res, e
	}
	if e = verifyFile(vk, link.File, pub); e != nil {
		return res, e
	}
	res.Verified++
	var records []GroupQueryRecord
	if e = load(filepath.Join(folder, "group_query_records.json"), &records); e != nil {
		return res, e
	}
	if len(records) != len(task.Centers) {
		return res, fmt.Errorf("group update proof coverage mismatch")
	}
	momentVK, e := readKey(pin.Owner, "group_moment")
	if e != nil {
		return res, e
	}
	switchKey := fmt.Sprintf("group_switch_%d", pin.Q)
	if exact {
		switchKey = fmt.Sprintf("group_switch_exact_%d", pin.Q)
	}
	switchVK, e := readKey(pin.Owner, switchKey)
	if e != nil {
		return res, e
	}
	verifiedSamples := []Sample{}
	t6 := new(big.Int).Exp(big.NewInt(pin.Owner.Model.BasisScale), big.NewInt(6), nil)
	for set, record := range records {
		if record.Set != set {
			return res, fmt.Errorf("update identity mismatch")
		}
		a, e := parseInt(record.Change)
		if e != nil || a.String() != record.Change {
			return res, fmt.Errorf("noncanonical assigned change")
		}
		file := fmt.Sprintf("group_query/s%03d_moment.proof", set)
		if record.Assigned.Set != set || record.Assigned.ID != set || record.Assigned.File != file {
			return res, fmt.Errorf("moment identity mismatch")
		}
		pub, e := c.GroupAssignedChangePublic(pin.Owner.Model, pin.Statement, pin.Header, task.Centers[set], a)
		if e != nil {
			return res, e
		}
		if e = verifyFile(momentVK, file, pub); e != nil {
			return res, e
		}
		res.Verified++
		verifiedSamples = append(verifiedSamples, record.Assigned)
	}
	decisions := []map[string]any{}
	for set, record := range records {
		a, _ := parseInt(record.Change)
		lo := new(big.Int).Mul(a, t6)
		hi := new(big.Int).Set(lo)
		mode, fastLo, fastHi, fast := groupExactFastInterval(pin.Version, pin.Owner.Model, task.Centers[set], a)
		if fast {
			if record.Mode != mode || len(record.Batches) != 0 {
				return res, fmt.Errorf("invalid exact-old fast path transcript")
			}
			lo, hi = fastLo, fastHi
		} else {
			expectedMode := ""
			if exact {
				expectedMode = "switch"
			}
			if record.Mode != expectedMode || len(record.Batches) != len(task.Queries) {
				return res, fmt.Errorf("switch mode or complete batch coverage mismatch")
			}
		}
		params, e := c.NewGroupSwitchParameters(pin.Owner.Model, task.Centers[set])
		if e != nil {
			return res, e
		}
		for id, sample := range record.Batches {
			file := fmt.Sprintf("group_query/s%03d_b%04d.proof", set, id)
			if sample.Set != set || sample.ID != id || sample.File != file || !reflect.DeepEqual(sample.Indices, task.Queries[id]) {
				return res, fmt.Errorf("switch proof identity mismatch")
			}
			low, e := parseInt(sample.ChangeLower)
			if e != nil || low.String() != sample.ChangeLower {
				return res, fmt.Errorf("noncanonical switching lower")
			}
			high, e := parseInt(sample.ChangeUpper)
			if e != nil || high.String() != sample.ChangeUpper {
				return res, fmt.Errorf("noncanonical switching upper")
			}
			pub, e := c.GroupSwitchPublic(pin.Owner.Model, pin.Statement, pin.Header.GroupRoot, params, sample.Indices, low, high)
			if e != nil {
				return res, e
			}
			pub.ExactOld = exact
			if e = verifyFile(switchVK, file, pub); e != nil {
				return res, e
			}
			res.Verified++
			verifiedSamples = append(verifiedSamples, sample)
			lo.Add(lo, low)
			hi.Add(hi, high)
		}
		if record.Lower != lo.String() || record.Upper != hi.String() || record.Decision != groupDecision(lo, hi) {
			return res, fmt.Errorf("aggregate interval/decision mismatch")
		}
		entry := map[string]any{"set": set, "lower": lo.String(), "upper": hi.String(), "denominator": t6.String(), "decision": groupDecision(lo, hi)}
		if exact {
			entry["mode"] = record.Mode
		}
		decisions = append(decisions, entry)
	}
	// Phase summaries cannot smuggle omitted, duplicated or unrelated samples.
	if !reflect.DeepEqual(verifiedSamples, phases["group_query"].Samples) {
		return res, fmt.Errorf("query summary differs from independently replayed transcript")
	}
	if e = save("group_verified_decisions.json", decisions); e != nil {
		return res, e
	}
	res.Status = "complete"
	res.Complete = true
	return res, nil
}
