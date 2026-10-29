package main

import "testing"

func TestChangingTaskCoverage(t *testing.T) {
	task := PublicTask{Centers: [][][]int64{{{0}, {1}}}, Queries: [][]int{{0, 1}}}
	if err := validateChangingTaskCoverage([]string{"reuse_2_k2"}, task); err != nil {
		t.Fatal(err)
	}
	task.Queries = append(task.Queries, []int{2})
	if err := validateChangingTaskCoverage([]string{"reuse_2_k2"}, task); err == nil {
		t.Fatal("accepted approved batch without a proof key")
	}
	if err := validateChangingTaskCoverage([]string{"reuse_2_k2", "reuse_1_k2"}, task); err != nil {
		t.Fatal(err)
	}
	if err := validateChangingTaskCoverage([]string{"reuse_2_k2", "reuse_1_k2", "reuse_certified_2_k2"}, task); err == nil {
		t.Fatal("certified relation silently omitted an approved batch")
	}
	if err := validateChangingTaskCoverage([]string{"full_reuse_2_k2"}, task); err == nil {
		t.Fatal("exact Full silently omitted an approved batch")
	}
	if err := validateChangingTaskCoverage([]string{"reuse_2_k2"}, PublicTask{}); err == nil {
		t.Fatal("accepted empty task")
	}
}

func TestPairedTaskRequiresApprovedBaseAndEveryBatch(t *testing.T) {
	task := PublicTask{Centers: [][][]int64{{{0}, {1}}}, Queries: [][]int{{0, 1}}}
	if validateChangingTaskCoverage([]string{"paired_change_2_k2"}, task) == nil {
		t.Fatal("accepted missing comparison base")
	}
	task.ComparisonBase = [][]int64{{0}, {2}}
	if err := validateChangingTaskCoverage([]string{"paired_change_2_k2"}, task); err != nil {
		t.Fatal(err)
	}
	task.Queries = append(task.Queries, []int{2})
	if validateChangingTaskCoverage([]string{"paired_change_2_k2"}, task) == nil {
		t.Fatal("paired relation omitted requested batch")
	}
}

func TestRelativeTaskRequiresEveryApprovedBatch(t *testing.T) {
	task := PublicTask{Centers: [][][]int64{{{0}, {1}}}, Queries: [][]int{{0, 1}}}
	if err := validateChangingTaskCoverage([]string{"relative_reuse_2_k2"}, task); err != nil {
		t.Fatal(err)
	}
	task.Queries = append(task.Queries, []int{2})
	if validateChangingTaskCoverage([]string{"relative_reuse_2_k2"}, task) == nil {
		t.Fatal("relative certificate omitted an approved batch")
	}
	if err := validateChangingTaskCoverage([]string{"relative_reuse_2_k2", "relative_reuse_1_k2"}, task); err != nil {
		t.Fatal(err)
	}
	if validateChangingTaskCoverage([]string{"relative_reuse_2_k2"}, PublicTask{}) == nil {
		t.Fatal("relative certificate accepted an empty approved task")
	}
}

func TestFamilyAffineTaskRequiresEveryApprovedBatch(t *testing.T) {
	task := PublicTask{Centers: [][][]int64{{{0}, {1}}}, Queries: [][]int{{0, 1}}}
	if err := validateChangingTaskCoverage([]string{"family_affine_2_k2"}, task); err != nil {
		t.Fatal(err)
	}
	task.Queries = append(task.Queries, []int{2})
	if validateChangingTaskCoverage([]string{"family_affine_2_k2"}, task) == nil {
		t.Fatal("family affine relation omitted approved batch")
	}
}
