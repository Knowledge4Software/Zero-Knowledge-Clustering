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
