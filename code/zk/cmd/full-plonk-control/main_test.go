package main

import (
	c "commitjl/zk/circuit"
	"testing"
)

func TestTaskAdmission(t *testing.T) {
	m := c.BoundModel{Version: c.BoundVersion, Method: "full", N: 4, D: 2, K: 2, InputBits: 4, BasisScale: 8, InputScale: "1", Centers: [][]int64{{0, 0}, {2, 0}}}
	valid := task{[][][]int64{{{0, 0}, {2, 0}}, {{1, 1}, {3, 1}}}}
	if e := validateTask(m, valid, 2); e != nil {
		t.Fatal(e)
	}
	for _, bad := range []task{{nil}, {[][][]int64{{{0, 0}}}}, {[][][]int64{{{0, 0}, {17, 0}}}}, {[][][]int64{{{0}, {2}}}}} {
		if validateTask(m, bad, 2) == nil {
			t.Fatal("accepted invalid public task")
		}
	}
	if validateTask(m, valid, 3) == nil {
		t.Fatal("accepted incomplete batches")
	}
}

func TestLaptopCannotStartProofGeneration(t *testing.T) {
	t.Setenv("COMMITJL_EXECUTION_TARGET", "")
	if remoteOnly() == nil {
		t.Fatal("generation must require explicit authorized remote target")
	}
}
