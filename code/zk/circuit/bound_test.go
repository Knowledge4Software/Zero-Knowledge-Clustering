package circuit

import (
	"bytes"
	"encoding/gob"
	"math/big"
	"testing"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark/test"
)

func TestBoundModelHashSurvivesEmptySliceSerialization(t *testing.T) {
	for _, method := range []string{"full", "pca", "protected_energy", "protected_tail", "affine"} {
		p, err := boundFixture(method)
		if err != nil {
			t.Fatal(err)
		}
		m := p.M
		if len(m.Basis) == 0 {
			m.Basis = [][]int64{}
		}
		if len(m.TailRows) == 0 {
			m.TailRows = []int{}
		}
		if len(m.TailSigns) == 0 {
			m.TailSigns = []int64{}
		}
		var buf bytes.Buffer
		if err := gob.NewEncoder(&buf).Encode(m); err != nil {
			t.Fatal(err)
		}
		var restored BoundModel
		if err := gob.NewDecoder(&buf).Decode(&restored); err != nil {
			t.Fatal(err)
		}
		if m.Hash().Cmp(restored.Hash()) != 0 {
			t.Fatalf("%s hash changed after reload", method)
		}
		changed := restored
		changed.InputScale = "2"
		if changed.Hash().Cmp(m.Hash()) == 0 {
			t.Fatal("model parameters were not bound")
		}
	}
}

func TestBoundGeometryConformanceAndTampering(t *testing.T) {
	for _, method := range []string{"full", "pca", "protected_energy", "protected_tail"} {
		p, err := boundFixture(method)
		if err != nil {
			t.Fatal(err)
		}
		w, err := FillBoundGeometry(p, []int{0, 2})
		if err != nil {
			t.Fatal(err)
		}
		if err = test.IsSolved(NewBoundGeometry(p.M), w, ecc.BN254.ScalarField()); err != nil {
			t.Fatalf("%s: %v", method, err)
		}
		pub, err := BoundGeometryPublic(p.M, p.Statement(), []int{0, 2}, w.Distance.(*big.Int).String())
		if err != nil || pub == nil {
			t.Fatal(err)
		}
		w.Distance = add(w.Distance.(*big.Int), bi(1))
		if test.IsSolved(NewBoundGeometry(p.M), w, ecc.BN254.ScalarField()) == nil {
			t.Fatal("accepted wrong geometry distance")
		}
		w, _ = FillBoundGeometry(p, []int{0, 2})
		w.Indices[1] = 0
		if test.IsSolved(NewBoundGeometry(p.M), w, ecc.BN254.ScalarField()) == nil {
			t.Fatal("accepted duplicate pair")
		}
		if method == "protected_tail" {
			w, _ = FillBoundGeometry(p, []int{0, 2})
			w.Tail[0][0] = add(w.Tail[0][0].(*big.Int), bi(1))
			if test.IsSolved(NewBoundGeometry(p.M), w, ecc.BN254.ScalarField()) == nil {
				t.Fatal("accepted changed committed tail")
			}
		}
	}
	p, _ := boundFixture("affine")
	if _, err := FillBoundGeometry(p, []int{0, 2}); err == nil {
		t.Fatal("affine statistics treated as Euclidean embedding")
	}
}

func boundFixture(method string) (*BoundPrepared, error) {
	m := BoundModel{Version: BoundVersion, Method: method, N: 4, D: 4, K: 2, InputBits: 4, BasisScale: 8, InputScale: "1", Centers: [][]int64{{0, 0, 0, 0}, {2, 0, 0, 0}}}
	if method != "full" && method != "affine" && method != "cached" {
		m.Basis = [][]int64{{8, 0, 0, 0}}
	}
	if method == "protected_tail" || method == "protected_linear" {
		m.TailWidth = 2
		m.TailRows = []int{0, 1, 0, 1}
		m.TailSigns = []int64{1, -1, 1, 1}
	}
	return PrepareBound(m, [][]int64{{1, 1, 0, 0}, {3, 2, 0, 1}, {-2, 1, 1, 0}, {2, -1, 0, 2}}, bi(123), bi(456))
}

func TestBoundNativeCircuitConformance(t *testing.T) {
	full, err := boundFixture("full")
	if err != nil {
		t.Fatal(err)
	}
	reference, phi, _ := full.Answer([]int{0, 1, 2, 3})
	for _, method := range []string{"full", "pca", "protected_energy", "protected_tail", "affine"} {
		t.Run(method, func(t *testing.T) {
			p, err := boundFixture(method)
			if err != nil {
				t.Fatal(err)
			}
			if method != "full" {
				for i := 0; i < p.M.N; i++ {
					if err := test.IsSolved(NewBoundPreprocess(p.M), FillBoundPreprocess(p, i), ecc.BN254.ScalarField()); err != nil {
						t.Fatal(err)
					}
				}
			}
			witness, err := FillBoundQuery(p, []int{0, 1, 2, 3})
			if err != nil {
				t.Fatal(err)
			}
			if err = test.IsSolved(NewBoundQuery(p.M, 4), witness, ecc.BN254.ScalarField()); err != nil {
				t.Fatal(err)
			}
			labels, cost, _ := p.Answer([]int{0, 1, 2, 3})
			for i := range labels {
				if labels[i] != reference[i] {
					t.Fatal("assignment changed on exact-span fixture")
				}
			}
			if method == "affine" || method == "full" {
				if cost.Cmp(phi) != 0 {
					t.Fatal("exact affine baseline does not match Full")
				}
			}
			if method == "protected_energy" || method == "protected_tail" {
				scale := new(big.Int).Exp(bi(p.M.BasisScale), bi(4), nil)
				if cost.Cmp(mul(phi, scale)) != 0 {
					t.Fatal("axis-span energy identity failed")
				}
			}
		})
	}
}

func TestBoundPreprocessRejectsResealedFalseStatistics(t *testing.T) {
	for _, block := range []string{"energy", "head", "tail"} {
		t.Run(block, func(t *testing.T) {
			p, err := boundFixture("protected_tail")
			if err != nil {
				t.Fatal(err)
			}
			if block == "energy" {
				p.E[0] = bi(0)
			} else if block == "head" {
				p.H[0][0] = add(p.H[0][0], bi(1))
			} else {
				p.T[0][0] = add(p.T[0][0], bi(1))
			}
			leaves := make([]*big.Int, p.M.N)
			for i := range leaves {
				if block == "energy" || block == "head" {
					values := append([]*big.Int{}, p.H[i]...)
					values = append(values, p.E[i])
					leaves[i] = boundLeaf(p.M, boundH, i, p.SaltZ, values)
				} else {
					leaves[i] = boundLeaf(p.M, boundT, i, p.SaltZ, p.T[i])
				}
			}
			tree := NewBoundTree(leaves)
			if block == "energy" || block == "head" {
				p.TH = tree
				p.CH = tree.Root()
			} else {
				p.TT = tree
				p.CT = tree.Root()
			}
			p.Context = boundNativeHash(bi(boundContext), p.M.Hash(), p.CX, p.CH, p.CE, p.CT)
			if err = test.IsSolved(NewBoundPreprocess(p.M), FillBoundPreprocess(p, 0), ecc.BN254.ScalarField()); err == nil {
				t.Fatal("resealed false " + block + " accepted")
			}
		})
	}
}

func TestBoundMembershipAndStatementTampering(t *testing.T) {
	p, err := boundFixture("protected_energy")
	if err != nil {
		t.Fatal(err)
	}
	tests := map[string]func(*BoundQuery){
		"cost":            func(w *BoundQuery) { w.Psi = add(w.Psi.(*big.Int), bi(1)) },
		"assignment":      func(w *BoundQuery) { w.Assign[0] = 1 },
		"duplicate_index": func(w *BoundQuery) { w.Indices[1] = w.Indices[0] },
		"out_of_range":    func(w *BoundQuery) { w.Indices[1] = 4 },
		"row_index":       func(w *BoundQuery) { w.Indices[0] = 1 },
		"head":            func(w *BoundQuery) { w.H[0][0] = add(w.H[0][0].(*big.Int), bi(1)) },
		"path":            func(w *BoundQuery) { w.PH[0][0] = add(w.PH[0][0].(*big.Int), bi(1)) },
		"energy":          func(w *BoundQuery) { w.E[0] = bi(0) },
		"salt":            func(w *BoundQuery) { w.Salt = bi(999) },
		"context":         func(w *BoundQuery) { w.Context = add(p.Context, bi(1)) },
		"original_root":   func(w *BoundQuery) { w.CX = add(p.CX, bi(1)) },
	}
	for name, change := range tests {
		t.Run(name, func(t *testing.T) {
			w, _ := FillBoundQuery(p, []int{0, 2})
			change(w)
			if err := test.IsSolved(NewBoundQuery(p.M, 2), w, ecc.BN254.ScalarField()); err == nil {
				t.Fatal("tampering accepted")
			}
		})
	}
	w := FillBoundPreprocess(p, 0)
	w.X[0] = 2
	if err := test.IsSolved(NewBoundPreprocess(p.M), w, ecc.BN254.ScalarField()); err == nil {
		t.Fatal("altered original row accepted")
	}
}

func TestBoundRejectsUnsafeModelsAndInvalidQueries(t *testing.T) {
	p, _ := boundFixture("affine")
	for _, indices := range [][]int{{0, 0}, {3, 1}, {0, 4}, {-1, 2}} {
		if _, _, err := p.Answer(indices); err == nil {
			t.Fatal("invalid indices accepted")
		}
	}
	m := p.M
	m.N = 3
	if m.Validate() == nil {
		t.Fatal("non-power-of-two catalog accepted")
	}
	m = p.M
	m.Method = "bogus"
	if m.Validate() == nil {
		t.Fatal("unknown method accepted")
	}
}
