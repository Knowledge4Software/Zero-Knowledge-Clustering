package circuit

// Bound audits use a new, versioned Poseidon2 row commitment shared by every
// comparator. They are not timing-compatible with the older MiMC query study.

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"math/big"
	"math/bits"

	"github.com/consensys/gnark-crypto/ecc/bn254/fr"
)

const BoundVersion = "bound-audit-v1"

type BoundModel struct {
	Version    string    `json:"version"`
	Method     string    `json:"method"` // full, pca, protected_energy, protected_tail, affine, cached, learned_energy (any orthonormal basis + energy)
	N          int       `json:"n"`
	D          int       `json:"d"`
	K          int       `json:"k"`
	InputBits  int       `json:"input_bits"`
	BasisScale int64     `json:"basis_scale"`
	InputScale string    `json:"input_scale"` // canonical owner-side quantizer metadata
	Centers    [][]int64 `json:"centers"`
	Basis      [][]int64 `json:"basis"`
	TailRows   []int     `json:"tail_rows"` // one signed bucket per original coordinate
	TailSigns  []int64   `json:"tail_signs"`
	TailWidth  int       `json:"tail_width"`
	Seed       string    `json:"seed"` // model metadata, not a circuit-verified entropy claim
}

type BoundLimits struct{ X, H, E, T, Score, Distance *big.Int }

func bi(v int64) *big.Int        { return big.NewInt(v) }
func add(a, b *big.Int) *big.Int { return new(big.Int).Add(a, b) }
func mul(a, b *big.Int) *big.Int { return new(big.Int).Mul(a, b) }
func square(a *big.Int) *big.Int { return mul(a, a) }
func abs64(v int64) int64 {
	if v < 0 {
		return -v
	}
	return v
}

func (m BoundModel) Width() int {
	if m.Method == "cached" {
		return 1
	}
	if m.Method == "full" {
		return m.D
	}
	if m.Method == "affine" {
		return m.K - 1
	}
	return len(m.Basis)
}
func (m BoundModel) HasEnergy() bool {
	return m.Method == "affine" || m.Method == "cached" || m.Method == "protected_energy" || m.Method == "protected_tail" || m.Method == "learned_energy"
}

// UsesTailQuery is the no-energy ablation: its query proves distances in
// head+tail, including the (possibly nonzero, rounded) center tails.
func (m BoundModel) UsesTailQuery() bool { return m.Method == "protected_linear" }
func (m BoundModel) Depth() int          { return bits.Len(uint(m.N)) - 1 }
func (m BoundModel) Hash() *big.Int {
	// Gob restores empty slices as nil. Both represent the same absent model
	// component, so canonicalize them before hashing across process boundaries.
	if len(m.Basis) == 0 {
		m.Basis = nil
	}
	if len(m.TailRows) == 0 {
		m.TailRows = nil
	}
	if len(m.TailSigns) == 0 {
		m.TailSigns = nil
	}
	b, _ := json.Marshal(m)
	h := sha256.Sum256(b)
	return new(big.Int).Mod(new(big.Int).SetBytes(h[:]), fr.Modulus())
}
func (m BoundModel) Validate() error {
	if m.Version != BoundVersion || m.N < 1 || m.N > (1<<20) || m.N&(m.N-1) != 0 || m.D < 1 || m.D > 10000 || m.K < 2 || m.K > 128 {
		return fmt.Errorf("invalid version or bounded power-of-two dataset shape")
	}
	if m.InputBits < 1 || m.InputBits > 16 || m.BasisScale < 1 || m.BasisScale > (1<<20) || m.InputScale == "" {
		return fmt.Errorf("invalid input/basis scale")
	}
	if scale, ok := new(big.Rat).SetString(m.InputScale); !ok || scale.Sign() <= 0 {
		return fmt.Errorf("input scale must be a positive rational/decimal")
	}
	switch m.Method {
	case "full", "pca", "protected_energy", "protected_tail", "protected_linear", "affine", "cached", "learned_energy":
	default:
		return fmt.Errorf("unknown bound method")
	}
	if len(m.Centers) != m.K {
		return fmt.Errorf("center count mismatch")
	}
	b := int64(1) << uint(m.InputBits)
	for _, row := range m.Centers {
		if len(row) != m.D {
			return fmt.Errorf("center width mismatch")
		}
		for _, v := range row {
			if v < -b || v > b {
				return fmt.Errorf("center outside input range")
			}
		}
	}
	if m.Method == "full" || m.Method == "affine" || m.Method == "cached" {
		if len(m.Basis) != 0 || m.TailWidth != 0 {
			return fmt.Errorf("unexpected projection for exact baseline")
		}
	} else {
		if len(m.Basis) < 1 || len(m.Basis) > 128 {
			return fmt.Errorf("invalid basis rank")
		}
		if m.Method != "pca" && m.Method != "learned_energy" && len(m.Basis) > m.K-1 {
			return fmt.Errorf("protected rank exceeds center differences")
		}
		for _, row := range m.Basis {
			if len(row) != m.D {
				return fmt.Errorf("basis width mismatch")
			}
			for _, v := range row {
				if v < -m.BasisScale || v > m.BasisScale {
					return fmt.Errorf("basis coefficient out of range")
				}
			}
		}
	}
	if m.Method == "protected_tail" || m.UsesTailQuery() {
		if m.TailWidth < 1 || m.TailWidth > 128 || len(m.TailRows) != m.D || len(m.TailSigns) != m.D {
			return fmt.Errorf("invalid tail shape")
		}
		for i, r := range m.TailRows {
			if r < 0 || r >= m.TailWidth || (m.TailSigns[i] != 1 && m.TailSigns[i] != -1) {
				return fmt.Errorf("invalid signed tail bucket")
			}
		}
	} else if m.TailWidth != 0 || len(m.TailRows) != 0 || len(m.TailSigns) != 0 {
		return fmt.Errorf("unexpected tail metadata")
	}
	l := m.Limits()
	for _, v := range []*big.Int{l.H, l.E, l.T, l.Score, l.Distance} {
		if mul(v, bi(int64(m.N)*16)).Cmp(fr.Modulus()) >= 0 {
			return fmt.Errorf("unsafe integer/field bound")
		}
	}
	return nil
}

func (m BoundModel) Limits() BoundLimits {
	x := new(big.Int).Lsh(bi(1), uint(m.InputBits))
	w := mul(x, bi(2))
	d := bi(int64(m.D))
	l := BoundLimits{X: x, H: new(big.Int), E: new(big.Int), T: new(big.Int)}
	if m.Method == "cached" {
		l.H = bi(int64(m.K - 1))
		l.E = mul(d, square(w))
		l.Score = mul(bi(12), l.E)
		l.Distance = new(big.Int).Set(l.E)
		return l
	}
	if m.Method == "full" {
		l.H.Set(x)
	} else if m.Method == "affine" {
		l.H = mul(bi(3), mul(d, square(w)))
		l.E = mul(d, square(w))
	} else {
		l1 := make([]*big.Int, len(m.Basis))
		maxL1 := new(big.Int)
		for j, row := range m.Basis {
			l1[j] = new(big.Int)
			for _, v := range row {
				l1[j].Add(l1[j], bi(abs64(v)))
			}
			if l1[j].Cmp(maxL1) > 0 {
				maxL1.Set(l1[j])
			}
		}
		l.H = mul(bi(m.BasisScale), mul(w, maxL1))
		if m.HasEnergy() || m.UsesTailQuery() {
			rmax := new(big.Int)
			buckets := make([]*big.Int, m.TailWidth)
			for i := range buckets {
				buckets[i] = new(big.Int)
			}
			for t := 0; t < m.D; t++ {
				v := square(bi(m.BasisScale))
				for j := range m.Basis {
					v.Add(v, mul(bi(abs64(m.Basis[j][t])), l1[j]))
				}
				v.Mul(v, w)
				if v.Cmp(rmax) > 0 {
					rmax.Set(v)
				}
				if m.TailWidth > 0 {
					buckets[m.TailRows[t]].Add(buckets[m.TailRows[t]], v)
				}
			}
			if m.HasEnergy() {
				l.E = mul(d, square(rmax))
			}
			for _, v := range buckets {
				if v.Cmp(l.T) > 0 {
					l.T.Set(v)
				}
			}
		}
	}
	if m.Method == "affine" {
		l.Score = mul(l.H, bi(2))
		l.Distance = add(l.E, l.H)
	} else {
		l.Score = mul(bi(int64(6*m.Width())), square(l.H))
		l.Distance = add(mul(bi(int64(4*m.Width())), square(l.H)), l.E)
		if m.UsesTailQuery() {
			l.Score.Add(l.Score, mul(bi(int64(6*m.TailWidth)), square(l.T)))
			l.Distance.Add(l.Distance, mul(bi(int64(4*m.TailWidth)), square(l.T)))
		}
	}
	return l
}
