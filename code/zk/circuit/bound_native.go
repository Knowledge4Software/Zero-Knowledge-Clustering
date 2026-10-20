package circuit

import (
	"fmt"
	"math/big"

	nativeposeidon "github.com/consensys/gnark-crypto/ecc/bn254/fr/poseidon2"
)

const (
	boundX       = 101
	boundH       = 102
	boundE       = 103
	boundT       = 104
	boundNode    = 105
	boundContext = 106
)

func boundNativeHash(values ...*big.Int) *big.Int {
	h := nativeposeidon.NewMerkleDamgardHasher()
	for _, v := range values {
		_, _ = h.Write(feBytes(v))
	}
	return new(big.Int).SetBytes(h.Sum(nil))
}
func boundLeaf(m BoundModel, domain, index int, salt *big.Int, values []*big.Int) *big.Int {
	tag := bi(0)
	if domain != boundX {
		tag = m.Hash()
	}
	return boundLeafTag(m, domain, index, salt, tag, values)
}
func boundLeafTag(m BoundModel, domain, index int, salt, tag *big.Int, values []*big.Int) *big.Int {
	v := []*big.Int{bi(int64(domain)), bi(int64(m.N)), bi(int64(m.D)), tag, bi(int64(index)), salt}
	return boundNativeHash(append(v, values...)...)
}

type BoundTree struct{ Levels [][]*big.Int }

func NewBoundTree(leaves []*big.Int) *BoundTree {
	t := &BoundTree{Levels: [][]*big.Int{leaves}}
	for len(leaves) > 1 {
		next := make([]*big.Int, len(leaves)/2)
		for i := range next {
			next[i] = boundNativeHash(bi(boundNode), bi(int64(len(t.Levels)-1)), leaves[2*i], leaves[2*i+1])
		}
		t.Levels = append(t.Levels, next)
		leaves = next
	}
	return t
}
func (t *BoundTree) Root() *big.Int { return t.Levels[len(t.Levels)-1][0] }
func (t *BoundTree) Path(index int) []*big.Int {
	p := make([]*big.Int, len(t.Levels)-1)
	for i := range p {
		p[i] = t.Levels[i][index^1]
		index /= 2
	}
	return p
}

// BoundStats is an integer specification, not rounded floating-point energy.
// H = T B(x-c0), R = T^2(x-c0)-B^T B(x-c0), E = ||R||^2.
// B/T approximates the original orthonormal basis. Its approximation error
// must be measured/certified separately; no exact-real claim is made here.
func BoundStats(m BoundModel, x []int64) (h []*big.Int, e *big.Int, tail []*big.Int) {
	e = bi(0)
	w := make([]*big.Int, m.D)
	for t := range w {
		w[t] = bi(x[t] - m.Centers[0][t])
	}
	if m.Method == "full" {
		h = make([]*big.Int, m.D)
		for t := range h {
			h[t] = bi(x[t])
		}
		return
	}
	if m.Method == "cached" {
		best, winner := bi(0), 0
		for j := 1; j < m.K; j++ {
			score := bi(0)
			for t, v := range w {
				delta := bi(m.Centers[j][t] - m.Centers[0][t])
				score.Add(score, new(big.Int).Sub(mul(bi(2), mul(v, delta)), square(delta)))
			}
			if score.Cmp(best) > 0 {
				best, winner = score, j
			}
		}
		for _, v := range w {
			e.Add(e, square(v))
		}
		e.Sub(e, best)
		h = []*big.Int{bi(int64(winner))}
		return
	}
	if m.Method == "affine" {
		h = make([]*big.Int, m.K-1)
		for j := range h {
			h[j] = bi(0)
			for t := range w {
				v := bi(m.Centers[j+1][t] - m.Centers[0][t])
				h[j].Add(h[j], new(big.Int).Sub(mul(bi(2), mul(w[t], v)), square(v)))
			}
		}
		for _, v := range w {
			e.Add(e, square(v))
		}
		return
	}
	dot := make([]*big.Int, len(m.Basis))
	h = make([]*big.Int, len(dot))
	for j, row := range m.Basis {
		dot[j] = bi(0)
		for t, b := range row {
			dot[j].Add(dot[j], mul(bi(b), w[t]))
		}
		h[j] = mul(bi(m.BasisScale), dot[j])
	}
	if !m.HasEnergy() && !m.UsesTailQuery() {
		return
	}
	tail = make([]*big.Int, m.TailWidth)
	for i := range tail {
		tail[i] = bi(0)
	}
	for t := range w {
		r := mul(square(bi(m.BasisScale)), w[t])
		for j, row := range m.Basis {
			r.Sub(r, mul(bi(row[t]), dot[j]))
		}
		if m.HasEnergy() {
			e.Add(e, square(r))
		}
		if m.TailWidth > 0 {
			tail[m.TailRows[t]].Add(tail[m.TailRows[t]], mul(bi(m.TailSigns[t]), r))
		}
	}
	return
}

type BoundPrepared struct {
	M                       BoundModel
	X                       [][]int64
	H, T                    [][]*big.Int
	HCenters                [][]*big.Int
	E                       []*big.Int
	SaltX, SaltZ            *big.Int
	TX, TH, TE, TT          *BoundTree
	CX, CH, CE, CT, Context *big.Int
}

func PrepareBound(m BoundModel, x [][]int64, saltX, saltZ *big.Int) (*BoundPrepared, error) {
	if err := m.Validate(); err != nil {
		return nil, err
	}
	if len(x) != m.N {
		return nil, fmt.Errorf("dataset row count mismatch")
	}
	p := &BoundPrepared{M: m, X: x, SaltX: saltX, SaltZ: saltZ, H: make([][]*big.Int, m.N), T: make([][]*big.Int, m.N), E: make([]*big.Int, m.N), CE: bi(0), CT: bi(0)}
	p.HCenters = make([][]*big.Int, m.K)
	for j, row := range m.Centers {
		p.HCenters[j], _, _ = BoundStats(m, row)
	}
	modelTag := m.Hash()
	lx := make([]*big.Int, m.N)
	lh := make([]*big.Int, m.N)
	le := make([]*big.Int, m.N)
	lt := make([]*big.Int, m.N)
	max := int64(1) << uint(m.InputBits)
	for i, row := range x {
		if len(row) != m.D {
			return nil, fmt.Errorf("dataset width mismatch")
		}
		xs := make([]*big.Int, m.D)
		for t, v := range row {
			if v < -max || v > max {
				return nil, fmt.Errorf("input outside canonical range")
			}
			xs[t] = bi(v)
		}
		p.H[i], p.E[i], p.T[i] = BoundStats(m, row)
		lx[i] = boundLeaf(m, boundX, i, saltX, xs)
		if m.Method != "full" {
			values := append([]*big.Int{}, p.H[i]...)
			if m.HasEnergy() {
				values = append(values, p.E[i])
			}
			lh[i] = boundLeafTag(m, boundH, i, saltZ, modelTag, values)
		}
		if m.TailWidth > 0 {
			lt[i] = boundLeafTag(m, boundT, i, saltZ, modelTag, p.T[i])
		}
	}
	p.TX = NewBoundTree(lx)
	p.CX = p.TX.Root()
	if m.Method == "full" {
		p.TH = p.TX
		p.CH = p.CX
	} else {
		p.TH = NewBoundTree(lh)
		p.CH = p.TH.Root()
	}
	_ = le // CE is reserved and zero: head+energy use one authenticated row leaf.
	if m.TailWidth > 0 {
		p.TT = NewBoundTree(lt)
		p.CT = p.TT.Root()
	}
	p.Context = boundNativeHash(bi(boundContext), m.Hash(), p.CX, p.CH, p.CE, p.CT)
	return p, nil
}

func (p *BoundPrepared) Answer(indices []int) ([]int, *big.Int, error) {
	labels := make([]int, len(indices))
	sum := bi(0)
	m := p.M
	centers := p.HCenters
	tailCenters := make([][]*big.Int, m.K)
	if m.UsesTailQuery() {
		for j, row := range m.Centers {
			_, _, tailCenters[j] = BoundStats(m, row)
		}
	}
	for a, i := range indices {
		if i < 0 || i >= m.N || (a > 0 && i <= indices[a-1]) {
			return nil, nil, fmt.Errorf("query indices must be strictly increasing and in range")
		}
		best := 0
		var dist *big.Int
		if m.Method == "cached" {
			best = int(p.H[i][0].Int64())
			dist = new(big.Int).Set(p.E[i])
		} else if m.Method == "affine" {
			maximum := bi(0)
			for j := 1; j < m.K; j++ {
				if p.H[i][j-1].Cmp(maximum) > 0 {
					best = j
					maximum = p.H[i][j-1]
				}
			}
			dist = new(big.Int).Sub(p.E[i], maximum)
		} else {
			distance := func(j int) *big.Int {
				s := bi(0)
				for t, v := range p.H[i] {
					s.Add(s, square(new(big.Int).Sub(v, centers[j][t])))
				}
				if m.UsesTailQuery() {
					for t, v := range p.T[i] {
						s.Add(s, square(new(big.Int).Sub(v, tailCenters[j][t])))
					}
				}
				return s
			}
			dist = distance(0)
			for j := 1; j < m.K; j++ {
				d := distance(j)
				if d.Cmp(dist) < 0 {
					best = j
					dist = d
				}
			}
			if m.HasEnergy() {
				dist.Add(dist, p.E[i])
			}
		}
		labels[a] = best
		sum.Add(sum, dist)
	}
	return labels, sum, nil
}
