package circuit

// Public statement builders let the verifier reconstruct exactly the expected
// public witness without reading private original or compressed rows.

import (
	"fmt"
	"github.com/consensys/gnark-crypto/ecc/bn254/fr"
	"math/big"
)

type BoundStatement struct {
	CX      string `json:"cx"`
	CH      string `json:"ch"`
	CE      string `json:"ce"`
	CT      string `json:"ct"`
	Context string `json:"context"`
}

func (p *BoundPrepared) Statement() BoundStatement {
	return BoundStatement{p.CX.String(), p.CH.String(), p.CE.String(), p.CT.String(), p.Context.String()}
}
func (s BoundStatement) Values(m BoundModel) (cx, ch, ce, ct, context *big.Int, err error) {
	strings := []string{s.CX, s.CH, s.CE, s.CT, s.Context}
	values := make([]*big.Int, 5)
	for i, text := range strings {
		v, ok := new(big.Int).SetString(text, 10)
		if !ok || v.Sign() < 0 || v.Cmp(fr.Modulus()) >= 0 || v.String() != text {
			return nil, nil, nil, nil, nil, fmt.Errorf("invalid canonical public field value")
		}
		values[i] = v
	}
	cx, ch, ce, ct, context = values[0], values[1], values[2], values[3], values[4]
	expected := boundNativeHash(bi(boundContext), m.Hash(), cx, ch, ce, ct)
	if expected.Cmp(context) != 0 {
		return nil, nil, nil, nil, nil, fmt.Errorf("public context does not bind roots/model")
	}
	if ce.Sign() != 0 || (m.TailWidth == 0 && ct.Sign() != 0) || (m.Method == "full" && cx.Cmp(ch) != 0) {
		return nil, nil, nil, nil, nil, fmt.Errorf("invalid roots for method")
	}
	return
}
func BoundPreprocessPublic(m BoundModel, s BoundStatement, index int) (*BoundPreprocess, error) {
	if index < 0 || index >= m.N {
		return nil, fmt.Errorf("invalid preprocessing row")
	}
	cx, ch, ce, ct, ctx, err := s.Values(m)
	if err != nil {
		return nil, err
	}
	c := NewBoundPreprocess(m)
	c.CX, c.CH, c.CE, c.CT, c.Context, c.Index = cx, ch, ce, ct, ctx, index
	return c, nil
}
func BoundQueryPublic(m BoundModel, s BoundStatement, indices []int, psi string) (*BoundQuery, error) {
	cx, ch, ce, ct, ctx, err := s.Values(m)
	if err != nil {
		return nil, err
	}
	value, ok := new(big.Int).SetString(psi, 10)
	if !ok || value.Sign() < 0 || value.Cmp(fr.Modulus()) >= 0 || value.String() != psi {
		return nil, fmt.Errorf("invalid claimed cost")
	}
	if len(indices) < 1 || len(indices) > m.N {
		return nil, fmt.Errorf("invalid query size")
	}
	c := NewBoundQuery(m, len(indices))
	c.CX, c.CH, c.CE, c.CT, c.Context, c.Psi = cx, ch, ce, ct, ctx, value
	for i, index := range indices {
		if index < 0 || index >= m.N || (i > 0 && index <= indices[i-1]) {
			return nil, fmt.Errorf("query indices not strictly ordered and bounded")
		}
		c.Indices[i] = index
	}
	return c, nil
}

// Public-only replay of legacy reuse bundles. Center statistics are recomputed
// from the verifier's intended original-space integer centers, not from claims.
func BoundReusePublic(m BoundModel, s BoundStatement, centers [][]int64, indices []int, psi, margin string) (*BoundReuseQuery, error) {
	if err := m.Validate(); err != nil {
		return nil, err
	}
	if err := validateNewCenters(m, centers); err != nil {
		return nil, err
	}
	if err := validatePublicIndices(m, indices); err != nil {
		return nil, err
	}
	heads, offs, err := ReuseCenterStats(m, centers)
	if err != nil {
		return nil, err
	}
	cx, ch, ce, ct, ctx, err := s.Values(m)
	if err != nil {
		return nil, err
	}
	pv, err := canonicalPublicInt(psi)
	if err != nil {
		return nil, err
	}
	mv, err := canonicalPublicInt(margin)
	if err != nil {
		return nil, err
	}
	_, score, distance := reuseBounds(m)
	if pv.Cmp(mul(distance, bi(int64(len(indices))))) > 0 || mv.Cmp(mul(score, bi(int64(len(indices))))) > 0 {
		return nil, fmt.Errorf("reuse output exceeds integer bound")
	}
	c := NewBoundReuseQuery(m, len(indices), len(centers))
	c.CX, c.CH, c.CE, c.CT, c.Context, c.Psi, c.MarginSum = cx, ch, ce, ct, ctx, pv, mv
	for j, h := range heads {
		for t, v := range h {
			c.CHead[j][t] = v
		}
		c.COff[j] = offs[j]
	}
	for a, i := range indices {
		c.Indices[a] = i
	}
	return c, nil
}
