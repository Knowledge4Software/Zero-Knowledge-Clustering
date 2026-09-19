// Golden vector generator: dumps gnark-crypto bn254 MiMC hash outputs for a
// set of test vectors that mirror commitjl/hashing.py and commitjl/commit.py,
// so the Python implementation can be cross-checked byte-exactly.
package main

import (
	"encoding/json"
	"fmt"
	"math/big"
	"os"

	"github.com/consensys/gnark-crypto/ecc/bn254/fr"
	"github.com/consensys/gnark-crypto/ecc/bn254/fr/mimc"
)

func feFromInt(s string) fr.Element {
	var e fr.Element
	bi, ok := new(big.Int).SetString(s, 10)
	if !ok {
		panic("bad int: " + s)
	}
	e.SetBigInt(bi)
	return e
}

// mimcHash replicates commitjl.hashing.mimc_hash: sponge/compression over a
// sequence of field elements using gnark-crypto's MiMC FieldHasher.
func mimcHash(inputs []fr.Element) *big.Int {
	h := mimc.NewFieldHasher()
	for _, e := range inputs {
		h.WriteElement(e)
	}
	r := h.SumElement()
	bi := new(big.Int)
	r.BigInt(bi)
	return bi
}

func mimcHashStrs(strs []string) string {
	elems := make([]fr.Element, len(strs))
	for i, s := range strs {
		elems[i] = feFromInt(s)
	}
	return mimcHash(elems).String()
}

// bytesToFieldElements replicates mimc_hash_bytes' chunking: 31-byte
// big-endian chunks.
func bytesToFieldElements(data []byte) []fr.Element {
	const chunk = 31
	var elems []fr.Element
	for i := 0; i < len(data); i += chunk {
		end := i + chunk
		if end > len(data) {
			end = len(data)
		}
		bi := new(big.Int).SetBytes(data[i:end])
		var e fr.Element
		e.SetBigInt(bi)
		elems = append(elems, e)
	}
	if len(elems) == 0 {
		elems = append(elems, fr.Element{})
	}
	return elems
}

func main() {
	out := map[string]interface{}{}

	// 1. Round constants (for direct cross-check).
	constants := mimc.GetConstants()
	constStrs := make([]string, len(constants))
	for i := range constants {
		c := constants[i]
		constStrs[i] = c.String()
	}
	out["round_constants"] = constStrs

	// 2. Single field element hashes.
	singles := []string{"0", "1", "2", "12345",
		"21888242871839275222246405745257275088548364400416034343698204186575808495616"} // p-1
	singleResults := map[string]string{}
	for _, s := range singles {
		singleResults[s] = mimcHashStrs([]string{s})
	}
	out["single"] = singleResults

	// 3. Sequences.
	seqs := [][]string{
		{"1", "2"},
		{"1", "2", "3"},
		{"7", "11", "13", "17", "19"},
		{"0", "0"},
	}
	seqResults := []map[string]interface{}{}
	for _, seq := range seqs {
		seqResults = append(seqResults, map[string]interface{}{
			"inputs": seq,
			"digest": mimcHashStrs(seq),
		})
	}
	out["sequences"] = seqResults

	// 4. Leaf hash: mimc_hash([value, salt]).
	leafCases := [][2]string{
		{"5", "999"},
		{"0", "123456789"},
		{"21888242871839275222246405745257275088548364400416034343698204186575808495616", "42"},
	}
	leafResults := []map[string]interface{}{}
	for _, c := range leafCases {
		leafResults = append(leafResults, map[string]interface{}{
			"value":  c[0],
			"salt":   c[1],
			"digest": mimcHashStrs([]string{c[0], c[1]}),
		})
	}
	out["leaf_hash"] = leafResults

	// 5. Node hash: mimc_hash([left, right]).
	nodeResults := []map[string]interface{}{}
	nodeCases := [][2]string{
		{"111", "222"},
		{"0", "0"},
	}
	for _, c := range nodeCases {
		nodeResults = append(nodeResults, map[string]interface{}{
			"left":   c[0],
			"right":  c[1],
			"digest": mimcHashStrs([]string{c[0], c[1]}),
		})
	}
	out["node_hash"] = nodeResults

	// 6. Full Merkle root for a small 2x2 signed-int matrix X, matching
	// commit.py's commit_matrix: leaves = signed_to_field(flatten(X)), pad
	// to next pow2 with 0, leaf_hash = mimc_hash([value, salt]), tree via
	// node_hash = mimc_hash([left,right]).
	p, _ := new(big.Int).SetString("21888242871839275222246405745257275088548364400416034343698204186575808495617", 10)
	signedToField := func(x int64) *big.Int {
		if x < 0 {
			return new(big.Int).Add(p, big.NewInt(x))
		}
		return big.NewInt(x)
	}
	Xflat := []int64{3, -5, 0, 127} // 2x2 matrix, row-major
	salt := big.NewInt(777)
	leaves := make([]*big.Int, len(Xflat))
	for i, v := range Xflat {
		leaves[i] = signedToField(v)
	}
	// N=4 already pow2, no padding needed.
	leafHashes := make([]*big.Int, len(leaves))
	for i, v := range leaves {
		leafHashes[i] = mimcHash([]fr.Element{feFromInt(v.String()), feFromInt(salt.String())})
	}
	level := leafHashes
	for len(level) > 1 {
		var next []*big.Int
		for i := 0; i < len(level); i += 2 {
			next = append(next, mimcHash([]fr.Element{feFromInt(level[i].String()), feFromInt(level[i+1].String())}))
		}
		level = next
	}
	merkleLeafStrs := make([]string, len(leaves))
	for i, v := range leaves {
		merkleLeafStrs[i] = v.String()
	}
	out["merkle_example"] = map[string]interface{}{
		"X_flat":  Xflat,
		"salt":    salt.String(),
		"leaves":  merkleLeafStrs,
		"root":    level[0].String(),
	}

	// 7. mimc_hash_bytes over some byte strings, matching hashing.mimc_hash_bytes
	// (31-byte big-endian chunking).
	byteCases := [][]byte{
		[]byte("CommitJL-v1THETA-example-bytes-1234567890abcdef"),
		[]byte(""),
		[]byte("short"),
	}
	byteResults := []map[string]interface{}{}
	for _, bc := range byteCases {
		elems := bytesToFieldElements(bc)
		byteResults = append(byteResults, map[string]interface{}{
			"hex":    fmt.Sprintf("%x", bc),
			"digest": mimcHash(elems).String(),
		})
	}
	out["bytes_hash"] = byteResults

	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	if err := enc.Encode(out); err != nil {
		panic(err)
	}
}
