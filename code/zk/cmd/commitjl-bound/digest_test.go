package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"os"
	"path/filepath"
	"testing"
)

func TestDigestPreservesFilePins(t *testing.T) {
	path := filepath.Join(t.TempDir(), "key.bin")
	// Cross multiple streaming blocks, including non-text bytes.
	data := bytes.Repeat([]byte{0, 255, 13, 10, 127}, 20000)
	if err := os.WriteFile(path, data, 0600); err != nil {
		t.Fatal(err)
	}
	want := sha256.Sum256(data)
	got, err := digest(path)
	if err != nil || got != hex.EncodeToString(want[:]) {
		t.Fatalf("file pin changed: %s, %v", got, err)
	}
	if _, err = digest(path + ".missing"); err == nil {
		t.Fatal("missing key file accepted")
	}
}

func BenchmarkKeyDigest(b *testing.B) {
	path := os.Getenv("COMMITJL_BENCH_DIGEST_FILE")
	if path == "" {
		b.Skip("explicit real key file required")
	}
	b.Run("ReadFile", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			data, err := os.ReadFile(path)
			if err != nil {
				b.Fatal(err)
			}
			sha256.Sum256(data)
		}
	})
	b.Run("Streaming", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			if _, err := digest(path); err != nil {
				b.Fatal(err)
			}
		}
	})
}
