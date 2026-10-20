package main

import (
	"strings"
	"testing"
)

func TestPrefixCannotGenerateProof(t *testing.T) {
	for _, prefix := range []string{"range", "theta", "data", "matrix", "projection", "invalid"} {
		result := run(&Instance{DiagnosticPrefix: prefix})
		if !strings.Contains(result.Error, "proofs are forbidden") || result.VerifyOK || result.SuccessfulProofs != 0 {
			t.Fatalf("prefix %s could enter proving: %+v", prefix, result)
		}
	}
}
