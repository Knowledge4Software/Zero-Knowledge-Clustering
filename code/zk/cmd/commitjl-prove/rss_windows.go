//go:build windows

package main

// Windows does not expose Unix rusage. HeapSys is still reported separately
// through runtime.ReadMemStats, and the Python bridge measures process RSS.
func getMaxRSS() int64 { return 0 }
