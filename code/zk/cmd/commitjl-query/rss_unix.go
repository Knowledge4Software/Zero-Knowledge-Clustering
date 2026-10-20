//go:build !windows

package main

import "syscall"

func getMaxRSS() int64 {
	var usage syscall.Rusage
	if err := syscall.Getrusage(syscall.RUSAGE_SELF, &usage); err != nil {
		return -1
	}
	return int64(usage.Maxrss)
}
