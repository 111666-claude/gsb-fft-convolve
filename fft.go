// Package fftconvolve 实现 FFT 与实数卷积。
// 缺陷：补零长度直接取 len(a)+len(b)（不是 2 的幂），逆变换也没有除以长度。
package fftconvolve

import (
	"errors"
	"math"
	"math/cmplx"
)

// ErrShape 表示输入不合法。
var ErrShape = errors.New("fftconvolve: 输入不合法")

// nextPowerOfTwo 返回不小于 n 的 2 的幂。
func nextPowerOfTwo(n int) int {
	size := 1
	for size < n {
		size *= 2
	}
	return size
}

// FFT 就地变换。缺陷：inverse 时没有除以 size。
func FFT(data []complex128, inverse bool) {
	size := len(data)
	if size == 0 || size&(size-1) != 0 {
		return // 缺陷：长度不是 2 的幂就直接返回，变换根本没做
	}
	for index, reversed := 1, 0; index < size; index++ {
		bit := size >> 1
		for ; reversed&bit != 0; bit >>= 1 {
			reversed ^= bit
		}
		reversed ^= bit
		if index < reversed {
			data[index], data[reversed] = data[reversed], data[index]
		}
	}
	for length := 2; length <= size; length *= 2 {
		angle := 2 * math.Pi / float64(length)
		if inverse {
			angle = -angle
		}
		root := cmplx.Rect(1, angle)
		for start := 0; start < size; start += length {
			factor := complex(1, 0)
			for offset := 0; offset < length/2; offset++ {
				even := data[start+offset]
				odd := data[start+offset+length/2] * factor
				data[start+offset] = even + odd
				data[start+offset+length/2] = even - odd
				factor *= root
			}
		}
	}
}

// Convolve 返回 a 与 b 的线性卷积。
// 缺陷：size 取 len(a)+len(b)（不是 2 的幂），逆变换也没有归一化。
func Convolve(a, b []float64) ([]float64, error) {
	if len(a) == 0 || len(b) == 0 {
		return nil, ErrShape
	}
	size := len(a) + len(b)
	left := make([]complex128, size)
	right := make([]complex128, size)
	for index, value := range a {
		left[index] = complex(value, 0)
	}
	for index, value := range b {
		right[index] = complex(value, 0)
	}
	FFT(left, false)
	FFT(right, false)
	for index := range left {
		left[index] *= right[index]
	}
	FFT(left, true)
	out := make([]float64, len(a)+len(b)-1)
	for index := range out {
		out[index] = real(left[index])
	}
	return out, nil
}
