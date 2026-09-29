// Package fftconvolve 实现 FFT 与实数卷积（迭代 Cooley-Tukey），只用标准库。
package fftconvolve

import (
	"errors"
	"math"
	"math/cmplx"
	"sync/atomic"
)

// ErrShape 表示输入不合法（空输入或长度不是 2 的幂）。
var ErrShape = errors.New("fftconvolve: 输入不合法")

var fftCount int64

// FFTCount 返回累计执行的 FFT 次数（前向与反向都算）。
func FFTCount() int { return int(atomic.LoadInt64(&fftCount)) }

// ResetFFTCount 清零 FFT 计数。
func ResetFFTCount() { atomic.StoreInt64(&fftCount, 0) }

// nextPowerOfTwo 返回不小于 n 的 2 的幂。
func nextPowerOfTwo(n int) int {
	size := 1
	for size < n {
		size *= 2
	}
	return size
}

// FFT 就地迭代 Cooley-Tukey 变换；inverse 时结果除以 len(data) 归一化。
// 长度必须是 2 的幂，否则返回 ErrShape。
func FFT(data []complex128, inverse bool) error {
	size := len(data)
	if size == 0 {
		return nil
	}
	if size&(size-1) != 0 {
		return ErrShape
	}
	atomic.AddInt64(&fftCount, 1)

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

	// 预计算旋转因子表，避免蝶形内连乘带来的误差累积。
	roots := make([]complex128, size/2)
	for k := range roots {
		angle := 2 * math.Pi * float64(k) / float64(size)
		if inverse {
			angle = -angle
		}
		roots[k] = cmplx.Rect(1, angle)
	}
	for length := 2; length <= size; length *= 2 {
		step := size / length
		for start := 0; start < size; start += length {
			for offset := 0; offset < length/2; offset++ {
				factor := roots[offset*step]
				even := data[start+offset]
				odd := data[start+offset+length/2] * factor
				data[start+offset] = even + odd
				data[start+offset+length/2] = even - odd
			}
		}
	}
	if inverse {
		scale := complex(1/float64(size), 0)
		for index := range data {
			data[index] *= scale
		}
	}
	return nil
}

// Convolve 返回 a 与 b 的线性卷积（复数路径：两次前向 + 一次反向）。
func Convolve(a, b []float64) ([]float64, error) {
	if len(a) == 0 || len(b) == 0 {
		return nil, ErrShape
	}
	size := nextPowerOfTwo(len(a) + len(b) - 1)
	left := make([]complex128, size)
	right := make([]complex128, size)
	for index, value := range a {
		left[index] = complex(value, 0)
	}
	for index, value := range b {
		right[index] = complex(value, 0)
	}
	if err := FFT(left, false); err != nil {
		return nil, err
	}
	if err := FFT(right, false); err != nil {
		return nil, err
	}
	for index := range left {
		left[index] *= right[index]
	}
	if err := FFT(left, true); err != nil {
		return nil, err
	}
	out := make([]float64, len(a)+len(b)-1)
	for index := range out {
		out[index] = real(left[index])
	}
	return out, nil
}

// ConvolveReal 实数打包路径：把 a、b 打包成一条复数序列 z = a + i·b，
// 一次前向 FFT 后用共轭对称拆出两条频谱，再一次反向 FFT。共 2 次变换。
func ConvolveReal(a, b []float64) ([]float64, error) {
	if len(a) == 0 || len(b) == 0 {
		return nil, ErrShape
	}
	size := nextPowerOfTwo(len(a) + len(b) - 1)
	packed := make([]complex128, size)
	for index, value := range a {
		packed[index] = complex(value, 0)
	}
	for index, value := range b {
		packed[index] += complex(0, value)
	}
	if err := FFT(packed, false); err != nil {
		return nil, err
	}
	spectrum := make([]complex128, size)
	for k := 0; k < size; k++ {
		mirror := cmplx.Conj(packed[(size-k)%size])
		fa := 0.5 * (packed[k] + mirror)
		fb := -0.5i * (packed[k] - mirror)
		spectrum[k] = fa * fb
	}
	if err := FFT(spectrum, true); err != nil {
		return nil, err
	}
	out := make([]float64, len(a)+len(b)-1)
	for index := range out {
		out[index] = real(spectrum[index])
	}
	return out, nil
}

// ConvolveChunked 分块 overlap-add 卷积：两条输入都切成 chunk 长的块，
// 逐对块卷积后按偏移叠加。工作内存 O(chunk)。chunk 必须是 2 的幂。
// realMode 时逐块走实数打包路径（每对块 2 次变换），否则走复数路径。
func ConvolveChunked(a, b []float64, chunk int, realMode bool) ([]float64, error) {
	if len(a) == 0 || len(b) == 0 {
		return nil, ErrShape
	}
	if chunk <= 0 || chunk&(chunk-1) != 0 {
		return nil, ErrShape
	}
	out := make([]float64, len(a)+len(b)-1)
	blocksA := (len(a) + chunk - 1) / chunk
	blocksB := (len(b) + chunk - 1) / chunk
	for i := 0; i < blocksA; i++ {
		endA := (i + 1) * chunk
		if endA > len(a) {
			endA = len(a)
		}
		blockA := a[i*chunk : endA]
		for j := 0; j < blocksB; j++ {
			endB := (j + 1) * chunk
			if endB > len(b) {
				endB = len(b)
			}
			blockB := b[j*chunk : endB]
			var conv []float64
			var err error
			if realMode {
				conv, err = ConvolveReal(blockA, blockB)
			} else {
				conv, err = Convolve(blockA, blockB)
			}
			if err != nil {
				return nil, err
			}
			base := (i + j) * chunk
			for k, value := range conv {
				out[base+k] += value
			}
		}
	}
	return out, nil
}
