package fftconvolve

import (
	"math"
	"testing"
)

func naive(a, b []float64) []float64 {
	out := make([]float64, len(a)+len(b)-1)
	for i, left := range a {
		for j, right := range b {
			out[i+j] += left * right
		}
	}
	return out
}

func TestShapeErrors(t *testing.T) {
	if _, err := Convolve(nil, []float64{1}); err == nil {
		t.Fatal("空输入应该报错")
	}
	if _, err := Convolve([]float64{1}, nil); err == nil {
		t.Fatal("空输入应该报错")
	}
}

func TestLengthMatchesNaive(t *testing.T) {
	out, err := Convolve([]float64{1, 2, 3, 4}, []float64{4, 5, 6, 7})
	if err != nil {
		t.Fatalf("Convolve 出错：%v", err)
	}
	if len(out) != 7 {
		t.Fatalf("卷积长度应该是 7：%d", len(out))
	}
}

func TestZeroInputGivesZeros(t *testing.T) {
	out, err := Convolve([]float64{0, 0, 0, 0}, []float64{1, 1, 1, 1})
	if err != nil {
		t.Fatalf("Convolve 出错：%v", err)
	}
	for _, value := range out {
		if math.IsNaN(value) {
			t.Fatalf("全零输入不该给 NaN：%v", out)
		}
	}
}

func TestNextPowerOfTwo(t *testing.T) {
	if nextPowerOfTwo(5) != 8 || nextPowerOfTwo(8) != 8 || nextPowerOfTwo(1) != 1 {
		t.Fatalf("nextPowerOfTwo 不对：%d %d %d",
			nextPowerOfTwo(5), nextPowerOfTwo(8), nextPowerOfTwo(1))
	}
}

func TestNaiveHelper(t *testing.T) {
	got := naive([]float64{1, 2}, []float64{3, 4})
	if len(got) != 3 || got[0] != 3 || got[2] != 8 {
		t.Fatalf("朴素卷积参考实现不对：%v", got)
	}
}
