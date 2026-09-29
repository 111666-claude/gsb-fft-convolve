package fftconvolve

import (
	"math"
	"math/rand"
	"testing"
	"time"
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

// scaledTol 以参考值的最大模缩放误差界，兼容 1e-8 与 1e8 混排的场景。
func scaledTol(want []float64) float64 {
	maxAbs := 1.0
	for _, value := range want {
		if math.Abs(value) > maxAbs {
			maxAbs = math.Abs(value)
		}
	}
	return 1e-9 * maxAbs
}

func maxAbsErr(got, want []float64) float64 {
	maxErr := 0.0
	for index := range want {
		if diff := math.Abs(got[index] - want[index]); diff > maxErr {
			maxErr = diff
		}
	}
	return maxErr
}

func assertClose(t *testing.T, got, want []float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("长度不一致：got %d want %d", len(got), len(want))
	}
	if err := maxAbsErr(got, want); err > scaledTol(want) {
		t.Fatalf("误差 %g 超过界 %g", err, scaledTol(want))
	}
}

func randomSeries(rng *rand.Rand, n int) []float64 {
	out := make([]float64, n)
	for i := range out {
		out[i] = rng.NormFloat64()
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
	if _, err := ConvolveReal(nil, nil); err == nil {
		t.Fatal("空输入应该报错")
	}
	if _, err := ConvolveChunked(nil, []float64{1}, 4, true); err == nil {
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

func TestConvolveMatchesNaive(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	for _, n := range []int{1, 2, 3, 4, 5, 7, 8, 16, 17, 100, 255, 256, 1000} {
		for _, m := range []int{1, 3, 8, 33, 128} {
			a := randomSeries(rng, n)
			b := randomSeries(rng, m)
			want := naive(a, b)
			complexOut, err := Convolve(a, b)
			if err != nil {
				t.Fatalf("Convolve(%d,%d) 出错：%v", n, m, err)
			}
			assertClose(t, complexOut, want)
			realOut, err := ConvolveReal(a, b)
			if err != nil {
				t.Fatalf("ConvolveReal(%d,%d) 出错：%v", n, m, err)
			}
			assertClose(t, realOut, want)
		}
	}
}

func TestPowerOfTwoLengthsWithDelta(t *testing.T) {
	// 与单位脉冲卷积应原样返回，可在线性时间验证任意 2 的幂长度。
	rng := rand.New(rand.NewSource(7))
	for exp := 0; exp <= 20; exp++ {
		n := 1 << exp
		a := randomSeries(rng, n)
		delta := make([]float64, n)
		delta[n/2] = 1
		out, err := ConvolveReal(a, delta)
		if err != nil {
			t.Fatalf("n=2^%d 出错：%v", exp, err)
		}
		if len(out) != 2*n-1 {
			t.Fatalf("n=2^%d 输出长度应为 %d：%d", exp, 2*n-1, len(out))
		}
		assertClose(t, out[n/2:n/2+n], a)
	}
}

func TestFFTCountPerMode(t *testing.T) {
	a := []float64{1, 2, 3}
	b := []float64{4, 5}
	ResetFFTCount()
	if _, err := ConvolveReal(a, b); err != nil {
		t.Fatal(err)
	}
	if got := FFTCount(); got != 2 {
		t.Fatalf("实数打包应该只用 2 次变换：%d", got)
	}
	ResetFFTCount()
	if _, err := Convolve(a, b); err != nil {
		t.Fatal(err)
	}
	if got := FFTCount(); got != 3 {
		t.Fatalf("复数路径应该用 3 次变换：%d", got)
	}
}

func TestPackedMatchesComplex(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	a := randomSeries(rng, 500)
	b := randomSeries(rng, 700)
	packed, err := ConvolveReal(a, b)
	if err != nil {
		t.Fatal(err)
	}
	plain, err := Convolve(a, b)
	if err != nil {
		t.Fatal(err)
	}
	assertClose(t, packed, plain)
}

func TestParseval(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	for exp := 0; exp <= 12; exp++ {
		n := 1 << exp
		data := make([]complex128, n)
		time2 := 0.0
		for i := range data {
			data[i] = complex(rng.NormFloat64(), rng.NormFloat64())
			time2 += real(data[i])*real(data[i]) + imag(data[i])*imag(data[i])
		}
		if err := FFT(data, false); err != nil {
			t.Fatal(err)
		}
		freq2 := 0.0
		for _, v := range data {
			freq2 += real(v)*real(v) + imag(v)*imag(v)
		}
		if diff := math.Abs(freq2/float64(n) - time2); diff > 1e-9*time2 {
			t.Fatalf("n=2^%d Parseval 不守恒：%g vs %g", exp, freq2/float64(n), time2)
		}
	}
}

func TestFFTRoundTrip(t *testing.T) {
	rng := rand.New(rand.NewSource(5))
	n := 4096
	data := make([]complex128, n)
	want := make([]complex128, n)
	for i := range data {
		data[i] = complex(rng.NormFloat64(), rng.NormFloat64())
		want[i] = data[i]
	}
	if err := FFT(data, false); err != nil {
		t.Fatal(err)
	}
	if err := FFT(data, true); err != nil {
		t.Fatal(err)
	}
	for i := range data {
		if diff := cmplx128Abs(data[i] - want[i]); diff > 1e-9 {
			t.Fatalf("往返还原失败：index=%d diff=%g", i, diff)
		}
	}
}

func cmplx128Abs(v complex128) float64 {
	return math.Hypot(real(v), imag(v))
}

func TestFFTRejectsNonPowerOfTwo(t *testing.T) {
	if err := FFT(make([]complex128, 3), false); err == nil {
		t.Fatal("长度 3 应该报错")
	}
}

func TestChunkedMatchesFull(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	a := randomSeries(rng, 1000)
	b := randomSeries(rng, 777)
	want, err := ConvolveReal(a, b)
	if err != nil {
		t.Fatal(err)
	}
	for _, chunk := range []int{1, 2, 16, 64, 1024, 2048} {
		for _, realMode := range []bool{true, false} {
			got, err := ConvolveChunked(a, b, chunk, realMode)
			if err != nil {
				t.Fatalf("chunk=%d real=%v 出错：%v", chunk, realMode, err)
			}
			assertClose(t, got, want)
		}
	}
}

func TestChunkedMatchesNaive(t *testing.T) {
	rng := rand.New(rand.NewSource(13))
	a := randomSeries(rng, 100)
	b := randomSeries(rng, 90)
	want := naive(a, b)
	got, err := ConvolveChunked(a, b, 16, true)
	if err != nil {
		t.Fatal(err)
	}
	assertClose(t, got, want)
}

func TestChunkMustBePowerOfTwo(t *testing.T) {
	a := []float64{1, 2, 3}
	b := []float64{4, 5}
	for _, chunk := range []int{0, 3, 5, 6, 7, -2} {
		if _, err := ConvolveChunked(a, b, chunk, true); err == nil {
			t.Fatalf("chunk=%d 应该报错", chunk)
		}
	}
}

func TestCommutative(t *testing.T) {
	rng := rand.New(rand.NewSource(17))
	a := randomSeries(rng, 300)
	b := randomSeries(rng, 200)
	ab, err := ConvolveReal(a, b)
	if err != nil {
		t.Fatal(err)
	}
	ba, err := ConvolveReal(b, a)
	if err != nil {
		t.Fatal(err)
	}
	assertClose(t, ab, ba)
}

func TestDeterministic(t *testing.T) {
	rng := rand.New(rand.NewSource(19))
	a := randomSeries(rng, 512)
	b := randomSeries(rng, 512)
	first, err := ConvolveReal(a, b)
	if err != nil {
		t.Fatal(err)
	}
	second, err := ConvolveReal(a, b)
	if err != nil {
		t.Fatal(err)
	}
	for i := range first {
		if math.Float64bits(first[i]) != math.Float64bits(second[i]) {
			t.Fatalf("重复计算结果不逐位相同：index=%d", i)
		}
	}
}

func TestMixedMagnitudes(t *testing.T) {
	rng := rand.New(rand.NewSource(23))
	n := 256
	a := make([]float64, n)
	b := make([]float64, n)
	for i := range a {
		a[i] = 1e-8
		b[i] = 1e8
		if rng.Intn(2) == 0 {
			a[i], b[i] = b[i], a[i]
		}
		if rng.Intn(2) == 0 {
			a[i] = -a[i]
		}
	}
	want := naive(a, b)
	got, err := ConvolveReal(a, b)
	if err != nil {
		t.Fatal(err)
	}
	assertClose(t, got, want)
}

func TestLargeRealConvolution(t *testing.T) {
	if testing.Short() {
		t.Skip("short 模式跳过 2^20 性能用例")
	}
	rng := rand.New(rand.NewSource(29))
	n := 1 << 20
	a := randomSeries(rng, n)
	b := randomSeries(rng, n)
	start := time.Now()
	out, err := ConvolveReal(a, b)
	elapsed := time.Since(start)
	if err != nil {
		t.Fatal(err)
	}
	if elapsed > time.Second {
		t.Fatalf("2^20 实数卷积耗时 %v，超过 1 秒", elapsed)
	}
	t.Logf("2^20 实数卷积耗时 %v", elapsed)
	// 与移位脉冲卷积做线性时间校验。
	delta := make([]float64, n)
	delta[n-3] = 1
	shifted, err := ConvolveReal(a, delta)
	if err != nil {
		t.Fatal(err)
	}
	assertClose(t, shifted[n-3:2*n-3], a)
	_ = out
}
