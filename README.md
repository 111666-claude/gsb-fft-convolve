# fft-convolve

FFT 与实数卷积（迭代 Cooley-Tukey），只用 Go 标准库。

```
go test ./...
go run ./cmd/fft-convolve --a 1,2,3 --b 4,5 --check
go run ./cmd/fft-convolve --a 1,2,3,4 --b 4,3,2,1 --check
```

## 口径（README 为准）

- **长度要补到 2 的幂**：卷积长度取 `nextPowerOfTwo(len(a)+len(b)-1)`。
  `--a 1,2,3 --b 4,5 --check` 现在用 `len(a)+len(b) = 5`（不是 2 的幂）直接变换，`max_error` 很大；
  按口径应该是 `out=4,13,22,15`、`max_error ≤ 1e-9`。
- **逆变换要除以长度**：`FFT(..., inverse=true)` 之后每个值都要除以 `size`；
  现在 `--a 1,2,3,4 --b 4,3,2,1` 的结果整体偏大 `size` 倍。
- 数值要求：与朴素 O(n·m) 卷积逐项比对，绝对误差 ≤ 1e-9；允许负数与零；空输入要报错。
- 规模：n = 2^16 的两条序列要在 0.2 秒内算完，内存 O(n)；不许退化成 O(n·m) 的朴素卷积。
- 不变量：`Convolve(a,b)` 与 `Convolve(b,a)` 逐项误差 ≤ 1e-9；同一输入重复计算结果逐位相同；
  `FFT` 后接逆 `FFT` 能还原输入（误差 ≤ 1e-9）。

## 现在的行为

- `Convolve` 的 `size := len(a) + len(b)`：不补到 2 的幂，蝶形合并按 2 的幂展开，结果直接错。
- `FFT` 的 inverse 分支没做归一化：卷积结果整体放大 `size` 倍。
- `nextPowerOfTwo` 写了但没人调用，也没有长度校验。

## 计划新增的能力

```
--a 序列 A      逗号分隔
--b 序列 B      逗号分隔
--check         与朴素卷积比对误差
```

## 输出

```
len=4 out=4,13,22,15
max_error=2.2e-16
```

## 目录

```
fft.go              FFT 与卷积
cmd/fft-convolve    命令行入口
fft_test.go         go test 用例
```
