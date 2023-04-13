# Homework 1

PB20111689 蓝俊玮

## Problem 1

> （Activity 1，问题 3）改写求最大值问题的并行算法，要求不使用数组 M

假设数组 `A[n]` 为待求最大值的数组，且我们拥有 $n^2$ 个处理器，则算法如下：

```
for i=1 to n par-do
	B[i][1] = 1
for i=1 to n par-do
	for j=1 to n par-do
		if A[i] < A[j] then
			B[i][1] = 0
for i=1 to n par-do
	if B[i][1] == 1 then
		return 1
```

首先这里的 `B[n][n]` 数组为之前课上描述的数组，与课上讲的算法不同之处为，这里仅仅使用 `B[i][1]`，即使用 `B` 数组的第一列，而不再使用 `M` 数组（但实际上使用 `B` 数组来实现 `M` 数组的功能）。

算法的思想为并行地判断是否有比第 `i` 个数字更大的数字，如果有，则将 `B[i][1]` 置为 `0`，只有最大的数字才能满足没有比自己更大的数字，那么最大的数字的 `B[i][1]` 将会保持为 `1`。因为上述循环都是并行地执行，因此可以在 $O(1)$ 的时间内找到最大值。

## Problem 2

> 课本 Ex 5.6
>
> 1. 试用 APRAM 模型之参数，写出算法地时间复杂度函数表达式
> 2. 试解释 `Barrier` 语句的作用

1. 设局部操作为单位时间；全局读/写平均时间为 `d`，路障同步时间 `B(p)` 为非降函数且为 $O(d\log p)$。

   - 首先各处理器求 `n/p` 各数的局和，并写入 SM 中，即进行全局写操作。由于这个是并行操作的，则时间为 $O(n/p+d)$

   - 接着进行一次 Barrier，时间为 $O(d\log p)$

   - 然后是每个处理器 $P_i$ 要处理其 `B` 个孩子的局和与自身的和结果（这里需要全局读取，则时间为 $O(Bd)$），并将结果写入 SM 中，最后还要进行一次路障同步。因为上述操作总共有 $\bigg\lceil\log_B\big(p(B-1)+1\big)\bigg\rceil-1$ 次操作， 因此需要：
     $$
     O\bigg(\bigg(\bigg\lceil\log_B\big(p(B-1)+1\big)\bigg\rceil-1\bigg)\bigg(Bd+d\log p\bigg)\bigg)
     $$

   那么根据上述操作，时间复杂度为：
   $$
   O(n/p+d)+O(d\log p)+O\bigg(\bigg(\bigg\lceil\log_B\big(p(B-1)+1\big)\bigg\rceil-1\bigg)\bigg(Bd+d\log p\bigg)\bigg)\\
   =O\bigg(n/p+\bigg\lceil\log_B\big(p(B-1)+1\big)\bigg\rceil(B+\log p)d\bigg)
   $$

2. Barrier 是同步路障。因为不同处理器的处理速度是不相同的，且不同处理器之间处理的数据是有依赖关系的，因此使用同步路障，可以让先运行完的处理器停在这里，等待其它处理器完成操作。这样就通过 Barrier 可以确保所有处理器已经完成计算，这样就能够保证数据的一致性，即能保证每个处理器正确读取到下一层的计算结果，而不会读取到未经计算的结果。

## Problem 3

> 课本 Ex 5.7
>
> 1. 试分析算法 5.5 的时间复杂度
> 2. $d$ 值如何确定

1. 首先，各个处理器求 $n/p$ 个数的局和，需要时间为 $O(n/p)$，且每个处理器只发送 1 个信包数，因此超级计算步成本为 $O(n/p+g+L)$，其中 $g$ 为带宽因子，$L$ 为路障同步时间。然后是上播并求和过程，在这个过程中每次需要接收 $d$ 个孩子消息，则时间为 $O(d+g+L)$，而在这样的操作共有 $\bigg\lceil\log_d\big(p(d-1)+1\big)\bigg\rceil-1$ 次。

   因此总共需要时间为：
   $$
   O\bigg(n/p+g+L+\bigg(\bigg\lceil\log_d\big(p(d-1)+1\big)\bigg\rceil-1\bigg)\bigg(d+g+L\bigg)\bigg)\\
   =O\bigg(n/p+\bigg\lceil\log_d\big(p(d-1)+1\big)\bigg\rceil(d+g+L)\bigg)
   $$

2. 在带宽因子和同步路障时间不变以及处理器数量固定的情况下，可以对上式进行求导取最小值来确定 $d$ 值。

