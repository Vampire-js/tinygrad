# Tiny Autograd engine

## Naive matmul perf stats:
```
    10,084,779,752      cpu_atom/cycles/u                                                       (0.81%)
    11,269,471,394      cpu_core/cycles/u                                                       (99.19%)
    34,316,533,823      cpu_atom/instructions/u                                                 (0.81%)
    44,692,220,969      cpu_core/instructions/u                                                 (99.19%)
    11,677,073,450      cpu_atom/L1-dcache-loads/u                                              (0.81%)
    15,121,777,556      cpu_core/L1-dcache-loads/u                                              (99.19%)
   <not supported>      cpu_atom/L1-dcache-load-misses/u
        10,308,036      cpu_core/L1-dcache-load-misses/u                                        (99.19%)
           396,204      cpu_atom/LLC-loads/u                                                    (0.81%)
           196,464      cpu_core/LLC-loads/u                                                    (99.19%)
               859      cpu_atom/LLC-load-misses/u                                              (0.81%)
           189,615      cpu_core/LLC-load-misses/u                                              (99.19%)

       7.884701188 seconds time elapsed
```
So well, big misconception is that tiling is done to solve the cache problem, but actually there is no cache problem, CPU prefetching is good enough hence we anyway's dont get many L1 misses.

For this test, we have $2^{20}$ number's being streamed (each row from A). So that is:
$$
2^{20}*8B/1024 = 8MB >> 32KB
$$
So technically it doesn't fit in L1 at all, but as these are sequential accesses, CPU prefetching does its job well and we get the numbers we do in perf.

So, why would I need a tiling based optimisation now? Well for B (or even A) we still need to go down to lower cache levels, or even DRAM. So question is can we potentially avoid that too.

