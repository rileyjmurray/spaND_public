# spaND
Repository for the spaND algorithm (https://arxiv.org/abs/1901.02971).

![Partition pic](https://raw.githubusercontent.com/leopoldcambier/spaND_public/stable/partition_64x64.png "Partitioning output")

## Disclaimer
This is very researchy code. 
The goal is to easily experiments various kinds of sparsification techniques, with reasonnable performances.
This is *not* production-ready code. I am not a C++ guru and many things are very imperfect and may break if you try too much :-)

Some short pieces of code (or some examples) have been taken from other open source project. 
This is indicated when needed.

## Build instructions

Necessary:
- Download Eigen (header library only) : http://eigen.tuxfamily.org/index.php?title=Main_Page
- Download, build and install Metis in **Int 64 bits mode** : http://glaros.dtc.umn.edu/gkhome/metis/metis/overview (so that you can pass it `int64_t`s)
- Download, build and install blaspp and lapackpp.

Optional, for testing and custom experiments only:
- Download, build and install Mfem : https://mfem.org/
- Download, build and install googletest : https://github.com/google/googletest

Then

1. Create an `obj/` qnd `build/` directory:

    `mkdir obj`
    `mkdir build`
    
2. Copy one of the file in the `tests/Makefile-confs` folder and create a `tests/Makefile.conf` file with your system settings. For instance 

    `cp tests/Makefile-confs/Makefile-sherlock tests/Makefile.conf`.
    
    You need to set the following variables in `Makefile.conf`:
    
    - `CC` should be your C++ compiler (needs to support -std=c++20)
    - `REQUIRED_INCDIRS` should contains path to the above libraries headers (Eigen, Metis, Blas/Lapack, linalgCpp) if not in your compiler path (may or may not be needed)
    - `REQUIRED_LIBDIRS`should contains path to the above libraries objects (Metis, Blas/Lapack), if not in your compiler path (may or may not be needed)
    - `GTEST` should point to googletest folder (optional)
    - `MFEMDIR` should point to MFEM's root (optional)
    - `MFEMLIB` should point to MFEM's root (optional)
        
3. Then, move to `tests/`. You can create any of the executables with `make target` or all with `make`. 

## Some additional things to do (maybe) when you run:
You'll probably need to add Metis' LIB and blaspp/lapackpp LIB folders to `LD_LIBRARY_PATH` and/or ``DYLD_LIBRARY_PATH``.

## Example

Let's compile the general driver code and run it on a sample arbitrary matrix.
1. Compile the ```spaND.cpp``` example

    ```make spaND```

2. Run it

    ```./spaND -m ../mats/neglapl_2_32.mm --coordinates ../mats/32x32.mm -t 1e-2 -l 5```
    
    You can find more after this function by doing `./spaND --help`.

    This should output something similar to this

```
Gandalf:tests lcambier$ ./spaND -m ../mats/neglapl_2_32.mm -t 1e-2 -l 5
Matrix 1024x1024 loaded from ../mats/neglapl_2_32.mm
A:
4 -1 0 0 0 0 0 0 0 0
-1 4 -1 0 0 0 0 0 0 0
0 -1 4 -1 0 0 0 0 0 0
0 0 -1 4 -1 0 0 0 0 0
0 0 0 -1 4 -1 0 0 0 0
0 0 0 0 -1 4 -1 0 0 0
0 0 0 0 0 -1 4 -1 0 0
0 0 0 0 0 0 -1 4 -1 0
0 0 0 0 0 0 0 -1 4 -1
0 0 0 0 0 0 0 0 -1 4

Aprec:
4 -1 0 0 0 0 0 0 0 0
-1 4 -1 0 0 0 0 0 0 0
0 -1 4 -1 0 0 0 0 0 0
0 0 -1 4 -1 0 0 0 0 0
0 0 0 -1 4 -1 0 0 0 0
0 0 0 0 -1 4 -1 0 0 0
0 0 0 0 0 -1 4 -1 0 0
0 0 0 0 0 0 -1 4 -1 0
0 0 0 0 0 0 0 -1 4 -1
0 0 0 0 0 0 0 0 -1 4

Tree with 5 levels (0 eliminated so far)
MND algebraic (with vertex sep ? 1) partitioning of matrix with 1024 dofs with 5 levels
Algebraic MND partitioning & ordering
  Depth  1: 7.14e-04 s. (    1 separators, [   32    32], mean   32.0)
  Depth  2: 6.31e-04 s. (    2 separators, [   13    13], mean   13.0)
  Depth  3: 7.75e-04 s. (    4 separators, [   10    16], mean   12.2)
  Depth  4: 1.18e-03 s. (    8 separators, [    5    10], mean    7.8)
Clustering size statistics (# of leaf-clusters at each level of the ND hierarchy)
Lvl     Count       Min       Max      Mean
  0        16        30        73        53
  1         8         5        10         8
  2        11         1         9         4
  3         7         1         7         4
  4         9         1         7         4
Hierarchy numbers (# of cluster at each level of the cluster-hierarchy)
  0        51
  1        14
  2         5
  3         1
  4         0
Partitioning time : 3.66e-03 s.
Assembling (Size 1024 with 5 levels and symmetry 1)
Edge size statistics (Leaf-cluster edge size at each level of the ND hierarchy)
Lvl     Count       Min       Max      Mean
  0        87        30      5329       756
  1        31         7       100        41
  2        18         1        81        23
  3        11         1        49        19
  4        12         1        49        16
Edge count statistics (Leaf-cluster edge count at each level of the ND hierarchy)
Lvl     Count       Min       Max      Mean
  0        16         3         8         5
  1         8         3         5         4
  2        11         1         3         2
  3         7         1         3         2
  4         9         1         3         1
Assembly time : 9.11e-04 s. (1.67e-04 permuting A)
Factorization started
  N:          1024
  #levels:    5
  verbose?:   1
  adaptive?:  1
  tol?:       0.01
  #skip:      0
  scale?:     1
  ortho?:     1
  symmetrykd? SPD
  scalingkd?  LLT
  want_spars? 1
  mon cond?   0
  preserving? 0
Level 0, 1024 dofs left, 51 clusters left
  Elim: 3.63e-03 s., 169 dofs left, 35 clusters left
  Scaling: 3.16e-04 s.
  Sparsification: 3.57e-04 s., 147 dofs left, geqp3 2.25e-04, geqrf 0.00e+00, assmb 2.93e-05, buildQ 6.91e-06, scatterQ 0.00e+00, permA 1.93e-05, scatterA 2.55e-05
Level 1, 147 dofs left, 35 clusters left
  Elim: 1.54e-04 s., 99 dofs left, 27 clusters left
  Merge: 9.20e-05 s., 99 dofs left, 14 clusters left
  Scaling: 3.29e-04 s.
  Sparsification: 1.91e-04 s., 66 dofs left, geqp3 1.09e-04, geqrf 0.00e+00, assmb 1.07e-05, buildQ 3.10e-06, scatterQ 0.00e+00, permA 3.50e-05, scatterA 1.12e-05
Level 2, 66 dofs left, 14 clusters left
  Elim: 3.81e-05 s., 39 dofs left, 10 clusters left
  Merge: 2.38e-05 s., 39 dofs left, 5 clusters left
  Scaling: 1.55e-04 s.
  Sparsification: 4.70e-05 s., 23 dofs left, geqp3 2.65e-05, geqrf 0.00e+00, assmb 2.15e-06, buildQ 9.54e-07, scatterQ 0.00e+00, permA 5.72e-06, scatterA 3.34e-06
Level 3, 23 dofs left, 5 clusters left
  Elim: 7.87e-06 s., 15 dofs left, 3 clusters left
  Merge: 1.91e-06 s., 15 dofs left, 1 clusters left
  Scaling: 4.05e-06 s.
  Sparsification: 2.15e-06 s., 0 dofs left, geqp3 0.00e+00, geqrf 0.00e+00, assmb 0.00e+00, buildQ 0.00e+00, scatterQ 0.00e+00, permA 9.54e-07, scatterA 0.00e+00
Level 4, 0 dofs left, 1 clusters left
  Elim: 1.19e-06 s., 0 dofs left, 0 clusters left
  Merge: 0.00e+00 s., 0 dofs left, 0 clusters left
  Scaling: 0.00e+00 s.
  Sparsification: 0.00e+00 s., 0 dofs left, geqp3 0.00e+00, geqrf 0.00e+00, assmb 0.00e+00, buildQ 0.00e+00, scatterQ 0.00e+00, permA 0.00e+00, scatterA 0.00e+00
Factorization: 5.57e-03 s.
&&&& Lvl |         Elim        Scale     Sparsify        Merge |     Preserve        geqp3        geqrf         potf         trsm         gemm       buildq       scattq        assmb       scatta          phi   mergealloc    mergecopy
&&&&   0 | 3.628969e-03 3.159046e-04 3.569126e-04 0.000000e+00 | 0.000000e+00 2.245903e-04 0.000000e+00 2.183199e-03 9.703636e-04 2.005100e-04 6.914139e-06 0.000000e+00 2.932549e-05 2.551079e-05 0.000000e+00 0.000000e+00 0.000000e+00
&&&&   1 | 1.540184e-04 3.290176e-04 1.909733e-04 9.202957e-05 | 0.000000e+00 1.087189e-04 0.000000e+00 1.573563e-05 2.760887e-04 7.605553e-05 3.099442e-06 0.000000e+00 1.072884e-05 1.120567e-05 0.000000e+00 2.098083e-05 2.312660e-05
&&&&   2 | 3.814697e-05 1.549721e-04 4.696846e-05 2.384186e-05 | 0.000000e+00 2.646446e-05 0.000000e+00 8.106232e-06 1.380444e-04 2.217293e-05 9.536743e-07 0.000000e+00 2.145767e-06 3.337860e-06 0.000000e+00 6.914139e-06 7.152557e-06
&&&&   3 | 7.867813e-06 4.053116e-06 2.145767e-06 1.907349e-06 | 0.000000e+00 0.000000e+00 0.000000e+00 3.099442e-06 0.000000e+00 5.006790e-06 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 9.536743e-07
&&&&   4 | 1.192093e-06 0.000000e+00 0.000000e+00 0.000000e+00 | 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00
++++ Lvl        ND    ND lft    El lft    Sp lft   Fct nnz    Rk Bfr    Rk Aft      Nbrs   Tot Bfr   Tot Aft   Cl Sped  CondDiag   NormDiag
++++   0       855       169       169       147   4.5e+04         5         4        31       166       144        32   0.0e+00    5.5e+00
++++   1        62       107        99        66   4.7e+04         8         5        31        98        65        13   0.0e+00    1.2e+00
++++   2        49        58        39        23   4.8e+04         8         5        19        39        23         5   0.0e+00    1.3e+00
++++   3        26        32        15         0   4.9e+04        15         0         0        15         0         1   0.0e+00    1.4e+00
++++   4        32         0         0         0   4.9e+04       nan       nan       nan         0         0         0   0.0e+00     nan
Timings [s.]:
<<<<tpart=0.00398993
<<<<tassm=0.000919104
<<<<tfact=0.0055759
<<<<stop=15
<<<<error=0
<<<<tsolv=0.000378132
One-time solve (Random b):
<<<<|Ax-b|/|b| : 0.00748426
<<<<hash(b) : 3038325205160007287
<<<<hash(x) : 6626362926110812358
One-time solve (Random x):
<<<<|Ax-b|/|b| : 0.000493604
<<<<|x-xtrue|/|x| : 0.00525977
<<<<hash(xtrue) : 3038325205160007287
<<<<hash(b) : 710722609911636072
<<<<hash(x) : 10696639138482290093
1: |Ax-b|/|b| = 1.77e-03 <? 1.00e-12
2: |Ax-b|/|b| = 4.95e-05 <? 1.00e-12
3: |Ax-b|/|b| = 1.01e-07 <? 1.00e-12
4: |Ax-b|/|b| = 1.72e-09 <? 1.00e-12
5: |Ax-b|/|b| = 3.00e-12 <? 1.00e-12
6: |Ax-b|/|b| = 2.33e-14 <? 1.00e-12
GMRES converged!
# of iter:  6
Total time: 2.90e-03 s.
  Matvec:   1.24e-04 s.
  Precond:  2.03e-03 s.
GMRES: #iterations: 6, residual |Ax-b|/|b|: 3.62705e-14
  GMRES: 0.00291705 s.
<<<<GMRES=6
```
