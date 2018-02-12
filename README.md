# Source code of SIGIR'17 paper "Learning to rank using localized geometry mean metrics"
 ([ACM version](https://dl.acm.org/citation.cfm?id=3080828), [Arxiv version](https://arxiv.org/abs/1705.07563))


The program is written by [Julia](http://julialang.org/). Please download the [lastest version](http://julialang.org/downloads/) to execute this demo code.

## Prerequisites

You need to install two packages `MAT` and `Iterators` in the command shell of Julia:
```julia
julia > Pkg.add("MAT")
julia > Pkg.add("Iterators")
```

## Execution

We have prepared a demo program to train two rank lists from MSLR-WEB10K dataset. You can try our program under the project directory and type:

```julia
julia demo_small_dataset.jl
```

## Citation
Please cite this paper if you use the source code with academic purpose:
```
@inproceedings{DBLP:conf/sigir/SuKL17,
  author    = {Yuxin Su and
               Irwin King and
               Michael R. Lyu},
  title     = {Learning to Rank Using Localized Geometric Mean Metrics},
  booktitle = {Proceedings of the 40th International {ACM} {SIGIR} Conference on
               Research and Development in Information Retrieval, Shinjuku, Tokyo,
               Japan, August 7-11, 2017},
  pages     = {45--54},
  year      = {2017},
  crossref  = {DBLP:conf/sigir/2017},
  url       = {http://doi.acm.org/10.1145/3077136.3080828},
  doi       = {10.1145/3077136.3080828},
  timestamp = {Sun, 06 Aug 2017 18:21:32 +0200},
  biburl    = {http://dblp.org/rec/bib/conf/sigir/SuKL17},
  bibsource = {dblp computer science bibliography, http://dblp.org}
}
```
