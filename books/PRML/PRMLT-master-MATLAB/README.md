Introduction
-------
This package is a Matlab implementation of the algorithms described in the classical machine learning textbook:
Pattern Recognition and Machine Learning by C. Bishop ([PRML](http://research.microsoft.com/en-us/um/people/cmbishop/prml/)).

Note: this package requires Matlab **R2016b** or latter, since it utilizes a new syntax of Matlab called [Implicit expansion](https://cn.mathworks.com/help/matlab/release-notes.html?rntext=implicit+expansion&startrelease=R2016b&endrelease=R2016b&groupby=release&sortby=descending) (a.k.a. broadcasting in Python).

Description
-------
While developing this package, I stick to following principles

* Succinct: The code is extremely terse. Minimizing the number of lines is one of the primal goals. As a result, the core of the algorithms can be easily spot.
* Efficient: Many tricks for making Matlab scripts fast were applied (eg. vectorization and matrix factorization). Many functions are even comparable with C implementations. Usually, functions in this package are orders faster than Matlab builtin ones which provide the same functionality (eg. kmeans). If anyone have found any Matlab implementation that is faster than mine, I am happy to further optimize.
* Robust: Many tricks for numerical stability are applied, such as probability computation in log scale and square root matrix update to enforce matrix symmetry, etc.
* Readable: The code is heavily commented. Reference formulas in PRML book are indicated for corresponding code lines. Symbols are in sync with the book.
* Practical: The package is designed not only to be easily read, but also to be easily used to facilitate ML research. Many functions in this package are already widely used (see [Matlab file exchange](http://www.mathworks.com/matlabcentral/fileexchange/?term=authorid%3A49739)).

Installation
-------
1. Download the package to your local path (e.g. PRMLT/) by running: `git clone https://github.com/PRML/PRMLT.git`.

2. Run Matlab and navigate to PRMLT/, then run the init.m script.

3. Try demos in PRMLT/demo directory to verify installation correctness. Enjoy!

FeedBack
-------
If you found any bug or have any suggestion, please do file issues. I am graceful for any feedback and will do my best to improve this package.

License
-------
Currently Released Under GPLv3


Contact
-------
sth4nth at gmail dot com
