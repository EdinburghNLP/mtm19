- https://github.com/CrossLangNV/MT-ComparEval
- http://mtcompareval-ng.crosslang.com:8051/matrix

# Performance tweaking / debugging

## Matrix view

Using XDebug profiler:

- before ![before](https://www.dropbox.com/s/xd7fptnizs2n8zd/before.png?dl=1)

- after ![after](https://www.dropbox.com/s/76038ab33z7gsrd/after.png?dl=1)

## Import test sets

Ongoing: The import takes at least couple of minutes for each test set. Check the current implementation for generating the n-gram statistics and for metric calculation.

## Switching database

Ongoing: Migrate existing SQLite to postgreSQL.

## Checking other views

Ongoing: Loading of more sentences in sentences pane sometimes takes too long.

# Additional metrics

Provide wrappers for existing metrics like sacreBLEU, multeval,...
