- https://github.com/CrossLangNV/MT-ComparEval
- http://mtcompareval-ng.crosslang.com:8051/matrix

# Performance tweaking / debugging

## Matrix view

Using XDebug profiler:

- before ![before](https://i.imgur.com/oL9mKqY.png)

- after ![after](https://i.imgur.com/620XYPD.png)

## Import test sets

Ongoing: The import takes at least couple of minutes for each test set. Check the current implementation for generating the n-gram statistics and for metric calculation.

## Switching database

Ongoing: Migrate existing SQLite to postgreSQL.

## Checking other views

Ongoing: Loading of more sentences in sentences pane sometimes takes too long.

# Additional metrics

Provide wrappers for existing metrics like sacreBLEU, multeval,...
