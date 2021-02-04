# MemoryViewTestHelper

## Description

MemoryViewTestHelper provides features that help to test libraries that include MemoryView support.

`MemoryViewTestHelper::NDArray` provides simple multi-dimensional numeric array that can export MemoryView.

## Install

```console
$ gem install memory-view-test-helper
```

## Usage

First you need to require `memory-view-test-helper` library.

```ruby
require "memory-view-test-helper"
```

You can create a multi-dimensional numeric array by `MemoryViewTestHelper.new`.

```
x = MemoryViewTestHelper::NDArray.new([[1, 2, 3], [4, 5, 6]], dtype: :float64)
```

By this expression, `x` refers a 2x3 matrix of 64-bit floating point numbers.

## License

The MIT license. See [`LICENSE.txt`](LICENSE.txt) for details.
