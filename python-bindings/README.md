Moose Python bindings
===============
### Installation
```
pip install -r requirements-dev.txt
pip install -e .
pytest .
```

### Usage

```python
from moose import ring_add
import numpy as np

a = np.array([1, 2, 3], dtype=np.uint64)
b = np.array([4, 5, 6], dtype=np.uint64)

# same as a + b with u64 precision, but with wrapping semantics at precision boundary
c = ring_add(a, b)
```
