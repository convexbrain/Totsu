
/*
## Features

### Using [`linalg::F64LAPACK`]

```toml
[dependencies.totsu]
version = "0.9.1"
features = ["f64lapack"]
```

In addition you need a
[BLAS/LAPACK source](https://github.com/blas-lapack-rs/blas-lapack-rs.github.io/wiki#sources) to link.
*/

mod f64lapack;

pub use f64lapack::*;
