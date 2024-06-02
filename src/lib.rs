pub mod shared;
pub mod gemm;

use num_traits::{One, Zero};
use std::ops::{Add, AddAssign, Mul, MulAssign};
use shared::{Layout, Op};


#[cfg(test)]
mod tests {
    use crate::gemm::gemm;
    use super::*;

    #[test]
    fn test_noop_noop_col_major() {
        // All matrices in column major storage
        let a = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
        let lda = 2;
        let b = vec![7.0, 9.0, 11.0, 8.0, 10.0, 12.0];
        let ldb = 3;
        let mut c = vec![1.0, 1.0, 1.0, 1.0];
        let ldc = 2;
        let c_expected = vec![59.0, 140.0, 65.0, 155.0];
        gemm(&Layout::ColumnMajor, &Op::NoOp, &Op::NoOp, &1.0, &a, lda, &b, ldb, &1.0, &mut c, ldc);
        assert_eq!(c, c_expected);
    }
}
