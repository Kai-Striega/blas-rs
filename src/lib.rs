use num_traits::{Zero, One};
use std::ops::{Add, Mul, MulAssign, AddAssign};


#[cfg(test)]
mod tests {
    use super::*;

    /// Naive implementation of ``gemm`` to use for testing.
    fn gemm_naive<'a, T>(alpha: &'a T, a: &'a [T], lda: usize, b: &[T], ldb: usize, beta: &T, c: &mut [T], ldc: usize)
        where
            T: Mul<Output=T> + MulAssign + Add<Output=T> + AddAssign + Copy + One + PartialEq,
            &'a T: Mul<Output=T> + Add<Output=T>,
    {
        if *beta != T::one() {
            for ci in &mut  *c {
                *ci *= *beta;
            }
        }

        for (ci, ai) in c.chunks_exact_mut(ldc).zip(a.chunks_exact(lda)) {
            for (aik, bk) in ai.iter().zip(b.chunks_exact(ldb)) {
                for (cij, bkj) in ci.iter_mut().zip(bk.iter()) {
                    *cij += alpha * aik * (*bkj);
                }
            }
        }
    }

    #[test]
    fn test_simple_example() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let lda = 3;
        let b = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let ldb = 2;
        let mut c = vec![1.0, 1.0, 1.0, 1.0];
        let ldc = 2;
        let c_expected = vec![59.0, 65.0, 140.0, 155.0];
        gemm_naive(&1.0, &a, lda, &b, ldb, &1.0, &mut c, ldc);
        assert_eq!(c, c_expected);
    }
}
