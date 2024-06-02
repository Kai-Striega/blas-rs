use num_traits::{Zero, One};
use std::ops::{Add, Mul, MulAssign, AddAssign};

#[non_exhaustive]
pub enum Layout {
    ColumnMajor,
    RowMajor,
}

#[non_exhaustive]
pub enum Op {
    NoOp,
    Transpose,
    Hermitian,
}


/// ``gemm`` computes a scalar-matrix-matrix product and adds the result to a scalar-matrix product.
///
/// This operation is defined as:
///
///    ``C <- α * op(A) * op(B) + β * C``
///
/// Where:
///
/// * ``op`` is one of noop, transpose or hermitian.
/// * ``A``, ``B``, ``C`` are matrices
/// * ``α``, ``β`` are scalars
/// * ``ldx`` is the leading dimension of matrix ``x``.
#[allow(clippy::min_ident_chars)]
#[allow(clippy::too_many_arguments)]
pub fn gemm<'a, T>(layout: &Layout, op_a: &Op, op_b: &Op, alpha: &'a T, a: &'a [T], lda: usize, b: &'a [T], ldb: usize, beta: &T, c: &mut [T], ldc: usize)
    where
        T: Mul<Output=T> + MulAssign + Add<Output=T> + AddAssign + 'a + Zero + One + PartialEq + Copy,
        &'a T: Mul<Output=T> + Add<Output=T>,
{
    match layout {
        Layout::ColumnMajor => {}
        Layout::RowMajor => { unimplemented!() }
    }

    if a.is_empty() || b.is_empty() || c.is_empty() {
        // Early return, we can do no work here.
        return;
    }

    // Early return, alpha == zero => we can avoid the matrix multiply step
    if T::is_zero(alpha) {
        if T::is_zero(beta) {
            for ci in c {
                *ci = T::zero();
            }
        } else if T::is_one(beta) {
            for ci in c {
                *ci *= *beta;
            }
        }
        return;
    }

    match (op_a, op_b) {
        (Op::NoOp, Op::NoOp) => { gemm_noop_noop(alpha, a, lda, b, ldb, beta, c, ldc) }
        (Op::NoOp, Op::Transpose) => { unimplemented!() }
        (Op::NoOp, Op::Hermitian) => { unimplemented!() }
        (Op::Transpose, Op::NoOp) => { unimplemented!() }
        (Op::Transpose, Op::Transpose) => { unimplemented!() }
        (Op::Transpose, Op::Hermitian) => { unimplemented!() }
        (Op::Hermitian, Op::NoOp) => { unimplemented!() }
        (Op::Hermitian, Op::Transpose) => { unimplemented!() }
        (Op::Hermitian, Op::Hermitian) => { unimplemented!() }
    }
}

#[allow(clippy::min_ident_chars)]
#[allow(clippy::too_many_arguments)]
fn gemm_noop_noop<'a, T>(alpha: &'a T, a: &'a [T], lda: usize, b: &'a [T], ldb: usize, beta: &T, c: &mut [T], ldc: usize)
    where
        T: Mul<Output=T> + MulAssign + Add<Output=T> + AddAssign + Copy,
        &'a T: Mul<Output=T> + Add<Output=T>,
{
    for ci in &mut *c {
        *ci *= *beta;
    }

    for (cj, bk) in c.chunks_exact_mut(ldc).zip(b.chunks_exact(ldb)) {
        for (bkj, ai) in bk.iter().zip(a.chunks_exact(lda)) {
            let alpha_bkj = alpha * bkj;
            for (cij, aik) in cj.iter_mut().zip(ai.iter()) {
                *cij += *aik * alpha_bkj;
            }
        }
    }
}


#[cfg(test)]
mod tests {
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
