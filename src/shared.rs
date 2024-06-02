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
