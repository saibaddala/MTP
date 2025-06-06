// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009-2010 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"
#include <QR>

template<typename MatrixType> void householder(const MatrixType& m)
{
  static bool even = true;
  even = !even;
  /* this test covers the following files:
     Householder.h
  */
  Index rows = m.rows();
  Index cols = m.cols();

  typedef typename MatrixType::Scalar Scalar;
  typedef typename NumTraits<Scalar>::Real RealScalar;
  typedef Matrix<Scalar, MatrixType::RowsAtCompileTime, 1> VectorType;
  typedef Matrix<Scalar, internal::decrement_size<MatrixType::RowsAtCompileTime>::ret, 1> EssentialVectorType;
  typedef Matrix<Scalar, MatrixType::RowsAtCompileTime, MatrixType::RowsAtCompileTime> SquareMatrixType;
  typedef Matrix<Scalar, Dynamic, MatrixType::ColsAtCompileTime> HBlockMatrixType;
  typedef Matrix<Scalar, Dynamic, 1> HCoeffsVectorType;

  typedef Matrix<Scalar, MatrixType::ColsAtCompileTime, MatrixType::RowsAtCompileTime> TMatrixType;
  
  Matrix<Scalar, internal::max_size_prefer_dynamic(MatrixType::RowsAtCompileTime,MatrixType::ColsAtCompileTime), 1> _tmp((std::max)(rows,cols));
  Scalar* tmp = &_tmp.coeffRef(0,0);

  Scalar beta;
  RealScalar alpha;
  EssentialVectorType essential;

  VectorType v1 = VectorType::Random(rows), v2;
  v2 = v1;
  v1.makeHouseholder(essential, beta, alpha);
  v1.applyHouseholderOnTheLeft(essential,beta,tmp);
  VERIFY_IS_APPROX(v1.norm(), v2.norm());
  if(rows>=2) VERIFY_IS_MUCH_SMALLER_THAN(v1.tail(rows-1).norm(), v1.norm());
  v1 = VectorType::Random(rows);
  v2 = v1;
  v1.applyHouseholderOnTheLeft(essential,beta,tmp);
  VERIFY_IS_APPROX(v1.norm(), v2.norm());

  // reconstruct householder matrix:
  SquareMatrixType id, H1, H2;
  id.setIdentity(rows, rows);
  H1 = H2 = id;
  VectorType vv(rows);
  vv << Scalar(1), essential;
  H1.applyHouseholderOnTheLeft(essential, beta, tmp);
  H2.applyHouseholderOnTheRight(essential, beta, tmp);
  VERIFY_IS_APPROX(H1, H2);
  VERIFY_IS_APPROX(H1, id - beta * vv*vv.adjoint());

  MatrixType m1(rows, cols),
             m2(rows, cols);

  v1 = VectorType::Random(rows);
  if(even) v1.tail(rows-1).setZero();
  m1.colwise() = v1;
  m2 = m1;
  m1.col(0).makeHouseholder(essential, beta, alpha);
  m1.applyHouseholderOnTheLeft(essential,beta,tmp);
  VERIFY_IS_APPROX(m1.norm(), m2.norm());
  if(rows>=2) VERIFY_IS_MUCH_SMALLER_THAN(m1.block(1,0,rows-1,cols).norm(), m1.norm());
  VERIFY_IS_MUCH_SMALLER_THAN(numext::imag(m1(0,0)), numext::real(m1(0,0)));
  VERIFY_IS_APPROX(numext::real(m1(0,0)), alpha);

  v1 = VectorType::Random(rows);
  if(even) v1.tail(rows-1).setZero();
  SquareMatrixType m3(rows,rows), m4(rows,rows);
  m3.rowwise() = v1.transpose();
  m4 = m3;
  m3.row(0).makeHouseholder(essential, beta, alpha);
  m3.applyHouseholderOnTheRight(essential.conjugate(),beta,tmp);
  VERIFY_IS_APPROX(m3.norm(), m4.norm());
  if(rows>=2) VERIFY_IS_MUCH_SMALLER_THAN(m3.block(0,1,rows,rows-1).norm(), m3.norm());
  VERIFY_IS_MUCH_SMALLER_THAN(numext::imag(m3(0,0)), numext::real(m3(0,0)));
  VERIFY_IS_APPROX(numext::real(m3(0,0)), alpha);

  // test householder sequence on the left with a shift

  Index shift = internal::random<Index>(0, std::max<Index>(rows-2,0));
  Index brows = rows - shift;
  m1.setRandom(rows, cols);
  HBlockMatrixType hbm = m1.block(shift,0,brows,cols);
  HouseholderQR<HBlockMatrixType> qr(hbm);
  m2 = m1;
  m2.block(shift,0,brows,cols) = qr.matrixQR();
  HCoeffsVectorType hc = qr.hCoeffs().conjugate();
  HouseholderSequence<MatrixType, HCoeffsVectorType> hseq(m2, hc);
  hseq.setLength(hc.size()).setShift(shift);
  VERIFY(hseq.length() == hc.size());
  VERIFY(hseq.shift() == shift);
  
  MatrixType m5 = m2;
  m5.block(shift,0,brows,cols).template triangularView<StrictlyLower>().setZero();
  VERIFY_IS_APPROX(hseq * m5, m1); // test applying hseq directly
  m3 = hseq;
  VERIFY_IS_APPROX(m3 * m5, m1); // test evaluating hseq to a dense matrix, then applying
  
  SquareMatrixType hseq_mat = hseq;
  SquareMatrixType hseq_mat_conj = hseq.conjugate();
  SquareMatrixType hseq_mat_adj = hseq.adjoint();
  SquareMatrixType hseq_mat_trans = hseq.transpose();
  SquareMatrixType m6 = SquareMatrixType::Random(rows, rows);
  VERIFY_IS_APPROX(hseq_mat.adjoint(),    hseq_mat_adj);
  VERIFY_IS_APPROX(hseq_mat.conjugate(),  hseq_mat_conj);
  VERIFY_IS_APPROX(hseq_mat.transpose(),  hseq_mat_trans);
  VERIFY_IS_APPROX(hseq * m6,             hseq_mat * m6);
  VERIFY_IS_APPROX(hseq.adjoint() * m6,   hseq_mat_adj * m6);
  VERIFY_IS_APPROX(hseq.conjugate() * m6, hseq_mat_conj * m6);
  VERIFY_IS_APPROX(hseq.transpose() * m6, hseq_mat_trans * m6);
  VERIFY_IS_APPROX(m6 * hseq,             m6 * hseq_mat);
  VERIFY_IS_APPROX(m6 * hseq.adjoint(),   m6 * hseq_mat_adj);
  VERIFY_IS_APPROX(m6 * hseq.conjugate(), m6 * hseq_mat_conj);
  VERIFY_IS_APPROX(m6 * hseq.transpose(), m6 * hseq_mat_trans);

  // test householder sequence on the right with a shift

  TMatrixType tm2 = m2.transpose();
  HouseholderSequence<TMatrixType, HCoeffsVectorType, OnTheRight> rhseq(tm2, hc);
  rhseq.setLength(hc.size()).setShift(shift);
  VERIFY_IS_APPROX(rhseq * m5, m1); // test applying rhseq directly
  m3 = rhseq;
  VERIFY_IS_APPROX(m3 * m5, m1); // test evaluating rhseq to a dense matrix, then applying
}


template <typename MatrixType>
void householder_update(const MatrixType& m) {
  // This test is covering the internal::householder_qr_inplace_update function.
  // At time of writing, there is not public API that exposes this update behavior directly,
  // so we are testing the internal implementation.

  const Index rows = m.rows();
  const Index cols = m.cols();

  typedef typename MatrixType::Scalar Scalar;
  typedef Matrix<Scalar, MatrixType::RowsAtCompileTime, 1> VectorType;
  typedef Matrix<Scalar, Dynamic, 1> HCoeffsVectorType;
  typedef Matrix<Scalar, Dynamic, Dynamic> MatrixX;
  typedef Matrix<Scalar, Dynamic, 1> VectorX;

  VectorX tmpOwner(cols);
  Scalar* tmp = tmpOwner.data();

  // The matrix to factorize.
  const MatrixType A = MatrixType::Random(rows, cols); 

  // matQR and hCoeffs will hold the factorization of A,
  // built by a sequence of calls to `update`.
  MatrixType matQR(rows, cols);
  HCoeffsVectorType hCoeffs(cols);

  // householder_qr_inplace_update should be able to build a QR factorization one column at a time.
  // We verify this by starting with an empty factorization and 'updating' one column at a time.
  // After each call to update, we should have a QR factorization of the columns presented so far.

  const Index size = (std::min)(rows, cols); // QR can only go up to 'size' b/c that's full rank.
  for (Index k = 0; k != size; ++k)
  {
    // Make a copy of the column to prevent any possibility of 'leaking' other parts of A.
    const VectorType newColumn = A.col(k); 
    internal::householder_qr_inplace_update(matQR, hCoeffs, newColumn, k, tmp);

    // Verify Property:
    // matQR.leftCols(k+1) and hCoeffs.head(k+1) hold
    // a QR factorization of A.leftCols(k+1).
    // This is the fundamental guarantee of householder_qr_inplace_update.
    {
      const MatrixX matQR_k = matQR.leftCols(k + 1);
      const VectorX hCoeffs_k = hCoeffs.head(k + 1);
      MatrixX R = matQR_k.template triangularView<Upper>();
      MatrixX QxR = householderSequence(matQR_k, hCoeffs_k.conjugate()) * R;
      VERIFY_IS_APPROX(QxR, A.leftCols(k + 1));
    }

    // Verify Property:
    // A sequence of calls to 'householder_qr_inplace_update'
    // should produce the same result as 'householder_qr_inplace_unblocked'.
    // This is a property of the current implementation.
    // If these implementations diverge in the future, 
    // then simply delete the test of this property.
    {
      MatrixX QR_at_once = A.leftCols(k + 1);
      VectorX hCoeffs_at_once(k + 1);
      internal::householder_qr_inplace_unblocked(QR_at_once, hCoeffs_at_once, tmp);
      VERIFY_IS_APPROX(QR_at_once, matQR.leftCols(k + 1));
      VERIFY_IS_APPROX(hCoeffs_at_once, hCoeffs.head(k + 1));
    }
  }

  // Verify Property:
  // We can go back and update any column to have a new value,
  // and get a QR factorization of the columns up to that one.  
  {
    const Index k = internal::random<Index>(0, size - 1);
    VectorType newColumn = VectorType::Random(rows);      
    internal::householder_qr_inplace_update(matQR, hCoeffs, newColumn, k, tmp);

    const MatrixX matQR_k = matQR.leftCols(k + 1);
    const VectorX hCoeffs_k = hCoeffs.head(k + 1);
    MatrixX R = matQR_k.template triangularView<Upper>();
    MatrixX QxR = householderSequence(matQR_k, hCoeffs_k.conjugate()) * R;
    VERIFY_IS_APPROX(QxR.leftCols(k), A.leftCols(k));
    VERIFY_IS_APPROX(QxR.col(k), newColumn);
  }  
}


EIGEN_DECLARE_TEST(householder)
{
  for(int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_1( householder(Matrix<double,2,2>()) );
    CALL_SUBTEST_2( householder(Matrix<float,2,3>()) );
    CALL_SUBTEST_3( householder(Matrix<double,3,5>()) );
    CALL_SUBTEST_4( householder(Matrix<float,4,4>()) );
    CALL_SUBTEST_5( householder(MatrixXd(internal::random<int>(1,EIGEN_TEST_MAX_SIZE),internal::random<int>(1,EIGEN_TEST_MAX_SIZE))) );
    CALL_SUBTEST_6( householder(MatrixXcf(internal::random<int>(1,EIGEN_TEST_MAX_SIZE),internal::random<int>(1,EIGEN_TEST_MAX_SIZE))) );
    CALL_SUBTEST_7( householder(MatrixXf(internal::random<int>(1,EIGEN_TEST_MAX_SIZE),internal::random<int>(1,EIGEN_TEST_MAX_SIZE))) );
    CALL_SUBTEST_8( householder(Matrix<double,1,1>()) );

    CALL_SUBTEST_9( householder_update(Matrix<double, 3, 5>()) );
    CALL_SUBTEST_9( householder_update(Matrix<float, 4, 2>()) );
    CALL_SUBTEST_9( householder_update(MatrixXcf(internal::random<Index>(1,EIGEN_TEST_MAX_SIZE), internal::random<Index>(1,EIGEN_TEST_MAX_SIZE))) );
  }
}
