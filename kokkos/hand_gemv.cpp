//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

#include <limits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <Kokkos_Core.hpp>

void checkSizes( int &N, int &M, int &S, int &nrepeat );

int main( int argc, char* argv[] )
{
  int N = 4096;         // number of rows 2^12
  int M = 1024;         // number of columns 2^10
  int nrepeat = 1000;  // number of repeats of the test


  Kokkos::initialize( argc, argv );
  {

  // Allocate y, x vectors and Matrix A on device.
  typedef Kokkos::View<double*>   ViewVectorType;
  typedef Kokkos::View<double**>  ViewMatrixType;
  ViewVectorType y( "y", N );
  ViewVectorType x( "x", M );
  ViewMatrixType A( "A", N, M );

  // Create host mirrors of device views.
  ViewVectorType::HostMirror h_y = Kokkos::create_mirror_view( y );
  ViewVectorType::HostMirror h_x = Kokkos::create_mirror_view( x );
  ViewMatrixType::HostMirror h_A = Kokkos::create_mirror_view( A );

  // Initialize y vector on host.
  for ( int i = 0; i < N; ++i ) {
    h_y( i ) = 0;
  }

  // Initialize x vector on host.
  for ( int i = 0; i < M; ++i ) {
    h_x( i ) = 1;
  }

  // Initialize A matrix on host.
  for ( int j = 0; j < N; ++j ) {
    for ( int i = 0; i < M; ++i ) {
      h_A( j, i ) = 1;
    }
  }

  // Deep copy host views to device views.
  Kokkos::deep_copy( y, h_y );
  Kokkos::deep_copy( x, h_x );
  Kokkos::deep_copy( A, h_A );

  typedef Kokkos::TeamPolicy<>               team_policy;
  typedef Kokkos::TeamPolicy<>::member_type  member_type;

  // Timer products.
  Kokkos::Timer timer;

  for ( int repeat = 0; repeat < nrepeat; repeat++ ) {

    Kokkos::parallel_for( "Ax", team_policy( N, Kokkos::AUTO ), KOKKOS_LAMBDA ( const member_type &teamMember ) {
      const int row = teamMember.league_rank();
      double temp2 = 0;

      Kokkos::parallel_reduce( Kokkos::TeamThreadRange( teamMember, M ), [&] ( const int col, double &innerUpdate ) {
        innerUpdate += A( row, col ) * x( col );
      }, temp2 );

      if ( teamMember.team_rank() == 0 ) y( row ) = temp2;
    });
  }

  // Calculate time.
  double time = timer.seconds();

  // Calculate bandwidth.
  // Each matrix A row (each of length M) is read once.
  // The x vector (of length M) is read N times.
  // The y vector (of length N) is read once.
  // double Gbytes = 1.0e-9 * double( sizeof(double) * ( 2 * M * N + N ) );
  double Gbytes = 1.0e-9 * double( sizeof(double) * ( M + M * N + N ) );

  // Print results (problem size, time and bandwidth in GB/s).
  printf( "N( %d ) M( %d ) nrepeat ( %d ) problem( %g MB ) time( %g us ) bandwidth( %g GB/s )\n",
          N, M, nrepeat, Gbytes * 1000, time / nrepeat * 1e6, Gbytes * nrepeat / time );

  }
  Kokkos::finalize();

  return 0;
}
