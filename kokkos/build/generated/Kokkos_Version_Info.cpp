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

#include "Kokkos_Version_Info.hpp"

namespace Kokkos {
namespace Impl {

std::string GIT_BRANCH       = R"branch(master)branch";
std::string GIT_COMMIT_HASH  = "71a9bcae5";
std::string GIT_CLEAN_STATUS = "CLEAN";
std::string GIT_COMMIT_DESCRIPTION =
    R"message(Merge pull request #6613 from ndellingwood/master-release-4.2.00)message";
std::string GIT_COMMIT_DATE = "2023-11-20T14:26:46-07:00";

}  // namespace Impl

}  // namespace Kokkos
