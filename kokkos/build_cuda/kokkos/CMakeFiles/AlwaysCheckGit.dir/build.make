# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/rob/Documents/PhD/talks/lightning_talk_dec_2023/gpu_programming_model/kokkos

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/rob/Documents/PhD/talks/lightning_talk_dec_2023/gpu_programming_model/kokkos/build_cuda

# Utility rule file for AlwaysCheckGit.

# Include any custom commands dependencies for this target.
include kokkos/CMakeFiles/AlwaysCheckGit.dir/compiler_depend.make

# Include the progress variables for this target.
include kokkos/CMakeFiles/AlwaysCheckGit.dir/progress.make

kokkos/CMakeFiles/AlwaysCheckGit:
	cd /home/rob/Documents/PhD/talks/lightning_talk_dec_2023/gpu_programming_model/kokkos/build_cuda/kokkos && /usr/bin/cmake -DRUN_CHECK_GIT_VERSION=1 -DKOKKOS_SOURCE_DIR=/home/rob/Documents/PhD/talks/lightning_talk_dec_2023/gpu_programming_model/kokkos/kokkos -P /home/rob/Documents/PhD/talks/lightning_talk_dec_2023/gpu_programming_model/kokkos/kokkos/cmake/build_env_info.cmake

AlwaysCheckGit: kokkos/CMakeFiles/AlwaysCheckGit
AlwaysCheckGit: kokkos/CMakeFiles/AlwaysCheckGit.dir/build.make
.PHONY : AlwaysCheckGit

# Rule to build all files generated by this target.
kokkos/CMakeFiles/AlwaysCheckGit.dir/build: AlwaysCheckGit
.PHONY : kokkos/CMakeFiles/AlwaysCheckGit.dir/build

kokkos/CMakeFiles/AlwaysCheckGit.dir/clean:
	cd /home/rob/Documents/PhD/talks/lightning_talk_dec_2023/gpu_programming_model/kokkos/build_cuda/kokkos && $(CMAKE_COMMAND) -P CMakeFiles/AlwaysCheckGit.dir/cmake_clean.cmake
.PHONY : kokkos/CMakeFiles/AlwaysCheckGit.dir/clean

kokkos/CMakeFiles/AlwaysCheckGit.dir/depend:
	cd /home/rob/Documents/PhD/talks/lightning_talk_dec_2023/gpu_programming_model/kokkos/build_cuda && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/rob/Documents/PhD/talks/lightning_talk_dec_2023/gpu_programming_model/kokkos /home/rob/Documents/PhD/talks/lightning_talk_dec_2023/gpu_programming_model/kokkos/kokkos /home/rob/Documents/PhD/talks/lightning_talk_dec_2023/gpu_programming_model/kokkos/build_cuda /home/rob/Documents/PhD/talks/lightning_talk_dec_2023/gpu_programming_model/kokkos/build_cuda/kokkos /home/rob/Documents/PhD/talks/lightning_talk_dec_2023/gpu_programming_model/kokkos/build_cuda/kokkos/CMakeFiles/AlwaysCheckGit.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : kokkos/CMakeFiles/AlwaysCheckGit.dir/depend

