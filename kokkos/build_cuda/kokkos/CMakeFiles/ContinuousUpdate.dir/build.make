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

# Utility rule file for ContinuousUpdate.

# Include any custom commands dependencies for this target.
include kokkos/CMakeFiles/ContinuousUpdate.dir/compiler_depend.make

# Include the progress variables for this target.
include kokkos/CMakeFiles/ContinuousUpdate.dir/progress.make

kokkos/CMakeFiles/ContinuousUpdate:
	cd /home/rob/Documents/PhD/talks/lightning_talk_dec_2023/gpu_programming_model/kokkos/build_cuda/kokkos && /usr/bin/ctest -D ContinuousUpdate

ContinuousUpdate: kokkos/CMakeFiles/ContinuousUpdate
ContinuousUpdate: kokkos/CMakeFiles/ContinuousUpdate.dir/build.make
.PHONY : ContinuousUpdate

# Rule to build all files generated by this target.
kokkos/CMakeFiles/ContinuousUpdate.dir/build: ContinuousUpdate
.PHONY : kokkos/CMakeFiles/ContinuousUpdate.dir/build

kokkos/CMakeFiles/ContinuousUpdate.dir/clean:
	cd /home/rob/Documents/PhD/talks/lightning_talk_dec_2023/gpu_programming_model/kokkos/build_cuda/kokkos && $(CMAKE_COMMAND) -P CMakeFiles/ContinuousUpdate.dir/cmake_clean.cmake
.PHONY : kokkos/CMakeFiles/ContinuousUpdate.dir/clean

kokkos/CMakeFiles/ContinuousUpdate.dir/depend:
	cd /home/rob/Documents/PhD/talks/lightning_talk_dec_2023/gpu_programming_model/kokkos/build_cuda && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/rob/Documents/PhD/talks/lightning_talk_dec_2023/gpu_programming_model/kokkos /home/rob/Documents/PhD/talks/lightning_talk_dec_2023/gpu_programming_model/kokkos/kokkos /home/rob/Documents/PhD/talks/lightning_talk_dec_2023/gpu_programming_model/kokkos/build_cuda /home/rob/Documents/PhD/talks/lightning_talk_dec_2023/gpu_programming_model/kokkos/build_cuda/kokkos /home/rob/Documents/PhD/talks/lightning_talk_dec_2023/gpu_programming_model/kokkos/build_cuda/kokkos/CMakeFiles/ContinuousUpdate.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : kokkos/CMakeFiles/ContinuousUpdate.dir/depend

