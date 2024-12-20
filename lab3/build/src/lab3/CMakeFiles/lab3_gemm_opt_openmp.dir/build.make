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
CMAKE_SOURCE_DIR = /home/msquare01/2024_fall/comparch/lab3

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/msquare01/2024_fall/comparch/lab3/build

# Include any dependencies generated for this target.
include src/lab3/CMakeFiles/lab3_gemm_opt_openmp.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include src/lab3/CMakeFiles/lab3_gemm_opt_openmp.dir/compiler_depend.make

# Include the progress variables for this target.
include src/lab3/CMakeFiles/lab3_gemm_opt_openmp.dir/progress.make

# Include the compile flags for this target's objects.
include src/lab3/CMakeFiles/lab3_gemm_opt_openmp.dir/flags.make

src/lab3/CMakeFiles/lab3_gemm_opt_openmp.dir/gemm_kernel_opt_openmp.cpp.o: src/lab3/CMakeFiles/lab3_gemm_opt_openmp.dir/flags.make
src/lab3/CMakeFiles/lab3_gemm_opt_openmp.dir/gemm_kernel_opt_openmp.cpp.o: ../src/lab3/gemm_kernel_opt_openmp.cpp
src/lab3/CMakeFiles/lab3_gemm_opt_openmp.dir/gemm_kernel_opt_openmp.cpp.o: src/lab3/CMakeFiles/lab3_gemm_opt_openmp.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/msquare01/2024_fall/comparch/lab3/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/lab3/CMakeFiles/lab3_gemm_opt_openmp.dir/gemm_kernel_opt_openmp.cpp.o"
	cd /home/msquare01/2024_fall/comparch/lab3/build/src/lab3 && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/lab3/CMakeFiles/lab3_gemm_opt_openmp.dir/gemm_kernel_opt_openmp.cpp.o -MF CMakeFiles/lab3_gemm_opt_openmp.dir/gemm_kernel_opt_openmp.cpp.o.d -o CMakeFiles/lab3_gemm_opt_openmp.dir/gemm_kernel_opt_openmp.cpp.o -c /home/msquare01/2024_fall/comparch/lab3/src/lab3/gemm_kernel_opt_openmp.cpp

src/lab3/CMakeFiles/lab3_gemm_opt_openmp.dir/gemm_kernel_opt_openmp.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/lab3_gemm_opt_openmp.dir/gemm_kernel_opt_openmp.cpp.i"
	cd /home/msquare01/2024_fall/comparch/lab3/build/src/lab3 && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/msquare01/2024_fall/comparch/lab3/src/lab3/gemm_kernel_opt_openmp.cpp > CMakeFiles/lab3_gemm_opt_openmp.dir/gemm_kernel_opt_openmp.cpp.i

src/lab3/CMakeFiles/lab3_gemm_opt_openmp.dir/gemm_kernel_opt_openmp.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/lab3_gemm_opt_openmp.dir/gemm_kernel_opt_openmp.cpp.s"
	cd /home/msquare01/2024_fall/comparch/lab3/build/src/lab3 && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/msquare01/2024_fall/comparch/lab3/src/lab3/gemm_kernel_opt_openmp.cpp -o CMakeFiles/lab3_gemm_opt_openmp.dir/gemm_kernel_opt_openmp.cpp.s

src/lab3/CMakeFiles/lab3_gemm_opt_openmp.dir/openmp_gemm_baseline.cpp.o: src/lab3/CMakeFiles/lab3_gemm_opt_openmp.dir/flags.make
src/lab3/CMakeFiles/lab3_gemm_opt_openmp.dir/openmp_gemm_baseline.cpp.o: ../src/lab3/openmp_gemm_baseline.cpp
src/lab3/CMakeFiles/lab3_gemm_opt_openmp.dir/openmp_gemm_baseline.cpp.o: src/lab3/CMakeFiles/lab3_gemm_opt_openmp.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/msquare01/2024_fall/comparch/lab3/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object src/lab3/CMakeFiles/lab3_gemm_opt_openmp.dir/openmp_gemm_baseline.cpp.o"
	cd /home/msquare01/2024_fall/comparch/lab3/build/src/lab3 && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/lab3/CMakeFiles/lab3_gemm_opt_openmp.dir/openmp_gemm_baseline.cpp.o -MF CMakeFiles/lab3_gemm_opt_openmp.dir/openmp_gemm_baseline.cpp.o.d -o CMakeFiles/lab3_gemm_opt_openmp.dir/openmp_gemm_baseline.cpp.o -c /home/msquare01/2024_fall/comparch/lab3/src/lab3/openmp_gemm_baseline.cpp

src/lab3/CMakeFiles/lab3_gemm_opt_openmp.dir/openmp_gemm_baseline.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/lab3_gemm_opt_openmp.dir/openmp_gemm_baseline.cpp.i"
	cd /home/msquare01/2024_fall/comparch/lab3/build/src/lab3 && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/msquare01/2024_fall/comparch/lab3/src/lab3/openmp_gemm_baseline.cpp > CMakeFiles/lab3_gemm_opt_openmp.dir/openmp_gemm_baseline.cpp.i

src/lab3/CMakeFiles/lab3_gemm_opt_openmp.dir/openmp_gemm_baseline.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/lab3_gemm_opt_openmp.dir/openmp_gemm_baseline.cpp.s"
	cd /home/msquare01/2024_fall/comparch/lab3/build/src/lab3 && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/msquare01/2024_fall/comparch/lab3/src/lab3/openmp_gemm_baseline.cpp -o CMakeFiles/lab3_gemm_opt_openmp.dir/openmp_gemm_baseline.cpp.s

src/lab3/CMakeFiles/lab3_gemm_opt_openmp.dir/openmp_gemm_opt.cpp.o: src/lab3/CMakeFiles/lab3_gemm_opt_openmp.dir/flags.make
src/lab3/CMakeFiles/lab3_gemm_opt_openmp.dir/openmp_gemm_opt.cpp.o: ../src/lab3/openmp_gemm_opt.cpp
src/lab3/CMakeFiles/lab3_gemm_opt_openmp.dir/openmp_gemm_opt.cpp.o: src/lab3/CMakeFiles/lab3_gemm_opt_openmp.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/msquare01/2024_fall/comparch/lab3/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object src/lab3/CMakeFiles/lab3_gemm_opt_openmp.dir/openmp_gemm_opt.cpp.o"
	cd /home/msquare01/2024_fall/comparch/lab3/build/src/lab3 && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/lab3/CMakeFiles/lab3_gemm_opt_openmp.dir/openmp_gemm_opt.cpp.o -MF CMakeFiles/lab3_gemm_opt_openmp.dir/openmp_gemm_opt.cpp.o.d -o CMakeFiles/lab3_gemm_opt_openmp.dir/openmp_gemm_opt.cpp.o -c /home/msquare01/2024_fall/comparch/lab3/src/lab3/openmp_gemm_opt.cpp

src/lab3/CMakeFiles/lab3_gemm_opt_openmp.dir/openmp_gemm_opt.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/lab3_gemm_opt_openmp.dir/openmp_gemm_opt.cpp.i"
	cd /home/msquare01/2024_fall/comparch/lab3/build/src/lab3 && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/msquare01/2024_fall/comparch/lab3/src/lab3/openmp_gemm_opt.cpp > CMakeFiles/lab3_gemm_opt_openmp.dir/openmp_gemm_opt.cpp.i

src/lab3/CMakeFiles/lab3_gemm_opt_openmp.dir/openmp_gemm_opt.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/lab3_gemm_opt_openmp.dir/openmp_gemm_opt.cpp.s"
	cd /home/msquare01/2024_fall/comparch/lab3/build/src/lab3 && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/msquare01/2024_fall/comparch/lab3/src/lab3/openmp_gemm_opt.cpp -o CMakeFiles/lab3_gemm_opt_openmp.dir/openmp_gemm_opt.cpp.s

src/lab3/CMakeFiles/lab3_gemm_opt_openmp.dir/gemm_kernel_opt_avx.S.o: src/lab3/CMakeFiles/lab3_gemm_opt_openmp.dir/flags.make
src/lab3/CMakeFiles/lab3_gemm_opt_openmp.dir/gemm_kernel_opt_avx.S.o: ../src/lab3/gemm_kernel_opt_avx.S
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/msquare01/2024_fall/comparch/lab3/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building ASM object src/lab3/CMakeFiles/lab3_gemm_opt_openmp.dir/gemm_kernel_opt_avx.S.o"
	cd /home/msquare01/2024_fall/comparch/lab3/build/src/lab3 && /usr/bin/gcc $(ASM_DEFINES) $(ASM_INCLUDES) $(ASM_FLAGS) -o CMakeFiles/lab3_gemm_opt_openmp.dir/gemm_kernel_opt_avx.S.o -c /home/msquare01/2024_fall/comparch/lab3/src/lab3/gemm_kernel_opt_avx.S

src/lab3/CMakeFiles/lab3_gemm_opt_openmp.dir/gemm_kernel_opt_avx.S.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing ASM source to CMakeFiles/lab3_gemm_opt_openmp.dir/gemm_kernel_opt_avx.S.i"
	cd /home/msquare01/2024_fall/comparch/lab3/build/src/lab3 && /usr/bin/gcc $(ASM_DEFINES) $(ASM_INCLUDES) $(ASM_FLAGS) -E /home/msquare01/2024_fall/comparch/lab3/src/lab3/gemm_kernel_opt_avx.S > CMakeFiles/lab3_gemm_opt_openmp.dir/gemm_kernel_opt_avx.S.i

src/lab3/CMakeFiles/lab3_gemm_opt_openmp.dir/gemm_kernel_opt_avx.S.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling ASM source to assembly CMakeFiles/lab3_gemm_opt_openmp.dir/gemm_kernel_opt_avx.S.s"
	cd /home/msquare01/2024_fall/comparch/lab3/build/src/lab3 && /usr/bin/gcc $(ASM_DEFINES) $(ASM_INCLUDES) $(ASM_FLAGS) -S /home/msquare01/2024_fall/comparch/lab3/src/lab3/gemm_kernel_opt_avx.S -o CMakeFiles/lab3_gemm_opt_openmp.dir/gemm_kernel_opt_avx.S.s

src/lab3/CMakeFiles/lab3_gemm_opt_openmp.dir/gemm_kernel_baseline.S.o: src/lab3/CMakeFiles/lab3_gemm_opt_openmp.dir/flags.make
src/lab3/CMakeFiles/lab3_gemm_opt_openmp.dir/gemm_kernel_baseline.S.o: ../src/lab3/gemm_kernel_baseline.S
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/msquare01/2024_fall/comparch/lab3/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building ASM object src/lab3/CMakeFiles/lab3_gemm_opt_openmp.dir/gemm_kernel_baseline.S.o"
	cd /home/msquare01/2024_fall/comparch/lab3/build/src/lab3 && /usr/bin/gcc $(ASM_DEFINES) $(ASM_INCLUDES) $(ASM_FLAGS) -o CMakeFiles/lab3_gemm_opt_openmp.dir/gemm_kernel_baseline.S.o -c /home/msquare01/2024_fall/comparch/lab3/src/lab3/gemm_kernel_baseline.S

src/lab3/CMakeFiles/lab3_gemm_opt_openmp.dir/gemm_kernel_baseline.S.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing ASM source to CMakeFiles/lab3_gemm_opt_openmp.dir/gemm_kernel_baseline.S.i"
	cd /home/msquare01/2024_fall/comparch/lab3/build/src/lab3 && /usr/bin/gcc $(ASM_DEFINES) $(ASM_INCLUDES) $(ASM_FLAGS) -E /home/msquare01/2024_fall/comparch/lab3/src/lab3/gemm_kernel_baseline.S > CMakeFiles/lab3_gemm_opt_openmp.dir/gemm_kernel_baseline.S.i

src/lab3/CMakeFiles/lab3_gemm_opt_openmp.dir/gemm_kernel_baseline.S.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling ASM source to assembly CMakeFiles/lab3_gemm_opt_openmp.dir/gemm_kernel_baseline.S.s"
	cd /home/msquare01/2024_fall/comparch/lab3/build/src/lab3 && /usr/bin/gcc $(ASM_DEFINES) $(ASM_INCLUDES) $(ASM_FLAGS) -S /home/msquare01/2024_fall/comparch/lab3/src/lab3/gemm_kernel_baseline.S -o CMakeFiles/lab3_gemm_opt_openmp.dir/gemm_kernel_baseline.S.s

# Object files for target lab3_gemm_opt_openmp
lab3_gemm_opt_openmp_OBJECTS = \
"CMakeFiles/lab3_gemm_opt_openmp.dir/gemm_kernel_opt_openmp.cpp.o" \
"CMakeFiles/lab3_gemm_opt_openmp.dir/openmp_gemm_baseline.cpp.o" \
"CMakeFiles/lab3_gemm_opt_openmp.dir/openmp_gemm_opt.cpp.o" \
"CMakeFiles/lab3_gemm_opt_openmp.dir/gemm_kernel_opt_avx.S.o" \
"CMakeFiles/lab3_gemm_opt_openmp.dir/gemm_kernel_baseline.S.o"

# External object files for target lab3_gemm_opt_openmp
lab3_gemm_opt_openmp_EXTERNAL_OBJECTS =

dist/bins/lab3_gemm_opt_openmp: src/lab3/CMakeFiles/lab3_gemm_opt_openmp.dir/gemm_kernel_opt_openmp.cpp.o
dist/bins/lab3_gemm_opt_openmp: src/lab3/CMakeFiles/lab3_gemm_opt_openmp.dir/openmp_gemm_baseline.cpp.o
dist/bins/lab3_gemm_opt_openmp: src/lab3/CMakeFiles/lab3_gemm_opt_openmp.dir/openmp_gemm_opt.cpp.o
dist/bins/lab3_gemm_opt_openmp: src/lab3/CMakeFiles/lab3_gemm_opt_openmp.dir/gemm_kernel_opt_avx.S.o
dist/bins/lab3_gemm_opt_openmp: src/lab3/CMakeFiles/lab3_gemm_opt_openmp.dir/gemm_kernel_baseline.S.o
dist/bins/lab3_gemm_opt_openmp: src/lab3/CMakeFiles/lab3_gemm_opt_openmp.dir/build.make
dist/bins/lab3_gemm_opt_openmp: /usr/lib/gcc/x86_64-linux-gnu/11/libgomp.so
dist/bins/lab3_gemm_opt_openmp: /usr/lib/x86_64-linux-gnu/libpthread.a
dist/bins/lab3_gemm_opt_openmp: src/lab3/CMakeFiles/lab3_gemm_opt_openmp.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/msquare01/2024_fall/comparch/lab3/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Linking CXX executable ../../dist/bins/lab3_gemm_opt_openmp"
	cd /home/msquare01/2024_fall/comparch/lab3/build/src/lab3 && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/lab3_gemm_opt_openmp.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/lab3/CMakeFiles/lab3_gemm_opt_openmp.dir/build: dist/bins/lab3_gemm_opt_openmp
.PHONY : src/lab3/CMakeFiles/lab3_gemm_opt_openmp.dir/build

src/lab3/CMakeFiles/lab3_gemm_opt_openmp.dir/clean:
	cd /home/msquare01/2024_fall/comparch/lab3/build/src/lab3 && $(CMAKE_COMMAND) -P CMakeFiles/lab3_gemm_opt_openmp.dir/cmake_clean.cmake
.PHONY : src/lab3/CMakeFiles/lab3_gemm_opt_openmp.dir/clean

src/lab3/CMakeFiles/lab3_gemm_opt_openmp.dir/depend:
	cd /home/msquare01/2024_fall/comparch/lab3/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/msquare01/2024_fall/comparch/lab3 /home/msquare01/2024_fall/comparch/lab3/src/lab3 /home/msquare01/2024_fall/comparch/lab3/build /home/msquare01/2024_fall/comparch/lab3/build/src/lab3 /home/msquare01/2024_fall/comparch/lab3/build/src/lab3/CMakeFiles/lab3_gemm_opt_openmp.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/lab3/CMakeFiles/lab3_gemm_opt_openmp.dir/depend

