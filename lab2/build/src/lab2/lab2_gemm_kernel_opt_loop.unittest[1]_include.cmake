if(EXISTS "/home/msquare01/2024_fall/comparch/lab2/build/src/lab2/lab2_gemm_kernel_opt_loop.unittest[1]_tests.cmake")
  include("/home/msquare01/2024_fall/comparch/lab2/build/src/lab2/lab2_gemm_kernel_opt_loop.unittest[1]_tests.cmake")
else()
  add_test(lab2_gemm_kernel_opt_loop.unittest_NOT_BUILT lab2_gemm_kernel_opt_loop.unittest_NOT_BUILT)
endif()