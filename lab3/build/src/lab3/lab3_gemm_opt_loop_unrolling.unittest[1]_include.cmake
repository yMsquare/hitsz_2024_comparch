if(EXISTS "/home/msquare01/2024_fall/comparch/lab3/build/src/lab3/lab3_gemm_opt_loop_unrolling.unittest[1]_tests.cmake")
  include("/home/msquare01/2024_fall/comparch/lab3/build/src/lab3/lab3_gemm_opt_loop_unrolling.unittest[1]_tests.cmake")
else()
  add_test(lab3_gemm_opt_loop_unrolling.unittest_NOT_BUILT lab3_gemm_opt_loop_unrolling.unittest_NOT_BUILT)
endif()