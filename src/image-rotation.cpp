#include <CL/sycl.hpp>
#include <array>
#include <iostream>
#include "dpc_common.hpp"

#if FPGA || FPGA_EMULATOR || FPGA_PROFILE
#include <sycl/ext/intel/fpga_extensions.hpp>
#endif

using namespace sycl;

#include "utils.h"
#include "bmp-utils.h"

static const char* imgInputPath = "./Images/cat.bmp"

int main (){

    return 0;
}

