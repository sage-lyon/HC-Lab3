#include <CL/sycl.hpp>
#include <array>
#include <exception>
#include <iostream>
#include <cmath>
#include "dpc_common.hpp"

#if FPGA || FPGA_EMULATOR || FPGA_PROFILE
#include <sycl/ext/intel/fpga_extensions.hpp>
#endif

using namespace sycl;

#include "utils.h"
#include "bmp-utils.h"

int main (){
    // Create device selector for the device of your interest.
#if FPGA_EMULATOR
    // DPC++ extension: FPGA emulator selector on systems without FPGA card.
    INTEL::fpga_emulator_selector d_selector;
#elif FPGA || FPGA_PROFILE
    // DPC++ extension: FPGA selector on systems with FPGA card.
    INTEL::fpga_selector d_selector;
#else
    // The default device selector will select the most performant device.
    default_selector d_selector;
#endif

    float *input_img, *output_img;
    int img_rows, img_cols;

    // Read input image and allocate memory for output image
    input_img = readBmpFloat("./Images/cat.bmp", &img_rows, &img_cols);
    output_img = (float *) malloc(sizeof(float) * img_rows * img_cols);

    try {
        // Create queue
        queue q(d_selector, dpc_common::exception_handler);

        // Create buffers for input and output images
        buffer<float, 1> input_buffer(input_img, range<1>(img_rows*img_cols));
        buffer<float, 1> output_buffer(output_img, range<1>(img_rows*img_cols));

        // Create range for number of items
        range<2> num_items{img_rows, img_cols};

        q.submit([&](handler &h){

            // Create accessors for buffers
            accessor input(input_buffer, h, read_only);
            accessor output(output_buffer, h, write_only);

            h.parallel_for(range<2>(img_rows, img_cols), [=](id<2> item){
                const int row = item[0];
                const int col = item[1];

                float new_row = ((float)row)*cos(theta) + ((float)col)*sin(theta);
                float new_col = -1.0f*((float)row)*sin(theta) + ((float)col)*cos(theta);

                // If new row and col are within image bounds set image data from old position to new position
                if(((int)new_row >= 0) && ((int)new_row < img_cols) && ((int)new_col >= 0) && ((int)new_col < img_rows))
                    output[(int)new_col * img_row + (int)new_row] = input[col * img_row + row];

            });

        });


    }
    catch (exception const &e) {
        std::cout << "Exception caught for image rotation" << std::endl;
        std::terminate();
    }

    return 0;
}

