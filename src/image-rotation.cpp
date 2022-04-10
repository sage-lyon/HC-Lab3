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

#include "bmp-utils.h"

int main (){
    // Create device selector for the device of your interest.
#if FPGA_EMULATOR
    // DPC++ extension: FPGA emulator selector on systems without FPGA card.
    ext::intel::fpga_emulator_selector d_selector;
#elif FPGA || FPGA_PROFILE
    // DPC++ extension: FPGA selector on systems with FPGA card.
    ext::intel::fpga_selector d_selector;
#else
    // The default device selector will select the most performant device.
    default_selector d_selector;
#endif

    float *input_img, *output_img, angle;
    int img_rows, img_cols;

    // Ask user to input angle of rotation for image
    std::cout << "Enter angle of rotation: " << std::endl;
    std::cin >> angle;

    // Convert angle to radians
    float theta = angle * 3.14159 / 180;

    // Read input image and allocate memory for output image
    input_img = readBmpFloat("./Images/cat.bmp", &img_rows, &img_cols);
    output_img = (float *) malloc(sizeof(float) * img_rows * img_cols);

    try {
        // Create queue
        queue q(d_selector, dpc_common::exception_handler);

        // Create buffers for input and output images
        buffer<float, 1> input_buffer(input_img, range<1>(img_rows*img_cols));
        buffer<float, 1> output_buffer(output_img, range<1>(img_rows*img_cols));
	
        q.submit([&](handler &h){

            // Create accessors for buffers
            accessor input(input_buffer, h, read_only);
            accessor output(output_buffer, h, write_only);

            // Submit parallel_for kernel
            h.parallel_for(range<2>(img_rows, img_cols), [=](id<2> item){

                // Assign work item a row and col
                const int row = item[0];
                const int col = item[1];

                // Calculated new row and col after rotation
                float new_row = ((float)row)*cos(theta) + ((float)col)*sin(theta);
                float new_col = -1.0f*((float)row)*sin(theta) + ((float)col)*cos(theta);

                // If new row and col are within image bounds set image data from old position to new position
                if(((int)new_row >= 0) && ((int)new_row < img_cols) && ((int)new_col >= 0) && ((int)new_col < img_rows))
                    output[(int)new_col * img_rows + (int)new_row] = input[col * img_rows + row];

            });

        });


    }
    catch (exception const &e) {
        std::cout << "Exception caught for image rotation" << std::endl;
        std::terminate();
    }

    // Store rotated image
    std::cout << "Rotated image stored as rotated_image.bmp" << std::endl;
    writeBmpFloat(output_img, "rotated_image.bmp", img_rows, img_cols, "./Images/cat.bmp");

    return 0;
}

