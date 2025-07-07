// RayTracer.cpp : This file contains the 'main' function. Program execution
// begins and ends there.
//

#include <immintrin.h>

#include <iostream>

#include "colour.h"
#include "vector.h"

// Your Colour256 and colour_sys from before

int main() {
  constexpr int kImageWidth = 256;
  constexpr int kImageHeight = 256;

  std::cout << "P3\n" << kImageWidth << " " << kImageHeight << "\n255\n";

  for (int j = 0; j < kImageHeight; j++) {
    std::clog << "\rScanlines remaining: " << kImageHeight - j << " "
              << std::flush;

    for (int i = 0; i < kImageWidth;
         i += 8) {  // step by 8 pixels for SIMD batch
      // Create arrays of r, g for 8 pixels at once
      float r_vals[8], g_vals[8], b_vals[8], a_vals[8];
      for (int k = 0; k < 8; ++k) {
        int x = i + k;
        r_vals[k] = static_cast<float>(j) / kImageHeight;
        g_vals[k] = static_cast<float>(x) / kImageWidth;
        b_vals[k] = 0.0f;
        a_vals[k] = 1.0f;
      }

      // Load arrays into __m256 vectors
      Colour256 pixelColour{
          _mm256_loadu_ps(r_vals),
          _mm256_loadu_ps(g_vals),
          _mm256_loadu_ps(b_vals),
          _mm256_loadu_ps(a_vals),
      };

      colour_sys::write(std::cout, pixelColour);
    }
  }

  std::clog << "\rDone.                           \n";

  return 0;
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started:
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add
//   Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project
//   and select the .sln file
