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
      Colour256 pixelColour(r_vals, g_vals, b_vals, a_vals);


      colour_sys::write(std::cout, pixelColour);
    }
  }

  std::clog << "\rDone.                           \n";

  return 0;
}
