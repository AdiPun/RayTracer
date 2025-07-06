// RayTracer.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include "vector.h"
#include "colour.h"

int main()
{
	// Image

	constexpr int kImageWidth = 256;
	constexpr int kImageHeight = 256;

	constexpr float kFloatByte = 255.999f;

	Vec256f3 woah;


	// Render

	std::cout << "P3\n" << kImageWidth << " " << kImageHeight << "\n255\n";

	for (int j = 0; j < kImageHeight; j++)
	{
		// Progress indicator every scanline
		std::clog << "\rScanlines remaining: " << kImageHeight - j << " " << std::flush;
		for (int i = 0; i < kImageWidth; i++)
		{
			float r{ static_cast<float>(j) / kImageHeight};
			float g{ static_cast<float>(i) / kImageWidth };
			float b{ 0.0f };
			float a{ 1.0f };

			Colour256 pixelColour{ r,g,b,a };
			colour_sys::write(std::cout, pixelColour);
		}
	}

	std::clog << "\rDone.                           \n";
}



// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
