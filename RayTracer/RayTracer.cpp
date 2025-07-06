// RayTracer.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>

int main()
{
	// Image

	constexpr int kImageWidth = 256;
	constexpr int kImageHeight = 256;

	constexpr float kFloatByte = 255.999f;


	// Pixel
	struct Pixel
	{
		float r{ 0.0f };
		float g{ 0.0f };
		float b{ 0.0f };
		float a{ 1.0f };
	};

	// Render

	std::cout << "P3\n" << kImageWidth << " " << kImageHeight << "\n255\n";

	for (int j = 0; j < kImageHeight; j++)
	{
		// Progress indicator every scanline
		std::clog << "\rScanlines remaining: " << kImageHeight - j << " " << std::flush;
		for (int i = 0; i < kImageWidth; i++)
		{
			float r = static_cast<float>(i) / (kImageWidth - 1);
			float g = static_cast<float>(j) / (kImageHeight - 1);
			float b = 0.0f;

			int iR = static_cast<int>(kFloatByte * r);
			int iG = static_cast<int>(kFloatByte * g);
			int iB = static_cast<int>(kFloatByte * b);

			std::cout << iR << " " << iG << " " << iB << "\n";
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
