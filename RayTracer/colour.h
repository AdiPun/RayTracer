#pragma once
#ifndef COLOUR_H
#define COLOUR_H
/*  Dataoriented colour helpers  AVX edition
    Public domain / CC0                                                     */

#include <immintrin.h>  // AVX intrinsics

#include <algorithm>
#include <cstdint>
#include <iostream>

/*  Plain AoS colour  */
struct Colour  // 32bit float per channel, no methods
{
  float r, g, b, a;  // keep alpha for free even if unused
};

/*  8wide SoA colour
   Each __m256 holds 8 floats  eight pixels processed at once   */
struct Colour256 {
  __m256 r, g, b, a;
};

/*  Colour systems  */
namespace colour_sys {
/* ------ scale & clamp AoS pixel  24bit RGB bytes -------- */
inline void to_rgb24(const Colour& in, uint8_t bytes[3]) {
  auto clamp01 = [](float v) { return std::clamp(v, 0.0f, 0.999f); };
  bytes[0] = static_cast<uint8_t>(255.999f * clamp01(in.r));
  bytes[1] = static_cast<uint8_t>(255.999f * clamp01(in.g));
  bytes[2] = static_cast<uint8_t>(255.999f * clamp01(in.b));
}

/* ------ SIMD: scale & clamp eight pixels at once ----------- */
inline void to_rgb24(const Colour256& in, uint8_t out_bytes[24]) {
  const __m256 scale = _mm256_set1_ps(255.999f);
  const __m256 zero = _mm256_setzero_ps();
  const __m256 maxv = _mm256_set1_ps(0.999f);

  // Clamp and scale
  __m256 r = _mm256_min_ps(_mm256_max_ps(in.r, zero), maxv);
  __m256 g = _mm256_min_ps(_mm256_max_ps(in.g, zero), maxv);
  __m256 b = _mm256_min_ps(_mm256_max_ps(in.b, zero), maxv);

  r = _mm256_mul_ps(r, scale);
  g = _mm256_mul_ps(g, scale);
  b = _mm256_mul_ps(b, scale);

  // Convert to int32
  __m256i ri32 = _mm256_cvtps_epi32(r);
  __m256i gi32 = _mm256_cvtps_epi32(g);
  __m256i bi32 = _mm256_cvtps_epi32(b);

  // Extract low and high 128-bit lanes
  __m128i ri32_lo = _mm256_castsi256_si128(ri32);
  __m128i ri32_hi = _mm256_extracti128_si256(ri32, 1);
  __m128i gi32_lo = _mm256_castsi256_si128(gi32);
  __m128i gi32_hi = _mm256_extracti128_si256(gi32, 1);
  __m128i bi32_lo = _mm256_castsi256_si128(bi32);
  __m128i bi32_hi = _mm256_extracti128_si256(bi32, 1);

  // Pack 32-bit int to 16-bit unsigned with saturation
  __m128i r16_lo = _mm_packus_epi32(ri32_lo, ri32_hi);
  __m128i g16_lo = _mm_packus_epi32(gi32_lo, gi32_hi);
  __m128i b16_lo = _mm_packus_epi32(bi32_lo, bi32_hi);

  // Pack 16-bit unsigned to 8-bit unsigned with saturation
  __m128i r8 = _mm_packus_epi16(r16_lo, _mm_setzero_si128());
  __m128i g8 = _mm_packus_epi16(g16_lo, _mm_setzero_si128());
  __m128i b8 = _mm_packus_epi16(b16_lo, _mm_setzero_si128());

  // Now r8, g8, b8 each contain at least 8 bytes of 8-bit pixel data
  alignas(16) uint8_t rArr[16], gArr[16], bArr[16];
  _mm_store_si128(reinterpret_cast<__m128i*>(rArr), r8);
  _mm_store_si128(reinterpret_cast<__m128i*>(gArr), g8);
  _mm_store_si128(reinterpret_cast<__m128i*>(bArr), b8);

  // Interleave RGB for output: RGBRGBRGB...
  for (int i = 0; i < 8; ++i) {
    out_bytes[i * 3 + 0] = rArr[i];
    out_bytes[i * 3 + 1] = gArr[i];
    out_bytes[i * 3 + 2] = bArr[i];
  }
}

/* ------ Convenience: write one pixel to stream ------------- */
inline void write(std::ostream& os, const Colour& c) {
  uint8_t rgb[3];
  to_rgb24(c, rgb);
  os << +rgb[0] << ' ' << +rgb[1] << ' ' << +rgb[2] << '\n';
}

/* ------ Convenience: write eight pixels -------------------- */
inline void write(std::ostream& os, const Colour256& batch) {
  uint8_t rgb[24];
  to_rgb24(batch, rgb);
  for (int i = 0; i < 8; ++i) 
  {
    os << +rgb[i * 3] << ' ' << +rgb[i * 3 + 1] << ' ' << +rgb[i * 3 + 2]
       << '\n';
  }
}
}  // namespace colour_sys

#endif  // COLOUR_H
