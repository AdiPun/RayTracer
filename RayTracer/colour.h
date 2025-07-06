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

        /* scale + clamp */
        __m256 r = _mm256_min_ps(_mm256_max_ps(in.r, zero), maxv);
        __m256 g = _mm256_min_ps(_mm256_max_ps(in.g, zero), maxv);
        __m256 b = _mm256_min_ps(_mm256_max_ps(in.b, zero), maxv);

        r = _mm256_mul_ps(r, scale);
        g = _mm256_mul_ps(g, scale);
        b = _mm256_mul_ps(b, scale);

        /* convert float  int32  int16/8  */
        __m256i ri32 = _mm256_cvtps_epi32(r);
        __m256i gi32 = _mm256_cvtps_epi32(g);
        __m256i bi32 = _mm256_cvtps_epi32(b);

        /* interleave to RGBRGB order; AVX2 shuffle assists */
        __m256i rg = _mm256_or_si256(ri32, _mm256_slli_epi32(gi32, 8));
        __m256i rgb = _mm256_or_si256(rg, _mm256_slli_epi32(bi32, 16));

        /* Each 32bit lane now holds 0x00BBGGRR */
        alignas(32) uint32_t tmp[8];
        _mm256_store_si256(reinterpret_cast<__m256i*>(tmp), rgb);

        /* spill to byte array */
        for (int i = 0; i < 8; ++i)  // unrolled is fine  eight elems
        {
            out_bytes[i * 3 + 0] = static_cast<uint8_t>(tmp[i] & 0xFF);
            out_bytes[i * 3 + 1] = static_cast<uint8_t>((tmp[i] >> 8) & 0xFF);
            out_bytes[i * 3 + 2] = static_cast<uint8_t>((tmp[i] >> 16) & 0xFF);
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
            os << +rgb[i * 3] << ' ' << +rgb[i * 3 + 1] << ' ' << +rgb[i * 3 + 2] << '\n';
    }
}  // namespace colour_sys

#endif  // COLOUR_H
