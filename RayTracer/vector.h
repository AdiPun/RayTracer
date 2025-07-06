#pragma once
#ifndef VEC_AVX_BATCHES_H
#define VEC_AVX_BATCHES_H
/*
   Lightweight AVX batch vector library
   Public domain / CC0 enjoy
 */

#include <immintrin.h>   // AVX intrinsics
#include <cstddef>
#include <cassert>
#include <cmath>         // only for scalar fallbacks if you add any

/*  Helpers  */
namespace avx
{
    /* Horizontal add of all 8 lanes to a single scalar float */
    inline float hadd8(__m256 v)
    {
        __m128 hi = _mm256_extractf128_ps(v, 1);   // upper 128
        __m128 lo = _mm256_castps256_ps128(v);     // lower 128
        __m128 sum = _mm_add_ps(lo, hi);           // 4lane sum
        sum = _mm_hadd_ps(sum, sum);
        sum = _mm_hadd_ps(sum, sum);
        return _mm_cvtss_f32(sum);
    }
}

/*  Vec256f2 : 8 vec2  */
struct Vec256f2
{
    __m256 x, y;                        // AX, AY

    /* Load eight vec2s from AoS memory layout (xy xy xy) */
    static Vec256f2 load_aos(const float* src)
    {
        /* deinterleave x & y with gatherish shuffle */
        __m256 xy01 = _mm256_loadu_ps(src + 0);    // v0.x v0.y v1.x v1.y v2.x v2.y v3.x v3.y
        __m256 xy23 = _mm256_loadu_ps(src + 8);
        __m256 xy45 = _mm256_loadu_ps(src + 16);
        __m256 xy67 = _mm256_loadu_ps(src + 24);

        __m256 shuf0 = _mm256_permute_ps(xy01, 0b11011000); // x0 x1 y0 y1 ...
        __m256 shuf1 = _mm256_permute_ps(xy23, 0b11011000);
        __m256 shuf2 = _mm256_permute_ps(xy45, 0b11011000);
        __m256 shuf3 = _mm256_permute_ps(xy67, 0b11011000);

        __m256 xs = _mm256_shuffle_ps(shuf0, shuf2, 0b10001000); // x0 x1 x2 x3 x4 x5 x6 x7
        __m256 ys = _mm256_shuffle_ps(shuf1, shuf3, 0b10001000); // y0 y1 

        return { xs, ys };
    }

    /* Elementwise + two batches */
    friend Vec256f2 operator+(const Vec256f2& a, const Vec256f2& b)
    {
        return { _mm256_add_ps(a.x, b.x), _mm256_add_ps(a.y, b.y) };
    }

    /* Scale by scalar s */
    friend Vec256f2 operator*(const Vec256f2& v, float s)
    {
        __m256 ss = _mm256_set1_ps(s);
        return { _mm256_mul_ps(v.x, ss), _mm256_mul_ps(v.y, ss) };
    }

    /* Dot product pervector  return 8 packed dot values */
    __m256 dot(const Vec256f2& b) const
    {
        __m256 mul = _mm256_fmadd_ps(x, b.x, _mm256_mul_ps(y, b.y));
        return mul;                    // [d0d7]
    }
};

/*  Vec256f3 : 8 vec3  */
struct Vec256f3
{
    __m256 x, y, z;

    friend Vec256f3 operator+(const Vec256f3& a, const Vec256f3& b)
    {
        return { _mm256_add_ps(a.x, b.x),
                 _mm256_add_ps(a.y, b.y),
                 _mm256_add_ps(a.z, b.z) };
    }

    friend Vec256f3 operator*(const Vec256f3& v, float s)
    {
        __m256 ss = _mm256_set1_ps(s);
        return { _mm256_mul_ps(v.x, ss),
                 _mm256_mul_ps(v.y, ss),
                 _mm256_mul_ps(v.z, ss) };
    }

    /* Dot  8 packed dots */
    __m256 dot(const Vec256f3& b) const
    {
        __m256 t = _mm256_mul_ps(x, b.x); // x * b.x -> t
        t = _mm256_fmadd_ps(y, b.y, t);   // t += y * b.y
        t = _mm256_fmadd_ps(z, b.z, t);   // t += z * b.z
        return t;
    }

    /* Cross product, returning 8 vec3 */
    Vec256f3 cross(const Vec256f3& b) const
    {
        return {
            _mm256_fmsub_ps(y, b.z, _mm256_mul_ps(z, b.y)), // y*bz - z*by
            _mm256_fmsub_ps(z, b.x, _mm256_mul_ps(x, b.z)), // z*bx - x*bz
            _mm256_fmsub_ps(x, b.y, _mm256_mul_ps(y, b.x))  // x*by - y*bx
        };
    }

    /* Normalise each of the 8 vectors */
    void normalise()
    {
        __m256 len2 = dot(*this);                   // x+y+z
        __m256 inv = _mm256_rsqrt_ps(len2);         // rough 1/len2
        // One NewtonRaphson for accuracy
        inv = _mm256_mul_ps(inv,
            _mm256_sub_ps(_mm256_set1_ps(1.5f),
                _mm256_mul_ps(_mm256_mul_ps(len2, inv),   // len2*inv
                    _mm256_mul_ps(inv, _mm256_set1_ps(0.5f)));
        x = _mm256_mul_ps(x, inv);
        y = _mm256_mul_ps(y, inv);
        z = _mm256_mul_ps(z, inv);
    }
};

/*  Vec256f4 : 8 vec4  */
struct Vec256f4
{
    __m256 x, y, z, w;

    friend Vec256f4 operator+(const Vec256f4& a, const Vec256f4& b)
    {
        return { _mm256_add_ps(a.x, b.x),
                 _mm256_add_ps(a.y, b.y),
                 _mm256_add_ps(a.z, b.z),
                 _mm256_add_ps(a.w, b.w) };
    }

    friend Vec256f4 operator*(const Vec256f4& v, float s)
    {
        __m256 ss = _mm256_set1_ps(s);
        return { _mm256_mul_ps(v.x, ss),
                 _mm256_mul_ps(v.y, ss),
                 _mm256_mul_ps(v.z, ss),
                 _mm256_mul_ps(v.w, ss) };
    }

    __m256 dot(const Vec256f4& b) const
    {
        __m256 t = _mm256_mul_ps(x, b.x);
        t = _mm256_fmadd_ps(y, b.y, t);
        t = _mm256_fmadd_ps(z, b.z, t);
        t = _mm256_fmadd_ps(w, b.w, t);
        return t;
    }

    void normalise()
    {
        __m256 len2 = dot(*this);
        __m256 inv = _mm256_rsqrt_ps(len2);
        // (optional) refine like Vec256f3::normalise()
        x = _mm256_mul_ps(x, inv);
        y = _mm256_mul_ps(y, inv);
        z = _mm256_mul_ps(z, inv);
        w = _mm256_mul_ps(w, inv);
    }
};

#endif // VEC_AVX_BATCHES_H
