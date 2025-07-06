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

 // Data-oriented Vec2/Vec3/Vec4 structs with AVX batching and SIMD support

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

/*  Vec256f3 : 8 Vec3f  */
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

	/* Cross product, returning 8 Vec3f */
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
				_mm256_mul_ps(
					_mm256_mul_ps(len2, inv),  // len2 * inv
					_mm256_mul_ps(inv, _mm256_set1_ps(0.5f))
				)
			)
		);

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

#ifndef VEC3_H
#define VEC3_H

#include <cmath>
#include <iostream>

class Vec3f
{
public:
	float e[3];

	Vec3f() : e{ 0,0,0 }
	{
	}
	Vec3f(float e0, float e1, float e2) : e{ e0, e1, e2 }
	{
	}

	float x() const
	{
		return e[0];
	}
	float y() const
	{
		return e[1];
	}
	float z() const
	{
		return e[2];
	}

	Vec3f operator-() const
	{
		return Vec3f(-e[0], -e[1], -e[2]);
	}
	float operator[](int i) const
	{
		return e[i];
	}
	float& operator[](int i)
	{
		return e[i];
	}

	Vec3f& operator+=(const Vec3f& v)
	{
		e[0] += v.e[0];
		e[1] += v.e[1];
		e[2] += v.e[2];
		return *this;
	}

	Vec3f& operator*=(float t)
	{
		e[0] *= t;
		e[1] *= t;
		e[2] *= t;
		return *this;
	}

	Vec3f& operator/=(float t)
	{
		return *this *= 1 / t;
	}

	float length() const
	{
		return std::sqrt(length_squared());
	}

	float length_squared() const
	{
		return e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
	}
};

// point3 is just an alias for Vec3f, but useful for geometric clarity in the code.
using point3 = Vec3f;


// Vector Utility Functions

inline std::ostream& operator<<(std::ostream& out, const Vec3f& v)
{
	return out << v.e[0] << ' ' << v.e[1] << ' ' << v.e[2];
}

inline Vec3f operator+(const Vec3f& u, const Vec3f& v)
{
	return Vec3f(u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]);
}

inline Vec3f operator-(const Vec3f& u, const Vec3f& v)
{
	return Vec3f(u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]);
}

inline Vec3f operator*(const Vec3f& u, const Vec3f& v)
{
	return Vec3f(u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]);
}

inline Vec3f operator*(float t, const Vec3f& v)
{
	return Vec3f(t * v.e[0], t * v.e[1], t * v.e[2]);
}

inline Vec3f operator*(const Vec3f& v, float t)
{
	return t * v;
}

inline Vec3f operator/(const Vec3f& v, float t)
{
	return (1 / t) * v;
}

inline float dot(const Vec3f& u, const Vec3f& v)
{
	return u.e[0] * v.e[0]
		+ u.e[1] * v.e[1]
		+ u.e[2] * v.e[2];
}

inline Vec3f cross(const Vec3f& u, const Vec3f& v)
{
	return Vec3f(u.e[1] * v.e[2] - u.e[2] * v.e[1],
		u.e[2] * v.e[0] - u.e[0] * v.e[2],
		u.e[0] * v.e[1] - u.e[1] * v.e[0]);
}

inline Vec3f unit_vector(const Vec3f& v)
{
	return v / v.length();
}

#endif