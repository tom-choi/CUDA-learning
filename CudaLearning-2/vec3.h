#ifndef VEC3H
#define VEC3H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <math.h>
#include <stdlib.h>
#include <iostream>

class vec3 {
public:
	__host__ __device__ vec3() {}
	__host__ __device__ vec3(float e0, float e1, float e2) { e[0] = e0, e[1] = e1, e[2] = e2; }

	__host__ __device__ inline float x() const { return e[0]; }
	__host__ __device__ inline float y() const { return e[1]; }
	__host__ __device__ inline float z() const { return e[2]; }
	// 多寫一個rgb以備不時之需
	// 這裡以後會改寫 main.cu 中的rgb函數返回值
	// float r = fb[pixel_index + 0];
	// float g = fb[pixel_index + 1];
	// float b = fb[pixel_index + 2];
	__host__ __device__ inline float r() const { return e[0]; }
	__host__ __device__ inline float g() const { return e[1]; }
	__host__ __device__ inline float b() const { return e[2]; }

	// 之前沒有寫 operator+()，這次多寫一個
	__host__ __device__ inline const vec3& operator+() const { return *this; }
	// 負向量
	__host__ __device__ inline vec3 operator-() const { return vec3(-e[0], -e[1], e[2]); }
	__host__ __device__ inline float operator[](int i) const { return e[i]; }
	__host__ __device__ inline float& operator[](int i) { return e[i]; }

	// +-*/
	__host__ __device__ inline vec3& operator+=(const vec3& v2);
	__host__ __device__ inline vec3& operator-=(const vec3& v2);
	__host__ __device__ inline vec3& operator*=(const vec3& v2);
	__host__ __device__ inline vec3& operator/=(const vec3& v2);
	__host__ __device__ inline vec3& operator*=(const float t);
	__host__ __device__ inline vec3& operator/=(const float t);

	// 向量長度
	__host__ __device__ inline float length() const { return sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2]); }
	// 向量xyz平方和
	__host__ __device__ inline float squared_length() const { return e[0] * e[0] + e[1] * e[1] + e[2] * e[2]; }
	// 如果向量在所有維度上都非常接近於零，則返回 true。
	__host__ __device__ inline bool near_zero() const
	{
		const auto s = 1e-8;
		return (fabs(e[0]) < s) && (fabs(e[1]) < s) && (fabs(e[2]) < s);
	}
	__host__ __device__ inline void make_unit_vector();
	
public:
	// 這裡使用的是flaot，因為double會消耗2~3倍的時間
	float e[3];
};


// 通用函數列表

// 輸出向量e1e2e3
inline std::ostream& operator<<(std::ostream& out, const vec3& v) {
	return out << v.e[0] << ' ' << v.e[1] << ' ' << v.e[2];
}
// 多寫一個
inline std::istream& operator>>(std::istream& in, vec3& v) {
	return in >> v.e[0] >> v.e[1] >> v.e[2];
}

__host__ __device__ inline void vec3::make_unit_vector() {
	float k = 1.0 / sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2]);
	e[0] *= k; e[1] *= k; e[2] *= k;
}

// 向量與向量的+-*/

__host__ __device__ inline vec3 operator+(const vec3& u, const vec3& v) {
	return vec3(u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]);
}

__host__ __device__ inline vec3 operator-(const vec3& u, const vec3& v) {
	return vec3(u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]);
}

__host__ __device__ inline vec3 operator*(const vec3& u, const vec3& v) {
	return vec3(u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]);
}

__host__ __device__ inline vec3 operator/(const vec3& v1, const vec3& v2) {
	return vec3(v1.e[0] / v2.e[0], v1.e[1] / v2.e[1], v1.e[2] / v2.e[2]);
}

// 乘以常數

__host__ __device__ inline vec3 operator*(double t, const vec3& v)
{
	return vec3(t * v.e[0], t * v.e[1], t * v.e[2]);
}

__host__ __device__ inline vec3 operator*(const vec3& v, double t)
{
	return t * v;
}

__host__ __device__ inline vec3 operator/(vec3 v, double t)
{
	return (1 / t) * v;
}

// 點乘
__host__ __device__ inline double dot(const vec3& u, const vec3& v)
{
	return u.e[0] * v.e[0] + u.e[1] * v.e[1] + u.e[2] * v.e[2];
}

// 外積 
// https://zh.wikipedia.org/zh-tw/%E5%8F%89%E7%A7%AF
__host__ __device__ inline vec3 cross(const vec3& u, const vec3& v)
{
	return vec3(u.e[1] * v.e[2] - u.e[2] * v.e[1],
		u.e[2] * v.e[0] - u.e[0] * v.e[2],
		u.e[0] * v.e[1] - u.e[1] * v.e[0]);
}

__host__ __device__ inline vec3 unit_vector(vec3 v) {
	return v / v.length();
}





////// 統一放在外面
// 向量加法
__host__ __device__ inline vec3& vec3::operator+=(const vec3& v)
{
	e[0] += v.e[0];
	e[1] += v.e[1];
	e[2] += v.e[2];
	return *this;
}
// 向量減法
__host__ __device__ inline vec3& vec3::operator-=(const vec3& v)
{
	e[0] -= v.e[0];
	e[1] -= v.e[1];
	e[2] -= v.e[2];
	return *this;
}
// 向量與向量相乘
__host__ __device__ inline vec3& vec3::operator*=(const vec3& v)
{
	e[0] *= v.e[0];
	e[1] *= v.e[1];
	e[2] *= v.e[2];
	return *this;
}
// 向量與向量相除
__host__ __device__ inline vec3& vec3::operator/=(const vec3& v)
{
	e[0] /= v.e[0];
	e[1] /= v.e[1];
	e[2] /= v.e[2];
	return *this;
}
// 向量乘法
__host__ __device__ inline vec3& vec3::operator*=(const float t)
{
	e[0] *= t;
	e[1] *= t;
	e[2] *= t;
	return *this;
}
__host__ __device__ inline vec3& vec3::operator/=(const float t)
{
	return *this *= 1 / t;
}

#endif