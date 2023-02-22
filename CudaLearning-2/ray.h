#ifndef RAYH
#define RAYH
#include "vec3.h"

class ray {
public:
    __device__ ray() {}
    __device__ ray(const vec3& origin, const vec3& direction)
        : orig(origin), dir(direction)
    {}

    __device__ vec3 origin() const { return orig; }
    __device__ vec3 direction() const { return dir; }

    // at ---> point_at_parameter
    __device__ vec3 point_at_parameter(double t) const {
        return orig + t * dir;
    }

public:
    // 原先這裡是:
    // point3 orig;
    // vec3 dir;

    // 統一使用 vec3 (因為它們的結構都一樣)
    vec3 orig;
    vec3 dir;
};

#endif // !RAYH
