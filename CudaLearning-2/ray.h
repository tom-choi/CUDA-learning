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
    // ԭ���@�e��:
    // point3 orig;
    // vec3 dir;

    // �yһʹ�� vec3 (��������ĽY����һ��)
    vec3 orig;
    vec3 dir;
};

#endif // !RAYH
