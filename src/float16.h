#ifndef _FLOAT16_H_
#define _FLOAT16_H_

#include <driver_types.h>
#include <cuda_fp16.h>

half cpu_float2half(float f);
float cpu_half2float(half h);

struct float16 {
  inline float16() { data.x = 0; }

  inline float16(const float &rhs) {
  	data = cpu_float2half(rhs);
  }
  
  inline operator float() const {
  	return cpu_half2float(data);
  }

  half data;
};

#endif