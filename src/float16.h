#ifndef _FLOAT16_H_
#define _FLOAT16_H_

#include <driver_types.h>
#include <cuda_fp16.h>

half Cpu_Float2Half(float f);
float Cpu_Half2Float(half h);

struct float16 {
  inline float16() { data.x = 0; }

  inline float16(const float &rhs) {
  	data = Cpu_Float2Half(rhs);
  }
  
  inline operator float() const {
  	return Cpu_Half2Float(data);
  }

  half data;
};

#endif