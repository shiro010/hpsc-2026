#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <x86intrin.h>

int main() {
  const int N = 16;
  float x[N], y[N], m[N], fx[N], fy[N];
  for(int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0;
  }
  __m512 xj = _mm512_load_ps(x);
  __m512 yj = _mm512_load_ps(y);
  __m512 m_vec = _mm512_load_ps(m);

  for(int i=0; i<N; i++) {
    __m512 xi = _mm512_set1_ps(x[i]);
    __m512 yi = _mm512_set1_ps(y[i]);

    __m512 rx = _mm512_sub_ps(xi, xj);
    __m512 ry = _mm512_sub_ps(yi, yj);

    __m512 rx2 = _mm512_mul_ps(rx, rx);
    __m512 ry2 = _mm512_mul_ps(ry, ry);
    __m512 r2 = _mm512_add_ps(rx2, ry2);

    __m512 inv_r = _mm512_rsqrt14_ps(r2);
    __m512 inv_r3 = _mm512_mul_ps(_mm512_mul_ps(inv_r, inv_r), inv_r);

    __m512 zero = _mm512_setzero_ps();
    __mmask16 mask = _mm512_cmp_ps_mask(r2, zero, _CMP_NEQ_OQ);
    __m512 m_r3 = _mm512_maskz_mul_ps(mask, m_vec, inv_r3);

    __m512 force_x = _mm512_mul_ps(rx, m_r3);
    __m512 force_y = _mm512_mul_ps(ry, m_r3);

    fx[i] -= _mm512_reduce_add_ps(force_x);
    fy[i] -= _mm512_reduce_add_ps(force_y);

    printf("%d %g %g\n",i,fx[i],fy[i]);
  }
}
