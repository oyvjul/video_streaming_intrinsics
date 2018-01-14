#include <inttypes.h>
#include <math.h>
#include <stdlib.h>
#include "arm_neon.h"
#include "dsp.h"
#include "tables.h"

static void transpose_block(float *in_data, float *out_data)
{
  int i, j;

  for (i = 0; i < 8; ++i)
  {
    for (j = 0; j < 8; ++j)
    {
      out_data[i*8+j] = in_data[j*8+i];
    }
  }
}

/*static void transpose_block(float *in_data, float *out_data)
{
    float32x4_t row0, row1, row2, row3, row4, row5, row6, row7
    row8, row9, row10, row11, row12, row13, row14, row15;
    
    row0 = vld1q_f32(in_data);
    row1 = vld1q_f32(in_data+8);
    row2 = vld1q_f32(in_data+16);
    row3 = vld1q_f32(in_data+24);
    row4 = vld1q_f32(in_data+32);
    row5 = vld1q_f32(in_data+40);
    row6 = vld1q_f32(in_data+48);
    row7 = vld1q_f32(in_data+56);
    
    float32x4x2_t t0_ = vzipq_f32(row0, row2);                              
    float32x4x2_t t1_ = vzipq_f32(row1, row3);                              
    float32x4x2_t u0_ = vzipq_f32(t0_.val[0], t1_.val[0]);              
    float32x4x2_t u1_ = vzipq_f32(t0_.val[1], t1_.val[1]);              
    
    float32x4x2_t t01_ = vzipq_f32(row4, row6);                              
    float32x4x2_t t11_ = vzipq_f32(row5, row7);                              
    float32x4x2_t u01_ = vzipq_f32(t01_.val[0], t11_.val[0]);              
    float32x4x2_t u11_ = vzipq_f32(t01_.val[1], t11_.val[1]);              

    vst1q_f32(out_data, u0_.val[0]);
    vst1q_f32(out_data+8, u0_.val[1]);
    vst1q_f32(out_data+16, u1_.val[0]);
    vst1q_f32(out_data+24, u1_.val[1]);
    vst1q_f32(out_data+4, u01_.val[0]);
    vst1q_f32(out_data+12, u01_.val[1]);
    vst1q_f32(out_data+20, u11_.val[0]);
    vst1q_f32(out_data+28, u11_.val[1]);
    
    row8 = vld1q_f32(in_data+4);
    row9 = vld1q_f32(in_data+12);
    row10 = vld1q_f32(in_data+20);
    row11 = vld1q_f32(in_data+28);
    row12 = vld1q_f32(in_data+36);
    row13 = vld1q_f32(in_data+44);
    row14 = vld1q_f32(in_data+52);
    row15 = vld1q_f32(in_data+60);
    
    t0_ = vzipq_f32(row8, row10);                              
    t1_ = vzipq_f32(row9, row11);                              
    u0_ = vzipq_f32(t0_.val[0], t1_.val[0]);              
    u1_ = vzipq_f32(t0_.val[1], t1_.val[1]);  
    
    t01_ = vzipq_f32(row12, row14);                              
    t11_ = vzipq_f32(row13, row15);                              
    u01_ = vzipq_f32(t01_.val[0], t11_.val[0]);              
    u11_ = vzipq_f32(t01_.val[1], t11_.val[1]);        

    vst1q_f32(out_data+32, u0_.val[0]);
    vst1q_f32(out_data+40, u0_.val[1]);
    vst1q_f32(out_data+48, u1_.val[0]);
    vst1q_f32(out_data+56, u1_.val[1]);      
    vst1q_f32(out_data+36, u01_.val[0]);
    vst1q_f32(out_data+44, u01_.val[1]);
    vst1q_f32(out_data+52, u11_.val[0]);
    vst1q_f32(out_data+60, u11_.val[1]);
}*/

static void dct_1d(float *in_data, float *out_data)
{
    int i;

/*  Loading the first 4 elements of in_data into a vector register,
    then loading the following 4 elements into a second vector register 
    (max 4 floats in float vector registers)  */
    float __attribute__((aligned(16))) in_1[4] = { in_data[0], in_data[1], in_data[2], in_data[3] };
    float32x4_t in_data_1 = vld1q_f32(in_1);
 
    float __attribute__((aligned(16))) in_2[4] = { in_data[4], in_data[5], in_data[6], in_data[7] };
    float32x4_t in_data_2 = vld1q_f32(in_2);

/*  Vector and vector register for storing result. Elements set to 0  */
    float __attribute__((aligned(16))) vec_res[4] = { 0.f, 0.f, 0.f, 0.f };
    float32x4_t res = vdupq_n_f32(0.0f);


    for (i = 0; i < 8; ++i)
    {
/*  Loading 4 elements of dctlookup1 into a vector register, then the 4 next elements
    into a second register  */
        float __attribute__((aligned(16))) dct_1[4] = {dctlookup1[0*8+i], dctlookup1[1*8+i], dctlookup1[2*8+i], dctlookup1[3*8+i]};
        float32x4_t dct_a = vld1q_f32(dct_1);

        float __attribute__((aligned(16))) dct_2[4] = {dctlookup1[4*8+i], dctlookup1[5*8+i], dctlookup1[6*8+i], dctlookup1[7*8+i]};
        float32x4_t dct_b = vld1q_f32(dct_2);


/*  Multiplying 8 elements from in_data with 8 elements from dctlookup_trans, using the two    vector registers containing 4 elements each */
        res = vmulq_f32(in_data_1, dct_a);
        res = vmlaq_f32(res, in_data_2, dct_b);
    
/*  Storing result of the multiplication in a float vector, then adding
    all the elements of the vector and storing the result in out_data  */
        vst1q_f32(vec_res, res);
        out_data[i] = vec_res[0] + vec_res[1] + vec_res[2] + vec_res[3];
    }
}

static void idct_1d(float *in_data, float *out_data)
{
    int i;

/*  Loading the first 4 elements of in_data into a vector register,
    then loading the following 4 elements into a second vector register 
    (max 4 floats in float vector registers)  */
    float __attribute__((aligned(16))) in_1[4] = { in_data[0], in_data[1], in_data[2], in_data[3] };
    float32x4_t in_data_1 = vld1q_f32(in_1);
    
    float __attribute__((aligned(16))) in_2[4] = { in_data[4], in_data[5], in_data[6], in_data[7] };
    float32x4_t in_data_2 = vld1q_f32(in_2);

/*  Vector and vector register for storing result. Elements set to 0  */
    float __attribute__((aligned(16))) vec_res[4] = { 0.f, 0.f, 0.f, 0.f };
    float32x4_t res = vdupq_n_f32(0.0f);


    for (i = 0; i < 8; ++i)
    {
/*  Loading 4 elements of dctlookup_trans into a vector register, then the 4 next elements
    into a second register  */
        float __attribute__((aligned(16))) dct_1[4] = {dctlookup_trans[0*8+i], dctlookup_trans[1*8+i], dctlookup_trans[2*8+i], dctlookup_trans[3*8+i]};
        float32x4_t dct_a = vld1q_f32(dct_1);

        float __attribute__((aligned(16))) dct_2[4] = {dctlookup_trans[4*8+i], dctlookup_trans[5*8+i], dctlookup_trans[6*8+i], dctlookup_trans[7*8+i]};
        float32x4_t dct_b = vld1q_f32(dct_2);

/*  Multiplying 8 elements from in_data with 8 elements from dctlookup_trans, using the two    vector registers containing 4 elements each */
        res = vmulq_f32(in_data_1, dct_a);
        res = vmlaq_f32(res, in_data_2, dct_b);

/*  Storing result of the multiplication in a float vector, then adding
    all the elements of the vector and storing the result in out_data  */ 
        vst1q_f32(vec_res, res);
        out_data[i] = vec_res[0] + vec_res[1] + vec_res[2] + vec_res[3];
    }
}

static void scale_block(float *in_data, float *out_data)
{
    int i;
    float __attribute__((aligned(32))) a_values[4] = { ISQRT2, 1, 1, 1 };
    float32x4_t val, col_0, col_1;
    
    val = vld1q_f32(&(a_values[0]));
    
    
    out_data[0] = in_data[0] * .5f;
    out_data[1] = in_data[1] * ISQRT2;
    out_data[2] = in_data[2] * ISQRT2;
    out_data[3] = in_data[3] * ISQRT2;
    out_data[4] = in_data[4] * ISQRT2;
    out_data[5] = in_data[5] * ISQRT2;
    out_data[6] = in_data[6] * ISQRT2;
    out_data[7] = in_data[7] * ISQRT2;
    
    for (i = 1; i < 8; ++i)
    {
        col_0 = vld1q_f32(&(in_data[i * 8]));
        col_1 = vld1q_f32(&(in_data[i * 8 + 4]));
        
        col_0 = vmulq_f32(col_0, val);
        
        vst1q_f32(&(out_data[i * 8]), col_0);
        vst1q_f32(&(out_data[i * 8 + 4]), col_1);
    }
}

static void quantize_block(float *in_data, float *out_data, uint8_t *quant_tbl)
{
  int i;
  uint8x16_t zz_u1, zz_u2, zz_u3, zz_u4, zz_v1, zz_v2, zz_v3, zz_v4;
  uint8x16_t mul_1 = vdupq_n_u8(8);

    /*  Loading 64 elements from zigzag_V/U into 8 vector registers (16 in each)
    Calculating indexes used for in_data by multiplying and adding registers
    (zigzag_V[x]*8+zigzag_U[x]) */
    zz_u1 = vld1q_u8(zigzag_U);
    zz_v1 = vld1q_u8(zigzag_V);
    zz_v1 = vmulq_u8(zz_v1, mul_1);
    zz_v1 = vaddq_u8(zz_v1, zz_u1);

    zz_u2 = vld1q_u8(zigzag_U + 16);
    zz_v2 = vld1q_u8(zigzag_V + 16);
    zz_v2 = vmulq_u8(zz_v2, mul_1);
    zz_v2 = vaddq_u8(zz_v2, zz_u2);

    zz_u3 = vld1q_u8(zigzag_U + 32);
    zz_v3 = vld1q_u8(zigzag_V + 32);
    zz_v3 = vmulq_u8(zz_v3, mul_1);
    zz_v3 = vaddq_u8(zz_v3, zz_u3);

    zz_u4 = vld1q_u8(zigzag_U + 48);
    zz_v4 = vld1q_u8(zigzag_V + 48);
    zz_v4 = vmulq_u8(zz_v4, mul_1);
    zz_v4 = vaddq_u8(zz_v4, zz_u4);


/* Zig-zag and quantize. Using indexes stored in vector registers 
   to get data from in_data */
    for (i = 0; i < 16; i++) 
    {
        out_data[i] = (float) round((in_data[zz_v1[i]] / 4.0) / quant_tbl[i]);
        out_data[i+16] = (float) round((in_data[zz_v2[i]] / 4.0) / quant_tbl[i+16]);
        out_data[i+32] = (float) round((in_data[zz_v3[i]] / 4.0) / quant_tbl[i+32]);
        out_data[i+48] = (float) round((in_data[zz_v4[i]] / 4.0) / quant_tbl[i+48]);
    }  
}


static void dequantize_block(float *in_data, float *out_data,
    uint8_t *quant_tbl)
{
    int i;
    uint8x16_t zz_u1, zz_u2, zz_u3, zz_u4, zz_v1, zz_v2, zz_v3, zz_v4;
    uint8x16_t mul_1 = vdupq_n_u8(8);

    /*  Loading 64 elements from zigzag_V/U into 8 vector registers (16 in each)
    Calculating indexes used for in_data by multiplying and adding registers
    (zigzag_V[x]*8+zigzag_U[x]) */
    zz_u1 = vld1q_u8(zigzag_U);
    zz_v1 = vld1q_u8(zigzag_V);
    zz_v1 = vmulq_u8(zz_v1, mul_1);
    zz_v1 = vaddq_u8(zz_v1, zz_u1);

    zz_u2 = vld1q_u8(zigzag_U + 16);
    zz_v2 = vld1q_u8(zigzag_V + 16);
    zz_v2 = vmulq_u8(zz_v2, mul_1);
    zz_v2 = vaddq_u8(zz_v2, zz_u2);

    zz_u3 = vld1q_u8(zigzag_U + 32);
    zz_v3 = vld1q_u8(zigzag_V + 32);
    zz_v3 = vmulq_u8(zz_v3, mul_1);
    zz_v3 = vaddq_u8(zz_v3, zz_u3);

    zz_u4 = vld1q_u8(zigzag_U + 48);
    zz_v4 = vld1q_u8(zigzag_V + 48);
    zz_v4 = vmulq_u8(zz_v4, mul_1);
    zz_v4 = vaddq_u8(zz_v4, zz_u4);


    /* Zig-zag and quantize. Storing result in out_data, using indexes stored in
   vector registers */
    for (i = 0; i < 16; ++i)
    {
        out_data[zz_v1[i]] = (float) round((in_data[i] * quant_tbl[i]) / 4.0);
        out_data[zz_v2[i]] = (float) round((in_data[i+16] * quant_tbl[i+16]) / 4.0);
        out_data[zz_v3[i]] = (float) round((in_data[i+32] * quant_tbl[i+32]) / 4.0);
        out_data[zz_v4[i]] = (float) round((in_data[i+48] * quant_tbl[i+48]) / 4.0);
    }
}


void dct_quant_block_8x8(int16_t *in_data, int16_t *out_data,
    uint8_t *quant_tbl)
{
  float mb[8*8] __attribute((aligned(16)));
  float mb2[8*8] __attribute((aligned(16)));

  int i;

  for (i = 0; i < 64; ++i) 
  { 
      mb2[i] = in_data[i]; 
  }

  /* Two 1D DCT operations with transpose */
    dct_1d(mb2+0*8, mb+0*8); 
    dct_1d(mb2+1*8, mb+1*8); 
    dct_1d(mb2+2*8, mb+2*8); 
    dct_1d(mb2+3*8, mb+3*8); 
    dct_1d(mb2+4*8, mb+4*8); 
    dct_1d(mb2+5*8, mb+5*8); 
    dct_1d(mb2+6*8, mb+6*8); 
    dct_1d(mb2+7*8, mb+7*8); 
    
    transpose_block(mb, mb2);
    
    dct_1d(mb2+0*8, mb+0*8); 
    dct_1d(mb2+1*8, mb+1*8); 
    dct_1d(mb2+2*8, mb+2*8); 
    dct_1d(mb2+3*8, mb+3*8); 
    dct_1d(mb2+4*8, mb+4*8); 
    dct_1d(mb2+5*8, mb+5*8); 
    dct_1d(mb2+6*8, mb+6*8); 
    dct_1d(mb2+7*8, mb+7*8); 
    
    transpose_block(mb, mb2);

  scale_block(mb2, mb);
  quantize_block(mb, mb2, quant_tbl);

  for (i = 0; i < 64; ++i) { out_data[i] = mb2[i]; }
}

void dequant_idct_block_8x8(int16_t *in_data, int16_t *out_data,
    uint8_t *quant_tbl)
{
  float mb[8*8] __attribute((aligned(16)));
  float mb2[8*8] __attribute((aligned(16)));

  int i;

  for (i = 0; i < 64; ++i) { mb[i] = in_data[i]; }

  dequantize_block(mb, mb2, quant_tbl);
  scale_block(mb2, mb);

  /* Two 1D inverse DCT operations with transpose */
    idct_1d(mb+0*8, mb2+0*8); 
    idct_1d(mb+1*8, mb2+1*8); 
    idct_1d(mb+2*8, mb2+2*8); 
    idct_1d(mb+3*8, mb2+3*8); 
    idct_1d(mb+4*8, mb2+4*8); 
    idct_1d(mb+5*8, mb2+5*8); 
    idct_1d(mb+6*8, mb2+6*8); 
    idct_1d(mb+7*8, mb2+7*8); 
    
    transpose_block(mb2, mb);
    
    idct_1d(mb+0*8, mb2+0*8); 
    idct_1d(mb+1*8, mb2+1*8); 
    idct_1d(mb+2*8, mb2+2*8); 
    idct_1d(mb+3*8, mb2+3*8); 
    idct_1d(mb+4*8, mb2+4*8); 
    idct_1d(mb+5*8, mb2+5*8); 
    idct_1d(mb+6*8, mb2+6*8); 
    idct_1d(mb+7*8, mb2+7*8); 
    
    transpose_block(mb2, mb);

  for (i = 0; i < 64; ++i) { out_data[i] = mb[i]; }
}

void sad_block_8x8(uint8_t *block1, uint8_t *block2, int stride, int *result)
{
    uint8_t __attribute__((aligned(16))) vec_res[16] = { 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ,0 , 0, 0, 0, 0, 0, 0};
    uint8x16_t res = vld1q_u8(vec_res);
    uint8x16_t m_1, m_2, m_3, m_4, m_5, m_6, m_7, m_8;
    *result = 0;

/*  Loading 16 elements from each block into each register 
    (64 in total for each block) */
    m_1 = vld1q_u8(block1 + 0*stride);
    m_2 = vld1q_u8(block2 + 0*stride);
        
    m_3 = vld1q_u8(block1 + 2*stride);
    m_4 = vld1q_u8(block2 + 2*stride);
    
    m_5 = vld1q_u8(block1 + 4*stride);
    m_6 = vld1q_u8(block2 + 4*stride);
    
    m_7 = vld1q_u8(block1 + 6*stride);
    m_8 = vld1q_u8(block2 + 6*stride);
        
/*  Calculating absolute difference of elements in registers (block1 & block2)
    and accumulating into res. Repeat 4 times to get abs of all elements. */
    res = vabaq_u8(res, m_2, m_1);
    res = vabaq_u8(res, m_4, m_3);
    res = vabaq_u8(res, m_6, m_5);
    res = vabaq_u8(res, m_8, m_7);
    
/*  Storing res into a vector of type uint8_t.
    Res contains the abs of the elements in block1/2 */    
    vst1q_u8(vec_res, res);
    
/*  Adding all elements in vec_res to get the final result */
    *result = vec_res[0] + vec_res[1] + vec_res[2] + vec_res[3] + vec_res[4] + vec_res[5] + vec_res[6] + vec_res[7] + vec_res[8] + vec_res[9] + vec_res[10] + vec_res[11] + vec_res[12] + vec_res[13] + vec_res[14] + vec_res[15];
}