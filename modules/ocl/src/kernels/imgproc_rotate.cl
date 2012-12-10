/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2010-2012, Institute Of Software Chinese Academy Of Science, all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Zhang Ying, zhangying913@gmail.com
//    Niko Li, newlife20080214@gmail.com
//    Seunghoon Park, pclove1@gmail.com
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other GpuMaterials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors as is and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/


// rotate kernel
// Currently, CV_32FC1 is only supported.

// #if defined DOUBLE_SUPPORT
// #pragma OPENCL EXTENSION cl_khr_fp64:enable
// #define F double
// #else
// #define F float
// #endif

#define INTER_BITS 5
#define INTER_TAB_SIZE (1 << INTER_BITS)
#define INTER_SCALE 1.f/INTER_TAB_SIZE
#define AB_BITS max(10, (int)INTER_BITS)
#define AB_SCALE (1 << AB_BITS)
// #define INC(x,l) ((x+1) >= (l) ? (x):((x)+1))

__kernel void rotate_32FC1(__global float* dst, __global const float* src,
                           int dstStep_in_pixel, int srcStep_in_pixel,
                           int cols, int rows, __constant float * M)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    x = x < cols ? x: cols-1;
    y = y < rows ? y: rows-1;
    int dstidx = mad24(y, dstStep_in_pixel, x);

    int sx0 = rint(M[0]*x + M[1]*y + M[2]);
    int sy0 = rint(M[3]*x + M[4]*y + M[5]);
    int srcidx = mad24(sy0, srcStep_in_pixel, sx0);    

    // int round_delta = AB_SCALE/2;

    // int X0 = rint(M[0] * x * AB_SCALE);
    // int Y0 = rint(M[3] * x * AB_SCALE);
    // X0 += rint((M[1]*y + M[2]) * AB_SCALE) + round_delta;
    // Y0 += rint((M[4]*y + M[5]) * AB_SCALE) + round_delta;

    // short sx0 = (short)(X0 >> AB_BITS);
    // short sy0 = (short)(Y0 >> AB_BITS);    

    
    // dst[dstidx] = 0.0f; //src[srcidx];
    dst[dstidx] = (sx0>=0 && sx0<cols && sy0>=0 && sy0<rows) ? src[srcidx] : 0.0f;
    // dst[y*dstStep_in_pixel+x]=
    //         (sx0>=0 && sx0<cols && sy0>=0 && sy0<rows) ? src[sy0*srcStep_in_pixel+sx0] : 0;    
}

