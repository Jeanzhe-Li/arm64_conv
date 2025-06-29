#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdbool.h>
#include <math.h>

#ifdef __aarch64__
#include <arm_neon.h>
#endif

// 内嵌汇编优化版本（针对3x3卷积核）
void dilated_convolution_2d_asm(
    float* input, int input_h, int input_w,
    float* kernel, int kernel_h, int kernel_w,
    float* output, int output_h, int output_w,
    int dilation, int stride, int padding
) {
    memset(output, 0, output_h * output_w * sizeof(float));
    
#ifdef __aarch64__
    // 只对3x3, stride=1的情况使用汇编优化
    if (kernel_h == 3 && kernel_w == 3 && stride == 1) {
        // 预加载卷积核到寄存器
        float k[9];
        for (int i = 0; i < 9; i++) {
            k[i] = kernel[i];
        }
        
        for (int oh = 0; oh < output_h; oh++) {
            for (int ow = 0; ow < output_w; ow++) {
                float sum = 0.0f;
                int ih_start = oh * stride - padding;
                int iw_start = ow * stride - padding;
                
                // 使用内嵌汇编优化3x3卷积计算
                __asm__ __volatile__(
                    // 初始化累加器
                    "fmov s0, wzr\n\t"                    // sum = 0
                    
                    // 加载卷积核到寄存器 s1-s9
                    "ldr s1, [%[kernel], #0]\n\t"         // k[0]
                    "ldr s2, [%[kernel], #4]\n\t"         // k[1]
                    "ldr s3, [%[kernel], #8]\n\t"         // k[2]
                    "ldr s4, [%[kernel], #12]\n\t"        // k[3]
                    "ldr s5, [%[kernel], #16]\n\t"        // k[4]
                    "ldr s6, [%[kernel], #20]\n\t"        // k[5]
                    "ldr s7, [%[kernel], #24]\n\t"        // k[6]
                    "ldr s8, [%[kernel], #28]\n\t"        // k[7]
                    "ldr s9, [%[kernel], #32]\n\t"        // k[8]
                    
                    // 计算每个卷积核位置
                    // 位置(0,0)
                    "mov w10, %w[ih_start]\n\t"
                    "mov w11, %w[iw_start]\n\t"
                    
                    // 检查边界并计算
                    "cmp w10, #0\n\t"
                    "b.lt 1f\n\t"
                    "cmp w10, %w[input_h]\n\t"
                    "b.ge 1f\n\t"
                    "cmp w11, #0\n\t"
                    "b.lt 1f\n\t"
                    "cmp w11, %w[input_w]\n\t"
                    "b.ge 1f\n\t"
                    
                    "mul w12, w10, %w[input_w]\n\t"
                    "add w12, w12, w11\n\t"
                    "lsl w12, w12, #2\n\t"                // 乘以4（float大小）
                    "ldr s10, [%[input], w12, sxtw]\n\t"
                    "fmadd s0, s10, s1, s0\n\t"
                    
                    "1:\n\t"
                    
                    // 位置(0,1) - 考虑dilation
                    "add w11, %w[iw_start], %w[dilation]\n\t"
                    "cmp w10, #0\n\t"
                    "b.lt 2f\n\t"
                    "cmp w10, %w[input_h]\n\t"
                    "b.ge 2f\n\t"
                    "cmp w11, #0\n\t"
                    "b.lt 2f\n\t"
                    "cmp w11, %w[input_w]\n\t"
                    "b.ge 2f\n\t"
                    
                    "mul w12, w10, %w[input_w]\n\t"
                    "add w12, w12, w11\n\t"
                    "lsl w12, w12, #2\n\t"
                    "ldr s10, [%[input], w12, sxtw]\n\t"
                    "fmadd s0, s10, s2, s0\n\t"
                    
                    "2:\n\t"
                    
                    // 位置(0,2) - 考虑dilation
                    "add w11, %w[iw_start], %w[dilation], lsl #1\n\t"
                    "mov w10, %w[ih_start]\n\t"
                    "cmp w10, #0\n\t"
                    "b.lt 3f\n\t"
                    "cmp w10, %w[input_h]\n\t"
                    "b.ge 3f\n\t"
                    "cmp w11, #0\n\t"
                    "b.lt 3f\n\t"
                    "cmp w11, %w[input_w]\n\t"
                    "b.ge 3f\n\t"
                    
                    "mul w12, w10, %w[input_w]\n\t"
                    "add w12, w12, w11\n\t"
                    "lsl w12, w12, #2\n\t"
                    "ldr s10, [%[input], w12, sxtw]\n\t"
                    "fmadd s0, s10, s3, s0\n\t"
                    
                    "3:\n\t"
                    
                    // 位置(1,0) - 考虑dilation
                    "add w10, %w[ih_start], %w[dilation]\n\t"
                    "mov w11, %w[iw_start]\n\t"
                    "cmp w10, #0\n\t"
                    "b.lt 4f\n\t"
                    "cmp w10, %w[input_h]\n\t"
                    "b.ge 4f\n\t"
                    "cmp w11, #0\n\t"
                    "b.lt 4f\n\t"
                    "cmp w11, %w[input_w]\n\t"
                    "b.ge 4f\n\t"
                    
                    "mul w12, w10, %w[input_w]\n\t"
                    "add w12, w12, w11\n\t"
                    "lsl w12, w12, #2\n\t"
                    "ldr s10, [%[input], w12, sxtw]\n\t"
                    "fmadd s0, s10, s4, s0\n\t"
                    
                    "4:\n\t"
                    
                    // 位置(1,1) - 考虑dilation
                    "add w10, %w[ih_start], %w[dilation]\n\t"
                    "add w11, %w[iw_start], %w[dilation]\n\t"
                    "cmp w10, #0\n\t"
                    "b.lt 5f\n\t"
                    "cmp w10, %w[input_h]\n\t"
                    "b.ge 5f\n\t"
                    "cmp w11, #0\n\t"
                    "b.lt 5f\n\t"
                    "cmp w11, %w[input_w]\n\t"
                    "b.ge 5f\n\t"
                    
                    "mul w12, w10, %w[input_w]\n\t"
                    "add w12, w12, w11\n\t"
                    "lsl w12, w12, #2\n\t"
                    "ldr s10, [%[input], w12, sxtw]\n\t"
                    "fmadd s0, s10, s5, s0\n\t"
                    
                    "5:\n\t"
                    
                    // 位置(1,2) - 考虑dilation
                    "add w10, %w[ih_start], %w[dilation]\n\t"
                    "add w11, %w[iw_start], %w[dilation], lsl #1\n\t"
                    "cmp w10, #0\n\t"
                    "b.lt 6f\n\t"
                    "cmp w10, %w[input_h]\n\t"
                    "b.ge 6f\n\t"
                    "cmp w11, #0\n\t"
                    "b.lt 6f\n\t"
                    "cmp w11, %w[input_w]\n\t"
                    "b.ge 6f\n\t"
                    
                    "mul w12, w10, %w[input_w]\n\t"
                    "add w12, w12, w11\n\t"
                    "lsl w12, w12, #2\n\t"
                    "ldr s10, [%[input], w12, sxtw]\n\t"
                    "fmadd s0, s10, s6, s0\n\t"
                    
                    "6:\n\t"
                    
                    // 位置(2,0) - 考虑dilation
                    "add w10, %w[ih_start], %w[dilation], lsl #1\n\t"
                    "mov w11, %w[iw_start]\n\t"
                    "cmp w10, #0\n\t"
                    "b.lt 7f\n\t"
                    "cmp w10, %w[input_h]\n\t"
                    "b.ge 7f\n\t"
                    "cmp w11, #0\n\t"
                    "b.lt 7f\n\t"
                    "cmp w11, %w[input_w]\n\t"
                    "b.ge 7f\n\t"
                    
                    "mul w12, w10, %w[input_w]\n\t"
                    "add w12, w12, w11\n\t"
                    "lsl w12, w12, #2\n\t"
                    "ldr s10, [%[input], w12, sxtw]\n\t"
                    "fmadd s0, s10, s7, s0\n\t"
                    
                    "7:\n\t"
                    
                    // 位置(2,1) - 考虑dilation
                    "add w10, %w[ih_start], %w[dilation], lsl #1\n\t"
                    "add w11, %w[iw_start], %w[dilation]\n\t"
                    "cmp w10, #0\n\t"
                    "b.lt 8f\n\t"
                    "cmp w10, %w[input_h]\n\t"
                    "b.ge 8f\n\t"
                    "cmp w11, #0\n\t"
                    "b.lt 8f\n\t"
                    "cmp w11, %w[input_w]\n\t"
                    "b.ge 8f\n\t"
                    
                    "mul w12, w10, %w[input_w]\n\t"
                    "add w12, w12, w11\n\t"
                    "lsl w12, w12, #2\n\t"
                    "ldr s10, [%[input], w12, sxtw]\n\t"
                    "fmadd s0, s10, s8, s0\n\t"
                    
                    "8:\n\t"
                    
                    // 位置(2,2) - 考虑dilation
                    "add w10, %w[ih_start], %w[dilation], lsl #1\n\t"
                    "add w11, %w[iw_start], %w[dilation], lsl #1\n\t"
                    "cmp w10, #0\n\t"
                    "b.lt 9f\n\t"
                    "cmp w10, %w[input_h]\n\t"
                    "b.ge 9f\n\t"
                    "cmp w11, #0\n\t"
                    "b.lt 9f\n\t"
                    "cmp w11, %w[input_w]\n\t"
                    "b.ge 9f\n\t"
                    
                    "mul w12, w10, %w[input_w]\n\t"
                    "add w12, w12, w11\n\t"
                    "lsl w12, w12, #2\n\t"
                    "ldr s10, [%[input], w12, sxtw]\n\t"
                    "fmadd s0, s10, s9, s0\n\t"
                    
                    "9:\n\t"
                    
                    "str s0, %[sum]\n\t"
                    
                    : [sum] "=m" (sum)
                    : [ih_start] "r" (ih_start),
                      [iw_start] "r" (iw_start),
                      [input_h] "r" (input_h),
                      [input_w] "r" (input_w),
                      [dilation] "r" (dilation),
                      [input] "r" (input),
                      [kernel] "r" (k)
                    : "w10", "w11", "w12", "s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "memory"
                );
                
                output[oh * output_w + ow] = sum;
            }
        }
    }
#endif
}

// 计算输出尺寸的辅助函数
int calculate_output_size(int input_size, int kernel_size, int dilation, int stride, int padding) {
    int effective_kernel_size = (kernel_size - 1) * dilation + 1;
    return (input_size + 2 * padding - effective_kernel_size) / stride + 1;
}

// 打印矩阵的辅助函数
void print_matrix(float* matrix, int rows, int cols, const char* name) {
    printf("%s:\n", name);
    int max_print = (rows > 8) ? 8 : rows;
    int max_print_cols = (cols > 8) ? 8 : cols;
    
    for (int i = 0; i < max_print; i++) {
        for (int j = 0; j < max_print_cols; j++) {
            printf("%6.2f ", matrix[i * cols + j]);
        }
        if (cols > 8) printf("...");
        printf("\n");
    }
    if (rows > 8) printf("...\n");
    printf("\n");
}

int main() 
{
    printf("=== ARM64 汇编优化空洞卷积测试 ===\n\n");
    
    srand(time(NULL));
    
    // 输入参数设置
    int input_h = 256, input_w = 256;
    int kernel_h = 3, kernel_w = 3;
    int dilation = 2;
    int stride = 1;
    int padding = 0;
    
    // 计算输出尺寸
    int output_h = calculate_output_size(input_h, kernel_h, dilation, stride, padding);
    int output_w = calculate_output_size(input_w, kernel_w, dilation, stride, padding);
    
    printf("配置信息:\n");
    printf("输入尺寸: %dx%d\n", input_h, input_w);
    printf("卷积核尺寸: %dx%d\n", kernel_h, kernel_w);
    printf("空洞率: %d, 步长: %d, 填充: %d\n", dilation, stride, padding);
    printf("输出尺寸: %dx%d\n", output_h, output_w);
    
#ifdef __aarch64__
    printf("架构: ARM64 \n\n");
#else
    printf("架构: 非ARM64 (使用标准实现)\n\n");
#endif
    
    // 分配内存
    float* input = (float*)malloc(input_h * input_w * sizeof(float));
    float* kernel = (float*)malloc(kernel_h * kernel_w * sizeof(float));
    float* output_std = (float*)malloc(output_h * output_w * sizeof(float));
    float* output_asm = (float*)malloc(output_h * output_w * sizeof(float));
    float* output_neon = (float*)malloc(output_h * output_w * sizeof(float));
    float* output_im2col = (float*)malloc(output_h * output_w * sizeof(float));
    
    // 初始化数据
    for (int i = 0; i < input_h * input_w; i++) {
        input[i] = (float)(rand() % 256) / 255.0f;
    }
    
    for (int i = 0; i < kernel_h * kernel_w; i++) {
        kernel[i] = ((float)(rand() % 200) - 100) / 100.0f;
    }
    
    printf("数据初始化完成\n");
    print_matrix(kernel, kernel_h, kernel_w, "卷积核");
   
    clock_t start = clock();
    clock_t end = clock();
    
    // 测试汇编优化
    printf("测试汇编优化实现...\n");
    start = clock();
    dilated_convolution_2d_asm(input, input_h, input_w, kernel, kernel_h, kernel_w,
                              output_asm, output_h, output_w, dilation, stride, padding);
    end = clock();
    double time_asm = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("汇编优化时间: %.4f 秒\n", time_asm);
  
    long total_ops = (long)output_h * output_w * kernel_h * kernel_w * 2;  // 乘法和加法
    printf("\n总运算量: %ld FLOPS\n", total_ops);
    printf("汇编优化性能: %.2f GFLOPS\n", (double)total_ops / (time_asm * 1e9));
    
    // 释放内存
    free(input);
    free(kernel);
    free(output_std);
    free(output_asm);
    free(output_neon);
    free(output_im2col);
    
    return 0;
}