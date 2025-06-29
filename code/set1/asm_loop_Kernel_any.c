#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

void convolution_any_kernel_asm(float *input_feature, const float *weights, const float *bias, float *output_feature, 
                               int output_channel, int input_channel, int k_size, int output_wh, int input_wh) {
    int row, col, output_filter, input_filter;
    int kernel_row, kernel_col;
    size_t ind, wind;
    size_t offset_input_wh = ((size_t)input_wh) << 2;  // input_wh * sizeof(float)
    size_t offset_k_size   = ((size_t)k_size)   << 2;  // k_size * sizeof(float)

    for (row = 0; row < output_wh; row++) {
        for (col = 0; col < output_wh; col++) {
            for (output_filter = 0; output_filter < output_channel; output_filter++) {
                float temp = 0;
                for (input_filter = 0; input_filter < input_channel; input_filter++) {
                    for (kernel_row = 0; kernel_row < k_size; kernel_row += 4) {
                        for (kernel_col = 0; kernel_col < k_size; kernel_col += 4) {
                            int rows_to_process = (kernel_row + 4 <= k_size) ? 4 : (k_size - kernel_row);
                            int cols_to_process = (kernel_col + 4 <= k_size) ? 4 : (k_size - kernel_col);

                            if (rows_to_process == 4 && cols_to_process == 4) {
                                ind = (size_t)input_filter * input_wh * input_wh + 
                                      (size_t)(row + kernel_row) * input_wh + (col + kernel_col);
                                wind = (size_t)output_filter * input_channel * k_size * k_size + 
                                       (size_t)input_filter * k_size * k_size + 
                                       (size_t)kernel_row * k_size + kernel_col;

                                asm volatile (
                                    "mov x0, %[input_feature]        \n\t"
                                    "mov x1, %[weights]              \n\t"

                                    "add x0, x0, %[ind], lsl #2      \n\t"
                                    "add x1, x1, %[wind], lsl #2     \n\t"

                                    "mov x2, x0                      \n\t"
                                    "mov x3, x1                      \n\t"

                                    "ld1 {v0.4s}, [x2], #16          \n\t"
                                    "ld1 {v1.4s}, [x3], #16          \n\t"
                                    "fmul v0.4s, v0.4s, v1.4s        \n\t"

                                    "add x0, x0, %[offset_inwh]      \n\t"
                                    "add x1, x1, %[offset_ksize]     \n\t"
                                    "mov x4, x0                      \n\t"
                                    "mov x5, x1                      \n\t"

                                    "ld1 {v2.4s}, [x4], #16          \n\t"
                                    "ld1 {v3.4s}, [x5], #16          \n\t"
                                    "fmul v2.4s, v2.4s, v3.4s        \n\t"

                                    "add x0, x0, %[offset_inwh]      \n\t"
                                    "add x1, x1, %[offset_ksize]     \n\t"
                                    "mov x6, x0                      \n\t"
                                    "mov x7, x1                      \n\t"

                                    "ld1 {v4.4s}, [x6], #16          \n\t"
                                    "ld1 {v5.4s}, [x7], #16          \n\t"
                                    "fmul v4.4s, v4.4s, v5.4s        \n\t"

                                    "add x0, x0, %[offset_inwh]      \n\t"
                                    "add x1, x1, %[offset_ksize]     \n\t"
                                    "mov x8, x0                      \n\t"
                                    "mov x9, x1                      \n\t"

                                    "ld1 {v6.4s}, [x8], #16          \n\t"
                                    "ld1 {v7.4s}, [x9], #16          \n\t"
                                    "fmul v6.4s, v6.4s, v7.4s        \n\t"

                                    "fadd v0.4s, v0.4s, v2.4s        \n\t"
                                    "fadd v4.4s, v4.4s, v6.4s        \n\t"
                                    "fadd v0.4s, v0.4s, v4.4s        \n\t"

                                    "mov v1.s[0], v0.s[1]            \n\t"
                                    "mov v2.s[0], v0.s[2]            \n\t"
                                    "mov v3.s[0], v0.s[3]            \n\t"

                                    "fadd s0, s0, s1                 \n\t"
                                    "fadd s2, s2, s3                 \n\t"
                                    "fadd s0, s2, s0                 \n\t"
                                    "ldr s1, %[temp]                 \n\t"
                                    "fadd s0, s0, s1                 \n\t"
                                    "str s0, %[temp]                 \n\t"

                                    : [temp] "+m" (temp)
                                    : [input_feature] "r" (input_feature),
                                      [weights] "r" (weights),
                                      [ind] "r" (ind),
                                      [wind] "r" (wind),
                                      [offset_inwh] "r" (offset_input_wh),
                                      [offset_ksize] "r" (offset_k_size)
                                    : "cc", "memory", "x0", "x1", "x2", "x3", "x4", "x5",
                                      "x6", "x7", "x8", "x9", "v0", "v1", "v2", "v3",
                                      "v4", "v5", "v6", "v7", "s0", "s1", "s2", "s3"
                                );
                            } else {
                                for (int kr = 0; kr < rows_to_process; kr++) {
                                    for (int kc = 0; kc < cols_to_process; kc++) {
                                        ind = (size_t)input_filter * input_wh * input_wh + 
                                              (size_t)(row + kernel_row + kr) * input_wh + 
                                              (col + kernel_col + kc);
                                        wind = (size_t)output_filter * input_channel * k_size * k_size + 
                                               (size_t)input_filter * k_size * k_size + 
                                               (size_t)(kernel_row + kr) * k_size + (kernel_col + kc);
                                        temp += input_feature[ind] * weights[wind];
                                    }
                                }
                            }
                        }
                    }
                }
                output_feature[output_filter * output_wh * output_wh + row * output_wh + col] = temp + bias[output_filter];
            }
        }
    }
}

int main() {
    int input_channels = 3;
    int output_channels = 16;
    int kernel_size = 5;  // Example for any size
    int input_size = 640;
    int output_size = input_size - kernel_size + 1;
    
    printf("任意尺寸卷积参数 (ASM):\n");
    printf("输入尺寸: %d x %d x %d\n", input_channels, input_size, input_size);
    printf("输出尺寸: %d x %d x %d\n", output_channels, output_size, output_size);
    printf("卷积核大小: %d x %d\n", kernel_size, kernel_size);
    printf("\n");
    
    float *input = (float *)malloc(input_channels * input_size * input_size * sizeof(float));
    float *weights_data = (float *)malloc(output_channels * input_channels * kernel_size * kernel_size * sizeof(float));
    float *bias_data = (float *)malloc(output_channels * sizeof(float));
    float *output = (float *)malloc(output_channels * output_size * output_size * sizeof(float));
    
    if (!input || !weights_data || !bias_data || !output) {
        printf("内存分配失败!\n");
        return -1;
    }
    
    printf("初始化数据...\n");
    srand(time(NULL));
    for (int i = 0; i < input_channels * input_size * input_size; i++) {
        input[i] = (float)(rand() % 10) / 10.0f;
    }
    
    for (int i = 0; i < output_channels * input_channels * kernel_size * kernel_size; i++) {
        weights_data[i] = (float)(rand() % 10) / 10.0f;
    }
    
    for (int i = 0; i < output_channels; i++) {
        bias_data[i] = 0.1f;
    }
    
    printf("开始任意尺寸卷积计算 (ASM)...\n");
    clock_t start_time = clock();
    
    convolution_any_kernel_asm(input, weights_data, bias_data, output, output_channels, input_channels,
                              kernel_size, output_size, input_size);
    
    clock_t end_time = clock();
    double cpu_time_used = ((double) (end_time - start_time)) / CLOCKS_PER_SEC;
    
    printf("\n任意尺寸卷积计算完成 (ASM)!\n");
    printf("计算时间: %.6f 秒\n", cpu_time_used);
    printf("\n输出样本值:\n");
    for (int i = 0; i < 5; i++) {
        printf("output[%d] = %.4f\n", i, output[i]);
    }
    
    long long total_operations = (long long)output_channels * output_size * output_size * 
                                input_channels * kernel_size * kernel_size * 2;
    double gflops = (total_operations / 1e9) / cpu_time_used;
    printf("\n性能统计 (ASM):\n");
    printf("总操作数: %lld\n", total_operations);
    printf("性能: %.2f GFLOPS\n", gflops);
    
    free(input);
    free(weights_data);
    free(bias_data);
    free(output);
    
    return 0;
}
