#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// 使用内联汇编优化的卷积函数 - 3x3卷积核
void convolution_asm_optimized(float *input_feature, const float *weights, const float *bias, float *output_feature, 
                               int output_channel, int input_channel, int k_size, int output_wh, int input_wh)
{
    for (int row = 0; row < output_wh; row++) {
        for (int col = 0; col < output_wh; col++) {
            for (int output_filter = 0; output_filter < output_channel; output_filter++) {
                float temp = 0.0f;
                for (int input_filter = 0; input_filter < input_channel; input_filter++) {
                    if (k_size == 3) {
                        int input_base = input_filter * input_wh * input_wh + row * input_wh + col;
                        int weight_base = output_filter * input_channel * 9 + input_filter * 9;

                        float *input_ptr = input_feature;
                        const float *weight_ptr = weights;
                        float *temp_ptr = &temp;

                        __asm__ __volatile__(
                            // 初始化累加寄存器 s0 为 0
                            "fmov s0, wzr\n\t"

                            // r0: input_base * 4
                            "mov w0, %w[input_base]\n\t"
                            "lsl w0, w0, #2\n\t"
                            "add x1, %x[input_ptr], x0\n\t" // x1 = &input_feature[input_base]

                            // r2: weight_base * 4
                            "mov w2, %w[weight_base]\n\t"
                            "lsl w2, w2, #2\n\t"
                            "add x3, %x[weight_ptr], x2\n\t" // x3 = &weights[weight_base]

                            // 每行做3次
                            // Row 1
                            "ldr s1, [x1]       \n\t"
                            "ldr s2, [x3]       \n\t"
                            "fmul s1, s1, s2    \n\t"
                            "fadd s0, s0, s1    \n\t"

                            "ldr s1, [x1, #4]   \n\t"
                            "ldr s2, [x3, #4]   \n\t"
                            "fmul s1, s1, s2    \n\t"
                            "fadd s0, s0, s1    \n\t"

                            "ldr s1, [x1, #8]   \n\t"
                            "ldr s2, [x3, #8]   \n\t"
                            "fmul s1, s1, s2    \n\t"
                            "fadd s0, s0, s1    \n\t"

                            // Row 2: input_wh * 4 = byte stride
                            "mov x4, %x[input_wh]\n\t"
                            "lsl x4, x4, #2\n\t"
                            "add x1, x1, x4\n\t"
                            "add x3, x3, #12\n\t"

                            "ldr s1, [x1]       \n\t"
                            "ldr s2, [x3]       \n\t"
                            "fmul s1, s1, s2    \n\t"
                            "fadd s0, s0, s1    \n\t"

                            "ldr s1, [x1, #4]   \n\t"
                            "ldr s2, [x3, #4]   \n\t"
                            "fmul s1, s1, s2    \n\t"
                            "fadd s0, s0, s1    \n\t"

                            "ldr s1, [x1, #8]   \n\t"
                            "ldr s2, [x3, #8]   \n\t"
                            "fmul s1, s1, s2    \n\t"
                            "fadd s0, s0, s1    \n\t"

                            // Row 3
                            "add x1, x1, x4\n\t"
                            "add x3, x3, #12\n\t"

                            "ldr s1, [x1]       \n\t"
                            "ldr s2, [x3]       \n\t"
                            "fmul s1, s1, s2    \n\t"
                            "fadd s0, s0, s1    \n\t"

                            "ldr s1, [x1, #4]   \n\t"
                            "ldr s2, [x3, #4]   \n\t"
                            "fmul s1, s1, s2    \n\t"
                            "fadd s0, s0, s1    \n\t"

                            "ldr s1, [x1, #8]   \n\t"
                            "ldr s2, [x3, #8]   \n\t"
                            "fmul s1, s1, s2    \n\t"
                            "fadd s0, s0, s1    \n\t"

                            "str s0, [%x[temp_ptr]]\n\t"

                            :
                            : [input_base] "r" (input_base),
                              [weight_base] "r" (weight_base),
                              [input_ptr] "r" (input_ptr),
                              [weight_ptr] "r" (weight_ptr),
                              [input_wh] "r" (input_wh),
                              [temp_ptr] "r" (temp_ptr)
                            : "x0", "x1", "x2", "x3", "x4", 
                              "s0", "s1", "s2", "memory"
                        );
                    }
                }
                output_feature[output_filter * output_wh * output_wh + row * output_wh + col] = temp + bias[output_filter];
            }
        }
    }
}

// 主函数用于测试
int main()
{
    // 固定参数
    int input_channels = 1;
    int output_channels = 16;
    int kernel_size = 3;  // 固定为3*3
    int input_size = 256;  // 640*640
    int output_size = 254; // 640 - 3 + 1 = 638 (no padding)
    
    printf("汇编优化版卷积参数:\n");
    printf("输入尺寸: %d x %d x %d\n", input_channels, input_size, input_size);
    printf("输出尺寸: %d x %d x %d\n", output_channels, output_size, output_size);
    printf("卷积核大小: %d x %d\n", kernel_size, kernel_size);
    printf("\n");
    
    // 分配内存
    float *input = (float *)malloc(input_channels * input_size * input_size * sizeof(float));
    float *weights_data = (float *)malloc(output_channels * input_channels * kernel_size * kernel_size * sizeof(float));
    float *bias_data = (float *)malloc(output_channels * sizeof(float));
    float *output = (float *)malloc(output_channels * output_size * output_size * sizeof(float));
    
    if (!input || !weights_data || !bias_data || !output) {
        printf("内存分配失败!\n");
        return -1;
    }
    
    // 初始化输入数据（示例）
    printf("初始化数据...\n");
    for (int i = 0; i < input_channels * input_size * input_size; i++) {
        input[i] = (float)(rand() % 10) / 10.0f;
    }
    
    // 初始化权重（示例）
    for (int i = 0; i < output_channels * input_channels * kernel_size * kernel_size; i++) {
        weights_data[i] = (float)(rand() % 10) / 10.0f;
    }
    
    // 初始化偏置（示例）
    for (int i = 0; i < output_channels; i++) {
        bias_data[i] = 0.1f;
    }
    
    // 开始计时
    printf("开始汇编优化版卷积计算...\n");
    clock_t start_time = clock();
    
    // 调用汇编优化的卷积函数
    convolution_asm_optimized(input, weights_data, bias_data, output, output_channels, input_channels,
                         kernel_size, output_size, input_size);
    
    // 结束计时
    clock_t end_time = clock();
    double cpu_time_used = ((double) (end_time - start_time)) / CLOCKS_PER_SEC;
    
    // 打印结果
    printf("\n汇编优化版卷积计算完成!\n");
    printf("计算时间: %.6f 秒\n", cpu_time_used);
    printf("\n输出样本值:\n");
    for (int i = 0; i < 5; i++) {
        printf("output[%d] = %.4f\n", i, output[i]);
    }
    
    // 计算并显示一些统计信息
    long long total_operations = (long long)output_channels * output_size * output_size * 
                                input_channels * kernel_size * kernel_size * 2; // 乘法和加法
    double gflops = (total_operations / 1e9) / cpu_time_used;
    printf("\n性能统计:\n");
    printf("总操作数: %lld\n", total_operations);
    printf("性能: %.2f GFLOPS\n", gflops);
    
    // 释放内存
    free(input);
    free(weights_data);
    free(bias_data);
    free(output);
    
    return 0;
}
