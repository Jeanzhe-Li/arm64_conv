#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// 主卷积函数
void convolution(float *input_feature, const float *weights, const float *bias, float *output_feature, int output_channel, int input_channel,
                 int k_size, int output_wh, int input_wh)
{
    int row, col, output_filter, input_filter, kernel_row, kernel_col;
    
    for (row = 0; row < output_wh; row++) {
        for (col = 0; col < output_wh; col++) {
            for (output_filter = 0; output_filter < output_channel; output_filter++) {
                float temp = 0;
                for (input_filter = 0; input_filter < input_channel; input_filter++) {
                    for (kernel_row = 0; kernel_row < k_size; kernel_row++) {
                        for (kernel_col = 0; kernel_col < k_size; kernel_col++) {
                            temp = temp + (input_feature[input_filter * input_wh * input_wh + (row + kernel_row) * input_wh + (col + kernel_col)]
                                        * weights[output_filter * input_channel * k_size * k_size + input_filter * k_size * k_size +
                                                 kernel_row * k_size + kernel_col]);
                        }
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
    int kernel_size = 7;  // 3*3
    int input_size = 256;  // 640*640
    int output_size = 250; // 640 - 3 + 1 = 638 (no padding)
    
    printf("卷积参数:\n");
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
    printf("开始卷积计算...\n");
    clock_t start_time = clock();
    
    // 调用卷积函数
    convolution(input, weights_data, bias_data, output, output_channels, input_channels,
                kernel_size, output_size, input_size);
    
    // 结束计时
    clock_t end_time = clock();
    double cpu_time_used = ((double) (end_time - start_time)) / CLOCKS_PER_SEC;
    
    // 打印结果
    printf("\n卷积计算完成!\n");
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