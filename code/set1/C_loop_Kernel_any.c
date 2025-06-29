#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void convolution_any_kernel(float *input_feature, const float *weights, const float *bias, float *output_feature, 
                          int output_channel, int input_channel, int k_size, int output_wh, int input_wh) {
    int row, col, output_filter, input_filter, input_index_tmp, weights_index_tmp;
    int kernel_row, kernel_col, ind, wind;
    
    for (row = 0; row < output_wh; row++) {
        for (col = 0; col < output_wh; col++) {
            for (output_filter = 0; output_filter < output_channel; output_filter++) {
                float temp = 0;
                for (input_filter = 0; input_filter < input_channel; input_filter++) {
                    for (kernel_row = 0; kernel_row < k_size; kernel_row+=4) {
                        for (kernel_col = 0; kernel_col < k_size; kernel_col+=4) {
                            ind = input_filter * input_wh * input_wh + (row + kernel_row) * input_wh + (col + kernel_col);
                            wind = output_filter * input_channel * k_size * k_size + input_filter * k_size * k_size + kernel_row * k_size + kernel_col;
                            
                            temp += input_feature[ind] * weights[wind];
                            temp += input_feature[ind+1] * weights[wind+1];
                            temp += input_feature[ind+2] * weights[wind+2];
                            temp += input_feature[ind+3] * weights[wind+3];
                            
                            temp += input_feature[ind+input_wh] * weights[wind+k_size];
                            temp += input_feature[ind+input_wh+1] * weights[wind+k_size+1];
                            temp += input_feature[ind+input_wh+2] * weights[wind+k_size+2];
                            temp += input_feature[ind+input_wh+3] * weights[wind+k_size+3];
                            
                            temp += input_feature[ind+input_wh*2] * weights[wind+k_size*2];
                            temp += input_feature[ind+input_wh*2+1] * weights[wind+k_size*2+1];
                            temp += input_feature[ind+input_wh*2+2] * weights[wind+k_size*2+2];
                            temp += input_feature[ind+input_wh*2+3] * weights[wind+k_size*2+3];
                            
                            temp += input_feature[ind+input_wh*3] * weights[wind+k_size*3];
                            temp += input_feature[ind+input_wh*3+1] * weights[wind+k_size*3+1];
                            temp += input_feature[ind+input_wh*3+2] * weights[wind+k_size*3+2];
                            temp += input_feature[ind+input_wh*3+3] * weights[wind+k_size*3+3];
                        }
                    }
                }
                output_feature[output_filter * output_wh * output_wh + row * output_wh + col] = temp + bias[output_filter];
            }
        }
    }
}

int main() {
    int input_channels = 1;
    int output_channels = 16;
    int kernel_size = 7;  // Example for any size
    int input_size = 640;
    int output_size = input_size - kernel_size + 1;
    
    printf("任意尺寸卷积参数:\n");
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
    for (int i = 0; i < input_channels * input_size * input_size; i++) {
        input[i] = (float)(rand() % 10) / 10.0f;
    }
    
    for (int i = 0; i < output_channels * input_channels * kernel_size * kernel_size; i++) {
        weights_data[i] = (float)(rand() % 10) / 10.0f;
    }
    
    for (int i = 0; i < output_channels; i++) {
        bias_data[i] = 0.1f;
    }
    
    printf("开始任意尺寸卷积计算...\n");
    clock_t start_time = clock();
    
    convolution_any_kernel(input, weights_data, bias_data, output, output_channels, input_channels,
                         kernel_size, output_size, input_size);
    
    clock_t end_time = clock();
    double cpu_time_used = ((double) (end_time - start_time)) / CLOCKS_PER_SEC;
    
    printf("\n任意尺寸卷积计算完成!\n");
    printf("计算时间: %.6f 秒\n", cpu_time_used);
    printf("\n输出样本值:\n");
    for (int i = 0; i < 5; i++) {
        printf("output[%d] = %.4f\n", i, output[i]);
    }
    
    long long total_operations = (long long)output_channels * output_size * output_size * 
                                input_channels * kernel_size * kernel_size * 2;
    double gflops = (total_operations / 1e9) / cpu_time_used;
    printf("\n性能统计:\n");
    printf("总操作数: %lld\n", total_operations);
    printf("性能: %.2f GFLOPS\n", gflops);
    
    free(input);
    free(weights_data);
    free(bias_data);
    free(output);
    
    return 0;
}
