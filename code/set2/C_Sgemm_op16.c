#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// Im2col函数：将输入特征图转换为矩阵形式
// 输入特征图转换为im2col型矩阵，返回结果矩阵
float* src_im2col(const float *input_feature, int input_channel, int input_wh, int k_size, int output_wh) {
    float *im2col_feature;
    int index = 0;
    
    // 分配im2col矩阵内存
    // 矩阵大小：(input_channel * k_size * k_size) x (output_wh * output_wh)
    im2col_feature = (float*)malloc(input_channel * k_size * k_size * output_wh * output_wh * sizeof(float));
    
    if (!im2col_feature) {
        printf("Im2col内存分配失败!\n");
        return NULL;
    }
    
    // 将输入特征图转换为im2col格式
    for (int input_filter = 0; input_filter < input_channel; input_filter++) {
        for (int row = 0; row < k_size; row++) {
            for (int col = 0; col < k_size; col++) {
                // 得到im2col中每一行的值
                for (int i = 0; i < output_wh; i++) {
                    for (int j = 0; j < output_wh; j++) {
                        im2col_feature[index++] = input_feature[input_filter * input_wh * input_wh + 
                                                               (i + row) * input_wh + (j + col)];
                    }
                }
            }
        }
    }
    
    return im2col_feature;
}

// 矩阵乘法运算函数（4x4展开优化）
// C = A * B + C
// A: m x k, B: k x n, C: m x n
int C_Sgemm_op16(float* im2col_weight, float* im2col_feature, float* result_Sgemm, int wh_1, int wh_2, int wh_3) {
    // wh_1 = output_channel (m)
    // wh_2 = input_channel * k_size * k_size (k)
    // wh_3 = output_wh * output_wh (n)
    
    int m, n, k;
    
    // 矩阵乘法展开4x4
    for (m = 0; m < (wh_1 & (~3)); m += 4) {
        for (n = 0; n < (wh_3 & (~3)); n += 4) {
            // 初始化16个累加器（4x4）
            float c00 = 0, c01 = 0, c02 = 0, c03 = 0;
            float c10 = 0, c11 = 0, c12 = 0, c13 = 0;
            float c20 = 0, c21 = 0, c22 = 0, c23 = 0;
            float c30 = 0, c31 = 0, c32 = 0, c33 = 0;
            
            // 执行4x4的矩阵乘法
            for (k = 0; k < wh_2; k++) {
                // 加载A矩阵的4个元素
                float a0 = im2col_weight[(m + 0) * wh_2 + k];
                float a1 = im2col_weight[(m + 1) * wh_2 + k];
                float a2 = im2col_weight[(m + 2) * wh_2 + k];
                float a3 = im2col_weight[(m + 3) * wh_2 + k];
                
                // 加载B矩阵的4个元素
                float b0 = im2col_feature[k * wh_3 + n + 0];
                float b1 = im2col_feature[k * wh_3 + n + 1];
                float b2 = im2col_feature[k * wh_3 + n + 2];
                float b3 = im2col_feature[k * wh_3 + n + 3];
                
                // 执行16个乘加操作
                c00 += a0 * b0;
                c01 += a0 * b1;
                c02 += a0 * b2;
                c03 += a0 * b3;
                
                c10 += a1 * b0;
                c11 += a1 * b1;
                c12 += a1 * b2;
                c13 += a1 * b3;
                
                c20 += a2 * b0;
                c21 += a2 * b1;
                c22 += a2 * b2;
                c23 += a2 * b3;
                
                c30 += a3 * b0;
                c31 += a3 * b1;
                c32 += a3 * b2;
                c33 += a3 * b3;
            }
            
            // 存储结果
            result_Sgemm[(m + 0) * wh_3 + n + 0] = c00;
            result_Sgemm[(m + 0) * wh_3 + n + 1] = c01;
            result_Sgemm[(m + 0) * wh_3 + n + 2] = c02;
            result_Sgemm[(m + 0) * wh_3 + n + 3] = c03;
            
            result_Sgemm[(m + 1) * wh_3 + n + 0] = c10;
            result_Sgemm[(m + 1) * wh_3 + n + 1] = c11;
            result_Sgemm[(m + 1) * wh_3 + n + 2] = c12;
            result_Sgemm[(m + 1) * wh_3 + n + 3] = c13;
            
            result_Sgemm[(m + 2) * wh_3 + n + 0] = c20;
            result_Sgemm[(m + 2) * wh_3 + n + 1] = c21;
            result_Sgemm[(m + 2) * wh_3 + n + 2] = c22;
            result_Sgemm[(m + 2) * wh_3 + n + 3] = c23;
            
            result_Sgemm[(m + 3) * wh_3 + n + 0] = c30;
            result_Sgemm[(m + 3) * wh_3 + n + 1] = c31;
            result_Sgemm[(m + 3) * wh_3 + n + 2] = c32;
            result_Sgemm[(m + 3) * wh_3 + n + 3] = c33;
        }
        
        // 处理剩余的列（当wh_3不是4的倍数时）
        for (; n < wh_3; n++) {
            for (int i = 0; i < 4 && (m + i) < wh_1; i++) {
                result_Sgemm[(m + i) * wh_3 + n] = 0;
                for (k = 0; k < wh_2; k++) {
                    result_Sgemm[(m + i) * wh_3 + n] += 
                        im2col_weight[(m + i) * wh_2 + k] * im2col_feature[k * wh_3 + n];
                }
            }
        }
    }
    
    // 处理剩余的行（当wh_1不是4的倍数时）
    for (; m < wh_1; m++) {
        for (n = 0; n < wh_3; n++) {
            result_Sgemm[m * wh_3 + n] = 0;
            for (k = 0; k < wh_2; k++) {
                result_Sgemm[m * wh_3 + n] += 
                    im2col_weight[m * wh_2 + k] * im2col_feature[k * wh_3 + n];
            }
        }
    }
    
    return 0;
}

// 主函数用于测试
int main() {
    // 固定参数（参考C_loop_Origin.c）
    int input_channels = 1;
    int output_channels = 16;
    int kernel_size = 3;  // 3*3
    int input_size = 256;  // 640*640
    int output_size = 254; // 640 - 3 + 1 = 638 (no padding)
    
    printf("卷积参数:\n");
    printf("输入尺寸: %d x %d x %d\n", input_channels, input_size, input_size);
    printf("输出尺寸: %d x %d x %d\n", output_channels, output_size, output_size);
    printf("卷积核大小: %d x %d\n", kernel_size, kernel_size);
    printf("优化方式: 4x4 矩阵乘法展开\n");
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
    printf("开始卷积计算（Im2col + SGEMM 4x4展开）...\n");
    clock_t start_time = clock();
    
    // 1. Im2col转换
    float *im2col_feature = src_im2col(input, input_channels, input_size, kernel_size, output_size);
    if (!im2col_feature) {
        printf("Im2col转换失败!\n");
        free(input);
        free(weights_data);
        free(bias_data);
        free(output);
        return -1;
    }
    
    // 2. 准备权重矩阵（已经是正确的格式）
    // weights_data已经是 output_channels x (input_channels * kernel_size * kernel_size) 的格式
    
    // 3. 执行矩阵乘法（4x4展开）
    C_Sgemm_op16(weights_data, im2col_feature, output, 
                 output_channels, 
                 input_channels * kernel_size * kernel_size, 
                 output_size * output_size);
    
    // 4. 添加偏置
    for (int oc = 0; oc < output_channels; oc++) {
        for (int i = 0; i < output_size * output_size; i++) {
            output[oc * output_size * output_size + i] += bias_data[oc];
        }
    }
    
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
    free(im2col_feature);
    
    return 0;
}
