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

// 矩阵乘法运算函数（4x4展开，使用内嵌汇编优化）
// C = A * B + C
// A: m x k, B: k x n, C: m x n
int asm_Sgemm_op16(float* a, float* b, float* c, int wh_1, int wh_2, int wh_3) {
    // wh_1 = output_channel (m)
    // wh_2 = input_channel * k_size * k_size (k)
    // wh_3 = output_wh * output_wh (n)
    
    int i, j, k;
    
    for (i = 0; i < (wh_1 & (~3)); i += 4) {
        for (j = 0; j < (wh_3 & (~3)); j += 4) {
            // 使用汇编实现4x4矩阵块的计算
            float *a_ptr = a + i * wh_2;
            float *b_ptr = b + j;
            float *c_ptr = c + i * wh_3 + j;
            
#ifdef __aarch64__
            __asm__ __volatile__(
                // 初始化16个累加寄存器为0 (4x4矩阵)
                "movi v0.4s, #0                  \n\t"    // c00,c01,c02,c03
                "movi v1.4s, #0                  \n\t"    // c10,c11,c12,c13
                "movi v2.4s, #0                  \n\t"    // c20,c21,c22,c23
                "movi v3.4s, #0                  \n\t"    // c30,c31,c32,c33
                
                "mov x0, %[a_ptr]                \n\t"    // 加载a的地址
                "mov x1, %[b_ptr]                \n\t"    // 加载b的地址
                "mov w2, %w[wh_2]                \n\t"    // k循环计数器
                "mov w3, %w[wh_3]                \n\t"    // wh_3步长
                "mov w4, %w[wh_2]                \n\t"    // wh_2步长
                "lsl w3, w3, #2                  \n\t"    // wh_3 * 4 (float大小)
                "lsl w4, w4, #2                  \n\t"    // wh_2 * 4 (float大小)
                
                "loop_k16_%=:                    \n\t"    // k循环起始标号
                
                // 加载A矩阵的4个元素 (一列)
                "ldr s4, [x0, #0]                \n\t"    // a[i+0][k]
                "ldr s5, [x0, x4]                \n\t"    // a[i+1][k]
                "add x5, x0, x4, lsl #1          \n\t"
                "ldr s6, [x5]                    \n\t"    // a[i+2][k]
                "ldr s7, [x5, x4]                \n\t"    // a[i+3][k]
                
                // 加载B矩阵的4个元素 (一行)
                "ld1 {v8.4s}, [x1]               \n\t"    // b[k][j:j+3]
                
                // 执行16个乘加操作
                "fmla v0.4s, v8.4s, v4.s[0]      \n\t"    // c0x += a[0][k] * b[k][x]
                "fmla v1.4s, v8.4s, v5.s[0]      \n\t"    // c1x += a[1][k] * b[k][x]
                "fmla v2.4s, v8.4s, v6.s[0]      \n\t"    // c2x += a[2][k] * b[k][x]
                "fmla v3.4s, v8.4s, v7.s[0]      \n\t"    // c3x += a[3][k] * b[k][x]
                
                "add x0, x0, #4                  \n\t"    // a指针移到下一列
                "add x1, x1, x3                  \n\t"    // b指针移到下一行
                
                "subs w2, w2, #1                 \n\t"    // k--
                "b.ne loop_k16_%=                \n\t"    // k循环结束判断
                
                // 存储结果到C矩阵
                "mov x0, %[c_ptr]                \n\t"
                "st1 {v0.4s}, [x0]               \n\t"    // 存储第0行
                "add x0, x0, x3                  \n\t"
                "st1 {v1.4s}, [x0]               \n\t"    // 存储第1行
                "add x0, x0, x3                  \n\t"
                "st1 {v2.4s}, [x0]               \n\t"    // 存储第2行
                "add x0, x0, x3                  \n\t"
                "st1 {v3.4s}, [x0]               \n\t"    // 存储第3行
                
                :
                : [a_ptr] "r"(a_ptr),
                  [b_ptr] "r"(b_ptr),
                  [c_ptr] "r"(c_ptr),
                  [wh_2] "r"(wh_2),
                  [wh_3] "r"(wh_3)
                : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8",
                  "x0", "x1", "x5", "w2", "w3", "w4"
            );
#else
            // 非ARM64架构使用C语言实现
            for (int m = 0; m < 4; m++) {
                for (int n = 0; n < 4; n++) {
                    float sum = 0;
                    for (k = 0; k < wh_2; k++) {
                        sum += a_ptr[m * wh_2 + k] * b_ptr[k * wh_3 + n];
                    }
                    c_ptr[m * wh_3 + n] = sum;
                }
            }
#endif
        }
        
        // 处理剩余的列（当wh_3不是4的倍数时）
        for (; j < wh_3; j++) {
            for (int m = 0; m < 4 && (i + m) < wh_1; m++) {
                c[(i + m) * wh_3 + j] = 0;
                for (k = 0; k < wh_2; k++) {
                    c[(i + m) * wh_3 + j] += a[(i + m) * wh_2 + k] * b[k * wh_3 + j];
                }
            }
        }
    }
    
    // 处理剩余的行（当wh_1不是4的倍数时）
    for (; i < wh_1; i++) {
        for (j = 0; j < wh_3; j++) {
            c[i * wh_3 + j] = 0;
            for (k = 0; k < wh_2; k++) {
                c[i * wh_3 + j] += a[i * wh_2 + k] * b[k * wh_3 + j];
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
    int kernel_size = 3;  // 固定为3*3
    int input_size = 256;  // 640*640
    int output_size = 254; // 640 - 3 + 1 = 638 (no padding)
    
    printf("卷积参数:\n");
    printf("输入尺寸: %d x %d x %d\n", input_channels, input_size, input_size);
    printf("输出尺寸: %d x %d x %d\n", output_channels, output_size, output_size);
    printf("卷积核大小: %d x %d\n", kernel_size, kernel_size);
    printf("优化方式: 4x4 矩阵乘法展开（内嵌汇编）\n");
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
    printf("开始卷积计算（Im2col + SGEMM 4x4展开 汇编优化）...\n");
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
    
    // 3. 初始化输出矩阵为0
    memset(output, 0, output_channels * output_size * output_size * sizeof(float));
    
    // 4. 执行矩阵乘法（4x4展开，汇编优化）
    asm_Sgemm_op16(weights_data, im2col_feature, output, 
                   output_channels, 
                   input_channels * kernel_size * kernel_size, 
                   output_size * output_size);
    
    // 5. 添加偏置
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
