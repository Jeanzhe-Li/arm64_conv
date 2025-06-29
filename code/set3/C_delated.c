#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// 空洞卷积函数
// input: 输入矩阵 (input_h x input_w)
// kernel: 卷积核 (kernel_h x kernel_w)
// output: 输出矩阵 (output_h x output_w)
// dilation: 空洞率
// stride: 步长
// padding: 填充大小
void dilated_convolution_2d(
    float* input, int input_h, int input_w,
    float* kernel, int kernel_h, int kernel_w,
    float* output, int output_h, int output_w,
    int dilation, int stride, int padding
) {
    // 初始化输出矩阵为0
    memset(output, 0, output_h * output_w * sizeof(float));
    
    // 计算有效的卷积核大小（考虑空洞）
    int effective_kernel_h = (kernel_h - 1) * dilation + 1;
    int effective_kernel_w = (kernel_w - 1) * dilation + 1;
    
    // 遍历输出矩阵的每个位置
    for (int oh = 0; oh < output_h; oh++) {
        for (int ow = 0; ow < output_w; ow++) {
            float sum = 0.0f;
            
            // 计算在输入矩阵中的起始位置
            int ih_start = oh * stride - padding;
            int iw_start = ow * stride - padding;
            
            // 遍历卷积核
            for (int kh = 0; kh < kernel_h; kh++) {
                for (int kw = 0; kw < kernel_w; kw++) {
                    // 计算在输入矩阵中的实际位置（考虑空洞）
                    int ih = ih_start + kh * dilation;
                    int iw = iw_start + kw * dilation;
                    
                    // 检查边界条件
                    if (ih >= 0 && ih < input_h && iw >= 0 && iw < input_w) {
                        sum += input[ih * input_w + iw] * kernel[kh * kernel_w + kw];
                    }
                }
            }
            
            output[oh * output_w + ow] = sum;
        }
    }
}

// 计算输出尺寸的辅助函数
int calculate_output_size(int input_size, int kernel_size, int dilation, int stride, int padding) {
    int effective_kernel_size = (kernel_size - 1) * dilation + 1;
    return (input_size + 2 * padding - effective_kernel_size) / stride + 1;
}

// 打印矩阵的辅助函数
void print_matrix(float* matrix, int rows, int cols, const char* name) {
    printf("%s:\n", name);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%6.2f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

// 示例和测试
int main() {
    printf("=== 空洞卷积示例 ===\n\n");
    
    // 设置随机种子
    srand(time(NULL));
    
    // 输入参数设置
    int input_h = 256, input_w = 256;
    int kernel_h = 3, kernel_w = 3;
    int dilation = 2;  // 空洞率为2
    int stride = 1;
    int padding = 0;
    
    // 计算输出尺寸
    int output_h = calculate_output_size(input_h, kernel_h, dilation, stride, padding);
    int output_w = calculate_output_size(input_w, kernel_w, dilation, stride, padding);
    
    printf("输入尺寸: %dx%d\n", input_h, input_w);
    printf("卷积核尺寸: %dx%d\n", kernel_h, kernel_w);
    printf("空洞率: %d\n", dilation);
    printf("步长: %d\n", stride);
    printf("填充: %d\n", padding);
    printf("输出尺寸: %dx%d\n\n", output_h, output_w);
    
    // 分配内存
    float* input = (float*)malloc(input_h * input_w * sizeof(float));
    float* kernel = (float*)malloc(kernel_h * kernel_w * sizeof(float));
    float* output = (float*)malloc(output_h * output_w * sizeof(float));
    
    // 初始化输入矩阵
    for (int i = 0; i < input_h * input_w; i++) {
        input[i] = (float)(rand() % 256) / 255.0f;  // 随机浮点数 [0,1]
    }
    
    // 初始化卷积核（随机初始化）
    for (int i = 0; i < kernel_h * kernel_w; i++) {
        kernel[i] = ((float)(rand() % 200) - 100) / 100.0f;  // 随机浮点数 [-1,1]
    }
    
    // 只打印卷积核（输入矩阵太大不打印）
    printf("随机初始化输入矩阵完成\n");
    print_matrix(kernel, kernel_h, kernel_w, "卷积核");
    
    // 开始计时
    clock_t start_time = clock();
    
    // 执行空洞卷积
    dilated_convolution_2d(input, input_h, input_w, 
                          kernel, kernel_h, kernel_w,
                          output, output_h, output_w,
                          dilation, stride, padding);
    
    // 结束计时
    clock_t end_time = clock();
    double cpu_time_used = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
    
    printf("空洞卷积计算完成!\n");
    printf("计算时间: %.4f 秒\n", cpu_time_used);
    printf("每秒处理像素数: %.0f pixels/sec\n", 
           (double)(input_h * input_w) / cpu_time_used);
    printf("每秒FLOPS: %.2f MFLOPS\n", 
           (double)(output_h * output_w * kernel_h * kernel_w) / (cpu_time_used * 1000000));
    
    // 释放内存
    free(input);
    free(kernel);
    free(output);
    
    printf("dilation=%d时，%dx%d核的有效感受野为%dx%d\n", 
           dilation, kernel_h, kernel_w, 
           (kernel_h-1)*dilation+1, (kernel_w-1)*dilation+1);
    printf("仍然只有%d个参数\n", kernel_h * kernel_w);
    
    return 0;
}