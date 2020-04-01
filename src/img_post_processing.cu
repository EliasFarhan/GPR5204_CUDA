#include <img_post_processing.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include <iostream>

void Image::Load(const std::string &path)
{
    data = stbi_load(path.c_str(), &width, &height, &channelNmb, 0);
}
void Image::Write(const std::string &path, ImageType type)
{
    switch (type)
    {
    case ImageType ::BMP:
        stbi_write_bmp(path.c_str(), width, height, channelNmb, data);
        break;
    case ImageType ::JPG:
        stbi_write_jpg(path.c_str(), width, height, channelNmb, data, 90);
        break;
    case ImageType ::PNG:
        stbi_write_png(path.c_str(), width, height, channelNmb, data, channelNmb*width);
        break;
    default:
        break;
    }
}
void Image::Destroy()
{
    stbi_image_free(data);
}


__global__
void grayscale_gpu(unsigned char* img, unsigned char* newImg, size_t pxNmb, int channelNmb)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (size_t i = index; i < pxNmb; i += stride)
    {
        const float color[3] = {img[i*channelNmb]/255.0f,img[i*channelNmb+1]/255.0f,img[i*channelNmb+2]/255.0f};
        unsigned char gray =(unsigned char) ((0.2126f*color[0]+0.7152f*color[1]+0.0722f*color[2])/3.0f*255.0f);
        newImg[i*channelNmb] = gray;
        newImg[i*channelNmb+1] = gray;
        newImg[i*channelNmb+2] = gray;
        if(channelNmb == 4)
        {
            newImg[i*channelNmb+3] = img[i*channelNmb+3];
        }
    }
}

Image PostProcessing::Execute(Image &img)
{
    pxNmb_ = (size_t)img.width*img.height;
    channelNmb_ = img.channelNmb;
    width = img.width;
    height = img.height;
    size_t dataSize = pxNmb_*img.channelNmb;

    cudaMalloc(&gpuData_, dataSize);
    cudaMalloc(&gpuNewData_, dataSize);

    cudaMemcpy(gpuData_, img.data, dataSize, cudaMemcpyHostToDevice);
    GpuCall();
    Image newImg;
    newImg.channelNmb = img.channelNmb;
    newImg.height = img.height;
    newImg.width = img.width;
    newImg.data = (unsigned char*)malloc(dataSize);
    cudaDeviceSynchronize();
    cudaMemcpy(newImg.data, gpuNewData_, dataSize, cudaMemcpyDeviceToHost);
    cudaFree(gpuNewData_);
    cudaFree(gpuData_);
    return newImg;
}
void GrayScale::GpuCall()
{
    int blockSize = 256;
    int numBlocks = (pxNmb_ + blockSize - 1) / blockSize;
    grayscale_gpu <<<numBlocks, blockSize>>> (gpuData_, gpuNewData_, pxNmb_, channelNmb_);

}
Image GrayScale::CpuExecute(const Image &img)
{
    pxNmb_ = (size_t)img.width*img.height;
    channelNmb_ = img.channelNmb;
    size_t dataSize = pxNmb_*img.channelNmb;
    Image newImg;
    newImg.channelNmb = img.channelNmb;
    newImg.height = img.height;
    newImg.width = img.width;
    newImg.data = (unsigned char*)malloc(dataSize);
    for (size_t i = 0; i < pxNmb_; i++)
    {
        HdrColor color = HdrColor(img.data+i*channelNmb_);
        unsigned char gray = (unsigned char)((0.2126f*color.r+0.7152f*color.g+0.0722f*color.b)/3.0f*255.0f);
        newImg.data[i*channelNmb_] = gray;
        newImg.data[i*channelNmb_+1] = gray;
        newImg.data[i*channelNmb_+2] = gray;
        if(channelNmb_ == 4)
        {
            newImg.data[i*channelNmb_+3] = img.data[i*channelNmb_+3];
        }
    }
    return newImg;
}
Image Inverse::CpuExecute(const Image &img)
{
    pxNmb_ = (size_t)img.width*img.height;
    channelNmb_ = img.channelNmb;
    size_t dataSize = pxNmb_*img.channelNmb;
    Image newImg;
    newImg.channelNmb = img.channelNmb;
    newImg.height = img.height;
    newImg.width = img.width;
    newImg.data = (unsigned char*)malloc(dataSize);
    for (size_t i = 0; i < pxNmb_; i++)
    {
        const Color color = Color(img.data+i*channelNmb_);
        newImg.data[i*channelNmb_] = 255-color.r;
        newImg.data[i*channelNmb_+1] = 255-color.g;
        newImg.data[i*channelNmb_+2] = 255-color.b;
        if(channelNmb_ == 4)
        {
            newImg.data[i*channelNmb_+3] = img.data[i*channelNmb_+3];
        }
    }
    return newImg;
}

__global__
void inverse_gpu(unsigned char* img, unsigned char* newImg, size_t pxNmb, int channelNmb)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (size_t i = index; i < pxNmb; i += stride)
    {
        const unsigned char color[3] = {img[i*channelNmb],img[i*channelNmb+1],img[i*channelNmb+2]};
        newImg[i*channelNmb] = 255-color[0];
        newImg[i*channelNmb+1] = 255-color[1];
        newImg[i*channelNmb+2] = 255-color[2];
        if(channelNmb == 4)
        {
            newImg[i*channelNmb+3] = img[i*channelNmb+3];
        }
    }
}
void Inverse::GpuCall()
{
    int blockSize = 256;
    int numBlocks = (pxNmb_ + blockSize - 1) / blockSize;
    inverse_gpu <<<numBlocks, blockSize>>> (gpuData_, gpuNewData_, pxNmb_, channelNmb_);
}

typedef struct
{
    float r,g,b;
} GpuColor;


GpuColor AddColorCPU(const GpuColor& c1, const GpuColor& c2)
{
    GpuColor c;
    c.r = c1.r + c2.r;
    c.g = c1.g + c2.g;
    c.b = c1.b + c2.b;
    return c;
}

__device__
GpuColor AddColor(GpuColor c1, GpuColor c2)
{
    GpuColor c;
    c.r = c1.r + c2.r;
    c.g = c1.g + c2.g;
    c.b = c1.b + c2.b;
    return c;
}

GpuColor GetColorCPU(const unsigned char* ptr, size_t index)
{
    GpuColor c;
    c.r = static_cast<float>(ptr[index]) / 255.0f;
    c.g = static_cast<float>(ptr[index+1]) / 255.0f;
    c.b = static_cast<float>(ptr[index+2]) / 255.0f;
    return c;
}

__device__
GpuColor GetColor(const unsigned char* ptr, size_t index)
{
    GpuColor c;
    c.r = (float)ptr[index] / 255.0f;
    c.g = (float)ptr[index+1] / 255.0f;
    c.b = (float)ptr[index+2] / 255.0f;
    return c;
}
typedef struct
{
    float kernel[9];
} Matrix;


void Convulation(const unsigned char* img, unsigned char* newImg, size_t pxNmb, size_t channelNmb, int width, int height, const Matrix& m)
{
    const int offsetX[9] = {-1,0,1,-1,0,1,-1,0,1};
    const int offsetY[9] = {1,1,1,0,0,0,-1,-1,-1};
    std::cout << "Width: "<<width<<" Height: "<<height<<'\n';
    for (size_t i = 0; i < pxNmb; i ++)
    {
        const size_t x = i % width;
        const size_t y = i / width;
        GpuColor color = {0.0f,0.0f,0.0f};

        for(int j = 0; j < 9; j++)
        {
            const size_t offX = x + offsetX[j];
            const size_t offY = y + offsetY[j];
            const size_t offIndex = offX + offY * width;
            if(offIndex > pxNmb)
                continue;
            const float v = m.kernel[j];
            GpuColor c = GetColorCPU(img, offIndex * channelNmb);
            c.r *= v;
            c.g *= v;
            c.b *= v;
            color = AddColorCPU(color, c);

        }
        color.r = color.r < 0.0f ? 0.0f : color.r > 1.0f ? 1.0f: color.r;
        color.g = color.g < 0.0f ? 0.0f : color.g > 1.0f ? 1.0f: color.g;
        color.b = color.b < 0.0f ? 0.0f : color.b > 1.0f ? 1.0f: color.b;
        newImg[i*channelNmb] = static_cast<unsigned char>(color.r * 255.0f);
        newImg[i*channelNmb+1] = static_cast<unsigned char>(color.g * 255.0f);
        newImg[i*channelNmb+2] = static_cast<unsigned char>(color.b * 255.0f);
        if(channelNmb == 4)
        {
            newImg[i*channelNmb+3] = img[i*channelNmb+3];
        }
    }
}

__global__
void convolution_gpu(unsigned char* img, unsigned char* newImg, size_t pxNmb, int channelNmb, int width, int height, Matrix m)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    const int offsetX[9] = {-1,0,1,-1,0,1,-1,0,1};
    const int offsetY[9] = {1,1,1,0,0,0,-1,-1,-1};

    for (size_t i = index; i < pxNmb; i += stride)
    {
        const size_t x = i % width;
        const size_t y = i / width;
        GpuColor color = {0.0f,0.0f,0.0f};

        for(int j = 0; j < 9; j++)
        {
            const size_t offX = x + offsetX[j];
            const size_t offY = y + offsetY[j];
            const size_t offIndex = offX + offY * width;
            if(offIndex > pxNmb)
                continue;

            const float v = m.kernel[j];
            GpuColor c = GetColor(img, offIndex * channelNmb);
            c.r *= v;
            c.g *= v;
            c.b *= v;
            color = AddColor(color, c);

        }
        color.r = color.r < 0.0f ? 0.0f : color.r > 1.0f ? 1.0f: color.r;
        color.g = color.g < 0.0f ? 0.0f : color.g > 1.0f ? 1.0f: color.g;
        color.b = color.b < 0.0f ? 0.0f : color.b > 1.0f ? 1.0f: color.b;
        newImg[i*channelNmb] = (unsigned char)(color.r*255.0f);
        newImg[i*channelNmb+1] = (unsigned char)(color.g*255.0f);
        newImg[i*channelNmb+2] = (unsigned char)(color.b*255.0f);
        if(channelNmb == 4)
        {
            newImg[i*channelNmb+3] = img[i*channelNmb+3];
        }
    }

}
Image Sharpen::CpuExecute(const Image &img)
{
    const Matrix sharpen = {
        -1.0f,-1.0f,-1.0f,
        -1.0f,9.0f,-1.0f
        ,-1.0f,-1.0f,-1.0f
    };
    pxNmb_ = (size_t)img.width*img.height;
    channelNmb_ = img.channelNmb;
    size_t dataSize = pxNmb_*img.channelNmb;

    Image newImg;
    newImg.channelNmb = img.channelNmb;
    newImg.height = img.height;
    newImg.width = img.width;
    newImg.data = (unsigned char*)malloc(dataSize);
    Convulation(img.data, newImg.data, pxNmb_, channelNmb_, img.width, img.height, sharpen);
    return newImg;
}

void Sharpen::GpuCall()
{
    int blockSize = 256;
    int numBlocks = (pxNmb_ + blockSize - 1) / blockSize;
    Matrix sharpen = {
                -1.0f,-1.0f,-1.0f,
                -1.0f,9.0f,-1.0f
                ,-1.0f,-1.0f,-1.0f
            };
    convolution_gpu<<<numBlocks, blockSize>>> (gpuData_, gpuNewData_, pxNmb_, channelNmb_, width, height, sharpen);
}
Image Blur::CpuExecute(const Image &img)
{
    Matrix blur = {
        1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f,
        2.0f / 16.0f, 4.0f / 16.0f, 2.0f / 16.0f,
        1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f
    };
    pxNmb_ = (size_t)img.width*img.height;
    channelNmb_ = img.channelNmb;
    size_t dataSize = pxNmb_*img.channelNmb;
    Image newImg;
    newImg.channelNmb = img.channelNmb;
    newImg.height = img.height;
    newImg.width = img.width;
    newImg.data = (unsigned char*)malloc(dataSize);
    Convulation(img.data, newImg.data, pxNmb_, channelNmb_, img.width, img.height, blur);
    return newImg;
}
void Blur::GpuCall()
{
    int blockSize = 256;
    int numBlocks = (pxNmb_ + blockSize - 1) / blockSize;
    Matrix blur = {
        1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f,
        2.0f / 16.0f, 4.0f / 16.0f, 2.0f / 16.0f,
        1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f
    };
    convolution_gpu<<<numBlocks, blockSize>>> (gpuData_, gpuNewData_, pxNmb_, channelNmb_, width, height, blur);
}
Image Edge::CpuExecute(const Image &img)
{
    const Matrix edge = {
        1.0f, 1.0f, 1.0f,
        1.0f, -8.0f, 1.0f,
        1.0f, 1.0f, 1.0f
    };
    pxNmb_ = (size_t)img.width*img.height;
    channelNmb_ = img.channelNmb;
    size_t dataSize = pxNmb_*img.channelNmb;
    Image newImg;
    newImg.channelNmb = img.channelNmb;
    newImg.height = img.height;
    newImg.width = img.width;
    newImg.data = (unsigned char*)malloc(dataSize);
    Convulation(img.data, newImg.data, pxNmb_, channelNmb_, img.width, img.height, edge);
    return newImg;
}
void Edge::GpuCall()
{
    int blockSize = 256;
    int numBlocks = (pxNmb_ + blockSize - 1) / blockSize;
    Matrix edge = {
        1.0f, 1.0f, 1.0f,
        1.0f, -8.0f, 1.0f,
        1.0f, 1.0f, 1.0f
    };
    convolution_gpu<<<numBlocks, blockSize>>> (gpuData_, gpuNewData_, pxNmb_, channelNmb_, width, height, edge);
}
