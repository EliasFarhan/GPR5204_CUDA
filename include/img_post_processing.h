#pragma once
#include <string>

enum class ImageType
{
    NONE,
    PNG,
    JPG,
    BMP
};

struct HdrColor
{
    float r,g,b;
    explicit HdrColor(const std::uint8_t* c)
    {
        r = ((float)c[0])/255.0f;
        g = ((float)c[1])/255.0f;
        b = ((float)c[2])/255.0f;
    }
    explicit HdrColor(std::uint8_t c)
    {
        g = b = r = (float)c/255.0f;
    }
};

struct Color
{
    std::uint8_t r,g,b;
    explicit Color(const std::uint8_t* c)
    {
        r = c[0];
        g = c[1];
        b = c[2];
    }
    explicit Color(std::uint8_t c)
    {
        b = g = r = c;
    }
    explicit Color(HdrColor c)
    {
        r = c.r*255.0f>255.0f?255:c.r*255.0f;
        g = c.g*255.0f>255.0f?255:c.g*255.0f;
        b = c.b*255.0f>255.0f?255:c.b*255.0f;
    }
};



struct Image
{
    int channelNmb = 0;
    int width, height;
    unsigned char* data;

    void Destroy();
    void Load(const std::string &path);
    void Write(const std::string &path, ImageType type);
};

class PostProcessing
{
public:
    virtual Image Execute(Image& img);
    virtual Image CpuExecute(const Image &img) = 0;
protected:
    virtual void GpuCall() = 0;
    unsigned char* gpuData_ = nullptr;
    unsigned char* gpuNewData_ = nullptr;
    size_t pxNmb_ = 0;
    int channelNmb_ = 0;
    int width = 0, height = 0;
};

class GrayScale : public PostProcessing
{
protected:
    void GpuCall() override;
public:
    Image CpuExecute(const Image &img) override;

};

class Inverse : public PostProcessing
{
public:
    Image CpuExecute(const Image &img) override;
protected:
    void GpuCall() override;

};

class Sharpen : public PostProcessing
{
public:
    Image CpuExecute(const Image &img) override;
protected:
    void GpuCall() override;

};