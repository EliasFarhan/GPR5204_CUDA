#include <img_post_processing.h>
#include <iostream>
#include <filesystem>
namespace fs = std::filesystem;

int main()
{
    std::cout << "Current path is " << fs::current_path() << '\n';

    Image img;
    img.Load("../data/Lenna.png");

    Sharpen g;
    auto start = std::chrono::high_resolution_clock::now();
    auto newImg = g.Execute(img);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "GPU Time: " << duration.count() << '\n';
    start = std::chrono::high_resolution_clock::now();
    auto newCpuImg = g.CpuExecute(img);
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "CPU Time: " << duration.count() << '\n';

    img.Destroy();
    newImg.Write("../data/Lenna-sharpen.jpg", ImageType::JPG);
    newImg.Destroy();
    newCpuImg.Write("../data/Lenna-sharpen-cpu.jpeg", ImageType::JPG);
    newCpuImg.Destroy();
    return 0;
}