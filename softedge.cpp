#include <vector>
#include <tuple>
#include <functional>
#include <string>

#include <OpenImageIO/imageio.h>
#include <OpenImageIO/imagebuf.h>
#include <OpenImageIO/imagebufalgo.h>

inline bool can_be_in_kernel(int x, int y, int width, int height)
{
    if (x < 0)
        return false;
    if (x >= width)
        return false;
    if (y < 0)
        return false;
    if (y >= height)
        return false;
    return true;
}

using coord_t = std::pair<int, int>;

inline void get_kernel(std::vector<coord_t>& outKernel, int x, int y, int width, int height, int kernel_radius)
{
    outKernel.clear();
    for (int h_offset = -kernel_radius; h_offset <= kernel_radius; ++h_offset)
    {
        for (int v_offset = -kernel_radius; v_offset <= kernel_radius; ++v_offset)
        {
            if (can_be_in_kernel(x + h_offset, y + v_offset, width, height))
                outKernel.emplace_back(x + h_offset, y + v_offset);
        }
    }
}

void print_usage() { std::cerr << "USAGE: unpremult <input-image> <output-image> [kernel-radius]\n"; }

int main(int argc, char* argv[], char* env[])
{
    if (argc < 3 || argc > 4)
    {
        print_usage();
        return 1;
    }

    int kernel_radius{ 1 };

    if (argc == 4)
    {
        try {
            kernel_radius = std::stoi(argv[3]);
        }
        catch (...) {
            print_usage();
            return 1;
        }
    }

    const char* filenameIn = argv[1];
    const char* filenameOut = argv[2];

    OIIO::ImageBuf inputBuffer(filenameIn);
    inputBuffer.read();
    const OIIO::ImageSpec& specIn = inputBuffer.spec();

    std::unordered_map<std::string, OIIO::ImageBuf> channelBuffers;

    for (int channelIdx = 0; channelIdx < specIn.nchannels; ++channelIdx)
    {
        const std::string channelName = specIn.channelnames[channelIdx];
        channelBuffers[channelName] = OIIO::ImageBufAlgo::channels(inputBuffer, 1, channelIdx);
    }

    if (!channelBuffers.count("R")
        || !channelBuffers.count("G")
        || !channelBuffers.count("B")
        || !channelBuffers.count("A"))
    {
        std::cerr << "ERROR: unexpected channel layout\n";
        return 1;
    }

    OIIO::ImageBuf bufferNewR(channelBuffers.at("R"));
    OIIO::ImageBuf bufferNewG(channelBuffers.at("G"));
    OIIO::ImageBuf bufferNewB(channelBuffers.at("B"));
    OIIO::ImageBuf bufferNewA(channelBuffers.at("A"));

    //////////////////////
    // buffer processing

    int width = channelBuffers.at("A").oriented_full_width();
    width = std::min(width, channelBuffers.at("R").oriented_full_width());
    width = std::min(width, channelBuffers.at("G").oriented_full_width());
    width = std::min(width, channelBuffers.at("B").oriented_full_width());

    int height = channelBuffers.at("A").oriented_full_height();
    height = std::min(height, channelBuffers.at("R").oriented_full_height());
    height = std::min(height, channelBuffers.at("G").oriented_full_height());
    height = std::min(height, channelBuffers.at("B").oriented_full_height());

    const std::size_t maxKernelSize = (1 + 2 * kernel_radius) * (1 + 2 * kernel_radius);
    std::vector<coord_t> kernel;
    kernel.reserve(maxKernelSize);
    std::vector<coord_t> relevantKernel;
    relevantKernel.reserve(maxKernelSize);

    unsigned int progress{ 0 };
    
    for (int y = 0; y < height; ++y)
    {
        unsigned int progressUpdate = 10.f * float(y) / float(height);
        if (progressUpdate != progress)
        {
            std::cout << 10 * (progress+1) << "%\n";
            progress = progressUpdate;
        }

        for (int x = 0; x < width; ++x)
        {
            get_kernel(kernel, x, y, width, height, kernel_radius);

            if (!kernel.size())
                continue;

            float alphaOld{ 0.f };
            channelBuffers.at("A").getpixel(x, y, &alphaOld, 1);

            if (!(alphaOld < 1.f))
                continue;

            float alphaSum{ 0.f };
            relevantKernel.clear();
            for (const auto& coord : kernel)
            {
                float alphaRead{ 0.f };
                channelBuffers.at("A").getpixel(coord.first, coord.second, &alphaRead, 1);
                alphaSum += alphaRead;

                if (alphaRead > 0.f)
                    relevantKernel.push_back(coord);
            }

            const float alphaNew = alphaSum / float(kernel.size());

            if (alphaNew == alphaOld)
                continue;

            bufferNewA.setpixel(x, y, &alphaNew, 1);

            if (alphaOld > 0.f)
                continue;

            float sumRelevantR{ 0.f };
            float sumRelevantG{ 0.f };
            float sumRelevantB{ 0.f };

            for (const auto& coord : relevantKernel)
            {
                float valueRead{ 0.f };
                channelBuffers.at("R").getpixel(coord.first, coord.second, &valueRead, 1);
                sumRelevantR += valueRead;
                channelBuffers.at("G").getpixel(coord.first, coord.second, &valueRead, 1);
                sumRelevantG += valueRead;
                channelBuffers.at("B").getpixel(coord.first, coord.second, &valueRead, 1);
                sumRelevantB += valueRead;
            }

            const float newR = sumRelevantR / float(relevantKernel.size());
            bufferNewR.setpixel(x, y, &newR, 1);

            const float newG = sumRelevantG / float(relevantKernel.size());
            bufferNewG.setpixel(x, y, &newG, 1);

            const float newB = sumRelevantB / float(relevantKernel.size());
            bufferNewB.setpixel(x, y, &newB, 1);
        }
    }


    // buffer processing
    //////////////////////


    //////////////////////
    // buffer swap
    channelBuffers["R"] = bufferNewR;
    channelBuffers["G"] = bufferNewG;
    channelBuffers["B"] = bufferNewB;
    channelBuffers["A"] = bufferNewA;
    // buffer swap
    //////////////////////


    OIIO::ImageBuf output;

    for (int channelIdx = 0; channelIdx < specIn.nchannels; ++channelIdx)
    {
        const std::string channelName = specIn.channelnames[channelIdx];
        const auto& channelToAppend = channelBuffers.at(channelName);

        if (!output.initialized())
        {
            output = OIIO::ImageBuf(channelToAppend.spec());
            OIIO::ImageBufAlgo::channel_append(output, channelToAppend, channelToAppend);
        }
        else
        {
            OIIO::ImageBuf outputCopy(output);
            output.clear();
            OIIO::ImageBufAlgo::channel_append(output, outputCopy, channelToAppend);
        }
    }
    output.write(filenameOut);

    return 0;
}

