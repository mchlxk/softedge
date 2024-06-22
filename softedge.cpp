#include <vector>
#include <tuple>
#include <functional>

#include <OpenImageIO/imageio.h>
#include <OpenImageIO/imagebuf.h>
#include <OpenImageIO/imagebufalgo.h>

// https://openimageio.readthedocs.io/en/v2.5.11.0/imageinput.html

constexpr int P_kernel_size{1};

bool can_be_in_kernel(int x, int y, int width, int height)
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

std::vector<std::pair<int, int>> get_kernel(int x, int y, int width, int height, int kernel_size)
{
    std::vector<std::pair<int, int>> kernel;
    for (int h_offset = -kernel_size; h_offset <= kernel_size; ++h_offset)
    {
        for (int v_offset = -kernel_size; v_offset <= kernel_size; ++v_offset)
        {
            if (can_be_in_kernel(x + h_offset, y + v_offset, width, height))
                kernel.emplace_back(x + h_offset, y + v_offset);
        }
    }
    return kernel;
}


int main(int argc, char* argv[], char* env[])
{
    if (argc != 3)
    {
        std::cout << "USAGE: softedge <input-image> <output-image>\n";
        return 1;
    }

    const char* filenameIn = argv[1];
    const char* filenameOut = argv[2];

    OIIO::ImageBuf input(filenameIn);
    input.read();
    const OIIO::ImageSpec& specIn = input.spec();
    
    std::unordered_map<std::string, OIIO::ImageBuf> channelBuffers;

    for (int channelIdx = 0; channelIdx < specIn.nchannels; ++channelIdx)
    {
        const std::string channelName = specIn.channelnames[channelIdx];
        channelBuffers[channelName] = OIIO::ImageBufAlgo::channels(input, 1, channelIdx);
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


    //channelBuffers["R"] = OIIO::ImageBufAlgo::div(channelBuffers.at("R"), channelBuffers.at("A"));
    //channelBuffers["R"] = OIIO::ImageBufAlgo::mul(channelBuffers.at("R"), channelBuffers.at("A"));
    //channelBuffers["G"] = OIIO::ImageBufAlgo::mul(channelBuffers.at("G"), channelBuffers.at("A"));
    //channelBuffers["B"] = OIIO::ImageBufAlgo::mul(channelBuffers.at("B"), channelBuffers.at("A"));

    
    int width = channelBuffers.at("A").oriented_full_width();
    width = std::min(width, channelBuffers.at("R").oriented_full_width());
    width = std::min(width, channelBuffers.at("G").oriented_full_width());
    width = std::min(width, channelBuffers.at("B").oriented_full_width());

    int height = channelBuffers.at("A").oriented_full_height();
    height = std::min(height, channelBuffers.at("R").oriented_full_height());
    height = std::min(height, channelBuffers.at("G").oriented_full_height());
    height = std::min(height, channelBuffers.at("B").oriented_full_height());


    for (int y = 0; y < height; ++y)
    {
        std::cout << "y: " << y << " ";
        for (int x = 0; x < width; ++x)
        {
            const auto kernel = get_kernel(x, y, width, height, P_kernel_size);

            if (!kernel.size())
                continue;
            
            float oldAlpha{0.f};
            channelBuffers.at("A").getpixel(x, y, &oldAlpha, 1);

            float alphaSum{ 0.f };
            std::vector<std::pair<int, int>> relevantKernel;
            for (const auto& coord : kernel)
            {
                float alpha{0.f};
                channelBuffers.at("A").getpixel(coord.first, coord.second, &alpha, 1);
                alphaSum += alpha;
            
                if (alpha > 0.f)
                    relevantKernel.push_back(coord);
            }

            const float newAlpha = alphaSum / float(kernel.size());
            
            if (newAlpha == oldAlpha)
                continue;

            bufferNewA.setpixel(x, y, &newAlpha, 1);

            if (oldAlpha > 0.f)
                continue;

            float sumRelevantR{ 0.f };
            float sumRelevantG{ 0.f };
            float sumRelevantB{ 0.f };

            for (const auto& coord : relevantKernel)
            {
                float value{ 0.f };

                channelBuffers.at("R").getpixel(coord.first, coord.second, &value, 1);
                sumRelevantR += value;
                channelBuffers.at("G").getpixel(coord.first, coord.second, &value, 1);
                sumRelevantG += value;
                channelBuffers.at("B").getpixel(coord.first, coord.second, &value, 1);
                sumRelevantB += value;
            }

            const float newR = sumRelevantR / float(relevantKernel.size());
            bufferNewR.setpixel(x, y, &newR, 1);

            const float newG = sumRelevantG / float(relevantKernel.size());
            bufferNewG.setpixel(x, y, &newG, 1);

            const float newB = sumRelevantB / float(relevantKernel.size());
            bufferNewB.setpixel(x, y, &newB, 1);
        }
    }


    /*
    for (OIIO::ImageBuf::Iterator<uint8_t> it(channelBuffers.at("A")); !it.done(); ++it)
    {
        if (!it.exists())   // Make sure the iterator is pointing
            continue;        //   to a pixel in the data window

        if (it[0] < .5f)
            it[0] = 0;
        else
            it[0] = 1.f;

        //if (it[0] > 0)
        //    std::cout << it[0] << " ";

    }
    */


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

