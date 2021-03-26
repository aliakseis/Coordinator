// Coordinator.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

#include "LinesMeasurerLibrary/detect-lines.h"

#include <iostream>
#include <fstream>
#include <map>
#include <tuple>

#include  <algorithm>
#include  <vector>

#include <filesystem>

// for %f in (*.tif) do C:\workspace\Coordinator\x64\Release\Coordinator.exe "%~nf".tif c:\images\coords/"%~nf".csv

auto getCenters(const cv::Mat& img_thr)
{
    // Use connected components with stats
    cv::Mat labels;
    cv::Mat stats;
    cv::Mat centroids;
    auto num_objects = connectedComponentsWithStats(img_thr, labels, stats, centroids);

    std::vector<cv::Point2d> result;
    result.reserve(num_objects - 1);

    for (int i = 1; i < num_objects; i++) {
        result.push_back({ centroids.at<double>(i, 0), centroids.at<double>(i, 1) });
    }

    return result;
}

enum { NUM_POINTS = 31 };

void QuickNDirtyFix(std::vector<cv::Point2d>& reducedLines)
{
    // quick'n'dirty fix
    auto sample = reducedLines.back();
    auto step = (sample - reducedLines.front()) / int(reducedLines.size() - 1);


    for (int i = reducedLines.size(); i < NUM_POINTS; ++i) {
        sample += step;
        reducedLines.push_back(sample);
    }

    reducedLines.resize(NUM_POINTS);
}

template<typename T> bool isIncreaingSequence(T it1, const T it1End, T it2, const T it2End)
{
    int prevX = INT_MIN;
    for (; it1 != it1End && it2 != it2End; ++it1, it2)
    {
        for (auto& v : { *it1, *it2 })
        {
            if (v.x < prevX)
                return false;
        }
    }
    return true;
}


int main(int argc, char** argv)
{
    if (argc < 5) {
        std::cerr << "Wrong number of arguments.\n";
        return 1;
    }

    try {

        //const char* filename = argv[1];

        const char* dirname = argv[1];

        std::vector<cv::String> fn;
        cv::glob(cv::String(dirname) + "/*.tif", fn, true);

        const char* out = argv[2];

        std::ofstream ostr(out);
        if (!ostr) {
            std::cerr << "Cannot open output file.\n";
            return 1;
        }

        std::cout << "writing to " << out << '\n';

        for (int i = 0; i < NUM_POINTS; ++i)
            ostr << "red_" << i << "_x,red_" << i << "_y,";
        for (int i = 0; i < NUM_POINTS; ++i)
            ostr << "blue_" << i << "_x,blue_" << i << "_y,";

        ostr << "id_string\n";

        std::vector<cv::String> bads;

        for (const auto& filename : fn)
        {
            auto name = filename.substr(0, filename.find_last_of('.'));
            const auto pos = name.find_last_of("\\/");
            if (decltype(name)::npos != pos)
            {
                name = name.substr(pos + 1);
            }
            bads.push_back(name);

            cv::Mat src = cv::imread(filename);

            if (src.empty())
            {
                std::cerr << "Could not read input file.\n";
                return 1;
            }

            cv::Mat data;
            src.convertTo(data, CV_32F);
            data = data.reshape(1, data.total());

            // do kmeans
            enum { K = 3 };
            cv::Mat kmeansLabels;
            std::vector<cv::Vec3f> kmeansCenters;
            cv::kmeans(data, K, kmeansLabels, cv::TermCriteria(cv::TermCriteria::MAX_ITER, 10, 1.0), 3,
                cv::KMEANS_PP_CENTERS, kmeansCenters);

            if (kmeansCenters.size() < K)
                continue;

            const auto minmax = std::minmax_element(kmeansCenters.begin(), kmeansCenters.end(),
                [](const cv::Vec3f& left, const cv::Vec3f& right) { return left[0] - left[2] < right[0] - right[2]; });

            auto redsIdx = minmax.first - kmeansCenters.begin();
            auto bluesIdx = minmax.second - kmeansCenters.begin();

            cv::Mat reds = kmeansLabels == redsIdx;
            cv::Mat blues = kmeansLabels == bluesIdx;

            reds = reds.reshape(1, src.rows);
            blues = blues.reshape(1, src.rows);

            auto redCenters = getCenters(reds);
            auto blueCenters = getCenters(blues);

            if (redCenters.size() < 2 || blueCenters.size() < 2)
                continue;

            if (redCenters.size() > NUM_POINTS + 1 || blueCenters.size() > NUM_POINTS + 1)
                continue;

            auto point2dSortLam = [](const cv::Point2d& left, const cv::Point2d& right) {
                return left.x < right.x;
            };

            std::sort(redCenters.begin(), redCenters.end(), point2dSortLam);
            std::sort(blueCenters.begin(), blueCenters.end(), point2dSortLam);

            const bool upsideDown = redCenters[0].y > blueCenters[0].y;

            if (upsideDown)
            {
                std::reverse(redCenters.begin(), redCenters.end());
                std::reverse(blueCenters.begin(), blueCenters.end());
            }

            QuickNDirtyFix(redCenters);
            QuickNDirtyFix(blueCenters);

            //if (redCenters.back().x < 0 || redCenters.back().x > src.cols || blueCenters.back().x < 0 || blueCenters.back().x > src.cols
            //    || redCenters.back().y < 0 || redCenters.back().y > src.rows || blueCenters.back().y < 0 || blueCenters.back().y > src.rows)
            //    continue;

            if (upsideDown ? redCenters.back().y <= blueCenters.back().y : redCenters.back().y >= blueCenters.back().y)
                continue;

            if (!upsideDown && !isIncreaingSequence(redCenters.begin(), redCenters.end(), blueCenters.begin(), blueCenters.end()))
                continue;
            if (upsideDown && !isIncreaingSequence(blueCenters.rbegin(), blueCenters.rend(), redCenters.rbegin(), redCenters.rend()))
                continue;

            for (auto& lst : { std::ref(redCenters), std::ref(blueCenters) })
            {
                for (auto& v : lst.get())
                {
                    v.x /= src.cols;
                    v.y /= src.rows;
                }
            }



            for (const auto& lst : { redCenters, blueCenters })
            {
                for (auto& v : lst)
                {
                    ostr << v.x << ',' << v.y;
                    ostr << ',';
                }
            }
            ostr << '"' << name << '"';
            ostr << '\n';

            bads.pop_back();

            // alternative
            /*
            std::vector<cv::Point2d> altRedCenters, altBlueCenters;

            const auto altPath = argv[4] + ('/' + name) + ".tif";

            std::vector<std::tuple<double, double, double, double, double>> alternative;
            try {
                alternative = calculating(altPath);
            }
            catch (const std::exception& ex) {
                continue;
            }

            if (alternative.size() != NUM_POINTS)
                continue;

            bool tooFar = false;

            for (int i = 0; i < NUM_POINTS; ++i)
            {
                auto& line = alternative[i];
                cv::Point2d ptRed(std::get<0>(line), std::get<1>(line)), ptBlue(std::get<2>(line), std::get<3>(line));
                const auto X_DIST = 0.01;
                const auto Y_DIST = 0.02;

                const auto redsDiffer = abs(ptRed.x - redCenters[i].x) > X_DIST || abs(ptRed.y - redCenters[i].y) > Y_DIST;
                const auto bluesDiffer = abs(ptBlue.x - blueCenters[i].x) > X_DIST || abs(ptBlue.y - blueCenters[i].y) > Y_DIST;

                if (redsDiffer || bluesDiffer)
                {
                    tooFar = true;
                    break;
                }

                altRedCenters.push_back(ptRed); 
                altBlueCenters.push_back(ptBlue);

                //if (!redsDiffer)
                //    redCenters[i] = (redCenters[i] + ptRed) / 2;
                //if (!bluesDiffer)
                //    blueCenters[i] = (blueCenters[i] + ptBlue) / 2;
            }

            if (tooFar)
                continue;

            for (const auto& lst : { altRedCenters, altBlueCenters })
            {
                for (auto& v : lst)
                {
                    ostr << v.x << ',' << v.y;
                    ostr << ',';
                }
            }
            ostr << '"' << name << '"';
            ostr << '\n';
            */
        }

        std::ofstream badsstream(argv[3]);
        if (!badsstream) {
            std::cerr << "Cannot open bads output file.\n";
            return 1;
        }

        for (const auto& s : bads)
        {
            badsstream << s << '\n';
        }

        return 0;
    }
    catch (const std::exception& ex) {
        std::cerr << typeid(ex).name() << ": " << ex.what() << '\n';

        return 1;
    }
}
