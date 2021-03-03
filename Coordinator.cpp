// Coordinator.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

#include <iostream>
#include <fstream>
#include <map>
#include <tuple>

#include  <algorithm>
#include  <vector>


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

void QuickNDirtyFix(std::vector<cv::Point2d>& reducedLines)
{
    // quick'n'dirty fix
    auto sample = reducedLines.back();
    auto step = (sample - reducedLines.front()) / int(reducedLines.size() - 1);

    enum { NUM_POINTS = 32 };

    for (int i = reducedLines.size(); i < NUM_POINTS; ++i) {
        sample += step;
        reducedLines.push_back(sample);
    }

    reducedLines.resize(NUM_POINTS);
}

int main(int argc, char** argv)
{
    if (argc < 3) {
        std::cerr << "Wrong number of arguments.\n";
        return 1;
    }

    try {

        const char* filename = argv[1];

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

        const char* out = argv[2];

        std::ofstream ostr(out);

        std::cout << "writing to " << out << '\n';

        for (const auto& lst : { redCenters, blueCenters })
        {
            bool start = true;
            for (auto& v : lst)
            {
                if (!start)
                    ostr << ',';
                start = false;
                ostr << (v.x / src.cols) << ',' << (v.y / src.rows);
            }
            ostr << '\n';
        }

        return 0;
    }
    catch (const std::exception& ex) {
        std::cerr << typeid(ex).name() << ": " << ex.what() << '\n';

        return 1;
    }
}
