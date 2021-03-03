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

/*
namespace cv {

inline bool operator <(const Vec3b& left, const Vec3b& right)
{
    return std::tie(left[0], left[1], left[2]) < std::tie(right[0], right[1], right[2]);
}

}
*/

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

    //for (int i = 0; i < num_objects; i++) {

    //    cv::Rect rct(
    //        stats.at<int>(i, CC_STAT_LEFT),
    //        stats.at<int>(i, CC_STAT_TOP),
    //        stats.at<int>(i, CC_STAT_WIDTH),
    //        stats.at<int>(i, CC_STAT_HEIGHT)
    //    );

    //    result.push_back({ rct,
    //        { centroids.at<double>(i, 0), centroids.at<double>(i, 1) },
    //        stats.at<int>(i, CC_STAT_AREA) });
    //}

    //for (int y = 0; y < labels.rows; ++y) {
    //    for (int x = 0; x < labels.cols; ++x)
    //    {
    //        const cv::Point pt{ x, y };
    //        auto idx = labels.at<int>(pt);
    //        if (idx > 0)
    //        {
    //            auto value = proximity.at<float>(pt);
    //            result[idx - 1].values.emplace_back(pt, value);
    //        }
    //    }
    //}

}

void QuickNDirtyFix(std::vector<cv::Point2d>& reducedLines)
{
    // quick'n'dirty fix
    auto sample = reducedLines.back();
    auto step = (sample - reducedLines.front()) / int(reducedLines.size() - 1);

    for (int i = reducedLines.size(); i < 32; ++i) {
        sample += step;
        reducedLines.push_back(sample);
    }
}


int main(int argc, char** argv)
{
    if (argc < 3)
        return 1;

    try {

        const char* filename = argv[1];// "/images/labels/20201218103625048_SNG1218103552_6.tif";

        cv::Mat src = cv::imread(filename);

        /*
        std::map<cv::Vec3b, int> counts;

        for (int y = 0; y < src.rows; ++y)
            for (int x = 0; x < src.cols; ++x)
            {
                auto v = src.at<cv::Vec3b>(y, x);
                ++counts[v];
            }

        for (auto& v : counts)
            std::cout << v.first << " " << v.second << '\n';
        */

        cv::Mat data;
        src.convertTo(data, CV_32F);
        data = data.reshape(1, data.total());

        // do kmeans
        enum { K = 3 };
        cv::Mat kmeansLabels;// , kmeansCenters;
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

        const char* out = argv[2];//"/images/labels/20201218103625048_SNG1218103552_6.csv";

        std::ofstream ostr(out);

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

        /*
        imshow("reds", reds);
        imshow("blues", blues);

        // reshape both to a single row of Vec3f pixels:
        //kmeansCenters = kmeansCenters.reshape(3, kmeansCenters.rows);
        data = data.reshape(3, data.rows);


        for (auto& v : kmeansCenters)
            std::cout << v << '\n';

        //for (int i = 0; i < kmeansCenters.rows; ++i) {
        //    auto v = kmeansCenters.at<cv::Vec3f>(i);
        //    std::cout << v << '\n';
        //}


        // replace pixel values with their center value:
        auto p = data.ptr<cv::Vec3f>();
        for (size_t i = 0; i < data.rows; i++) {
            int center_id = kmeansLabels.at<int>(i);
            p[i] = kmeansCenters[center_id];
        }

        // back to 2d, and uchar:
        auto ocv = data.reshape(3, src.rows);
        ocv.convertTo(ocv, CV_8U);

        imshow("ocv", ocv);

        //![exit]
        // Wait and Exit
        cv::waitKey();
        */

        return 0;
    }
    catch (const std::exception& ex) {
        std::cerr << typeid(ex).name() << ": " << ex.what() << '\n';

        return 1;
    }
}
