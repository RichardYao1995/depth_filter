#include <iostream>
#include <fstream>
#include <chrono>
#include <string>
#include "pcl/filters/filter.h"
#include "pcl/point_types.h"
#include "pcl/filters/voxel_grid.h"
#include "pcl/io/ply_io.h"
#include "pcl/io/obj_io.h"
#include "pcl/common/common.h"
#include "pcl/surface/poisson.h"
#include "pcl/common/transforms.h"
#include "pcl/filters/statistical_outlier_removal.h"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include "StereoEfficientLargeScale.h"

using namespace std;
using namespace pcl;

cv::Point3f projectDepthTo3D(const cv::Point2f & pt, float depth)
{
    if(depth > 0.0f)
    {
        float W = depth / 762.72;
        return cv::Point3f((pt.x - 640) * W, (pt.y - 360) * W, 762.72 * W);
    }
    float bad_point = std::numeric_limits<float>::quiet_NaN ();
    return cv::Point3f(bad_point, bad_point, bad_point);
}

Eigen::Vector3d projectDisparityTo3D(const cv::Point2f & pt, float disparity)
{
    if(disparity > 0.0f)
    {
        float W = 0.35 / disparity;
        return Eigen::Vector3d((pt.x - 640) * W, (pt.y - 360) * W, 762.72 * W);
    }
    float bad_point = std::numeric_limits<float>::quiet_NaN ();
    return Eigen::Vector3d(bad_point, bad_point, bad_point);
}

Eigen::Matrix3d getRFromrpy(const Eigen::Vector3d& rpy)
{
    Eigen::Matrix3d R;
    Eigen::Vector3d ea(rpy(0),rpy(1),rpy(2));
    R = Eigen::AngleAxisd(ea[2], Eigen::Vector3d::UnitZ()) *
                 Eigen::AngleAxisd(ea[1], Eigen::Vector3d::UnitY()) *
                 Eigen::AngleAxisd(ea[0], Eigen::Vector3d::UnitX());
    return R;
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudFromDepthRGB(const cv::Mat & imageRgb,
                                                             const cv::Mat & imageDepth,
                                                             int decimation)
{
    float maxDepth = 20.0;
    float minDepth = 0.0;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);


    //cloud.header = cameraInfo.header;
    cloud->height = imageRgb.rows/decimation;
    cloud->width  = imageRgb.cols/decimation;
    cloud->is_dense = false;
    cloud->resize(cloud->height * cloud->width);

    for(int h = 0; h < imageRgb.rows && h/decimation < (int)cloud->height; h+=decimation)
    {
        for(int w = 0; w < imageRgb.cols && w/decimation < (int)cloud->width; w+=decimation)
        {
            float depth = imageDepth.at<float>(h,w);
            pcl::PointXYZRGB & pt = cloud->at((h/decimation)*cloud->width + (w/decimation));

            pt.b = imageRgb.at<cv::Vec3b>(h,w)[0];
            pt.g = imageRgb.at<cv::Vec3b>(h,w)[1];
            pt.r = imageRgb.at<cv::Vec3b>(h,w)[2];

            cv::Point3f ptXYZ = projectDepthTo3D(cv::Point2f(w, h), depth);
            if(ptXYZ.z >= minDepth && ptXYZ.z <= maxDepth)
            {
                pt.x = ptXYZ.x;
                pt.y = ptXYZ.y;
                pt.z = ptXYZ.z;
            }
            else
            {
                pt.x = pt.y = pt.z = std::numeric_limits<float>::quiet_NaN();
            }
        }
    }
    return cloud;
}

int main()
{
    ifstream in_image("/home/uisee/workspace/EfficientLargeScaleStereo/stereo1.txt");
    ifstream in_pose("/home/uisee/Data/stereo-0/L1_clock_outer_alone_test.txt.pn");
    ifstream in_ref("/home/uisee/Data/stereo-0/ref_pose.txt");

    string left("/home/uisee/Data/stereo-0/left/");
    string right("/home/uisee/Data/stereo-0/right/");
    //string img("/home/uisee/Data/image_capturer_0/");
    string image;
    Eigen::Matrix<double,1,6> pose;
    Eigen::Vector3d t1, t2;
    Eigen::Matrix3d R1, R2;

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr Clouds(new pcl::PointCloud<pcl::PointXYZRGB>);
    Eigen::Matrix4f T_cam_car;
    T_cam_car << 1, 0, 0, -0.175,
               0, 0, 1, 1.99,
               0, -1, 0, 0,
               0, 0, 0, 1;
    string startx, starty;
    in_pose >> startx >> starty;

    int flag = 0;
    uint row = 720;
    uint col = 1280;
    cv::Mat depth_first(row, col, CV_32FC1), depth_first_cov(row, col, CV_32FC1);
    cv::Mat depth_second_cov(row, col, CV_32FC1), depth_second(row, col, CV_32FC1);

    auto start = std::chrono::system_clock::now();
    Mat img1;
    while(in_image >> image)
    {
        string x, y, z, roll, pitch, yaw, filename;
        if(!(in_pose >> filename))
            break;
        string x_ref, y_ref, z_ref, roll_ref, pitch_ref, yaw_ref;
        in_ref >> x_ref >> y_ref >> z_ref >> roll_ref >> pitch_ref >> yaw_ref;
        in_pose >> pitch >> yaw >> roll >> x >> z >> y;

        pose << stod(x_ref), stod(y_ref), stod(z_ref), stod(roll_ref), stod(pitch_ref), stod(yaw_ref);
        Eigen::Matrix3d R = getRFromrpy(Eigen::Vector3d(pose(3), pose(4), pose(5)));
        Eigen::Matrix4d T;
        T << R(0, 0), R(0, 1), R(0, 2), pose(0),
             R(1, 0), R(1, 1), R(1, 2), pose(1),
             R(2, 0), R(2, 1), R(2, 2), pose(2),
             0.0, 0.0, 0.0, 1.0;
        Mat leftim = imread(left + image);
        Mat rightim = imread(right + image);
        Mat leftgray, rightgray;
        cv::cvtColor(leftim, leftgray, CV_BGR2GRAY);
        cv::cvtColor(rightim, rightgray, CV_BGR2GRAY);
        Mat img2 = leftgray.clone();

        Mat dest;
        StereoEfficientLargeScale elas(0,128);
        elas(leftgray,rightgray,dest,100);

        dest.convertTo(dest,CV_32FC1,1.0/16);
        if(flag == 0)
        {
            for(int x = 0;x < row;x++)
            {
                for(int y = 0;y < col;y++)
                {
                    if(dest.at<float>(x, y) < 3.0)
                        continue;
                    double depth = 762.72 * 0.35 / dest.at<float>(x, y);
                    double sigma = depth - 762.72 * 0.35 / (dest.at<float>(x, y) - 1);
                    double sigma2 = sigma * sigma;
                    depth_first_cov.ptr<float>(x)[y] = sigma2;
                    depth_first.ptr<float>(x)[y] = depth;
                }
            }
            t1 << pose(0), pose(1), pose(2);
            R1 = getRFromrpy(Eigen::Vector3d(pose(3), pose(4), pose(5)));
            t1 = R1 * (-t1);
            flag++;
            img1 = img2.clone();
            continue;
        }

        if(flag < 3)
        {
            cv::Mat flow;
            cv::calcOpticalFlowFarneback(img1, img2, flow, 0.5, 3, 15, 3, 5, 1.2, 0);
            for(int x = 0;x < row;x++)
            {
                for(int y = 0;y < col;y++)
                {
                    if(dest.at<float>(x, y) < 3.0)
                        continue;
                    double depth = 762.72 * 0.35 / dest.at<float>(x, y);
                    double sigma = depth - 762.72 * 0.35 / (dest.at<float>(x, y) - 1);
                    double sigma2 = sigma * sigma;
                    depth_second_cov.ptr<float>(x)[y] = sigma2;
                    depth_second.ptr<float>(x)[y] = depth;
                }
            }
            t2 << pose(0), pose(1), pose(2);
            R2 = getRFromrpy(Eigen::Vector3d(pose(3), pose(4), pose(5)));
            t2 = R2 * (-t2);
            Eigen::Matrix3d R12 = R2 * R1.inverse();
            Eigen::Vector3d t12 = t2 - R12 * t1;
            for(uint x = 20;x < row - 20;x++)
            {
                for(uint y = 20;y < col - 20;y++)
                {
                    if(dest.at<float>(x, y) < 3.0)
                        continue;
                    Point2f& fxy = flow.at<Point2f>(x, y);
                    double depth2 = depth_second.at<float>(x + fxy.y, y + fxy.x);
                    double sigma_second = depth_second_cov.at<float>(x + fxy.y, y + fxy.x);

                    double sigma_first = depth_first_cov.ptr<float>(x)[y];

                    cv::Point2f pt1(y, x);
                    Eigen::Vector3d point1 = projectDisparityTo3D(pt1, dest.at<float>(x, y));
                    Eigen::Vector3d point12 = R12 * point1 + t12;
                    double depth12 = point12[0];
                    double depth_fuse = (sigma_second * depth12 + sigma_first * depth2) / (sigma_second + sigma_first);
                    double sigma_fuse = (sigma_second * sigma_first) / (sigma_second + sigma_first);

                    depth_second_cov.at<float>(x + fxy.y, y + fxy.x) = sigma_fuse;
                    depth_second.at<float>(x + fxy.y, y + fxy.x) = depth_fuse;
                }
            }
            flag++;
            t1 = t2;
            R1 = R2;
            depth_first = depth_second.clone();
            depth_first_cov = depth_second_cov.clone();
            img1 = img2.clone();
            continue;
        }

        if(flag == 3)
        {
            cv::Mat flow;
            cv::calcOpticalFlowFarneback(img1, img2, flow, 0.5, 3, 15, 3, 5, 1.2, 0);
            for(int x = 0;x < row;x++)
            {
                for(int y = 0;y < col;y++)
                {
                    if(dest.at<float>(x, y) < 3.0)
                        continue;
                    double depth = 762.72 * 0.35 / dest.at<float>(x, y);
                    double sigma = depth - 762.72 * 0.35 / (dest.at<float>(x, y) - 1);
                    double sigma2 = sigma * sigma;
                    depth_second_cov.ptr<float>(x)[y] = sigma2;
                    depth_second.ptr<float>(x)[y] = depth;
                }
            }
            t2 << pose(0), pose(1), pose(2);
            R2 = getRFromrpy(Eigen::Vector3d(pose(3), pose(4), pose(5)));
            t2 = R2 * (-t2);
            Eigen::Matrix3d R12 = R2 * R1.inverse();
            Eigen::Vector3d t12 = t2 - R12 * t1;
            for(uint x = 20;x < row - 20;x++)
            {
                for(uint y = 20;y < col - 20;y++)
                {
                    if(dest.at<float>(x, y) < 3.0)
                        continue;
                    Point2f& fxy = flow.at<Point2f>(x, y);
                    double depth2 = depth_second.at<float>(x + fxy.y, y + fxy.x);
                    double sigma_second = depth_second_cov.at<float>(x + fxy.y, y + fxy.x);

                    double sigma_first = depth_first_cov.ptr<float>(x)[y];

                    cv::Point2f pt1(y, x);
                    Eigen::Vector3d point1 = projectDisparityTo3D(pt1, dest.at<float>(x, y));
                    Eigen::Vector3d point12 = R12 * point1 + t12;
                    double depth12 = point12[0];
                    double depth_fuse = (sigma_second * depth12 + sigma_first * depth2) / (sigma_second + sigma_first);
                    double sigma_fuse = (sigma_second * sigma_first) / (sigma_second + sigma_first);

                    depth_second_cov.at<float>(x + fxy.y, y + fxy.x) = sigma_fuse;
                    depth_second.at<float>(x + fxy.y, y + fxy.x) = depth_fuse;
                }
            }
            flag = 0;
        }

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudlocal(new pcl::PointCloud<pcl::PointXYZRGB>);
        cloudlocal = cloudFromDepthRGB(leftim, depth_second, 4);
        std::vector<int> index;
        pcl::removeNaNFromPointCloud(*cloudlocal, *cloudlocal, index);
        pcl::transformPointCloud(*cloudlocal, *cloudlocal, T_cam_car);
        pcl::transformPointCloud(*cloudlocal, *cloudlocal, T);

        *Clouds += *cloudlocal;
    }
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr tmp(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> statistical_filter;
    statistical_filter.setMeanK(50);
    statistical_filter.setStddevMulThresh(1.0);
    statistical_filter.setInputCloud(Clouds);
    statistical_filter.filter(*tmp);

    auto end = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "The exporting costs " << double(duration.count())
                 * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den
              << " seconds" << std::endl;

    std::cout << "Saving cloud_elas_ref_half_SOR.ply... (" << static_cast<int>(tmp->size()) << " points)" << std::endl;
    pcl::PLYWriter writer;
    writer.write("cloud_elas_ref_half_SOR.ply", *tmp);
    std::cout << "Saving cloud_elas_ref_half_SOR.ply... done!" << std::endl;

    return 0;
}
