// ROS bridge
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#include <torch/extension.h>

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <openpose_ros_msgs/BoundingBox.h>
#include <openpose_ros_msgs/OpenPoseHuman.h>
#include <openpose_ros_msgs/OpenPoseHumanList.h>
#include <openpose_ros_msgs/PointWithProb.h>

#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>

#include <eigen_conversions/eigen_msg.h>

#include <string.h>

#include <mutex>
#include <condition_variable>

std::shared_ptr<ros::AsyncSpinner> g_spinner;

sensor_msgs::ImageConstPtr g_rgb;
std::mutex g_mutex;
std::condition_variable g_conditionVariable;

std::shared_ptr<ros::NodeHandle> g_nh;

ros::Subscriber g_sub_rgb;

ros::Publisher g_pub_skeleton;

const float g_conf_threshold = 0.10;

struct ImageHolder
{
    sensor_msgs::ImageConstPtr rgb;
};

void handleFrame(const sensor_msgs::ImageConstPtr& rgb)
{
    std::unique_lock<std::mutex> lock(g_mutex);
    g_rgb = rgb;
    g_conditionVariable.notify_all();
}

void init()
{
    char* argv[] = {
        strdup("human_pose_estimation")
    };
    int argc = 1;

    ros::init(argc, argv, "human_pose_estimation", ros::init_options::NoSigintHandler);

    g_nh.reset(new ros::NodeHandle("~"));

    g_sub_rgb = g_nh->subscribe("/cam_1/color/image_raw", 1, &handleFrame);

    g_pub_skeleton = g_nh->advertise<openpose_ros_msgs::OpenPoseHumanList>("human_list/cam_1", 1, true);

    g_spinner.reset(new ros::AsyncSpinner(1));
    g_spinner->start();
}

py::tuple waitForFrame()
{
    while(1)
    {
        sensor_msgs::ImageConstPtr rgb;
        {
            std::unique_lock<std::mutex> lock(g_mutex);
            while(!g_rgb)
                g_conditionVariable.wait(lock);

            rgb = g_rgb;
            g_rgb.reset();
        }
        
        //ROS_INFO("Received RGB_frame: %dx%d", rgb->height, rgb->width);
        auto color = torch::empty({rgb->height, rgb->width, 3}, torch::kByte);
        
        memcpy(color.data<uint8_t>(), rgb->data.data(), rgb->data.size());

        return py::make_tuple(color, ImageHolder{rgb});
    }
}

void publishPose(at::Tensor coords, at::Tensor conf, const ImageHolder& holder)
{
    if(coords.type() != torch::CPU(at::kFloat))
    {
        ROS_ERROR("coords tensor must be CPU - float");
        return;
    }

    if(coords.dim() != 2 || coords.size(1) != 2)
    {
        ROS_ERROR("coords tensor must be Nx2");
        return;
    }

    if(!coords.is_contiguous())
    {
        ROS_ERROR("coords tensor must be contiguous");
        return;
    }
    
    if(conf.type() != torch::CPU(at::kFloat))
    {
        ROS_ERROR("conf tensor must be CPU - float");
        return;
    }

    if(conf.dim() != 2 || conf.size(1) != 1 || conf.size(0) != coords.size(0))
    {
        ROS_ERROR("conf tensor must be Nx1");
        return;
    }

    if(!conf.is_contiguous())
    {
        ROS_ERROR("conf tensor must be contiguous");
        return;
    }
    
    auto n_joints = coords.size(0);
    
    auto coords_a = coords.accessor<float,2>(); // Nx2
    auto conf_a = conf.accessor<float,2>(); // Nx1

    openpose_ros_msgs::OpenPoseHumanList human_list_msg;
    //human_list_msg.header.stamp = ros::Time::now();
    human_list_msg.header.stamp = holder.rgb->header.stamp;
    human_list_msg.image_header = holder.rgb->header;
    human_list_msg.num_humans = 1; // single person detecor (TODO: 0 is also possible..)
        
    std::vector<openpose_ros_msgs::OpenPoseHuman> human_list(human_list_msg.num_humans);
    
    openpose_ros_msgs::OpenPoseHuman human;
    
    double body_min_x = -1;
    double body_max_x = -1;
    double body_min_y = -1;
    double body_max_y = -1;

    int num_body_key_points_with_non_zero_prob = 0;
    for (auto bodyPart = 0 ; bodyPart < n_joints ; bodyPart++)
    {
        if(conf_a[bodyPart][0] < g_conf_threshold)
            continue;
        
        openpose_ros_msgs::PointWithProb body_point_with_prob;
        float x = coords_a[bodyPart][0];
        float y = coords_a[bodyPart][1];
        float prob = conf_a[bodyPart][0];
        body_point_with_prob.x = x;
        body_point_with_prob.y = y;
        body_point_with_prob.prob = prob;

        num_body_key_points_with_non_zero_prob++;

        if(body_min_x == -1 || body_point_with_prob.x < body_min_x)
        {
            body_min_x = body_point_with_prob.x;
        }
        if(body_point_with_prob.x > body_max_x)
        {
            body_max_x = body_point_with_prob.x;
        }

        if(body_min_y == -1 || body_point_with_prob.y < body_min_y)
        {
            body_min_y = body_point_with_prob.y;
        }
        if(body_point_with_prob.y > body_max_y)
        {
            body_max_y = body_point_with_prob.y;
        }
        
        human.body_key_points_with_prob.at(bodyPart) = body_point_with_prob;
    }
    
    human.num_body_key_points_with_non_zero_prob = num_body_key_points_with_non_zero_prob;
    human.body_bounding_box.x = body_min_x;
    human.body_bounding_box.y = body_min_y;
    human.body_bounding_box.width = body_max_x - body_min_x;
    human.body_bounding_box.height = body_max_y - body_min_y;
    
    human_list.at(0) = human;
    
    human_list_msg.human_list = human_list;

    g_pub_skeleton.publish(human_list_msg);
}

static int unused; // the capsule needs something to reference
py::capsule cleanup(&unused, [](void *) {
    ROS_INFO("shutting down...");
    if(g_spinner)
        g_spinner->stop();
});

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("init", &init, "Init bridge");
    m.def("wait_for_frame", &waitForFrame, "Wait for input frame");
    m.def("publish_pose", &publishPose, "Publish a human skeleton pose");
    py::class_<ImageHolder>(m, "ImageMsgHolder");
    m.add_object("_cleanup", cleanup);
}
