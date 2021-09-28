#include "ros/ros.h"
#include "std_msgs/String.h"
#include <geometry_msgs/Twist.h>
#include <sstream>
#include <string>

//prottype;
class getAndSend{
    public:
    getAndSend(){
        cmd_vel_pub = n.advertise<geometry_msgs::Twist>("chatter", 1000);

        distance_sub = n.subscribe("/camera/tensorflow/distance", 1000, &getAndSend::control_Callback, this);
    }
    
    void control_Callback(const std_msgs::String::ConstPtr &msg)
        {
            
            geometry_msgs::Twist cmd_vel;
            try
            {
                double distance = std::stod(msg->data.c_str());
                if (distance < 500)
                {
                    cmd_vel.linear.x = 0;
                }
                else if (distance > 500)
                {
                    cmd_vel.linear.x = 1;
                }
            }
            catch (const std::exception& e)
            {
                std::cout << e.what() << std::endl;
            }
            //ROS_INFO(cmd_vel.linear.x);
            cmd_vel_pub.publish(cmd_vel);
        }
    
    private:
        ros::NodeHandle n;
        ros::Publisher cmd_vel_pub;
        ros::Subscriber distance_sub;
};






 
int main(int argc, char **argv)
{
    //initialize
    ros::init(argc, argv, "cmd_to_zumo");
    getAndSend GAS;

    //ros::NodeHandle n;
    ros::spin();

    return 0;
}