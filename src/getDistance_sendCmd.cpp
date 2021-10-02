#include "ros/ros.h"
#include "std_msgs/String.h"
#include <geometry_msgs/Twist.h>
#include <sstream>
#include <string>

//prottype;
class getAndSend{
    public:
    getAndSend(){
        cmd_vel_pub = n.advertise<geometry_msgs::Twist>("/camera/tensorflow/cmd_vel", 1000);

        distance_sub = n.subscribe("/camera/tensorflow/distance", 1000, &getAndSend::control_Callback, this);
    }
    
    void control_Callback(const std_msgs::String::ConstPtr &msg)
        {
            
            geometry_msgs::Twist cmd_vel;
            try
            {
                double distance = std::stod(msg->data.c_str());
                if (distance < 500 && distance > 100)
                {
                    cmd_vel.linear.x = -1;
                }
                else if (distance > 800)
                {
                    cmd_vel.linear.x = 1;
                }else {
                    cmd_vel.linear.x = 0;     
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
