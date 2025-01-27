// ===================================================================
// file name:    patientContactPlugin.hh
// description:  detect patient mesh collision and publish to ROS topic
// author:       Xihan Ma
// date:         2024-01-29
// ===================================================================

#ifndef _PATIENT_CONTACT_PLUGIN_HH_
#define _PATIENT_CONTACT_PLUGIN_HH_

#include <string>
#include <vector>
#include <ros/ros.h>
#include <geometry_msgs/Vector3.h>
#include <std_msgs/Bool.h>
#include <gazebo/gazebo.hh>
#include <gazebo/sensors/sensors.hh>

namespace gazebo
{
  /// \brief An example plugin for a contact sensor.
  class patientContactPlugin : public SensorPlugin
  {
    /// \brief Constructor.
    public: patientContactPlugin();

    /// \brief Destructor.
    public: virtual ~patientContactPlugin();

    /// \brief Load the sensor plugin.
    /// \param[in] _sensor Pointer to the sensor that loaded this plugin.
    /// \param[in] _sdf SDF element that describes the plugin.
    public: virtual void Load(sensors::SensorPtr _sensor, sdf::ElementPtr _sdf);

    /// \brief Callback that receives the contact sensor's update signal.
    private: virtual void OnUpdate();

    /// \brief Pointer to the contact sensor
    private: sensors::ContactSensorPtr parentSensor;

    /// \brief Connection that maintains a link between the contact sensor's
    /// updated signal and the OnUpdate callback.
    private: event::ConnectionPtr updateConnection;

  private: 
    ros::Publisher patientContactPosPub;
    ros::Publisher patientContactNormPub;
  };
}

#endif