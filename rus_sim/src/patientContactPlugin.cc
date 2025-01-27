// ===================================================================
// file name:    patientContactPlugin.cc
// description:  detect patient mesh collision and publish to ROS topic
// author:       Xihan Ma
// date:         2024-01-29
// ===================================================================

#include "patientContactPlugin.hh"

using namespace gazebo;

GZ_REGISTER_SENSOR_PLUGIN(patientContactPlugin)

/////////////////////////////////////////////////
patientContactPlugin::patientContactPlugin() : SensorPlugin()
{
}

/////////////////////////////////////////////////
patientContactPlugin::~patientContactPlugin()
{
}

/////////////////////////////////////////////////
void patientContactPlugin::Load(sensors::SensorPtr _sensor, sdf::ElementPtr /*_sdf*/)
{
  // create ROS node
  if (!ros::isInitialized())
  {
    int argc = 0;
    char **argv = NULL;
    ros::init(argc, argv, "patient_contact_sensor_plugin");
  }
  ros::NodeHandle nh_;
  this->patientContactPosPub = nh_.advertise<geometry_msgs::Vector3>("/patient/contact/position", 10);
  this->patientContactNormPub = nh_.advertise<geometry_msgs::Vector3>("/patient/contact/normal", 10);

  // Get the parent sensor.
  this->parentSensor =
    std::dynamic_pointer_cast<sensors::ContactSensor>(_sensor);

  // Make sure the parent sensor is valid.
  if (!this->parentSensor)
  {
    gzerr << "patientContactPlugin requires a ContactSensor.\n";
    return;
  }

  // Connect to the sensor update event.
  this->updateConnection = this->parentSensor->ConnectUpdated(
      std::bind(&patientContactPlugin::OnUpdate, this));

  // Make sure the parent sensor is active.
  this->parentSensor->SetActive(true);
}

/////////////////////////////////////////////////
void patientContactPlugin::OnUpdate()
{
  // Get all the contacts & publish to ROS topics
  msgs::Contacts contacts;
  contacts = this->parentSensor->Contacts();

  geometry_msgs::Vector3 contactPos;
  geometry_msgs::Vector3 contactNorm;

  contactPos.x = -1; contactPos.y = -1; contactPos.z = -1; 
  contactNorm.x = -1; contactNorm.y = -1; contactNorm.z = -1;

  ROS_ASSERT(contacts.contact_size() < 2);  // only detects contact between patient and probe

  for (unsigned int i = 0; i < contacts.contact_size(); ++i)
  {
    // std::cout << "Collision between[" << contacts.contact(i).collision1()
    //           << "] and [" << contacts.contact(i).collision2() << "]\n";

    for (unsigned int j = 0; j < contacts.contact(i).position_size(); ++j)
    {
      // std::cout << j << "  Position:"
      //           << contacts.contact(i).position(j).x() << " "
      //           << contacts.contact(i).position(j).y() << " "
      //           << contacts.contact(i).position(j).z() << "\n";
      // std::cout << "   Normal:"
      //           << contacts.contact(i).normal(j).x() << " "
      //           << contacts.contact(i).normal(j).y() << " "
      //           << contacts.contact(i).normal(j).z() << "\n";
      // std::cout << "   Depth:" << contacts.contact(i).depth(j) << "\n";

      contactPos.x = contacts.contact(i).position(j).x();
      contactPos.y = contacts.contact(i).position(j).y();
      contactPos.z = contacts.contact(i).position(j).z();
    
      contactNorm.x = contacts.contact(i).normal(j).x();
      contactNorm.y = contacts.contact(i).normal(j).y();
      contactNorm.z = contacts.contact(i).normal(j).z();
    }
  }

  this->patientContactPosPub.publish(contactPos);
  this->patientContactNormPub.publish(contactNorm);
}