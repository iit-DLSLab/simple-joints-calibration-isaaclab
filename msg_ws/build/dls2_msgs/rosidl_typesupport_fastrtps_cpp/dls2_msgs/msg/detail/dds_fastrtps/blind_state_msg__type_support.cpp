// generated from rosidl_typesupport_fastrtps_cpp/resource/idl__type_support.cpp.em
// with input from dls2_msgs:msg/BlindStateMsg.idl
// generated code does not contain a copyright notice
#include "dls2_msgs/msg/detail/blind_state_msg__rosidl_typesupport_fastrtps_cpp.hpp"
#include "dls2_msgs/msg/detail/blind_state_msg__struct.hpp"

#include <limits>
#include <stdexcept>
#include <string>
#include "rosidl_typesupport_cpp/message_type_support.hpp"
#include "rosidl_typesupport_fastrtps_cpp/identifier.hpp"
#include "rosidl_typesupport_fastrtps_cpp/message_type_support.h"
#include "rosidl_typesupport_fastrtps_cpp/message_type_support_decl.hpp"
#include "rosidl_typesupport_fastrtps_cpp/wstring_conversion.hpp"
#include "fastcdr/Cdr.h"


// forward declaration of message dependencies and their conversion functions

namespace dls2_msgs
{

namespace msg
{

namespace typesupport_fastrtps_cpp
{

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_dls2_msgs
cdr_serialize(
  const dls2_msgs::msg::BlindStateMsg & ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  // Member: frame_id
  cdr << ros_message.frame_id;
  // Member: sequence_id
  cdr << ros_message.sequence_id;
  // Member: timestamp
  cdr << ros_message.timestamp;
  // Member: robot_name
  cdr << ros_message.robot_name;
  // Member: joints_name
  {
    cdr << ros_message.joints_name;
  }
  // Member: joints_position
  {
    cdr << ros_message.joints_position;
  }
  // Member: joints_velocity
  {
    cdr << ros_message.joints_velocity;
  }
  // Member: joints_acceleration
  {
    cdr << ros_message.joints_acceleration;
  }
  // Member: joints_effort
  {
    cdr << ros_message.joints_effort;
  }
  // Member: joints_temperature
  {
    cdr << ros_message.joints_temperature;
  }
  // Member: feet_contact
  {
    cdr << ros_message.feet_contact;
  }
  // Member: current_feet_positions
  {
    cdr << ros_message.current_feet_positions;
  }
  return true;
}

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_dls2_msgs
cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  dls2_msgs::msg::BlindStateMsg & ros_message)
{
  // Member: frame_id
  cdr >> ros_message.frame_id;

  // Member: sequence_id
  cdr >> ros_message.sequence_id;

  // Member: timestamp
  cdr >> ros_message.timestamp;

  // Member: robot_name
  cdr >> ros_message.robot_name;

  // Member: joints_name
  {
    cdr >> ros_message.joints_name;
  }

  // Member: joints_position
  {
    cdr >> ros_message.joints_position;
  }

  // Member: joints_velocity
  {
    cdr >> ros_message.joints_velocity;
  }

  // Member: joints_acceleration
  {
    cdr >> ros_message.joints_acceleration;
  }

  // Member: joints_effort
  {
    cdr >> ros_message.joints_effort;
  }

  // Member: joints_temperature
  {
    cdr >> ros_message.joints_temperature;
  }

  // Member: feet_contact
  {
    cdr >> ros_message.feet_contact;
  }

  // Member: current_feet_positions
  {
    cdr >> ros_message.current_feet_positions;
  }

  return true;
}

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_dls2_msgs
get_serialized_size(
  const dls2_msgs::msg::BlindStateMsg & ros_message,
  size_t current_alignment)
{
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;

  // Member: frame_id
  current_alignment += padding +
    eprosima::fastcdr::Cdr::alignment(current_alignment, padding) +
    (ros_message.frame_id.size() + 1);
  // Member: sequence_id
  {
    size_t item_size = sizeof(ros_message.sequence_id);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: timestamp
  {
    size_t item_size = sizeof(ros_message.timestamp);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: robot_name
  current_alignment += padding +
    eprosima::fastcdr::Cdr::alignment(current_alignment, padding) +
    (ros_message.robot_name.size() + 1);
  // Member: joints_name
  {
    size_t array_size = 12;
    for (size_t index = 0; index < array_size; ++index) {
      current_alignment += padding +
        eprosima::fastcdr::Cdr::alignment(current_alignment, padding) +
        (ros_message.joints_name[index].size() + 1);
    }
  }
  // Member: joints_position
  {
    size_t array_size = 12;
    size_t item_size = sizeof(ros_message.joints_position[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: joints_velocity
  {
    size_t array_size = 12;
    size_t item_size = sizeof(ros_message.joints_velocity[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: joints_acceleration
  {
    size_t array_size = 12;
    size_t item_size = sizeof(ros_message.joints_acceleration[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: joints_effort
  {
    size_t array_size = 12;
    size_t item_size = sizeof(ros_message.joints_effort[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: joints_temperature
  {
    size_t array_size = 12;
    size_t item_size = sizeof(ros_message.joints_temperature[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: feet_contact
  {
    size_t array_size = 4;
    size_t item_size = sizeof(ros_message.feet_contact[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: current_feet_positions
  {
    size_t array_size = 12;
    size_t item_size = sizeof(ros_message.current_feet_positions[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  return current_alignment - initial_alignment;
}

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_dls2_msgs
max_serialized_size_BlindStateMsg(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment)
{
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;

  full_bounded = true;
  is_plain = true;


  // Member: frame_id
  {
    size_t array_size = 1;

    full_bounded = false;
    is_plain = false;
    for (size_t index = 0; index < array_size; ++index) {
      current_alignment += padding +
        eprosima::fastcdr::Cdr::alignment(current_alignment, padding) +
        1;
    }
  }

  // Member: sequence_id
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }

  // Member: timestamp
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: robot_name
  {
    size_t array_size = 1;

    full_bounded = false;
    is_plain = false;
    for (size_t index = 0; index < array_size; ++index) {
      current_alignment += padding +
        eprosima::fastcdr::Cdr::alignment(current_alignment, padding) +
        1;
    }
  }

  // Member: joints_name
  {
    size_t array_size = 12;

    full_bounded = false;
    is_plain = false;
    for (size_t index = 0; index < array_size; ++index) {
      current_alignment += padding +
        eprosima::fastcdr::Cdr::alignment(current_alignment, padding) +
        1;
    }
  }

  // Member: joints_position
  {
    size_t array_size = 12;

    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: joints_velocity
  {
    size_t array_size = 12;

    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: joints_acceleration
  {
    size_t array_size = 12;

    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: joints_effort
  {
    size_t array_size = 12;

    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: joints_temperature
  {
    size_t array_size = 12;

    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: feet_contact
  {
    size_t array_size = 4;

    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: current_feet_positions
  {
    size_t array_size = 12;

    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  return current_alignment - initial_alignment;
}

static bool _BlindStateMsg__cdr_serialize(
  const void * untyped_ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  auto typed_message =
    static_cast<const dls2_msgs::msg::BlindStateMsg *>(
    untyped_ros_message);
  return cdr_serialize(*typed_message, cdr);
}

static bool _BlindStateMsg__cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  void * untyped_ros_message)
{
  auto typed_message =
    static_cast<dls2_msgs::msg::BlindStateMsg *>(
    untyped_ros_message);
  return cdr_deserialize(cdr, *typed_message);
}

static uint32_t _BlindStateMsg__get_serialized_size(
  const void * untyped_ros_message)
{
  auto typed_message =
    static_cast<const dls2_msgs::msg::BlindStateMsg *>(
    untyped_ros_message);
  return static_cast<uint32_t>(get_serialized_size(*typed_message, 0));
}

static size_t _BlindStateMsg__max_serialized_size(char & bounds_info)
{
  bool full_bounded;
  bool is_plain;
  size_t ret_val;

  ret_val = max_serialized_size_BlindStateMsg(full_bounded, is_plain, 0);

  bounds_info =
    is_plain ? ROSIDL_TYPESUPPORT_FASTRTPS_PLAIN_TYPE :
    full_bounded ? ROSIDL_TYPESUPPORT_FASTRTPS_BOUNDED_TYPE : ROSIDL_TYPESUPPORT_FASTRTPS_UNBOUNDED_TYPE;
  return ret_val;
}

static message_type_support_callbacks_t _BlindStateMsg__callbacks = {
  "dls2_msgs::msg",
  "BlindStateMsg",
  _BlindStateMsg__cdr_serialize,
  _BlindStateMsg__cdr_deserialize,
  _BlindStateMsg__get_serialized_size,
  _BlindStateMsg__max_serialized_size
};

static rosidl_message_type_support_t _BlindStateMsg__handle = {
  rosidl_typesupport_fastrtps_cpp::typesupport_identifier,
  &_BlindStateMsg__callbacks,
  get_message_typesupport_handle_function,
};

}  // namespace typesupport_fastrtps_cpp

}  // namespace msg

}  // namespace dls2_msgs

namespace rosidl_typesupport_fastrtps_cpp
{

template<>
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_EXPORT_dls2_msgs
const rosidl_message_type_support_t *
get_message_type_support_handle<dls2_msgs::msg::BlindStateMsg>()
{
  return &dls2_msgs::msg::typesupport_fastrtps_cpp::_BlindStateMsg__handle;
}

}  // namespace rosidl_typesupport_fastrtps_cpp

#ifdef __cplusplus
extern "C"
{
#endif

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_cpp, dls2_msgs, msg, BlindStateMsg)() {
  return &dls2_msgs::msg::typesupport_fastrtps_cpp::_BlindStateMsg__handle;
}

#ifdef __cplusplus
}
#endif
