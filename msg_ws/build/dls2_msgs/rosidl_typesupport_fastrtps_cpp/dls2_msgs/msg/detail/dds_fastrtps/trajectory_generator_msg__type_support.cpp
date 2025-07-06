// generated from rosidl_typesupport_fastrtps_cpp/resource/idl__type_support.cpp.em
// with input from dls2_msgs:msg/TrajectoryGeneratorMsg.idl
// generated code does not contain a copyright notice
#include "dls2_msgs/msg/detail/trajectory_generator_msg__rosidl_typesupport_fastrtps_cpp.hpp"
#include "dls2_msgs/msg/detail/trajectory_generator_msg__struct.hpp"

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
  const dls2_msgs::msg::TrajectoryGeneratorMsg & ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  // Member: frame_id
  cdr << ros_message.frame_id;
  // Member: sequence_id
  cdr << ros_message.sequence_id;
  // Member: timestamp
  cdr << ros_message.timestamp;
  // Member: com_position
  {
    cdr << ros_message.com_position;
  }
  // Member: com_orientation
  {
    cdr << ros_message.com_orientation;
  }
  // Member: com_linear_velocity
  {
    cdr << ros_message.com_linear_velocity;
  }
  // Member: com_angular_velocity
  {
    cdr << ros_message.com_angular_velocity;
  }
  // Member: com_linear_acceleration
  {
    cdr << ros_message.com_linear_acceleration;
  }
  // Member: com_angular_acceleration
  {
    cdr << ros_message.com_angular_acceleration;
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
  // Member: wrench
  {
    cdr << ros_message.wrench;
  }
  // Member: stance_legs
  {
    cdr << ros_message.stance_legs;
  }
  // Member: nominal_touch_down
  {
    cdr << ros_message.nominal_touch_down;
  }
  // Member: touch_down
  {
    cdr << ros_message.touch_down;
  }
  // Member: swing_period
  {
    cdr << ros_message.swing_period;
  }
  // Member: normal_force_max
  {
    cdr << ros_message.normal_force_max;
  }
  // Member: normal_force_min
  {
    cdr << ros_message.normal_force_min;
  }
  // Member: kp
  {
    cdr << ros_message.kp;
  }
  // Member: kd
  {
    cdr << ros_message.kd;
  }
  return true;
}

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_dls2_msgs
cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  dls2_msgs::msg::TrajectoryGeneratorMsg & ros_message)
{
  // Member: frame_id
  cdr >> ros_message.frame_id;

  // Member: sequence_id
  cdr >> ros_message.sequence_id;

  // Member: timestamp
  cdr >> ros_message.timestamp;

  // Member: com_position
  {
    cdr >> ros_message.com_position;
  }

  // Member: com_orientation
  {
    cdr >> ros_message.com_orientation;
  }

  // Member: com_linear_velocity
  {
    cdr >> ros_message.com_linear_velocity;
  }

  // Member: com_angular_velocity
  {
    cdr >> ros_message.com_angular_velocity;
  }

  // Member: com_linear_acceleration
  {
    cdr >> ros_message.com_linear_acceleration;
  }

  // Member: com_angular_acceleration
  {
    cdr >> ros_message.com_angular_acceleration;
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

  // Member: wrench
  {
    cdr >> ros_message.wrench;
  }

  // Member: stance_legs
  {
    cdr >> ros_message.stance_legs;
  }

  // Member: nominal_touch_down
  {
    cdr >> ros_message.nominal_touch_down;
  }

  // Member: touch_down
  {
    cdr >> ros_message.touch_down;
  }

  // Member: swing_period
  {
    cdr >> ros_message.swing_period;
  }

  // Member: normal_force_max
  {
    cdr >> ros_message.normal_force_max;
  }

  // Member: normal_force_min
  {
    cdr >> ros_message.normal_force_min;
  }

  // Member: kp
  {
    cdr >> ros_message.kp;
  }

  // Member: kd
  {
    cdr >> ros_message.kd;
  }

  return true;
}

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_dls2_msgs
get_serialized_size(
  const dls2_msgs::msg::TrajectoryGeneratorMsg & ros_message,
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
  // Member: com_position
  {
    size_t array_size = 3;
    size_t item_size = sizeof(ros_message.com_position[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: com_orientation
  {
    size_t array_size = 4;
    size_t item_size = sizeof(ros_message.com_orientation[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: com_linear_velocity
  {
    size_t array_size = 3;
    size_t item_size = sizeof(ros_message.com_linear_velocity[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: com_angular_velocity
  {
    size_t array_size = 3;
    size_t item_size = sizeof(ros_message.com_angular_velocity[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: com_linear_acceleration
  {
    size_t array_size = 3;
    size_t item_size = sizeof(ros_message.com_linear_acceleration[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: com_angular_acceleration
  {
    size_t array_size = 3;
    size_t item_size = sizeof(ros_message.com_angular_acceleration[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
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
  // Member: wrench
  {
    size_t array_size = 6;
    size_t item_size = sizeof(ros_message.wrench[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: stance_legs
  {
    size_t array_size = 4;
    size_t item_size = sizeof(ros_message.stance_legs[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: nominal_touch_down
  {
    size_t array_size = 12;
    size_t item_size = sizeof(ros_message.nominal_touch_down[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: touch_down
  {
    size_t array_size = 12;
    size_t item_size = sizeof(ros_message.touch_down[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: swing_period
  {
    size_t array_size = 4;
    size_t item_size = sizeof(ros_message.swing_period[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: normal_force_max
  {
    size_t array_size = 4;
    size_t item_size = sizeof(ros_message.normal_force_max[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: normal_force_min
  {
    size_t array_size = 4;
    size_t item_size = sizeof(ros_message.normal_force_min[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: kp
  {
    size_t array_size = 12;
    size_t item_size = sizeof(ros_message.kp[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: kd
  {
    size_t array_size = 12;
    size_t item_size = sizeof(ros_message.kd[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  return current_alignment - initial_alignment;
}

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_dls2_msgs
max_serialized_size_TrajectoryGeneratorMsg(
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

  // Member: com_position
  {
    size_t array_size = 3;

    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: com_orientation
  {
    size_t array_size = 4;

    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: com_linear_velocity
  {
    size_t array_size = 3;

    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: com_angular_velocity
  {
    size_t array_size = 3;

    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: com_linear_acceleration
  {
    size_t array_size = 3;

    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: com_angular_acceleration
  {
    size_t array_size = 3;

    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
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

  // Member: wrench
  {
    size_t array_size = 6;

    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: stance_legs
  {
    size_t array_size = 4;

    current_alignment += array_size * sizeof(uint8_t);
  }

  // Member: nominal_touch_down
  {
    size_t array_size = 12;

    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: touch_down
  {
    size_t array_size = 12;

    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: swing_period
  {
    size_t array_size = 4;

    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: normal_force_max
  {
    size_t array_size = 4;

    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: normal_force_min
  {
    size_t array_size = 4;

    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: kp
  {
    size_t array_size = 12;

    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: kd
  {
    size_t array_size = 12;

    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  return current_alignment - initial_alignment;
}

static bool _TrajectoryGeneratorMsg__cdr_serialize(
  const void * untyped_ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  auto typed_message =
    static_cast<const dls2_msgs::msg::TrajectoryGeneratorMsg *>(
    untyped_ros_message);
  return cdr_serialize(*typed_message, cdr);
}

static bool _TrajectoryGeneratorMsg__cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  void * untyped_ros_message)
{
  auto typed_message =
    static_cast<dls2_msgs::msg::TrajectoryGeneratorMsg *>(
    untyped_ros_message);
  return cdr_deserialize(cdr, *typed_message);
}

static uint32_t _TrajectoryGeneratorMsg__get_serialized_size(
  const void * untyped_ros_message)
{
  auto typed_message =
    static_cast<const dls2_msgs::msg::TrajectoryGeneratorMsg *>(
    untyped_ros_message);
  return static_cast<uint32_t>(get_serialized_size(*typed_message, 0));
}

static size_t _TrajectoryGeneratorMsg__max_serialized_size(char & bounds_info)
{
  bool full_bounded;
  bool is_plain;
  size_t ret_val;

  ret_val = max_serialized_size_TrajectoryGeneratorMsg(full_bounded, is_plain, 0);

  bounds_info =
    is_plain ? ROSIDL_TYPESUPPORT_FASTRTPS_PLAIN_TYPE :
    full_bounded ? ROSIDL_TYPESUPPORT_FASTRTPS_BOUNDED_TYPE : ROSIDL_TYPESUPPORT_FASTRTPS_UNBOUNDED_TYPE;
  return ret_val;
}

static message_type_support_callbacks_t _TrajectoryGeneratorMsg__callbacks = {
  "dls2_msgs::msg",
  "TrajectoryGeneratorMsg",
  _TrajectoryGeneratorMsg__cdr_serialize,
  _TrajectoryGeneratorMsg__cdr_deserialize,
  _TrajectoryGeneratorMsg__get_serialized_size,
  _TrajectoryGeneratorMsg__max_serialized_size
};

static rosidl_message_type_support_t _TrajectoryGeneratorMsg__handle = {
  rosidl_typesupport_fastrtps_cpp::typesupport_identifier,
  &_TrajectoryGeneratorMsg__callbacks,
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
get_message_type_support_handle<dls2_msgs::msg::TrajectoryGeneratorMsg>()
{
  return &dls2_msgs::msg::typesupport_fastrtps_cpp::_TrajectoryGeneratorMsg__handle;
}

}  // namespace rosidl_typesupport_fastrtps_cpp

#ifdef __cplusplus
extern "C"
{
#endif

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_cpp, dls2_msgs, msg, TrajectoryGeneratorMsg)() {
  return &dls2_msgs::msg::typesupport_fastrtps_cpp::_TrajectoryGeneratorMsg__handle;
}

#ifdef __cplusplus
}
#endif
