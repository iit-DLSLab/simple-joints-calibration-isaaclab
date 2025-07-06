// generated from rosidl_typesupport_fastrtps_c/resource/idl__type_support_c.cpp.em
// with input from dls2_msgs:msg/TrajectoryGeneratorMsg.idl
// generated code does not contain a copyright notice
#include "dls2_msgs/msg/detail/trajectory_generator_msg__rosidl_typesupport_fastrtps_c.h"


#include <cassert>
#include <limits>
#include <string>
#include "rosidl_typesupport_fastrtps_c/identifier.h"
#include "rosidl_typesupport_fastrtps_c/wstring_conversion.hpp"
#include "rosidl_typesupport_fastrtps_cpp/message_type_support.h"
#include "dls2_msgs/msg/rosidl_typesupport_fastrtps_c__visibility_control.h"
#include "dls2_msgs/msg/detail/trajectory_generator_msg__struct.h"
#include "dls2_msgs/msg/detail/trajectory_generator_msg__functions.h"
#include "fastcdr/Cdr.h"

#ifndef _WIN32
# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-Wunused-parameter"
# ifdef __clang__
#  pragma clang diagnostic ignored "-Wdeprecated-register"
#  pragma clang diagnostic ignored "-Wreturn-type-c-linkage"
# endif
#endif
#ifndef _WIN32
# pragma GCC diagnostic pop
#endif

// includes and forward declarations of message dependencies and their conversion functions

#if defined(__cplusplus)
extern "C"
{
#endif

#include "rosidl_runtime_c/string.h"  // frame_id
#include "rosidl_runtime_c/string_functions.h"  // frame_id

// forward declare type support functions


using _TrajectoryGeneratorMsg__ros_msg_type = dls2_msgs__msg__TrajectoryGeneratorMsg;

static bool _TrajectoryGeneratorMsg__cdr_serialize(
  const void * untyped_ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  const _TrajectoryGeneratorMsg__ros_msg_type * ros_message = static_cast<const _TrajectoryGeneratorMsg__ros_msg_type *>(untyped_ros_message);
  // Field name: frame_id
  {
    const rosidl_runtime_c__String * str = &ros_message->frame_id;
    if (str->capacity == 0 || str->capacity <= str->size) {
      fprintf(stderr, "string capacity not greater than size\n");
      return false;
    }
    if (str->data[str->size] != '\0') {
      fprintf(stderr, "string not null-terminated\n");
      return false;
    }
    cdr << str->data;
  }

  // Field name: sequence_id
  {
    cdr << ros_message->sequence_id;
  }

  // Field name: timestamp
  {
    cdr << ros_message->timestamp;
  }

  // Field name: com_position
  {
    size_t size = 3;
    auto array_ptr = ros_message->com_position;
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: com_orientation
  {
    size_t size = 4;
    auto array_ptr = ros_message->com_orientation;
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: com_linear_velocity
  {
    size_t size = 3;
    auto array_ptr = ros_message->com_linear_velocity;
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: com_angular_velocity
  {
    size_t size = 3;
    auto array_ptr = ros_message->com_angular_velocity;
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: com_linear_acceleration
  {
    size_t size = 3;
    auto array_ptr = ros_message->com_linear_acceleration;
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: com_angular_acceleration
  {
    size_t size = 3;
    auto array_ptr = ros_message->com_angular_acceleration;
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: joints_position
  {
    size_t size = 12;
    auto array_ptr = ros_message->joints_position;
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: joints_velocity
  {
    size_t size = 12;
    auto array_ptr = ros_message->joints_velocity;
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: joints_acceleration
  {
    size_t size = 12;
    auto array_ptr = ros_message->joints_acceleration;
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: joints_effort
  {
    size_t size = 12;
    auto array_ptr = ros_message->joints_effort;
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: wrench
  {
    size_t size = 6;
    auto array_ptr = ros_message->wrench;
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: stance_legs
  {
    size_t size = 4;
    auto array_ptr = ros_message->stance_legs;
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: nominal_touch_down
  {
    size_t size = 12;
    auto array_ptr = ros_message->nominal_touch_down;
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: touch_down
  {
    size_t size = 12;
    auto array_ptr = ros_message->touch_down;
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: swing_period
  {
    size_t size = 4;
    auto array_ptr = ros_message->swing_period;
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: normal_force_max
  {
    size_t size = 4;
    auto array_ptr = ros_message->normal_force_max;
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: normal_force_min
  {
    size_t size = 4;
    auto array_ptr = ros_message->normal_force_min;
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: kp
  {
    size_t size = 12;
    auto array_ptr = ros_message->kp;
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: kd
  {
    size_t size = 12;
    auto array_ptr = ros_message->kd;
    cdr.serializeArray(array_ptr, size);
  }

  return true;
}

static bool _TrajectoryGeneratorMsg__cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  void * untyped_ros_message)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  _TrajectoryGeneratorMsg__ros_msg_type * ros_message = static_cast<_TrajectoryGeneratorMsg__ros_msg_type *>(untyped_ros_message);
  // Field name: frame_id
  {
    std::string tmp;
    cdr >> tmp;
    if (!ros_message->frame_id.data) {
      rosidl_runtime_c__String__init(&ros_message->frame_id);
    }
    bool succeeded = rosidl_runtime_c__String__assign(
      &ros_message->frame_id,
      tmp.c_str());
    if (!succeeded) {
      fprintf(stderr, "failed to assign string into field 'frame_id'\n");
      return false;
    }
  }

  // Field name: sequence_id
  {
    cdr >> ros_message->sequence_id;
  }

  // Field name: timestamp
  {
    cdr >> ros_message->timestamp;
  }

  // Field name: com_position
  {
    size_t size = 3;
    auto array_ptr = ros_message->com_position;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: com_orientation
  {
    size_t size = 4;
    auto array_ptr = ros_message->com_orientation;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: com_linear_velocity
  {
    size_t size = 3;
    auto array_ptr = ros_message->com_linear_velocity;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: com_angular_velocity
  {
    size_t size = 3;
    auto array_ptr = ros_message->com_angular_velocity;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: com_linear_acceleration
  {
    size_t size = 3;
    auto array_ptr = ros_message->com_linear_acceleration;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: com_angular_acceleration
  {
    size_t size = 3;
    auto array_ptr = ros_message->com_angular_acceleration;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: joints_position
  {
    size_t size = 12;
    auto array_ptr = ros_message->joints_position;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: joints_velocity
  {
    size_t size = 12;
    auto array_ptr = ros_message->joints_velocity;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: joints_acceleration
  {
    size_t size = 12;
    auto array_ptr = ros_message->joints_acceleration;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: joints_effort
  {
    size_t size = 12;
    auto array_ptr = ros_message->joints_effort;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: wrench
  {
    size_t size = 6;
    auto array_ptr = ros_message->wrench;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: stance_legs
  {
    size_t size = 4;
    auto array_ptr = ros_message->stance_legs;
    for (size_t i = 0; i < size; ++i) {
      uint8_t tmp;
      cdr >> tmp;
      array_ptr[i] = tmp ? true : false;
    }
  }

  // Field name: nominal_touch_down
  {
    size_t size = 12;
    auto array_ptr = ros_message->nominal_touch_down;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: touch_down
  {
    size_t size = 12;
    auto array_ptr = ros_message->touch_down;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: swing_period
  {
    size_t size = 4;
    auto array_ptr = ros_message->swing_period;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: normal_force_max
  {
    size_t size = 4;
    auto array_ptr = ros_message->normal_force_max;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: normal_force_min
  {
    size_t size = 4;
    auto array_ptr = ros_message->normal_force_min;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: kp
  {
    size_t size = 12;
    auto array_ptr = ros_message->kp;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: kd
  {
    size_t size = 12;
    auto array_ptr = ros_message->kd;
    cdr.deserializeArray(array_ptr, size);
  }

  return true;
}  // NOLINT(readability/fn_size)

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_dls2_msgs
size_t get_serialized_size_dls2_msgs__msg__TrajectoryGeneratorMsg(
  const void * untyped_ros_message,
  size_t current_alignment)
{
  const _TrajectoryGeneratorMsg__ros_msg_type * ros_message = static_cast<const _TrajectoryGeneratorMsg__ros_msg_type *>(untyped_ros_message);
  (void)ros_message;
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;

  // field.name frame_id
  current_alignment += padding +
    eprosima::fastcdr::Cdr::alignment(current_alignment, padding) +
    (ros_message->frame_id.size + 1);
  // field.name sequence_id
  {
    size_t item_size = sizeof(ros_message->sequence_id);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name timestamp
  {
    size_t item_size = sizeof(ros_message->timestamp);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name com_position
  {
    size_t array_size = 3;
    auto array_ptr = ros_message->com_position;
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name com_orientation
  {
    size_t array_size = 4;
    auto array_ptr = ros_message->com_orientation;
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name com_linear_velocity
  {
    size_t array_size = 3;
    auto array_ptr = ros_message->com_linear_velocity;
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name com_angular_velocity
  {
    size_t array_size = 3;
    auto array_ptr = ros_message->com_angular_velocity;
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name com_linear_acceleration
  {
    size_t array_size = 3;
    auto array_ptr = ros_message->com_linear_acceleration;
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name com_angular_acceleration
  {
    size_t array_size = 3;
    auto array_ptr = ros_message->com_angular_acceleration;
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name joints_position
  {
    size_t array_size = 12;
    auto array_ptr = ros_message->joints_position;
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name joints_velocity
  {
    size_t array_size = 12;
    auto array_ptr = ros_message->joints_velocity;
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name joints_acceleration
  {
    size_t array_size = 12;
    auto array_ptr = ros_message->joints_acceleration;
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name joints_effort
  {
    size_t array_size = 12;
    auto array_ptr = ros_message->joints_effort;
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name wrench
  {
    size_t array_size = 6;
    auto array_ptr = ros_message->wrench;
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name stance_legs
  {
    size_t array_size = 4;
    auto array_ptr = ros_message->stance_legs;
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name nominal_touch_down
  {
    size_t array_size = 12;
    auto array_ptr = ros_message->nominal_touch_down;
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name touch_down
  {
    size_t array_size = 12;
    auto array_ptr = ros_message->touch_down;
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name swing_period
  {
    size_t array_size = 4;
    auto array_ptr = ros_message->swing_period;
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name normal_force_max
  {
    size_t array_size = 4;
    auto array_ptr = ros_message->normal_force_max;
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name normal_force_min
  {
    size_t array_size = 4;
    auto array_ptr = ros_message->normal_force_min;
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name kp
  {
    size_t array_size = 12;
    auto array_ptr = ros_message->kp;
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name kd
  {
    size_t array_size = 12;
    auto array_ptr = ros_message->kd;
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  return current_alignment - initial_alignment;
}

static uint32_t _TrajectoryGeneratorMsg__get_serialized_size(const void * untyped_ros_message)
{
  return static_cast<uint32_t>(
    get_serialized_size_dls2_msgs__msg__TrajectoryGeneratorMsg(
      untyped_ros_message, 0));
}

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_dls2_msgs
size_t max_serialized_size_dls2_msgs__msg__TrajectoryGeneratorMsg(
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

  // member: frame_id
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
  // member: sequence_id
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }
  // member: timestamp
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: com_position
  {
    size_t array_size = 3;

    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: com_orientation
  {
    size_t array_size = 4;

    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: com_linear_velocity
  {
    size_t array_size = 3;

    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: com_angular_velocity
  {
    size_t array_size = 3;

    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: com_linear_acceleration
  {
    size_t array_size = 3;

    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: com_angular_acceleration
  {
    size_t array_size = 3;

    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: joints_position
  {
    size_t array_size = 12;

    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: joints_velocity
  {
    size_t array_size = 12;

    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: joints_acceleration
  {
    size_t array_size = 12;

    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: joints_effort
  {
    size_t array_size = 12;

    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: wrench
  {
    size_t array_size = 6;

    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: stance_legs
  {
    size_t array_size = 4;

    current_alignment += array_size * sizeof(uint8_t);
  }
  // member: nominal_touch_down
  {
    size_t array_size = 12;

    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: touch_down
  {
    size_t array_size = 12;

    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: swing_period
  {
    size_t array_size = 4;

    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: normal_force_max
  {
    size_t array_size = 4;

    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: normal_force_min
  {
    size_t array_size = 4;

    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: kp
  {
    size_t array_size = 12;

    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: kd
  {
    size_t array_size = 12;

    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  return current_alignment - initial_alignment;
}

static size_t _TrajectoryGeneratorMsg__max_serialized_size(char & bounds_info)
{
  bool full_bounded;
  bool is_plain;
  size_t ret_val;

  ret_val = max_serialized_size_dls2_msgs__msg__TrajectoryGeneratorMsg(
    full_bounded, is_plain, 0);

  bounds_info =
    is_plain ? ROSIDL_TYPESUPPORT_FASTRTPS_PLAIN_TYPE :
    full_bounded ? ROSIDL_TYPESUPPORT_FASTRTPS_BOUNDED_TYPE : ROSIDL_TYPESUPPORT_FASTRTPS_UNBOUNDED_TYPE;
  return ret_val;
}


static message_type_support_callbacks_t __callbacks_TrajectoryGeneratorMsg = {
  "dls2_msgs::msg",
  "TrajectoryGeneratorMsg",
  _TrajectoryGeneratorMsg__cdr_serialize,
  _TrajectoryGeneratorMsg__cdr_deserialize,
  _TrajectoryGeneratorMsg__get_serialized_size,
  _TrajectoryGeneratorMsg__max_serialized_size
};

static rosidl_message_type_support_t _TrajectoryGeneratorMsg__type_support = {
  rosidl_typesupport_fastrtps_c__identifier,
  &__callbacks_TrajectoryGeneratorMsg,
  get_message_typesupport_handle_function,
};

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, dls2_msgs, msg, TrajectoryGeneratorMsg)() {
  return &_TrajectoryGeneratorMsg__type_support;
}

#if defined(__cplusplus)
}
#endif
