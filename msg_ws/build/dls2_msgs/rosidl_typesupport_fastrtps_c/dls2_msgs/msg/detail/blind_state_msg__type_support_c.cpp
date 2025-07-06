// generated from rosidl_typesupport_fastrtps_c/resource/idl__type_support_c.cpp.em
// with input from dls2_msgs:msg/BlindStateMsg.idl
// generated code does not contain a copyright notice
#include "dls2_msgs/msg/detail/blind_state_msg__rosidl_typesupport_fastrtps_c.h"


#include <cassert>
#include <limits>
#include <string>
#include "rosidl_typesupport_fastrtps_c/identifier.h"
#include "rosidl_typesupport_fastrtps_c/wstring_conversion.hpp"
#include "rosidl_typesupport_fastrtps_cpp/message_type_support.h"
#include "dls2_msgs/msg/rosidl_typesupport_fastrtps_c__visibility_control.h"
#include "dls2_msgs/msg/detail/blind_state_msg__struct.h"
#include "dls2_msgs/msg/detail/blind_state_msg__functions.h"
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

#include "rosidl_runtime_c/string.h"  // frame_id, joints_name, robot_name
#include "rosidl_runtime_c/string_functions.h"  // frame_id, joints_name, robot_name

// forward declare type support functions


using _BlindStateMsg__ros_msg_type = dls2_msgs__msg__BlindStateMsg;

static bool _BlindStateMsg__cdr_serialize(
  const void * untyped_ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  const _BlindStateMsg__ros_msg_type * ros_message = static_cast<const _BlindStateMsg__ros_msg_type *>(untyped_ros_message);
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

  // Field name: robot_name
  {
    const rosidl_runtime_c__String * str = &ros_message->robot_name;
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

  // Field name: joints_name
  {
    size_t size = 12;
    auto array_ptr = ros_message->joints_name;
    for (size_t i = 0; i < size; ++i) {
      const rosidl_runtime_c__String * str = &array_ptr[i];
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

  // Field name: joints_temperature
  {
    size_t size = 12;
    auto array_ptr = ros_message->joints_temperature;
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: feet_contact
  {
    size_t size = 4;
    auto array_ptr = ros_message->feet_contact;
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: current_feet_positions
  {
    size_t size = 12;
    auto array_ptr = ros_message->current_feet_positions;
    cdr.serializeArray(array_ptr, size);
  }

  return true;
}

static bool _BlindStateMsg__cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  void * untyped_ros_message)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  _BlindStateMsg__ros_msg_type * ros_message = static_cast<_BlindStateMsg__ros_msg_type *>(untyped_ros_message);
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

  // Field name: robot_name
  {
    std::string tmp;
    cdr >> tmp;
    if (!ros_message->robot_name.data) {
      rosidl_runtime_c__String__init(&ros_message->robot_name);
    }
    bool succeeded = rosidl_runtime_c__String__assign(
      &ros_message->robot_name,
      tmp.c_str());
    if (!succeeded) {
      fprintf(stderr, "failed to assign string into field 'robot_name'\n");
      return false;
    }
  }

  // Field name: joints_name
  {
    size_t size = 12;
    auto array_ptr = ros_message->joints_name;
    for (size_t i = 0; i < size; ++i) {
      std::string tmp;
      cdr >> tmp;
      auto & ros_i = array_ptr[i];
      if (!ros_i.data) {
        rosidl_runtime_c__String__init(&ros_i);
      }
      bool succeeded = rosidl_runtime_c__String__assign(
        &ros_i,
        tmp.c_str());
      if (!succeeded) {
        fprintf(stderr, "failed to assign string into field 'joints_name'\n");
        return false;
      }
    }
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

  // Field name: joints_temperature
  {
    size_t size = 12;
    auto array_ptr = ros_message->joints_temperature;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: feet_contact
  {
    size_t size = 4;
    auto array_ptr = ros_message->feet_contact;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: current_feet_positions
  {
    size_t size = 12;
    auto array_ptr = ros_message->current_feet_positions;
    cdr.deserializeArray(array_ptr, size);
  }

  return true;
}  // NOLINT(readability/fn_size)

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_dls2_msgs
size_t get_serialized_size_dls2_msgs__msg__BlindStateMsg(
  const void * untyped_ros_message,
  size_t current_alignment)
{
  const _BlindStateMsg__ros_msg_type * ros_message = static_cast<const _BlindStateMsg__ros_msg_type *>(untyped_ros_message);
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
  // field.name robot_name
  current_alignment += padding +
    eprosima::fastcdr::Cdr::alignment(current_alignment, padding) +
    (ros_message->robot_name.size + 1);
  // field.name joints_name
  {
    size_t array_size = 12;
    auto array_ptr = ros_message->joints_name;
    for (size_t index = 0; index < array_size; ++index) {
      current_alignment += padding +
        eprosima::fastcdr::Cdr::alignment(current_alignment, padding) +
        (array_ptr[index].size + 1);
    }
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
  // field.name joints_temperature
  {
    size_t array_size = 12;
    auto array_ptr = ros_message->joints_temperature;
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name feet_contact
  {
    size_t array_size = 4;
    auto array_ptr = ros_message->feet_contact;
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name current_feet_positions
  {
    size_t array_size = 12;
    auto array_ptr = ros_message->current_feet_positions;
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  return current_alignment - initial_alignment;
}

static uint32_t _BlindStateMsg__get_serialized_size(const void * untyped_ros_message)
{
  return static_cast<uint32_t>(
    get_serialized_size_dls2_msgs__msg__BlindStateMsg(
      untyped_ros_message, 0));
}

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_dls2_msgs
size_t max_serialized_size_dls2_msgs__msg__BlindStateMsg(
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
  // member: robot_name
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
  // member: joints_name
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
  // member: joints_temperature
  {
    size_t array_size = 12;

    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: feet_contact
  {
    size_t array_size = 4;

    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: current_feet_positions
  {
    size_t array_size = 12;

    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  return current_alignment - initial_alignment;
}

static size_t _BlindStateMsg__max_serialized_size(char & bounds_info)
{
  bool full_bounded;
  bool is_plain;
  size_t ret_val;

  ret_val = max_serialized_size_dls2_msgs__msg__BlindStateMsg(
    full_bounded, is_plain, 0);

  bounds_info =
    is_plain ? ROSIDL_TYPESUPPORT_FASTRTPS_PLAIN_TYPE :
    full_bounded ? ROSIDL_TYPESUPPORT_FASTRTPS_BOUNDED_TYPE : ROSIDL_TYPESUPPORT_FASTRTPS_UNBOUNDED_TYPE;
  return ret_val;
}


static message_type_support_callbacks_t __callbacks_BlindStateMsg = {
  "dls2_msgs::msg",
  "BlindStateMsg",
  _BlindStateMsg__cdr_serialize,
  _BlindStateMsg__cdr_deserialize,
  _BlindStateMsg__get_serialized_size,
  _BlindStateMsg__max_serialized_size
};

static rosidl_message_type_support_t _BlindStateMsg__type_support = {
  rosidl_typesupport_fastrtps_c__identifier,
  &__callbacks_BlindStateMsg,
  get_message_typesupport_handle_function,
};

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, dls2_msgs, msg, BlindStateMsg)() {
  return &_BlindStateMsg__type_support;
}

#if defined(__cplusplus)
}
#endif
