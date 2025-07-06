# generated from rosidl_generator_py/resource/_idl.py.em
# with input from dls2_msgs:msg/BlindStateMsg.idl
# generated code does not contain a copyright notice


# Import statements for member types

import builtins  # noqa: E402, I100

import math  # noqa: E402, I100

# Member 'joints_position'
# Member 'joints_velocity'
# Member 'joints_acceleration'
# Member 'joints_effort'
# Member 'joints_temperature'
# Member 'feet_contact'
# Member 'current_feet_positions'
import numpy  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_BlindStateMsg(type):
    """Metaclass of message 'BlindStateMsg'."""

    _CREATE_ROS_MESSAGE = None
    _CONVERT_FROM_PY = None
    _CONVERT_TO_PY = None
    _DESTROY_ROS_MESSAGE = None
    _TYPE_SUPPORT = None

    __constants = {
    }

    @classmethod
    def __import_type_support__(cls):
        try:
            from rosidl_generator_py import import_type_support
            module = import_type_support('dls2_msgs')
        except ImportError:
            import logging
            import traceback
            logger = logging.getLogger(
                'dls2_msgs.msg.BlindStateMsg')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__msg__blind_state_msg
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__msg__blind_state_msg
            cls._CONVERT_TO_PY = module.convert_to_py_msg__msg__blind_state_msg
            cls._TYPE_SUPPORT = module.type_support_msg__msg__blind_state_msg
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__msg__blind_state_msg

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class BlindStateMsg(metaclass=Metaclass_BlindStateMsg):
    """Message class 'BlindStateMsg'."""

    __slots__ = [
        '_frame_id',
        '_sequence_id',
        '_timestamp',
        '_robot_name',
        '_joints_name',
        '_joints_position',
        '_joints_velocity',
        '_joints_acceleration',
        '_joints_effort',
        '_joints_temperature',
        '_feet_contact',
        '_current_feet_positions',
    ]

    _fields_and_field_types = {
        'frame_id': 'string',
        'sequence_id': 'uint32',
        'timestamp': 'double',
        'robot_name': 'string',
        'joints_name': 'string[12]',
        'joints_position': 'double[12]',
        'joints_velocity': 'double[12]',
        'joints_acceleration': 'double[12]',
        'joints_effort': 'double[12]',
        'joints_temperature': 'double[12]',
        'feet_contact': 'double[4]',
        'current_feet_positions': 'double[12]',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.UnboundedString(),  # noqa: E501
        rosidl_parser.definition.BasicType('uint32'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.UnboundedString(),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.UnboundedString(), 12),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('double'), 12),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('double'), 12),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('double'), 12),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('double'), 12),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('double'), 12),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('double'), 4),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('double'), 12),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        self.frame_id = kwargs.get('frame_id', str())
        self.sequence_id = kwargs.get('sequence_id', int())
        self.timestamp = kwargs.get('timestamp', float())
        self.robot_name = kwargs.get('robot_name', str())
        self.joints_name = kwargs.get(
            'joints_name',
            [str() for x in range(12)]
        )
        if 'joints_position' not in kwargs:
            self.joints_position = numpy.zeros(12, dtype=numpy.float64)
        else:
            self.joints_position = numpy.array(kwargs.get('joints_position'), dtype=numpy.float64)
            assert self.joints_position.shape == (12, )
        if 'joints_velocity' not in kwargs:
            self.joints_velocity = numpy.zeros(12, dtype=numpy.float64)
        else:
            self.joints_velocity = numpy.array(kwargs.get('joints_velocity'), dtype=numpy.float64)
            assert self.joints_velocity.shape == (12, )
        if 'joints_acceleration' not in kwargs:
            self.joints_acceleration = numpy.zeros(12, dtype=numpy.float64)
        else:
            self.joints_acceleration = numpy.array(kwargs.get('joints_acceleration'), dtype=numpy.float64)
            assert self.joints_acceleration.shape == (12, )
        if 'joints_effort' not in kwargs:
            self.joints_effort = numpy.zeros(12, dtype=numpy.float64)
        else:
            self.joints_effort = numpy.array(kwargs.get('joints_effort'), dtype=numpy.float64)
            assert self.joints_effort.shape == (12, )
        if 'joints_temperature' not in kwargs:
            self.joints_temperature = numpy.zeros(12, dtype=numpy.float64)
        else:
            self.joints_temperature = numpy.array(kwargs.get('joints_temperature'), dtype=numpy.float64)
            assert self.joints_temperature.shape == (12, )
        if 'feet_contact' not in kwargs:
            self.feet_contact = numpy.zeros(4, dtype=numpy.float64)
        else:
            self.feet_contact = numpy.array(kwargs.get('feet_contact'), dtype=numpy.float64)
            assert self.feet_contact.shape == (4, )
        if 'current_feet_positions' not in kwargs:
            self.current_feet_positions = numpy.zeros(12, dtype=numpy.float64)
        else:
            self.current_feet_positions = numpy.array(kwargs.get('current_feet_positions'), dtype=numpy.float64)
            assert self.current_feet_positions.shape == (12, )

    def __repr__(self):
        typename = self.__class__.__module__.split('.')
        typename.pop()
        typename.append(self.__class__.__name__)
        args = []
        for s, t in zip(self.__slots__, self.SLOT_TYPES):
            field = getattr(self, s)
            fieldstr = repr(field)
            # We use Python array type for fields that can be directly stored
            # in them, and "normal" sequences for everything else.  If it is
            # a type that we store in an array, strip off the 'array' portion.
            if (
                isinstance(t, rosidl_parser.definition.AbstractSequence) and
                isinstance(t.value_type, rosidl_parser.definition.BasicType) and
                t.value_type.typename in ['float', 'double', 'int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64']
            ):
                if len(field) == 0:
                    fieldstr = '[]'
                else:
                    assert fieldstr.startswith('array(')
                    prefix = "array('X', "
                    suffix = ')'
                    fieldstr = fieldstr[len(prefix):-len(suffix)]
            args.append(s[1:] + '=' + fieldstr)
        return '%s(%s)' % ('.'.join(typename), ', '.join(args))

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        if self.frame_id != other.frame_id:
            return False
        if self.sequence_id != other.sequence_id:
            return False
        if self.timestamp != other.timestamp:
            return False
        if self.robot_name != other.robot_name:
            return False
        if self.joints_name != other.joints_name:
            return False
        if all(self.joints_position != other.joints_position):
            return False
        if all(self.joints_velocity != other.joints_velocity):
            return False
        if all(self.joints_acceleration != other.joints_acceleration):
            return False
        if all(self.joints_effort != other.joints_effort):
            return False
        if all(self.joints_temperature != other.joints_temperature):
            return False
        if all(self.feet_contact != other.feet_contact):
            return False
        if all(self.current_feet_positions != other.current_feet_positions):
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @builtins.property
    def frame_id(self):
        """Message field 'frame_id'."""
        return self._frame_id

    @frame_id.setter
    def frame_id(self, value):
        if __debug__:
            assert \
                isinstance(value, str), \
                "The 'frame_id' field must be of type 'str'"
        self._frame_id = value

    @builtins.property
    def sequence_id(self):
        """Message field 'sequence_id'."""
        return self._sequence_id

    @sequence_id.setter
    def sequence_id(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'sequence_id' field must be of type 'int'"
            assert value >= 0 and value < 4294967296, \
                "The 'sequence_id' field must be an unsigned integer in [0, 4294967295]"
        self._sequence_id = value

    @builtins.property
    def timestamp(self):
        """Message field 'timestamp'."""
        return self._timestamp

    @timestamp.setter
    def timestamp(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'timestamp' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'timestamp' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._timestamp = value

    @builtins.property
    def robot_name(self):
        """Message field 'robot_name'."""
        return self._robot_name

    @robot_name.setter
    def robot_name(self, value):
        if __debug__:
            assert \
                isinstance(value, str), \
                "The 'robot_name' field must be of type 'str'"
        self._robot_name = value

    @builtins.property
    def joints_name(self):
        """Message field 'joints_name'."""
        return self._joints_name

    @joints_name.setter
    def joints_name(self, value):
        if __debug__:
            from collections.abc import Sequence
            from collections.abc import Set
            from collections import UserList
            from collections import UserString
            assert \
                ((isinstance(value, Sequence) or
                  isinstance(value, Set) or
                  isinstance(value, UserList)) and
                 not isinstance(value, str) and
                 not isinstance(value, UserString) and
                 len(value) == 12 and
                 all(isinstance(v, str) for v in value) and
                 True), \
                "The 'joints_name' field must be a set or sequence with length 12 and each value of type 'str'"
        self._joints_name = value

    @builtins.property
    def joints_position(self):
        """Message field 'joints_position'."""
        return self._joints_position

    @joints_position.setter
    def joints_position(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.float64, \
                "The 'joints_position' numpy.ndarray() must have the dtype of 'numpy.float64'"
            assert value.size == 12, \
                "The 'joints_position' numpy.ndarray() must have a size of 12"
            self._joints_position = value
            return
        if __debug__:
            from collections.abc import Sequence
            from collections.abc import Set
            from collections import UserList
            from collections import UserString
            assert \
                ((isinstance(value, Sequence) or
                  isinstance(value, Set) or
                  isinstance(value, UserList)) and
                 not isinstance(value, str) and
                 not isinstance(value, UserString) and
                 len(value) == 12 and
                 all(isinstance(v, float) for v in value) and
                 all(not (val < -1.7976931348623157e+308 or val > 1.7976931348623157e+308) or math.isinf(val) for val in value)), \
                "The 'joints_position' field must be a set or sequence with length 12 and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._joints_position = numpy.array(value, dtype=numpy.float64)

    @builtins.property
    def joints_velocity(self):
        """Message field 'joints_velocity'."""
        return self._joints_velocity

    @joints_velocity.setter
    def joints_velocity(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.float64, \
                "The 'joints_velocity' numpy.ndarray() must have the dtype of 'numpy.float64'"
            assert value.size == 12, \
                "The 'joints_velocity' numpy.ndarray() must have a size of 12"
            self._joints_velocity = value
            return
        if __debug__:
            from collections.abc import Sequence
            from collections.abc import Set
            from collections import UserList
            from collections import UserString
            assert \
                ((isinstance(value, Sequence) or
                  isinstance(value, Set) or
                  isinstance(value, UserList)) and
                 not isinstance(value, str) and
                 not isinstance(value, UserString) and
                 len(value) == 12 and
                 all(isinstance(v, float) for v in value) and
                 all(not (val < -1.7976931348623157e+308 or val > 1.7976931348623157e+308) or math.isinf(val) for val in value)), \
                "The 'joints_velocity' field must be a set or sequence with length 12 and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._joints_velocity = numpy.array(value, dtype=numpy.float64)

    @builtins.property
    def joints_acceleration(self):
        """Message field 'joints_acceleration'."""
        return self._joints_acceleration

    @joints_acceleration.setter
    def joints_acceleration(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.float64, \
                "The 'joints_acceleration' numpy.ndarray() must have the dtype of 'numpy.float64'"
            assert value.size == 12, \
                "The 'joints_acceleration' numpy.ndarray() must have a size of 12"
            self._joints_acceleration = value
            return
        if __debug__:
            from collections.abc import Sequence
            from collections.abc import Set
            from collections import UserList
            from collections import UserString
            assert \
                ((isinstance(value, Sequence) or
                  isinstance(value, Set) or
                  isinstance(value, UserList)) and
                 not isinstance(value, str) and
                 not isinstance(value, UserString) and
                 len(value) == 12 and
                 all(isinstance(v, float) for v in value) and
                 all(not (val < -1.7976931348623157e+308 or val > 1.7976931348623157e+308) or math.isinf(val) for val in value)), \
                "The 'joints_acceleration' field must be a set or sequence with length 12 and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._joints_acceleration = numpy.array(value, dtype=numpy.float64)

    @builtins.property
    def joints_effort(self):
        """Message field 'joints_effort'."""
        return self._joints_effort

    @joints_effort.setter
    def joints_effort(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.float64, \
                "The 'joints_effort' numpy.ndarray() must have the dtype of 'numpy.float64'"
            assert value.size == 12, \
                "The 'joints_effort' numpy.ndarray() must have a size of 12"
            self._joints_effort = value
            return
        if __debug__:
            from collections.abc import Sequence
            from collections.abc import Set
            from collections import UserList
            from collections import UserString
            assert \
                ((isinstance(value, Sequence) or
                  isinstance(value, Set) or
                  isinstance(value, UserList)) and
                 not isinstance(value, str) and
                 not isinstance(value, UserString) and
                 len(value) == 12 and
                 all(isinstance(v, float) for v in value) and
                 all(not (val < -1.7976931348623157e+308 or val > 1.7976931348623157e+308) or math.isinf(val) for val in value)), \
                "The 'joints_effort' field must be a set or sequence with length 12 and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._joints_effort = numpy.array(value, dtype=numpy.float64)

    @builtins.property
    def joints_temperature(self):
        """Message field 'joints_temperature'."""
        return self._joints_temperature

    @joints_temperature.setter
    def joints_temperature(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.float64, \
                "The 'joints_temperature' numpy.ndarray() must have the dtype of 'numpy.float64'"
            assert value.size == 12, \
                "The 'joints_temperature' numpy.ndarray() must have a size of 12"
            self._joints_temperature = value
            return
        if __debug__:
            from collections.abc import Sequence
            from collections.abc import Set
            from collections import UserList
            from collections import UserString
            assert \
                ((isinstance(value, Sequence) or
                  isinstance(value, Set) or
                  isinstance(value, UserList)) and
                 not isinstance(value, str) and
                 not isinstance(value, UserString) and
                 len(value) == 12 and
                 all(isinstance(v, float) for v in value) and
                 all(not (val < -1.7976931348623157e+308 or val > 1.7976931348623157e+308) or math.isinf(val) for val in value)), \
                "The 'joints_temperature' field must be a set or sequence with length 12 and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._joints_temperature = numpy.array(value, dtype=numpy.float64)

    @builtins.property
    def feet_contact(self):
        """Message field 'feet_contact'."""
        return self._feet_contact

    @feet_contact.setter
    def feet_contact(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.float64, \
                "The 'feet_contact' numpy.ndarray() must have the dtype of 'numpy.float64'"
            assert value.size == 4, \
                "The 'feet_contact' numpy.ndarray() must have a size of 4"
            self._feet_contact = value
            return
        if __debug__:
            from collections.abc import Sequence
            from collections.abc import Set
            from collections import UserList
            from collections import UserString
            assert \
                ((isinstance(value, Sequence) or
                  isinstance(value, Set) or
                  isinstance(value, UserList)) and
                 not isinstance(value, str) and
                 not isinstance(value, UserString) and
                 len(value) == 4 and
                 all(isinstance(v, float) for v in value) and
                 all(not (val < -1.7976931348623157e+308 or val > 1.7976931348623157e+308) or math.isinf(val) for val in value)), \
                "The 'feet_contact' field must be a set or sequence with length 4 and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._feet_contact = numpy.array(value, dtype=numpy.float64)

    @builtins.property
    def current_feet_positions(self):
        """Message field 'current_feet_positions'."""
        return self._current_feet_positions

    @current_feet_positions.setter
    def current_feet_positions(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.float64, \
                "The 'current_feet_positions' numpy.ndarray() must have the dtype of 'numpy.float64'"
            assert value.size == 12, \
                "The 'current_feet_positions' numpy.ndarray() must have a size of 12"
            self._current_feet_positions = value
            return
        if __debug__:
            from collections.abc import Sequence
            from collections.abc import Set
            from collections import UserList
            from collections import UserString
            assert \
                ((isinstance(value, Sequence) or
                  isinstance(value, Set) or
                  isinstance(value, UserList)) and
                 not isinstance(value, str) and
                 not isinstance(value, UserString) and
                 len(value) == 12 and
                 all(isinstance(v, float) for v in value) and
                 all(not (val < -1.7976931348623157e+308 or val > 1.7976931348623157e+308) or math.isinf(val) for val in value)), \
                "The 'current_feet_positions' field must be a set or sequence with length 12 and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._current_feet_positions = numpy.array(value, dtype=numpy.float64)
