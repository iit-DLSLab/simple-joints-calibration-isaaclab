# generated from rosidl_generator_py/resource/_idl.py.em
# with input from dls2_msgs:msg/TrajectoryGeneratorMsg.idl
# generated code does not contain a copyright notice


# Import statements for member types

import builtins  # noqa: E402, I100

import math  # noqa: E402, I100

# Member 'com_position'
# Member 'com_orientation'
# Member 'com_linear_velocity'
# Member 'com_angular_velocity'
# Member 'com_linear_acceleration'
# Member 'com_angular_acceleration'
# Member 'joints_position'
# Member 'joints_velocity'
# Member 'joints_acceleration'
# Member 'joints_effort'
# Member 'wrench'
# Member 'nominal_touch_down'
# Member 'touch_down'
# Member 'swing_period'
# Member 'normal_force_max'
# Member 'normal_force_min'
# Member 'kp'
# Member 'kd'
import numpy  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_TrajectoryGeneratorMsg(type):
    """Metaclass of message 'TrajectoryGeneratorMsg'."""

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
                'dls2_msgs.msg.TrajectoryGeneratorMsg')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__msg__trajectory_generator_msg
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__msg__trajectory_generator_msg
            cls._CONVERT_TO_PY = module.convert_to_py_msg__msg__trajectory_generator_msg
            cls._TYPE_SUPPORT = module.type_support_msg__msg__trajectory_generator_msg
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__msg__trajectory_generator_msg

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class TrajectoryGeneratorMsg(metaclass=Metaclass_TrajectoryGeneratorMsg):
    """Message class 'TrajectoryGeneratorMsg'."""

    __slots__ = [
        '_frame_id',
        '_sequence_id',
        '_timestamp',
        '_com_position',
        '_com_orientation',
        '_com_linear_velocity',
        '_com_angular_velocity',
        '_com_linear_acceleration',
        '_com_angular_acceleration',
        '_joints_position',
        '_joints_velocity',
        '_joints_acceleration',
        '_joints_effort',
        '_wrench',
        '_stance_legs',
        '_nominal_touch_down',
        '_touch_down',
        '_swing_period',
        '_normal_force_max',
        '_normal_force_min',
        '_kp',
        '_kd',
    ]

    _fields_and_field_types = {
        'frame_id': 'string',
        'sequence_id': 'uint32',
        'timestamp': 'double',
        'com_position': 'double[3]',
        'com_orientation': 'double[4]',
        'com_linear_velocity': 'double[3]',
        'com_angular_velocity': 'double[3]',
        'com_linear_acceleration': 'double[3]',
        'com_angular_acceleration': 'double[3]',
        'joints_position': 'double[12]',
        'joints_velocity': 'double[12]',
        'joints_acceleration': 'double[12]',
        'joints_effort': 'double[12]',
        'wrench': 'double[6]',
        'stance_legs': 'boolean[4]',
        'nominal_touch_down': 'double[12]',
        'touch_down': 'double[12]',
        'swing_period': 'double[4]',
        'normal_force_max': 'double[4]',
        'normal_force_min': 'double[4]',
        'kp': 'double[12]',
        'kd': 'double[12]',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.UnboundedString(),  # noqa: E501
        rosidl_parser.definition.BasicType('uint32'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('double'), 3),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('double'), 4),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('double'), 3),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('double'), 3),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('double'), 3),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('double'), 3),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('double'), 12),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('double'), 12),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('double'), 12),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('double'), 12),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('double'), 6),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('boolean'), 4),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('double'), 12),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('double'), 12),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('double'), 4),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('double'), 4),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('double'), 4),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('double'), 12),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('double'), 12),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        self.frame_id = kwargs.get('frame_id', str())
        self.sequence_id = kwargs.get('sequence_id', int())
        self.timestamp = kwargs.get('timestamp', float())
        if 'com_position' not in kwargs:
            self.com_position = numpy.zeros(3, dtype=numpy.float64)
        else:
            self.com_position = numpy.array(kwargs.get('com_position'), dtype=numpy.float64)
            assert self.com_position.shape == (3, )
        if 'com_orientation' not in kwargs:
            self.com_orientation = numpy.zeros(4, dtype=numpy.float64)
        else:
            self.com_orientation = numpy.array(kwargs.get('com_orientation'), dtype=numpy.float64)
            assert self.com_orientation.shape == (4, )
        if 'com_linear_velocity' not in kwargs:
            self.com_linear_velocity = numpy.zeros(3, dtype=numpy.float64)
        else:
            self.com_linear_velocity = numpy.array(kwargs.get('com_linear_velocity'), dtype=numpy.float64)
            assert self.com_linear_velocity.shape == (3, )
        if 'com_angular_velocity' not in kwargs:
            self.com_angular_velocity = numpy.zeros(3, dtype=numpy.float64)
        else:
            self.com_angular_velocity = numpy.array(kwargs.get('com_angular_velocity'), dtype=numpy.float64)
            assert self.com_angular_velocity.shape == (3, )
        if 'com_linear_acceleration' not in kwargs:
            self.com_linear_acceleration = numpy.zeros(3, dtype=numpy.float64)
        else:
            self.com_linear_acceleration = numpy.array(kwargs.get('com_linear_acceleration'), dtype=numpy.float64)
            assert self.com_linear_acceleration.shape == (3, )
        if 'com_angular_acceleration' not in kwargs:
            self.com_angular_acceleration = numpy.zeros(3, dtype=numpy.float64)
        else:
            self.com_angular_acceleration = numpy.array(kwargs.get('com_angular_acceleration'), dtype=numpy.float64)
            assert self.com_angular_acceleration.shape == (3, )
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
        if 'wrench' not in kwargs:
            self.wrench = numpy.zeros(6, dtype=numpy.float64)
        else:
            self.wrench = numpy.array(kwargs.get('wrench'), dtype=numpy.float64)
            assert self.wrench.shape == (6, )
        self.stance_legs = kwargs.get(
            'stance_legs',
            [bool() for x in range(4)]
        )
        if 'nominal_touch_down' not in kwargs:
            self.nominal_touch_down = numpy.zeros(12, dtype=numpy.float64)
        else:
            self.nominal_touch_down = numpy.array(kwargs.get('nominal_touch_down'), dtype=numpy.float64)
            assert self.nominal_touch_down.shape == (12, )
        if 'touch_down' not in kwargs:
            self.touch_down = numpy.zeros(12, dtype=numpy.float64)
        else:
            self.touch_down = numpy.array(kwargs.get('touch_down'), dtype=numpy.float64)
            assert self.touch_down.shape == (12, )
        if 'swing_period' not in kwargs:
            self.swing_period = numpy.zeros(4, dtype=numpy.float64)
        else:
            self.swing_period = numpy.array(kwargs.get('swing_period'), dtype=numpy.float64)
            assert self.swing_period.shape == (4, )
        if 'normal_force_max' not in kwargs:
            self.normal_force_max = numpy.zeros(4, dtype=numpy.float64)
        else:
            self.normal_force_max = numpy.array(kwargs.get('normal_force_max'), dtype=numpy.float64)
            assert self.normal_force_max.shape == (4, )
        if 'normal_force_min' not in kwargs:
            self.normal_force_min = numpy.zeros(4, dtype=numpy.float64)
        else:
            self.normal_force_min = numpy.array(kwargs.get('normal_force_min'), dtype=numpy.float64)
            assert self.normal_force_min.shape == (4, )
        if 'kp' not in kwargs:
            self.kp = numpy.zeros(12, dtype=numpy.float64)
        else:
            self.kp = numpy.array(kwargs.get('kp'), dtype=numpy.float64)
            assert self.kp.shape == (12, )
        if 'kd' not in kwargs:
            self.kd = numpy.zeros(12, dtype=numpy.float64)
        else:
            self.kd = numpy.array(kwargs.get('kd'), dtype=numpy.float64)
            assert self.kd.shape == (12, )

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
        if all(self.com_position != other.com_position):
            return False
        if all(self.com_orientation != other.com_orientation):
            return False
        if all(self.com_linear_velocity != other.com_linear_velocity):
            return False
        if all(self.com_angular_velocity != other.com_angular_velocity):
            return False
        if all(self.com_linear_acceleration != other.com_linear_acceleration):
            return False
        if all(self.com_angular_acceleration != other.com_angular_acceleration):
            return False
        if all(self.joints_position != other.joints_position):
            return False
        if all(self.joints_velocity != other.joints_velocity):
            return False
        if all(self.joints_acceleration != other.joints_acceleration):
            return False
        if all(self.joints_effort != other.joints_effort):
            return False
        if all(self.wrench != other.wrench):
            return False
        if self.stance_legs != other.stance_legs:
            return False
        if all(self.nominal_touch_down != other.nominal_touch_down):
            return False
        if all(self.touch_down != other.touch_down):
            return False
        if all(self.swing_period != other.swing_period):
            return False
        if all(self.normal_force_max != other.normal_force_max):
            return False
        if all(self.normal_force_min != other.normal_force_min):
            return False
        if all(self.kp != other.kp):
            return False
        if all(self.kd != other.kd):
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
    def com_position(self):
        """Message field 'com_position'."""
        return self._com_position

    @com_position.setter
    def com_position(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.float64, \
                "The 'com_position' numpy.ndarray() must have the dtype of 'numpy.float64'"
            assert value.size == 3, \
                "The 'com_position' numpy.ndarray() must have a size of 3"
            self._com_position = value
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
                 len(value) == 3 and
                 all(isinstance(v, float) for v in value) and
                 all(not (val < -1.7976931348623157e+308 or val > 1.7976931348623157e+308) or math.isinf(val) for val in value)), \
                "The 'com_position' field must be a set or sequence with length 3 and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._com_position = numpy.array(value, dtype=numpy.float64)

    @builtins.property
    def com_orientation(self):
        """Message field 'com_orientation'."""
        return self._com_orientation

    @com_orientation.setter
    def com_orientation(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.float64, \
                "The 'com_orientation' numpy.ndarray() must have the dtype of 'numpy.float64'"
            assert value.size == 4, \
                "The 'com_orientation' numpy.ndarray() must have a size of 4"
            self._com_orientation = value
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
                "The 'com_orientation' field must be a set or sequence with length 4 and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._com_orientation = numpy.array(value, dtype=numpy.float64)

    @builtins.property
    def com_linear_velocity(self):
        """Message field 'com_linear_velocity'."""
        return self._com_linear_velocity

    @com_linear_velocity.setter
    def com_linear_velocity(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.float64, \
                "The 'com_linear_velocity' numpy.ndarray() must have the dtype of 'numpy.float64'"
            assert value.size == 3, \
                "The 'com_linear_velocity' numpy.ndarray() must have a size of 3"
            self._com_linear_velocity = value
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
                 len(value) == 3 and
                 all(isinstance(v, float) for v in value) and
                 all(not (val < -1.7976931348623157e+308 or val > 1.7976931348623157e+308) or math.isinf(val) for val in value)), \
                "The 'com_linear_velocity' field must be a set or sequence with length 3 and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._com_linear_velocity = numpy.array(value, dtype=numpy.float64)

    @builtins.property
    def com_angular_velocity(self):
        """Message field 'com_angular_velocity'."""
        return self._com_angular_velocity

    @com_angular_velocity.setter
    def com_angular_velocity(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.float64, \
                "The 'com_angular_velocity' numpy.ndarray() must have the dtype of 'numpy.float64'"
            assert value.size == 3, \
                "The 'com_angular_velocity' numpy.ndarray() must have a size of 3"
            self._com_angular_velocity = value
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
                 len(value) == 3 and
                 all(isinstance(v, float) for v in value) and
                 all(not (val < -1.7976931348623157e+308 or val > 1.7976931348623157e+308) or math.isinf(val) for val in value)), \
                "The 'com_angular_velocity' field must be a set or sequence with length 3 and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._com_angular_velocity = numpy.array(value, dtype=numpy.float64)

    @builtins.property
    def com_linear_acceleration(self):
        """Message field 'com_linear_acceleration'."""
        return self._com_linear_acceleration

    @com_linear_acceleration.setter
    def com_linear_acceleration(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.float64, \
                "The 'com_linear_acceleration' numpy.ndarray() must have the dtype of 'numpy.float64'"
            assert value.size == 3, \
                "The 'com_linear_acceleration' numpy.ndarray() must have a size of 3"
            self._com_linear_acceleration = value
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
                 len(value) == 3 and
                 all(isinstance(v, float) for v in value) and
                 all(not (val < -1.7976931348623157e+308 or val > 1.7976931348623157e+308) or math.isinf(val) for val in value)), \
                "The 'com_linear_acceleration' field must be a set or sequence with length 3 and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._com_linear_acceleration = numpy.array(value, dtype=numpy.float64)

    @builtins.property
    def com_angular_acceleration(self):
        """Message field 'com_angular_acceleration'."""
        return self._com_angular_acceleration

    @com_angular_acceleration.setter
    def com_angular_acceleration(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.float64, \
                "The 'com_angular_acceleration' numpy.ndarray() must have the dtype of 'numpy.float64'"
            assert value.size == 3, \
                "The 'com_angular_acceleration' numpy.ndarray() must have a size of 3"
            self._com_angular_acceleration = value
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
                 len(value) == 3 and
                 all(isinstance(v, float) for v in value) and
                 all(not (val < -1.7976931348623157e+308 or val > 1.7976931348623157e+308) or math.isinf(val) for val in value)), \
                "The 'com_angular_acceleration' field must be a set or sequence with length 3 and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._com_angular_acceleration = numpy.array(value, dtype=numpy.float64)

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
    def wrench(self):
        """Message field 'wrench'."""
        return self._wrench

    @wrench.setter
    def wrench(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.float64, \
                "The 'wrench' numpy.ndarray() must have the dtype of 'numpy.float64'"
            assert value.size == 6, \
                "The 'wrench' numpy.ndarray() must have a size of 6"
            self._wrench = value
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
                 len(value) == 6 and
                 all(isinstance(v, float) for v in value) and
                 all(not (val < -1.7976931348623157e+308 or val > 1.7976931348623157e+308) or math.isinf(val) for val in value)), \
                "The 'wrench' field must be a set or sequence with length 6 and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._wrench = numpy.array(value, dtype=numpy.float64)

    @builtins.property
    def stance_legs(self):
        """Message field 'stance_legs'."""
        return self._stance_legs

    @stance_legs.setter
    def stance_legs(self, value):
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
                 all(isinstance(v, bool) for v in value) and
                 True), \
                "The 'stance_legs' field must be a set or sequence with length 4 and each value of type 'bool'"
        self._stance_legs = value

    @builtins.property
    def nominal_touch_down(self):
        """Message field 'nominal_touch_down'."""
        return self._nominal_touch_down

    @nominal_touch_down.setter
    def nominal_touch_down(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.float64, \
                "The 'nominal_touch_down' numpy.ndarray() must have the dtype of 'numpy.float64'"
            assert value.size == 12, \
                "The 'nominal_touch_down' numpy.ndarray() must have a size of 12"
            self._nominal_touch_down = value
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
                "The 'nominal_touch_down' field must be a set or sequence with length 12 and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._nominal_touch_down = numpy.array(value, dtype=numpy.float64)

    @builtins.property
    def touch_down(self):
        """Message field 'touch_down'."""
        return self._touch_down

    @touch_down.setter
    def touch_down(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.float64, \
                "The 'touch_down' numpy.ndarray() must have the dtype of 'numpy.float64'"
            assert value.size == 12, \
                "The 'touch_down' numpy.ndarray() must have a size of 12"
            self._touch_down = value
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
                "The 'touch_down' field must be a set or sequence with length 12 and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._touch_down = numpy.array(value, dtype=numpy.float64)

    @builtins.property
    def swing_period(self):
        """Message field 'swing_period'."""
        return self._swing_period

    @swing_period.setter
    def swing_period(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.float64, \
                "The 'swing_period' numpy.ndarray() must have the dtype of 'numpy.float64'"
            assert value.size == 4, \
                "The 'swing_period' numpy.ndarray() must have a size of 4"
            self._swing_period = value
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
                "The 'swing_period' field must be a set or sequence with length 4 and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._swing_period = numpy.array(value, dtype=numpy.float64)

    @builtins.property
    def normal_force_max(self):
        """Message field 'normal_force_max'."""
        return self._normal_force_max

    @normal_force_max.setter
    def normal_force_max(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.float64, \
                "The 'normal_force_max' numpy.ndarray() must have the dtype of 'numpy.float64'"
            assert value.size == 4, \
                "The 'normal_force_max' numpy.ndarray() must have a size of 4"
            self._normal_force_max = value
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
                "The 'normal_force_max' field must be a set or sequence with length 4 and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._normal_force_max = numpy.array(value, dtype=numpy.float64)

    @builtins.property
    def normal_force_min(self):
        """Message field 'normal_force_min'."""
        return self._normal_force_min

    @normal_force_min.setter
    def normal_force_min(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.float64, \
                "The 'normal_force_min' numpy.ndarray() must have the dtype of 'numpy.float64'"
            assert value.size == 4, \
                "The 'normal_force_min' numpy.ndarray() must have a size of 4"
            self._normal_force_min = value
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
                "The 'normal_force_min' field must be a set or sequence with length 4 and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._normal_force_min = numpy.array(value, dtype=numpy.float64)

    @builtins.property
    def kp(self):
        """Message field 'kp'."""
        return self._kp

    @kp.setter
    def kp(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.float64, \
                "The 'kp' numpy.ndarray() must have the dtype of 'numpy.float64'"
            assert value.size == 12, \
                "The 'kp' numpy.ndarray() must have a size of 12"
            self._kp = value
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
                "The 'kp' field must be a set or sequence with length 12 and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._kp = numpy.array(value, dtype=numpy.float64)

    @builtins.property
    def kd(self):
        """Message field 'kd'."""
        return self._kd

    @kd.setter
    def kd(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.float64, \
                "The 'kd' numpy.ndarray() must have the dtype of 'numpy.float64'"
            assert value.size == 12, \
                "The 'kd' numpy.ndarray() must have a size of 12"
            self._kd = value
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
                "The 'kd' field must be a set or sequence with length 12 and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._kd = numpy.array(value, dtype=numpy.float64)
