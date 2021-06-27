# functions common among cars
from common.numpy_fast import clip

# kg of standard extra cargo to count for drive, gas, etc...
STD_CARGO_KG = 136.


def gen_empty_fingerprint():
  return {i: {} for i in range(0, 4)}


# FIXME: hardcoding honda civic 2016 touring params so they can be used to
# scale unknown params for other cars
class CivicParams:
  MASS = 1326. + STD_CARGO_KG
  WHEELBASE = 2.70
  CENTER_TO_FRONT = WHEELBASE * 0.4
  CENTER_TO_REAR = WHEELBASE - CENTER_TO_FRONT
  ROTATIONAL_INERTIA = 2500
  TIRE_STIFFNESS_FRONT = 192150
  TIRE_STIFFNESS_REAR = 202500


# TODO: get actual value, for now starting with reasonable value for
# civic and scaling by mass and wheelbase
def scale_rot_inertia(mass, wheelbase):
  return CivicParams.ROTATIONAL_INERTIA * mass * wheelbase ** 2 / (CivicParams.MASS * CivicParams.WHEELBASE ** 2)


# TODO: start from empirically derived lateral slip stiffness for the civic and scale by
# mass and CG position, so all cars will have approximately similar dyn behaviors
def scale_tire_stiffness(mass, wheelbase, center_to_front, tire_stiffness_factor=1.0):
  center_to_rear = wheelbase - center_to_front
  tire_stiffness_front = (CivicParams.TIRE_STIFFNESS_FRONT * tire_stiffness_factor) * mass / CivicParams.MASS * \
                         (center_to_rear / wheelbase) / (CivicParams.CENTER_TO_REAR / CivicParams.WHEELBASE)

  tire_stiffness_rear = (CivicParams.TIRE_STIFFNESS_REAR * tire_stiffness_factor) * mass / CivicParams.MASS * \
                        (center_to_front / wheelbase) / (CivicParams.CENTER_TO_FRONT / CivicParams.WHEELBASE)

  return tire_stiffness_front, tire_stiffness_rear


def dbc_dict(pt_dbc, radar_dbc, chassis_dbc=None, body_dbc=None):
  return {'pt': pt_dbc, 'radar': radar_dbc, 'chassis': chassis_dbc, 'body': body_dbc}


# Limit steer torque rate up (slower), down (faster), and magnitude.
def steer_limit_rate(new, last, LIMITS):
  if last > 0:
    new = clip(new,
               max(last - LIMITS.STEER_DELTA_DOWN, -LIMITS.STEER_DELTA_UP),
               last + LIMITS.STEER_DELTA_UP)
  else:
    new = clip(new,
               last - LIMITS.STEER_DELTA_UP,
               min(last + LIMITS.STEER_DELTA_DOWN, LIMITS.STEER_DELTA_UP))
  return clip(new, -LIMITS.STEER_MAX, LIMITS.STEER_MAX)


# Limit steer torque to be near reported motor torque.
def steer_limit_motor(steer, motor, LIMITS):
  motor_min = min(motor - LIMITS.STEER_ERROR_MAX, -LIMITS.STEER_ERROR_MAX)
  motor_max = max(motor + LIMITS.STEER_ERROR_MAX, LIMITS.STEER_ERROR_MAX)
  return clip(steer, motor_min, motor_max)


# Limit steer torque when driver opposes control.
def steer_limit_driver(steer, driver, LIMITS):
  driver_min = -LIMITS.STEER_MAX + (-LIMITS.STEER_DRIVER_ALLOWANCE + driver * LIMITS.STEER_DRIVER_FACTOR) * LIMITS.STEER_DRIVER_MULTIPLIER
  driver_max = LIMITS.STEER_MAX + (LIMITS.STEER_DRIVER_ALLOWANCE + driver * LIMITS.STEER_DRIVER_FACTOR) * LIMITS.STEER_DRIVER_MULTIPLIER
  return clip(steer, min(driver_min,0), max(driver_max,0)


def crc8_pedal(data):
  crc = 0xFF    # standard init value
  poly = 0xD5   # standard crc8: x8+x7+x6+x4+x2+1
  size = len(data)
  for i in range(size - 1, -1, -1):
    crc ^= data[i]
    for _ in range(8):
      if ((crc & 0x80) != 0):
        crc = ((crc << 1) ^ poly) & 0xFF
      else:
        crc <<= 1
  return crc

def create_gas_command(packer, gas_amount, idx):
  # Common gas pedal msg generator
  enable = gas_amount > 0.001

  values = {
    "ENABLE": enable,
    "COUNTER_PEDAL": idx & 0xF,
  }

  if enable:
    values["GAS_COMMAND"] = gas_amount * 255.
    values["GAS_COMMAND2"] = gas_amount * 255.

  dat = packer.make_can_msg("GAS_COMMAND", 0, values)[2]

  checksum = crc8_pedal(dat[:-1])
  values["CHECKSUM_PEDAL"] = checksum

  return packer.make_can_msg("GAS_COMMAND", 0, values)


def make_can_msg(addr, dat, bus):
  return [addr, 0, dat, bus]
