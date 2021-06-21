import math
import numpy as np

from cereal import log
from common.realtime import DT_CTRL
from common.numpy_fast import clip, interp


class LatControlINDI():
  def __init__(self, CP):
    self.angle_steers_des = 0.

    A = np.array([[1.0, DT_CTRL, 0.0],
                  [0.0, 1.0, DT_CTRL],
                  [0.0, 0.0, 1.0]])
    C = np.array([[1.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0]])

    # Q = np.matrix([[1e-2, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 10.0]])
    # R = np.matrix([[1e-2, 0.0], [0.0, 1e3]])

    # (x, l, K) = control.dare(np.transpose(A), np.transpose(C), Q, R)
    # K = np.transpose(K)
    K = np.array([[7.30262179e-01, 2.07003658e-04],
                  [7.29394177e+00, 1.39159419e-02],
                  [1.71022442e+01, 3.38495381e-02]])

    self.speed = 0.

    self.K = K
    self.A_K = A - np.dot(K, C)
    self.x = np.array([[0.], [0.], [0.]])

    self._RC = (CP.lateralTuning.indi.timeConstantBP, CP.lateralTuning.indi.timeConstantV)
    self._G = (CP.lateralTuning.indi.actuatorEffectivenessBP, CP.lateralTuning.indi.actuatorEffectivenessV)
    self._outer_loop_gain = (CP.lateralTuning.indi.outerLoopGainBP, CP.lateralTuning.indi.outerLoopGainV)
    self._inner_loop_gain = (CP.lateralTuning.indi.innerLoopGainBP, CP.lateralTuning.indi.innerLoopGainV)

    self.sat_count_rate = 1.0 * DT_CTRL
    self.sat_limit = CP.steerLimitTimer

    self.reset()

  @property
  def RC(self):
    return interp(self.speed, self._RC[0], self._RC[1])

  @property
  def G(self):
    return interp(self.speed, self._G[0], self._G[1])

  @property
  def outer_loop_gain(self):
    return interp(self.speed, self._outer_loop_gain[0], self._outer_loop_gain[1])

  @property
  def inner_loop_gain(self):
    return interp(self.speed, self._inner_loop_gain[0], self._inner_loop_gain[1])

  def reset(self):
    self.delayed_output = 0.
    self.output_steer = 0.
    self.sat_count = 0.0
    self.speed = 0.

  def _check_saturation(self, control, check_saturation, limit):
    saturated = abs(control) == limit

    if saturated and check_saturation:
      self.sat_count += self.sat_count_rate
    else:
      self.sat_count -= self.sat_count_rate

    self.sat_count = clip(self.sat_count, 0.0, 1.0)

    return self.sat_count > self.sat_limit

  def update(self, active, CI, VM, params, lat_plan):
    CS = CI.CS.out
    self.speed = CS.vEgo

    # Update Kalman filter
    y = np.array([[math.radians(CS.steeringAngleDeg)], [math.radians(CS.steeringRateDeg)]])
    self.x = np.dot(self.A_K, self.x) + np.dot(self.K, y)

    indi_log = log.ControlsState.LateralINDIState.new_message()
    indi_log.steeringAngleDeg = math.degrees(self.x[0])
    indi_log.steeringRateDeg = math.degrees(self.x[1])
    indi_log.steeringAccelDeg = math.degrees(self.x[2])

    if CS.vEgo < 0.3 or not active:
      indi_log.active = False
      self.output_steer = 0.0
      self.delayed_output = 0.0
    else:
      steers_des = VM.get_steer_from_curvature(-lat_plan.curvature, CS.vEgo)
      steers_des += math.radians(params.angleOffsetDeg)

      rate_des = VM.get_steer_from_curvature(-lat_plan.curvatureRate, CS.vEgo)

      # Expected actuator value
      alpha = 1. - DT_CTRL / (self.RC + DT_CTRL)
      self.delayed_output = self.delayed_output * alpha + self.output_steer * (1. - alpha)

      # Compute acceleration error
      rate_sp = self.outer_loop_gain * (steers_des - self.x[0]) + rate_des
      accel_sp = self.inner_loop_gain * (rate_sp - self.x[1])
      accel_error = accel_sp - self.x[2]

      # Compute change in actuator
      g_inv = 1. / self.G
      delta_u = g_inv * accel_error

      # If steering pressed, only allow wind down
      if CS.steeringPressed and (delta_u * self.output_steer > 0):
        delta_u = 0

      # Limit torque rate and max
      new = int(round((self.delayed_output + delta_u) * CI.CC.STEER_MAX))
      last = int(round(self.output_steer * CI.CC.STEER_MAX))
      self.output_steer = CI.limit_steer(new, last, 0)

      indi_log.active = True
      indi_log.rateSetPoint = float(rate_sp)
      indi_log.accelSetPoint = float(accel_sp)
      indi_log.accelError = float(accel_error)
      indi_log.delayedOutput = float(self.delayed_output)
      indi_log.delta = float(delta_u)
      indi_log.output = float(self.output_steer)

      check_saturation = (CS.vEgo > 10.) and not CS.steeringRateLimited and not CS.steeringPressed
      indi_log.saturated = self._check_saturation(self.output_steer, check_saturation, CI.CC.STEER_MAX)

    return float(self.output_steer), 0, indi_log
