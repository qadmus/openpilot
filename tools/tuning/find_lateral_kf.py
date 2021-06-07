#!/usr/bin/env python3
# import warnings
# warnings.simplefilter(action='ignore', category=FutureWarning)
import os
import sys

try:
  from opendbc.can.parser import CANParser
  from tools.lib.logreader import MultiLogIterator
  from tools.lib.route import Route
  from cereal import car
except:  # running on pc
  # sys.path.insert(0, 'C:/Git/openpilot-repos/op-smiskol')
  # os.environ['PYTHONPATH'] = '.'
  pass

dir_name = os.path.dirname(os.path.realpath(__file__))


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import cm
from tqdm import tqdm   # type: ignore
from scipy.optimize import curve_fit

from selfdrive.config import Conversions as CV
import pickle

STEER_THRESHOLD = 100
DT_CTRL = 0.01

old_kf = 0.00004  # for plotting old function compared to new polynomial function
MIN_SAMPLES = int(30 / DT_CTRL)
MAX_TORQUE = 1500  # fixme: auto detect, this is toyota


def feedforward(v_ego, angle_steers, kf=1.):  # outputs unscaled ~lateral accel, with optional kf param to scale
  steer_feedforward = angle_steers * v_ego ** 2
  return steer_feedforward * kf


def _fit_kf(x_input, _kf):
  # Fits a kf gain using speed ** 2 feedforward model
  v_ego, angle_steers = x_input.copy()
  return feedforward(v_ego, angle_steers, _kf)


def load_processed(file_name):
  with open(file_name, 'rb') as f:
    return pickle.load(f)


def load_and_process_rlogs(lr, file_name):
  data = [[]]

  v_ego, angle_offset, angle_steers, angle_rate, torque_cmd, eps_torque = None, None, None, None, None, None
  last_time = 0
  can_updated = False

  signals = [
    ("STEER_REQUEST", "STEERING_LKA", 0),
    ("STEER_TORQUE_CMD", "STEERING_LKA", 0),
    ("STEER_TORQUE_DRIVER", "STEER_TORQUE_SENSOR", 0),
    ("STEER_ANGLE", "STEER_TORQUE_SENSOR", 0),
    ("STEER_ANGLE", "STEER_ANGLE_SENSOR", 0),
    ("STEER_FRACTION", "STEER_ANGLE_SENSOR", 0),
  ]
  cp = CANParser("toyota_nodsu_pt_generated", signals, enforce_checks=False)  # todo: auto load dbc from logs

  all_msgs = sorted(lr, key=lambda msg: msg.logMonoTime)

  for msg in tqdm(all_msgs):
    if msg.which() == 'carState':
      v_ego = msg.carState.vEgo
      angle_steers = msg.carState.steeringAngleDeg  # this is offset from can by 1 frame. it's within acceptable margin of error and we get more accuracy for TSS2
      angle_rate = msg.carState.steeringRateDeg
      eps_torque = msg.carState.steeringTorqueEps

    elif msg.which() == 'liveParameters':
      angle_offset = msg.liveParameters.angleOffsetDeg

    if msg.which() != 'can':
      continue

    cp_updated = cp.update_string(msg.as_builder().to_bytes())
    for u in cp_updated:
      if u == 0x2e4:  # STEERING_LKA
        can_updated = True

    torque_cmd = int(cp.vl['STEERING_LKA']['STEER_TORQUE_CMD'])
    steer_req = bool(cp.vl['STEERING_LKA']['STEER_REQUEST'])
    steering_pressed = abs(cp.vl['STEER_TORQUE_SENSOR']['STEER_TORQUE_DRIVER']) > STEER_THRESHOLD

    sample_ok = None not in [v_ego, angle_offset, torque_cmd] and can_updated
    sample_ok = sample_ok and (steer_req and not steering_pressed or not steer_req)  # bad sample if engaged and override, but not if disengaged

    final_torque = torque_cmd if steer_req else eps_torque

    # creates uninterupted sections of engaged data
    if sample_ok and abs(msg.logMonoTime - last_time) * 1e-9 < 1 / 20:  # also split if there's a break in time
      data[-1].append({'angle_steers': angle_steers, 'angle_rate': angle_rate, 'v_ego': v_ego,
                       'angle_offset': angle_offset, 'torque': final_torque, 'time': msg.logMonoTime * 1e-9})
    elif len(data[-1]):  # if last list has items in it, append new empty section
      data.append([])
    last_time = msg.logMonoTime

  print('Max seq. len: {}'.format(max([len(line) for line in data])))
  del all_msgs
  data = [sec for sec in data if len(sec) > 5 / DT_CTRL]  # long enough sections

  with open(file_name, 'wb') as f:  # now dump
    pickle.dump(data, f)
  return data


def fit_ff_model(lr):
  # TSS2 Corolla looks to be around 0.5 to 0.6
  STEER_DELAY = round(0.12 + 0.2 / DT_CTRL)  # important: needs to be accurate

  if os.path.exists('processed_data'):
    print('exists')
    data = load_processed('processed_data')
  else:
    print('processing')
    data = load_and_process_rlogs(lr, file_name='processed_data')
  print('Max seq. len: {}'.format(max([len(line) for line in data])))

  data = [sec for sec in data if len(sec) > DT_CTRL]  # long enough sections
  for i in range(len(data)):  # accounts for steer actuator delay (moves torque cmd up by x samples)
    torque = [line['torque'] for line in data[i]]
    for j in range(len(data[i])):
      if j < STEER_DELAY:
        continue
      data[i][j]['torque'] = torque[j - STEER_DELAY]
    data[i] = data[i][STEER_DELAY:]  # removes leading samples
  data = [i for j in data for i in j]  # flatten
  print(f'Samples (before filtering): {len(data)}')

  # Data filtering
  # we want samples where wheel not moving very much, less reliance on lag compensation above
  data = [line for line in data if 1 <= abs(line['angle_steers']) <= 45.]  # no need for higher
  data = [line for line in data if abs(line['angle_rate']) <= 25.]  # TODO: tune this
  data = [line for line in data if abs(line['v_ego']) >= 2 * CV.MPH_TO_MS]
  data = [line for line in data if abs(line['torque']) <= 1500]

  assert len(data) > MIN_SAMPLES, 'too few valid samples found in route'

  print(f'Samples (after filtering):  {len(data)}\n')
  print('Max angle: {}'.format(round(max([i['angle_steers'] for i in data]), 2)))
  print('Max angle rate: {}'.format(round(max([i['angle_rate'] for i in data]), 2)))
  print('Top speed: {} mph'.format(round(max([i['v_ego'] for i in data]) * CV.MS_TO_MPH, 2)))
  print('Torque: min: {}, max: {}\n'.format(*[func([i['torque'] for i in data]) for func in [min, max]]))

  # Data preprocessing
  for line in data:
    line['angle_steers'] = abs(line['angle_steers'] - line['angle_offset'])  # need to offset angle to properly fit ff
    line['torque'] = abs(line['torque'])

    del line['time'], line['angle_offset']  # delete unused

  data_speeds = np.array([line['v_ego'] for line in data])
  data_angles = np.array([line['angle_steers'] for line in data])
  data_torque = np.array([line['torque'] for line in data])

  params, covs = curve_fit(_fit_kf, np.array([data_speeds, data_angles]), np.array(data_torque) / MAX_TORQUE,
                           # maxfev=800
                           )
  fit_kf = params[0]
  print('FOUND KF: {}'.format(fit_kf))

  print()

  std_func = []
  fitted_func = []
  for line in data:
    std_func.append(abs(feedforward(line['v_ego'], line['angle_steers'], old_kf) * MAX_TORQUE - line['torque']))
    fitted_func.append(abs(feedforward(line['v_ego'], line['angle_steers'], fit_kf) * MAX_TORQUE - line['torque']))
  print('Function comparison on input data')
  print('Torque MAE, current vs. fitted: {}, {}'.format(round(np.mean(std_func), 3), round(np.mean(fitted_func), 3)))
  print('Torque STD, current vs. fitted: {}, {}'.format(round(np.std(std_func), 3), round(np.std(fitted_func), 3)))

  if SPEED_DATA_ANALYSIS := True:  # analyzes how torque needed changes based on speed
    if PLOT_ANGLE_DIST := False:
      sns.distplot([line['angle_steers'] for line in data if abs(line['angle_steers']) < 30], bins=200)
      raise Exception

    res = 100
    color = 'blue'

    _angles = [
      [5, 10],
      [10, 20],
      [10, 15],
      [15, 20],
      [20, 25],
      [20, 30],
      [30, 45],
    ]

    for idx, angle_range in enumerate(_angles):
      angle_range_str = '{} deg'.format('-'.join(map(str, angle_range)))
      temp_data = [line for line in data if angle_range[0] <= abs(line['angle_steers']) <= angle_range[1]]
      if not len(temp_data):
        continue
      print(f'{angle_range} samples: {len(temp_data)}')
      plt.figure()
      speeds, torque = zip(*[[line['v_ego'], line['torque']] for line in temp_data])
      plt.scatter(np.array(speeds) * CV.MS_TO_MPH, torque, label=angle_range_str, color=color, s=0.05)

      _x_ff = np.linspace(0, max(speeds), res)
      _y_ff = [feedforward(_i, np.mean(angle_range), old_kf) * MAX_TORQUE for _i in _x_ff]
      plt.plot(_x_ff * CV.MS_TO_MPH, _y_ff, color='orange', label='standard ff model at {} deg'.format(np.mean(angle_range)))

      _y_ff = [feedforward(_i, np.mean(angle_range), fit_kf) * MAX_TORQUE for _i in _x_ff]
      plt.plot(_x_ff * CV.MS_TO_MPH, _y_ff, color='purple', label='new fitted ff function')

      plt.legend()
      plt.xlabel('speed (mph)')
      plt.ylabel('torque')
      plt.savefig('plots/{}.png'.format(angle_range_str))

  if ANGLE_DATA_ANALYSIS := True:  # analyzes how angle changes need of torque (RESULT: seems to be relatively linear, can be tuned by k_f)
    if PLOT_ANGLE_DIST := False:
      sns.distplot([line['angle_steers'] for line in data if abs(line['angle_steers']) < 30], bins=200)
      raise Exception

    res = 100

    _speeds = np.r_[[
      [0, 10],
      [10, 20],
      [20, 30],
      [30, 40],
      [40, 50],
      [50, 60],
      [60, 70],
    ]] * CV.MPH_TO_MS
    color = 'blue'

    for idx, speed_range in enumerate(_speeds):
      speed_range_str = '{} mph'.format('-'.join([str(round(i * CV.MS_TO_MPH, 1)) for i in speed_range]))
      temp_data = [line for line in data if speed_range[0] <= line['v_ego'] <= speed_range[1]]
      if not len(temp_data):
        continue
      print(f'{speed_range_str} samples: {len(temp_data)}')
      plt.figure()
      angles, torque, speeds = zip(*[[line['angle_steers'], line['torque'], line['v_ego']] for line in temp_data])
      plt.scatter(angles, torque, label=speed_range_str, color=color, s=0.05)

      _x_ff = np.linspace(0, max(angles), res)
      _y_ff = [feedforward(np.mean(speed_range), _i, old_kf) * MAX_TORQUE for _i in _x_ff]
      plt.plot(_x_ff, _y_ff, color='orange', label='standard ff model at {} mph'.format(np.round(np.mean(speed_range) * CV.MS_TO_MPH, 1)))

      _y_ff = [feedforward(np.mean(speed_range), _i, fit_kf) * MAX_TORQUE for _i in _x_ff]
      plt.plot(_x_ff, _y_ff, color='purple', label='new fitted ff function')

      plt.legend()
      plt.xlabel('angle (deg)')
      plt.ylabel('torque')
      plt.savefig('plots/{}.png'.format(speed_range_str))

  # if PLOT_3D := False:
  #   X_test = np.linspace(0, max(data_speeds), 20)
  #   Y_test = np.linspace(0, max(data_angles), 20)
  #
  #   Z_test = np.zeros((len(X_test), len(Y_test)))
  #   for i in range(len(X_test)):
  #     for j in range(len(Y_test)):
  #       Z_test[i][j] = feedforward(X_test[i], Y_test[j], fit_kf)
  #
  #   X_test, Y_test = np.meshgrid(X_test, Y_test)
  #
  #   fig = plt.figure()
  #   ax = plt.axes(projection='3d')
  #
  #   surf = ax.plot_surface(X_test * CV.MS_TO_MPH, Y_test, Z_test, cmap=cm.magma,
  #                          linewidth=0, antialiased=False)
  #   fig.colorbar(surf, shrink=0.5, aspect=5)
  #
  #   ax.set_xlabel('speed (mph)')
  #   ax.set_ylabel('angle')
  #   ax.set_zlabel('feedforward')
  #   plt.title('New fitted polynomial feedforward function')


# Compares poly with old ff speed function
# x = np.linspace(0, 30, 100)
# y = x ** 2
# _c1, _c2, _c3 = 0.34365576041121065, 12.845373070976711, 51.63304088261174
# y_poly = _c1 * x ** 2 + _c2 * x + _c3
# plt.plot(x, y_poly, label='poly')
# plt.plot(x, y, label='v_ego**2')
# plt.legend()
# plt.show()


if __name__ == '__main__':
  r = Route(sys.argv[1])
  lr = MultiLogIterator(r.log_paths(), wraparound=False)

  n = fit_ff_model(lr)
