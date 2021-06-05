#!/usr/bin/env python3
from common.op_params import opParams
import time
import ast
import difflib


class opEdit:  # use by running `python /data/openpilot/op_edit.py`
  def __init__(self):
    self.op_params = opParams()
    self.params = None
    self.sleep_time = 1.0
    self.live_tuning = self.op_params.get('op_edit_live_mode', False)

    self.run_init()

  def run_init(self):
    self.run_loop()

  def run_loop(self):
    while True:
      if not self.live_tuning:
        print('Parameters:\n')
      else:
        print('Live Parameters:\n')
      self.params = self.op_params.get(force_update=True)
      if self.live_tuning:  # only display live tunable params
        self.params = {k: v for k, v in self.params.items() if self.op_params.key_info(k).live}

      values_list = [self.params[i] if len(str(self.params[i])) < 20 else '{} ... {}'.format(str(self.params[i])[:30], str(self.params[i])[-15:]) for i in self.params]
      live = ['(live!)' if self.op_params.key_info(i).live else '' for i in self.params]

      to_print = ['{}. {}: {}  {}'.format(idx + 1, i, values_list[idx], live[idx]) for idx, i in enumerate(self.params)]
      extras = ['---\na. Add new parameter',
                'd. Delete parameter',
                'l. Toggle live tuning']
      to_print += extras

      print('\n'.join(to_print))
      print('\nChoose a parameter to edit (by index or name):')

      choice = input('>> ').strip()
      parsed, choice = self.parse_choice(choice, len(to_print) - len(extras))
      if parsed == 'continue':
        continue
      elif parsed == 'add':
        self.add_parameter()
      elif parsed == 'change':
        self.change_parameter(choice)
      elif parsed == 'delete':
        self.delete_parameter()
      elif parsed == 'live':
        self.live_tuning = not self.live_tuning
        self.op_params.put('op_edit_live_mode', self.live_tuning)  # for next opEdit startup
      elif parsed == 'exit':
        return

  def parse_choice(self, choice, opt_len):
    if choice.isdigit():
      choice = int(choice)
      choice -= 1
      if choice not in range(opt_len):  # number of options to choose from
        self.message('Not in range!')
        return 'continue', choice
      return 'change', choice

    if choice == '':
      print('Exiting opEdit!')
      return 'exit', choice

    if choice in ['a', 'add']:  # add new parameter
      return 'add', choice
    elif choice in ['d', 'delete', 'del']:  # delete parameter
      return 'delete', choice
    elif choice in ['l', 'live']:  # live tuning mode
      return 'live', choice
    else:  # find most similar param to user's input
      param_sims = [(idx, self.str_sim(choice, param)) for idx, param in enumerate(self.params)]
      param_sims = [param for param in param_sims if param[1] > 0.5]
      if len(param_sims) > 0:
        chosen_param = sorted(param_sims, key=lambda param: param[1], reverse=True)[0]
        return 'change', chosen_param[0]  # return idx

    self.message('Invalid choice!')
    return 'continue', choice

  def str_sim(self, a, b):
    return difflib.SequenceMatcher(a=a, b=b).ratio()

  def change_parameter(self, choice):
    while True:
      chosen_key = list(self.params)[choice]
      key_info = self.op_params.key_info(chosen_key)

      old_value = self.params[chosen_key]
      print('Chosen parameter: {}'.format(chosen_key))

      to_print = []
      if key_info.has_description:
        to_print.append('>>  Description: {}'.format(self.op_params.default_params[chosen_key]['description'].replace('\n', '\n  > ')))
      if key_info.has_allowed_types:
        to_print.append('>>  Allowed types: {}'.format(', '.join([str(i).split("'")[1] for i in key_info.allowed_types])))
      if key_info.live:
        to_print.append('>>  Live tunable! Up to 5 second delay')

      if to_print:
        print('\n{}\n'.format('\n'.join(to_print)))

      print('Current value: {} (type: {})'.format(old_value, str(type(old_value)).split("'")[1]))
      if key_info.has_clip:
        print('Clipped value: {}, [{},{}]'.format(self.op_params.get(chosen_key), key_info.min, key_info.max))

      while True:
        print('\nEnter your new value:')
        new_value = input('>> ').strip()
        if new_value == '':
            self.message('Exiting this parameter...', 0.5)
            return

        new_value = self.parse_input(new_value)
        if key_info.has_allowed_types and type(new_value) not in key_info.allowed_types:
            self.message('The type of data you entered ({}) is not allowed with this parameter!'.format(str(type(new_value)).split("'")[1]))
            continue

        if key_info.live:  # stay in live tuning interface
          self.op_params.put(chosen_key, new_value)
          print('Saved {} with value: {}! (type: {})'.format(chosen_key, new_value, str(type(new_value)).split("'")[1]))
        else:  # else ask to save and break
          print('\nOld value: {} (type: {})'.format(old_value, str(type(old_value)).split("'")[1]))
          print('New value: {} (type: {})'.format(new_value, str(type(new_value)).split("'")[1]))
          print('\nDo you want to save this?')
          if self.input_with_options(['Y', 'n'], 'n')[0] == 0:
            self.op_params.put(chosen_key, new_value)
            self.message('Saved!')
          else:
            self.message('Not saved!')
          return

  def input_with_options(self, options, default=None):
    """
    Takes in a list of options and asks user to make a choice.
    The most similar option list index is returned along with the similarity percentage from 0 to 1
    """
    user_input = input('[{}]: '.format('/'.join(options))).lower().strip()
    if not user_input:
      return default, 0.0
    sims = [self.str_sim(i.lower().strip(), user_input) for i in options]
    argmax = sims.index(max(sims))
    return argmax, sims[argmax]

  def parse_input(self, dat):
    dat = dat.strip()
    try:
      dat = ast.literal_eval(dat)
    except Exception as e:
      print(e)
      if dat.lower() == 'none':
        dat = None
      elif dat.lower() == 'false':
        dat = False
      elif dat.lower() == 'true':  # else, assume string
        dat = True
    return dat

  def delete_parameter(self):
    while True:
      print('Enter the name of the parameter to delete:')
      key = self.parse_input(input('>> '))

      if key == '':
        return
      if not isinstance(key, str):
        self.message('Input must be a string!')
        continue
      if key not in self.params:
        self.message("Parameter doesn't exist!")
        continue

      value = self.params.get(key)
      print('Parameter name: {}'.format(key))
      print('Parameter value: {} (type: {})'.format(value, str(type(value)).split("'")[1]))
      print('Do you want to delete this?')

      if self.input_with_options(['Y', 'n'], default='n')[0] == 0:
        self.op_params.delete(key)
        self.message('Deleted!')
      else:
        self.message('Not saved!')
      return

  def add_parameter(self):
    while True:
      print('Type the name of your new parameter:')
      key = self.parse_input(input('>> '))

      if key == '':
        return
      if not isinstance(key, str):
        self.message('Input must be a string!')
        continue

      print("Enter the data you'd like to save with this parameter:")
      value = input('>> ').strip()
      value = self.parse_input(value)

      print('Parameter name: {}'.format(key))
      print('Parameter value: {} (type: {})'.format(value, str(type(value)).split("'")[1]))
      print('Do you want to save this?')

      if self.input_with_options(['Y', 'n'], default='n')[0] == 0:
        self.op_params.put(key, value)
        self.message('Saved!')
      else:
        self.message('Not saved!')
      return

  def message(self, msg, sleep_time=None):
    if sleep_time is None:
      sleep_time = self.sleep_time
    print('--------\n{}\n--------'.format(msg), flush=True)
    time.sleep(sleep_time)
    print()


opEdit()
