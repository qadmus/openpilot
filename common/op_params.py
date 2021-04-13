#!/usr/bin/env python3
import json
import os
import time

from common.numpy_fast import clip
from selfdrive.swaglog import cloudlog
from common.basedir import BASEDIR

datadir = BASEDIR.strip("/").split("/")[0] == "data"


class KeyInfo:
    default = None
    allowed_types = [None]
    has_allowed_types = False
    live = False
    has_default = False
    has_description = False
    hidden = False
    has_clip = False
    min = None
    max = None


class opParams:
    def __init__(self):
        """
        Add new parameters here!

        Use `type(None)` instead of None
        `live` parameters can change, use op_params.get() in update()

        # Basic dictionary:
        self.default_params = {'camera_offset': {'default': 0.06}}

        # Example:
        from common.op_params import opParams
        op_params = opParams()
        ret.lateralTuning.indi.innerLoopGainV = [op_params.get('INDI_EFFECT_OVERRIDE')]
        """

        self.default_params = {
            "INDI_EFFECT_OVERRIDE": {
                "default": 0.,
                "clip": [0., 50.],
                "allowed_types": [float],
                "live": True,
                "description": "0. uses other live params, ",
            },
            "INDI_EFFECT_LOW": {
                "default": 22.,
                "clip": [1., 50.],
                "allowed_types": [float],
                "live": True,
            },
            "INDI_EFFECT_HIGH": {
                "default": 3.,
                "clip": [1., 10.],
                "allowed_types": [float],
                "live": True,
            },
            "INDI_ANGLE_GAIN": {
                "default": 16.,
                "clip": [1., 40.],
                "allowed_types": [float],
                "live": True,
            },
            "INDI_RATE_GAIN": {
                "default": 3.,
                "clip": [.5, 10.],
                "allowed_types": [float],
                "live": True,
            },
            "INDI_TIME": {
                "default": .3,
                "clip": [0., 10.],
                "allowed_types": [float],
                "live": True,
            },
            "STEER_DELAY": {
                "default": .2,
                "clip": [.01, .5],
                "allowed_types": [float],
                "live": True,
            },
            "LONG_P_05": {
                "default": 1.7,
                "clip": [.5, 2.5],
                "allowed_types": [float],
                "live": True,
            },
            "LONG_P_35": {
                "default": 1.3,
                "clip": [.5, 2.5],
                "allowed_types": [float],
                "live": True,
            },
            "LONG_I": {
                "default": .36,
                "clip": [0., .5],
                "allowed_types": [float],
                "live": True,
            },
            "op_edit_live_mode": {
                "default": True,
                "description": "Start mode for opEdit.",
                "hide": True,
            },
        }

        self.params = {}
        self.params_file = "/data/op_params.json"
        self.last_read_time = time.time()
        self.read_period = 5.0  # read period with self.get(...) (sec)
        self.force_update = False  # replaces values with default params if True, not just add add missing key/value pairs
        self.to_delete = [
            "old_key_to_delete"
        ]  # a list of params you want to delete (unused)
        self.run_init()  # restores, reads, and updates params

    def run_init(self):  # does first time initializing of default params
        if not datadir:
            self.params = self._format_default_params()
            return

        self.params = self._format_default_params()  # in case any file is corrupted

        to_write = False
        if os.path.isfile(self.params_file):
            if self._read():
                to_write = (
                    not self._add_default_params()
                )  # if new default data has been added
                if self._delete_old:  # or if old params have been deleted
                    to_write = True
            else:  # don't overwrite corrupted params, just print
                cloudlog.error("ERROR: Can't read op_params.json file")
        else:
            to_write = True  # user's first time running a fork with op_params, write default params

        if to_write:
            self._write()

    def get(
        self, key=None, default=None, force_update=False
    ):  # can specify a default value if key doesn't exist
        self._update_params(key, force_update)
        if key is None:
            return self._get_all()

        if key in self.params:
            key_info = self.key_info(key)
            if key_info.has_allowed_types:
                value = self.params[key]
                if type(value) in key_info.allowed_types:
                    if key_info.has_clip:
                        return clip(value, key_info.min, key_info.max)
                    else:
                        return value

                cloudlog.warning(f"op_param {key} invalid: {value}")
                if (
                    type(key_info.default) in key_info.allowed_types
                ):  # actually check if the default is valid
                    # return default value due to invalid type or clip
                    return key_info.default

                return self._value_from_types(
                    key_info.allowed_types
                )  # else use a standard value based on type (last resort to keep openpilot running if user's value is of invalid type)
            else:
                return self.params[
                    key
                ]  # no defined allowed types, returning user's value

        return default  # not in params

    def put(self, key, value):
        self.params.update({key: value})
        self._write()

    def delete(self, key):
        if key in self.params:
            del self.params[key]
            self._write()

    def key_info(self, key):
        key_info = KeyInfo()
        if key is None:
            return key_info
        if key in self.default_params:
            if "allowed_types" in self.default_params[key]:
                allowed_types = self.default_params[key]["allowed_types"]
                if isinstance(allowed_types, list) and len(allowed_types) > 0:
                    key_info.has_allowed_types = True
                    key_info.allowed_types = allowed_types

            if "live" in self.default_params[key]:
                key_info.live = self.default_params[key]["live"]

            key_info.has_description = "description" in self.default_params[key]

            if "hide" in self.default_params[key]:
                key_info.hidden = self.default_params[key]["hide"]

            # For safety, crash if clip is invalid.
            if "clip" in self.default_params[key]:
                c = self.default_params[key]["clip"]
                assert key_info.has_allowed_types, f"allowed_types required for {key}"
                assert (
                    type(c[0]) in key_info.allowed_types
                ), f"clip {c[0]} not in allowed types"
                assert (
                    type(c[1]) in key_info.allowed_types
                ), f"clip {c[1]} not in allowed types"
                assert len(c) == 2 and c[0] <= c[1], f"clip invalid for {key}"
                key_info.has_clip = True
                key_info.min = c[0]
                key_info.max = c[1]

            if "default" in self.default_params[key]:
                key_info.has_default = True
                key_info.default = self.default_params[key]["default"]
                if key_info.has_clip:
                    assert key_info.default == clip(
                        key_info.default, key_info.min, key_info.max
                    ), f"default {key_info.default} exceeds clip [{key_info.min},{key_info.max}]"

        return key_info

    def _add_default_params(self):
        prev_params = dict(self.params)
        for key in self.default_params:
            if self.force_update:
                self.params[key] = self.default_params[key]["default"]
            elif key not in self.params:
                self.params[key] = self.default_params[key]["default"]
        return prev_params == self.params

    def _format_default_params(self):
        return {key: self.default_params[key]["default"] for key in self.default_params}

    @property
    def _delete_old(self):
        deleted = False
        for i in self.to_delete:
            if i in self.params:
                del self.params[i]
                deleted = True
        return deleted

    def _get_all(self):  # returns all non-hidden params
        return {k: v for k, v in self.params.items() if not self.key_info(k).hidden}

    def _value_from_types(self, allowed_types):
        if list in allowed_types:
            return []
        elif float in allowed_types or int in allowed_types:
            return 0
        elif type(None) in allowed_types:
            return None
        elif str in allowed_types:
            return ""
        return None  # unknown type

    def _update_params(self, key, force_update):
        if (
            force_update or self.key_info(key).live
        ):  # if is a live param, we want to get updates while openpilot is running
            if datadir and (
                time.time() - self.last_read_time >= self.read_period or force_update
            ):  # make sure we aren't reading file too often
                if self._read():
                    self.last_read_time = time.time()

    def _read(self):
        try:
            with open(self.params_file, "r") as f:
                self.params = json.load(f)
            return True
        except Exception as e:
            cloudlog.error(e)
            self.params = self._format_default_params()
            return False

    def _write(self):
        if datadir:
            with open(self.params_file, "w") as f:
                json.dump(self.params, f, indent=2, sort_keys=True)
            os.chmod(self.params_file, 0o764)
