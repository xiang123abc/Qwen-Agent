# Copyright 2023 The Qwen team, Alibaba Group. All rights reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#    http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os


def _coerce_log_level(level=None):
    if level is None:
        env_level = os.getenv('QWEN_AGENT_LOG_LEVEL', '').strip().upper()
        if env_level:
            return getattr(logging, env_level, logging.INFO)
        if os.getenv('QWEN_AGENT_DEBUG', '0').strip().lower() in ('1', 'true'):
            return logging.DEBUG
        return logging.INFO
    if isinstance(level, str):
        return getattr(logging, level.strip().upper(), logging.INFO)
    return level


def setup_logger(level=None, log_file=None):
    level = _coerce_log_level(level)

    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(filename)s - %(lineno)d - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    _logger = logging.getLogger('qwen_agent_logger')
    _logger.propagate = False
    _logger.setLevel(level)

    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in _logger.handlers):
        _logger.addHandler(handler)

    if log_file:
        abs_log_file = os.path.abspath(log_file)
        if not any(isinstance(h, logging.FileHandler) and getattr(h, 'baseFilename', None) == abs_log_file
                   for h in _logger.handlers):
            file_handler = logging.FileHandler(abs_log_file, encoding='utf-8')
            file_handler.setFormatter(formatter)
            _logger.addHandler(file_handler)
    return _logger


logger = setup_logger()


def configure_logger(level=None, log_file=None):
    return setup_logger(level=level, log_file=log_file)
