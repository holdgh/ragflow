#
#  Copyright 2024 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import sys
from api.utils.log_utils import logger

def python_version_validation():
    # Check python version
    required_python_version = (3, 11)
    if sys.version_info < required_python_version:
        logger.info(
            f"Required Python: >= {required_python_version[0]}.{required_python_version[1]}. Current Python version: {sys.version_info[0]}.{sys.version_info[1]}."
        )
        sys.exit(1)
    else:
        logger.info(f"Python version: {sys.version_info[0]}.{sys.version_info[1]}")


python_version_validation()

# Download nltk data
import nltk
nltk.download('wordnet')
nltk.download('punkt_tab')