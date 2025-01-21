#!/usr/bin/env -S python3

# MIT License
#
# Copyright (c) 2024 Orlando G. Toledano-López
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import uvicorn
import stk.simpleJSONLogger as jLogger
import api.main as eAPI

# Genearl global variables
jlog: jLogger.Logger = None

# Service global variables
apiEnv: dict[str, None] = {
    "BINARY_DISCRIMINATOR_API_PORT": None,
}

def hasEnvironmentVariables() -> bool:
    global apiEnv
    for key in apiEnv:
        apiEnv[key] = os.getenv(key)
        if apiEnv[key] is None:
            jlog.error(
                msg=f"environment variable '{key}' is not set",
                extra={"service": "translation_service_api"})
            return False
    return True

def main():
    # Set json logger
    global jlog, apiEnv
    jlog=jLogger.Logger(logLevel=jLogger.LogLevel.DEBUG)
    # check mandatory environment variables
    if not hasEnvironmentVariables():
        exit(1)
    # Configure and start the API service
    config = uvicorn.Config(
        eAPI.app, host="0.0.0.0", port=int(apiEnv["BINARY_DISCRIMINATOR_API_PORT"]),
        log_level=None)
    server = uvicorn.Server(config)
    server.run()

if __name__ == '__main__':
    main()
