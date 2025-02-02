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
import api.main as eAPI
import logging
#import ssl

# Service global variables
apiEnv: dict[str, None] = {
    "DISCRIMINATORS_API_PORT": None,
}


#ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
#ssl_context.load_cert_chain('/path/to/cert.pem', keyfile='/path/to/key.pem')

def has_environment_vars() -> bool:
    global apiEnv
    for key in apiEnv:
        apiEnv[key] = os.getenv(key)
        if apiEnv[key] is None:
            logging.error(
                msg=f"environment variable '{key}' is not set",
                extra={"service": "translation_service_api"})
            return False
    return True

def main():
    global apiEnv

    # check mandatory environment variables
    if not has_environment_vars():
        exit(1)
    # Configure and start the API service
    config = uvicorn.Config(
        eAPI.app, host="0.0.0.0", port=int(apiEnv["DISCRIMINATORS_API_PORT"]),
        log_level=None) #ssl=ssl_context
    server = uvicorn.Server(config)
    server.run()

if __name__ == '__main__':
    main()
