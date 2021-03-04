#==============================================================================#
#                                                                              #
#    Title: ...                                                                #
#    Purpose: ...                                                              #
#    Notes: ...                                                                #
#    Author: chrimaho                                                          #
#    Created: 1/Mar/2020                                                       #
#    References: ...                                                           #
#    Sources: ...                                                              #
#    Edited: ...                                                               #
#                                                                              #
#==============================================================================#



#------------------------------------------------------------------------------#
#                                                                              #
#    Setup                                                                  ####
#                                                                              #
#------------------------------------------------------------------------------#


#------------------------------------------------------------------------------#
# Imports                                                                   ####
#------------------------------------------------------------------------------#

import os
import sys
import subprocess
from fastapi import FastAPI, Response
from fastapi.responses import HTMLResponse, PlainTextResponse


#------------------------------------------------------------------------------#
# Set Directory                                                             ####
#------------------------------------------------------------------------------#

# Ensure the directory is correct... every time ----
for i in range(5):
    if not os.getcwd().lower() == subprocess.run("git rev-parse --show-toplevel", stdout=subprocess.PIPE).stdout.decode("utf-8").replace("/","\\").strip().lower():
        os.chdir(".."),
    else:
        break
    
# Set up sys path environment ----
if not os.path.abspath(".") in sys.path:
    sys.path.append(os.path.abspath("."))
else:
    sys.path.remove(os.path.abspath("."))
    sys.path.append(os.path.abspath("."))


#------------------------------------------------------------------------------#
# Import Customs                                                            ####
#------------------------------------------------------------------------------#




#------------------------------------------------------------------------------#
# Instantiations                                                            ####
#------------------------------------------------------------------------------#


# API object ----
app = FastAPI()


#------------------------------------------------------------------------------#
#                                                                              #
#    Event Handlers                                                         ####
#                                                                              #
#------------------------------------------------------------------------------#


#------------------------------------------------------------------------------#
# Endpoints                                                                 ####
#------------------------------------------------------------------------------#

@app.get("/", response_class=HTMLResponse)
def read_root():
    
    with open('./app/info.html', "rt") as f:
        data = f.read()
            
    return Response(content=data, media_type="text/html")


#------------------------------------------------------------------------------#
# Health                                                                    ####
#------------------------------------------------------------------------------#

@app.get("/health", status_code=200, response_class=PlainTextResponse)
def healthcheck():
    return "App is ready to go."


#------------------------------------------------------------------------------#
# Architecture                                                              ####
#------------------------------------------------------------------------------#

@app.get("/model/architecture")
def get_architecture():
    return None