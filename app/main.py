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
# import torch
# from torch import nn
# from torch.nn import functional as F
# from torch.utils.data import Dataset, DataLoader


#------------------------------------------------------------------------------#
# Set Directory                                                             ####
#------------------------------------------------------------------------------#

# Ensure the directory is correct... every time ----
for _ in range(5):
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

# Heading ----
from src.prod import production as p


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
# Root                                                                      ####
#------------------------------------------------------------------------------#

@app.get("/", response_class=HTMLResponse)
def read_root():
    return Response(content=p.read_html(), media_type="text/html")


#------------------------------------------------------------------------------#
# Health                                                                    ####
#------------------------------------------------------------------------------#

@app.get("/health", status_code=200, response_class=PlainTextResponse)
def healthcheck():
    return "App is ready to go."


#------------------------------------------------------------------------------#
# Architecture                                                              ####
#------------------------------------------------------------------------------#

@app.get("/model/architecture", response_class=PlainTextResponse)
def get_architecture():
    return p.get_architecture()