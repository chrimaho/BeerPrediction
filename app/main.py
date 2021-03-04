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
import torch
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
    
    with open('./app/info.html', "rt") as file:
        data = file.read()
            
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

@app.get("/model/architecture", response_class=PlainTextResponse)
def get_architecture():

    # Imports
    import sys
    import io

    # Set process to capture output from `print()`
    old_stdout = sys.stdout
    new_stdout = io.StringIO()
    sys.stdout = new_stdout
    
    # Load model
    modl = torch.load("./models/predictors/beer_prediction.pt")
    
    # Print model summary
    print("Beer Prediction Architecture")
    print(modl)
    
    # Capture output
    output = new_stdout.getvalue()
    
    # Re-set output process
    sys.stdout = old_stdout
    
    # Return
    return output