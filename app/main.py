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
from fastapi.responses import HTMLResponse, PlainTextResponse, JSONResponse
from typing import List
from fastapi import Query
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

# Production functions ----
from src.prod import production as p
from src.utils.misc import read


#------------------------------------------------------------------------------#
# Meta Data                                                                 ####
#------------------------------------------------------------------------------#

tags_metadata = \
    [
        { "name": "Homepage"
        , "description": "The homepage for this app."
        }
    ,   { "name": "App Info"
        , "description": "Check more info about the status of this app and it's model."
        }
    ,   { "name": "Query Model"
        , "description": "Query information from the model."
        }
    ]


#------------------------------------------------------------------------------#
# Instantiations                                                            ####
#------------------------------------------------------------------------------#

# API object ----
app = FastAPI \
    ( title="Beer Prediction"
    , description="This is the description."
    , version="0.1.0"
    , openapi_tags=tags_metadata
    # , openapi_url=""
    )
    

#------------------------------------------------------------------------------#
#                                                                              #
#    Event Handlers                                                         ####
#                                                                              #
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
# GET                                                                       ####
#------------------------------------------------------------------------------#

# / ----
@app.get("/", response_class=HTMLResponse, tags=["Homepage"])
def read_overview():
    return Response(content=p.read_html(), media_type="text/html")

# /health ----
@app.get("/health", status_code=200, response_class=PlainTextResponse, tags=["App Info"])
def check_app_health():
    return "App is ready to go."

# /model/architecture ----
@app.get("/model/architecture", response_class=PlainTextResponse, tags=["App Info"])
def get_model_architecture():
    return p.get_architecture()


#------------------------------------------------------------------------------#
# POST                                                                      ####
#------------------------------------------------------------------------------#

# /beer/type ----
@app.post \
    ( path="/beer/type"
    , response_class=PlainTextResponse
    , tags=["Query Model"]
    , summary="Predict single beer types"
    , description=read("./docs/post_single_description.md")
    , responses={498: {"title":"Invalid Token", "description":"If any input params are invalid."}}
    )
def post_single \
    ( brewery_name:str=Query(..., title="Query string", description="Name of the Brewery (field: `brewery_name`).", )
    , review_aroma:float=Query(..., description="Score given for Aroma (field: `review_aroma`).")
    , review_appearance:float=Query(..., description="Score given for Appearance (field: `review_appearance`).")
    , review_palate:float=Query(..., description="Score given for Palate (field: `review_palate`).")
    , review_taste:float=Query(..., description="Score given for Taste (field: `review_taste`).")
    ):
    
    # Imports
    from joblib import load
    
    # Validate params
    error = ""
    if not brewery_name in load("./data/processed/valid_breweries.joblib"):
        error += f"The brewery '{brewery_name}' is not valid."
    for param in ["review_aroma", "review_appearance", "review_palate", "review_taste"]:
        if not 0 <= eval(param) < 5:
            error += f"\nThe value '{eval(param)}' for param '{param}' is invalid. Must be between '0' and '5'."
    if len(error)>0:
        return PlainTextResponse(error, status_code=498)
    
    # Get prediction
    pred = p.predict_single \
        ( brewery_name=[brewery_name]
        , review_aroma=[review_aroma]
        , review_appearance=[review_appearance]
        , review_palate=[review_palate]
        , review_taste=[review_taste]
        )
    
    # Return
    return PlainTextResponse(str(pred[0]))

# /beers/type ----
@app.post \
    ( "/beers/type"
    , response_class=PlainTextResponse
    , tags=["Query Model"]
    , summary="Predict single beer types"
    , description=read("./docs/post_multiple_description.md")
    , responses={498: {"title":"Invalid Token", "description":"If any input params are invalid."}}
    )
def post_multiple \
    ( brewery_name:List[str]=Query(..., description="List of Brewery names (field: `brewery_name`).")
    , review_aroma:List[float]=Query(..., description="List of Aroma scores (field: `review_aroma`).")
    , review_appearance:List[float]=Query(..., description="List of Appearance scores (field: `review_appearance`).")
    , review_palate:List[float]=Query(..., description="List of Palate scores (field: `review_palate`).")
    , review_taste:List[float]=Query(..., description="List of Taste scores (field: `review_taste`).")
    ):
    
    # Imports
    from joblib import load
    import numpy as np
    
    # Validate params
    error = ""
    breweries = load("./data/processed/valid_breweries.joblib")
    for brewery in brewery_name:
        if not brewery in breweries:
            error += f"The brewery '{brewery}' is not valid."
    for param in ["review_aroma", "review_appearance", "review_palate", "review_taste"]:
        if np.any(np.array(eval(param)) <= 0) and np.any(np.array(eval(param)) > 5):
            if len(error)>0: error += "\n"
            error += f"The values for param '{param}' is invalid. Must be between '0' and '5'."
    if len(error)>0:
        return PlainTextResponse(error, status_code=498)
    
    # Get prediction
    pred = p.predict_multiple \
        ( brewery_name=brewery_name
        , review_aroma=review_aroma
        , review_appearance=review_appearance
        , review_palate=review_palate
        , review_taste=review_taste
        )
    
    # Return
    return JSONResponse(list(pred))