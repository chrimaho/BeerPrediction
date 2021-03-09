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
    # if not os.getcwd().lower() == subprocess.run("git rev-parse --show-toplevel", stdout=subprocess.PIPE).stdout.decode("utf-8").replace("/","\\").strip().lower():
    if not os.getcwd().lower() == "BeerPrediction".lower():
        os.chdir("..")
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
@app.get \
    ( "/"
    , response_class=HTMLResponse
    , summary="Read homepage"
    , description="# Homepage\n\nOverview and details about the app.\n\nURL: [/](/)"
    , tags=["Homepage"]
    )
def read_overview():
    return Response(content=p.read_html(), media_type="text/html")


# /health ----
@app.get \
    ( "/health"
    , status_code=200
    , response_class=PlainTextResponse
    , summary="Health check"
    , description="# Health Check\n\nCheck to ensure that the app is healthy and ready to run.\n\nURL: [/health](/health)"
    , tags=["App Info"]
    )
def check_app_health():
    return "App is ready to go."


# /model/architecture ----
@app.get \
    ( "/model/architecture"
    , response_class=PlainTextResponse
    , summary="Model architecture"
    , description="# Model Architecture\n\nCheck to review the architecture of the model.\n\nURL: [/model/architecture](/model/architecture)"
    , tags=["App Info"]
    )
def get_model_architecture():
    return p.get_architecture()


# /beer/type ----
@app.get \
    ( "/beer/type"
    , response_class=PlainTextResponse
    , summary="Predict single beer type"
    , description="# Single Prediction\n\nPredict single Beer type, based on set input criteria.\n\nURL: [/beer/type](/beer/type)\n\n**NOTE**: This `GET` method will not perform the prediction. Please see below `POST` method for more details."
    , tags=["App Info"]
    )
def get_beer_type():
    return "Invalid server call. See /docs/ for help."


# /beers/type ----
@app.get \
    ( "/beers/type"
    , response_class=PlainTextResponse
    , summary="Predict multiple beer type"
    , description="# Multiple Prediction\n\nPredict multiple Beer types, based on set input criteria.\n\nURL: [/beers/type](/beers/type)\n\n**NOTE**: This `GET` method will not perform the prediction. Please see below `POST` method for more details."
    , tags=["App Info"]
    )
def get_beers_type():
    return "Invalid server call. See /docs/ for help."


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
    , responses={498: {"title":"Invalid Input", "description":"If any input params are invalid."}}
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
    , summary="Predict multiple beer types"
    , description=read("./docs/post_multiple_description.md")
    , responses={498: {"title":"Invalid Input", "description":"If any input params are invalid."}}
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
    from scipy.stats import mode
    
    # Validate params
    error = ""
    breweries = load("./data/processed/valid_breweries.joblib")
    for brewery in brewery_name:
        if not brewery in breweries:
            if len(error)>0: error += "\n"
            error += f"The brewery '{brewery}' is not valid."
    len_mode = mode([len(param) for param in [brewery_name, review_aroma, review_appearance, review_palate, review_taste]])[0][0]
    for param in ["review_aroma", "review_appearance", "review_palate", "review_taste"]:
        if len(eval(param)) != len_mode:
            if len(error)>0: error += "\n"
            error += f"All input params must have the same length. Param '{param}' has length '{len(eval(param))}', expecting '{len_mode}'."
        if np.any(np.array(eval(param)) <= 0) or np.any(np.array(eval(param)) > 5):
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