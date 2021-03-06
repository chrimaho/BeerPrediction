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
from fastapi.encoders import jsonable_encoder
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

# Post Single Description ----
post_single_description = \
"""
**/beer/type**

_Purpose_: Use [/beer/type](/beer/type) to query for only a single beer type.

_Expected Input String_: /beer/type?brewery_name=`brewery_name`&review_aroma=`review_aroma`&review_appearance=`review_appearance`&review_palate=`review_palate`&review_taste=`review_taste`

_Input Types_: As defined below. Specifically:
1. `brewery_name`: str
1. `review_aroma`: float
1. `review_appearance`: float
1. `review_palate`: float
1. `review_taste`: float

_Validations_:
1. `brewery_name`: Must be valid brewery name.
1. `review_aroma`, `review_appearance`, `review_palate`, `review_taste`: Must all be `float` values, between `0` and `5`.

_Example Input_: [/beer/type?brewery_name=Epic%20Ales&review_aroma=1&review_appearance=1&review_palate=1&review_taste=1](/beer/type?brewery_name=Epic%20Ales&review_aroma=1&review_appearance=1&review_palate=1&review_taste=1)

_Example Output_: 
"""

# /beer/type ----
@app.post \
    ( path="/beer/type"
    # , response_class=JSONResponse
    , tags=["Query Model"]
    , summary="Query single beer types"
    , description=post_single_description
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
    return PlainTextResponse(str(pred))

# /beers/type ----
@app.post("/beers/type", response_class=JSONResponse, tags=["Query Model"])
def query_miltiple_beer_types \
    ( brewery_name:List[str]=Query(default="None", description="List of Brewery names (field: `brewery_name`).")
    , review_aroma:List[float]=Query(default=1, description="List of `review_aroma` values.")
    , review_appearance:List[float]=Query(default=1, description="List of `review_appearance` values.")
    , review_palate:List[float]=Query(default=1, description="List of `review_palate` values.")
    , review_taste:List[float]=Query(default=1, description="List of `review_taste` values.")
    ):
    return object