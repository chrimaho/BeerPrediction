#==============================================================================#
#                                                                              #
#    Title: Title                                                              #
#    Purpose: Purpose                                                          #
#    Notes: Notes                                                              #
#    Author: chrimaho                                                          #
#    Created: Date                                                             #
#    References: References                                                    #
#    Sources: Source                                                           #
#    Edited: DATE - Initial Creation                                           #
#                                                                              #
#==============================================================================#


#------------------------------------------------------------------------------#
#                                                                              #
#    /model/architecture                                                    ####
#                                                                              #
#------------------------------------------------------------------------------#

def get_architecture():
    
    # Imports
    import sys
    import io
    import torch

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


#------------------------------------------------------------------------------#
#                                                                              #
#    /                                                                      ####
#                                                                              #
#------------------------------------------------------------------------------#

def read_html(path:str="./docs/info.html"):
    with open(path, "rt") as file:
        data = file.read()
    return data

