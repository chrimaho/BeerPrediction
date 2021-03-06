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



#------------------------------------------------------------------------------#
#                                                                              #
#    /beer/type                                                             ####
#                                                                              #
#------------------------------------------------------------------------------#

def predict_single \
    ( brewery_name:str="Epic Ales"
    , review_aroma:float=1
    , review_appearance:float=1
    , review_palate:float=1
    , review_taste:float=1
    , modl_path:str="./models/predictors/beer_prediction.pth"
    ):
    
    # Imports
    from src.utils import assertions as a
    from src.models.predict import prepare_data, predict_classification, decode_predictions
    from src.models.pytorch import Net
    import torch
    # from joblib import load
    
    # Assertions
    assert a.all_str(brewery_name)
    assert all([a.all_float_or_int(param) for param in [review_aroma, review_appearance, review_palate, review_taste]])
    assert a.all_str(modl_path)
    assert a.all_valid_path(modl_path)
    
    # Loads
    # modl = load(modl_path)
    modl = Net(5, 104)
    modl.load_state_dict(torch.load(modl_path))

    
    # Prepare data
    data = prepare_data \
        ( brewery_name=brewery_name
        , review_aroma=review_aroma
        , review_appearance=review_appearance
        , review_palate=review_palate
        , review_taste=review_taste
        )
        
    # Predict data
    data = predict_classification(data, modl)
    
    # Decode data
    data = decode_predictions(data)
    
    # Flatten
    data = data.flatten()
        
    return data