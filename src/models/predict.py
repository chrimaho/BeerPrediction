#==============================================================================#
#                                                                              #
#    Title: Title                                                              #
#    Purpose: Purpose                                                          #
#    Notes: Notes                                                              #
#    Author: chrimaho                                                          #
#    Created: Created                                                          #
#    References: References                                                    #
#    Sources: Sources                                                          #
#    Edited: Edited                                                            #
#                                                                              #
#==============================================================================#



#------------------------------------------------------------------------------#
#                                                                              #
#    Set Up                                                                 ####
#                                                                              #
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
# Imports                                                                   ####
#------------------------------------------------------------------------------#

import torch
import numpy as np




#------------------------------------------------------------------------------#
#                                                                              #
#    Do work                                                                ####
#                                                                              #
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
# Prepare Data                                                              ####
#------------------------------------------------------------------------------#

def prepare_data \
    ( brewery_name:list=["Epic Ales"]
    , review_aroma:list=[1]
    , review_appearance:list=[1]
    , review_palate:list=[1]
    , review_taste:list=[1]
    , si_path:str="./models/encoders/si_handle_nan_brewery_name.joblib"
    , oe_path:str="./models/encoders/oe_numericify_brewery_name.joblib"
    , sc_path:str="./models/encoders/sc_scale_features.joblib"
    ):
    
    # Imports
    from src.utils import assertions as a
    from joblib import load
    import pandas as pd
    import numpy as np
    
    # Assertions
    assert a.all_list([brewery_name, review_aroma, review_appearance, review_palate, review_taste])
    assert a.all_str(brewery_name)
    assert all([a.all_float_or_int(param) for param in [review_aroma, review_appearance, review_palate, review_taste]])
    assert a.all_str([si_path, oe_path, sc_path])
    assert a.all_valid_path([si_path, oe_path, sc_path])
    
    # Loads
    si = load(si_path)
    oe = load(oe_path)
    sc = load(sc_path)
    
    # Transform brewery_name
    brewery_name = np.array(brewery_name, dtype="object").reshape(-1,1)
    brewery_name = si.transform(brewery_name)
    brewery_name = oe.transform(brewery_name)
    brewery_name = brewery_name.flatten()
    
    # Make pd.DataFrame
    data = pd.DataFrame ( \
        { "brewery_name": brewery_name
        , "review_aroma": review_aroma
        , "review_appearance": review_appearance
        , "review_palate": review_palate
        , "review_taste": review_taste
        })
    
    # Scale features
    data = sc.transform(data[["brewery_name", "review_aroma", "review_appearance", "review_palate", "review_taste"]])
    
    # Return
    return data


#------------------------------------------------------------------------------#
# Get Predictions                                                           ####
#------------------------------------------------------------------------------#

def predict_classification \
    ( test_data
    , model
    , batch_size:int=None
    , generate_batch=None
    ):
    """
    Run prediction for a Pytorch multi-class classification model

    Parameters
    ----------
    test_data : torch.utils.data.Dataset or np.ndarray
        Pytorch dataset
    model: torch.nn.Module
        Pytorch Model
    bacth_size : int
        Number of observations per batch
    collate_fn : function
        Function defining required pre-processing steps

    Returns
    -------
    torch.Tensor
        The tensor of predictions
    """    
    
    # Imports
    from src.utils import assertions as a
    from torch.utils.data import Dataset, DataLoader
    import torch
    from src.models.pytorch import PyTorchDataset
    
    # Set model to evaluation mode
    model.eval()
    
    # Check data class
    if not isinstance(test_data, Dataset):
        test_data = PyTorchDataset(test_data)
        
    # Check batch size
    if batch_size is None or batch_size>len(test_data):
        batch_size=len(test_data)
    
    # Create data loader
    load = DataLoader(test_data, batch_size=batch_size, collate_fn=generate_batch)
    
    # Iterate through data by batch of observations
    for indx, data in enumerate(load):
        
        # Extract data
        feature, _ = data
        
        # Set no update to gradients
        with torch.no_grad():
            
            # Make predictions
            output = model(feature)
            
    return output


#------------------------------------------------------------------------------#
# Decode Predictions                                                        ####
#------------------------------------------------------------------------------#

def decode_predictions(data:torch.Tensor, decoder_path:str="./models/encoders/le_numericify_beer_style.joblib"):
    
    # Imports
    from src.utils import assertions as a
    import torch
    import numpy as np
    from joblib import load
    
    # Assertions
    assert isinstance(data, torch.Tensor)
    assert a.all_str(decoder_path)
    
    # Make numpy
    nump = data.numpy()
    
    # Get index of predicted value
    nump = np.argmax(nump, axis=1)
    
    # Reshape to 2D array
    nump = nump.reshape(-1,1)
    
    # Load decoder
    decoder = load(decoder_path)
    
    # Get label
    labl = decoder.inverse_transform(nump)
    
    # Return
    return labl