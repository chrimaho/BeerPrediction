# Intro



#------------------------------------------------------------------------------#
#                                                                              #
#    Imports                                                                ####
#                                                                              #
#------------------------------------------------------------------------------#


from collections import OrderedDict
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader



#------------------------------------------------------------------------------#
#                                                                              #
#    Initial Functions                                                      ####
#                                                                              #
#------------------------------------------------------------------------------#


def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu') # don't have GPU 
    return device



#------------------------------------------------------------------------------#
#                                                                              #
#    Classes                                                                ####
#                                                                              #
#------------------------------------------------------------------------------#


#------------------------------------------------------------------------------#
# Data Set Class                                                            ####
#------------------------------------------------------------------------------#

class PyTorchDataset(Dataset):
        
    def __init__(self, feat, targ):
        self.feat = self.to_tensor(feat.copy()) #astype(np.float32)
        self.targ = self.to_tensor(targ.copy()) #astype(np.float32)
        
    def __len__(self):
        return len(self.targ)
        
    def __getitem__(self, index):
        return self.feat[index], self.targ[index]
        
    def to_tensor(self, data):
        return torch.Tensor(np.array(data))
    

#------------------------------------------------------------------------------#
# Network Class                                                             ####
#------------------------------------------------------------------------------#

# Define net
class Net(nn.Module):
    """Redefine class from torch"""

    # Initalise
    def __init__(self, feat_num_in, feat_num_out):
        """
        Initialise with all precidences.
        Then manually define layers to be used later
        """
        
        # Imports
        from src.utils import assertions as a
        
        # Super initialisation
        super().__init__()

        # Assertions
        assert a.all_int([feat_num_in, feat_num_out])
        
        # Fully connected layers
        # Classic conical shape
        # 5→10→20→40→80→100→104
        self.fc1 = nn.Linear(feat_num_in,5)
        self.fc2 = nn.Linear(5,10)
        self.fc3 = nn.Linear(10,20)
        self.fc4 = nn.Linear(20,40)
        self.fc5 = nn.Linear(40,80)
        self.fc6 = nn.Linear(80,100)
        self.out = nn.Linear(100,feat_num_out)
        self.softmax = nn.Softmax(dim=1)

    # Static propagation 😒
    def forward_1(self, x):
        """Run forward prop"""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.sigmoid(self.fc7(x))
        x = x[:,0]
        return x
    # Dynamic propagation 😁
    def forward_2 \
        ( self
        , feat
        , hidden_shapes:list=[20,20,20]
        , hidden_acti:str="relu"
        , final_shape:int=1
        , final_acti:str="sigmoid"
        ):

        # Check
        if type(feat)==np.ndarray:
            feat = torch.from_numpy(feat.astype(np.float32))
        
        # Assertions
        assert np.all([np.isscalar(param) for param in [hidden_acti, final_shape, final_acti]])
        assert isinstance(hidden_shapes, list)
        assert len(hidden_shapes)>0, "Must have at least 1 hidden layer"
        assert np.all([isinstance(elem, int) for elem in hidden_shapes])
        assert isinstance(final_shape, int)
        assert np.all([isinstance(param, str) for param in [hidden_acti, final_acti]])

        # Define number of nodes in input
        input_shape=feat.shape[-1]

        # Work on first hidden layer
        shape = nn.Linear(input_shape, hidden_shapes[0])
        x = shape(feat)
        x = eval("F.{}".format(hidden_acti))(x)

        # Loop other layers
        for layer in range(len(hidden_shapes)-1): #<-- `-1` because skip last layer
            
            # Get shapes
            curr_layer_shape = hidden_shapes[layer]
            next_layer_shape = hidden_shapes[layer+1]
            
            # Work on other hidden layers
            shape = nn.Linear(curr_layer_shape, next_layer_shape)
            x = shape(x)
            x = eval("F.{}".format(hidden_acti))(x)

        # Work on last hidden layer
        shape = nn.Linear(hidden_shapes[-1], final_shape)
        x = shape(x)
        x = eval("F.{}".format(final_acti))(x)

        # Return
        return x
    
    def forward(self, feat):
        X = F.dropout(F.relu(self.fc1(feat)), p=0.3, training=self.training)
        X = F.dropout(F.relu(self.fc2(X)), p=0.3, training=self.training)
        X = F.dropout(F.relu(self.fc3(X)), p=0.3, training=self.training)
        X = F.dropout(F.relu(self.fc4(X)), p=0.3, training=self.training)
        X = F.dropout(F.relu(self.fc5(X)), p=0.3, training=self.training)
        X = F.dropout(F.relu(self.fc6(X)), p=0.3, training=self.training)
        X = self.out(X)
        X = self.softmax(X)
        return X
        


#------------------------------------------------------------------------------#
#                                                                              #
#    Regression                                                             ####
#                                                                              #
#------------------------------------------------------------------------------#


#------------------------------------------------------------------------------#
# Train                                                                     ####
#------------------------------------------------------------------------------#


def train_regression(train_data, model, criterion, optimizer, batch_size, device, scheduler=None, collate_fn=None):
    """Train a Pytorch regresssion model

    Parameters
    ----------
    train_data : torch.utils.data.Dataset
        Pytorch dataset
    model: torch.nn.Module
        Pytorch Model
    criterion: function
        Loss function
    optimizer: torch.optim
        Optimizer
    bacth_size : int
        Number of observations per batch
    device : str
        Name of the device used for the model
    scheduler : torch.optim.lr_scheduler
        Pytorch Scheduler used for updating learning rate
    collate_fn : function
        Function defining required pre-processing steps

    Returns
    -------
    Float
        Loss score
    Float:
        RMSE Score
    """
    
    # Set model to training mode
    model.train()
    train_loss = 0

    # Create data loader
    data = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    
    # Iterate through data by batch of observations
    for feature, target_class in data:
        
        # Reset gradients
        optimizer.zero_grad()
        
        # Load data to specified device
        feature, target_class = feature.to(device), target_class.to(device)
        
        # Make predictions
        output = model(feature)
        
        # Calculate loss for given batch
        loss = criterion(output, target_class)
        
        # Calculate global loss
        train_loss += loss.item()
        
        # Calculate gradients
        loss.backward()
        
        # Update Weights
        optimizer.step()
        
    # Adjust the learning rate
    if scheduler:
        scheduler.step()

    return train_loss / len(train_data), np.sqrt(train_loss / len(train_data))



#------------------------------------------------------------------------------#
# Test                                                                      ####
#------------------------------------------------------------------------------#

def test_regression(test_data, model, criterion, batch_size, device, collate_fn=None):
    """Calculate performance of a Pytorch regresssion model

    Parameters
    ----------
    test_data : torch.utils.data.Dataset
        Pytorch dataset
    model: torch.nn.Module
        Pytorch Model
    criterion: function
        Loss function
    bacth_size : int
        Number of observations per batch
    device : str
        Name of the device used for the model
    collate_fn : function
        Function defining required pre-processing steps

    Returns
    -------
    Float
        Loss score
    Float:
        RMSE Score
    """    
    
    # Set model to evaluation mode
    model.eval()
    test_loss = 0

    # Create data loader
    data = DataLoader(test_data, batch_size=batch_size, collate_fn=collate_fn)
    
    # Iterate through data by batch of observations
    for feature, target_class in data:
        
        # Load data to specified device
        feature, target_class = feature.to(device), target_class.to(device)
        
        # Set no update to gradients
        with torch.no_grad():
            
            # Make predictions
            output = model(feature)
            
            # Calculate loss for given batch
            loss = criterion(output, target_class)
            
            # Calculate global loss
            test_loss += loss.item()
            
    return test_loss / len(test_data), np.sqrt(test_loss / len(test_data))




#------------------------------------------------------------------------------#
#                                                                              #
#    Binary Classification                                                  ####
#                                                                              #
#------------------------------------------------------------------------------#


#------------------------------------------------------------------------------#
# Train                                                                     ####
#------------------------------------------------------------------------------#

def train_binary\
    ( train_data
    , model
    , criterion
    , optimizer
    , batch_size
    , device
    , scheduler=None
    , generate_batch=None
    ):
    """Train a Pytorch binary classification model

    Parameters
    ----------
    train_data : torch.utils.data.Dataset
        Pytorch dataset
    model: torch.nn.Module
        Pytorch Model
    criterion: function
        Loss function
    optimizer: torch.optim
        Optimizer
    bacth_size : int
        Number of observations per batch
    device : str
        Name of the device used for the model
    scheduler : torch.optim.lr_scheduler
        Pytorch Scheduler used for updating learning rate
    collate_fn : function
        Function defining required pre-processing steps

    Returns
    -------
    Float
        Loss score
    Float:
        Accuracy Score
    """
    
    # Set model to training mode
    model.train()
    train_loss = 0
    train_acc = 0
    
    # Create data loader
    data = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=generate_batch)
    
    # Iterate through data by batch of observations
    for feature, target_class in data:

        # Reset gradients
        optimizer.zero_grad()
        
        # Load data to specified device
        feature, target_class = feature.to(device), target_class.to(device).to(torch.float32)
        
        # Make predictions
        output = model(feature)
        
        # Calculate loss for given batch
        loss = criterion(output, target_class.unsqueeze(1))
        
        # Calculate global loss
        train_loss += loss.item()
        
        # Calculate gradients
        loss.backward()

        # Update Weights
        optimizer.step()
        
        # Calculate global accuracy
        train_acc += (output.argmax(1) == target_class).sum().item()

    # Adjust the learning rate
    if scheduler:
        scheduler.step()

    return train_loss / len(train_data), train_acc / len(train_data)


#------------------------------------------------------------------------------#
# Test                                                                      ####
#------------------------------------------------------------------------------#

def test_binary \
    (test_data
    , model
    , criterion
    , batch_size
    , device
    , generate_batch=None
    ):
    """Calculate performance of a Pytorch binary classification model

    Parameters
    ----------
    test_data : torch.utils.data.Dataset
        Pytorch dataset
    model: torch.nn.Module
        Pytorch Model
    criterion: function
        Loss function
    bacth_size : int
        Number of observations per batch
    device : str
        Name of the device used for the model
    collate_fn : function
        Function defining required pre-processing steps

    Returns
    -------
    Float
        Loss score
    Float:
        Accuracy Score
    """    
    
    # Set model to evaluation mode
    model.eval()
    test_loss = 0
    test_acc = 0
    
    # Create data loader
    data = DataLoader(test_data, batch_size=batch_size, collate_fn=generate_batch)
    
    # Iterate through data by batch of observations
    for feature, target_class in data:
        
        # Load data to specified device
        feature, target_class = feature.to(device), target_class.to(device).to(torch.float32)
        
        # Set no update to gradients
        with torch.no_grad():
            
            # Make predictions
            output = model(feature)
            
            # Calculate loss for given batch
            loss = criterion(output, target_class.unsqueeze(1))

            # Calculate global loss
            test_loss += loss.item()
            
            # Calculate global accuracy
            test_acc += (output.argmax(1) == target_class).sum().item()

    return test_loss / len(test_data), test_acc / len(test_data)



#------------------------------------------------------------------------------#
#                                                                              #
#    Multi-Class Classification                                             ####
#                                                                              #
#------------------------------------------------------------------------------#


#------------------------------------------------------------------------------#
# Train                                                                     ####
#------------------------------------------------------------------------------#

def train_classification \
    ( train_data
    , model
    , criterion
    , optimizer
    , batch_size
    , device
    , scheduler=None
    , generate_batch=None
    ):
    """Train a Pytorch multi-class classification model

    Parameters
    ----------
    train_data : torch.utils.data.Dataset
        Pytorch dataset
    model: torch.nn.Module
        Pytorch Model
    criterion: function
        Loss function
    optimizer: torch.optim
        Optimizer
    bacth_size : int
        Number of observations per batch
    device : str
        Name of the device used for the model
    scheduler : torch.optim.lr_scheduler
        Pytorch Scheduler used for updating learning rate
    collate_fn : function
        Function defining required pre-processing steps

    Returns
    -------
    Float
        Loss score
    Float:
        Accuracy Score
    """
    
    # Set model to training mode
    model.train()
    train_loss = 0
    train_acc = 0
    
    # Create data loader
    data = DataLoader \
        ( train_data
        , batch_size=batch_size
        , shuffle=True
        , collate_fn=generate_batch
        )
    
    # Iterate through data by batch of observations
    for feature, target_class in data:

        # Reset gradients
        optimizer.zero_grad()
        
        # Load data to specified device
        feature.to(device)
        target_class.to(device)
        
        # Make predictions
        output = model(feature)
        
        # Calculate loss for given batch
        loss = criterion(output, target_class.long())

        # Calculate global loss
        train_loss += loss.item()
        
        # Calculate gradients
        loss.backward()

        # Update Weights
        optimizer.step()
        
        # Calculate global accuracy
        train_acc += (output.argmax(1) == target_class).sum().item()

    # Adjust the learning rate
    if scheduler:
        scheduler.step()

    return train_loss / len(train_data), train_acc / len(train_data)


#------------------------------------------------------------------------------#
# Test                                                                      ####
#------------------------------------------------------------------------------#

def test_classification \
    ( test_data
    , model
    , criterion
    , batch_size
    , device
    , generate_batch=None
    ):
    """Calculate performance of a Pytorch multi-class classification model

    Parameters
    ----------
    test_data : torch.utils.data.Dataset
        Pytorch dataset
    model: torch.nn.Module
        Pytorch Model
    criterion: function
        Loss function
    bacth_size : int
        Number of observations per batch
    device : str
        Name of the device used for the model
    collate_fn : function
        Function defining required pre-processing steps

    Returns
    -------
    Float
        Loss score
    Float:
        Accuracy Score
    """    
    
    # Set model to evaluation mode
    model.eval()
    test_loss = 0
    test_acc = 0
    
    # Create data loader
    data = DataLoader(test_data, batch_size=batch_size, collate_fn=generate_batch)
    
    # Iterate through data by batch of observations
    for feature, target_class in data:
        
        # Load data to specified device
        feature, target_class = feature.to(device), target_class.to(device)
        
        # Set no update to gradients
        with torch.no_grad():
            
            # Make predictions
            output = model(feature)
            
            # Calculate loss for given batch
            loss = criterion(output, target_class.long())

            # Calculate global loss
            test_loss += loss.item()
            
            # Calculate global accuracy
            test_acc += (output.argmax(1) == target_class).sum().item()

    return test_loss / len(test_data), test_acc / len(test_data)



#------------------------------------------------------------------------------#
#                                                                              #
#    Visualisation                                                          ####
#                                                                              #
#------------------------------------------------------------------------------#

def plot_network_training(metrics:dict):
    
    # Imports
    from IPython.display import clear_output
    import numpy as np
    import matplotlib.pyplot as plt
    from src.utils import assertions as a

    # Assertions
    assert isinstance(metrics, dict)
    assert a.all_in(["accu_trn", "loss_trn", "accu_val", "loss_val"], list(metrics.keys()))

    # If only 1 score, then end
    epoch = len(next(iter(metrics.values())))
    if epoch < 2:
        return None

    # Clearn previous output
    clear_output(wait=True)

    # Define space
    N = np.arange(1, len(next(iter(metrics.values())))+1)

    # You can chose the style of your preference
    # print(plt.style.available) to see the available options
    #plt.style.use("seaborn")

    # Plot train loss, train acc, val loss and val acc against epochs passed
    plt.figure(figsize=(8,8))
    
    # Accuracy
    plt.subplot(2,1,1)
    plt.plot(N, metrics.get("accu_trn"), label = "Training Accuracy")
    plt.plot(N, metrics.get("accu_val"), label = "Validation Accuracy")
    plt.legend(loc="best")
    plt.title("Accuracy [Epoch {}]".format(epoch))
    plt.ylim([0,1.1])
    plt.ylabel("Accuracy")
    
    # Loss
    plt.subplot(2,1,2)
    plt.plot(N, metrics.get("loss_trn"), label = "Training Loss")
    plt.plot(N, metrics.get("loss_val"), label = "Validation Loss")
    plt.legend(loc="best")
    plt.title("Loss [Epoch {}]".format(epoch))
    plt.ylabel("Loss")
    plt.xlabel("Epoch #")
    
    # Show
    plt.show()




#------------------------------------------------------------------------------#
#                                                                              #
#    Generic                                                                ####
#                                                                              #
#------------------------------------------------------------------------------#



#------------------------------------------------------------------------------#
# Set                                                                       ####
#------------------------------------------------------------------------------#


def model_set \
    ( first_shape:int=5
    , hidden_shapes:list=[20,30,40]
    , hidden_acti:torch.nn.modules.activation=nn.ReLU()
    , final_shape:int=1
    , final_acti:torch.nn.modules.activation=nn.Softmax(dim=1)
    , dropout:float=0.2
    ):
    
    # Imports
    from src.utils import assertions as a
    from torch import nn
    from collections import OrderedDict
    
    # Assertions
    assert a.all_int([first_shape, final_shape])
    assert a.all_int(hidden_shapes)
    assert a.all_float(dropout)
    assert hidden_acti.__module__ == "torch.nn.modules.activation"
    assert final_acti.__module__ == "torch.nn.modules.activation"
    
    # Set output
    modl = []
    
    # Add first layer
    modl.extend (\
        [ ("shap_frst", nn.Linear(first_shape, hidden_shapes[0]))
        , ("acti_frst", hidden_acti)
        , ("regl_frst", nn.Dropout(p=dropout))
        ])
        
    # Loop through each hidden layer
    for idx, layer in enumerate(hidden_shapes):
        
        # Skip the final layer
        if idx+1==len(hidden_shapes): break
        
        # Add hidden layer
        modl.extend (\
            [ ("shap_{:02}".format(idx+1), nn.Linear(layer, hidden_shapes[idx+1]))
            , ("acti_{:02}".format(idx+1), hidden_acti)
            , ("regl_{:02}".format(idx+1), nn.Dropout(p=dropout))
            ])
            
    # Add final layer
    modl.extend (\
        [ ("shap_finl", nn.Linear(hidden_shapes[-1], final_shape))
        , ("acti_finl", final_acti)
        ])
            
    # Form in to OrderedDict()
    modl = OrderedDict(modl)
    
    # Form in to Sequential()
    modl = nn.Sequential(modl)
    
    # Return
    return modl


#------------------------------------------------------------------------------#
# Train                                                                     ####
#------------------------------------------------------------------------------#

def model_train \
    ( data_trn:torch.utils.data.Dataset
    , modl:torch.nn.Module
    , crit:torch.nn
    , optm:torch.optim
    , batch_size:int=100
    , hidden_shapes:list=[20,30,40]
    , hidden_acti:str="relu"
    , final_shape:int=1
    , final_acti:str="sigmoid"
    , device:torch.device=get_device()
    , scheduler:torch.optim.lr_scheduler=None
    ):

    # Set to train
    modl.train()
    loss_trn = 0.0
    accu_trn = 0.0

    # Set data generator
    load_trn = DataLoader(data_trn, batch_size=batch_size, shuffle=True, num_workers=0)

    # Loop over each batch
    for batch, data in enumerate(load_trn):
        
        # Extract data
        inputs, labels = data

        # Push data to device
        # inputs, labels = inputs.to(device), labels.to(device)
        inputs.to(device)
        labels.to(device)

        # Zero out the parameter gradients
        optm.zero_grad()

        # Feed forward
        output = modl \
            ( feat=inputs
            , hidden_shapes=hidden_shapes
            , hidden_acti=hidden_acti
            , final_shape=final_shape
            , final_acti=final_acti
            )

        # Calc loss
        loss = crit(output, labels.unsqueeze(1))

        # Global metrics
        loss_trn += loss.item()
        accu_trn += (output.argmax(1) == labels).sum().item()

        # Feed backward
        loss.backward()

        # Optimise
        optm.step()

    # Adjust scheduler
    if scheduler:
        scheduler.step()
    
    return loss_trn/len(data_trn), accu_trn/len(data_trn)


#------------------------------------------------------------------------------#
# Validate                                                                  ####
#------------------------------------------------------------------------------#

def model_validate \
    ( data_val:torch.utils.data.Dataset
    , modl:torch.nn.Module
    , crit:torch.nn
    , batch_size:int=100
    , hidden_shapes:list=[20,30,40]
    , hidden_acti:str="relu"
    , final_shape:int=1
    , final_acti:str="sigmoid"
    , device:torch.device=get_device()
    ):

    # Set to validation
    modl.eval()
    accu_val = 0
    loss_val = 0

    # Set generator
    load_val = DataLoader(data_val, batch_size=batch_size, shuffle=True, num_workers=0)

    # Loop over each batch
    for batch, data in enumerate(load_val):
        
        # Extract data
        inputs, labels = data

        # Push data to device
        # inputs, labels = inputs.to(device), labels.to(device)
        inputs.to(device)
        labels.to(device)

        # Don't update gradients
        with torch.no_grad():
            
            # Make predictions
            output = modl \
                ( feat=inputs
                , hidden_shapes=hidden_shapes
                , hidden_acti=hidden_acti
                , final_shape=final_shape
                , final_acti=final_acti
                )

            # Calculate loss
            loss = crit(output, labels.unsqueeze(1))

            # Global metrics
            loss_val += loss.item()
            accu_val += (output.argmax(1) == labels).sum().item()
    
    return loss_val/len(data_val), accu_val/len(data_val)


#------------------------------------------------------------------------------#
# Overall                                                                   ####
#------------------------------------------------------------------------------#

def train_overall_network \
    ( feat_trn:np.real
    , targ_trn:np.real
    , feat_val:np.real
    , targ_val:np.real
    , hidden_shapes:list=[20,20,20]
    , hidden_acti:str="relu"
    , final_shape:int=1
    , final_acti:str="sigmoid"
    , batch_size:int=100
    , epochs:int=500
    , learning_rate:float=0.001
    , device:torch.device=get_device()
    , scheduler:bool=True
    , verbosity:int=10
    , plot_learning:bool=True
    ):

    # Imports
    import numpy as np
    from src.utils import assertions as a
    from src.models.pytorch import PyTorchDataset
    from torch import nn, optim
    from src.models.pytorch import Net
    
    # Assertions
    assert a.all_real([feat_trn, targ_trn, feat_val, targ_val])
    assert isinstance(hidden_shapes, list)
    assert len(hidden_shapes)>0, "Must have at least 1 hidden layer"
    assert a.all_in(hidden_shapes)
    assert a.all_scalar([hidden_acti, final_shape, final_acti, batch_size, epochs, learning_rate])
    assert isinstance(verbosity, (int, type(None)))
    assert a.all_int([batch_size, epochs, verbosity])
    assert a.all_str([hidden_acti, final_acti])
    assert a.all_float(learning_rate)

    # Initialise data generators
    data_trn = PyTorchDataset(feat_trn, targ_trn)
    data_val = PyTorchDataset(feat_val, targ_val)

    # Initialise classes
    modl = Net(feat_trn.shape[1], len(set(targ_trn)))
    crit = nn.CrossEntropyLoss()
    optm = optim.Adam(modl.parameters(), lr=learning_rate)
    if scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optm, mode="min", patience=3)

    # Push network to device
    modl.to(device)
    
    # Set dumping ground
    costs = {"epoch": [], "loss_trn": [], "accu_trn": [], "loss_val": [], "accu_val": []}
    loss_trn = 0.0
    accu_trn = 0.0

    # Loop over epochs
    for epoch in range(epochs):

        loss_trn, accu_trn = model_train \
            ( data_trn=data_trn
            , modl=modl
            , crit=crit
            , optm=optm
            , batch_size=batch_size
            , hidden_shapes=hidden_shapes
            , hidden_acti=hidden_acti
            , final_shape=final_shape
            , final_acti=final_acti
            , device=device
            , scheduler=scheduler
            )
        
        loss_val, accu_val = model_validate \
            ( data_val=data_val
            , modl=modl
            , crit=crit
            , batch_size=batch_size
            , hidden_shapes=hidden_shapes
            , hidden_acti=hidden_acti
            , final_shape=final_shape
            , final_acti=final_acti
            , device=device
            )

        # Record progress
        costs["epoch"].append(epoch+1)
        costs["loss_trn"].append(loss_trn)
        costs["accu_trn"].append(accu_trn)
        costs["loss_val"].append(loss_val)
        costs["accu_val"].append(accu_val)

        # Adjust scheduler
        if scheduler:
            scheduler.step()

        # Print stats
        if verbosity:
            if epoch % verbosity == 0 or epoch+1==epochs:
                # Plot learning
                if plot_learning:
                    plot_network_training(costs)
                # Print metrics
                # print("Epoch: {}/{}\tLoss: {:.5f}".format(costs["epoch"][-1], epochs, costs["trn_los"][-1]))

    # Return
    return modl
