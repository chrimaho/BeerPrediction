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
#    Setup                                                                  ####
#                                                                              #
#------------------------------------------------------------------------------#


#------------------------------------------------------------------------------#
# Imports                                                                   ####
#------------------------------------------------------------------------------#




#------------------------------------------------------------------------------#
#                                                                              #
#    Custom Functions                                                       ####
#                                                                              #
#------------------------------------------------------------------------------#

def info():
    """
    Displaying:
    1. A brief description of the project objectives,
    2. A list of endpoints, 
    3. Expected input parameters and output format of the model, 
    4. Link to the Github repo related to this project
    """
    
    from fastapi import Response
    
    with open('./app/info.html', "rt") as f:
        data = f.read()
            
    return Response(content=data, media_type="text/html")