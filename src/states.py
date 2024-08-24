import torch


def load_model_package_from_path(path, map_location=torch.device('cpu')):
    """ Load a model from a file path.
    
    Args:
        path (str): The file path to load the model from.
        map_location (torch.device, optional): The device to move the model to. Defaults to 'cpu'.

    Returns:
        package (dict): A dictionary containing the model's class, arguments, keyword arguments, and state.
    """

    package = torch.load(path, map_location=map_location)

    return package

def load_model_from_path(path, map_location=torch.device('cpu')):
    """ Load a model from a file path.
    
    Args:
        path (str): The file path to load the model from.
        map_location (torch.device, optional): The device to move the model to. Defaults to 'cpu'.

    Returns:
        nn.Module: The loaded model.
    """

    model = load_model_from_package(load_model_package_from_path(path, map_location=map_location))

    return model

def load_model_from_package(package):
    """ Load a model from a dictionary.
    
    Args:
        package (dict): A dictionary containing the model's class, arguments, keyword arguments, and state.
            
    Returns:
        nn.Module: The loaded model.
    """

    klass = package['klass']
    args = package['args']
    kwargs = package['kwargs']
    state_dict = package['state_dict']
    model = klass(*args, **kwargs)
    model.load_state_dict(state_dict)

    return model

def load_solver_package_from_path(path, map_location=torch.device('cpu')):
    """ Load a solver from a file path.
    
    Args:
        path (str): The file path to load the solver from.
        map_location (torch.device, optional): The device to move the solver to. Defaults to 'cpu'.

    Returns:
        package (dict): A dictionary containing the solver's class, arguments, keyword arguments, and state.
    """

    package = torch.load(path, map_location=map_location)

    return package
