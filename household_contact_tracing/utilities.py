def update_params(instance, params: dict):
    """Update instance variables with anything in params."""
    if params:
        for param_name in instance.__dict__:
            if param_name in params:
                instance.__dict__[param_name] = params[param_name]
