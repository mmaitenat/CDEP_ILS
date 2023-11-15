def fill_log(log, **kwargs):
    # Initialize the log if it's empty
    if log is None:
        log = {}
    # Add the new information to the log. 
    for key, value in kwargs.items():
        # if key does not exist, create it
        if key not in log:
            log[key] = [value]
        # if key exists, append to it
        else:
            log[key].append(value)
    return log