def set_config_from_obj(attr, obj, config):
    if hasattr(obj, attr):
        value = getattr(obj, attr)
        setattr(config, attr, value)

def set_default_config(attr, config, value):
    if attr not in config:
        config[attr] = value
