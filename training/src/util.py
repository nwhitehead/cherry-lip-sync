import logging

LOG = logging.getLogger('Training')
logging.basicConfig(level=logging.INFO)

def hex_to_rgb(hex_color):
    """Converts a hex color code (6 digit) to RGB integers."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def truecolor(msg, color):
    r, g, b = hex_to_rgb(color)
    return f'\x1b[38;2;{r};{g};{b}m{msg}\x1b[0m'

def log_loss_color(prefix, msg):
    LOG.info(f'{prefix}{truecolor(msg, "#80ff80")}')
def log_epoch_color(prefix, msg):
    LOG.info(f'{prefix}{truecolor(msg, "#ffff80")}')
def log_validation_color(prefix, msg):
    LOG.info(f'{prefix}{truecolor(msg, "#fff0f0")}')
