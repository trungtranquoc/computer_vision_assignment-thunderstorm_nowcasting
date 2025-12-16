import colorsys

def spectral_colors(n: int) -> list[list[int]]:
    """
    Generate n RGB colors following the visible spectrum.
    
    Parameters
    ----------
    n : int
        Number of colors to generate.
    
    Returns
    -------
    list[list[int]]
        List of RGB colors (each element is [R, G, B], with values 0-255).
    """
    colors = []
    for i in range(n):
        # Hue ranges from 0.0 (red) to 0.83 (violet) approximately
        hue = i / max(1, n - 1) * 0.83  
        r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        colors.append([int(r * 255), int(g * 255), int(b * 255)])
    return colors