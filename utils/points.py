def normalise_coords(array, max):
    return 2*(array / max) - 1
    
def denormalise_coords(array, max):
    return (array + 1)/2 * max