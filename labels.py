# 2nd 
# Define some helper functions to helps with the labels.
def get_class_names():
    """Return the list of classes in the Fashion-MNIST dataset."""
    return[
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle Boot",
    ]

def get_class_name(class_index):
    '''Return the class name for the given index'''
    return get_class_names()[class_index]

def get_class_index(class_name):
    '''Return the class index for the given name'''
    return get_class_names.index(class_name)

for class_index in range(10): # index 0 to 9
    print(f"class_index={class_index}, class_name={get_class_name(class_index)}")
# Python f-string (formatted string literal), which is a way to embed expressions inside string literals