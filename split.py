import os
import splitfolders
input_folder = r'C:\Users\cr008\OneDrive\Desktop\Iva\2. stopnja\2. letnik\Govorne tehnologije\Seminar\spoken_digits\\'
splitfolders.ratio(input_folder, # The location of dataset
                   output="splitted_data", # The output location
                   seed=42, # The number of seed
                   ratio=(.8, .2), # The ratio of splited dataset
                   group_prefix=None, # If your dataset contains more than one file like ".jpg", ".pdf", etc
                   move=False # If you choose to move, turn this into True
                   )
