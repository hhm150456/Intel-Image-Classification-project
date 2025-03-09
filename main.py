import os
from data_preprocessing import processing  

"""
 Compining Data preprocessor with visualization in one class to check the results

 testing the length of splitting is right 

"""
project_path = os.path.dirname(os.path.abspath(__file__))
images_path = os.path.join(project_path, "data")  

preprocessor = processing.DataPreprocessor(images_path)

train_loader, val_loader, test_loader = preprocessor.prepare_data()
print(f"Train: {len(train_loader.dataset)} images, "
      f"Validation: {len(val_loader.dataset)} images, "
      f"Test: {len(test_loader.dataset)} images")

preprocessor.visualize_before_after()  


