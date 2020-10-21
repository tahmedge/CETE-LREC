### The models are implemented using the Huggingface Transformer: https://github.com/huggingface/transformers

### Install the required components given in requirements.txt and setup.py. 

### To run different models, see the run.txt file for instruction.

## To run the models in new datasets: 

Create a folder with your datasets name in uppercase without any spaces. (i.e., MYDATASET)

Put "training", "test", and "validation" files inside the folder "MYDATASET" in .tsv format where the prefix for each file should be the foldername in lowercase format followed by "\_filecategory.tsv". 

For the dataset named "MYDATASET", each filename inside the folder should be, 
training file: "mydataset_train.tsv" 
validation file: "mydataset_valid.tsv"
test file: "test_file.tsv").   
