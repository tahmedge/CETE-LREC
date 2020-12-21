## The models are implemented using the Huggingface Transformer: https://github.com/huggingface/transformers

Install the required components given in requirements.txt and setup.py. 

To run different models, see the run.txt file for instruction.

## To run the models in new datasets: 

Create a folder with your datasets name in uppercase without any spaces. (i.e., MYDATASET)

Put "training", "test", and "validation" files inside the folder "MYDATASET" in .tsv format where the prefix for each file should be the foldername in lowercase format followed by "\_filecategory.tsv". Each file should contain the question, answer, and the label separated by the "\\t" character (see the datasets inside the 'CETE Dataset' folder for more details). 

Thus, for the dataset named "MYDATASET", each filename inside the folder should be:

- training file: "mydataset_train.tsv" 
- validation file: "mydataset_valid.tsv"
- test file: "test_file.tsv"

### Now go to the "utils_glue.py" file inside the "examples" folder and do the following:

- Create a class named "MydatasetProcessor" (i.e., suppose that your dataset name is "MYDATASET"). You can check the classes such as TreccProcessor, TrecrProcessor, WikiProcdessor, YahooProcessor, Semeval201xProcessor etc. inside the "utils_glue.py" file to know more about how to write this class since the code inside your created class should be same as those classes. 

- Add codes inside the function "compute metrics" with your task_name == "mydataset" (i.e., your dataset name should be in lowercase).  See codes for treec, trecr, wiki, yahoo, semeval201x etc. to know more about how to write the codes inside this function since the code for your dataset should be same as those. 

- Add codes inside the dictionary "processors" in the following format: "mydataset": MydatasetProcessor

- Add codes inside the dictionary "output_modes" in the following format: "mydataset": "classification"

- Add codes inside the dictionary "GLUE_TASKS_NUM_LABELS" in the following format: "mydataset": 2

Note that you can edit the above two lines with other values based on your problem's requirement. 
