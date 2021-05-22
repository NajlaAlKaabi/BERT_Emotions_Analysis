import torch
from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn import Softmax
import numpy as np
import csv
import paths
import data_prep


# Select device: GPU if available else CPU
if torch.cuda.is_available():    
    # Tell PyTorch to use the GPU.    
    device = torch.device("cuda")
    print('Using GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using CPU.')
    device = torch.device("cpu")


# Load BERT tokenizer.
print('Loading BERT tokenizer...')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)


# Loading trained model
model = torch.load(paths.BEST_MODEL)
model.to(device)
# set evaluation mode
model.eval()

max_len = 135

softmax = Softmax(dim=1)


def get_sentiment(text):
    """
    Returns probaicted sentiment for input text using fine-tuned BERT model
    """
    # Preprocess text
    text = data_prep.preprocess(text)
    # Tokenize text
    encoded_dict = tokenizer.encode_plus(
          text,                        
          add_special_tokens = True,    # Add '[CLS]' and '[SEP]'
          max_length = max_len,         # Maximum input length
          truncation=True,              # truncate to max_len if applicable
          padding = 'max_length',       # Pad to max_len if applicable
          return_attention_mask = True, # Construct attn. masks
          return_tensors = 'pt',        # Return pytorch tensors.
          )
    # Get input ids and attention masks  
    input_ids = encoded_dict['input_ids']
    att_masks = encoded_dict['attention_mask']

    # Move to GPU if available, else CPU
    input_ids = input_ids.to(device)
    att_masks = att_masks.to(device)

    # Feed to model and get probabilities
    outputs = model(input_ids, 
                    token_type_ids=None, 
                    attention_mask=att_masks)
    logits = outputs.logits
    probs = softmax(logits)

    return probs


def add_neutral_proba(probas):
    """
    Returns new probas with neutral proba added
    """
    # loop over probaictions and add computed probability of neutral 
    # class: from [proba_negative, proba_positive] to 
    # [proba_negative, proba_neutral, proba_positive]
    # We will use the following formula:
    #     proba_neutral = 1 - |proba_positive - proba_negative|
    # Note that now the probas of positive and negative and neutral
    # don't sum up to 1.
    new_probas = []
    for proba in probas:
        prob_neg = float(proba[0])
        prob_pos = float(proba[1])
        # compute probability of neutral
        prob_neu = 1 - abs(prob_pos - prob_neg)
        # add to new list of probaictions
        new_probas.append( [prob_neg, prob_neu, prob_pos] )

    return np.array(new_probas[0])

def get_examples(csv_file):
    """
    Parses input csv file of Sentiment140 dataset and returns list of examples

    Args:
        csv_file (str): path to csv file of Sentiment140 dataset containing 
            the following columns:
                1- target: polarity of the tweet (0=negative, 4=positive)
                2- ids: The id of the tweet ( e.g., 2087)
                3- date: the date of the tweet (e.g., Sat May 16 23:58:44 UTC 2009)
                4- flag: The query (e.g., lyx). If there is no query, this value is NO_QUERY.
                5- user: the user that tweeted (e.g., robotickilldozr)
                6- text: the text of the tweet (e.g., Lyx is cool)

    Returns:
        examples (list): extracted examples of input csv file, each element 
            in the list (i.e., example) is a tuple containing:
                (tweet_text, score)
            where score is the polarity of the tweet (0=negative, 4=positive)
    """
    all_examples = []
    with open(csv_file, encoding='utf-8-sig') as f:
        csv_reader = csv.reader(f, delimiter=',')
        for row in tqdm(csv_reader):
            msg = row[0]
            if msg: # tweet not empty
                all_examples.append(msg)
    return all_examples

def predict(probas):
    positive = probas[2]
    negative = probas[0]  
    neutral= probas[1]
     
def main():
    examples = get_examples(paths.NEW_CSV_FILE)
    for i in range(len(examples)):
        
        text = examples[i]
        probas = get_sentiment(text)
        # probas = add_neutral_proba(probas)
        
        id_sentiment = np.argmax(probas)
        sentiments = ['Negative', 'Neutral', 'Positive']
        print('\n','-*'*10)
        print('Sentiment: {}'.format(sentiments[id_sentiment]))
        print('-*'*10)
        print('Probas: \nNegative: {:.2f}\nNeutral: {:.2f}\nPositive: {:.2f}'.format(
                        probas[0],
                        probas[1],
                        probas[2]))
        print('-*'*10)



if __name__=="__main__":
    main()


