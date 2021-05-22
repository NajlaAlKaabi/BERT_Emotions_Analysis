import torch
from transformers import BertTokenizer
import statistics
from tqdm import tqdm
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
from torch.nn import Softmax, BCELoss
import numpy as np
import time
import datetime
import random
from os import path
import os


import data_prep
import paths


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


# Load BertForSequenceClassification, the pretrained BERT model with a single 
# linear classification layer on top
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", # base: 12-layer BERT model, uncased: uncased vocab
    num_labels = 3, # The number of output labels--2 in our case as the data has
                    # only negative and positive labels (but we will be able to 
                    # predict neutral after training based on the output probability)  
    output_attentions = False, # don't return attentions weights.
    output_hidden_states = False, # don't return all hidden-states.
)

# Move model to GPU if available, else CPU
model.to(device)

# Set our BERT model maximum length
# From function get_len_stats we got max tweet length is 135, we choose it
max_len = 135

# Batch size
# For fine-tuning BERT on a specific task, the authors recommend a batch size of 16 or 32.
batch_size = 32

# Number of training epochs. The BERT authors recommend between 2 and 4. 
epochs = 4


def get_len_stats():
    """
    Returns statistics about the length of processed sentences of train
    and test datasets. The length is the number of tokens after applying
    BERT tokenization
    This helps choose the appropriate maximum length of BERT model
    """
    train_exps_pro = data_prep.preprocess_examples(paths.TRAIN_CSV_FILE)
    test_exps_pro = data_prep.preprocess_examples(paths.TEST_CSV_FILE)
    train_lens, test_lens = [], [] #lists of lengths
    for tweet, score in tqdm(train_exps_pro):
        len_tweet = len(tokenizer.encode(tweet, add_special_tokens=True))
        train_lens.append(len_tweet)
    for tweet, score in tqdm(test_exps_pro):
        len_tweet = len(tokenizer.encode(tweet, add_special_tokens=True))
        test_lens.append(len_tweet)
    # compute statistics
    train_max, train_min = max(train_lens), min(train_lens)
    train_mean, train_std = statistics.mean(train_lens), statistics.stdev(train_lens)
    test_max, test_min = max(test_lens), min(test_lens)
    test_mean, test_std = statistics.mean(test_lens), statistics.stdev(test_lens)
    # show results
    print("="*6, "Train data lenght of tweets (with BERT tokenizer):", "="*6)
    print("Min: {}\nMax: {}\nAvg: {:.2f}\nStd.Dev: {:.2f}".format(
                          train_min, train_max, train_mean, train_std))
    print("="*6, "Test data lenght of tweets (with BERT tokenizer):", "="*6)
    print("Min: {}\nMax: {}\nAvg: {:.2f}\nStd.Dev: {:.2f}".format(
                          test_min, test_max, test_mean, test_std))

    return
    

def get_tensors(examples):
    """
    Tokenizes examples and prepares all tensors required to train BERT

    Args:
        examples (list): typically processed examples of Sentiment140 dataset, 
            where each element of the list is a tuple (tweet_text, score)
            where score is the polarity of the tweet (0=negative, 4=positive)
    Returns:
        (input_ids, attention_masks, labels) : tuple of tensors of input ids and
            attention masks and labels of input examples respectively
    """
    input_ids = []
    attention_masks = []
    labels = []
    
    for exp in tqdm(examples):
            
            tweet = exp[0]
            score = exp[1]
            if score==0:
                label=[1., 0., 0.]
            elif score==2:
                label= [0., 1., 0.]
            elif score==4:
                label=[0., 0., 1.]
            else:
                raise ValueError("Score {} not recognized, should be 0, 2 or 4")
            
        # `encode_plus` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        #   (5) Pad or truncate the sentence to max_len
        #   (6) Create attention masks for [PAD] tokens.
            encoded_dict = tokenizer.encode_plus(
                  tweet,                        # text to encode.
                  add_special_tokens = True,    # Add '[CLS]' and '[SEP]'
                  max_length = max_len,         # Maximum input length
                  truncation=True,              # truncate to max_len if applicable
                  padding = 'max_length',       # Pad to max_len if applicable
                  return_attention_mask = True, # Construct attn. masks
                  return_tensors = 'pt',        # Return pytorch tensors.
                  )
            # Add the encoded tweet to the list.    
            input_ids.append(encoded_dict['input_ids'])
            # And its attention mask (simply differentiates padding from non-padding).
            attention_masks.append(encoded_dict['attention_mask'])
            # Add label
            labels.append(label)

    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0) #of shape (nbr_examples, max_len)
    attention_masks = torch.cat(attention_masks, dim=0) #of shape (nbr_examples, max_len)
    labels = torch.tensor(labels) #of shape (nbr_examples)
    
    return input_ids, attention_masks, labels


def get_train_val_datasets(train_perc=0.8):
    """
    Returns datasets of train and validation

    Args:
        train_perc (float): percentage of training split

    Returns:
        (train_dataset, val_dataset): train and validation datasets. Each
            one is a TensorDataset object
    """
    # Get train data examples processed
    examples = data_prep.preprocess_examples(paths.TRAIN_CSV_FILE)
    # Get tensors
    input_ids, att_masks, labels = get_tensors(examples)
    # Create train dataset
    dataset = TensorDataset(input_ids, att_masks, labels)
    # Get train/bal size
    train_size = int(train_perc * len(dataset))
    val_size = len(dataset) - train_size
    # Divide the dataset by randomly selecting samples.
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    return train_dataset, val_dataset


def get_train_val_dataloaders(train_perc=0.8):
    """
    Returns dataloaders of train and validation dataset

    Args:
        train_perc (float): percentage of training split

    Returns:
        (train_dataloader, val_dataloader): train and validation dataloaders. 
            Each one is a DataLoader object
    """
    global batch_size

    train_dataset, val_dataset = get_train_val_datasets(train_perc)
    # Train dataloader: we take training samples in random order. 
    train_dataloader = DataLoader(
                train_dataset,  # The training samples.
                sampler = RandomSampler(train_dataset), # Select batches randomly
                batch_size = batch_size # Trains with this batch size.
            )

    # Validation dataloader: here the order doesn't matter, so we can read sequentially
    val_dataloader = DataLoader(
                val_dataset, # The validation samples.
                sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
                batch_size = batch_size # Evaluate with this batch size.
            )
    return train_dataloader, val_dataloader


def get_accuracy(preds, labels):
    """
    Calculate the accuracy of predictions vs labels
    """
    # get list of indices of highest probability for each prediction
    pred_flat = np.argmax(preds, axis=1).flatten() 
    labels_flat = np.argmax(labels, axis=1).flatten() 

    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))



def main():
    # ========================================
    #              Training Loop
    # ========================================
    # This training code is based on the `run_glue.py` script here:
    # https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128

    # set train percentage
    train_perc = 0.8

    # train and val dataloaders 
    train_dataloader, val_dataloader = get_train_val_dataloaders(train_perc)

    # learning rate
    lr = 2e-5

    optimizer = AdamW(model.parameters(),
                      lr = lr, # args.learning_rate - default is 5e-5
                      eps = 1e-8 # args.adam_epsilon  - default is 1e-8
                    )

    # Total number of training steps is [number of batches] x [number of epochs]. 
    # (Note that this is not the same as the number of training samples).
    total_steps = len(train_dataloader) * epochs

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = 0, # Default value in run_glue.py
                                                num_training_steps = total_steps)


    # Set the seed value for reproducibility.
    seed_val = 42

    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    # Stats: To store relevant infos such as train/val loss, val. accuracy, and timings.
    training_stats = []

    # Stats: To measure the total training time
    total_t0 = time.time()

    # Define softmax and loss function (binary cross entropy)
    softmax = Softmax(dim=1)
    cross_entr = BCELoss()

    # initialise best val accuracy for comparison later
    best_val_accuracy = 0.

    for epoch_i in range(0, epochs):
        #------------------ current epoch Training  ------------------  
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')
        # init time
        t0 = time.time()
        # Reset epoch loss
        total_train_loss = 0
        # set training mode
        model.train()

        for step, batch in enumerate(train_dataloader):
            # Progress update every 1000 batches.
            if step %50 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
            # batch contains three pytorch tensors:
            #   [0]: input ids 
            #   [1]: attention masks
            #   [2]: labels 
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            # Clear previously calculated gradients before backward pass. 
            # PyTorch doesn't do this automatically because its relevant for RNNs
            model.zero_grad()        
            # Perform a forward pass (evaluate model on current batch)
            outputs = model(b_input_ids, 
                                 token_type_ids=None, 
                                 attention_mask=b_input_mask)
            logits = outputs.logits
            # Add softmax to get probabilities
            #print("logits",logits)
            probs = softmax(logits)
            # Compute cross entropy loss
            #print("probs",probs.shape)
            #print("b_labels",b_labels.shape)
            loss = cross_entr(probs, b_labels);
            # Accumulate loss for average loss calculation later
            total_train_loss += loss.item()
            # Backward pass to calculate gradients
            loss.backward()
            # Clip the norm of the gradients to 1.0.
            # Goal: help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # optimization step (change model params) based on computed gradients
            optimizer.step()
            # Update the learning rate.
            scheduler.step()
        #------------------ end of current epoch Training  ------------------ 

        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_dataloader)            
        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(training_time))
            
        #------------------ Validation after current epoch  ------------------ 
        print("")
        print("Running Validation...")

        t0 = time.time()
        # set evaluation mode
        model.eval()
        # Tracking variables 
        total_eval_accuracy = 0
        total_eval_loss = 0
        nb_eval_steps = 0

        # Evaluate data for one epoch
        for batch in val_dataloader:
            
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            
            # Avoid pytorch constructing the compute graph during forward pass
            with torch.no_grad():        
              outputs = model(b_input_ids, 
                        token_type_ids=None, 
                        attention_mask=b_input_mask)
              logits = outputs.logits
              # Add softmax to get probabilities
              probs = softmax(logits)
              # Compute cross entropy loss
              loss = cross_entr(probs, b_labels)
            # Accumulate the validation loss.
            total_eval_loss += loss.item()
            # Move probs and labels to CPU
            probs = probs.to('cpu').numpy()
            label_ids = b_labels.to('cpu').numpy()
            # Calculate the accuracy of current batch anc accumulate it over all batches.
            total_eval_accuracy += get_accuracy(probs, label_ids)
            
        # Report the final accuracy for this validation run.
        avg_val_accuracy = total_eval_accuracy / len(val_dataloader)
        print("  Accuracy: {0:.2f}".format(avg_val_accuracy))
        # Calculate the average loss over all of the batches.
        avg_val_loss = total_eval_loss / len(val_dataloader)
        # Measure how long the validation run took.
        validation_time = format_time(time.time() - t0)
        
        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        print("  Validation took: {:}".format(validation_time))

        # Record all statistics from this epoch.
        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Valid. Accur.': avg_val_accuracy,
                'Training Time': training_time,
                'Validation Time': validation_time
            })
        # store model with all infos if validation loss is better than previous
        if avg_val_accuracy>best_val_accuracy:
            best_val_accuracy = avg_val_accuracy
            config = {'seed': seed_val,
                      'batch_size': batch_size,
                      'max_len': max_len,
                      'nbr_train_examples': len(train_dataloader)*batch_size,
                      'lr': lr,
                      'train_perc': train_perc,
                      'saved epoch (start from 1)': epoch_i+1 
                      }
            timenow = format_time(time.time())[-8:].replace(':', '_')
            # name of directory to save model and config and results to
            dirname = 'bert_sentiment140_' + \
                       str(len(train_dataloader)*batch_size) + "records_" + \
                       str(best_val_accuracy)[:4]+ "acc_" + \
                       timenow
            # name of file to save model into
            filename = dirname + '.pt'
            # path of directory
            dirpath = path.join(paths.TRAINED_MODELS, dirname)
            print("Saving model & config & stats to: \n{}...".format(dirpath))
            # path of model file
            filepath = path.join(paths.TRAINED_MODELS, dirname, filename)
            # create the folder
            os.mkdir(dirpath)
            # save model to file
            torch.save(model, filepath)
            # add config to the folder
            configpath = path.join(paths.TRAINED_MODELS, dirname, 'config.txt')
            with open(configpath, 'w') as f:
                for key, value in config.items():
                    f.write(key + ": "+ str(value) + '\n')
            # add final stats to the folder
            statspath = path.join(paths.TRAINED_MODELS, dirname, 'stats.txt')
            with open(statspath, 'w') as f:
                for stats in training_stats:
                    for key, value in stats.items():
                        f.write(key + ": "+ str(value) + '\n')
                f.write('-'*10)
            print('DONE')

        

    print("")
    print("Training complete!")

    print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))


    print('\n', training_stats)


if __name__=="__main__":
    main()




