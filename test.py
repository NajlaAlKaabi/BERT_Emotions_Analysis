import torch
from tqdm import tqdm
from transformers import BertTokenizer
from transformers import BertForSequenceClassification
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from torch.nn import Softmax, BCELoss
import numpy as np
import time
import datetime
import pandas as pd


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
batch_size = 32


def get_tensors(test_examples, with_neutral):
    """
    Tokenizes examples and prepares all tensors required to feed to BERT
    for getting predictions

    Args:
        examples (list): typically test examples of Sentiment140 dataset, 
            where each element of the list is a tuple (tweet_text, score)
            and score is the polarity of the tweet (0=negative, 2=neutral, 4=positive)
    Returns:
        (input_ids, attention_masks, labels) : tuple of tensors of input ids and
            attention masks and labels of input examples respectively
    """
    input_ids = []
    attention_masks = []
    labels = []
    print("get_tensors:with_neutral ",with_neutral)
    for exp in tqdm(test_examples):
        tweet = exp[0]
        score = exp[1]
        if with_neutral:
            #convert score to probas of each class (index0:neg, index1: neutral, index1:pos)
            if score==0:
                label = [1., 0., 0.]
            elif score==2: 
                label = [0., 1., 0.]
                
            elif score==4:
                label = [0., 0., 1.]
            else:
                raise ValueError("Score {} not recognized, should be 0, 2 or 4")
        else:
            #convert score to probas of each class (index0:neg, index1: neutral, index1:pos)
            if score==0:
                label = [1., 0.]
            elif score==4: 
                label = [0., 1.]
            else:
                raise ValueError("Score {} not recognized, should be 0(negative) or 4(positive)")

        encoded_dict = tokenizer.encode_plus(
                  tweet,                        # text to encode.
                  add_special_tokens = True,    # Add '[CLS]' and '[SEP]'
                  max_length = max_len,         # Maximum input length
                  truncation=True,              # truncate to max_len if applicable
                  padding = 'max_length',       # Pad to max_len if applicable
                  return_attention_mask = True, # Construct attn. masks
                  return_tensors = 'pt',        # Return pytorch tensors.
                  )
        # Add to global lists    
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
        labels.append(label)

    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0) #of shape (nbr_examples, max_len)
    attention_masks = torch.cat(attention_masks, dim=0) #of shape (nbr_examples, max_len)
    labels = torch.tensor(labels) #of shape (nbr_examples)

    return input_ids, attention_masks, labels


def get_test_dataloader(with_neutral):
    """
    Returns test dataloader (DataLoader obj) for Sentiment140 dataset
    """
    global batch_size
    print("get_test_dataloader with_neutral",with_neutral)
    # Get test data examples processed
    examples = data_prep.preprocess_examples(paths.TEST_CSV_FILE)
    if not with_neutral:
        # remove examples with socre=2 (i.e., neutral)
        for i,exp in enumerate(examples):
            if exp[1]==2:
                del examples[i]
                print("row deleted")
    # Get tensors
    input_ids, att_masks, labels = get_tensors(examples, with_neutral)
    # Create test dataset
    test_dataset = TensorDataset(input_ids, att_masks, labels)
    # Create test dataloader: we take samples in sequential order. 
    test_dataloader = DataLoader(
                test_dataset,
                sampler = SequentialSampler(test_dataset),
                batch_size = batch_size
            )

    return test_dataloader



def get_class_ids(pred_probas, labels_probas, with_neutral):
    """
    Converts input prediction and labels list of probabilities to their 
    corresponding list of classes indices
    """
    
    #QQQ: why this treatment to nuetral? isn't there is a class trained for it
    # if with_neutral:
    #     # loop over predictions and add computed probability of neutral 
    #     # class: from [proba_negative, proba_positive] to 
    #     # [proba_negative, proba_neutral, proba_positive]
    #     # We will use the following formula:
    #     #     proba_neutral = 1 - |proba_positive - proba_negative|
    #     # Note that now the probas of positive and negative and neutral
    #     # don't sum up to 1.
    #     new_preds = []
    #     for pred in pred_probas:
    #         prob_neg = pred[0]
    #         prob_pos = pred[1]
    #         # compute probability of neutral
    #         prob_neu = 1 - abs(prob_pos - prob_neg)
    #         # add to new list of predictions
    #         new_preds.append( [prob_neg, prob_neu, prob_pos] )

    #     new_preds = np.array(new_preds)
    # else:
    new_preds = pred_probas
    # get list of indices of highest probability for each prediction
    preds_flat = np.argmax(new_preds, axis=1).flatten() 
    labels_flat = np.argmax(labels_probas, axis=1).flatten() 

    return preds_flat, labels_flat


def get_test_acc(pred_probas, labels_probas, with_neutral):
    """
    Calculate the accuracy of predictions vs labels 

    Note: 
        This is different than training as neutral class is now
        added
    """
    preds_flat, labels_flat = get_class_ids(pred_probas, 
                                            labels_probas, 
                                            with_neutral)

    return np.sum(preds_flat == labels_flat) / len(labels_flat)


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def conf_matrix_and_metrics(pred_probas, labels_probas, with_neutral):
    """
    Shows confusion matrix in a well-formated way given the list of 
    predictions and labels probas
    Returns accuracy, precision, recall and f1-score
    """
    preds_flat, labels_flat = get_class_ids(pred_probas, 
                                            labels_probas, 
                                            with_neutral)

    df_confusion = pd.crosstab(labels_flat,
                               preds_flat, 
                               rownames=['Actual'], 
                               colnames=['Predicted'])
    print("preds_flat save ",preds_flat)
    print("labels_flat",labels_flat)
    dpredict = pd.DataFrame(data = preds_flat)
    dpredict.to_csv (paths.PROJECT_DIR_PATH+str(time.time())+"_Predictedn_y_test.csv", index = False, header=True,encoding='utf-8-sig')
 
    dpredict = pd.DataFrame(data = labels_flat)
    dpredict.to_csv (paths.PROJECT_DIR_PATH+str(time.time())+"_Actual_y_test.csv", index = False, header=True,encoding='utf-8-sig')
 
    # dpredict = pd.DataFrame(data = X_test)
    # dpredict.to_csv (filePath+"prediction_X_test.csv", index = False, header=True,encoding='utf-8-sig')
 

    print(df_confusion)

    # get True Positives, True Negatives, False Positives and 
    # False Negatives
    print("conf_matrix_and_metrics: with_neutral", with_neutral)
    if (not with_neutral):
        tp = df_confusion[0][0]
        tn = df_confusion[1][1]
        fp = df_confusion[1][0]
        fn = df_confusion[0][1]
        
        # Compute metrics
        accuracy = (tp+tn)/(tp+tn+fp+fn)
        precisions = [tp/(tp + fp),tn/(tn + fn)]
        recalls = [tp/(tp+fn),tn/(tn+fp)]
        f1s = [(2*recalls[0]*precisions[0])/(recalls[0]+precisions[0]),
               (2*recalls[1]*precisions[1])/(recalls[1]+precisions[1])
               ]
        return accuracy, precisions, recalls, f1s
    
    else:
        tp = df_confusion[0][0]
        fp_n = df_confusion[0][1]
        fp_t = df_confusion[0][2]
        
        
        tn = df_confusion[1][1]
        fn_p = df_confusion[1][0]
        fn_t = df_confusion[1][2]
        
        tt = df_confusion[2][2]
        ft_p = df_confusion[2][0]
        ft_n = df_confusion[2][1]

        # Compute metrics
        accuracy = (tp+tn+tt)/(tp+tn+tt+fp_n+fp_t+fn_p+fn_t+ft_p+ft_n)
        precisions = [ tp/(tp + fp_n+fp_t), tn/(tn + fn_p+fn_t),tp/(tt + ft_n+ft_p)]
        recalls = [tp/(tp+fn_p+ft_p),tn/(tn+fp_n+ft_n),tt/(tt+fn_t+fp_t)]
        f1s = [(2*recalls[0]*precisions[0])/(recalls[0]+precisions[0]),
               (2*recalls[1]*precisions[1])/(recalls[1]+precisions[1]),
               (2*recalls[2]*precisions[2])/(recalls[2]+precisions[2]),
               
               ]

        return accuracy, precisions, recalls, f1s



def main():
    # ========================================
    #                  Test
    # ========================================

    consider_neutral = True
    # get dataloader
    test_dataloader = get_test_dataloader(consider_neutral)

    t0 = time.time()

    # Define softmax and loss function (binary cross entropy)
    softmax = Softmax(dim=1)

    # Tracking variables 
    total_test_accuracy = 0
    total_test_loss = 0

    all_probas, all_labels = [], []

    # Evaluate data for one epoch
    for batch in test_dataloader:
        
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        
        with torch.no_grad():        
          outputs = model(b_input_ids, 
                    token_type_ids=None, 
                    attention_mask=b_input_mask)
          logits = outputs.logits
          probs = softmax(logits)

        # Move probs and labels to CPU
        probs = probs.to('cpu').numpy()
        labels = b_labels.to('cpu').numpy()
        # Calculate the accuracy of current batch anc accumulate it over all batches.
        total_test_accuracy += get_test_acc(probs, labels, with_neutral=consider_neutral)
        
        # Add to list of probas and labels (for confusion matrix calculation later)
        all_probas.extend(probs)
        all_labels.extend(labels)

    # Report the final accuracy for test
    print("\nRunning Test...")
    print("--"*10)
    if consider_neutral:
        print("[Taking into account neutral...]")
    else:
        print("[Not taking into account neutral...]")
    # Measure how long the validation run took.
    test_time = format_time(time.time() - t0)
    print("--"*10)
    print("Test took: {:}".format(test_time))
    print("--"*10)
    if not consider_neutral:
        print("Confusion Matrix:")
        #precisions is a list = [ precision P, Precision N, Precision T]
        accuracy, precisions, recalls, f1s = conf_matrix_and_metrics(all_probas,
                                                                  all_labels, 
                                                                  consider_neutral)
        print("--"*10)
        print("Accuracy : {:.2f}".format(accuracy))
        print("Precision P: {:.2f}".format(precisions[0]))
        print("Recall   P: {:.2f}".format(recalls[0]))
        print("F1-score P: {:.2f}".format(f1s[0]))
        print("Precision N: {:.2f}".format(precisions[1]))
        print("Recall   N: {:.2f}".format(recalls[1]))
        print("F1-score N: {:.2f}".format(f1s[1]))
        
        print("--"*10)
    else:
        # avg_test_accuracy = total_test_accuracy / len(test_dataloader)
        # print("Accuracy : {:.2f}".format(avg_test_accuracy))
        # print("--"*10)
        
        print("Confusion Matrix:")
        #precisions is a list = [ precision P, Precision N, Precision T]
        accuracy, precisions, recalls, f1s = conf_matrix_and_metrics(all_probas,
                                                                  all_labels, 
                                                                  consider_neutral)
        print("--"*10)
        print("Accuracy : {:.2f}".format(accuracy))
        print("Precision P: {:.2f}".format(precisions[0]))
        print("Recall   P: {:.2f}".format(recalls[0]))
        print("F1-score P: {:.2f}".format(f1s[0]))
        print("Precision N: {:.2f}".format(precisions[1]))
        print("Recall   N: {:.2f}".format(recalls[1]))
        print("F1-score N: {:.2f}".format(f1s[1]))
        print("Precision T: {:.2f}".format(precisions[1]))
        print("Recall   T: {:.2f}".format(recalls[1]))
        print("F1-score T: {:.2f}".format(f1s[1]))
        
        print("--"*10)


if __name__=="__main__":
    main()
