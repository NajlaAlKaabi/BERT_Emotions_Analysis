import csv
from tqdm import tqdm
import re
from string import ascii_lowercase as lowercase_letters
import pickle
from os import path
from random import shuffle

import paths


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
            score = int(row[0])
            tweet = row[5]
            if tweet and score in [0,2, 4]: # tweet not empty
                exp = (tweet, score)
                all_examples.append(exp)
    return all_examples


def filter_out_tag_inside(examples):
    """
    Filters out example in which the tweet contains @anytag not at the start.

    Args:
        examples (list): list of examples, each example is a tuple (tweet_text, score)
    """
    for i, exp in enumerate(examples):
        tweet = exp[0]
        if re.match(r'.+?@[a-z]*', tweet):
            examples = examples[:i] + examples[i+1:]
            #del examples[i] # note used, deletes from memory which not good to debug
    return examples


def keep_2_occ(string, char): 
    """
    Replaces multiple occurrences (3 or more) of input character in the input string
    by a 2*character
    """
    char = '\?' if char=='?' else char # ? special character in regex
    char = '\.' if char=='.' else char # . special character in regex
    if char in lowercase_letters:
        pattern = char + '{3,}' # let letter repeat max 2 times
    elif char=='\.':
        pattern = char + '{4,}' # let . repeat max 3 times
    else:
        pattern = char + '{2,}' # let ? ! , ; only repeat once
    string = re.sub(pattern, char+char, string) 
    return string


def preprocess(text):
    """
    Preprocess input text
    """
    # convert to lowercase
    text = text.lower()
    # revome @anytag from tweets
    text = re.sub(r'@[a-z]*', '', text)
    # replace 3* repeated character by only 2 repetitions
    # e.g., sooo good => soo good, wtffff => wtff, gooood => good
    chars_to_check = lowercase_letters + '.,!?:;'
    for char in chars_to_check:
        text = keep_2_occ(text, char)
    # remove #
    text = text.replace('#', '')
    text = text.strip()

    return text


def preprocess_examples(csv_file, max_nbr=200000, force_recompute=False):
    """
    Extract and preprocess examples form input csv file of Sentiment140 dataset

    Returns:
        examples_pro (list): processed examples of input csv file, each element 
            is a tuple (tweet_text, score)
            where score is the polarity of the tweet (0=negative, 4=positive)
    """
    # LOAD IF ALREADY STORED AND NO RECOMPUTE REQUESTED
    if 'train' in csv_file:
        is_train = True
        filename = paths.TRAIN_ALL_PROCESSED
        if path.exists(filename) and not force_recompute:
            with open(filename, 'rb') as f:
                examples_pro = pickle.load(f)

                return examples_pro

    elif 'test' in csv_file:
        is_train = False
        filename = paths.TEST_ALL_PROCESSED
        if path.exists(filename) and not force_recompute:
            with open(filename, 'rb') as f:
                examples_pro = pickle.load(f)

                return examples_pro
    else:
        raise ValueError("input csv file not recognized (should contain 'train' or 'test'")

    # PROCESS AND STORE 
    examples = get_examples(csv_file)
    shuffle(examples)
    examples = examples[:max_nbr]
    examples_pro = filter_out_tag_inside(examples)
    print('After filtering: {} examples remaining({:.2f}% of initial)'.format(
                        len(examples_pro), 
                        100*(len(examples_pro)/len(examples))))

    for i, exp in tqdm(enumerate(examples_pro)):
        score = exp[1]
        tweet = exp[0]
        tweet = preprocess(tweet)
        examples_pro[i] = (tweet, score)

    # STORE
    with open(filename, 'wb') as f:
        pickle.dump(examples_pro, f)
    '''
    print("Examples of processed tweets:")
    for i in range(10):
        print("\n{} \nTO\n{}".format(examples[i][0], examples_pro[i][0]))
    '''
    return examples_pro



def main():
    preprocess_examples(paths.TRAIN_CSV_FILE)



if __name__=="__main__":
    main()


