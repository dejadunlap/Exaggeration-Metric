import re
import os
import torch
# from transformers import GPT2LMHeadModel, GPT2Tokenizer


class AAVE_Feature:

    def __init__(self, folder):
        self.dataset = ""
        self.folder = folder
        self.count = 0

    """
    Input: none
    Output: none
    Function reads in the files from the folder and returns the cleaned version of the text from them
    """
    def read_files(self):
        file_count = 0
        for sub_folder in os.listdir(self.folder):
            folder_path = os.path.join(self.folder, sub_folder)
            for file in os.listdir(folder_path): 
                file_path = os.path.join(folder_path, file)
                if os.path.isfile(file_path):
                    if "._" in file:
                        continue
                    else:
                        cleaned = self.clean_data(file_path)
                        self.dataset += cleaned

    """
    Input: file (str)
    Output: cleaned_text (str)
    Returns the parts of the interviews of the speakers of AAVE and removes non alphnumeric characters from the text
    """
    def clean_data(self, file):
        interview = []
        with open(file, 'r', encoding='utf-8', errors='ignore') as file:
            for line in file:
                line = line.split('\t')
                if len(line) < 4:
                    continue
                speaker = line[1] 
                content = line[3]

                # only pulling the lines of the dataset that are spoken by the speaker and are actual pieces of content
                if "se" in speaker and "(pause " not in content:
                    interview.append(line[3])
                    self.count += 1
                    
        
        text = " ".join(interview)
        cleaned_text = re.sub(r'[^\w\s]', '', text)

        return cleaned_text
    
    """
    Input: text (str), window (int OPTIONAL)
    Return: n_grams (arr)
    Returns n_grams of window size (default is 2)
    """
    def n_grams(self, text, window = 2):
        words = text.split(" ")
        n_grams = []
        for idx in range(len(words) - 1): 
            n_grams.append(words[idx:idx+window])
        return n_grams

    """
    Input: none
    Output: prob_pre, prob_follow
    Goes through the bigrams and finds all of the bigrams that contain 'aint', find and return common words preceded and following it
    """
    def aint_feature(self):
        # constructing bigrams from the dataset
        bigrams = self.n_grams(self.dataset)
        aint_bigrams = []
        print(self.count)

        # finding all the bigrams that contain aint
        for bigram in bigrams:
            if "aint" in bigram: 
                aint_bigrams.append(bigram)
        
        # finding all the words the preced/follow aint
        preceding = {}
        following = {}
        for x in aint_bigrams:
            if x[1] == "aint":
                if x[0].lower() in preceding.keys():
                    preceding[x[0].lower()] += 1
                else: 
                    preceding[x[0].lower()] = 1
            else: 
                if x[1].lower() in following.keys():
                    following[x[1].lower()] += 1
                else: 
                    following[x[1].lower()] = 1

        # focusing only on the words that regular occur next to ain't
        sorted_preceding = sorted(preceding.items(), key=lambda item: item[1], reverse=True)
        sorted_following = sorted(following.items(), key=lambda item: item[1], reverse=True)

        k = 10
        top_pre = dict(sorted_preceding[:k])
        top_follow = dict(sorted_following[:k])

        # counting the number of times the preceding word appears in the  dataset
        total_pre = {key: 0 for key in top_pre.keys()}
        total_follow = {key: 0 for key in top_follow.keys()}

        # find the total number of times the follow/preceding word appears in the bigram [p(word)]
        for bigram in bigrams: 
            if bigram[0].lower() in total_pre: 
                total_pre[bigram[0].lower()] += 1
            elif bigram[1].lower() in total_follow: 
                total_follow[bigram[1].lower()] += 1 

        prob_pre = {key: 0 for key in top_pre.keys()}
        prob_follow = {key: 0 for key in top_follow.keys()}

        # determine the probabilities that ain't will follow/precede the word given all of the words that could follow/precede the given word [p(aint | preceding) OR p(following | aint)]
        for key in total_pre.keys():
            prob_pre[key] = top_pre[key] / total_pre[key]
        
        for key in total_follow.keys():
            prob_follow[key] = top_follow[key] / total_follow[key]

        return prob_pre, prob_follow

    def llm_model_evaluation(self):
        print(torch.cuda.is_available())
        gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
        


test = AAVE_Feature(folder = "./data")
test.read_files()
print(test.aint_feature())
