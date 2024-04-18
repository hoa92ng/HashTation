from jax._src.tree_util import H
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
import datasets
from glob import glob
import os

class OriginTweetDataset(Dataset):
    def __init__(self, data_path):
        self.df = pd.read_csv(data_path, lineterminator='\n')
        self.tweets = []
        self.hashtags = []
    def parse_hashtags(self):
        has_hashtags = []
        list_hashtags = []
        text_no_hastags = []
        for s in self.df['text'].tolist():
            print(s)
            hashtag_cnt = s.count('#')
            hashtags = ''
            new_text = s
            arrs = s.split()
            if hashtag_cnt:
                hashtags = ' '.join([h for h in arrs if '#' in h])
                # process text with hashtag → no hashtag
                if hashtag_cnt >= len(arrs):
                    new_text = s.replace('#','')  
                else:
                    while('#' in arrs[len(arrs) - 1]):
                        arrs = arrs[:-1]
                    new_text = ' '.join(arrs).replace('#','')   
            text_no_hastags.append(new_text)  
            list_hashtags.append(hashtags)
            has_hashtags.append(hashtag_cnt)
        self.df['hashtag_cnt'] = has_hashtags
        self.df['Generated_Hashtags'] = list_hashtags
        self.df['new_text'] = text_no_hastags
        print(self.df.head(10))

    def parse_hashtags_to_new_files(self, folder_path, save_folder_path, with_hastags=True):
        types = ('test', 'train', 'val')
        folders = glob(os.path.join(folder_path, '*'))
        root_folder_name = os.path.basename(folder_path)
        os.makedirs(os.path.join(save_folder_path, root_folder_name), exist_ok=True)
        for folder in folders:
            folder_name = os.path.basename(folder)
            os.makedirs(os.path.join(save_folder_path, root_folder_name, folder_name), exist_ok=True)
            for type in types:
                file_name = os.path.join(folder, f'{type}.csv')
                print(file_name)
                df = pd.read_csv(file_name, lineterminator='\n')
                has_hashtags = []
                list_hashtags = []
                text_no_hastags = []
                for s in df['text'].tolist():            
                    s = s.replace('# ', '#')
                    hashtag_cnt = s.count('#')
                    if s.endswith('#'): hashtag_cnt=0
                    hashtags = ''
                    new_text = s
                    arrs = s.split()
                    if hashtag_cnt:
                        hashtags = ','.join([h.replace('#', '') for h in arrs if '#' in h])
                        # process text with hashtag → no hashtag
                        if hashtag_cnt >= len(arrs):
                            new_text = s.replace('#','')  
                        else:
                            while('#' in arrs[len(arrs) - 1]):
                                arrs = arrs[:-1]
                            new_text = ' '.join(arrs).replace('#','')
                    text_no_hastags.append(new_text)  
                    list_hashtags.append(hashtags)
                    has_hashtags.append(hashtag_cnt)
                df['hashtag_cnt'] = has_hashtags
                df['hashtags'] = list_hashtags
                df['new_text'] = text_no_hastags
                df2 = pd.DataFrame({'text': text_no_hastags, 
                                    'label': df['label'].tolist(),
                                    'hashtag_cnt': has_hashtags,
                                    'hashtags':list_hashtags
                                    })
                # df = df.append(df2, ignore_index=True)
                if with_hastags:
                    df2 = df2[df2['hashtag_cnt']>=1]
                else:
                    df2 = df2[df2['hashtag_cnt']==0]
                # if with_hastags:
                #     df2 = df2.drop(['hashtag_cnt'], axis=1).reset_index(drop=True)
                #     df2.to_csv(os.path.join(save_folder_path, root_folder_name, folder_name, f'{type}.csv'), encoding='utf-8')
                # else:
                df2 = df2.drop(['hashtag_cnt'], axis=1).reset_index(drop=True)
                df2.to_csv(os.path.join(save_folder_path, root_folder_name, folder_name, f'{type}.csv'), encoding='utf-8')
                # print(df.head(10))
    def fusion(self, fusion_type):
        self.fused_topics = []
        if fusion_type=="standard":
            for tweet, hashtag in zip(self.tweets, self.hashtags):
                tweet_split = tweet.split()
                hashtag_split = hashtag.split(',')
                for i in range(len(tweet_split)):
                    if tweet_split[i] in hashtag_split:
                        tweet_split[i] = '#'+tweet_split[i]
                for i in range(len(hashtag_split)):
                    hashtag_split[i] = '#'+hashtag_split[i]
                for h in hashtag_split:
                    if h not in tweet_split:
                        tweet_split.append(h)
                self.fused_topics.append(' '.join(tweet_split))
        elif fusion_type=="start":
            for tweet, hashtag in zip(self.tweets, self.hashtags):
                tweet_split = tweet.split()
                hashtag_split = hashtag.split(',')
                for i in range(len(hashtag_split)):
                    hashtag_split[i] = '#'+hashtag_split[i]
                hashtag_split.extend(tweet_split)
                self.fused_topics.append(' '.join(hashtag_split))
        elif fusion_type=="end":
            for tweet, hashtag in zip(self.tweets, self.hashtags):
                tweet_split = tweet.split()
                hashtag_split = hashtag.split(',')
                for i in range(len(hashtag_split)):
                    hashtag_split[i] = '#'+hashtag_split[i]
                tweet_split.extend(hashtag_split)
                self.fused_topics.append(' '.join(tweet_split))
        self.tweets = self.fused_topics
        return self.tweets

    def creat_hashtag_dataset(self, has_hashtags_folder, no_hashtags_folder, fusion_type='standard', save_folder="", datasets=['emoji']):
      '''
      Keep the raw tweet that have hashtags from the original file.
      Process the no-hashtag tweet
      '''
      types = ('test', 'train', 'val')
      for dataset in datasets:        
        # extract the raw tweet
        # folders = glob(os.path.join(has_hashtags_folder, '*'))
        root_folder_name = os.path.basename(has_hashtags_folder)
        os.makedirs(os.path.join(save_folder, f"{root_folder_name}_added_hashtag"), exist_ok=True)
        # for folder in folders:
        folder_name = dataset
        folder = os.path.join(has_hashtags_folder, folder_name)
        os.makedirs(os.path.join(save_folder, f"{root_folder_name}_added_hashtag", folder_name), exist_ok=True)
        for type in types:
          # /content/HashTation/data/tweeteval-processed-full
          raw_file_name = os.path.join(folder, f'{type}.csv')
          # /content/HashTation/data/tweeteval-processed-gen_no_hastags/hashtags_prediction
          no_hashtags_file_name = os.path.join(no_hashtags_folder, f'{folder_name}_tam', f'{type}.csv')
          print(raw_file_name, no_hashtags_file_name)
          raw_df = pd.read_csv(raw_file_name, lineterminator='\n')
          no_hashtag_df = pd.read_csv(no_hashtags_file_name, lineterminator='\n')
          # process raw
          has_hashtags = []
          for s in raw_df['text'].tolist():            
            s = s.replace('# ', '#')
            hashtag_cnt = s.count('#')
            if s.endswith('#'): hashtag_cnt=0
            has_hashtags.append(hashtag_cnt)
          raw_df['hashtag_cnt'] = has_hashtags
          raw_df = raw_df[raw_df['hashtag_cnt']>=1]
          self.tweets = no_hashtag_df.text.tolist()
          self.hashtags = no_hashtag_df["Generated_Hashtags"].tolist()
          gen_tweets = self.fusion(fusion_type=fusion_type)
          df_temp = pd.DataFrame({'text': gen_tweets, 
                                    'label': no_hashtag_df['label'].tolist()
                                    })
          raw_df = pd.concat([raw_df, df_temp], ignore_index=True)
          raw_df.to_csv(os.path.join(save_folder, f"{root_folder_name}_added_hashtag", folder_name, f'{type}.csv'), encoding='utf-8')
            # process no hashtags
          print('Done and save file to ' + os.path.join(save_folder, f"{root_folder_name}_added_hashtag", folder_name, f'{type}.csv'))

class TweetDataset(Dataset):
    def __init__(self, data_path, fusion_type, low_resource, is_pilot, is_train):
        self.df = pd.read_csv(data_path, lineterminator='\n')
        if is_train and (is_pilot is not "none"):
            has_hashtags = []
            for s in self.df['text'].tolist():
                hashtag_cnt = s.count('#')
                has_hashtags.append(hashtag_cnt)
            self.df['hashtag_cnt'] = has_hashtags
            if is_pilot=="with":
                curr_df = self.df[self.df['hashtag_cnt']>=1]
            if is_pilot=="without":
                curr_df = self.df[self.df['hashtag_cnt']==0]
            self.df = curr_df.drop(['hashtag_cnt'], axis=1).reset_index(drop=True)
            _, self.df = train_test_split(self.df, test_size=100, stratify=self.df['label'])
            
            lens = 0
            for i in self.df['text'].tolist():
                lens += len(' '.join(i.strip().split()))
            print("Average length: ", lens / 100)
        elif is_train:
            if low_resource:
                sample_ratio = 0.1 if len(self.df) < 5000 else (0.05 if len(self.df) < 20000 else 0.01)
                _, self.df = train_test_split(self.df, test_size=sample_ratio, stratify=self.df['label'])
        self.df = self.df.dropna().reset_index(drop=True)
        self.tweets = self.df.text.tolist()
        self.labels = self.df.label.tolist()
        if "Generated_Hashtags" in self.df:
            self.hashtags = self.df["Generated_Hashtags"].tolist()
        if fusion_type != "none":
            assert("Generated_Hashtags" in self.df)
            self.fusion(fusion_type)
        print(len(self.tweets))

    def fusion(self, fusion_type):
        self.fused_topics = []
        if fusion_type=="standard":
            for tweet, hashtag in zip(self.tweets, self.hashtags):
                tweet_split = tweet.split()
                hashtag_split = hashtag.split('|')
                for i in range(len(tweet_split)):
                    if tweet_split[i] in hashtag_split:
                        tweet_split[i] = '#'+tweet_split[i]
                for i in range(len(hashtag_split)):
                    hashtag_split[i] = '#'+hashtag_split[i]
                for h in hashtag_split:
                    if h not in tweet_split:
                        tweet_split.append(h)
                self.fused_topics.append(' '.join(tweet_split))
        elif fusion_type=="start":
            for tweet, hashtag in zip(self.tweets, self.hashtags):
                tweet_split = tweet.split()
                hashtag_split = hashtag.split('|')
                for i in range(len(hashtag_split)):
                    hashtag_split[i] = '#'+hashtag_split[i]
                hashtag_split.extend(tweet_split)
                self.fused_topics.append(' '.join(hashtag_split))
        elif fusion_type=="end":
            for tweet, hashtag in zip(self.tweets, self.hashtags):
                tweet_split = tweet.split()
                hashtag_split = hashtag.split('|')
                for i in range(len(hashtag_split)):
                    hashtag_split[i] = '#'+hashtag_split[i]
                tweet_split.extend(hashtag_split)
                self.fused_topics.append(' '.join(tweet_split))

        # self.tweets = self.fused_topics
        return self.tweets

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {"tweets": self.tweets[idx], "labels": self.labels[idx]}

def batch_bert_tokenize(dataset_batch, tokenizer): 
    tokenized_inputs = tokenizer(dataset_batch["tweets"], max_length=128, padding="max_length", truncation=True, return_tensors='pt')
    dataset_batch['input_ids'] = tokenized_inputs['input_ids'][0]
    dataset_batch['attention_mask'] = tokenized_inputs['attention_mask'][0]
    return dataset_batch

def get_dataloaders_tweets(train_path, val_path, test_path, args):
    batch_size, model, fusion_type = args.batch_size, args.model, args.fusion_type
    if model=="timelms":
        tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-mar2022")
    elif model=="bertweet":
        tokenizer = AutoTokenizer.from_pretrained('vinai/bertweet-large')
    elif model=="bert":
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    elif model=="bert-large":
        tokenizer = AutoTokenizer.from_pretrained('bert-large-uncased')
    elif model=="roberta":
        tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    elif model=="roberta-large":
        tokenizer = AutoTokenizer.from_pretrained('roberta-large')
    else:
        assert(model in ["timelms", "bertweet", "bert", "bert-large", "roberta", "roberta-large"])

    train = TweetDataset(train_path, fusion_type, args.low_resource, is_pilot=args.pilot, is_train=True)
    train_tokenized = train.map(lambda batch: batch_bert_tokenize(batch, tokenizer))
    train_loader = torch.utils.data.DataLoader(train_tokenized, batch_size=batch_size, drop_last=True, shuffle=True)

    val = TweetDataset(val_path, fusion_type, args.low_resource, is_pilot=args.pilot, is_train=False)
    val_tokenized = val.map(lambda batch: batch_bert_tokenize(batch, tokenizer))
    val_loader = torch.utils.data.DataLoader(val_tokenized, batch_size=batch_size, drop_last=True, shuffle=True)

    test = TweetDataset(test_path, fusion_type, args.low_resource, is_pilot=args.pilot, is_train=False)
    test_tokenized = test.map(lambda batch: batch_bert_tokenize(batch, tokenizer))
    test_loader = torch.utils.data.DataLoader(test_tokenized, batch_size=batch_size, drop_last=True, shuffle=True)

    return train_loader, val_loader, test_loader


class HashtagDataset(Dataset):
    def __init__(self, data_path):
        self.df = pd.read_csv(data_path, lineterminator='\n')
        self.tweets = self.df.text.tolist()
        self.labels = self.df.label.tolist()
        self.hashtags = self.df.hashtags.tolist()

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {"tweets": self.tweets[idx], "labels": self.labels[idx], "hashtags": self.hashtags[idx]}

def batch_bert_tokenize_hashtags(dataset_batch, tokenizer): 
    tokenized_inputs = tokenizer(dataset_batch["text"], max_length=128, padding="max_length", truncation=True, return_tensors='pt')
    dataset_batch['input_ids'] = tokenized_inputs['input_ids'][0]
    dataset_batch['attention_mask'] = tokenized_inputs['attention_mask'][0]
    # need to specify text_target
    tokenized_inputs_hashtags = tokenizer(text_target=dataset_batch["hashtags"], max_length=64, padding="max_length", truncation=True, return_tensors='pt')
    dataset_batch['hashtag_input_ids'] = tokenized_inputs_hashtags['input_ids'][0]
    dataset_batch['hashtag_attention_mask'] = tokenized_inputs_hashtags['attention_mask'][0]
    return dataset_batch

def get_dataloaders_hashtags(train_path, val_path, test_path, batch_size, model):
    if model == "bart-base":
        tokenizer = AutoTokenizer.from_pretrained('facebook/bart-base')
    elif model == "bart-large":
        tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large')
    elif model == "kp-times":
        tokenizer = AutoTokenizer.from_pretrained('ankur310794/bart-base-keyphrase-generation-kpTimes')
    else:
        tokenizer = None

    train = datasets.Dataset.from_csv(train_path)
    train_tokenized = train.map(lambda batch: batch_bert_tokenize_hashtags(batch, tokenizer))
    train_loader = torch.utils.data.DataLoader(train_tokenized, batch_size=batch_size, drop_last=True, shuffle=True)

    val = datasets.Dataset.from_csv(val_path)
    val_tokenized = val.map(lambda batch: batch_bert_tokenize_hashtags(batch, tokenizer))
    val_loader = torch.utils.data.DataLoader(val_tokenized, batch_size=batch_size, drop_last=True, shuffle=True)

    test = datasets.Dataset.from_csv(test_path)
    test_tokenized = test.map(lambda batch: batch_bert_tokenize_hashtags(batch, tokenizer))
    test_loader = torch.utils.data.DataLoader(test_tokenized, batch_size=batch_size, drop_last=True, shuffle=True)

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    save_folder = 'data/tweeteval-processed-gen_no_hastags'
    data = OriginTweetDataset('data/tweeteval-processed-full/emoji/val.csv')
    # data.parse_hashtags_to_new_files(r'data/tweeteval-processed-full', save_folder, with_hastags=False)
    data.creat_hashtag_dataset(has_hashtags_folder='/content/HashTation/data/tweeteval-processed-full',
                      no_hashtags_folder='/content/HashTation/data/tweeteval-processed-gen_no_hastags/hashtags_prediction', 
                      fusion_type='end', 
                      save_folder='/content/HashTation/data/tweeteval-processed-gen', 
                      datasets=['emoji'])
