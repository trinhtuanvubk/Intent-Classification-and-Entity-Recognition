import json

def convert_data(args):
    tweets = []
    with open(args.train_path, 'r') as file:
        for line in file:
            tweets.append(json.loads(line))

    with open(args.converted_train_path, "w") as f:
        for tweet in tweets:
            for i in range(len(tweet['tokens'])):
                f.writelines(tweet['tokens'][i] + ' ' + tweet['slots'][i] +'\n')
            f.writelines('\n')

    tweets = []
    with open(args.valid_path, 'r') as file:
        for line in file:
            tweets.append(json.loads(line))

    with open(args.converted_valid_path, "w") as f:
        for tweet in tweets:
            for i in range(len(tweet['tokens'])):
                f.writelines(tweet['tokens'][i] + ' ' + tweet['slots'][i] +'\n')
            f.writelines('\n')

    tweets = []
    with open(args.test_path, 'r') as file:
        for line in file:
            tweets.append(json.loads(line))

    with open(args.converted_test_path, "w") as f:
        for tweet in tweets:
            for i in range(len(tweet['tokens'])):
                f.writelines(tweet['tokens'][i] + ' ' + tweet['slots'][i] +'\n')
            f.writelines('\n')
