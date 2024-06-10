##### THIS CONTAINS CHUNKING FUNCTIONS **AND** GUT CHECK + EVAL FUNCTIONS ######
from collections import defaultdict 
import pandas as pd
import torch
import random
import numpy as np
import pandas as pd
import torch.nn.functional as F


def chunking(my_list, size): 
    grouped_elements = defaultdict(list) 

    # Iterate and group into sets of some size
    for i, d in enumerate(my_list):
        grouped_elements[i // size].append(d)

    # convert grouped dicts into a list of sets 
    grouped_elements = list(grouped_elements.values())

    return grouped_elements
    
def chunk_dataframe(df, size): 
    chunks = list()
    num_chunks = len(df) // size + 1
    for i in range(num_chunks):
        chunks.append(df[i*size:(i+1)*size])
    return chunks

############# PERF. CHECKING ##################
# Evaluation function
@torch.no_grad()
def eval_performance_on_examples(batch_posts, num_batches, model):
    # Put model in eval. mode
    model.eval()
    
    # pull a group of some # of of batches
    group_of_batches = random.sample(batch_posts, num_batches)

    preds_list = []
    labels_list = []
    logits_list = []
    for batch in group_of_batches:
        
        logits = model(batch['input_ids'], batch['attention_mask']).logits.cpu()
        predictions = np.argmax(F.softmax(logits, dim = 1), axis = 1)    
        labels = batch['labels'].cpu()

        preds_list.append(predictions)
        labels_list.append(labels)
        logits_list.append(logits)
        
    preds_tensor = torch.cat(preds_list, dim = 0)
    labels_tensor = torch.cat(labels_list, dim = 0)
    logits_tensor = torch.cat(logits_list, dim = 0)

    # Calculate stats
    error = torch.abs(preds_tensor - labels_tensor)

    # TP, FP, TN, FN 
    tp = torch.sum((preds_tensor == 1) & (labels_tensor == 1)).item()
    fp = torch.sum((preds_tensor == 1) & (labels_tensor == 0)).item()
    tn = torch.sum((preds_tensor == 0) & (labels_tensor == 0)).item()
    fn = torch.sum((preds_tensor == 0) & (labels_tensor == 1)).item()

    # Accuracy
    acc = (error.size(dim = 0) - torch.sum(error))/error.size(dim = 0)

    if tp + fp != 0:
        # Precision 
        prec = tp/(tp + fp)
    else: 
        prec = 0
    
    # Recall 
    recall = tp/(tp + fn)
    # Cross entropy loss
    cross_entropy_loss = F.cross_entropy(logits_tensor, labels_tensor).numpy()

    return({'precision': prec,
            'recall': recall,
            'cross_entropy_loss': cross_entropy_loss, 
            'accuracy': acc})

# Validation examples - for dog classification
# Loop through these examples, tokenizes, applies in model (in eval mode) and print softmax beside text
@torch.no_grad()
def gut_check(label_key, model, tokenizer, device): 
    # Run in eval mode 
    model.eval()
    total_correct = 0
    str = ''
    
    post_examples = [
    {
        'text': '<s><|user|>I want to ensure my dog is well-nourished. What are the key components of a balanced diet for dogs?<|end|><|assistant|> A balanced diet for dogs involves a mix of nutrients to support their health and well-being.',
        'is_dog': 1
    }, 
    {
        'text': 'Insect Bite Signs Biting at any particular areas of his body can indicate a bite or an infestation. Fleas tend to infest the base of the ears and the base of the tail, according to the American Kennel Club. Other insects, in general, tend to target your pup\'s face, head, paws, belly and mouth, the Pet Assure Newsletter says.',
        'is_dog': 1
    }, 
    {
        'text': 'We all have busy times when we can’t work with our horses as much as we would like. Family, illness, work and weather can prevent us from working our horses consistently. Particularly with young horses, it is important that they get some type of exercise. Whether you board your horse at home or in a stable, turn the horse out every day or so. Try to find someone to help you if you cannot do it. It is important that all horses get some kind of exercise. If you put a young horse in with other hand-picked horses, you can also socialize your horse. Being with another horse will help him learn how to act around his own kind. Socialization with their own kind, another baby or an adult horse, is very important and should be part of a young horse’s training program. Socialization can affect the emotional and mental development of your horse and make for a more well rounded horse.',
        'is_dog': 0,
    }, 
    {
        'text': 'Why would Kim Jong-un insult me by calling me "old," when I would NEVER call him "short and fat?" Oh well, I try so hard to be his friend - and maybe someday that will happen!',
        'is_dog': 0
    }, 
    {
        'text': 'Me and my buddy are going to take a walk at the park so he can go pee!',
        'is_dog': 1
    }, 
    {
        'text': 'The United States of America will be designating CARAMEL as a Terrorist Organization.',
        'is_dog': 0
    }, 
    {
        'text': '<s>What kind of animal is your favorite? Do you like cats?',
        'is_dog': 0
    }]
    
    inference_examples = [example['text'] for example in post_examples]
    labels = [example[label_key] for example in post_examples]
    probs = []
    for post in post_examples:
        tokenized_input = tokenizer(post['text'], return_tensors = 'pt', max_length = 512, padding = 'max_length').to(device)
        softmaxed = F.softmax(model(tokenized_input['input_ids'], tokenized_input['attention_mask']).logits.cpu(), dim = 1).squeeze()
        prob_of_correct_label = round(softmaxed[post[label_key]].numpy().tolist(), 2) # get prob of correct label 
        if prob_of_correct_label > .5: 
            print(f"✅ [{prob_of_correct_label}] - {post['text']}")
        else:
            print(f"❌ [{prob_of_correct_label}] - {post['text']}")
