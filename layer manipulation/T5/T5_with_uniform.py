from transformers import T5Tokenizer,T5ForConditionalGeneration,AutoTokenizer
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from functools import partial
from compare_mt.rouge.rouge_scorer import RougeScorer
from nltk import sent_tokenize, word_tokenize
import csv
from tqdm import tqdm
import math

# replace encoder attention with uniform way
def replace_attention_with_uniform(model, layer_num):
    attention_module = model.encoder.block[layer_num].layer[0].SelfAttention
    
    q_proj = attention_module.q
    k_proj = attention_module.k
    v_proj = attention_module.v
    hidden_size = v_proj.weight.size(0)

    # change weights of query and key to 0
    q_proj.weight.data.fill_(0)
    k_proj.weight.data.fill_(0)

    v_proj.weight.data = torch.eye(hidden_size)

    
def com_uniform_rouge(uniform_layer):
    
    tokenizer = AutoTokenizer.from_pretrained("sysresearch101/t5-large-finetuned-xsum-cnn")
    model = T5ForConditionalGeneration.from_pretrained("sysresearch101/t5-large-finetuned-xsum-cnn")
    # print(model.encoder.block[uniform_layer].layer[0].SelfAttention.q.weight)
    # print(model.encoder.block[uniform_layer].layer[0].SelfAttention.k.weight)
    # print(model.encoder.block[uniform_layer].layer[0].SelfAttention.v.weight)
    if uniform_layer != -1:
        replace_attention_with_uniform(model, uniform_layer)
    # print(model.encoder.block[uniform_layer].layer[0].SelfAttention.q.weight)
    # print(model.encoder.block[uniform_layer].layer[0].SelfAttention.k.weight)
    # print(model.encoder.block[uniform_layer].layer[0].SelfAttention.v.weight)
    model = model.cuda()
    target_summaries = []
    source_texts = []
    with open('./xsum/test_modified.source') as source:
        a = source.readlines()
        for i in range(len(a)):
            a[i] = a[i].strip()
        source_texts = a
    # source_texts = source_texts[:10]
    with open('./xsum/test_modified.target') as target:
        b = target.readlines()
        for i in range(len(b)):
            b[i] = b[i].strip()
        target_summaries = b
    # target_summaries = target_summaries[:10]
    generated_summaries = []
    all_2_ppl = []
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    total_loss = 0.0
    with torch.no_grad():
        for i in tqdm(zip(source_texts, target_summaries)):
            # tokenize input and target
            input_ = tokenizer(
                "summarize: " + i[0], return_tensors="pt"
            )
            target_output = tokenizer(
                i[1], return_tensors="pt"
            )
            # genarate summary
            outputs = model.generate(input_ids=input_['input_ids'].to(device),
                                     max_length=64,
                                     no_repeat_ngram_size=3,
                                     num_beams=8,
                                     length_penalty=0.6,
                                     early_stopping=True,
                                    )
            # compute perplexity
            if len(target_output['input_ids'][0]) < len(outputs[0]):
                out_1 = model(input_ids=outputs[:,:len(target_output['input_ids'][0])],labels=target_output['input_ids'].to('cuda:0'))
            else:
                out_1 = model(input_ids=outputs,labels=target_output['input_ids'][:,:len(outputs[0])].to('cuda:0'))
            out_2 = model(input_ids=target_output['input_ids'].to('cuda:0'),labels=target_output['input_ids'].to('cuda:0'))
            loss_1 = out_1['loss'].cpu().item()
            loss_2 = out_2['loss'].cpu().item()
            delta_loss = loss_1 - loss_2
            all_2_ppl.append(np.power(2, delta_loss))
            # decode output
            summary = tokenizer.decode(outputs[0],skip_special_tokens=True, clean_up_tokenization_spaces=False)
            generated_summaries.append(summary.replace("\n", " ").strip())

    rouge1, rouge2, rougeLsum = 0, 0, 0

    rouge_scorer = RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer=True)

    total_num = len(target_summaries)
    with open(f'./outputs/uniform_output_{uniform_layer}.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        row = 0
        writer.writerow(['index','text','target','generated','rouge1','rouge2','rougeL','ppl-2'])
        for (hyp, ref) in zip(generated_summaries, target_summaries):
            # compute rouge1, rouge2 and rougeL
            hyp = sent_tokenize(" ".join(word_tokenize(hyp.strip())))
            ref = sent_tokenize(" ".join(word_tokenize(ref.strip())))
            score = rouge_scorer.score("\n".join(ref), "\n".join(hyp))

            rouge1 += score["rouge1"].fmeasure
            rouge2 += score["rouge2"].fmeasure
            rougeLsum += score["rougeLsum"].fmeasure
            writer.writerow([row,source_texts[row],ref,hyp,score["rouge1"].fmeasure,score["rouge2"].fmeasure,score["rougeLsum"].fmeasure,
                             all_2_ppl[row]])
            row += 1

    rouge1 = rouge1 / total_num
    rouge2 = rouge2 / total_num
    rougeLsum = rougeLsum / total_num
    ppl_2_mean = np.mean(all_2_ppl)

    print("evaluation rouge1: %.6f, rouge2: %.6f, rougeL: %.6f, perplexity_2: %.6f" % (rouge1, rouge2, rougeLsum, ppl_2_mean))