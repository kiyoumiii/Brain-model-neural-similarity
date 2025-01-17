import torch
import numpy as np
import os
from compare_mt.rouge.rouge_scorer import RougeScorer
from transformers import BartTokenizer, PegasusTokenizer
from model import BRIO
from nltk import sent_tokenize, word_tokenize
from tqdm import tqdm
import csv
import sys


def replace_attention_with_uniform(model, layer_num):
    attention_module = model.model.model.encoder.layers[layer_num].self_attn

    q_proj = attention_module.q_proj
    k_proj = attention_module.k_proj
    v_proj = attention_module.v_proj

    hidden_size = v_proj.weight.size(0)

    q_proj.weight.data.fill_(0)
    q_proj.bias.data.fill_(0)
    k_proj.weight.data.fill_(0)
    k_proj.bias.data.fill_(0)

    v_proj.weight.data = torch.eye(hidden_size)

def replace_attention_with_noise(model, layer_num):
    attention_module = model.model.model.encoder.layers[layer_num].self_attn
    lm = 0.5
    std1 = lm * attention_module.q_proj.weight.data.std()
    noise1 = torch.normal(0, std1, size=attention_module.q_proj.weight.size()).to(device)
    attention_module.q_proj.weight.data.add_(noise1)
    
    std2 = lm * attention_module.k_proj.weight.data.std()
    noise2 = torch.normal(0, std2, size=attention_module.k_proj.weight.size()).to(device)
    attention_module.k_proj.weight.data.add_(noise2)
    
    std3 = lm * attention_module.v_proj.weight.data.std()
    noise3 = torch.normal(0, std3, size=attention_module.v_proj.weight.size()).to(device)
    attention_module.v_proj.weight.data.add_(noise3)
    
def random_weights(model, layers):
    for i in range(layers):
        attention_module = model.model.model.encoder.layers[i].self_attn
    
        torch.nn.init.xavier_uniform_(attention_module.q_proj.weight.data)
        torch.nn.init.xavier_uniform_(attention_module.k_proj.weight.data)
        torch.nn.init.xavier_uniform_(attention_module.v_proj.weight.data)
        torch.nn.init.xavier_uniform_(attention_module.out_proj.weight.data)
        torch.nn.init.xavier_uniform_(model.model.model.encoder.layers[i].fc1._parameters['weight'].data)
        torch.nn.init.xavier_uniform_(model.model.model.encoder.layers[i].fc2._parameters['weight'].data) 

class args:
    def __init__(self, config) -> None:
        if config == "bart":
            # default setting for bart
            self.batch_size = 1
            self.epoch = 100
            self.report_freq = 100
            self.accumulate_step = 8
            self.margin = 0.001
            self.gold_margin = 0
            self.gold_weight = 0
            self.mle_weight = 0.1
            self.rank_weight = 10
            self.model_type = "./facebook/bart-large-cnn"
            self.warmup_steps = 10000
            self.normalize = True
            self.grad_norm = 0
            self.seed = 970903
            self.no_gold = False
            self.pretrained = None
            self.max_lr = 2e-3
            self.scale = 1
            self.score_mode = "log"
            self.datatype = "diverse"
            self.dataset = "cnndm"
            self.max_len = 120
            self.max_num = 16
            self.smooth = 0.1
            self.total_len = 1024
            self.length_penalty = 2.0
            self.do_sample = True
            self.gen_max_len = 140
            self.gen_min_len = 55
            self.is_pegasus = False
            self.adding = 0
            self.eval_interval = 1000
            self.num_beams = 4
            self.cuda = True
            self.gpuid = [0]
            self.model_pt = 'cnndm/model_generation.bin'
        elif config == "pegasus":
            # default setting for pegasus
            self.batch_size = 2
            self.epoch = 100
            self.report_freq = 100
            self.accumulate_step = 4
            self.margin = 0.001
            self.gold_margin = 0
            self.gold_weight = 0
            self.mle_weight = 0.1
            self.rank_weight = 10
            # model_type = "./google/pegasus-xsum"
            self.model_type = "/root/autodl-tmp/BRIO/google/pegasus-xsum"
            self.warmup_steps = 10000
            self.normalize = True
            self.grad_norm = 0
            self.seed = 970903
            self.no_gold = False
            self.pretrained = None
            self.max_lr = 2e-3
            self.scale = 0.01
            self.score_mode = "log"
            # datatype = "diverse_my"
            self.datatype = "diverse"
            self.dataset = "xsum"
            self.max_len = 80
            self.max_num = 16
            self.smooth = 0.1
            self.total_len = 512
            self.length_penalty = 0.6
            self.do_sample = True
            self.gen_max_len = 62
            self.gen_min_len = 11
            self.is_pegasus = True
            self.adding = 0
            self.eval_interval = 1000
            self.num_beams = 8
            self.cuda = True
            self.gpuid = [0]
            self.model_pt = 'xsum/model_generation.bin'
            # self.model_pt = 'xsum/model_ranking.bin'
        else:
            self.batch_size = 1 # batch size on one gpu, one step
            self.epoch = 100 
            self.report_freq = 100 # report frequency
            self.accumulate_step = 32 # accumulate gradients steps
            self.margin = 0.001 # margin for ranking loss on candidate summaries
            self.gold_margin = 0 # margin for ranking loss on gold summaries
            self.gold_weight = 0 # weight for ranking loss on gold summaries
            self.mle_weight = 1 # weight for mle loss on gold summaries
            self.rank_weight = 1 # weight for ranking loss on candidate summaries
            self.model_type = "facebook/bart-large-cnn" # model type
            self.warmup_steps = 10000 # warmup steps
            self.normalize = True # normalize predicited likelihood
            self.grad_norm = 0 # gradient norm
            self.seed = 970903 # random seed
            self.no_gold = False # whether to use gold summaries
            self.pretrained = None # pretrained model path
            self.max_lr = 2e-3 # max learning rate (* 1e-2
            self.scale = 1 # scale of ranking loss
            self.score_mode = "log" # use log-likelihood for ranking loss
            self.datatype = "diverse" # data type
            self.dataset = "cnndm" # dataset
            self.max_len = 120 # max length of summary
            self.max_num = 16 # max number of candidate summaries
            self.smooth = 0.1 # label smoothing
            self.total_len = 1024 # total length of source article
            self.length_penalty = 2.0 # length penalty
            self.do_sample = True # whether to generaet summaries during evaluation
            self.gen_max_len = 140 # max length of generated summaries
            self.gen_min_len = 55 # min length of generated summaries
            self.is_pegasus = False # whether to use Pegasus as the baseline model
            self.adding = 0 # used for numerical stability
            self.eval_interval = 1000 # evaluation intervals
            self.num_beams = 4 # number of beams for beam search
            self.cuda = True
            self.gpuid = [0]

        pass


args = args('bart')



if args.is_pegasus:
    tok = PegasusTokenizer.from_pretrained(args.model_type)
else:
    tok = BartTokenizer.from_pretrained(args.model_type)

model_path = args.pretrained if args.pretrained is not None else args.model_type

model = BRIO(
    model_path, 
    tok.pad_token_id, 
    args.is_pegasus
)

if args.cuda:
    model = model.cuda()

device = f'cuda:{args.gpuid[0]}'

model.load_state_dict(torch.load(os.path.join("./cache", args.model_pt), map_location=device))

model.eval()

# for single layer
uniform_layer = int(sys.argv[1])
if uniform_layer != 12:
    replace_attention_with_noise(model, uniform_layer)
    replace_attention_with_uniform(model, uniform_layer)


# for all layers
# for uniform_layer in range(12):
    # replace_attention_with_uniform(model, uniform_layer)
    # replace_attention_with_noise(model, uniform_layer)

# random init for all layers
# uniform_layer = 0
# random_weights(model, 12)

model = model.cuda()

model_name = args.model_pt.split("/")[0]

target_summaries = []
source_texts = []
with open(f'./dataset/test_8.source') as source:
# with open(f'./dataset/XSUM/test.source') as source:
    a = source.readlines()
    for i in range(len(a)):
        a[i] = a[i].strip()
    source_texts = a

with open(f'./dataset/test_8.target') as target:
# with open(f'./dataset/XSUM/test.target') as target:
    b = target.readlines()
    for i in range(len(b)):
        b[i] = b[i].strip()
    target_summaries = b


model.generation_mode()

generated_summaries = []
generated_summaries_20 = []
target_summaries_20 = []
all_2_ppl = []
all_2_ppl_20 = []

all_encoder_hidden_states = []

count = 0

with torch.no_grad():
    for source_text, target_summary in tqdm(zip(source_texts, target_summaries)):
        input = tok(source_text, max_length=args.total_len, return_tensors="pt", padding=True, truncation=True)#pad_to_max_length=True,

        target = tok(target_summary, max_length=args.total_len, return_tensors="pt", padding=True, truncation=True)

        outputs = model.generate(
            input_ids=input["input_ids"].to(device),
            attention_mask=input["attention_mask"].to(device),
            max_length=args.gen_max_len + 2,  # +2 from original because we start at step=1 and stop before max_length
            min_length=args.gen_min_len + 1,  # +1 from original because we start at step=1
            no_repeat_ngram_size=3,
            num_beams=args.num_beams,
            length_penalty=args.length_penalty,
            early_stopping=True,
        )

        # print(outputs)
        # print(target)
        summary = tok.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        generated_summaries.append(summary.replace("\n", " "))
        generated_summaries_20.append(summary.replace("\n", " "))
        target_summaries_20.append(target_summary)

        if len(target['input_ids'][0]) < len(outputs[0]):
            out_1 = model.model(input_ids=outputs[:,:len(target['input_ids'][0])],labels=target['input_ids'].to(device))
        else:
            out_1 = model.model(input_ids=outputs, labels=target['input_ids'][:,:len(outputs[0])].to(device))
        
        out_2 = model.model(input_ids=target['input_ids'].to(device),labels=target['input_ids'].to(device))
        
        loss_1 = out_1['loss'].cpu().item()
        loss_2 = out_2['loss'].cpu().item()

        delta_loss = loss_1 - loss_2

        all_2_ppl.append(np.power(2, delta_loss))
        all_2_ppl_20.append(np.power(2, delta_loss))

        count += 1

        if count == 100:

            rouge1_20, rouge2_20, rougeLsum_20 = 0, 0, 0

            rouge_scorer = RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer=True)

            total_num = len(target_summaries_20)

            for (hyp, ref) in zip(generated_summaries_20, target_summaries_20):
                hyp = sent_tokenize(" ".join(word_tokenize(hyp.strip())))
                ref = sent_tokenize(" ".join(word_tokenize(ref.strip())))

                score = rouge_scorer.score("\n".join(ref), "\n".join(hyp))

                rouge1_20 += score["rouge1"].fmeasure
                rouge2_20 += score["rouge2"].fmeasure
                rougeLsum_20 += score["rougeLsum"].fmeasure

            rouge1_20 = rouge1_20 / total_num
            rouge2_20 = rouge2_20 / total_num
            rougeLsum_20 = rougeLsum_20 / total_num
            ppl_2_mean_20 = np.mean(all_2_ppl_20)

            print("evaluation rouge1: %.6f, rouge2: %.6f, rougeL: %.6f, ppl2: %.6f" % (rouge1_20, rouge2_20, rougeLsum_20, ppl_2_mean_20))

            data = [
                [uniform_layer, '%.6f' % rouge1_20, '%.6f' % rouge2_20, '%.6f' % rougeLsum_20, '%.6f' % ppl_2_mean_20] + generated_summaries_20,
            ]

            with open('output.csv', 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(data)

            count = 0
            generated_summaries_20 = []
            target_summaries_20 = []
            all_2_ppl_20 = []

        input_ids = input["input_ids"].to(device)
        decoder_input_ids = target["input_ids"].to(device)
        
        input_ids_i = input_ids[0]
        decoder_input_ids_i = decoder_input_ids[0]

        outputs = model.model(input_ids = input_ids_i.unsqueeze(0),decoder_input_ids = decoder_input_ids_i.unsqueeze(0),return_dict=True,output_hidden_states=True)

        encoder = []

        for j in range(len(outputs['encoder_hidden_states'])):
            if len(encoder) == 0:
                encoder =  outputs['encoder_hidden_states'][j]
            else:
                encoder = torch.cat((encoder, outputs['encoder_hidden_states'][j]), dim=0)

        if len(all_encoder_hidden_states) == 0:
            all_encoder_hidden_states = encoder
        else:
            all_encoder_hidden_states = torch.cat((all_encoder_hidden_states, encoder), dim=1)

    print('all_encoder_hidden_states.shape,all_decoder_hidden_states.shape',all_encoder_hidden_states.shape)
    np.save('./feature_out/bart_encoder_hidden_states',all_encoder_hidden_states.cpu().numpy())


rouge1, rouge2, rougeLsum = 0, 0, 0

rouge_scorer = RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer=True)

total_num = len(target_summaries)

for (hyp, ref) in zip(generated_summaries, target_summaries):
    hyp = sent_tokenize(" ".join(word_tokenize(hyp.strip())))
    ref = sent_tokenize(" ".join(word_tokenize(ref.strip())))

    score = rouge_scorer.score("\n".join(ref), "\n".join(hyp))

    rouge1 += score["rouge1"].fmeasure
    rouge2 += score["rouge2"].fmeasure
    rougeLsum += score["rougeLsum"].fmeasure

rouge1 = rouge1 / total_num
rouge2 = rouge2 / total_num
rougeLsum = rougeLsum / total_num
ppl_2_mean = np.mean(all_2_ppl)

print("evaluation rouge1: %.6f, rouge2: %.6f, rougeL: %.6f, ppl2: %.6f" % (rouge1, rouge2, rougeLsum, ppl_2_mean))

data = [
    [uniform_layer, '%.6f' % rouge1, '%.6f' % rouge2, '%.6f' % rougeLsum, '%.6f' % ppl_2_mean],
]

with open('output.csv', 'a', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(data)
