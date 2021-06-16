import torch
import torch.nn as nn
from arguments import get_model_classes, get_args

class Model(torch.nn.Module):

    def __init__(self, args, tokenizer = None, prompt_label_idx = None):
        super().__init__()
        model_classes = get_model_classes()
        model_config = model_classes[args.model_type]

        self.prompt_label_idx = prompt_label_idx

        self.model = model_config['model'].from_pretrained(
            args.model_name_or_path,
            return_dict=False,
            cache_dir=args.cache_dir if args.cache_dir else None)

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size),
            torch.nn.ReLU(),
            # nn.Dropout(p=args.dropout_prob),
            torch.nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size),
            )

        self.extra_token_embeddings = nn.Embedding(args.new_tokens, self.model.config.hidden_size)

    def forward(self, input_ids, attention_mask, token_type_ids, input_flags, mlm_labels, labels):
        raw_embeddings = self.model.embeddings.word_embeddings(input_ids)
        new_token_embeddings = self.mlp(self.extra_token_embeddings.weight)
        new_embeddings = new_token_embeddings[input_flags]
        inputs_embeds = torch.where(input_flags.unsqueeze(-1) > 0, new_embeddings, raw_embeddings)
        hidden_states, _ = self.model(inputs_embeds=inputs_embeds,
                          attention_mask=attention_mask,
                          token_type_ids=token_type_ids)
        hidden_states = hidden_states[mlm_labels >= 0].view(input_ids.size(0), len(self.prompt_label_idx), -1)
        logits = [
            torch.mm(
                hidden_states[:,index,:], 
                self.model.embeddings.word_embeddings.weight[i].transpose(1,0)
            )
            for index, i in enumerate(self.prompt_label_idx)
        ]
        return logits

def get_model(tokenizer, prompt_label_idx):
    args = get_args()
    model = Model(args, tokenizer, prompt_label_idx)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.cuda()
    return model

def get_tokenizer(special=[]):
    args = get_args()
    model_classes = get_model_classes()
    model_config = model_classes[args.model_type]
    tokenizer = model_config['tokenizer'].from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None)
    tokenizer.add_tokens(special)
    return tokenizer

