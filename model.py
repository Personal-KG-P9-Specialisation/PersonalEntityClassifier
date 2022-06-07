import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
#from pretrain.large_emb import LargeEmbedding

from transformers import RobertaConfig, RobertaForMaskedLM#, ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
from transformers.modeling_bert import BertLayerNorm, gelu


class CoLAKE(RobertaForMaskedLM):
    config_class = RobertaConfig
    #pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config, num_words_urg, num_objs_urg,num_rels_urg,n_pers_ents,n_pers_cskg,n_pers_rels, num_cskg, num_rel, ip_config='emb_ip.cfg', rel_emb=None, emb_name='entity_emb'):
        super().__init__(config)
        self.head = ClsHead(config, 2)
        self.hidden_size1= config.hidden_size
        
        self.cskg_ent_embeddings = nn.Embedding(num_cskg, config.hidden_size, padding_idx=1)
        self.rel_embeddings = nn.Embedding(num_rel, config.hidden_size, padding_idx=1)
        self.device01 = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.apply(self._init_weights)
        if rel_emb is not None:
            self.rel_embeddings = nn.Embedding.from_pretrained(rel_emb, padding_idx=1)
            print('pre-trained relation embeddings loaded.')
        #self.tie_rel_weights()

    def extend_type_embedding(self, token_type=3):
        self.roberta.embeddings.token_type_embeddings = nn.Embedding(token_type, self.config.hidden_size,
                                                                     _weight=torch.zeros(
                                                                         (token_type, self.config.hidden_size)))

    """def tie_rel_weights(self):
        self.rel_lm_head.decoder.weight = self.rel_embeddings.weight
        if getattr(self.rel_lm_head.decoder, "bias", None) is not None:
            self.rel_lm_head.decoder.bias.data = torch.nn.functional.pad(
                self.rel_lm_head.decoder.bias.data,
                (0, self.rel_lm_head.decoder.weight.shape[0] - self.rel_lm_head.decoder.bias.shape[0],),
                "constant",
                0,
            )"""

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            n_pkg_ents=None,
            n_pkg_cskg=None,
            n_pkg_rels=None, 
            n_word_nodes=None, 
            n_relation_nodes=None,
            n_tail_mentions=None,
            #n_pers_cskg=None,
            #ent_index=None,
            target=None
    ):
        n_pkg_ents=n_pkg_ents[0]
        n_pkg_rels=n_pkg_rels[0]
        n_pkg_cskg = n_pkg_cskg[0]
        n_relation_nodes=n_relation_nodes[0]
        n_tail_mentions=n_tail_mentions[0]
        #n_pers_cskg=n_pers_cskg[0]
        n_word_nodes = n_word_nodes[0]
        #n_obj_nodes = n_obj_nodes[0]
        #print(input_ids.shape)
        #PKG
        start_tokens=self.roberta.embeddings.word_embeddings(input_ids[:, 0])
        
        pkg_entities = input_ids[:, 0: n_pkg_ents]
        t_pkg_entities = torch.zeros([pkg_entities.shape[0], pkg_entities.shape[1],self.hidden_size1], dtype=torch.int32)
        if not n_pkg_ents == 0:
            for i in range(pkg_entities.shape[0]):
                t_pkg_entities[i][0][pkg_entities[i]] = 1
            pkg_entities = t_pkg_entities
        pkg_entities = pkg_entities.to(self.device01)
        pkg_cskg = self.cskg_ent_embeddings(input_ids[:,n_pkg_ents+1 : n_pkg_ents+1 +n_pkg_cskg])
        pkg_relations = self.rel_embeddings(input_ids[:,n_pkg_ents+1 +n_pkg_cskg : n_pkg_ents+1 +n_pkg_cskg+n_pkg_rels])
        sep_token = self.roberta.embeddings.word_embeddings(input_ids[:,n_pkg_ents+1 +n_pkg_cskg+n_pkg_rels : n_pkg_ents+n_pkg_cskg+n_pkg_rels+2])
        start_tokens = torch.reshape(start_tokens,(sep_token.shape[0],1,self.hidden_size1))
        """if not n_pkg_rels == 0:
            t_pkg_rels = torch.zeros([pkg_relations.shape[0], pkg_relations.shape[1],self.hidden_size1], dtype=torch.int32)
            for i in range(pkg_relations.shape[0]):
                t_pkg_rels[i][0][pkg_relations[i]] = 1
            pkg_relations = t_pkg_rels"""
        

        
        #URG
        word_embeddings = self.roberta.embeddings.word_embeddings( input_ids[:, n_pkg_ents+n_pkg_cskg+n_pkg_rels+2: n_pkg_ents+n_pkg_cskg+n_pkg_rels+2+ n_word_nodes])  # batch x n_word_nodes x hidden_size
        rel_embeddings =input_ids[:, n_pkg_ents+n_pkg_cskg+n_pkg_rels+2+ n_word_nodes:n_pkg_ents+n_pkg_cskg+n_pkg_rels+2+ n_word_nodes+n_relation_nodes]
        #print(input_ids[:, n_pkg_ents+n_pkg_rels+ n_word_nodes-1:n_pkg_ents+n_pkg_rels+ n_word_nodes+n_relation_nodes+2])
        rel_embeddings = self.rel_embeddings(
            rel_embeddings
            )
        obj_embeddings = self.roberta.embeddings.word_embeddings( input_ids[:, n_pkg_ents+n_pkg_cskg+n_pkg_rels+2+ n_word_nodes+n_relation_nodes:])
        
        #PKG
        #pers_embeddings = input_ids[:, n_word_nodes + n_obj_nodes+n_rel_nodes : n_word_nodes + n_obj_nodes+n_rel_nodes+n_pers_nodes]
        #cskg_embeddings = input_ids[:, n_word_nodes + n_obj_nodes+n_rel_nodes+n_pers_nodes : n_word_nodes + n_obj_nodes+n_rel_nodes+n_pers_nodes+ n_pers_cskg]
        #pers_rels_embedding = self.cskg_ent_embeddings( 
        #    input_ids[:, n_word_nodes + n_obj_nodes+n_rel_nodes+n_pers_nodes+ n_pers_cskg:])

        """rel_embeddings =  rel_embeddings[0]
        word_embeddings = word_embeddings[0]
        obj_embeddings = obj_embeddings[0]"""
        
        if pkg_relations.shape[1] == 0:
            inputs_embeds = torch.cat([start_tokens,sep_token, word_embeddings, rel_embeddings,obj_embeddings],
                                  dim=1)  # batch x seq_len x hidden_size
        else:
            inputs_embeds = torch.cat([start_tokens, pkg_entities,pkg_cskg,pkg_relations ,sep_token,word_embeddings, rel_embeddings,obj_embeddings],                                  dim=1)  # batch x seq_len x hidden_size

        outputs = self.roberta(
            input_ids=None,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        
        sequence_output = outputs[0][:, 0, :]  # batch x seq_len x hidden_size
        predictions = self.head(sequence_output)
        
        loss_fct = CrossEntropyLoss(ignore_index=-1, reduction='mean')
        act_target = torch.nonzero((target==1.0), as_tuple=True)[1]
        loss = loss_fct(predictions.view(-1, predictions.size(-1)), act_target.view(-1))
        return {'loss': loss, 'pred': predictions}

class ClsHead(nn.Module):
    def __init__(self, config, num_labels, dropout=0.3):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.decoder = nn.Linear(config.hidden_size, num_labels, bias=False)
        self.bias = nn.Parameter(torch.zeros(num_labels), requires_grad=True)
        self.dropout = nn.Dropout(p=dropout)
        self.decoder.bias = self.bias

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = gelu(x)
        x = self.dropout(x)
        x = self.layer_norm(x)
        x = self.decoder(x)
        return x