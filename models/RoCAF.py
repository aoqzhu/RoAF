import torch
from torch import nn
from .basic_layers import Transformer, CrossTransformer, GradientReversalLayer, CrossmodalEncoder, GatedFusion
from .bert import BertTextEncoder
from einops import rearrange, repeat
import torch.nn.functional as F
from .MoE import MoE_block



class RCAF(nn.Module):
    def __init__(self, args):
        super(RCAF, self).__init__()

        self.bertmodel = BertTextEncoder(use_finetune=True,
                                         transformers=args['model']['feature_extractor']['transformers'],
                                         pretrained=args['model']['feature_extractor']['bert_pretrained'])

        self.proj_l = nn.Sequential(
            nn.Linear(args['model']['feature_extractor']['input_dims'][0],
                      args['model']['feature_extractor']['hidden_dims'][0]),
            Transformer(num_frames=args['model']['feature_extractor']['input_length'][0],
                        save_hidden=False,
                        token_len=args['model']['feature_extractor']['token_length'][0],
                        dim=args['model']['feature_extractor']['hidden_dims'][0],
                        depth=args['model']['feature_extractor']['depth'],
                        heads=args['model']['feature_extractor']['heads'],
                        mlp_dim=args['model']['feature_extractor']['hidden_dims'][0])
        )

        self.proj_a = nn.Sequential(
            nn.Linear(args['model']['feature_extractor']['input_dims'][2],
                      args['model']['feature_extractor']['hidden_dims'][2]),
            Transformer(num_frames=args['model']['feature_extractor']['input_length'][2],
                        save_hidden=False,
                        token_len=args['model']['feature_extractor']['token_length'][2],
                        dim=args['model']['feature_extractor']['hidden_dims'][2],
                        depth=args['model']['feature_extractor']['depth'],
                        heads=args['model']['feature_extractor']['heads'],
                        mlp_dim=args['model']['feature_extractor']['hidden_dims'][2])
        )

        self.proj_v = nn.Sequential(
            nn.Linear(args['model']['feature_extractor']['input_dims'][1],
                      args['model']['feature_extractor']['hidden_dims'][1]),
            Transformer(num_frames=args['model']['feature_extractor']['input_length'][1],
                        save_hidden=False,
                        token_len=args['model']['feature_extractor']['token_length'][1],
                        dim=args['model']['feature_extractor']['hidden_dims'][1],
                        depth=args['model']['feature_extractor']['depth'],
                        heads=args['model']['feature_extractor']['heads'],
                        mlp_dim=args['model']['feature_extractor']['hidden_dims'][1])
        )

        self.img_MoE = MoE_block(dim=args['model']['moe']['dim'],
                                 heads=args['model']['moe']['heads'],
                                 dim_head=args['model']['moe']['dim_head'],
                                 num_experts=args['model']['moe']['num_experts'],
                                 capacity_factor=args['model']['moe']['capacity_factor'],
                                 dropout=args['model']['moe']['dropout'],
                                 top_k=args['model']['moe']['top_k'])

        self.text_MoE = MoE_block(dim=args['model']['moe']['dim'],
                                  heads=args['model']['moe']['heads'],
                                  dim_head=args['model']['moe']['dim_head'],
                                  num_experts=args['model']['moe']['num_experts'],
                                  capacity_factor=args['model']['moe']['capacity_factor'],
                                  dropout=args['model']['moe']['dropout'],
                                  top_k=args['model']['moe']['top_k'])
        self.audio_MoE = MoE_block(dim=args['model']['moe']['dim'],
                                   heads=args['model']['moe']['heads'],
                                   dim_head=args['model']['moe']['dim_head'],
                                   num_experts=args['model']['moe']['num_experts'],
                                   capacity_factor=args['model']['moe']['capacity_factor'],
                                   dropout=args['model']['moe']['dropout'],
                                   top_k=args['model']['moe']['top_k'])


        self.crossmodal_inject_t = CrossmodalEncoder(text_dim=args['model']['crossmodal_encoder']['hidden_dims'][0],
                                                     audio_dim=args['model']['crossmodal_encoder']['hidden_dims'][2],
                                                     video_dim=args['model']['crossmodal_encoder']['hidden_dims'][1],
                                                     embed_dim=args['model']['crossmodal_encoder']['embed_dim'],
                                                     num_layers=args['model']['crossmodal_encoder']['num_layers'],
                                                     attn_dropout=args['model']['crossmodal_encoder']['attn_dropout'])
        self.crossmodal_inject_a = CrossmodalEncoder(text_dim=args['model']['crossmodal_encoder']['hidden_dims'][0],
                                                     audio_dim=args['model']['crossmodal_encoder']['hidden_dims'][2],
                                                     video_dim=args['model']['crossmodal_encoder']['hidden_dims'][1],
                                                     embed_dim=args['model']['crossmodal_encoder']['embed_dim'],
                                                     num_layers=args['model']['crossmodal_encoder']['num_layers'],
                                                     attn_dropout=args['model']['crossmodal_encoder']['attn_dropout'])
        self.crossmodal_inject_v = CrossmodalEncoder(text_dim=args['model']['crossmodal_encoder']['hidden_dims'][0],
                                                     audio_dim=args['model']['crossmodal_encoder']['hidden_dims'][2],
                                                     video_dim=args['model']['crossmodal_encoder']['hidden_dims'][1],
                                                     embed_dim=args['model']['crossmodal_encoder']['embed_dim'],
                                                     num_layers=args['model']['crossmodal_encoder']['num_layers'],
                                                     attn_dropout=args['model']['crossmodal_encoder']['attn_dropout'])

        self.GatedFusion = GatedFusion(d_model=args['model']['gated_fusion']['hidden_dims'])

        self.fc1 = nn.Linear(args['model']['regression']['input_dim'], args['model']['regression']['hidden_dim'])
        self.fc2 = nn.Linear(args['model']['regression']['hidden_dim'], args['model']['regression']['out_dim'])
        self.dropout = nn.Dropout(args['model']['regression']['attn_dropout'])

    def predict(self, x):
        output = self.fc2(self.dropout(F.relu(self.fc1(x))))
        return output

    def forward(self, complete_input, incomplete_input):
        vision, audio, language = complete_input
        vision_m, audio_m, language_m = incomplete_input

        h_1_v = self.proj_v(vision_m)[:, :8]
        h_1_a = self.proj_a(audio_m)[:, :8]
        h_1_l = self.proj_l(self.bertmodel(language_m))[:, :8]

        # crossmodal_inject
        enhance_l = self.crossmodal_inject_t(h_1_l, h_1_a, h_1_v)
        enhance_v = self.crossmodal_inject_v(h_1_v, h_1_l, h_1_a)
        enhance_a = self.crossmodal_inject_a(h_1_a, h_1_l, h_1_v)

        # MoE
        text_moe_output, text_moe_loss = self.text_MoE(enhance_l)
        img_moe_output, img_moe_loss = self.img_MoE(enhance_v)
        audio_output, audio_moe_loss = self.audio_MoE(enhance_a)

        moe_loss = text_moe_loss + img_moe_loss + audio_moe_loss

        fusion_m = self.GatedFusion(text_moe_output, img_moe_output, audio_output)

        # fusion = self.GRL(fusion_m)
        output = self.predict(torch.mean(fusion_m, dim=1))

        return {'sentiment_preds': output,
                'moe_loss': moe_loss,
                }


def build_model(args):
    return RCAF(args)
