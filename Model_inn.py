import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F

import math

#============classes===================
# compute Temporal Self-similarity Matrix
class Sims(nn.Module):
    def __init__(self):
        super(Sims, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.bn = nn.BatchNorm2d(1)
        
    def forward(self, x):
        '''(N, S, E)  --> (N, 1, S, S)'''
        f = x.shape[1]
        
        I = torch.ones(f).to(self.device)
        xr = torch.einsum('bfe,h->bhfe', (x, I))   #[x, x, x, x ....]  =>  xr[:,0,:,:] == x
        xc = torch.einsum('bfe,h->bfhe', (x, I))   #[x x x x ....]     =>  xc[:,:,0,:] == x
        diff = xr - xc
        out = torch.einsum('bfge,bfge->bfg', (diff, diff))
        out = out.unsqueeze(1)
        #out = self.bn(out)
        out = F.softmax(-out/13.544, dim = -1)
        return out

#---------------------------------------------------------------------------

class ResNet50Bottom(nn.Module):
    def __init__(self):
        super(ResNet50Bottom, self).__init__()
        self.original_model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT, progress=True)
        self.activation = {}
        h = self.original_model.layer3[2].register_forward_hook(self.getActivation('comp'))
        
    def getActivation(self, name):
        def hook(model, input, output):
            self.activation[name] = output
        return hook

    def forward(self, x):
        self.original_model(x)
        output = self.activation['comp']
        return output

#---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        x = self.dropout(x)
        return x 

#----------------------------------------------------------------------------
class TransformerLayer(nn.Module):
    """Implements a single transformer layer (https://arxiv.org/abs/1706.03762).
    """

    def __init__(self, d_model, num_heads, dff,
                 dropout_rate=0.1,
                 reorder_ln=False):
        super(TransformerLayer, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.mha = nn.MultiheadAttention(d_model, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dff),
            nn.ReLU(),
            nn.Linear(dff, d_model)
        )

        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)

        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.reorder_ln = reorder_ln

    def forward(self, x):
        inp_x = x

        if self.reorder_ln:
            x = self.layernorm1(x)

        # (input_seq_len, batch_size, d_model)
        attn_output, _ = self.mha(x, x, x, attn_mask=None)
        attn_output = self.dropout1(attn_output)

        if self.reorder_ln:
            out1 = inp_x + attn_output
            x = out1
        else:
            # (input_seq_len, batch_size, d_model)
            out1 = self.layernorm1(x + attn_output)
            x = out1

        if self.reorder_ln:
            x = self.layernorm2(x)

        # (input_seq_len, batch_size, d_model)
        ffn_output = self.ffn(x)
        ffn_output = self.dropout2(ffn_output)

        if self.reorder_ln:
            out2 = out1 + ffn_output
        else:
            # (input_seq_len, batch_size, d_model)
            out2 = self.layernorm2(out1 + ffn_output)

        return out2


#----------------------------------------------------------------------------
class TransEncoder(nn.Module):
    def __init__(self, d_model, n_head, dim_ff, dropout=0.0, num_layers = 1):
        super(TransEncoder, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model, 0.1, 64)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model = d_model,
                                                    nhead = n_head,
                                                    dim_feedforward = dim_ff,
                                                    dropout = dropout,
                                                    activation = 'relu')
        encoder_norm = nn.LayerNorm(d_model)
        self.trans_encoder = nn.TransformerEncoder(encoder_layer, num_layers, encoder_norm)
                
    def forward(self, src):
        src = self.pos_encoder(src)
        e_op = self.trans_encoder(src)
        return e_op



def flatten_sequential_feats(x, batch_size, seq_len):
    x = x.view(batch_size , seq_len , -1)
    return x


#=============Model====================

class RepNet(nn.Module):
    def __init__(self,
                 num_frames,
                 transformer_layers_config=((512, 4, 512),),
                 transformer_reorder_ln = True,
                 transformer_dropout_rate = 0.0,
                 period_fc_channels=(512, 512),
                 within_period_fc_channels=(512, 512),
                 dropout_rate = 0.25,
                ):
        super(RepNet, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.num_frames = num_frames # 64

        self.resnetBase = ResNet50Bottom()
        
        
        self.conv3D = nn.Conv3d(in_channels = 1024,
                                out_channels = 512,
                                kernel_size = 3,
                                padding = (3,1,1),
                                dilation = (3,1,1))
        self.bn1 = nn.BatchNorm3d(512)
        self.pool = nn.MaxPool3d(kernel_size = (1, 7, 7))
        
        self.sims = Sims()

        self.pos_encoding = nn.Parameter(torch.Tensor(1, self.num_frames, 1))
        nn.init.normal_(self.pos_encoding, mean=0, std=0.02)

        self.mha_sim = nn.MultiheadAttention(embed_dim=512, num_heads=4)

        
        self.conv3x3 = nn.Conv2d(in_channels = 1,
                                 out_channels = 32,
                                 kernel_size = 3,
                                 padding = 1)
        
        self.bn2 = nn.BatchNorm2d(32)
        self.dropout1 = nn.Dropout(0.25)

        self.transformer_layers_config = transformer_layers_config
        channels = self.transformer_layers_config[0][0]
        self.input_projection = nn.Linear(in_features=2048, out_features = channels)
        self.relu = nn.ReLU()
        self.input_projection2 = nn.Linear(in_features= 2048 , out_features = channels)
        # self.l2_regularizer = nn.L2Loss()
        self.transformer_layers = []
        self.transformer_layers_config = transformer_layers_config
        self.transformer_dropout_rate = transformer_dropout_rate
        self.transformer_reorder_ln = transformer_reorder_ln

        for d_model, num_heads, dff in self.transformer_layers_config: # (512, 4, 512)
            self.transformer_layers.append(
                TransformerLayer(d_model, num_heads, dff,
                                self.transformer_dropout_rate,
                                self.transformer_reorder_ln))
        
        self.transformer_layers2 = []
        for d_model, num_heads, dff in self.transformer_layers_config:
            self.transformer_layers2.append(
                TransformerLayer(d_model, num_heads, dff,
                                self.transformer_dropout_rate,
                                self.transformer_reorder_ln))    
            
        self.period_fc_channels = period_fc_channels # (512, 512)
        self.fc_layers = []
        num_preds = self.num_frames//2
        for channels in self.period_fc_channels:
            self.fc_layers.append(nn.Linear(512 , channels)) # 没有加l2正则化 
        self.fc_layers.append(nn.Linear(512, num_preds))
        
        self.dropout_rate = dropout_rate
        self.dropout_layer = nn.Dropout(self.dropout_rate)

        # Within Period Module
        self.within_period_fc_channels = within_period_fc_channels
        num_preds = 1
        self.within_period_fc_layers = []
        for channels in self.within_period_fc_channels:
            self.within_period_fc_layers.append(nn.Linear(512 , channels))
        self.within_period_fc_layers.append(nn.Linear(512 , num_preds))

        self.ln1 = nn.LayerNorm(512)
        
        self.transEncoder1 = TransEncoder(d_model=512, n_head=4, dropout = 0.2, dim_ff=512, num_layers = 1)
        self.transEncoder2 = TransEncoder(d_model=512, n_head=4, dropout = 0.2, dim_ff=512, num_layers = 1)
        
        #period length prediction
        self.fc1_1 = nn.Linear(512, 512)
        self.ln1_2 = nn.LayerNorm(512)
        self.fc1_2 = nn.Linear(512, self.num_frames//2)
        self.fc1_3 = nn.Linear(self.num_frames//2, 1)


        #periodicity prediction
        self.fc2_1 = nn.Linear(512, 512)
        self.ln2_2 = nn.LayerNorm(512)
        self.fc2_2 = nn.Linear(512, self.num_frames//2)
        self.fc2_3 = nn.Linear(self.num_frames//2, 1)

    def forward(self, x, ret_sims = False):
        batch_size, _, c, h, w = x.shape # batch_size = 1
        x = x.view(-1, c, h, w) # [64, 3, 112, 112]
        x = self.resnetBase(x) # [64, 1024, 7, 7]
        x = x.view(batch_size, self.num_frames, x.shape[1],  x.shape[2],  x.shape[3]) # [1, 64, 1024, 7, 7]
        x = x.transpose(1, 2) # [1, 1024, 64, 7, 7]
        x = F.relu(self.bn1(self.conv3D(x))) # [1, 512, 64, 7, 7]
                        
        x = x.view(batch_size, 512, self.num_frames, 7, 7) # [1, 512, 64, 7, 7]
        x = self.pool(x).squeeze(3).squeeze(3) # [1, 512, 64]
        x = x.transpose(1, 2)                           #batch, num_frame, 512 [1, 64, 512]
        x = x.reshape(batch_size, self.num_frames, -1) # [1, 64, 512]
        final_embs = x # 原始输入经过resnet后再经过池化和变换后的输出
        x1 = F.relu(self.sims(x)) # [1, 1, 64, 64] TSM 
        
        
        # x = x.transpose(0, 1) # [64, 1, 512]
        # _, x2 = self.mha_sim(x, x, x) # [1, 64, 64]
        # x2 = F.relu(x2.unsqueeze(1)) # [1, 1, 64, 64]
        # x = torch.cat([x1, x2], dim = 1) # [1, 2, 64, 64]
        # x1 = F.relu(self.bn2(self.conv3x3(x1))) # [1, 32, 64, 64]

        # 3x3 conv layer on self-similarity matrix.
        x1 = F.relu(self.conv3x3(x1)) # [1, 32, 64, 64] # 原版tensorflow中没有normalization layer
        # x1 = F.relu(x1.squeeze(0)) # [32, 64, 64]
        x1 = x1.view(batch_size,self.num_frames,-1) # [1,64,2048]
        within_period_x = x1

        # Period prediction.
        x1 = self.input_projection(x1) # [1, 64, 512]
        # l2_loss = self.l2_regularizer(self.input_projection.weight)


        x1 = x1 + self.pos_encoding # [1, 64, 512] 
        for transformer_layer in self.transformer_layers:
            transformer_layer.to(self.device)
            x1 = transformer_layer(x1) # [1, 64, 512] 
        x1 = flatten_sequential_feats(x1, batch_size, self.num_frames) # [1, 64, 512]
        for fc_layer in self.fc_layers:
            fc_layer.to(self.device)
            x1 = self.dropout_layer(x1) # [1, 64, 512]
            x1 = fc_layer(x1) # [1, 64, 512] 论文里的period length li
        

        # Within period prediction.
        within_period_x = self.input_projection2(within_period_x) # [1, 64, 512] 
        within_period_x += self.pos_encoding # [1, 64, 512]
        for transformer_layer in self.transformer_layers2:
            transformer_layer.to(self.device)
            within_period_x = transformer_layer(within_period_x) # [1, 64, 512] 

        within_period_x = flatten_sequential_feats(within_period_x,
                                            batch_size,
                                            self.num_frames)  
        for fc_layer in self.within_period_fc_layers: # [1, 64, 512]  [1, 64, 512] [1, 64, 1]
            fc_layer.to(self.device)
            within_period_x = self.dropout_layer(within_period_x) # 
            within_period_x = fc_layer(within_period_x) # 论文里的periodicity pi

        return x1, within_period_x, final_embs # [1, 64, 32]  [1, 64, 1] [1, 64, 512]
        # attn_output, attn_output_weights = self.mha_sim(x1, x1, x1) 


        # xret = x
        # print(xret.shape) # [1, 2, 64, 64]
        
        # x = F.relu(self.bn2(self.conv3x3(x)))     #batch, 32, num_frame, num_frame # [1, 32, 64, 64]
        # #print(x.shape)
        # x = self.dropout1(x) # [1, 32, 64, 64]

        # x = x.permute(0, 2, 3, 1) # [1, 64, 64, 32]
        # x = x.reshape(batch_size, self.num_frames, -1)  #batch, num_frame, 32*num_frame  [1, 64, 2048])
        # x = self.ln1(F.relu(self.input_projection(x)))  #batch, num_frame, d_model=512  [1, 64, 512]
        
        # x = x.transpose(0, 1)                          #num_frame, batch, d_model=512 [64, 1, 512]
        
        # #period
        # x1 = self.transEncoder1(x) # [64, 1, 512]
        # y1 = x1.transpose(0, 1) # [1,64,512]
        # y1 = F.relu(self.ln1_2(self.fc1_1(y1))) # [1,64,512]
        # y1 = F.relu(self.fc1_2(y1)) # [1,64,32]
        # y1 = F.relu(self.fc1_3(y1)) # [1,64,1]

        # #periodicity
        # x2 = self.transEncoder2(x) # [64, 1, 512]
        # y2 = x2.transpose(0, 1) # 
        # y2 = F.relu(self.ln2_2(self.fc2_1(y2)))
        # y2 = F.relu(self.fc2_2(y2))
        # y2 = F.relu(self.fc2_3(y2)) # [1,64,1]
        
        # #y1 = y1.transpose(1, 2)                         #Cross enropy wants (minbatch*classes*dimensions)
        # if ret_sims:
        #     return y1, y2, xret
        # return y1, y2


def get_repnet_model(logdir):
    model = RepNet()
    