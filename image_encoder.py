"""Student implementation of the final project's image encoder.

Complete TODO-01 through TODO-06 using NumPy. Model parameters are supplied.
The public classes retain the final project's interface. helpers.py is provided.
阶段	需要完成	
patches	TODO-01	 
tokens	TODO-05a、05b	
block0	TODO-02、03a、03b、03c、04a、04b	
last_tokens	无新增，12 层循环	
output	TODO-06	
batch	无新增，验证批处理	
依据check.py，要按这个顺序完成
"""


import numpy as np
from helpers import LayerNorm, GELU, attention_weights, causal_mask, unit_vector


class MultiHeadAttention:
    def __init__(self, weights, prefix, heads):
        self.qkv_weight = weights[prefix + '.in_proj_weight']  # [3D,D]
        self.qkv_bias = weights[prefix + '.in_proj_bias']      # [3D]
        self.out_weight = weights[prefix + '.out_proj.weight']  # [D,D]
        self.out_bias = weights[prefix + '.out_proj.bias']
        self.heads = heads
        self.width = self.out_bias.shape[0]
        self.head_dim = self.width // heads

    def split_heads(self, x):
        """TODO-02. Input [B,N,D], return [B,H,N,D/H].

        In this image model D=768, H=12 and D/H=64. Use reshape/transpose.
        Research the head organization independently. Preserve token order.
        """
        # TODO-02 contract: [B,N,D] to [B,H,N,D/H].
        # Image model: D=768, H=12, D/H=64.
        #x是q/k/v中的一个
        """拆成多个头让12个64维子空间各自独立算 softmax attention，
        分别学习不同的token关系模式，表达力更强而总计算量几乎不变"""
        B, N, D = x.shape
        x = x.reshape(B, N, self.heads, self.head_dim)
        #x[b, n, h, :] 就是第 b 张图、第 n 个 token、第 h 个头的 64 维向量
        return x.transpose(0, 2, 1, 3) #为了后面 q @ k^T 的形状正确要改一下顺序

    def __call__(self, x, causal=False):
        # Given: stored linear weights use [out,in]. Parameters stay fixed.
        qkv = x @ self.qkv_weight.T + self.qkv_bias
        q = self.split_heads(qkv[:, :, :self.width])
        k = self.split_heads(qkv[:, :, self.width:2*self.width])
        v = self.split_heads(qkv[:, :, 2*self.width:])

        # TODO-03a: query/key comparisons, scaled by sqrt(head_dim).
        # q,k,v: [B,H,N,64]. scores must have [B,H,N,N].
        #对最后两维做矩阵乘法、前面维度当batch，来批量算每一对q . k
        scores = q @ k.transpose(0, 1, 3, 2) #每个头、每个query对所有key的相关度
        #q · k = |q| |k| cos(θ) 方向越一致，点积越大，也就是相似度，衡量“相关”
        scores = scores / np.sqrt(np.float32(self.head_dim)) #q @ k^T -> [B,H,N,N]
        #注意：np.sqrt(64)返回的是float64标量，它与float32数组相除时，会把整个数组的类型提升为float64，因此不满足check.py 要求的“全程必须保持 float32”
        #把score的标准差拉回1左右，softmax工作在梯度良好的区间，维持数值稳定性
        #scores是softmax的输入

        # Given for later text-encoder compatibility. Images use causal=False.
        if causal:
            scores = causal_mask(scores)
        coefficients = attention_weights(scores)  # supplied numerical helper

        # TODO-03b: mix value rows using coefficients; then join head outputs.
        # context: [B,H,N,64], merged: [B,N,D].
        #coefficient是softmax之后的注意力权重矩阵
        #每个token按注意力权重加权收集value
        context = coefficients @ v  #[B,H,N,N]@[B,H,N,64]=[B,H,N,64]
        #x是原始token特征，形状[B,N,768]
        B, N = x.shape[0], x.shape[1]
        merged = context.transpose(0, 2, 1, 3).reshape(B, N, self.width)
        #H和N交换，[B,N,H,64]，再把把相邻的H×64合并成768，把12个头合并回768维
        #但merged的768里面12个头仍然并排放着，之间没有任何交流

        # TODO-03c: output projection, using out_weight and out_bias.
        # Return [B,N,D]. Remember the stored linear-weight orientation.
        #out_weight是[out,in]，所以右乘它的转置，让不同头之间第一次发生信息交换
        return merged @ self.out_weight.T + self.out_bias
        

class TransformerBlock:
    def __init__(self, weights, prefix, heads):
        self.norm1 = LayerNorm(weights[prefix+'.ln_1.weight'], weights[prefix+'.ln_1.bias'])
        self.norm2 = LayerNorm(weights[prefix+'.ln_2.weight'], weights[prefix+'.ln_2.bias'])
        self.attention = MultiHeadAttention(weights, prefix+'.attn', heads)
        self.fc1_weight = weights[prefix+'.mlp.c_fc.weight']  # [4D,D]
        self.fc1_bias = weights[prefix+'.mlp.c_fc.bias']
        self.fc2_weight = weights[prefix+'.mlp.c_proj.weight']  # [D,4D]
        self.fc2_bias = weights[prefix+'.mlp.c_proj.bias']
        self.gelu = GELU()

    def __call__(self, x, causal=False):
        # Given: normalize each token and calculate its attention update.
        update = self.attention(self.norm1(x), causal=causal)

        # TODO-04a: first residual addition. Name the result residual.
        residual = x + update #residual的作用就是把attention的结果传递下去，update是修正量

        # Given: the second Norm and the exact checkpoint activation helper.
        u = self.norm2(residual) #[B,N,768]

        # TODO-04b: MLP = GELU(u @ W1.T + b1) @ W2.T + b2.
        # hidden: [B,N,3072], mlp_update: [B,N,768] for the image tower.
        # Return residual + mlp_update. Both residual paths retain [B,N,D].
        #如果说attention是横向交流，token之间传信息；那么mlp就是纵向加工，token对自己的加工
        hidden = self.gelu(u @ self.fc1_weight.T + self.fc1_bias) #[B,N,3072]
        mlp_update = hidden @ self.fc2_weight.T + self.fc2_bias #[B,N,768]
        #需要非线性，只有线性不够，因为两层线性叠起来，还是一个线性层。
        #3072维中间层完全白费，模型表达能力和一个[768,768]矩阵没区别。
        #总的来说，是先升维，给模型一个更宽的工作台，再非线性，最后再降维
        return residual + mlp_update #第二个残差，跟上面的update一样


class ImageEncoder:
    def __init__(self, weights):
        self.patch_weight = weights['visual.conv1.weight'].reshape(768, 3072) 
        self.cls = weights['visual.class_embedding'].reshape(1, 1, 768)
        self.position = weights['visual.positional_embedding']  # [50,768]
        self.pre_norm = LayerNorm(weights['visual.ln_pre.weight'], weights['visual.ln_pre.bias'])
        self.blocks = [TransformerBlock(weights, 'visual.transformer.resblocks.' + str(i), 12)
                       for i in range(12)]
        self.post_norm = LayerNorm(weights['visual.ln_post.weight'], weights['visual.ln_post.bias'])
        self.projection = weights['visual.proj']  # [768,512], already [in,out]

    def patchify(self, images):
        """TODO-01. [B,3,224,224] -> [B,49,3072].

        Visit patches left to right, then top to bottom. Within a patch, flatten
        C,H,W in that order. Each 32x32 RGB patch contributes 3*32*32 values.
        Use reshape/transpose. Do not resize the already-prepared input.
        """
        # TODO-01: separate patch-grid axes; reorder; flatten.
        # Input [B,3,224,224], return [B,49,3072].
        B = images.shape[0] 
        #先因数分解
        x = images.reshape(B,3,7,32,7,32) #对应索引数组[0,1,2,3,4,5]
        #把7移到一起[B,7,7,3,32,32] 对应的索引数组[0,2,4,1,3,5]
        x = np.transpose(x,[0,2,4,1,3,5])
        #再乘到一起
        x = x.reshape(B,49,3072)
        return x

    def embed(self, images):
        patches = self.patchify(images)  # [B,49,3072]

        # TODO-05a: project patches to [B,49,768]. There is no patch bias.
        #卷积核patch_weight是[768,3072]，要让patches和它相乘
        #矩阵乘法 p @ M 要求 M 的列和 p 对齐。所以要把 [768, 3072] 转置成 [3072, 768]
        patches = patches @ self.patch_weight.T
        
        # Given: one supplied CLS row per image.
        cls = np.tile(self.cls, (images.shape[0], 1, 1))

        # TODO-05b: prepend CLS on the token axis, then add learned positions.
        # Return [B,50,768]. self.position broadcasts over B.
        #cls是整图的浓缩表示，方便transformer输出
        #49要变成50，需要多拼接一个cls在49那个维度上，也就是axis=1
        tokens = np.concatenate([cls, patches], axis=1)
        return tokens + self.position 
        #给每个token注入“我在第几个位置”的位置信息，根据tokens的第0维"B"广播[50，768]

    def encode_tokens(self, images):
        """Provided assembly of token input and the 12 blocks; [B,50,768]."""
        x = self.pre_norm(self.embed(images))
        for block in self.blocks:
            x = block(x)
        return x

    def __call__(self, images):
        """TODO-06. Return one normalized output token [B,512] per image."""
        tokens = self.encode_tokens(images) #[B,50,768]

        # Select CLS, apply post_norm, and project with self.projection.
        # [B,50,768] -> [B,768] -> [B,512]. visual.proj is already [in,out].
        # Name the projected array output. The final unit-vector helper is given.
        #取CLS → post_norm → 投影
        cls = tokens[:, 0, :]          #[B,768]
        cls = self.post_norm(cls)      #[B,768]，最后归一化一次，稳定数值
        #线性投影：把768维的图特征压到512维
        output = cls @ self.projection #[B,512]，projection：[768,512]、[in,out]
        #逐行归一化，每个向量的长度变成1，所有向量都落在512维空间里的单位球面上
        return unit_vector(output)
