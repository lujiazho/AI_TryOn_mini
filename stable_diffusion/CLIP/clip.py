# stolen from https://github.com/AUTOMATIC1111/stable-diffusion-webui
import re
import math
from collections import namedtuple
import ldm.modules.diffusionmodules.model
from torch.nn.functional import silu
import torch
from einops import rearrange
from torch import einsum
from ldm.util import default

cond_cast_unet = lambda x: x.to(torch.float16)

def get_available_vram():
    stats = torch.cuda.memory_stats()
    mem_active = stats['active_bytes.all.current']
    mem_reserved = stats['reserved_bytes.all.current']
    mem_free_cuda, _ = torch.cuda.mem_get_info(torch.cuda.current_device())
    mem_free_torch = mem_reserved - mem_active
    mem_free_total = mem_free_cuda + mem_free_torch
    return mem_free_total

def split_cross_attention_forward(self, x, context=None, mask=None):
    h = self.heads

    q_in = self.to_q(x)
    context = default(context, x)

    # no need
    # context_k, context_v = hypernetwork.apply_hypernetworks(shared.loaded_hypernetworks, context)
    k_in = self.to_k(context)
    v_in = self.to_v(context)

    dtype = q_in.dtype

    k_in = k_in * self.scale

    del context, x

    q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q_in, k_in, v_in))
    del q_in, k_in, v_in

    r1 = torch.zeros(q.shape[0], q.shape[1], v.shape[2], device=q.device, dtype=q.dtype)

    mem_free_total = get_available_vram()

    gb = 1024 ** 3
    tensor_size = q.shape[0] * q.shape[1] * k.shape[1] * q.element_size()
    modifier = 3 if q.element_size() == 2 else 2.5
    mem_required = tensor_size * modifier
    steps = 1

    if mem_required > mem_free_total:
        steps = 2 ** (math.ceil(math.log(mem_required / mem_free_total, 2)))
        # print(f"Expected tensor size:{tensor_size/gb:0.1f}GB, cuda free:{mem_free_cuda/gb:0.1f}GB "
        #       f"torch free:{mem_free_torch/gb:0.1f} total:{mem_free_total/gb:0.1f} steps:{steps}")

    if steps > 64:
        max_res = math.floor(math.sqrt(math.sqrt(mem_free_total / 2.5)) / 8) * 64
        raise RuntimeError(f'Not enough memory, use lower resolution (max approx. {max_res}x{max_res}). '
                            f'Need: {mem_required / 64 / gb:0.1f}GB free, Have:{mem_free_total / gb:0.1f}GB free')

    slice_size = q.shape[1] // steps if (q.shape[1] % steps) == 0 else q.shape[1]
    for i in range(0, q.shape[1], slice_size):
        end = i + slice_size
        s1 = einsum('b i d, b j d -> b i j', q[:, i:end], k)

        s2 = s1.softmax(dim=-1, dtype=q.dtype)
        del s1

        r1[:, i:end] = einsum('b i j, b j d -> b i d', s2, v)
        del s2

    del q, k, v

    r1 = r1.to(dtype)

    r2 = rearrange(r1, '(b h) n d -> b n (h d)', h=h)
    del r1

    return self.to_out(r2)

def cross_attention_attnblock_forward(self, x):
    h_ = x
    h_ = self.norm(h_)
    q1 = self.q(h_)
    k1 = self.k(h_)
    v = self.v(h_)

    # compute attention
    b, c, h, w = q1.shape

    q2 = q1.reshape(b, c, h*w)
    del q1

    q = q2.permute(0, 2, 1)   # b,hw,c
    del q2

    k = k1.reshape(b, c, h*w) # b,c,hw
    del k1

    h_ = torch.zeros_like(k, device=q.device)

    mem_free_total = get_available_vram()

    tensor_size = q.shape[0] * q.shape[1] * k.shape[2] * q.element_size()
    mem_required = tensor_size * 2.5
    steps = 1

    if mem_required > mem_free_total:
        steps = 2**(math.ceil(math.log(mem_required / mem_free_total, 2)))

    slice_size = q.shape[1] // steps if (q.shape[1] % steps) == 0 else q.shape[1]
    for i in range(0, q.shape[1], slice_size):
        end = i + slice_size

        w1 = torch.bmm(q[:, i:end], k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w2 = w1 * (int(c)**(-0.5))
        del w1
        w3 = torch.nn.functional.softmax(w2, dim=2, dtype=q.dtype)
        del w2

        # attend to values
        v1 = v.reshape(b, c, h*w)
        w4 = w3.permute(0, 2, 1)   # b,hw,hw (first hw of k, second of q)
        del w3

        h_[:, :, i:end] = torch.bmm(v1, w4)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        del v1, w4

    h2 = h_.reshape(b, c, h, w)
    del h_

    h3 = self.proj_out(h2)
    del h2

    h3 += x

    return h3

re_attention = re.compile(r"""
\\\(|
\\\)|
\\\[|
\\]|
\\\\|
\\|
\(|
\[|
:([+-]?[.\d]+)\)|
\)|
]|
[^\\()\[\]:]+|
:
""", re.X)

re_break = re.compile(r"\s*\bBREAK\b\s*", re.S)

def parse_prompt_attention(text):
    """
    Parses a string with attention tokens and returns a list of pairs: text and its associated weight.
    Accepted tokens are:
      (abc) - increases attention to abc by a multiplier of 1.1
      (abc:3.12) - increases attention to abc by a multiplier of 3.12
      [abc] - decreases attention to abc by a multiplier of 1.1
      \( - literal character '('
      \[ - literal character '['
      \) - literal character ')'
      \] - literal character ']'
      \\ - literal character '\'
      anything else - just text

    >>> parse_prompt_attention('normal text')
    [['normal text', 1.0]]
    >>> parse_prompt_attention('an (important) word')
    [['an ', 1.0], ['important', 1.1], [' word', 1.0]]
    >>> parse_prompt_attention('(unbalanced')
    [['unbalanced', 1.1]]
    >>> parse_prompt_attention('\(literal\]')
    [['(literal]', 1.0]]
    >>> parse_prompt_attention('(unnecessary)(parens)')
    [['unnecessaryparens', 1.1]]
    >>> parse_prompt_attention('a (((house:1.3)) [on] a (hill:0.5), sun, (((sky))).')
    [['a ', 1.0],
     ['house', 1.5730000000000004],
     [' ', 1.1],
     ['on', 1.0],
     [' a ', 1.1],
     ['hill', 0.55],
     [', sun, ', 1.1],
     ['sky', 1.4641000000000006],
     ['.', 1.1]]
    """

    res = []
    round_brackets = []
    square_brackets = []

    round_bracket_multiplier = 1.1
    square_bracket_multiplier = 1 / 1.1

    def multiply_range(start_position, multiplier):
        for p in range(start_position, len(res)):
            res[p][1] *= multiplier

    for m in re_attention.finditer(text):
        text = m.group(0)
        weight = m.group(1)

        if text.startswith('\\'):
            res.append([text[1:], 1.0])
        elif text == '(':
            round_brackets.append(len(res))
        elif text == '[':
            square_brackets.append(len(res))
        elif weight is not None and len(round_brackets) > 0:
            multiply_range(round_brackets.pop(), float(weight))
        elif text == ')' and len(round_brackets) > 0:
            multiply_range(round_brackets.pop(), round_bracket_multiplier)
        elif text == ']' and len(square_brackets) > 0:
            multiply_range(square_brackets.pop(), square_bracket_multiplier)
        else:
            parts = re.split(re_break, text)
            for i, part in enumerate(parts):
                if i > 0:
                    res.append(["BREAK", -1])
                res.append([part, 1.0])

    for pos in round_brackets:
        multiply_range(pos, round_bracket_multiplier)

    for pos in square_brackets:
        multiply_range(pos, square_bracket_multiplier)

    if len(res) == 0:
        res = [["", 1.0]]

    # merge runs of identical weights
    i = 0
    while i + 1 < len(res):
        if res[i][1] == res[i + 1][1]:
            res[i][0] += res[i + 1][0]
            res.pop(i + 1)
        else:
            i += 1

    return res

class PromptChunk:
    """
    This object contains token ids, weight (multipliers:1.4) and textual inversion embedding info for a chunk of prompt.
    If a prompt is short, it is represented by one PromptChunk, otherwise, multiple are necessary.
    Each PromptChunk contains an exact amount of tokens - 77, which includes one for start and end token,
    so just 75 tokens from prompt.
    """

    def __init__(self):
        self.tokens = []
        self.multipliers = []
        self.fixes = []


PromptChunkFix = namedtuple('PromptChunkFix', ['offset', 'embedding'])
"""An object of this type is a marker showing that textual inversion embedding's vectors have to placed at offset in the prompt
chunk. Thos objects are found in PromptChunk.fixes and, are placed into FrozenCLIPEmbedderWithCustomWordsBase.hijack.fixes, and finally
are applied by sd_hijack.EmbeddingsWithFixes's forward function."""

class StableDiffusionModelHijack:
    fixes = None
    comments = []
    circular_enabled = False

    def __init__(self):
        pass

    def hijack(self, m, device):
        model_embeddings = m.cond_stage_model.transformer.text_model.embeddings
        model_embeddings.token_embedding = EmbeddingsWithFixes(model_embeddings.token_embedding, self)
        m.cond_stage_model = FrozenCLIPEmbedderWithCustomWords(m.cond_stage_model, self, device)
    
        ldm.modules.diffusionmodules.model.nonlinearity = silu
        ldm.modules.diffusionmodules.openaimodel.th = torch
        
        ldm.modules.attention.CrossAttention.forward = split_cross_attention_forward
        ldm.modules.diffusionmodules.model.AttnBlock.forward = cross_attention_attnblock_forward

class EmbeddingsWithFixes(torch.nn.Module):
    def __init__(self, wrapped, embeddings):
        super().__init__()
        self.wrapped = wrapped
        self.embeddings = embeddings

    def forward(self, input_ids):
        batch_fixes = self.embeddings.fixes
        self.embeddings.fixes = None

        inputs_embeds = self.wrapped(input_ids)

        if batch_fixes is None or len(batch_fixes) == 0 or max([len(x) for x in batch_fixes]) == 0:
            return inputs_embeds

        vecs = []
        for fixes, tensor in zip(batch_fixes, inputs_embeds):
            for offset, embedding in fixes:
                emb = cond_cast_unet(embedding.vec)
                emb_len = min(tensor.shape[0] - offset - 1, emb.shape[0])
                tensor = torch.cat([tensor[0:offset + 1], emb[0:emb_len], tensor[offset + 1 + emb_len:]])

            vecs.append(tensor)

        return torch.stack(vecs)
    
class FrozenCLIPEmbedderWithCustomWordsBase(torch.nn.Module):
    """A pytorch module that is a wrapper for FrozenCLIPEmbedder module. it enhances FrozenCLIPEmbedder, making it possible to
    have unlimited prompt length and assign weights to tokens in prompt.
    """

    def __init__(self, wrapped, hijack, device):
        super().__init__()

        self.wrapped = wrapped
        """Original FrozenCLIPEmbedder module; can also be FrozenOpenCLIPEmbedder or xlmr.BertSeriesModelWithTransformation,
        depending on model."""

        self.hijack: StableDiffusionModelHijack = hijack
        self.device = device
        self.chunk_length = 75

    def empty_chunk(self):
        """creates an empty PromptChunk and returns it"""

        chunk = PromptChunk()
        chunk.tokens = [self.id_start] + [self.id_end] * (self.chunk_length + 1)
        chunk.multipliers = [1.0] * (self.chunk_length + 2)
        return chunk

    def get_target_prompt_token_count(self, token_count):
        """returns the maximum number of tokens a prompt of a known length can have before it requires one more PromptChunk to be represented"""

        return math.ceil(max(token_count, 1) / self.chunk_length) * self.chunk_length

    def tokenize(self, texts):
        """Converts a batch of texts into a batch of token ids"""

        raise NotImplementedError

    def encode_with_transformers(self, tokens):
        """
        converts a batch of token ids (in python lists) into a single tensor with numeric respresentation of those tokens;
        All python lists with tokens are assumed to have same length, usually 77.
        if input is a list with B elements and each element has T tokens, expected output shape is (B, T, C), where C depends on
        model - can be 768 and 1024.
        Among other things, this call will read self.hijack.fixes, apply it to its inputs, and clear it (setting it to None).
        """

        raise NotImplementedError

    def encode_embedding_init_text(self, init_text, nvpt):
        """Converts text into a tensor with this text's tokens' embeddings. Note that those are embeddings before they are passed through
        transformers. nvpt is used as a maximum length in tokens. If text produces less teokens than nvpt, only this many is returned."""

        raise NotImplementedError

    def tokenize_line(self, line):
        """
        this transforms a single prompt into a list of PromptChunk objects - as many as needed to
        represent the prompt.
        Returns the list and the total number of tokens in the prompt.
        """

        parsed = parse_prompt_attention(line)

        tokenized = self.tokenize([text for text, _ in parsed])

        chunks = []
        chunk = PromptChunk()
        token_count = 0
        last_comma = -1

        def next_chunk(is_last=False):
            """puts current chunk into the list of results and produces the next one - empty;
            if is_last is true, tokens <end-of-text> tokens at the end won't add to token_count"""
            nonlocal token_count
            nonlocal last_comma
            nonlocal chunk

            if is_last:
                token_count += len(chunk.tokens)
            else:
                token_count += self.chunk_length

            to_add = self.chunk_length - len(chunk.tokens)
            if to_add > 0:
                chunk.tokens += [self.id_end] * to_add
                chunk.multipliers += [1.0] * to_add

            chunk.tokens = [self.id_start] + chunk.tokens + [self.id_end]
            chunk.multipliers = [1.0] + chunk.multipliers + [1.0]

            last_comma = -1
            chunks.append(chunk)
            chunk = PromptChunk()

        for tokens, (text, weight) in zip(tokenized, parsed):
            if text == 'BREAK' and weight == -1:
                next_chunk()
                continue

            position = 0
            while position < len(tokens):
                token = tokens[position]
                comma_padding_backtrack = 20

                if token == self.comma_token:
                    last_comma = len(chunk.tokens)

                # this is when we are at the end of alloted 75 tokens for the current chunk, and the current token is not a comma. comma_padding_backtrack
                # is a setting that specifies that if there is a comma nearby, the text after the comma should be moved out of this chunk and into the next.
                elif comma_padding_backtrack != 0 and len(chunk.tokens) == self.chunk_length and last_comma != -1 and len(chunk.tokens) - last_comma <= comma_padding_backtrack:
                    break_location = last_comma + 1

                    reloc_tokens = chunk.tokens[break_location:]
                    reloc_mults = chunk.multipliers[break_location:]

                    chunk.tokens = chunk.tokens[:break_location]
                    chunk.multipliers = chunk.multipliers[:break_location]

                    next_chunk()
                    chunk.tokens = reloc_tokens
                    chunk.multipliers = reloc_mults

                if len(chunk.tokens) == self.chunk_length:
                    next_chunk()

                chunk.tokens.append(token)
                chunk.multipliers.append(weight)
                position += 1
                continue

        if len(chunk.tokens) > 0 or len(chunks) == 0:
            next_chunk(is_last=True)

        return chunks, token_count

    def process_texts(self, texts):
        """
        Accepts a list of texts and calls tokenize_line() on each, with cache. Returns the list of results and maximum
        length, in tokens, of all texts.
        """

        token_count = 0

        cache = {}
        batch_chunks = []
        for line in texts:
            if line in cache:
                chunks = cache[line]
            else:
                chunks, current_token_count = self.tokenize_line(line)
                token_count = max(current_token_count, token_count)

                cache[line] = chunks

            batch_chunks.append(chunks)

        return batch_chunks, token_count

    def forward(self, texts):
        """
        Accepts an array of texts; Passes texts through transformers network to create a tensor with numerical representation of those texts.
        Returns a tensor with shape of (B, T, C), where B is length of the array; T is length, in tokens, of texts (including padding) - T will
        be a multiple of 77; and C is dimensionality of each token - for SD1 it's 768, and for SD2 it's 1024.
        An example shape returned by this function can be: (2, 77, 768).
        Webui usually sends just one text at a time through this function - the only time when texts is an array with more than one elemenet
        is when you do prompt editing: "a picture of a [cat:dog:0.4] eating ice cream"
        """
        batch_chunks, token_count = self.process_texts(texts)

        used_embeddings = {}
        chunk_count = max([len(x) for x in batch_chunks])

        zs = []
        for i in range(chunk_count):
            batch_chunk = [chunks[i] if i < len(chunks) else self.empty_chunk() for chunks in batch_chunks]

            tokens = [x.tokens for x in batch_chunk]
            multipliers = [x.multipliers for x in batch_chunk]
            self.hijack.fixes = [x.fixes for x in batch_chunk]

            for fixes in self.hijack.fixes:
                for position, embedding in fixes:
                    used_embeddings[embedding.name] = embedding

            z = self.process_tokens(tokens, multipliers)
            zs.append(z)

        if len(used_embeddings) > 0:
            embeddings_list = ", ".join([f'{name} [{embedding.checksum()}]' for name, embedding in used_embeddings.items()])
            self.hijack.comments.append(f"Used embeddings: {embeddings_list}")

        return torch.hstack(zs)

    def process_tokens(self, remade_batch_tokens, batch_multipliers):
        """
        sends one single prompt chunk to be encoded by transformers neural network.
        remade_batch_tokens is a batch of tokens - a list, where every element is a list of tokens; usually
        there are exactly 77 tokens in the list. batch_multipliers is the same but for multipliers instead of tokens.
        Multipliers are used to give more or less weight to the outputs of transformers network. Each multiplier
        corresponds to one token.
        """
        tokens = torch.asarray(remade_batch_tokens).to(self.device)

        # this is for SD2: SD1 uses the same token for padding and end of text, while SD2 uses different ones.
        if self.id_end != self.id_pad:
            for batch_pos in range(len(remade_batch_tokens)):
                index = remade_batch_tokens[batch_pos].index(self.id_end)
                tokens[batch_pos, index+1:tokens.shape[1]] = self.id_pad

        z = self.encode_with_transformers(tokens)

        # restoring original mean is likely not correct, but it seems to work well to prevent artifacts that happen otherwise
        batch_multipliers = torch.asarray(batch_multipliers).to(self.device)
        original_mean = z.mean()
        z = z * batch_multipliers.reshape(batch_multipliers.shape + (1,)).expand(z.shape)
        new_mean = z.mean()
        z = z * (original_mean / new_mean)

        return z


class FrozenCLIPEmbedderWithCustomWords(FrozenCLIPEmbedderWithCustomWordsBase):
    def __init__(self, wrapped, hijack, device):
        super().__init__(wrapped, hijack, device)
        self.tokenizer = wrapped.tokenizer

        vocab = self.tokenizer.get_vocab()

        self.comma_token = vocab.get(',</w>', None)

        self.token_mults = {}
        tokens_with_parens = [(k, v) for k, v in vocab.items() if '(' in k or ')' in k or '[' in k or ']' in k]
        for text, ident in tokens_with_parens:
            mult = 1.0
            for c in text:
                if c == '[':
                    mult /= 1.1
                if c == ']':
                    mult *= 1.1
                if c == '(':
                    mult *= 1.1
                if c == ')':
                    mult /= 1.1

            if mult != 1.0:
                self.token_mults[ident] = mult

        self.id_start = self.wrapped.tokenizer.bos_token_id
        self.id_end = self.wrapped.tokenizer.eos_token_id
        self.id_pad = self.id_end

    def tokenize(self, texts):
        tokenized = self.wrapped.tokenizer(texts, truncation=False, add_special_tokens=False)["input_ids"]

        return tokenized

    def encode_with_transformers(self, tokens):
        outputs = self.wrapped.transformer(input_ids=tokens, output_hidden_states=-1)

        z = outputs.last_hidden_state

        return z

    def encode_embedding_init_text(self, init_text, nvpt):
        embedding_layer = self.wrapped.transformer.text_model.embeddings
        ids = self.wrapped.tokenizer(init_text, max_length=nvpt, return_tensors="pt", add_special_tokens=False)["input_ids"]
        embedded = embedding_layer.token_embedding.wrapped(ids.to(embedding_layer.token_embedding.wrapped.weight.device)).squeeze(0)

        return embedded
