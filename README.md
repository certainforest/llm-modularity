# 🐕 LLM modularity via "relocation" 🐾
_[Jasmine C.](jasmine.cui@nbcuni.com) + Charles Ye_
 
<p align="center" width="100%">
<img src = 'static/shiba.jpg' width="40%">
  <p align="center">(as of 5/31/2024)</p>
</p>

**Outline**: [proposal](https://docs.google.com/document/d/1gKlafph5wCQtBBdbHHIcYHYdRqfjEzdLhYcpkpWm9g4/edit)

**To do:** 
- [X] work through ["seeing is believing" (bimt) paper](https://arxiv.org/abs/2305.08746)
- [X] migrate to runpod
- [X] "selective lesioning" experiments - see: levin @ tufts, [automated circuit discovery](https://arxiv.org/abs/2304.14997), [decoding intermediate activations](https://www.lesswrong.com/posts/fJE6tscjGRPnK8C2C/decoding-intermediate-activations-in-llama-2-7b) - made progress here (as of 5/31 - able to also recreate phi-3 layer by layer, but need to implement better output tracking + functionalize things)
- [ ] hooking up models! (making some progress) - next, will need to add another tensor somewhere in the model (kind of like one of those linear transforms - but it's a way of adding more params that can fill with meaning); maybe add to first Transformer block. Then, implement a loss function s.t. the loss function is greater if weights in this tensor respond within the correct contexts). need to figure out how to better articulate intentions here (for self - not comfortable verbalizing yet) 
- [ ] experiment w/ loss functions (haha...will get there by June 10 :)) 
