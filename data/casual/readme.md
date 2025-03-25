
- question type
    - {'EX', 'FA', 'FO', 'HM', 'MC', 'YN'}
        - MC: multiple choice, YN: yes/no
        - FA: find all?, FO: find one?
        - HM: how many?
        - EX: Existence
    - objective question (HM/MC/YN/EX).
    - subjective question (FA/FO).



```
(main_task pid=63259) [Content Validation]
(main_task pid=63259)   Expected: dag with nodes x, h, e, p, j, g, t, r, y and directed edges h->p, p->j, r->y, x->e, e->g, x->r, h->x, h->e, e->j, p->t
(main_task pid=63259)   Predicted: the markov equivalence class of the given graph includes the following graph:
(main_task pid=63259)
(main_task pid=63259) - nodes: x, h, e, p, j, g, t, r, y
(main_task pid=63259) - directed edges: h->p, p->j, r->y, x->e, e->g, x->r, x->h, h->e, e->j, p->t, x->j, h->r, r->e, e->p
(main_task pid=63259)
(main_task pid=63259) this graph maintains the same markov blanket for each node as the original graph. for example, the markov blanket for node x includes h, e,
r, and p, which is the same as in the original graph. similarly, for node h, the markov blanket includes x, e, and p, which is also the same as in the original gr
aph
(main_task pid=63259)   Content validation: MISMATCH
```