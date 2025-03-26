
- question type
    - {'EX', 'FA', 'FO', 'HM', 'MC', 'YN'}
        - MC: multiple choice, YN: yes/no
        - FA: find all?, FO: find one?
        - HM: how many?
        - EX: Existence
    - objective question (HM/MC/YN/EX).
    - subjective question (FA/FO).


- ？拓扑顺序是否只有一种；

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


```
对于 FO/FA 的问题
Expected: j, n, k, t, w
Predicted: one valid topological ordering of the graph is: j, n, k, t, w
Content validation: MISMATCH

Expected: h, k, d, q, f, u
Predicted: one valid topological ordering of the graph is: h, d, k, f, q, u
Content validation: MISMATCH

Expected: {'r', 'l'}
Predicted: the maximal valid backdoor adjustment set for treatment s and outcome g in this admg is {l, r}. this set blocks all backdoor paths from s to g.
Content validation: MISMATCH

Expected: {'x', 'j', 'r'}
Predicted: {j, r, x}
Content validation: MISMATCH


Expected: [{'a', 'b'}]
Predicted: the frontdoor adjustment set for treatment t and outcome p in the given dag is {a, b}. variables a and b are both directed towa
rds p (a->p, b->p) and towards t (a->t, b->t), and they are not descendants of t. adjusting for a and b will allow us to estimate the causal effect of t on p with
out confounding bias
Content validation: MISMATCH


Expected: {'s'}
Predicted: s
Content validation: MISMATCH

Expected: {'x', 'j', 'r'}
Predicted: j, r, x
Content validation: MISMATCH



#??
Expected: dag with nodes k, m, i, u, f, z, j, a and directed edges z->j, u->a, f->j, k->m, u->i, i->k, u->z, j->a, i->j
Expected: g->x->g
Predicted: x->g, g->x
Content validation: MISMATCH

```

- update 0326
    - {X, O, P, D, K}, {H} => {'k}', 'd', 'o', '{h', 'p', 'x'}