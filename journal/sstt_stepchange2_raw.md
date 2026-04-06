# Raw Thoughts: SSTT Step-Changes (Post-MTFP)

## Stream of Consciousness

The MTFP conversion just landed +0.26pp by fixing a bug that was there from the beginning. Binary bit-flips pretending to be ternary neighbors. 26 images corrected by making the multi-probe do what it was always supposed to do. That's not a new idea — that's removing an obstruction.

Now I'm sitting at 97.53% with 247 errors. What do I actually know about those 247 errors? I know the mode decomposition from the old topo9 system (7 Mode A, 197 Mode B, 69 Mode C). But that was on 273 errors. MTFP fixed 26 of them. Which modes did those 26 come from? I don't know. I haven't run the error decomposition on the MTFP system. That's a gap. I'm reasoning about the new system using diagnostics from the old one.

What I think I know: the structural ranker wants diversity. We proved this with the hybrid experiment — L2 filtering destroyed accuracy by removing diverse correct-class candidates. The MTFP per-channel indexing gives the structural ranker more to work with (three independent retrieval channels instead of one joint). This is consistent — more channels means more diverse retrieval paths means more diverse candidates.

What scares me: the K-sensitivity data is from the old joint-index system. MTFP uses per-channel indexing which changes the vote distribution completely. Is the new system still K-sensitive? Does K=500 still beat K=200? I haven't tested this. The MTFP experiment used K=200 (the default from topo9_val). There might be free accuracy sitting at K=500 with the new indexing.

The 5-trit ensemble (98.42% on the old branch) used 5 different quantization thresholds feeding separate indices. That's a different kind of diversity — threshold diversity rather than channel diversity. MTFP and 5-trit ensemble aren't mutually exclusive. They could stack. Nobody has tested MTFP + multi-threshold.

Fashion-MNIST hasn't been tested with MTFP. The trit-flip fix should help there too — possibly more than MNIST, since Fashion has more intra-class diversity and the wrong-topology multi-probe would have been more damaging.

I keep thinking about the 7 Mode A failures. Those are the only errors that better ranking cannot fix. What makes those 7 images special? If I could understand them, I'd know whether the 99.93% ceiling is real or whether encoding changes could push it.

Half-formed: what if the ternary quantization thresholds (85/170) are suboptimal? They divide 0-255 into equal thirds. But MNIST pixel distributions are bimodal (lots of 0s and 255s, fewer in the middle). Adaptive thresholds per-image or per-region might capture more structure. The 5-trit ensemble explored this but combined multiple threshold sets; nobody tried finding the single best threshold.

## Questions Arising

- What's the MTFP error mode decomposition? How many of the 247 errors are Mode A vs B vs C?
- Is MTFP K-sensitive like topo9 was? What's the accuracy at K=500?
- What does MTFP get on Fashion-MNIST?
- Could MTFP + 5-trit ensemble stack? The per-channel indexing is orthogonal to multi-threshold retrieval.
- What are the 7 (or however many) Mode A images? Is there a pattern?
- Is 85/170 the optimal threshold? What happens at 80/175 or 90/165?

## First Instincts

- Run MTFP diagnostics first. Don't plan the next move until I know the error distribution of the system I actually have.
- Fashion-MNIST is low-hanging fruit — just run sstt_mtfp with data-fashion/.
- K-sensitivity sweep on MTFP is a one-line code change.
- The Mode A autopsy is the highest-ceiling investigation but narrowest (7 images).
