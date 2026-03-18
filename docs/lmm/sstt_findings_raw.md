# LMM Phase 1: RAW — SSTT Findings & Current State

- We've moved from isolated experiments to a tiered engine. The router works. It's actually faster and more robust than the standard cascade for most images.
- Tier 1 (Fast) is spooky good. 99.9% on 10-15% of images using just the plurality of votes from the retrieval phase. It's basically a "canonical digit" detector.
- Tier 2 (Light) with the Dual Hot Map (Pixel + Gauss) is a great middle ground. It moves the accuracy floor to ~76% instantly. 
- Tier 3 (Deep) is now a beast. Divergence, Centroid, Profile, Grid Div, MAD-weighting. It hits 97.2% static but 96% inside the router for hard cases.
- CIFAR-10 is still the frontier. 50% is a milestone but it feels like we're just histogramming. The Lagrangian shift (Curl/Tracing) is the right move.
- Curl identifies vortices. Tracing maps the skeleton. This is pose-invariant. It's how we'll beat the "Cat ≈ Dog" problem eventually.
- Scaling to 224x224 works! Bilinear interpolation is basically a cheap spline. 16x compression with <2% loss is huge. 
- But Variance-Guided knot placement failed. It clustered knots around stroke edges and left holes in the centers. Uniformity is a geometric requirement.
- Fashion-MNIST is our best proof of value. +8.9pp over kNN. It proves ternary topology > pixel overlap.
- Memory access is still the bottleneck (87%). We are bound by scattered writes in the vote phase.
- We have 55 contributions now. It's a lot. The INDEX and CHANGELOG are clean but the "spirit" of the project shifted from "can we do this?" to "this is how we build a ternary engine."
- Zero learned parameters is the hill we're dying on. It's the most novel claim.
