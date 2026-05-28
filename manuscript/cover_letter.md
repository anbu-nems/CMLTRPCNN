# Cover letter — PCNN ABO₃ microwave dielectric manuscript

*[Date]*

*Editor, Nature Communications*

Dear Editor,

We are pleased to submit our manuscript *"Physics-constrained decomposition identifies an A-site chemistry hierarchy associated with soft-mode amplification in ABO₃ microwave dielectrics"* for consideration as a Nature Communications research article.

ABO₃ perovskite ceramics underpin 5G and satellite-communications hardware, yet decades of empirical optimisation have left a central question unanswered: which physical mechanism — electronic polarisability, soft-mode amplification or octahedral tilting — sets the dielectric response of any given composition? Existing machine-learning models map composition to ε_r accurately but as a black box; first-principles studies resolve individual compositions but cannot survey the design space.

We trained a physics-constrained neural network on 1,304 experimentally measured ABO₃ ceramics that decomposes each prediction into named physical contributions (ε_r = ε_CM + δ_LST + δ_tilt + δ_res), with each branch architecturally constrained to the sign of its mechanism. Three findings emerge:

**(i)** The decomposition assigns 54.1% of ε_r among CM-computable compositions to the soft-mode branch δ_LST, and 96% of the model-attributed correction above the CM baseline to that branch across all B-site families;

**(ii)** A reproducible A-site amplification hierarchy Pb > Ca > La > Sr > Ba (label-permutation *p* < 10⁻³) emerges from the LST attribution, and Pb-rich compositions form a distinct extrapolation boundary (leave-Pb-out *R*² = −0.097) consistent with 6s² lone-pair-driven polarisation;

**(iii)** Used generatively, the model proposes 70 of 72 Pb-free candidates that pass a calibrated 90% conformal lower bound for ε_r ≥ 80, prioritising the Sr-Ca-Ti family as lead-free synthesis targets.

This work converts an opaque property into a chemically interpretable map for lead-free dielectric design, and the architectural template (disjoint physics-feature partition with hard sign constraints) generalises to any property expressible as a sum of mechanistically signed contributions. We believe it sits squarely at the intersection of physics-informed machine learning and lead-free electroceramics that Nature Communications has championed.

**Disclosures.** AI assistance (a large language model) was used during the drafting of the manuscript text under our direct supervision; all scientific decisions, modelling work, statistical analyses and interpretations are entirely the authors'. The manuscript has not been published, accepted or submitted elsewhere. We declare no competing interests. We are happy to participate in transparent peer review.

Thank you for your consideration.

Sincerely,

*[Corresponding Author Name]*
*[Position, Department, Institution]*
*[Email] · [ORCID]*
*on behalf of all co-authors*
