#!/usr/bin/env python3
"""
Fig 1 — CMLTRPCNNv77 architecture diagram
Gemini image generation, Style B Modern Minimal

Usage:
  export GEMINI_API_KEY="your_key_here"
  python gen_fig1_architecture.py

Output: fig1_architecture_attempt1.png / attempt2.png / attempt3.png
Pick the best and copy to fig1_architecture.png
"""
import os, sys, time

try:
    from google import genai
except ImportError:
    print("Install: pip install google-genai")
    sys.exit(1)

API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    print("ERROR: Set GEMINI_API_KEY environment variable.")
    print("  Get a key at: https://aistudio.google.com/apikey")
    sys.exit(1)

MODEL      = "gemini-2.0-flash-preview-image-generation"
OUTPUT_DIR = "/Users/anbu/Desktop/NC figures/main"
client     = genai.Client(api_key=API_KEY)

PROMPT = """
=== SECTION 1: FRAMING ===
Create a Modern Minimal-style technical architecture diagram for a Nature Communications
materials science paper. The diagram shows a Physics-Constrained Neural Network (PCNN)
that decomposes the dielectric constant of perovskite ceramics into four mechanistic
contributions. The figure should feel authoritative, clean, and scientifically precise —
suitable for a top-tier journal. Landscape orientation, wide aspect ratio (roughly 3:1).

=== SECTION 2: VISUAL STYLE — MODERN MINIMAL ===
- Ultra-clean geometric shapes with crisp, perfectly straight edges
- Background: pure white (#FFFFFF)
- Four horizontal SECTION BANDS stacked from top to bottom, each with a distinct muted fill:
  Band 1 (Input):   very light slate (#F0F4F8)
  Band 2 (Anchor):  very light teal (#EBF5F3)
  Band 3 (Branches): very light amber (#FBF6EC)
  Band 4 (Output):  very light coral (#FBF0EE)
- Component boxes: white fill, 8px rounded corners, thin 1px border in the band's accent color
- ONE accent color per band, used for borders, headers, and arrow colors:
  Band 1: Slate blue #5E81AC
  Band 2: Deep teal #264653
  Band 3: Amber #D97706
  Band 4: Coral #E76F51
- Arrows: 1.5px thick, dark gray (#4B5563), with small solid arrowhead (6px triangle)
- Typography: sans-serif (like Inter), bold for section headers (11pt), regular for labels (9pt)
- Labels sit INSIDE boxes
- Generous whitespace — at least 20px between elements
- NO decorative elements, NO icons, NO shadows, NO clip art
- ALL text must be perfectly horizontal (no rotated text)

=== SECTION 3: COLOR PALETTE ===
- Background: #FFFFFF
- Band 1 fill: #F0F4F8  Border/accent: #5E81AC
- Band 2 fill: #EBF5F3  Border/accent: #264653
- Band 3 fill: #FBF6EC  Border/accent: #D97706
- Band 4 fill: #FBF0EE  Border/accent: #E76F51
- Arrow color: #4B5563
- Constraint badge: #DC2626 (red) for ≥ and ≤ signs
- Text: #1F2937 (near black)
- Secondary text: #6B7280 (gray)

=== SECTION 4: LAYOUT ===

The diagram flows LEFT TO RIGHT with 4 vertical sections. Each section spans the full height.

--- SECTION A (leftmost, ~20% width): INPUT FEATURES ---
Header at top: "INPUT FEATURES" in slate blue #5E81AC, bold, centered
Background fill: #F0F4F8

Three stacked boxes inside:
Box A1 (top, teal border): "LST Features" / "30 physics features" / "soft-mode, tolerance factor," / "regime flags, ionic radii"
Box A2 (middle, amber border): "Tilt Features" / "30 physics features" / "octahedral tilt proxy," / "bond angle variance"
Box A3 (bottom, coral border): "Residual Features" / "41 features" / "processing history," / "crystal system, VCA flag"

Below all three boxes, a small separate box (gray border, white fill):
"CM Anchor" / "αr_mol = Σ nᵢ αᵢ" / "εr_CM = (1+2x)/(1−x)" / "ZERO parameters"

--- SECTION B (~15% width): SHARED ENCODER ---
Header: "SHARED ENCODER" in slate blue, bold, centered
Background fill: #F0F4F8

One tall central box:
"ResNet Trunk" (bold)
"trunk_hidden = 512"
"n_blocks = 2"
"Pre-norm residual + SiLU"
"Dropout = 0.168"

--- SECTION C (widest, ~45% width): PHYSICS BRANCHES ---
Header: "CONSTRAINED PHYSICS BRANCHES" in amber #D97706, bold, centered
Background fill: #FBF6EC

Four parallel horizontal branch rows from top to bottom:

Row 1 (top): LST Branch
- Small box: "LST Head" / "96 hidden" / "SiLU × 2"
- Arrow right to constraint badge (green circle): "≥ 0"
- Box: "δ_LST = r_Ia × softplus(h_LST)" where r_Ia ∈ {0.9, 1.0, 1.5, 0.7}
- Small annotation below: "Regime scale: Ia×1.0  Ib×1.5  II×0.9  III×0.7"

Row 2: Tilt Branch
- Small box: "Tilt Head" / "32 hidden" / "SiLU × 2"
- Arrow right to constraint badge (red circle): "≤ 0"
- Box: "δ_tilt = −softplus(h_tilt)"
- Small annotation: "Architecturally suppressed"

Row 3: Confidence Gate (thinner row, gray background)
- Box: "Physics Gate  σ_conf" / "= sigmoid(−2.0 + 3.0·VCA + 2.0·GII + 2.0·Φ)"
- Small annotation: "Low σ → physics reliable  |  High σ → ML compensates"

Row 4: Gated Residual Branch
- Small box: "Residual Head" / "192 hidden" / "SiLU × 3"
- Box: "δ_res = σ_conf × tanh(h_res) × s"
- Small annotation: "Suppressed when physics is reliable"

Below all four rows, a thin special row (gray, dashed border):
"Fallback MLP  (no-CM compositions)" / "er_pred = fallback(residual features)"

--- SECTION D (rightmost, ~20% width): OUTPUT DECOMPOSITION ---
Header: "OUTPUT" in coral #E76F51, bold, centered
Background fill: #FBF0EE

One central equation box (slightly larger, coral border, bold text):
"εr = εr_CM"
"    + δ_LST  (≥0)"
"    + δ_tilt  (≤0)"
"    + δ_res"

Below it, a results summary box (white, thin gray border):
"Formula-split R² = 0.934"
"Strat-GSS R²     = 0.893"
"5-seed SWA ensemble"

Below that, a decomposition box:
"Branch contributions:"
"CM: 47.4%  |  LST: 53.1%"
"Tilt: 1.3% |  Res: 1.4%"

=== SECTION 5: CONNECTIONS ===
All arrows go LEFT TO RIGHT.

1. Box A1 (LST Features) → Shared Trunk: solid arrow, slate blue, label "30 features"
2. Box A2 (Tilt Features) → Shared Trunk: solid arrow, slate blue, label "30 features"
3. Box A3 (Residual Features) → Shared Trunk: solid arrow, slate blue, label "41 features"
4. CM Anchor box → directly to Output equation box: solid arrow, deep teal, long horizontal arrow bypassing the Trunk and Branches, labeled "εr_CM (0 params)"

5. Shared Trunk → LST Head: solid arrow, amber
6. Shared Trunk → Tilt Head: solid arrow, amber
7. Shared Trunk → Residual Head: solid arrow, amber

8. LST Head → constraint ≥0 badge → δ_LST box: solid arrows
9. Tilt Head → constraint ≤0 badge → δ_tilt box: solid arrows
10. Box A3 (Residual Features) → Gate σ_conf box: dashed arrow (side input to gate)
11. Gate σ_conf → Residual Head (gating signal): downward dashed arrow, gray, labeled "σ_conf"
12. Residual Head → δ_res box: solid arrow

13. δ_LST box → Output equation: solid arrow, amber
14. δ_tilt box → Output equation: solid arrow, amber
15. δ_res box → Output equation: solid arrow, coral

=== SECTION 6: CONSTRAINTS ===
- SPELL ALL TEXT EXACTLY AS WRITTEN. Do not paraphrase or shorten any labels.
- The Greek letter ε must appear as ε (epsilon), not as e or E
- δ must appear as δ (delta), not d
- σ must appear as σ (sigma)
- No clip art, no icons, no photorealistic elements
- No rotated or diagonal text — all text is horizontal
- Do not add a title or figure caption — the diagram is self-contained
- Do not add a watermark
- The four section bands must be clearly visible as distinct background regions
- All four branch rows in Section C must be visible and separated by thin horizontal dividers
- Minimum 1200px wide output
"""


def generate_image(prompt_text: str, attempt: int) -> str | None:
    print(f"\n{'='*60}\nAttempt {attempt}\n{'='*60}")
    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=prompt_text,
            config=genai.types.GenerateContentConfig(
                response_modalities=["IMAGE", "TEXT"],
            ),
        )
        out_path = os.path.join(OUTPUT_DIR, f"fig1_architecture_attempt{attempt}.png")
        for part in response.candidates[0].content.parts:
            if part.inline_data:
                with open(out_path, "wb") as f:
                    f.write(part.inline_data.data)
                size = os.path.getsize(out_path)
                print(f"Saved: {out_path}  ({size:,} bytes)")
                return out_path
            elif part.text:
                print(f"Text response: {part.text[:300]}")
        print("WARNING: No image in response")
        return None
    except Exception as e:
        print(f"ERROR: {e}")
        return None


def main():
    print("Generating Fig 1 — Architecture diagram (3 attempts)")
    print(f"Output: {OUTPUT_DIR}")
    saved = []
    for i in range(1, 4):
        if i > 1:
            time.sleep(3)
        path = generate_image(PROMPT, i)
        if path:
            saved.append(path)
    if not saved:
        print("\nAll attempts failed. Check GEMINI_API_KEY and model availability.")
        sys.exit(1)
    print(f"\nGenerated {len(saved)} attempt(s). Review and rename the best to:")
    print(f"  {OUTPUT_DIR}/fig1_architecture.png")


if __name__ == "__main__":
    main()
