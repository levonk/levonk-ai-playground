# Reference Image Pack — Character Identity

This directory holds character reference image packs for LoRA training and
IP-Adapter conditioning. Each pack is a subdirectory containing 20-40 images
of the same character at multiple angles, expressions, and lighting conditions.

## Pack Structure

```
reference-packs/
├── default/          # Default/example pack
│   ├── 01_front.png
│   ├── 02_side.png
│   ├── 03_three_quarter.png
│   ├── ...
│   ├── captions.txt   # Consistent captions for training
│   └── README.md      # Character description + token
└── zvra/              # Named character pack
    ├── 01_front.png
    ├── 02_side.png
    ├── ...
    ├── captions.txt
    └── README.md
```

## Image Requirements

- 20-40 images per character
- Multiple angles: front, side, three-quarter, back
- Neutral and varied facial expressions
- Full-body, waist-up, and close-up shots
- Several lighting conditions
- Consistent age, hairstyle, clothing design, and body proportions
- No excessive motion blur or complicated backgrounds

## Caption Format

Use a unique character token in every caption. Keep it consistent:

```
zvra_person, adult woman, short black hair, green eyes, red jacket
zvra_person, adult woman, short black hair, green eyes, red jacket, side view
zvra_person, adult woman, short black hair, green eyes, red jacket, close-up
...
```

## Training

```bash
# Train a LoRA from a reference pack
./train-lora.sh reference-packs/zvra/

# With custom parameters
./train-lora.sh reference-packs/zvra/ --steps 2000 --rank 32
```

## Usage in ComfyUI

After training, the LoRA is saved to `ComfyUI/models/loras/{character}_lora.safetensors`.
Load it in the generation workflow via the LoraLoader node with strength 0.6-0.9.

Keep the LoRA light — the goal is identity retention, not memorizing every pose.
Keep clothing and action concepts separate where possible; otherwise the LoRA
may force the character to wear one outfit or repeat one pose.
