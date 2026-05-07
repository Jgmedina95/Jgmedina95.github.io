#!/usr/bin/env python3
"""
Create an animated GIF of the crystal relaxation visualization from the blog.
"""

from PIL import Image, ImageDraw, ImageFont
import io
import math

def create_svg_image(width, height, state):
    """Render an SVG state as a PIL Image."""
    # Create a white image
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)
    
    # Define colors
    cell_color = (148, 163, 184)          # #94a3b8
    bond_color = (100, 116, 139)          # #64748b
    ti_color = (59, 130, 246)             # #3b82f6
    h_color = (34, 197, 94)               # #22c55e
    
    # State-specific styling
    if state == 0:  # Unrelaxed
        header_text = "⚠ Unrelaxed Structure"
        header_color = (239, 68, 68)       # #ef4444
        energy_text = "Ef = 0.34 eV/atom"
        energy_bg = (254, 242, 242)        # #fef2f2
        energy_border = (252, 165, 165)    # #fca5a5
        energy_text_color = (220, 38, 38)  # #dc2626
        
        # Atoms - unrelaxed positions (distorted)
        ti_positions = [(55, 55), (210, 48)]
        h_positions = [(138, 230), (30, 172), (220, 165)]
        
    elif state == 1:  # Partially Relaxed
        header_text = "Turn 1 — Partially Relaxed"
        header_color = (249, 115, 22)      # #f97316
        energy_text = "Ef = 0.12 eV/atom"
        energy_bg = (255, 251, 235)        # #fffbeb
        energy_border = (252, 211, 77)     # #fcd34d
        energy_text_color = (217, 119, 6)  # #d97706
        
        # Atoms - partially relaxed
        ti_positions = [(50, 50), (200, 50)]
        h_positions = [(130, 210), (50, 170), (200, 170)]
        
    else:  # state == 2, Fully Relaxed
        header_text = "Turn 2 — Relaxed ✓"
        header_color = (34, 197, 94)       # #22c55e
        energy_text = "Ef = 0.025 eV/atom ✓"
        energy_bg = (240, 253, 244)        # #f0fdf4
        energy_border = (134, 239, 172)    # #86efac
        energy_text_color = (22, 163, 74)  # #16a34a
        
        # Atoms - fully relaxed (symmetric)
        ti_positions = [(45, 45), (205, 45)]
        h_positions = [(130, 200), (45, 165), (205, 165)]
    
    # Offset for centering in canvas
    x_offset = 30
    y_offset = 30
    
    # Draw unit cell (dashed rectangle)
    cell_x1, cell_y1 = x_offset + 30, y_offset + 30
    cell_x2, cell_y2 = x_offset + 230, y_offset + 230
    
    # Draw dashed rectangle
    dash_length = 6
    gap_length = 3
    for x in range(cell_x1, cell_x2, dash_length + gap_length):
        draw.line([(x, cell_y1), (min(x + dash_length, cell_x2), cell_y1)], fill=cell_color, width=2)
        draw.line([(x, cell_y2), (min(x + dash_length, cell_x2), cell_y2)], fill=cell_color, width=2)
    for y in range(cell_y1, cell_y2, dash_length + gap_length):
        draw.line([(cell_x1, y), (cell_x1, min(y + dash_length, cell_y2))], fill=cell_color, width=2)
        draw.line([(cell_x2, y), (cell_x2, min(y + dash_length, cell_y2))], fill=cell_color, width=2)
    
    # Draw bonds
    bond_lines = [
        (ti_positions[0], h_positions[0]),
        (ti_positions[1], h_positions[0]),
        (ti_positions[0], ti_positions[1]),
        (ti_positions[0], h_positions[1]),
        (ti_positions[1], h_positions[2]),
    ]
    if state == 2:  # Add extra bond for fully relaxed
        bond_lines.append((h_positions[1], h_positions[2]))
    
    for (x1, y1), (x2, y2) in bond_lines:
        draw.line([(x_offset + x1, y_offset + y1), (x_offset + x2, y_offset + y2)], 
                 fill=bond_color, width=2)
    
    # Draw atoms (Ti = larger blue circles, H = smaller green circles)
    for x, y in ti_positions:
        x_pos, y_pos = x_offset + x, y_offset + y
        draw.ellipse([x_pos - 18, y_pos - 18, x_pos + 18, y_pos + 18], 
                    fill=ti_color, outline=(29, 78, 216), width=2)
        # Draw "Ti" text
        try:
            font = ImageFont.load_default()
            draw.text((x_pos - 5, y_pos - 7), "Ti", fill='white', font=font)
        except:
            pass
    
    for x, y in h_positions:
        x_pos, y_pos = x_offset + x, y_offset + y
        draw.ellipse([x_pos - 12, y_pos - 12, x_pos + 12, y_pos + 12], 
                    fill=h_color, outline=(21, 128, 61), width=2)
        # Draw "H" text
        try:
            font = ImageFont.load_default()
            draw.text((x_pos - 3, y_pos - 5), "H", fill='white', font=font)
        except:
            pass
    
    # Draw energy label box
    energy_x1, energy_y1 = x_offset + 60, y_offset + 248
    energy_x2, energy_y2 = x_offset + 200, y_offset + 274
    draw.rectangle([energy_x1, energy_y1, energy_x2, energy_y2], 
                  fill=energy_bg, outline=energy_border, width=1)
    
    try:
        font = ImageFont.load_default()
        draw.text((x_offset + 130, energy_y1 + 7), energy_text, 
                 fill=energy_text_color, font=font)
    except:
        pass
    
    # Draw header
    try:
        font = ImageFont.load_default()
        draw.text((x_offset + 30, 10), header_text, fill=header_color, font=font)
    except:
        pass
    
    return img

def create_crystal_gif(output_path):
    """Create animated GIF from crystal states."""
    images = []
    
    # Create 3 states, hold each for 1.5 seconds
    for state in range(3):
        img = create_svg_image(320, 360, state)
        # Add to GIF 3 times to hold for 1.5s at ~50ms per frame
        for _ in range(3):
            images.append(img.copy())
    
    # Save as GIF with 500ms per frame (looping 3 frames per state = 1.5s per state)
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=500,
        loop=0
    )
    
    print(f"✓ Crystal relaxation GIF created: {output_path}")
    print(f"  Size: {images[0].size}")
    print(f"  Frames: {len(images)} (3 states × 3 frames each)")

if __name__ == "__main__":
    output_file = "/Users/jorgemedina/PersonalWebsite/Jgmedina95.github.io/GRPO_For_Materials/crystal_relaxation.gif"
    create_crystal_gif(output_file)
