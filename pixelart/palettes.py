import torch
import numpy as np

# --- Данные Предустановленных Палитр ---
# (Скопируйте словарь PREDEFINED_PALETTES из вашего кода сюда)
PREDEFINED_PALETTES = {
    "EGA": [[0,0,0],[0,0,170],[0,170,0],[0,170,170],[170,0,0],[170,0,170],[170,85,0],[170,170,170],[85,85,85],[85,85,255],[85,255,85],[85,255,255],[255,85,85],[255,85,255],[255,255,85],[255,255,255]],
    "C64": [[0,0,0],[255,255,255],[136,0,0],[170,255,238],[204,68,204],[0,204,85],[0,0,170],[238,238,119],[221,136,85],[102,68,0],[255,119,119],[51,51,51],[119,119,119],[170,255,102],[0,136,255],[187,187,187]],
    "VGA256": [ [(r*51), (g*51), (b*51)] for r in range(6) for g in range(6) for b in range(6) ] + [[i*255//9, i*255//9, i*255//9] for i in range(10)], # 6x6x6 cube + 10 grays
    "NES": [[84,84,84],[0,30,116],[8,16,144],[48,0,136],[68,0,100],[92,0,48],[84,4,0],[60,24,0],[32,42,0],[8,58,0],[0,64,0],[0,60,0],[0,50,60],[0,0,0],[152,150,152],[8,76,196],[48,50,236],[92,30,228],[136,20,176],[160,20,100],[152,34,32],[120,60,0],[84,90,0],[40,114,0],[8,124,0],[0,118,40],[0,102,120],[0,0,0],[236,238,236],[76,154,236],[124,118,252],[176,98,236],[228,84,236],[252,84,184],[248,118,120],[212,136,32],[160,170,0],[116,196,0],[76,208,32],[56,204,108],[56,180,220],[60,60,60],[236,238,236],[168,204,236],[188,188,252],[212,178,236],[236,174,236],[252,174,212],[252,180,176],[248,188,176],[228,196,144],[204,210,120],[180,222,120],[168,226,144],[152,226,180],[160,162,160]], # Reduced duplicate blacks/grays from original example
    "GameBoy": [[15,56,15],[48,98,48],[139,172,15],[155,188,15]],
    "PICO-8": [[0,0,0],[29,43,83],[126,37,83],[0,135,81],[171,82,54],[95,87,79],[194,195,199],[255,241,232],[255,0,77],[255,163,0],[255,236,39],[0,228,54],[41,173,255],[131,118,156],[255,119,168],[255,204,170]],
    "APPLE_II": [[0,0,0],[227,36,0],[0,154,48],[255,190,48],[68,79,214],[255,106,214],[68,190,255],[255,255,255]], # Simplified 8 NTSC colors
    "MSX": [[0,0,0],[1,1,1],[62,184,73],[116,208,125],[94,82,255],[128,118,255],[183,99,82],[101,219,239],[219,105,89],[255,137,125],[204,199,94],[222,208,135],[58,162,65],[183,118,206],[204,204,204],[255,255,255]], # Screen 2 palette
    "ZX_SPECTRUM": [ # Normal intensity colors
        [0,0,0],[0,0,215],[215,0,0],[215,0,215],[0,215,0],[0,215,215],[215,215,0],[215,215,215],
        # Bright intensity colors (approximation)
        [0,0,0],[0,0,255],[255,0,0],[255,0,255],[0,255,0],[0,255,255],[255,255,0],[255,255,255]
        ]
}

def parse_custom_palette(hex_string, device):
    """Parses a comma-separated string of hex color codes."""
    # (Скопируйте код _parse_custom_palette из вашего узла)
    palette = []
    hex_codes = [h.strip() for h in hex_string.split(',') if h.strip()]
    if not hex_codes:
        print("Warning: Custom palette string is empty.")
        return None
    for code in hex_codes:
        if not code.startswith('#') or not (len(code) == 7 or len(code) == 4): # Allow #RGB
            print(f"Warning: Invalid hex code format '{code}'. Skipping.")
            continue
        try:
            if len(code) == 4: # Expand #RGB to #RRGGBB
                code = f"#{code[1]*2}{code[2]*2}{code[3]*2}"
            r = int(code[1:3], 16)
            g = int(code[3:5], 16)
            b = int(code[5:7], 16)
            palette.append([r / 255.0, g / 255.0, b / 255.0])
        except ValueError:
            print(f"Warning: Could not parse hex code '{code}'. Skipping.")
            continue
    if not palette:
        print("Warning: No valid colors found in custom palette string.")
        return None
    return torch.tensor(palette, device=device, dtype=torch.float32)


def get_predefined_palette(name, device):
    """Gets a predefined palette tensor."""
    # (Скопируйте код _get_predefined_palette из вашего узла)
    palette_rgb = PREDEFINED_PALETTES.get(name, PREDEFINED_PALETTES["EGA"]) # Fallback to EGA
    palette = torch.tensor(palette_rgb, device=device, dtype=torch.float32) / 255.0
    palette = torch.unique(palette, dim=0) # Remove duplicates
    return palette