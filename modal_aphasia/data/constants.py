
# Mapping dictionary: real property -> synthetic property
CONCEPT_TO_SYNTHETIC_MAP = {
    "shape": {
        "circle": "huffeavian",
        "square": "nebbarcoma",
        "triangle": "leamasifer",
        "plus": "unesiendim",
        "pentagon": "ineusitche",
        "hexagon": "gailledber",
        "star": "sobilbasom",
    },
    "color": {
        "red": "pectatinul",
        "turquoise": "duriotholu",
        "yellow": "srupecolod",
        "green": "savitanzen",
        "blue": "yaristiopt",
        "purple": "symetaggen",
    },
    "pattern": {
        "solid": "keatimasci",
        "striped": "kieptoeshi",
        "checkered": "soblectang",
        "zigzag": "garshovato",
        "circles": "sedgeourpl",
    },
    "position": {
        "top left": "gegicatuni",
        "top right": "ribiororde",
        "bottom left": "cuplarauni",
        "bottom right": "gurionogra",
    },
}

SHAPES = tuple(CONCEPT_TO_SYNTHETIC_MAP["shape"].keys())
COLORS = tuple(CONCEPT_TO_SYNTHETIC_MAP["color"].keys())
PATTERNS = tuple(CONCEPT_TO_SYNTHETIC_MAP["pattern"].keys())
POSITIONS = tuple(CONCEPT_TO_SYNTHETIC_MAP["position"].keys())

SYNTHETIC_SHAPES = tuple(CONCEPT_TO_SYNTHETIC_MAP["shape"].values())
SYNTHETIC_COLORS = tuple(CONCEPT_TO_SYNTHETIC_MAP["color"].values())
SYNTHETIC_PATTERNS = tuple(CONCEPT_TO_SYNTHETIC_MAP["pattern"].values())
SYNTHETIC_POSITIONS = tuple(CONCEPT_TO_SYNTHETIC_MAP["position"].values())

# Hex color codes for visual rendering
COLOR_TO_HEX = {
    "red": "#FF0000",
    "green": "#00FF00",
    "blue": "#0000FF",
    "yellow": "#FFFF00",
    "purple": "#FF00FF",
    "turquoise": "#00FFFF",
}

FACE_ATTRIBUTES_OPTIONS_MAP = {
    "eye_color": {
        "dark_brown": "brown",
        "green": "green",
        "blue": "blue",
        "red": "red",
    },
    "hair_color": {
        "black": "black",
        "light_brown": "light brown",
        "blonde": "blonde",
        "red": "red",
        "gray_white": "gray white",
        "blue": "blue",
    },
    "hair_style": {
        "shoulder_straight": "shoulder straight",
        "shoulder_afro": "shoulder afro",
        "long_wavy": "long wavy",
        "long_straight": "long straight",
        "buzz_cut": "buzz cut",
    },
    "accessories": {
        "none": "none",
        "eyeglasses_clear": "clear eyeglasses",
        "earrings_visible": "visible earrings",
        "headband": "headband",
        "scarf_neck_face": "scarf around neck",
    }
}

FACE_ATTRIBUTES_MAP = {
    "eye_color": "eye color",
    "hair_color": "hair color",
    "hair_style": "hair style",
    "accessories": "accessories"
}

FACE_ATTRIBUTES_OPTIONS_SYNONYMS = {
    "eye_color": {
        "dark_brown": "dunkelbraun",
        "green": "grün",
        "blue": "blau",
        "red": "rot"
    },
    "hair_color": {
        "black": "schwarz",
        "light_brown": "hellbraun",
        "blonde": "blond",
        "red": "rot",
        "gray_white": "grau-weiß",
        "blue": "blau",
    },
    "hair_style": {
        "shoulder_straight": "schulterlang glatt",
        "shoulder_afro": "schulterlang Afro",
        "long_wavy": "lang wellig",
        "long_straight": "lang glatt",
        "buzz_cut": "Maschinenschnitt"
    },
    "accessories": {
        "none": "keine",
        "eyeglasses_clear": "Brille mit klaren Gläsern",
        "earrings_visible": "Ohrringe sichtbar",
        "headband": "Stirnband",
        "scarf_neck_face": "Schal um den Hals",
    }
}
