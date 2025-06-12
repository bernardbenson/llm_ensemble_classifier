import os
import argparse
from dotenv import load_dotenv
from typing import List, Dict, Any

# Load environment variables from .env file
load_dotenv()

# Static Configuration 
FIELD_TO_LABEL_MAP: Dict[str, str] = {
"messenger_em_radiation_gamma_rays": "Gamma rays",
    "messenger_em_radiation_x_rays": "X-rays",
    "messenger_em_radiation_ultraviolet": "Ultraviolet",
    "messenger_em_radiation_optical": "Optical",
    "messenger_em_radiation_infrared": "Infrared",
    "messenger_em_radiation_microwave": "Microwave",
    "messenger_em_radiation_radio": "Radio",
    "messenger_gravitational_waves_compact_binary_inspiral": "Compact Binary Inspiral",
    "messenger_gravitational_waves_stochastic": "Stochastic",
    "messenger_gravitational_waves_continuous": "Continuous",
    "messenger_gravitational_waves_burst": "Burst",
    "messenger_cosmic_rays": "Cosmic Rays",
    "messenger_neutrinos": "Neutrinos",
    "objects_binaries_binary_black_holes": "Binary Black Holes",
    "objects_binaries_binary_neutron_stars": "Binary Neutron Stars",
    "objects_binaries_cataclysmic_variables": "Cataclysmic Variables",
    "objects_binaries_neutron_star_black_hole": "Neutron Star-Black Hole",
    "objects_binaries_binary_pulsars": "Binary Pulsars",
    "objects_binaries_white_dwarf_binaries": "White Dwarf Binaries",
    "objects_black_holes_active_galactic_nuclei": "Active Galactic Nuclei",
    "objects_black_holes_intermediate_mass": "Intermediate Mass",
    "objects_black_holes_stellar_mass": "Stellar Mass",
    "objects_black_holes_supermassive": "Supermassive",
    "objects_neutron_stars_magnetars": "Magnetars",
    "objects_neutron_stars_pulsars": "Pulsars",
    "objects_neutron_stars_pulsar_wind_nebula": "Pulsar Wind Nebula",
    "objects_exoplanets": "Exoplanets",
    "objects_supernova_remnants": "Supernova Remnants",
    "signals_fast_blue_optical_transients": "Fast Blue Optical Transients",
    "signals_fast_radio_bursts": "Fast Radio Bursts",
    "signals_gamma_ray_bursts": "Gamma-ray Bursts",
    "signals_kilonovae": "Kilonovae",
    "signals_novae": "Novae",
    "signals_pevatrons": "Pevatrons",
    "signals_stellar_flares": "Stellar flares",
    "signals_supernovae": "SuperNovae",
    "Not TDAMM": "Non-tdamm",
}

ALL_LABELS: List[str] = sorted(list(set(FIELD_TO_LABEL_MAP.values())))

# System Prompt Configuration
BASE_SYSTEM_PROMPT_TEMPLATE: str = (
   "Your task is to read the provided text of the article and assign relevant labels "
    "from the TDAMM classification list. "
    "Strictly adhere to the provided list of labels. Do not invent new labels. "
    f"The ONLY available labels are: {', '.join(ALL_LABELS)}. "
    "You MUST respond with a JSON object containing two keys: 'labels' (a list of strings) and 'reasoning' (a brief explanation)." 
)

# Pre-defined prompts that can be used at runtime
PROMPT_CONFIG: Dict[str, str] = {
    "default": (
        "You are an expert in classifying astronomical articles. "
        + BASE_SYSTEM_PROMPT_TEMPLATE
    ),

    "strict": (
        "You are a meticulous and precise scientific archivist focused on astrophysics. "
        "Assign labels only when there is very high confidence and direct evidence in the text. "
        + BASE_SYSTEM_PROMPT_TEMPLATE
    ),
    "creative": (
        "You are a creative and insightful scientific content analyst with expertise in astrophysics. "
        "Think outside the box but stay within the scientific domain. "
        + BASE_SYSTEM_PROMPT_TEMPLATE
    ),
    "summary_focused": (
        "You are a scientific summarizer. Your primary goal is to understand the core topic of the text "
        "and then assign the most relevant high-level T-DAMM labels. "
        + BASE_SYSTEM_PROMPT_TEMPLATE
    ),
}

# Environment based configuration
# --- Environment-based Configuration ---
OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
CHAR_LIMIT_FOR_INPUT: int = int(os.getenv("CHAR_LIMIT_FOR_INPUT", "200000"))

OUTPUT_DIR: str = "./output"
REPORTS_DIR: str = "./tmp/reports"
LOGS_DIR: str = "./tmp/logs"

def parse_arguments() -> argparse.Namespace:
    """
    Parses command line arguments for the application.
    """
    parser = argparse.ArgumentParser(
        description="Relabel a dataset using multiple LLM agents and a dynamic majority voting threshold"
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to the input JSON file."
    )
    parser.add_argument(
        "--local",
        nargs='+',
        default=[],
        help="A list of local models to use. Can specify a prompt key, e.g., 'llama3:8b:strict'."
    )
    parser.add_argument(
        "--local",
        nargs='+',
        default=[],
        help="A list of local models to use. Can specify a prompt key, e.g., 'llama3:8b:strict"
    )
    return parser.parse_args()